"""
STAGE: Ingestion

Multi-family O*NET ingestion, writing directly to the local JSON store.

Data flow for one run:

    select_onet_codes(prefixes)          filter the O*NET catalogue by SOC family
      -> fetch_onet_tools_and_tech(code) tools and technologies for one occupation
        -> record EVERY tool in skills_master.json (occupation mapping only)
        -> for HOT tools only:
             scrape_and_validate_skill() resolve page -> 0.90 cross-encoder gate
                                         -> Gemini credibility audit (only if gated in)
             -> approved:  score with sortingalgorithmnew, write a snapshot
             -> otherwise: status stays pending, with the gate reason recorded

The network layer is REUSED from scraping.py rather than reimplemented. That module
owns the 429/Retry-After/5xx backoff, the persistent scrape cache, the hard-mapping
table and the disambiguation filter; a second implementation here would mean a second
set of rate-limit bugs.

Cost discipline, in the order it applies:
  1. Non-hot tools are recorded and skipped BEFORE any network call.
  2. A skill whose definition was refreshed within RESUMMARIZE_AFTER_DAYS is not
     re-audited, so a re-run costs no Gemini quota.
  3. The cross-encoder gate runs before Gemini, so a wrong page never costs a call.
"""

import datetime
import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

import backlog as backlog_store
import json_store
import run_state
from gemini_keys import DailyQuotaExhausted
from json_store import (
    STATUS_APPROVED,
    STATUS_PENDING,
    load_master,
    load_timeseries,
    save_master,
    save_timeseries,
    upsert_master_entry,
    upsert_snapshot,
)
from scraping import (
    fetch_onet_tools_and_tech,
    load_local_cache,
    normalize_skill_name,
    save_local_cache,
    scrape_and_validate_batch,
    scrape_and_validate_skill,
    search_onet_occupations,
)
from sortingalgorithmnew import calculate_ai_correlation

logger = logging.getLogger(__name__)

# A definition older than this is re-scraped and re-audited on the next run. This is
# the single biggest quota control: without it every run re-audits every skill.
RESUMMARIZE_AFTER_DAYS = 90

# How long an occupation waits to enter the store-write window before giving up and
# queueing itself. The only things it can be waiting on are a reviewer's commit and the
# second pass, both of which are measured in seconds, so reaching this means something
# is genuinely stuck and spinning would not help.
STORE_WAIT_SECONDS = 300.0

# How many skills are resolved before their audits are sent. A multiple of the Gemini
# BATCH_SIZE, so a chunk fills whole batched requests, and small enough that a quota
# wall discards at most this much resolve work. The resolve results are cached, so even
# that is re-derived without network cost.
INGEST_CHUNK = 24

# Gate reasons meaning "this page was supplied, not guessed". Entries carrying one are
# never re-scraped, not even under --force. Defined here rather than imported from
# review_actions to keep the ingestion path free of the review module.
SUPPLIED_PAGE_GATE_REASONS = ("reviewer_remediated", "machine_remediated")

# A SOC prefix is two digits and a hyphen: "15-", "11-". Validated before use so a
# typo like "15" or "l5-" fails loudly instead of silently matching nothing and
# reporting a successful run over zero occupations.
PREFIX_PATTERN = re.compile(r"^\d{2}-$")

# A full O*NET code ("15-2051.00") or a SOC family prefix ("15-"). The dot is what
# tells them apart, everywhere: here, in the web form, and when draining the backlog.
TARGET_PATTERN = re.compile(r"^\d{2}-(\d{4}\.\d{2})?$")


def normalize_targets(raw: Iterable[str]) -> List[str]:
    """
    Cleans a mixed list of O*NET codes and SOC prefixes, raising on anything malformed.

    Shared by the CLI, the web form and the backlog drain so all three agree on what a
    valid target looks like. A bare "15" becomes "15-"; a full code is left alone.
    """
    cleaned: List[str] = []
    for item in raw:
        value = str(item).strip()
        if not value:
            continue
        if "." not in value and not value.endswith("-"):
            value = f"{value}-"
        if not TARGET_PATTERN.match(value):
            raise ValueError(
                f"{item!r} is not a valid target. Use a full O*NET code like "
                f"'15-2051.00' or a SOC family prefix like '15-'."
            )
        if value not in cleaned:
            cleaned.append(value)

    if not cleaned:
        raise ValueError("No targets supplied; nothing to ingest.")
    return cleaned


# The complete SOC major-group set, all 23 of them. ingest.py's FAMILY_HINTS listed
# nine of these and called them "the common ones"; a full scrape needs every group, and
# one list beats two that drift apart.
SOC_MAJOR_GROUPS = (
    ("11-", "Management"),
    ("13-", "Business and Financial Operations"),
    ("15-", "Computer and Mathematical"),
    ("17-", "Architecture and Engineering"),
    ("19-", "Life, Physical, and Social Science"),
    ("21-", "Community and Social Service"),
    ("23-", "Legal"),
    ("25-", "Educational Instruction and Library"),
    ("27-", "Arts, Design, Entertainment, Sports, and Media"),
    ("29-", "Healthcare Practitioners and Technical"),
    ("31-", "Healthcare Support"),
    ("33-", "Protective Service"),
    ("35-", "Food Preparation and Serving Related"),
    ("37-", "Building and Grounds Cleaning and Maintenance"),
    ("39-", "Personal Care and Service"),
    ("41-", "Sales and Related"),
    ("43-", "Office and Administrative Support"),
    ("45-", "Farming, Fishing, and Forestry"),
    ("47-", "Construction and Extraction"),
    ("49-", "Installation, Maintenance, and Repair"),
    ("51-", "Production"),
    ("53-", "Transportation and Material Moving"),
    ("55-", "Military Specific"),
)

ALL_PREFIXES = [prefix for prefix, _ in SOC_MAJOR_GROUPS]


def _emit(progress: Optional[Callable[[dict], None]], event: dict) -> None:
    """
    Fires a progress callback, swallowing anything it raises.

    The callback belongs to the web layer and is a reporting concern only. An exception
    inside it -- a full disk while writing the status file, say -- must never abort an
    ingestion that has already spent hours of Gemini quota.
    """
    if progress is None:
        return
    try:
        progress(event)
    except Exception:
        logger.exception("Progress callback failed; ingestion continues.")


def normalize_prefixes(raw: Iterable[str]) -> List[str]:
    """
    Cleans and validates SOC prefixes, raising on anything malformed.

    Accepts "15" and "15-" alike and normalizes both to "15-", because the trailing
    hyphen is easy to forget and dropping it would make "1-" match nothing.
    """
    cleaned: List[str] = []
    for item in raw:
        value = str(item).strip()
        if not value:
            continue
        if not value.endswith("-"):
            value = f"{value}-"
        if not PREFIX_PATTERN.match(value):
            raise ValueError(
                f"{item!r} is not a valid SOC prefix. Expected two digits and a "
                f"hyphen, for example '15-'."
            )
        if value not in cleaned:
            cleaned.append(value)

    if not cleaned:
        raise ValueError("No SOC prefixes supplied; nothing to ingest.")
    return cleaned


def select_onet_codes(prefixes: List[str]) -> Dict[str, str]:
    """
    Returns {onet_code: onet_title} for occupations in the requested SOC families.

    One search per prefix, unioned. The SOC prefix works directly as a search keyword
    ("15-" yields the 38 computer and mathematical occupations), but the match is
    FUZZY: "11-" comes back with 80 records of which only 59 are really 11-. So every
    result is filtered against the requested prefixes before use. Without that filter,
    asking for management occupations would quietly ingest healthcare ones too.

    A prefix that returns nothing is reported individually. Lumping the failures
    together would hide "27- has no occupations" behind "the API is down".
    """
    selected: Dict[str, str] = {}
    for prefix in prefixes:
        found = search_onet_occupations(prefix)
        kept = {
            code: title
            for code, title in found.items()
            if str(code).startswith(prefix)
        }
        if not kept:
            logger.error(
                "No occupations matched %s (search returned %d records, none in that "
                "family). Check connectivity or the prefix.", prefix, len(found),
            )
        else:
            logger.info(
                "%s: %d occupations (%d search hits, %d discarded as other families).",
                prefix, len(kept), len(found), len(found) - len(kept),
            )
        selected.update(kept)

    logger.info(
        "Selected %d occupations in total across %s.", len(selected), ", ".join(prefixes)
    )
    return selected


def _needs_refresh(entry: Optional[Dict[str, Any]], force: bool) -> bool:
    """
    Decides whether a skill's definition must be re-scraped and re-audited.

    True when it has never been scraped, when its definition is older than the
    freshness window, or when the caller forces it. A pending skill is NOT refreshed
    just because it is pending: it is awaiting a human, and re-scraping would
    overwrite the very text the reviewer is looking at.

    Supplied pages outrank force. A page that a reviewer typed in, or that a second
    pass proposed and had audited, is not stale evidence the scraper should improve
    on -- it is the correction TO what the scraper produced, and re-resolving would
    walk the entry straight back to the wrong article it was rescued from. Without
    this check --force silently discarded every remediation ever made, because the
    force short-circuit sits above the pending guard that was meant to protect them.
    """
    if entry is not None:
        if entry.get("gate_reason") in SUPPLIED_PAGE_GATE_REASONS:
            return False
        # Model-authored text keeps its original no_candidate_found gate reason, since
        # that is still why it is in the queue. Protect it by its provenance instead:
        # re-scraping would find no candidate all over again and overwrite a written
        # definition with the O*NET boilerplate it replaced.
        block = entry.get("second_pass")
        if isinstance(block, dict) and block.get("outcome") == "model_authored":
            return False
    if force:
        return True
    if entry is None:
        return True
    if not entry.get("wikipedia_summary"):
        return True
    if entry.get("status") == STATUS_PENDING:
        return False

    last = entry.get("last_updated")
    if not last:
        return True
    try:
        stamp = datetime.date.fromisoformat(str(last)[:10])
    except ValueError:
        logger.warning(
            "Skill %r has an unparseable last_updated (%r); refreshing it.",
            entry.get("skill_name"), last,
        )
        return True
    return (datetime.date.today() - stamp).days >= RESUMMARIZE_AFTER_DAYS


def ingest_occupation(
    onet_code: str,
    onet_title: str,
    master: Dict[str, Any],
    timeseries: List[Dict[str, Any]],
    cache: Dict[str, Any],
    force: bool = False,
) -> Dict[str, int]:
    """
    Ingests one occupation into the in-memory stores. Does not write to disk.

    The caller owns persistence so that a full multi-occupation run saves once at the
    end rather than rewriting both files per occupation.
    """
    counts = {"tools": 0, "hot": 0, "scraped": 0, "approved": 0, "pending": 0, "skipped": 0}

    payload = fetch_onet_tools_and_tech(onet_code)
    if not payload or not payload.get("skills"):
        logger.warning("O*NET %s returned no tools; skipping.", onet_code)
        return counts

    resolved_title = payload.get("onet_title") or onet_title

    # Phase 1: record every tool and work out which ones need scraping at all. Nothing
    # here touches the network or a model.
    todo = []
    for item in payload["skills"]:
        skill_name = item["skill_name"]
        counts["tools"] += 1

        # Every tool is recorded, hot or not, so the occupation map is complete.
        # Occupation objects, not flat lists: hot status belongs to the (skill,
        # occupation) relationship. A tool hot for Data Scientists and standard for
        # Statisticians must record both, not collapse to one global flag.
        entry = upsert_master_entry(
            master,
            skill_name,
            category=item.get("category", ""),
            occupations=[{
                "onet_code": onet_code,
                "onet_title": resolved_title,
                "is_hot_tech": item.get("is_hot_tech", False),
            }],
        )

        # Non-hot tools stop here, BEFORE any network or model call. This ordering is
        # what keeps a full run affordable.
        if not item.get("is_hot_tech"):
            continue
        counts["hot"] += 1

        if not _needs_refresh(entry, force):
            counts["skipped"] += 1
            logger.debug("%r is fresh; not re-auditing.", skill_name)
            continue

        todo.append(item)

    # Phase 2: resolve, gate and audit the survivors, a chunk at a time. Chunked rather
    # than all at once so a quota wall costs at most one chunk of resolve work, and so
    # the writes below still happen while the run is progressing rather than only at the
    # end of an occupation with hundreds of tools.
    #
    # DailyQuotaExhausted is deliberately NOT caught. It means every key is spent, so
    # the skills in flight were never audited and nothing about them may be written:
    # not approved, and not queued as pending either, because a pending card asks a
    # human to judge a definition that no audit has seen. Letting it propagate leaves
    # those skills exactly as they were and hands the caller the job of recording the
    # occupation on the backlog.
    #
    # Batching is kept WITHIN one occupation on purpose: the backlog records unfinished
    # work per occupation, and a batch spanning two of them could not be resumed
    # cleanly from that record.
    for start in range(0, len(todo), INGEST_CHUNK):
        chunk = todo[start:start + INGEST_CHUNK]
        decisions = scrape_and_validate_batch(chunk, cache=cache)

        for item in chunk:
            skill_name = item["skill_name"]
            decision = decisions.get(normalize_skill_name(skill_name))
            if decision is None:
                logger.error(
                    "No decision came back for %r; leaving it untouched.", skill_name
                )
                continue

            counts["scraped"] += 1
            status = STATUS_APPROVED if decision["auto_approve"] else STATUS_PENDING

            # A re-scrape replaces the evidence a second-pass finding was about, so the
            # finding has to go with it. Leaving it would show a reviewer a suggestion
            # panel describing a page this entry no longer points at.
            entry = master.get(skill_name)
            if entry is not None:
                entry["second_pass"] = None

            upsert_master_entry(
                master,
                skill_name,
                resolved_title=decision.get("resolved_title"),
                wikipedia_summary=decision.get("summary") or "",
                best_source_name=decision.get("best_source_name"),
                status=status,
                gate_reason=decision.get("gate_reason"),
                cross_score=decision.get("cross_score"),
                is_credible=decision.get("is_credible"),
            )

            if status == STATUS_APPROVED:
                counts["approved"] += 1
                record_snapshot(master[skill_name], timeseries)
            else:
                counts["pending"] += 1
                logger.info(
                    "%r held for review (%s).", skill_name, decision.get("gate_reason")
                )

    logger.info(
        "%s (%s): %d tools, %d hot, %d scraped, %d approved, %d pending, %d fresh.",
        onet_code, resolved_title, counts["tools"], counts["hot"], counts["scraped"],
        counts["approved"], counts["pending"], counts["skipped"],
    )
    return counts


def record_snapshot(entry: Dict[str, Any], timeseries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Scores an approved skill and writes its snapshot for the current quarter.

    Shared with the Streamlit review path, so a human approval and an automatic one
    produce structurally identical records. Only approved skills reach here: an
    unreviewed definition must never appear in the trend data.
    """
    # flagship_note is absent for most skills and that is the normal case: it exists
    # only for generic terms and platforms the flagship pass has evaluated. Passing ""
    # scores exactly as before, so a skill the pass never reached is unaffected.
    #
    # embeds_ai is the same story and one degree more important: a skill embedding_pass
    # has not reached yet is None, which earns no boost and fires no rule. That is the
    # correct reading of "nobody has established this", and it is why approving a skill
    # before the pass runs is safe -- the next pass picks it up and rescores it.
    metrics = calculate_ai_correlation(
        entry["skill_name"],
        entry.get("category", ""),
        entry.get("wikipedia_summary", ""),
        entry.get("flagship_note", "") or "",
        entry.get("embeds_ai"),
    )
    upsert_snapshot(
        timeseries,
        entry["skill_name"],
        metrics,
        entry.get("onet_codes", []),
        entry.get("onet_titles", []),
    )
    logger.info(
        "Scored %r: ai=%+.4f (%s) tech=%+.4f ml=%+.4f embedded=%+.4f",
        entry["skill_name"], metrics["ai_score"], metrics["category_bucket"],
        metrics["tech_base_sim"], metrics["ml_pipeline_sim"], metrics["embedded_ai_sim"],
    )
    return metrics


def ingest_single_occupation(onet_code: str, force: bool = False) -> Dict[str, Any]:
    """
    Ingests exactly one occupation by O*NET code and saves.

    The web form's single-code path. run_ingestion() covers whole SOC families; this
    covers the common case of pulling one job without enumerating its family, and
    shares the same store-loading and saving discipline.

    On the daily quota wall the occupation goes back on the backlog and everything
    already earned is still saved.

    The store is loaded and saved inside one writer window, so a reviewer committing
    decisions at the same moment either goes first or goes second -- never underneath.
    """
    cache = load_local_cache()
    quota_exhausted = False
    counts: Dict[str, Any] = {}

    try:
        with run_state.store_writer(f"ingesting {onet_code}", timeout=STORE_WAIT_SECONDS):
            master = load_master()
            timeseries = load_timeseries()
            try:
                counts = ingest_occupation(
                    onet_code, onet_code, master, timeseries, cache, force=force
                )
            except DailyQuotaExhausted as err:
                quota_exhausted = True
                backlog_store.add_targets([onet_code], reason=backlog_store.REASON_QUOTA)
                logger.error("%s Occupation %s returned to the backlog.", err, onet_code)
            finally:
                # Inside the window, so the save that lands is the one built from the
                # copy loaded inside it.
                save_master(master)
                save_timeseries(timeseries)
    except run_state.StoreBusy as busy:
        backlog_store.add_targets([onet_code], reason=backlog_store.REASON_INTERRUPTED)
        logger.error(
            "%s Occupation %s was not ingested and stays queued.", busy, onet_code
        )
        return {"quota_exhausted": False, "store_busy": True}
    finally:
        # Outside the window on purpose: only ingestion writes the scrape cache, so it
        # contends with nothing and does not need to hold up a reviewer.
        save_local_cache(cache)

    return {**counts, "quota_exhausted": quota_exhausted}


def run_ingestion(
    prefixes: List[str],
    limit: int = None,
    force: bool = False,
    *,
    progress: Optional[Callable[[dict], None]] = None,
    should_continue: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Full run across every occupation in the requested SOC families.

    `limit` caps the number of occupations, which is what makes a first run against a
    new family survivable: 11- alone is dozens of occupations and hundreds of skills.

    THE STORE IS LOADED AND SAVED ONCE PER OCCUPATION, inside a writer window, rather
    than once for the whole family. Holding one in-memory copy across a family that
    takes hours meant every reviewer decision made in that window was erased by the save
    at the end, and -- once the window was locked against reviewers instead -- that no
    decision could be committed at all while a run was going. Per occupation, the two
    interleave safely and a reviewer waits seconds.

    The scrape cache is still saved once at the end. Nothing else writes it.

    `progress` and `should_continue` are optional and keyword-only, so existing callers
    are unaffected. A cooperative stop behaves exactly like the KeyboardInterrupt path
    below: everything unstarted goes back on the backlog.
    """
    prefixes = normalize_prefixes(prefixes)
    codes = select_onet_codes(prefixes)
    if not codes:
        return {"occupations": 0, "totals": {}}

    ordered = sorted(codes.items())
    if limit:
        ordered = ordered[:limit]
        logger.info("Limited to the first %d occupations.", len(ordered))

    cache = load_local_cache()

    totals = {"tools": 0, "hot": 0, "scraped": 0, "approved": 0, "pending": 0, "skipped": 0}
    processed = 0
    quota_exhausted = False
    stopped = False
    skills_total = 0

    try:
        for position, (onet_code, onet_title) in enumerate(ordered):
            # Cooperative stop, at the occupation boundary for the same reason as in
            # run_backlog: an occupation part-done is an occupation whose scraped work
            # would be discarded. Everything unstarted goes back on the backlog, which
            # is exactly what the KeyboardInterrupt handler below does.
            if should_continue is not None and not should_continue():
                stopped = True
                remaining = [code for code, _ in ordered[position:]]
                if remaining:
                    backlog_store.add_targets(
                        remaining, reason=backlog_store.REASON_INTERRUPTED
                    )
                logger.warning(
                    "Stop requested after %d occupation(s); %d returned to the backlog.",
                    processed, len(remaining),
                )
                break

            _emit(progress, {
                "event": "target_start", "target": onet_code, "title": onet_title,
            })

            try:
                with run_state.store_writer(
                    f"ingesting {onet_code}", timeout=STORE_WAIT_SECONDS
                ):
                    master = load_master()
                    timeseries = load_timeseries()
                    try:
                        counts = ingest_occupation(
                            onet_code, onet_title, master, timeseries, cache, force=force
                        )
                    finally:
                        # Inside the window and in a finally, so an occupation stopped
                        # by the quota wall still saves what it earned, and saves it
                        # from the copy loaded inside this window.
                        save_master(master)
                        save_timeseries(timeseries)
                        skills_total = len(master)
            except run_state.StoreBusy as busy:
                # Nothing else should be able to hold the store this long. Stop rather
                # than spin: the remaining occupations are queued and the next run picks
                # them up, and a run that cannot write is not a run.
                stopped = True
                remaining = [code for code, _ in ordered[position:]]
                backlog_store.add_targets(
                    remaining, reason=backlog_store.REASON_INTERRUPTED
                )
                logger.error(
                    "%s Stopped after %d occupation(s); %d returned to the backlog.",
                    busy, processed, len(remaining),
                )
                break
            except DailyQuotaExhausted as err:
                # The occupation that hit the wall AND everything after it go on the
                # backlog. The one that hit the wall is included because it is only
                # part-done; re-running it is cheap, since the freshness rule and the
                # scrape cache skip whatever already completed.
                quota_exhausted = True
                remaining = [onet_code] + [code for code, _ in ordered[position + 1:]]
                backlog_store.add_targets(remaining, reason=backlog_store.REASON_QUOTA)
                logger.error(
                    "%s Stopped after %d occupation(s); %d returned to the backlog.",
                    err, processed, len(remaining),
                )
                _emit(progress, {
                    "event": "quota_stopped", "target": onet_code,
                    "keys_tried": getattr(err, "keys_tried", 0),
                })
                break
            for key in totals:
                totals[key] += counts.get(key, 0)
            processed += 1
            _emit(progress, {
                "event": "target_done", "target": onet_code, "counts": counts,
            })
    except KeyboardInterrupt:
        # Everything not yet started is queued too, so a Ctrl-C during a long family
        # run is resumable rather than lost.
        remaining = [code for code, _ in ordered[processed:]]
        if remaining:
            backlog_store.add_targets(remaining, reason=backlog_store.REASON_INTERRUPTED)
        logger.warning("Interrupted after %d occupations; saving progress.", processed)
    finally:
        # The stores are already saved, per occupation, inside their writer windows.
        # Only the scrape cache is left, and nothing else writes it.
        save_local_cache(cache)

    logger.info(
        "Run complete. %d occupations, %d tools seen, %d hot, %d newly scraped, "
        "%d approved, %d awaiting review, %d already fresh.",
        processed, totals["tools"], totals["hot"], totals["scraped"],
        totals["approved"], totals["pending"], totals["skipped"],
    )
    return {
        "occupations": processed,
        "totals": totals,
        # Counted from the last save rather than a held reference: nothing here keeps
        # the store in memory between occupations any more.
        "skills": skills_total,
        "quota_exhausted": quota_exhausted,
        "stopped": stopped,
    }


def run_targets(targets: List[str], limit: int = None, force: bool = False) -> Dict[str, Any]:
    """
    Runs a mixed list of O*NET codes and SOC prefixes.

    Codes go one at a time; prefixes are batched into a single family run so the
    occupation catalogue is fetched once per family rather than once per target.
    Stops at the first quota wall -- continuing would just queue everything anyway.
    """
    targets = normalize_targets(targets)
    codes = [t for t in targets if "." in t]
    prefixes = [t for t in targets if "." not in t]

    totals = {"occupations": 0, "quota_exhausted": False, "skills": 0}

    for code in codes:
        result = ingest_single_occupation(code, force=force)
        totals["occupations"] += 1 if not result.get("quota_exhausted") else 0
        if result.get("quota_exhausted"):
            totals["quota_exhausted"] = True
            return totals

    if prefixes:
        result = run_ingestion(prefixes, limit=limit, force=force)
        totals["occupations"] += result.get("occupations", 0)
        totals["quota_exhausted"] = bool(result.get("quota_exhausted"))

    # Read the store size at the end rather than threading it out of each branch:
    # ingest_single_occupation does not report it, and a code-only run was printing
    # "0 skills in skills_master.json" while having just written hundreds.
    totals["skills"] = len(load_master())
    return totals


def run_backlog(
    force: bool = False,
    *,
    progress: Optional[Callable[[dict], None]] = None,
    should_continue: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Drains the backlog, removing each target only once it has completed.

    Removal happens per target rather than in one batch at the end: hitting the wall
    part-way through a drain must leave everything already finished finished, or
    tomorrow's run repeats work that has already been paid for.

    Codes and prefixes are both accepted, distinguished by the dot in a full O*NET code.

    `progress` and `should_continue` are optional and keyword-only, so every existing
    caller -- ingest.py included -- behaves exactly as it did. When supplied they let a
    UI report position and ask for a clean stop; see the stop check below.
    """
    targets = backlog_store.target_list()
    if not targets:
        return {"processed": 0, "remaining": 0, "quota_exhausted": False}

    logger.warning("Draining ingestion backlog: %s", backlog_store.describe())

    processed = 0
    quota_exhausted = False
    stopped = False

    for target in targets:
        # Cooperative stop, checked BEFORE dispatching a target and never inside one.
        # An occupation is an indivisible unit of paid work: it holds an in-memory
        # master that only its caller's finally block writes, so tearing out of the
        # middle would throw away every skill it had already scraped and audited.
        # Stopping here costs nothing, because the target is still on the backlog --
        # it is only removed once it completes -- so resuming is just another drain.
        if should_continue is not None and not should_continue():
            stopped = True
            logger.warning(
                "Stop requested. Ending the drain after %d target(s); %d remain queued.",
                processed, len(backlog_store.target_list()),
            )
            break

        _emit(progress, {"event": "target_start", "target": target})

        try:
            if "." in target:
                result = ingest_single_occupation(target, force=force)
            else:
                result = run_ingestion([target], force=force)
        except Exception:
            # One bad target must not abandon the rest of the queue. It stays on the
            # backlog so it is retried, and is visible in the log rather than silent.
            logger.exception("Backlog target %s failed; leaving it queued.", target)
            continue

        if result.get("quota_exhausted"):
            quota_exhausted = True
            _emit(progress, {"event": "quota_stopped", "target": target})
            break

        # A target that resolved to nothing is not proof the target is bad: the
        # CareerOneStop catalogue call fails the same way when the API is down. Count
        # the attempt and keep it queued, giving up only after several tries so a real
        # typo cannot wedge the queue permanently.
        produced_nothing = (
            result.get("occupations") == 0
            and not result.get("totals", {}).get("tools")
        )
        if produced_nothing:
            attempts = backlog_store.record_attempt(target)
            if attempts >= backlog_store.MAX_ATTEMPTS:
                backlog_store.remove_target(target)
                logger.error(
                    "Backlog target %s produced nothing on %d attempts. Giving up on "
                    "it; check that the code or prefix is valid.", target, attempts,
                )
            else:
                logger.warning(
                    "Backlog target %s produced nothing (attempt %d of %d). Keeping it "
                    "queued in case this was a connectivity failure.",
                    target, attempts, backlog_store.MAX_ATTEMPTS,
                )
            continue

        backlog_store.remove_target(target)
        processed += 1
        _emit(progress, {
            "event": "target_done",
            "target": target,
            "counts": result.get("totals") or {},
        })

    remaining = len(backlog_store.target_list())
    logger.info(
        "Backlog drain finished: %d processed, %d still queued.", processed, remaining
    )
    return {
        "processed": processed,
        "remaining": remaining,
        "quota_exhausted": quota_exhausted,
        # Present only so a caller that asked for a cooperative stop can tell that from
        # a natural finish. Existing callers never read it.
        "stopped": stopped,
    }


def plan_full_scrape(prefixes: List[str] = None) -> Dict[str, Any]:
    """
    Expands SOC prefixes into concrete O*NET codes and queues every one of them.

    Queueing CODES rather than the prefixes themselves is what makes a progress bar
    possible at all. select_onet_codes costs one network call per prefix, so a run that
    stored prefixes would have to re-expand them on every resume and could never state
    a total. Expanded once, up front, the backlog itself becomes the odometer: it
    already removes exactly one entry per completed occupation.

    Codes already queued are skipped by add_targets, so re-planning is idempotent and a
    second click cannot double the total.

    Slow by nature -- 23 network calls for the full set. Callers must run it on a
    background thread, not inside a request.
    """
    prefixes = normalize_prefixes(prefixes or ALL_PREFIXES)
    baseline = len(backlog_store.target_list())

    logger.info("Planning a full scrape over %d SOC group(s).", len(prefixes))
    codes = select_onet_codes(prefixes)

    added = backlog_store.add_targets(
        sorted(codes), reason=backlog_store.REASON_REQUESTED
    )
    logger.warning(
        "Planned %d occupation(s) across %d SOC group(s); %d newly queued, %d were "
        "already owed.",
        len(codes), len(prefixes), added, baseline,
    )
    return {
        "total": len(codes),
        "added": added,
        "baseline": baseline,
        "prefixes": prefixes,
    }

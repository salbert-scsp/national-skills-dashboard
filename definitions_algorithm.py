"""
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
from typing import Any, Dict, Iterable, List, Optional

import backlog as backlog_store
import json_store
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
    save_local_cache,
    scrape_and_validate_skill,
    search_onet_occupations,
)
from sortingalgorithmnew import calculate_ai_correlation

logger = logging.getLogger(__name__)

# A definition older than this is re-scraped and re-audited on the next run. This is
# the single biggest quota control: without it every run re-audits every skill.
RESUMMARIZE_AFTER_DAYS = 90

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

        # Resolve, gate, and conditionally audit. Everything expensive is in here.
        #
        # DailyQuotaExhausted is deliberately NOT caught. It means every key is spent,
        # so this skill was never audited and nothing about it may be written: not
        # approved, and not queued as pending either, because a pending card asks a
        # human to judge a definition that no audit has seen. Letting it propagate
        # leaves the skill exactly as it was and hands the caller the job of recording
        # the occupation on the backlog.
        counts["scraped"] += 1
        decision = scrape_and_validate_skill(item, cache=cache)

        status = STATUS_APPROVED if decision["auto_approve"] else STATUS_PENDING

        # A re-scrape replaces the evidence a second-pass finding was about, so the
        # finding has to go with it. Leaving it would show a reviewer a suggestion
        # panel describing a page this entry no longer points at.
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
    metrics = calculate_ai_correlation(
        entry["skill_name"],
        entry.get("category", ""),
        entry.get("wikipedia_summary", ""),
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
    """
    master = load_master()
    timeseries = load_timeseries()
    cache = load_local_cache()
    quota_exhausted = False
    counts: Dict[str, Any] = {}

    try:
        counts = ingest_occupation(
            onet_code, onet_code, master, timeseries, cache, force=force
        )
    except DailyQuotaExhausted as err:
        quota_exhausted = True
        backlog_store.add_targets([onet_code], reason=backlog_store.REASON_QUOTA)
        logger.error("%s Occupation %s returned to the backlog.", err, onet_code)
    finally:
        save_master(master)
        save_timeseries(timeseries)
        save_local_cache(cache)

    return {**counts, "quota_exhausted": quota_exhausted}


def run_ingestion(prefixes: List[str], limit: int = None, force: bool = False) -> Dict[str, Any]:
    """
    Full run across every occupation in the requested SOC families.

    `limit` caps the number of occupations, which is what makes a first run against a
    new family survivable: 11- alone is dozens of occupations and hundreds of skills.

    Both JSON files and the scrape cache are saved once at the end, and also saved if
    the run is interrupted -- a KeyboardInterrupt part-way through a long ingest must
    not throw away the work already paid for.
    """
    prefixes = normalize_prefixes(prefixes)
    codes = select_onet_codes(prefixes)
    if not codes:
        return {"occupations": 0, "totals": {}}

    ordered = sorted(codes.items())
    if limit:
        ordered = ordered[:limit]
        logger.info("Limited to the first %d occupations.", len(ordered))

    master = load_master()
    timeseries = load_timeseries()
    cache = load_local_cache()

    totals = {"tools": 0, "hot": 0, "scraped": 0, "approved": 0, "pending": 0, "skipped": 0}
    processed = 0
    quota_exhausted = False

    try:
        for position, (onet_code, onet_title) in enumerate(ordered):
            try:
                counts = ingest_occupation(
                    onet_code, onet_title, master, timeseries, cache, force=force
                )
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
                break
            for key in totals:
                totals[key] += counts.get(key, 0)
            processed += 1
    except KeyboardInterrupt:
        # Everything not yet started is queued too, so a Ctrl-C during a long family
        # run is resumable rather than lost.
        remaining = [code for code, _ in ordered[processed:]]
        if remaining:
            backlog_store.add_targets(remaining, reason=backlog_store.REASON_INTERRUPTED)
        logger.warning("Interrupted after %d occupations; saving progress.", processed)
    finally:
        save_master(master)
        save_timeseries(timeseries)
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
        "skills": len(master),
        "quota_exhausted": quota_exhausted,
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


def run_backlog(force: bool = False) -> Dict[str, Any]:
    """
    Drains the backlog, removing each target only once it has completed.

    Removal happens per target rather than in one batch at the end: hitting the wall
    part-way through a drain must leave everything already finished finished, or
    tomorrow's run repeats work that has already been paid for.

    Codes and prefixes are both accepted, distinguished by the dot in a full O*NET code.
    """
    targets = backlog_store.target_list()
    if not targets:
        return {"processed": 0, "remaining": 0, "quota_exhausted": False}

    logger.warning("Draining ingestion backlog: %s", backlog_store.describe())

    processed = 0
    quota_exhausted = False

    for target in targets:
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

    remaining = len(backlog_store.target_list())
    logger.info(
        "Backlog drain finished: %d processed, %d still queued.", processed, remaining
    )
    return {
        "processed": processed,
        "remaining": remaining,
        "quota_exhausted": quota_exhausted,
    }

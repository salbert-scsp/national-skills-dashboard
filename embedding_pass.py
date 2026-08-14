"""
Establishes which skills actually embed AI, by searching for each one.

    python3.11 embedding_pass.py                  # dry run, spends nothing
    python3.11 embedding_pass.py --preview 12     # probe 12, still write nothing
    python3.11 embedding_pass.py --apply          # probe, store, and rescore
    python3.11 embedding_pass.py --apply --force "LaTeX" "Slack"

WHY: embedded_ai_sim measured how a definition READS, not what a product SHIPS. Across
the live store it put LaTeX (a typesetting system from 1984), Thomson EndNote, SofTech
CADRA and Transoft AutoTURN into a class called "Non-Technical Embedded AI", because the
anchor's vocabulary -- workspace, workflow, integration, automated -- is the vocabulary of
any productivity tool. So the fact is searched for instead of inferred, and it enters the
engine as a three-state boolean worth EMBEDDED_AI_BOOST on the raw score.

WHAT IT SPENDS, and the two costs are wildly different:

  - DUCKDUCKGO: one search per skill at a MEASURED sustainable cadence of 25 seconds.
    This is the binding constraint. A full first pass over ~785 scored skills is about
    five and a half hours. Faster is not an option -- six queries inside ten seconds gets
    the IP blocked for minutes, and a blocked run returns `unknown` for everything while
    looking like progress.
  - GEMINI: one request per EMBEDDING_BATCH_SIZE skills, so about 131 for a full pass, on
    the embedding role's own key.

WHAT IT WRITES: embeds_ai and its evidence on the master entry, then re-scores the skill
so the verdict and the boost reach the time series.

THE RECHECK POLICY, which is what makes a second run nearly free:

    True      never re-probed. A product does not un-ship its AI features.
    False     re-probed only once RECHECK_AFTER_DAYS have passed. This is the state that
              genuinely goes stale: a 2026 negative for a tool that ships a copilot in
              2027 is the entire reason a window exists.
    unknown   re-probed EVERY run. It is not a finding, it is a gap -- a blocked search,
              a timeout, a quota wall -- and retrying is the only thing that closes it.

SCOPE: approved skills that have been scored, because those are the ones on the dashboard.
A skill approved later is probed the next time this runs; the pass is idempotent and
re-running it after a batch of approvals is the intended remedy rather than a workaround.
"""

import argparse
import datetime
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

import json_store
import storage
from agentic_source_check import (
    EMBEDDING_BATCH_SIZE,
    AuditUnavailable,
    get_source_checker,
)
from embedding_probe import search_embedding_evidence
from gemini_keys import ROLE_EMBEDDING, DailyQuotaExhausted
from json_store import STATUS_APPROVED
from sortingalgorithmnew import calculate_ai_correlation

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("embedding")

# How long a NEGATIVE verdict stands before it is worth asking again. Three months is a
# compromise the user set: long enough that a routine run costs almost nothing, short
# enough that a product shipping AI is picked up within a quarter -- which is also the
# granularity of the time series, so a changed verdict lands in its own snapshot rather
# than overwriting one that was true when it was written.
#
# A POSITIVE verdict has no window at all. See is_due.
RECHECK_AFTER_DAYS = 90

STATE_YES = "yes"
STATE_NO = "no"
STATE_UNKNOWN = "unknown"


def today() -> datetime.date:
    return datetime.date.today()


def state_of(record: Optional[Dict[str, Any]]) -> str:
    """The three-state verdict a cache record carries."""
    if not record:
        return STATE_UNKNOWN
    value = record.get("embeds_ai")
    if value is True:
        return STATE_YES
    if value is False:
        return STATE_NO
    return STATE_UNKNOWN


def embeds_ai_of(record: Optional[Dict[str, Any]]):
    """The cached verdict as the tri-state the scoring engine takes."""
    state = state_of(record)
    if state == STATE_YES:
        return True
    if state == STATE_NO:
        return False
    return None


def _checked_at(record: Dict[str, Any]) -> Optional[datetime.date]:
    try:
        return datetime.date.fromisoformat(str(record.get("checked_at") or ""))
    except ValueError:
        # A record with an unparseable date is treated as never checked rather than as
        # fresh. Re-probing costs 25 seconds; trusting a corrupt date costs a wrong
        # verdict standing indefinitely, since nothing else would ever revisit it.
        return None


def is_due(record: Optional[Dict[str, Any]], now: datetime.date = None) -> bool:
    """
    Whether this skill should be probed on this run.

    The whole recheck policy, in one place, so the cost of a run is a property of the
    cache rather than of whichever caller happens to be looping.
    """
    now = now or today()
    state = state_of(record)

    if state == STATE_UNKNOWN:
        return True
    if state == STATE_YES:
        return False

    checked = _checked_at(record)
    if checked is None:
        return True
    return (now - checked).days >= RECHECK_AFTER_DAYS


def is_in_scope(entry: Dict[str, Any], snapshot: Dict[str, Any]) -> bool:
    """
    Whether this skill is one the pass covers at all.

    Approved and scored. An unapproved skill is not on the dashboard, so a tag for it
    would describe something nobody can see; an unscored one has no snapshot for the
    verdict to land in.
    """
    return entry.get("status") == STATUS_APPROVED and bool(snapshot)


def plan_run(
    master: Dict[str, Any],
    timeseries: List[Dict[str, Any]],
    cache: Dict[str, Any],
    forced: Optional[set] = None,
    now: datetime.date = None,
) -> Dict[str, Any]:
    """
    Works out what this run would do, WITHOUT touching the network.

    Returned whole rather than printed, so the dry run and the real run report from the
    same computation and cannot disagree about what a run costs.
    """
    now = now or today()
    forced = forced or set()
    newest = json_store.latest_snapshots(timeseries)

    due: List[Tuple[str, Dict[str, Any]]] = []
    counts = {"in_scope": 0, "cached_yes": 0, "cached_no_fresh": 0,
              "due_recheck": 0, "due_unknown": 0, "forced": 0}

    for name, entry in sorted(master.items()):
        snapshot = newest.get(name) or {}
        if not is_in_scope(entry, snapshot):
            continue
        counts["in_scope"] += 1

        record = cache.get(name)
        if name in forced:
            counts["forced"] += 1
            due.append((name, entry))
            continue

        if not is_due(record, now):
            if state_of(record) == STATE_YES:
                counts["cached_yes"] += 1
            else:
                counts["cached_no_fresh"] += 1
            continue

        if state_of(record) == STATE_UNKNOWN:
            counts["due_unknown"] += 1
        else:
            counts["due_recheck"] += 1
        due.append((name, entry))

    return {"due": due, "counts": counts}


def chunks(items: List[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def record_verdict(
    cache: Dict[str, Any],
    name: str,
    verdict: Optional[Dict[str, Any]],
    error: Optional[str] = None,
    now: datetime.date = None,
) -> Dict[str, Any]:
    """
    Writes one verdict to the cache, including an unknown one.

    An unknown IS recorded, with the reason it failed, and that is not a contradiction of
    the retry rule -- is_due ignores the date on an unknown, so recording it changes
    nothing about when it is asked again. What it buys is a store that can answer "how
    many skills were never actually established, and why", instead of leaving that to be
    inferred from an absence.
    """
    now = now or today()
    if verdict is None:
        record = {
            "embeds_ai": None,
            "evidence": "",
            "evidence_url": None,
            "confidence": None,
            "error": error or "no_verdict",
            "checked_at": now.isoformat(),
        }
    else:
        record = {
            "embeds_ai": bool(verdict.get("embeds_ai")),
            "evidence": str(verdict.get("evidence") or "").strip(),
            "evidence_url": verdict.get("evidence_url") or None,
            "confidence": verdict.get("confidence"),
            "error": None,
            "checked_at": now.isoformat(),
        }
    cache[name] = record
    return record


def apply_to_entry(entry: Dict[str, Any], record: Dict[str, Any]) -> None:
    """Copies a cache record onto the master entry the dashboard reads."""
    entry["embeds_ai"] = record.get("embeds_ai")
    entry["embeds_ai_evidence"] = record.get("evidence") or ""
    entry["embeds_ai_evidence_url"] = record.get("evidence_url")
    entry["embeds_ai_confidence"] = record.get("confidence")
    entry["embeds_ai_checked_at"] = record.get("checked_at")


def score_entry(name: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Re-scores one skill with its verdict applied.

    Deliberately NOT flagship substitution. That is flagship_pass's job and it reads
    fields this pass does not write; calling calculate_ai_correlation here keeps the two
    passes independent, and re-running either one after the other converges.
    """
    return calculate_ai_correlation(
        name,
        entry.get("category", ""),
        entry.get("wikipedia_summary", "") or "",
        entry.get("flagship_note", "") or "",
        entry.get("embeds_ai"),
    )


def estimate(due_count: int) -> str:
    searches = due_count
    requests = -(-due_count // EMBEDDING_BATCH_SIZE)
    from embedding_probe import POLITENESS_DELAY
    hours = (searches * POLITENESS_DELAY) / 3600.0
    return (f"{searches} search(es) at {POLITENESS_DELAY:.0f}s each, about {hours:.1f} "
            f"hour(s), plus {requests} Gemini request(s)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--apply", action="store_true",
                        help="store the verdicts and rescore; without it nothing is saved")
    parser.add_argument("--preview", type=int, default=0, metavar="N",
                        help="probe only the first N due skills")
    parser.add_argument("--show", type=int, default=15,
                        help="how many due skills to list in the dry run (default 15)")
    parser.add_argument("--force", nargs="*", default=[], metavar="SKILL",
                        help="re-probe these skills whatever the cache says")
    args = parser.parse_args(argv)

    master = json_store.load_master()
    timeseries = json_store.load_timeseries()
    cache = storage.load_embedding_cache()

    plan = plan_run(master, timeseries, cache, forced=set(args.force))
    due, counts = plan["due"], plan["counts"]

    limit = args.preview if args.preview > 0 else len(due)
    selected = due[:limit]

    print(f"In scope (approved and scored): {counts['in_scope']}")
    print(f"  cached yes, never re-asked ..... {counts['cached_yes']}")
    print(f"  cached no, still fresh ......... {counts['cached_no_fresh']}")
    print(f"  due, negative older than {RECHECK_AFTER_DAYS}d ... {counts['due_recheck']}")
    print(f"  due, never established ......... {counts['due_unknown']}")
    if counts["forced"]:
        print(f"  forced ......................... {counts['forced']}")
    print(f"\nDue this run: {len(due)}   selected: {len(selected)}")
    print(f"Cost: {estimate(len(selected))}\n")

    for name, _ in selected[:args.show]:
        print("  %-9s %s" % (state_of(cache.get(name)), name[:60]))
    if len(selected) > args.show:
        print(f"  ... and {len(selected) - args.show} more")

    if not args.apply:
        print("\nDry run. Nothing was searched, sent or written. "
              "Re-run with --apply to spend the above.")
        return 0

    if not selected:
        print("Nothing to do.")
        return 0

    checker = get_source_checker(ROLE_EMBEDDING)
    yes = no = unknown = 0
    moved: List[str] = []

    for batch in chunks(selected, EMBEDDING_BATCH_SIZE):
        # Search first, then grade what was actually found. Searching the whole batch
        # before spending a Gemini request means a rate-limited stretch costs nothing but
        # time -- the items with no results are simply not sent.
        gradeable = []
        for name, entry in batch:
            found = search_embedding_evidence(name)
            if found["error"]:
                record_verdict(cache, name, None, error=found["error"])
                unknown += 1
                continue
            gradeable.append({
                "item_id": name,
                "skill_name": name,
                "definition": entry.get("wikipedia_summary") or "",
                "results": found["results"],
            })

        if not gradeable:
            continue

        try:
            answers = checker.grade_embedding_batch(gradeable)
        except DailyQuotaExhausted as err:
            # Every key available to this role is spent. Stop and save what has already
            # been established rather than losing the search time already paid for.
            logger.warning("Daily quota exhausted: %s. Saving what was answered.", err)
            break
        except AuditUnavailable:
            logger.warning("Gemini unreachable for a batch of %d. They stay unknown.",
                           len(gradeable))
            for item in gradeable:
                record_verdict(cache, item["item_id"], None, error="grader_unavailable")
                unknown += 1
            continue

        for name, entry in batch:
            if name not in {item["item_id"] for item in gradeable}:
                continue
            verdict = answers.get(name)
            if verdict is None:
                # Absent from the answer. Never False -- see grade_embedding_batch.
                record_verdict(cache, name, None, error="missing_from_batch")
                unknown += 1
                continue

            record = record_verdict(cache, name, verdict)
            previous = entry.get("embeds_ai")
            apply_to_entry(entry, record)

            if record["embeds_ai"]:
                yes += 1
            else:
                no += 1

            metrics = score_entry(name, entry)
            json_store.upsert_snapshot(
                timeseries, name, metrics,
                entry.get("onet_codes", []), entry.get("onet_titles", []),
            )
            if previous != record["embeds_ai"]:
                moved.append(name)
                logger.info(
                    "%s: embeds_ai %s -> %s, ai %.3f (+%.2f) -> %s. %s",
                    name, previous, record["embeds_ai"],
                    metrics["ai_score_base"], metrics["embedded_ai_boost"],
                    metrics["category_bucket"], record["evidence"][:90],
                )

        storage.save_embedding_cache(cache)

    storage.save_embedding_cache(cache)
    json_store.save_master(master)
    json_store.save_timeseries(timeseries)

    print(f"\nProbed {yes + no + unknown}: {yes} embed AI, {no} do not, "
          f"{unknown} could not be established.")
    print(f"{len(moved)} skill(s) changed verdict.")
    if unknown:
        print("The unknowns are retried on the next run; nothing was concluded about them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
STAGE: Storage

Resumable ingestion backlog.

When the Gemini daily quota runs out mid-run, the work that never happened has to go
somewhere. It must NOT go into the review queue: those skills were never audited, so
queueing them would ask a human to approve a definition nothing has vetted, and a single
quota wall would produce hundreds of such cards. Instead the OCCUPATIONS still owed are
written here, and the next run drains this file before asking for anything new.

    ingestion_backlog.json
    {"updated": "2026-08-06",
     "reason": "gemini_daily_quota_exhausted",
     "targets": [{"target": "15-2051.00", "requested": "2026-08-06", "reason": "..."}]}

Granularity is the occupation, not the skill. Re-running an occupation is cheap because
the 90-day freshness rule and the scrape cache skip everything already finished, and
tracking half-finished occupations at skill level would be a second, subtler store to
keep in agreement with the first.

Writes go through storage.save_backlog_raw, which is atomic, so a crash cannot leave a
truncated backlog -- the one file whose loss would silently drop queued work.
"""

import datetime
import logging
from typing import Any, Dict, List

import storage

logger = logging.getLogger(__name__)

BACKLOG_FILE = storage.BACKLOG_FILE

REASON_QUOTA = "gemini_daily_quota_exhausted"
REASON_REQUESTED = "requested"
REASON_INTERRUPTED = "interrupted"


def _empty() -> Dict[str, Any]:
    return {"updated": None, "reason": None, "targets": []}


def load_backlog() -> Dict[str, Any]:
    """
    Reads the backlog, tolerating absence and corruption.

    Unlike the skills store, a damaged backlog is NOT fatal. Losing it means re-entering
    a handful of O*NET codes; refusing to start the app over it would be a worse trade.
    It is still logged as an error rather than passed over.
    """
    try:
        data = storage.load_backlog_raw()
    except storage.StorageUnreadable:
        logger.exception("%s could not be read. Starting with an empty backlog.", BACKLOG_FILE)
        return _empty()

    if data is None:
        return _empty()

    if not isinstance(data, dict) or not isinstance(data.get("targets"), list):
        logger.error("%s has an unexpected shape. Ignoring it.", BACKLOG_FILE)
        return _empty()

    # Drop malformed entries rather than letting one bad record break the drain loop.
    data["targets"] = [
        entry for entry in data["targets"]
        if isinstance(entry, dict) and str(entry.get("target", "")).strip()
    ]
    return data


def save_backlog(backlog: Dict[str, Any]) -> None:
    backlog["updated"] = datetime.date.today().isoformat()
    storage.save_backlog_raw(backlog)


def is_empty() -> bool:
    return not load_backlog()["targets"]


def target_list() -> List[str]:
    """Just the target strings, in queue order."""
    return [entry["target"] for entry in load_backlog()["targets"]]


def add_targets(targets: List[str], reason: str = REASON_QUOTA) -> int:
    """
    Queues targets for a later run, skipping any already present.

    Deduping matters: a run that hits the wall twice on the same occupation would
    otherwise queue it twice and process it twice tomorrow. Returns how many were
    actually added.
    """
    cleaned = [str(t).strip() for t in targets if str(t).strip()]
    if not cleaned:
        return 0

    backlog = load_backlog()
    existing = {entry["target"] for entry in backlog["targets"]}
    today = datetime.date.today().isoformat()

    added = 0
    for target in cleaned:
        if target in existing:
            continue
        backlog["targets"].append(
            {"target": target, "requested": today, "reason": reason}
        )
        existing.add(target)
        added += 1

    if added:
        backlog["reason"] = reason
        save_backlog(backlog)
        logger.warning(
            "Queued %d target(s) for a later run (%s): %s",
            added, reason, ", ".join(cleaned[:8]) + ("..." if len(cleaned) > 8 else ""),
        )
    return added


MAX_ATTEMPTS = 3


def record_attempt(target: str) -> int:
    """
    Counts a failed attempt at a target and returns the new total.

    A target that produces nothing is NOT dropped on the first try. "No occupations
    matched" looks identical whether the family genuinely does not exist or
    CareerOneStop was unreachable, and silently discarding queued work because of a
    momentary outage is the worse of the two errors. After MAX_ATTEMPTS it is given up
    on loudly, so a typo cannot wedge the queue forever either.
    """
    data = load_backlog()
    for entry in data["targets"]:
        if entry["target"] == target:
            entry["attempts"] = int(entry.get("attempts", 0)) + 1
            save_backlog(data)
            return entry["attempts"]
    return 0


def attempts_for(target: str) -> int:
    for entry in load_backlog()["targets"]:
        if entry["target"] == target:
            return int(entry.get("attempts", 0))
    return 0


def remove_target(target: str) -> bool:
    """
    Removes one completed target and saves immediately.

    Saved per target rather than once at the end of the drain: if the quota wall is hit
    part-way through draining, everything already finished must stay finished. Batching
    the save would replay completed occupations tomorrow.
    """
    backlog = load_backlog()
    before = len(backlog["targets"])
    backlog["targets"] = [e for e in backlog["targets"] if e["target"] != target]

    if len(backlog["targets"]) == before:
        return False

    if not backlog["targets"]:
        backlog["reason"] = None
    save_backlog(backlog)
    return True


def clear_backlog() -> None:
    save_backlog(_empty())
    logger.info("Ingestion backlog cleared.")


def describe() -> str:
    """One-line human summary, for the CLI banner and the web notice."""
    backlog = load_backlog()
    targets = backlog["targets"]
    if not targets:
        return "Backlog empty."

    names = ", ".join(e["target"] for e in targets[:6])
    if len(targets) > 6:
        names += f", and {len(targets) - 6} more"
    stamp = backlog.get("updated") or "an earlier run"
    return f"{len(targets)} target(s) queued since {stamp}: {names}"

"""
Live state of a web-triggered ingestion run, and the lock that keeps writers apart.

Two halves, on purpose.

The IN-MEMORY half -- a Lock, an Event, and a running flag -- is the control surface.
It decides who may start, whether to keep going, and who may write the store. It is
consulted between every occupation, so it must not touch disk, and it must be atomic
against a second request arriving on another thread a millisecond later.

The ON-DISK half is the reporting surface: the facts a restart or a new day would
otherwise lose. How many occupations the run planned, when it hit the quota wall,
whether the last run died mid-flight.

NEITHER HALF STORES WHAT WORK REMAINS. That is ingestion_backlog.json, which already
removes each target only once it has completed and saves immediately when it does.
Duplicating it here would create two answers to the same question, and the one written
less often would win. `completed` is therefore derived on read, never stored.

Extending backlog.py to carry this instead was rejected for a correctness reason rather
than a stylistic one: its `reason` is a single scalar for the whole file, so a
REASON_PAUSED would be silently overwritten by the next add_targets(REASON_QUOTA).

THE WRITER LOCK
---------------
Also here, because it is the same concern: who is allowed to touch skills_master.json
right now. Every background writer of the store -- the scrape and the second pass --
holds STORE_WRITER_LOCK for its whole run. Without it, two of them doing
load_master -> mutate -> save_master concurrently means whichever saves last silently
erases the other's work, INCLUDING any human approvals made in between. That hazard
predates this module; the scrape button is what makes it near-certain rather than rare.
"""

import datetime
import logging
import threading
from typing import Any, Callable, Dict, Optional

import backlog as backlog_store
import storage

logger = logging.getLogger(__name__)

SCHEMA = 1

# Run states. `planning` is a sub-state of running rather than its own value: the run
# slot is held, but the total is not known yet because select_onet_codes has not
# finished. The UI branches on total == 0 to tell them apart.
IDLE = "idle"
RUNNING = "running"
PAUSED = "paused"
QUOTA_STOPPED = "quota_stopped"
FINISHED = "finished"
INTERRUPTED = "interrupted"

TERMINAL_STATES = (IDLE, PAUSED, QUOTA_STOPPED, FINISHED, INTERRUPTED)

# ---------------------------------------------------------------------------
# In-memory control surface
# ---------------------------------------------------------------------------

# Guards _running and every write to the state file below.
_lock = threading.Lock()

# True between a successful try_claim() and its release().
_running = False

# Set to ask the worker to stop cleanly after the current occupation.
_stop = threading.Event()

# Held for the whole of any background run that writes skills_master.json. Reentrant is
# NOT wanted here: a writer that somehow tried to claim it twice is a bug worth
# deadlocking on in development rather than papering over.
STORE_WRITER_LOCK = threading.Lock()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty() -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "state": IDLE,
        "run_id": None,
        "label": "",
        "total": 0,
        "baseline": 0,
        "current_target": None,
        "current_title": None,
        "started": None,
        "updated": None,
        "finished": None,
        "counts": {},
        "quota": {"stopped_at": None, "resets_on": None, "keys_tried": 0},
        "last_error": None,
    }


def _load() -> Dict[str, Any]:
    """
    Reads the run record, tolerating absence and corruption.

    Like the backlog and unlike the skills store, a damaged record here is not fatal:
    it describes a run, it is not the run's output. Losing it costs a progress bar.
    """
    try:
        data = storage.load_run_state_raw()
    except storage.StorageUnreadable:
        logger.exception("%s could not be read. Reporting an idle run.",
                         storage.RUN_STATE_FILE)
        return _empty()

    if not isinstance(data, dict):
        return _empty()

    record = _empty()
    record.update(data)
    return record


def _write(record: Dict[str, Any]) -> None:
    record["updated"] = _now()
    try:
        storage.save_run_state_raw(record)
    except OSError:
        # Not fatal. Losing the record costs the progress bar its position, which is
        # a far better trade than aborting a run that has already spent hours of quota.
        logger.exception("Could not write %s; progress reporting will be stale.",
                         storage.RUN_STATE_FILE)


# ---------------------------------------------------------------------------
# Claiming and releasing the run slot
# ---------------------------------------------------------------------------

def try_claim(label: str) -> Optional[str]:
    """
    Atomically takes the run slot, or returns None if a run already holds it.

    Compare-and-set under one lock, NOT "read the file, then write it". Two clicks
    arriving on two threadpool threads a millisecond apart would both see state=idle on
    disk and both start; both would then load_master(), ingest different occupations,
    and save_master() over each other. Nothing would be corrupt -- the writes are
    atomic -- but one thread's entire run would be silently discarded.

    The disabled button in the UI is cosmetic. This is what makes it safe.
    """
    global _running
    with _lock:
        if _running:
            return None
        _running = True
        _stop.clear()

        run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        record = _empty()
        record.update({
            "state": RUNNING,
            "run_id": run_id,
            "label": label,
            "started": _now(),
            "finished": None,
        })
        _write(record)
        return run_id


def release(final_state: str, *, error: str = None) -> None:
    """Clears the run slot and stamps how the run ended."""
    global _running
    with _lock:
        _running = False
        _stop.clear()
        record = _load()
        record["state"] = final_state
        record["finished"] = _now()
        record["current_target"] = None
        record["current_title"] = None
        if error:
            record["last_error"] = error
        _write(record)
    logger.info("Ingestion run ended: %s.", final_state)


def is_running() -> bool:
    with _lock:
        return _running


def request_stop() -> bool:
    """
    Asks the worker to stop after the occupation it is currently in.

    Returns False if nothing is running. Does NOT block: the worker returns, its thread
    exits, and everything unfinished is still on the backlog. Resume is then a fresh
    drain -- the same code path as start, so there is one running state rather than two.
    """
    with _lock:
        if not _running:
            return False
    _stop.set()
    logger.info("Stop requested; the run will end after the current occupation.")
    return True


def should_continue() -> bool:
    """The cooperative-stop predicate handed to the ingestion loop."""
    return not _stop.is_set()


def was_stopped() -> bool:
    return _stop.is_set()


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

def set_total(total: int, baseline: int) -> None:
    """
    Records the planned size of the run, once planning has finished.

    `baseline` is how many unrelated targets were already queued when the run was
    planned. Without subtracting it, an ad-hoc code submitted through the ingest form
    would make the progress bar walk backwards.
    """
    with _lock:
        record = _load()
        record["total"] = int(total)
        record["baseline"] = int(baseline)
        _write(record)


def on_event(event: Dict[str, Any]) -> None:
    """
    Progress callback handed to the ingestion loop.

    Deliberately tolerant: this is a reporting concern, and the ingestion module wraps
    every call to it in a blanket except for the same reason. Nothing in here may be
    able to abort a run.
    """
    kind = event.get("event")
    with _lock:
        record = _load()

        if kind == "target_start":
            record["current_target"] = event.get("target")
            record["current_title"] = event.get("title")
        elif kind == "target_done":
            counts = record.get("counts") or {}
            for key, value in (event.get("counts") or {}).items():
                counts[key] = counts.get(key, 0) + int(value or 0)
            counts["occupations"] = counts.get("occupations", 0) + 1
            record["counts"] = counts
        elif kind == "quota_stopped":
            record["quota"] = {
                "stopped_at": _now(),
                "resets_on": quota_reset_date().isoformat(),
                "keys_tried": int(event.get("keys_tried") or 0),
            }
        else:
            return

        _write(record)


def quota_reset_date() -> datetime.date:
    """
    The date the daily Gemini quota resets, which is always tomorrow.

    Derived rather than stored: gemini_keys._load_state discards any state whose date
    is not today, so local midnight IS the reset as far as this system is concerned.

    Caveat worth knowing before quoting a time to anyone: Gemini's free-tier daily
    window is Pacific. On a machine in another timezone the real reset can be hours off
    this date's midnight, which is why callers state a date and never a time.
    """
    return datetime.date.today() + datetime.timedelta(days=1)


def snapshot() -> Dict[str, Any]:
    """
    The run record, with progress derived from the backlog.

    `completed` is derived rather than stored because the backlog is the only thing
    that knows, across a restart and across days, what has actually finished -- it
    removes exactly one entry per completed occupation and saves as it does. A stored
    counter would be a second answer that drifts every time the CLI drains a target.
    """
    record = _load()
    queued = len(backlog_store.target_list())
    total = int(record.get("total") or 0)
    baseline = int(record.get("baseline") or 0)

    record["queued"] = queued
    record["completed"] = max(0, min(total, total + baseline - queued))
    record["percent"] = round(100.0 * record["completed"] / total, 1) if total else 0.0
    record["running"] = is_running()
    # Planning is running-with-no-total-yet. Named here so the template does not have
    # to know that rule.
    record["planning"] = record.get("state") == RUNNING and total == 0
    return record


def reconcile_on_startup() -> None:
    """
    Repairs a record left claiming a run that this process is plainly not doing.

    Called from the app's lifespan. In-memory state does not survive a restart, so a
    record still saying `running` means the server died mid-run. Nothing is lost: the
    backlog still holds every occupation that had not completed, and at most the one in
    flight is replayed -- which the 90-day freshness rule and the scrape cache make
    nearly free.
    """
    record = _load()
    if record.get("state") == RUNNING:
        logger.warning(
            "Run %s was still marked running at startup, so the server stopped during "
            "it. Marking it interrupted; %d target(s) remain queued and resume where "
            "they left off.",
            record.get("run_id"), len(backlog_store.target_list()),
        )
        record["state"] = INTERRUPTED
        record["finished"] = _now()
        record["current_target"] = None
        record["current_title"] = None
        _write(record)

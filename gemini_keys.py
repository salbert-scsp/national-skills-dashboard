"""
Gemini API key pool with daily-quota rotation.

The free tier caps requests per DAY as well as per minute. When a key hits its daily
cap, no amount of backoff helps -- the only options are another key or tomorrow. This
module owns which key is in use, which ones are spent today, and when to give up.

Key order is deliberate: GEMINI_API_KEY from .env first, then gemini_keys.json in file
order. The pool is rebuilt from scratch every process, so each run starts at the .env
key again, which is what makes "try the main key again next time you run" true without
any extra bookkeeping.

Exhaustion state lives in gemini_key_state.json, stamped with the date. A state file
from a previous day means the quota has reset, so it is discarded and the pool starts
clean. That date check IS the resume mechanism -- there is no scheduler and nothing to
remember to reset.

The state file stores FINGERPRINTS, never keys. gemini_keys.json is already a file full
of live secrets; a second one would be a second thing to leak.
"""

import datetime
import hashlib
import logging
import os
from typing import List, Optional

from dotenv import load_dotenv

import storage

logger = logging.getLogger(__name__)

load_dotenv()

KEYS_FILE = storage.GEMINI_KEYS_FILE
STATE_FILE = storage.GEMINI_KEY_STATE_FILE

ENV_KEY_LABEL = "env:GEMINI_API_KEY"


class DailyQuotaExhausted(RuntimeError):
    """
    Every configured key has hit its daily Gemini quota.

    Raised rather than returned so it cannot be mistaken for an audit verdict and
    cannot be accidentally ignored by a caller that only checks a return value. The
    ingestion loop catches it, records what is left on the backlog, and stops.
    """

    def __init__(self, keys_tried: int):
        super().__init__(
            f"All {keys_tried} Gemini key(s) have hit their daily quota. "
            f"Remaining work has been saved to the backlog; re-run tomorrow."
        )
        self.keys_tried = keys_tried


def fingerprint(key: str) -> str:
    """Short, stable, non-reversible id for a key, safe to write to disk and logs."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _load_key_file() -> List[dict]:
    """
    Reads gemini_keys.json.

    Accepts either a bare list of key strings or a list of {"label", "key"} objects, so
    the file can be written the obvious way and still carry names that make the log
    readable. A missing file is normal -- it means only the .env key is configured.
    """
    try:
        raw = storage.load_gemini_keys_raw()
    except storage.StorageUnreadable:
        # Loud, not silent: a malformed key file would otherwise look identical to
        # "no spare keys configured", and the first quota wall would stop the run
        # with no hint that four perfectly good keys were sitting unread on disk.
        logger.exception("%s could not be read. Falling back to the .env key only.", KEYS_FILE)
        return []

    if raw is None:
        return []

    # Some editors leave a top-level {"keys": [...]} wrapper; accept it rather than
    # failing on a shape that is obviously intended.
    if isinstance(raw, dict):
        raw = raw.get("keys", [])

    if not isinstance(raw, list):
        logger.error("%s should hold a list of keys; ignoring it.", KEYS_FILE)
        return []

    entries = []
    for index, item in enumerate(raw, 1):
        if isinstance(item, str):
            key = item.strip()
            label = f"{KEYS_FILE}#{index}"
        elif isinstance(item, dict):
            key = str(item.get("key", "")).strip()
            label = str(item.get("label") or f"{KEYS_FILE}#{index}")
        else:
            continue
        if key:
            entries.append({"label": label, "key": key})
    return entries


def _load_state() -> dict:
    """
    Reads today's exhaustion state, discarding anything from a previous day.

    A stale file is not an error, it is the normal signal that the quota reset.
    """
    today = datetime.date.today().isoformat()

    try:
        state = storage.load_gemini_key_state_raw()
    except storage.StorageUnreadable:
        logger.warning("%s is unreadable; treating every key as available.", STATE_FILE)
        return {"date": today, "exhausted": []}

    if not isinstance(state, dict):
        return {"date": today, "exhausted": []}

    if state.get("date") != today:
        logger.info(
            "Gemini key state is from %s, not today. Daily quotas have reset.",
            state.get("date", "an unknown date"),
        )
        return {"date": today, "exhausted": []}

    exhausted = [f for f in state.get("exhausted", []) if isinstance(f, str)]
    return {"date": today, "exhausted": exhausted}


def _save_state(state: dict) -> None:
    try:
        storage.save_gemini_key_state_raw(state)
    except OSError:
        # Not fatal. Losing the state means retrying a spent key on the next run,
        # which costs one 429 -- far better than aborting a working run.
        logger.exception("Could not write %s; rotation state will not survive.", STATE_FILE)


class GeminiKeyPool:
    """
    Ordered pool of Gemini keys with per-day exhaustion tracking.

    Construct one per process. `current()` gives the key to use; when it hits the daily
    wall, `mark_exhausted_and_advance()` retires it and moves on, returning False when
    nothing is left.
    """

    def __init__(self) -> None:
        self.keys: List[dict] = []

        env_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if env_key:
            self.keys.append({"label": ENV_KEY_LABEL, "key": env_key})

        # Deduped against the .env key: the same key listed twice would be "rotated"
        # to itself and report two failures for one quota.
        seen = {fingerprint(entry["key"]) for entry in self.keys}
        for entry in _load_key_file():
            if fingerprint(entry["key"]) not in seen:
                seen.add(fingerprint(entry["key"]))
                self.keys.append(entry)

        if not self.keys:
            raise ValueError(
                "No Gemini keys configured. Set GEMINI_API_KEY in .env, or add keys to "
                f"{KEYS_FILE}."
            )

        self.state = _load_state()
        self.index = 0
        self._advance_past_exhausted()

        available = sum(1 for entry in self.keys if not self._is_exhausted(entry))
        logger.info(
            "Gemini key pool: %d key(s) configured, %d available today.",
            len(self.keys), available,
        )

    # -- internals ---------------------------------------------------------

    def _is_exhausted(self, entry: dict) -> bool:
        return fingerprint(entry["key"]) in self.state["exhausted"]

    def _advance_past_exhausted(self) -> None:
        """Moves the cursor to the first key not already spent today."""
        while self.index < len(self.keys) and self._is_exhausted(self.keys[self.index]):
            logger.info(
                "Skipping Gemini key %s: already exhausted today.",
                self.keys[self.index]["label"],
            )
            self.index += 1

    # -- public ------------------------------------------------------------

    def all_exhausted(self) -> bool:
        return self.index >= len(self.keys)

    def current(self) -> Optional[dict]:
        """The key to use right now, or None when every key is spent."""
        if self.all_exhausted():
            return None
        return self.keys[self.index]

    def current_key(self) -> Optional[str]:
        entry = self.current()
        return entry["key"] if entry else None

    def current_label(self) -> str:
        entry = self.current()
        return entry["label"] if entry else "none"

    def mark_exhausted_and_advance(self) -> bool:
        """
        Retires the current key for the rest of today and moves to the next.

        Returns True if another key is now available, False if the pool is spent.
        Persists immediately, so a crash mid-run does not cause tomorrow's first
        request to re-probe a key that is already known to be finished today.
        """
        entry = self.current()
        if entry is None:
            return False

        marker = fingerprint(entry["key"])
        if marker not in self.state["exhausted"]:
            self.state["exhausted"].append(marker)
            _save_state(self.state)

        logger.warning(
            "Gemini key %s has hit its daily quota. Retired for today.", entry["label"]
        )

        self.index += 1
        self._advance_past_exhausted()

        if self.all_exhausted():
            logger.error(
                "All %d Gemini key(s) are exhausted for today.", len(self.keys)
            )
            return False

        logger.warning("Rotating to Gemini key %s.", self.current_label())
        return True

    def total_keys(self) -> int:
        return len(self.keys)

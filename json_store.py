"""
STAGE: Storage

The two-file local JSON store.

With no database there is no second copy of anything, so this module is deliberately
paranoid about the two ways flat-file storage loses data: a truncated write, and a
read that silently swallows corruption and hands back an empty dict.

    skills_master.json      object keyed by skill name; upserted every run
    skills_timeseries.json  flat list of snapshots; one record per (skill, quarter)

Every module that touches those files goes through here. Four modules read them
(definitions_algorithm, dashboardtables, review_actions, second_pass) and three write
them, and duplicating the atomic-write dance in each is how the two files drift apart.

This module owns what a skill RECORD is: the upsert rules, the never-regress
invariants, the occupations upgrade. It no longer owns where the bytes live -- that is
storage.py, which every I/O call below delegates to. The split is what lets the store
move to Cloud SQL without any of the record logic here noticing.
"""

import datetime
import logging
from typing import Any, Dict, List

import storage

logger = logging.getLogger(__name__)

# Re-exported from storage so existing callers and log lines keep working unchanged.
MASTER_FILE = storage.MASTER_FILE
TIMESERIES_FILE = storage.TIMESERIES_FILE

# Review states a master entry can hold. Strings rather than the SQL integers,
# because a JSON file is read by humans and `"status": -2` explains nothing.
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
VALID_STATUSES = (STATUS_PENDING, STATUS_APPROVED, STATUS_REJECTED)

# Set on a skill a viewer reported from the dashboard. It lives HERE, beside the statuses,
# because it is the one gate reason that qualifies a status: a reported skill is pending
# AND still published, and both the writer (review_actions) and the reader
# (dashboardtables) have to agree on the spelling for that to hold. It was previously a
# constant in review_actions and a bare string literal in two other modules.
REPORT_GATE_REASON = "reported_by_viewer"


class StoreCorrupted(RuntimeError):
    """
    A store file exists but could not be parsed.

    Raised rather than returning empty. Falling back to {} would make a corrupted
    master file look like a first run, and the next save would overwrite the damaged
    file with a nearly-empty one -- turning a recoverable problem into data loss.
    """


# --------------------------------------------------------------------------
# Low-level IO
# --------------------------------------------------------------------------

# Distinguishes "the store has never been written" from "the store holds null". Only
# the former is a normal first run; storage returns None for both, so the sentinel is
# what keeps the two apart.
_MISSING = object()


def _loaded(raw, path: str, default):
    """
    Turns a storage read into a value, reporting absence exactly as before.

    Absence is normal and logged at INFO; corruption never reaches here, because
    storage raises and the callers below translate that into StoreCorrupted.
    """
    if raw is None:
        logger.info("%s does not exist yet; starting from empty.", path)
        return default
    return raw


def _read_json(path: str, default):
    """
    Reads a JSON file, tolerating absence but never corruption.

    Kept as a private generic for any caller that needs a one-off read against the
    store's failure policy. The two real stores use storage's named helpers instead.
    """
    try:
        raw = storage.read_json(path, default=_MISSING)
    except storage.StorageUnreadable as err:
        raise StoreCorrupted(
            f"{path} exists but could not be read ({err}). Refusing to continue and "
            f"overwrite it. Inspect or delete the file, then re-run."
        ) from err
    if raw is _MISSING:
        logger.info("%s does not exist yet; starting from empty.", path)
        return default
    return raw


def atomic_write(path: str, payload) -> None:
    """
    Writes JSON so that the file on disk is either the old content or the new one.

    Now a thin alias over storage.write_json, which holds the temp-file-and-replace
    dance. Kept under this name because it is part of this module's published surface;
    the keywords below reproduce the exact bytes this function used to emit.
    """
    storage.write_json(path, payload, indent=2, ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------
# Master store
# --------------------------------------------------------------------------

# Set once the occupations upgrade has been reported, so a long-running server does
# not repeat the same warning on every page render.
_UPGRADE_REPORTED = False


def _upgrade_occupations(master: Dict[str, Any]) -> int:
    """
    Rebuilds the `occupations` object list on entries that predate it.

    Early versions of this store kept `onet_codes` and `onet_titles` as parallel flat
    lists plus one global `is_hot_tech_anywhere` flag. That threw away which
    occupations a skill is Hot Tech FOR -- Python is hot for Data Scientists and not
    for Statisticians -- which the dashboard drawer displays per occupation.

    The per-occupation detail is genuinely unrecoverable from what was stored, so this
    reconstructs the honest approximation: zip the two flat lists and seed every
    is_hot_tech from the single global flag. The next ingestion of each occupation
    overwrites the guess with the real value. Logged loudly for that reason.

    Returns the number of entries upgraded, so the caller can decide whether to save.
    """
    upgraded = 0
    for entry in master.values():
        if entry.get("occupations") is not None:
            continue
        codes = entry.get("onet_codes") or []
        titles = entry.get("onet_titles") or []
        hot = bool(entry.get("is_hot_tech_anywhere"))
        entry["occupations"] = [
            {
                "onet_code": code,
                # zip() would silently drop codes when the lists disagree in length,
                # which is exactly the case worth keeping visible.
                "onet_title": titles[index] if index < len(titles) else "",
                "is_hot_tech": hot,
            }
            for index, code in enumerate(codes)
        ]
        upgraded += 1
    return upgraded


def load_master() -> Dict[str, Any]:
    """
    Loads skills_master.json as {skill_name: record}.

    Applies the occupations upgrade in memory. The upgrade is NOT written back here --
    loading is a read, and a read that rewrites the file would make every dashboard
    render a disk write. The next save persists it.
    """
    try:
        data = _loaded(storage.load_master_raw(), MASTER_FILE, {})
    except storage.StorageUnreadable as err:
        raise StoreCorrupted(
            f"{MASTER_FILE} exists but could not be read ({err}). Refusing to continue "
            f"and overwrite it. Inspect or delete the file, then re-run."
        ) from err

    if not isinstance(data, dict):
        raise StoreCorrupted(
            f"{MASTER_FILE} should hold a JSON object keyed by skill name, "
            f"found {type(data).__name__}."
        )

    upgraded = _upgrade_occupations(data)
    if upgraded and not _UPGRADE_REPORTED:
        # Once per process, not once per request. The upgrade is not written back, so
        # every dashboard render re-applies it, and warning each time would bury the
        # log under a line that says nothing new.
        logger.warning(
            "Reconstructed per-occupation records for %d skills that predate them. "
            "Hot Tech status is seeded from the old global flag and is an "
            "approximation until each occupation is re-ingested. This is reported "
            "once per process.",
            upgraded,
        )
        globals()["_UPGRADE_REPORTED"] = True
    return data


def save_master(master: Dict[str, Any]) -> None:
    storage.save_master_raw(master)
    logger.info("Wrote %d skills to %s.", len(master), MASTER_FILE)


def load_timeseries() -> List[Dict[str, Any]]:
    """Loads skills_timeseries.json as a flat list of snapshot records."""
    try:
        data = _loaded(storage.load_timeseries_raw(), TIMESERIES_FILE, [])
    except storage.StorageUnreadable as err:
        raise StoreCorrupted(
            f"{TIMESERIES_FILE} exists but could not be read ({err}). Refusing to "
            f"continue and overwrite it. Inspect or delete the file, then re-run."
        ) from err

    if not isinstance(data, list):
        raise StoreCorrupted(
            f"{TIMESERIES_FILE} should hold a JSON array of snapshots, "
            f"found {type(data).__name__}."
        )
    return data


def save_timeseries(records: List[Dict[str, Any]]) -> None:
    storage.save_timeseries_raw(records)
    logger.info("Wrote %d snapshots to %s.", len(records), TIMESERIES_FILE)


# --------------------------------------------------------------------------
# Record helpers
# --------------------------------------------------------------------------

def quarter_for(when: datetime.date) -> str:
    """Calendar quarter label, e.g. 2026-08-06 -> '2026Q3'."""
    return f"{when.year}Q{(when.month - 1) // 3 + 1}"


def today() -> datetime.date:
    return datetime.date.today()


def merge_unique(existing: List[str], incoming: List[str]) -> List[str]:
    """
    Union of two string lists, order-preserving.

    This is what makes one skill spanning five occupations a single master entry with
    five codes rather than five entries. Order is preserved rather than sorted so the
    first occupation a skill was seen under stays first, which reads better in the UI.
    """
    merged = list(existing or [])
    for value in incoming or []:
        if value and value not in merged:
            merged.append(value)
    return merged


def merge_occupation(
    entry: Dict[str, Any], onet_code: str, onet_title: str, is_hot_tech: bool
) -> None:
    """
    Adds or updates one occupation on a master entry, keyed on onet_code.

    Hot status is ORed PER OCCUPATION, not globally: a skill can be Hot Tech for Data
    Scientists and standard for Statisticians, and that distinction is the whole point
    of tracking occupations as objects rather than as a flat code list.

    The flat onet_codes / onet_titles lists and is_hot_tech_anywhere are derived from
    the object list here, so they can never drift out of agreement with it.
    """
    if not onet_code:
        return

    occupations = entry.setdefault("occupations", [])
    for existing in occupations:
        if existing.get("onet_code") == onet_code:
            if onet_title:
                existing["onet_title"] = onet_title
            if is_hot_tech:
                existing["is_hot_tech"] = True
            break
    else:
        occupations.append({
            "onet_code": onet_code,
            "onet_title": onet_title or "",
            "is_hot_tech": bool(is_hot_tech),
        })

    entry["onet_codes"] = [occ["onet_code"] for occ in occupations]
    entry["onet_titles"] = merge_unique([], [occ["onet_title"] for occ in occupations])
    entry["is_hot_tech_anywhere"] = any(occ.get("is_hot_tech") for occ in occupations)


def upsert_master_entry(
    master: Dict[str, Any],
    skill_name: str,
    *,
    category: str = "",
    resolved_title: str = None,
    wikipedia_summary: str = None,
    best_source_name: str = None,
    occupations: List[Dict[str, Any]] = None,
    onet_codes: List[str] = None,
    onet_titles: List[str] = None,
    status: str = None,
    gate_reason: str = None,
    cross_score: float = None,
    is_credible: bool = None,
    is_hot_tech: bool = None,
    report_note: str = None,
    second_pass: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Creates or updates one master entry in place, and returns it.

    Only non-None arguments overwrite. That matters because ingestion calls this
    twice for the same skill -- once at discovery with just the occupation mapping,
    and again after scraping with the summary -- and a plain dict.update() would blank
    the summary on the discovery pass.

    Two fields never regress:
      - `status` is not downgraded from approved back to pending by a later discovery
        pass. Only an explicit review action changes an approved skill.
      - `is_hot_tech_anywhere` is an OR across occupations. Hot for Data Scientists and
        not for Statisticians means hot somewhere, and a later non-hot sighting must
        not clear it.
    """
    entry = master.get(skill_name)
    if entry is None:
        entry = {
            "skill_name": skill_name,
            "category": category,
            "resolved_title": None,
            # Set only when the reference is NOT a Wikipedia page. Wikipedia entries
            # keep storing a title and letting the UI build the link, so this stays
            # None for them rather than duplicating a derivable URL.
            "reference_url": None,
            "wikipedia_summary": None,
            "best_source_name": None,
            "last_updated": None,
            "occupations": [],
            "onet_codes": [],
            "onet_titles": [],
            "status": STATUS_PENDING,
            "gate_reason": None,
            "cross_score": None,
            "is_credible": None,
            "is_hot_tech_anywhere": False,
            "report_note": None,
            # Set by second_pass.py via review_actions, never by ingestion. Records
            # that an automated re-resolution has already looked at this entry and
            # what it concluded, so a later run does not spend quota re-deciding it.
            "second_pass": None,
        }
        master[skill_name] = entry

    entry.setdefault("occupations", [])

    if category:
        entry["category"] = category
    if resolved_title is not None:
        entry["resolved_title"] = resolved_title
    if best_source_name is not None:
        entry["best_source_name"] = best_source_name
    if wikipedia_summary is not None:
        entry["wikipedia_summary"] = wikipedia_summary
        entry["last_updated"] = today().isoformat()

    # Preferred form: full occupation objects, which carry per-occupation hot status.
    for occupation in occupations or []:
        merge_occupation(
            entry,
            occupation.get("onet_code"),
            occupation.get("onet_title", ""),
            occupation.get("is_hot_tech", False),
        )

    # Legacy form, kept so older callers keep working. is_hot_tech applies to the
    # occupations named in THIS call only, which is why it is passed through here
    # rather than being set globally on the entry.
    if onet_codes:
        titles = onet_titles or []
        for index, code in enumerate(onet_codes):
            merge_occupation(
                entry, code,
                titles[index] if index < len(titles) else "",
                bool(is_hot_tech),
            )
    elif is_hot_tech and not entry["occupations"]:
        # Hot with no occupation to attach it to. Rare, but losing the flag entirely
        # would be worse than recording it at the entry level.
        entry["is_hot_tech_anywhere"] = True

    if gate_reason is not None:
        entry["gate_reason"] = gate_reason
    if cross_score is not None:
        entry["cross_score"] = cross_score
    if is_credible is not None:
        entry["is_credible"] = is_credible
    if report_note is not None:
        entry["report_note"] = report_note
    if second_pass is not None:
        entry["second_pass"] = second_pass

    if status is not None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Unknown status {status!r}; expected one of {VALID_STATUSES}.")
        # Never silently un-approve. A re-ingestion pass that re-queued an
        # already-reviewed skill would throw away the human decision.
        if not (entry["status"] == STATUS_APPROVED and status == STATUS_PENDING):
            entry["status"] = status

    return entry


def upsert_snapshot(
    records: List[Dict[str, Any]],
    skill_name: str,
    metrics: Dict[str, Any],
    onet_codes: List[str],
    onet_titles: List[str],
    when: datetime.date = None,
) -> List[Dict[str, Any]]:
    """
    Adds or replaces the snapshot for one skill in the CURRENT quarter.

    One record per (skill, quarter). A literal append-on-every-run would add a point
    each time the pipeline executes, and the trend chart would draw stair-steps from
    repeated same-day runs that look exactly like real score drift. A new quarter
    creates a new record; a re-run inside the same quarter refreshes the existing one.

    Returns the list (mutated in place) for chaining.
    """
    when = when or today()
    quarter = quarter_for(when)

    record = {
        "skill_name": skill_name,
        "quarter": quarter,
        "snapshot_date": when.isoformat(),
        "ai_score": metrics["ai_score"],
        # The pole measurement with nothing added, and what was added. Stored separately
        # so a boosted 2026 snapshot stays comparable with an unboosted 2025 one: any
        # trend that spans the change has to read ai_score_base, or it draws a 0.05 step
        # that is a policy change rather than drift. Defaulted to ai_score with a zero
        # boost for callers that predate the split, which is exactly what those were.
        "ai_score_base": metrics.get("ai_score_base", metrics["ai_score"]),
        "embedded_ai_boost": metrics.get("embedded_ai_boost", 0.0),
        # The deterministic half, and the phrases that earned it. The terms are stored,
        # not just the number: "this scored 0.05 higher" is unarguable only when the
        # words that did it are on the record next to it.
        "lexical_ai_boost": metrics.get("lexical_ai_boost", 0.0),
        "lexical_ai_terms": metrics.get("lexical_ai_terms") or [],
        "tech_base_sim": metrics["tech_base_sim"],
        "ml_pipeline_sim": metrics["ml_pipeline_sim"],
        "embedded_ai_sim": metrics["embedded_ai_sim"],
        # Diagnostic only, gating nothing today. Recorded so there is real data to argue
        # from if it is ever proposed as a rule -- see calculate_ai_correlation.
        "contrast_sim": metrics.get("contrast_sim"),
        # Same discipline as contrast_sim: measured, stored, shown, and read by no rule.
        # None on records written before it landed. Nothing compares it, so nothing can
        # raise on the None, and defaulting it to 0.0 would assert a measurement that was
        # never taken. See LEGACY_STATIC_ANCHOR for why it does not gate anything.
        "legacy_sim": metrics.get("legacy_sim"),
        # The two poles behind ai_score, kept apart. ai_engineering_sim is READ BY THE
        # TOP BUCKET -- see AI_ENGINEERING_FLOOR -- so a snapshot without it cannot be
        # reclassified faithfully. Absent on records written before this landed, which
        # classify() reads as "skip the floor" rather than "fail it".
        "ai_engineering_sim": metrics.get("ai_engineering_sim"),
        "ai_generative_sim": metrics.get("ai_generative_sim"),
        # Three-state and stored as such. None means nobody has established it, which is
        # NOT the same as False, and collapsing the two here would make a rate-limited
        # afternoon indistinguishable from a finding about the product.
        "embeds_ai": metrics.get("embeds_ai"),
        "category_bucket": metrics["category_bucket"],
        # The rules engine's verdict and its working. Read with .get() so a snapshot
        # written before the engine landed still loads: the time series is appended to
        # across quarters and old records are never rewritten.
        #
        # FIELD NAMING: snake_case, without exception. Every key in both store files uses
        # it, and these are iterated generically by the dashboard row builders and the
        # JSON export. A second convention would not be a style disagreement, it would be
        # a key some iterator silently misses.
        "sub_category": metrics.get("sub_category"),
        "is_flagship_version_evaluated": metrics.get("is_flagship_version_evaluated", False),
        # Provenance for a substituted score. A category term measured as Microsoft Word
        # reads "ai 0.192" exactly like a measurement of the term itself, so which product
        # produced it has to travel with it.
        "is_generic_category": metrics.get("is_generic_category", False),
        "flagship_version": metrics.get("flagship_version"),
        "flagship_source": metrics.get("flagship_source"),
        "decision_metric": metrics.get("decision_metric"),
        "decision_threshold": metrics.get("decision_threshold"),
        "decision_margin": metrics.get("decision_margin"),
        "in_semantic_variance_band": metrics.get("in_semantic_variance_band", False),
        "classification_confidence": metrics.get("classification_confidence"),
        "onet_codes": list(onet_codes or []),
        "onet_titles": list(onet_titles or []),
    }

    for index, existing in enumerate(records):
        if existing.get("skill_name") == skill_name and existing.get("quarter") == quarter:
            records[index] = record
            return records

    records.append(record)
    return records


def latest_snapshots(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Newest snapshot per skill, keyed by skill name.

    Sorts on (quarter, snapshot_date) rather than trusting file order, because the
    time series is appended to over many runs and nothing guarantees it stays ordered.
    Quarter labels sort correctly as strings ('2026Q1' < '2026Q3' < '2027Q1').
    """
    newest: Dict[str, Dict[str, Any]] = {}
    for record in records:
        name = record.get("skill_name")
        if not name:
            continue
        current = newest.get(name)
        key = (record.get("quarter", ""), record.get("snapshot_date", ""))
        if current is None or key > (current.get("quarter", ""), current.get("snapshot_date", "")):
            newest[name] = record
    return newest

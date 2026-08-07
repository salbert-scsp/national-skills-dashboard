"""
The two-file local JSON store.

With no database there is no second copy of anything, so this module is deliberately
paranoid about the two ways flat-file storage loses data: a truncated write, and a
read that silently swallows corruption and hands back an empty dict.

    skills_master.json      object keyed by skill name; upserted every run
    skills_timeseries.json  flat list of snapshots; one record per (skill, quarter)

Every module that touches those files goes through here. Three modules read them
(definitions_algorithm, dashboardtables, app_interface) and two write them, and
duplicating the atomic-write dance in each is how the two files drift apart.
"""

import datetime
import json
import logging
import os
import tempfile
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

MASTER_FILE = os.getenv("SKILLS_MASTER_FILE", "skills_master.json")
TIMESERIES_FILE = os.getenv("SKILLS_TIMESERIES_FILE", "skills_timeseries.json")

# Review states a master entry can hold. Strings rather than the SQL integers,
# because a JSON file is read by humans and `"status": -2` explains nothing.
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
VALID_STATUSES = (STATUS_PENDING, STATUS_APPROVED, STATUS_REJECTED)


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

def _read_json(path: str, default):
    """Reads a JSON file, tolerating absence but never corruption."""
    if not os.path.exists(path):
        logger.info("%s does not exist yet; starting from empty.", path)
        return default

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (ValueError, UnicodeDecodeError) as err:
        raise StoreCorrupted(
            f"{path} exists but is not valid JSON ({err}). Refusing to continue and "
            f"overwrite it. Inspect or delete the file, then re-run."
        ) from err
    except OSError as err:
        raise StoreCorrupted(f"{path} could not be read: {err}") from err


def atomic_write(path: str, payload) -> None:
    """
    Writes JSON so that the file on disk is either the old content or the new one.

    Serializes to a temp file in the SAME directory (os.replace is only atomic within
    a filesystem), flushes and fsyncs so the bytes are really on disk before the
    rename, then replaces. A crash at any point leaves the previous file intact.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, prefix=".tmp_", suffix=".json",
        delete=False,
    )
    temp_path = handle.name
    try:
        with handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        # Never leave the temp file behind on failure; the directory would silently
        # fill with .tmp_*.json across failed runs.
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


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
    data = _read_json(MASTER_FILE, {})
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
    atomic_write(MASTER_FILE, master)
    logger.info("Wrote %d skills to %s.", len(master), MASTER_FILE)


def load_timeseries() -> List[Dict[str, Any]]:
    """Loads skills_timeseries.json as a flat list of snapshot records."""
    data = _read_json(TIMESERIES_FILE, [])
    if not isinstance(data, list):
        raise StoreCorrupted(
            f"{TIMESERIES_FILE} should hold a JSON array of snapshots, "
            f"found {type(data).__name__}."
        )
    return data


def save_timeseries(records: List[Dict[str, Any]]) -> None:
    atomic_write(TIMESERIES_FILE, records)
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
        "tech_base_sim": metrics["tech_base_sim"],
        "ml_pipeline_sim": metrics["ml_pipeline_sim"],
        "embedded_ai_sim": metrics["embedded_ai_sim"],
        "category_bucket": metrics["category_bucket"],
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

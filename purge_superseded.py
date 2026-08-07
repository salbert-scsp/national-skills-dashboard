"""
One-time purge of superseded HITL queue rows.

Standalone and NOT part of repair_schema on purpose. repair_schema runs on every boot
and every statement in it is additive; a DELETE has no business in that path.

Background: an earlier three-source pipeline queued rows with no summary text, which no
human could ever approve and which shadowed their skills out of every subsequent
ingestion run. database._supersede_legacy_queue_rows moved them to is_approved = -2,
which took them out of the pending queue and out of the shadow check while leaving them
on disk. This script removes them from disk.

The DELETE is irreversible, so a full JSON dump is written first and its record count is
verified against the row count about to be deleted. If they disagree, nothing is deleted.

Usage:
    python3.11 purge_superseded.py              # dry run, the default
    python3.11 purge_superseded.py --confirm    # dump, verify, then delete
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from database import QUEUE_SUPERSEDED, check_db_health, db_cursor

logger = logging.getLogger("purge_superseded")

BACKUP_PREFIX = "superseded_queue_backup_"


def _fetch_superseded() -> list:
    """Returns every superseded row as a dict, all columns, whatever they are."""
    with db_cursor(autocommit=True) as cursor:
        cursor.execute(
            f"""
            SELECT * FROM HITL_Validation_Queue
            WHERE is_approved = {QUEUE_SUPERSEDED}
            ORDER BY QueueID
            """
        )
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _write_backup(rows: list) -> Path:
    """
    Dumps rows to a timestamped JSON file and returns its path.

    default=str covers datetime and Decimal, neither of which json handles natively.
    The dump is not meant to be re-imported by machine -- it is the record of what was
    destroyed, so readability beats round-trip fidelity.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(f"{BACKUP_PREFIX}{stamp}.json")
    path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return path


def purge(confirm: bool) -> int:
    """Returns a process exit code."""
    if not check_db_health():
        logger.error("Database is not reachable. Nothing done.")
        return 1

    rows = _fetch_superseded()
    count = len(rows)
    logger.info("Found %d superseded queue rows (is_approved = %d).", count, QUEUE_SUPERSEDED)

    if count == 0:
        logger.info("Nothing to purge.")
        return 0

    if not confirm:
        logger.info(
            "Dry run. Re-run with --confirm to dump these %d rows to JSON and delete them.",
            count,
        )
        return 0

    path = _write_backup(rows)

    # Read the file back rather than trusting the write. The dump is the only way back
    # from the DELETE, so it gets verified, not assumed.
    try:
        restored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.exception("Backup at %s could not be read back. Nothing deleted.", path)
        return 1

    if not isinstance(restored, list) or len(restored) != count:
        logger.error(
            "Backup at %s holds %s records but %d rows were read. Nothing deleted.",
            path, len(restored) if isinstance(restored, list) else "a non-list", count,
        )
        return 1

    logger.info("Backup verified: %d records in %s.", len(restored), path)

    with db_cursor() as cursor:
        cursor.execute(
            f"DELETE FROM HITL_Validation_Queue WHERE is_approved = {QUEUE_SUPERSEDED}"
        )
        deleted = cursor.rowcount

    logger.warning("Deleted %d superseded queue rows. Backup: %s", deleted, path)
    if deleted != count:
        logger.warning(
            "Deleted %d rows but %d were backed up. Rows changed between the dump and "
            "the delete; the backup is still a superset of what was removed.",
            deleted, count,
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually dump and delete. Without this the script only reports.",
    )
    sys.exit(purge(parser.parse_args().confirm))

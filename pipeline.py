"""
Ingestion orchestrator.

Responsibilities, deliberately narrow:
  - Record discovery: Occupations_Master, Skills_Master, and Skill_Occupation_Map
    first_seen_date / last_seen_date / is_hot_tech tracking.
  - Decide whether a skill needs re-summarizing (90-day rolling snapshot check).
  - Gather candidates, gate them on cross-encoder relevance, audit the survivors,
    and queue the results.
  - Hand auto-approved items to score.py.

It writes NO rows to Skills_Historical_Metrics. score.py owns that table exclusively,
so each promotion produces exactly one snapshot and rejected skills leave none.

Two structural rules worth preserving:

1. ALL tools are recorded, only HOT ones are audited. Hot Tech is a property of the
   (skill, occupation) relationship, so every tool gets a Skill_Occupation_Map row
   with its true flag, while the expensive path (Wikipedia, cross-encoder, Gemini)
   runs only for the hot subset. The `continue` that enforces this sits above every
   expensive call, making the guarantee structural rather than incidental.

2. Gathering happens BEFORE auditing, per occupation code. Phase 1 does all the
   scraping and local scoring with no API calls; phase 2 spends Gemini quota on the
   survivors. Checkpointing stays per-occupation: gathering all 1000 codes before the
   first audit would mean a crash at 95 percent loses the entire run.
"""

import datetime
import logging
from typing import Any, Dict, List, Optional

from database import QUEUE_APPROVED, QUEUE_PENDING, db_cursor, initialize_schema
from score import auto_approve_and_score_skill
from scraping import (
    CROSS_ENCODER_THRESHOLD,
    fetch_all_onet_codes,
    fetch_onet_tools_and_tech,
    load_local_cache,
    save_local_cache,
    scrape_and_validate_skill,
)

logger = logging.getLogger(__name__)

RESUMMARIZE_AFTER_DAYS = 90


def _upsert_occupation(cursor, onet_code: str, onet_title: str) -> None:
    cursor.execute(
        """
        IF NOT EXISTS (SELECT 1 FROM Occupations_Master WHERE onet_code = ?)
            INSERT INTO Occupations_Master (onet_code, onet_title, last_updated)
            VALUES (?, ?, GETDATE())
        ELSE
            UPDATE Occupations_Master SET onet_title = ?, last_updated = GETDATE()
            WHERE onet_code = ?
        """,
        (onet_code, onet_code, onet_title, onet_title, onet_code),
    )


def _upsert_skill(cursor, skill_name: str, category: str) -> int:
    """
    Records skill discovery and returns its skill_id.

    is_approved stays 0 on insert; only score.py promotes a skill, so discovery no
    longer pre-approves everything it finds.

    Note for future optimization: do NOT narrow this to "only newly inserted skills".
    A tool first seen as non-hot has a Skills_Master row but no metrics row, and it
    must be able to enter the expensive path later when O*NET flags it hot for some
    occupation. Skipping already-known skills would orphan those permanently.
    """
    cursor.execute(
        """
        IF NOT EXISTS (SELECT 1 FROM Skills_Master WHERE skill_name = ?)
            INSERT INTO Skills_Master (skill_name, category, is_approved) VALUES (?, ?, 0)
        ELSE
            UPDATE Skills_Master SET category = ? WHERE skill_name = ?
        """,
        (skill_name, skill_name, category, category, skill_name),
    )
    cursor.execute("SELECT skill_id FROM Skills_Master WHERE skill_name = ?", (skill_name,))
    return cursor.fetchone()[0]


def _track_occupation_map(
    cursor,
    skill_id: int,
    onet_code: str,
    today: datetime.date,
    is_hot_tech: bool,
) -> None:
    """
    Sets first_seen_date on initial discovery, then refreshes last_seen_date and
    is_hot_tech on every subsequent sighting.

    Hot status is OVERWRITTEN with the latest O*NET truth rather than latched to 1.
    Latching would monotonically drift every row toward "hot" and recreate exactly the
    information-free flag this design exists to eliminate. A downgrade is safe: the
    skill's Skills_Historical_Metrics rows and its Skills_Master.is_approved flag are
    keyed on skill_id and untouched, so an already-scored skill keeps its history.

    Read-then-branch rather than one IF NOT EXISTS statement, so a reclassification
    can be logged and a legacy NULL flag is distinguishable from a missing row.
    """
    cursor.execute(
        "SELECT is_hot_tech FROM Skill_Occupation_Map WHERE skill_id = ? AND onet_code = ?",
        (skill_id, onet_code),
    )
    row = cursor.fetchone()
    new_flag = 1 if is_hot_tech else 0

    if row is None:
        cursor.execute(
            """
            INSERT INTO Skill_Occupation_Map
                (skill_id, onet_code, first_seen_date, last_seen_date, is_hot_tech)
            VALUES (?, ?, ?, ?, ?)
            """,
            (skill_id, onet_code, today, today, new_flag),
        )
        return

    # NULL means the row predates per-relationship tracking and is presumed hot.
    previous = row[0]
    if previous is not None and int(previous) != new_flag:
        if new_flag == 0:
            logger.warning(
                "O*NET reclassified skill_id=%s for %s: hot -> not hot.", skill_id, onet_code
            )
        else:
            logger.info(
                "O*NET reclassified skill_id=%s for %s: not hot -> hot.", skill_id, onet_code
            )

    cursor.execute(
        """
        UPDATE Skill_Occupation_Map SET last_seen_date = ?, is_hot_tech = ?
        WHERE skill_id = ? AND onet_code = ?
        """,
        (today, new_flag, skill_id, onet_code),
    )


def _latest_snapshot_date(cursor, skill_id: int) -> Optional[datetime.date]:
    cursor.execute(
        """
        SELECT TOP 1 snapshot_date FROM Skills_Historical_Metrics
        WHERE skill_id = ? ORDER BY snapshot_date DESC
        """,
        (skill_id,),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def _has_pending_review(cursor, skill_name: str) -> bool:
    """
    True if this skill is already awaiting human review by someone who could act on it.

    Without this check, a queued-but-unreviewed skill has no metrics row, so the
    90-day test keeps firing and every run re-scrapes it, re-audits it, and stacks
    another duplicate card in the reviewer's queue.

    Keyed on skill_name rather than (skill_name, onet_code) on purpose: a skill needs
    auditing once, not once per occupation. The complete per-occupation truth lives in
    Skill_Occupation_Map, so the queue row only records the occupation that triggered
    the audit. Do not weaken this to per-occupation without also solving the duplicate
    cards it would create.

    INVARIANT: the rows that block a skill must be a subset of the rows a human can
    actually approve. An earlier pipeline generation queued 388 rows with no summary
    text, which score.promote_queue_item refuses to promote -- so they could never be
    cleared, and they silently shadowed 35 of 53 hot skills for one occupation alone,
    keeping them from ever being scraped or scored.

    Hence the summary test, which mirrors promote_queue_item's own refusal condition.
    Those two must change together. Note what is deliberately NOT tested here:
    gate_reason. Filtering on it would encode "written by the generation that stamps
    this column", a fact about today's writer rather than about approvability; a future
    version that legitimately queued without one would then be re-scraped every run,
    which is the duplicate-card bug this function exists to prevent.

    is_approved values 1, -1 and -2 all correctly fail to block.
    """
    cursor.execute(
        """
        SELECT
            SUM(CASE WHEN wiki_summary IS NOT NULL AND LTRIM(RTRIM(wiki_summary)) <> ''
                     THEN 1 ELSE 0 END),
            COUNT(*)
        FROM HITL_Validation_Queue
        WHERE skill_name = ? AND is_approved = 0
        """,
        (skill_name,),
    )
    blocking, total = cursor.fetchone()
    blocking = int(blocking or 0)
    total = int(total or 0)

    # Counted rather than filtered out, so a recurrence of the unapprovable-row bug is
    # visible in the log instead of silently changing behavior.
    if total > blocking:
        logger.warning(
            "%d pending queue row(s) for %r have no summary and cannot be approved; "
            "not blocking re-audit.",
            total - blocking, skill_name,
        )

    return blocking > 0


def _insert_queue_item(cursor, skill_item: dict, decision: dict, onet_code: str,
                       onet_title: str, is_approved: int, is_hot_tech: bool) -> int:
    """
    Inserts the queue row and returns its QueueID.

    SET NOCOUNT ON is required: without it the INSERT's row count arrives as the first
    result, and pyodbc raises "Previous SQL was not a query" on the fetch below.
    """
    cursor.execute(
        """
        SET NOCOUNT ON;
        INSERT INTO HITL_Validation_Queue (
            skill_name, category, wiki_title, wiki_summary, wiki_score,
            best_source_name, is_credible, gate_reason,
            onet_code, onet_title, is_hot_tech, is_approved
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        SELECT CAST(SCOPE_IDENTITY() AS INT);
        """,
        (
            skill_item["skill_name"],
            skill_item["category"],
            decision["resolved_title"],
            decision["summary"],
            decision["cross_score"],
            decision["best_source_name"],
            decision["is_credible"],
            decision["gate_reason"],
            onet_code,
            onet_title,
            1 if is_hot_tech else 0,
            is_approved,
        ),
    )
    return int(cursor.fetchone()[0])


def _record_discovery(
    skills: List[Dict[str, Any]],
    onet_code: str,
    onet_title: str,
    today: datetime.date,
    stale_before: datetime.date,
) -> List[Dict[str, Any]]:
    """
    Records every tool against the occupation and returns the hot skills needing audit.

    Non-hot skills are fully handled here: a Skills_Master row and a map row with
    is_hot_tech = 0, and nothing else. They are re-visited on every run, but that costs
    three local statements and zero network calls, and the visit is required so
    last_seen_date stays current.
    """
    needs_audit: List[Dict[str, Any]] = []
    non_hot = 0

    with db_cursor() as cursor:
        _upsert_occupation(cursor, onet_code, onet_title)

        for skill_item in skills:
            skill_name = skill_item["skill_name"]
            is_hot = bool(skill_item["is_hot_tech"])

            skill_id = _upsert_skill(cursor, skill_name, skill_item["category"])
            _track_occupation_map(cursor, skill_id, onet_code, today, is_hot)

            # Everything below this point is expensive. Non-hot tools stop here.
            if not is_hot:
                non_hot += 1
                continue

            if _has_pending_review(cursor, skill_name):
                logger.info("Skipping %r: already awaiting human review.", skill_name)
                continue

            last_snapshot = _latest_snapshot_date(cursor, skill_id)
            if last_snapshot is not None and last_snapshot >= stale_before:
                logger.info(
                    "Skipping %r: snapshot dated %s is within %d days.",
                    skill_name, last_snapshot, RESUMMARIZE_AFTER_DAYS,
                )
                continue

            needs_audit.append(dict(skill_item, _last_snapshot=last_snapshot))

    logger.info(
        "O*NET %s: %d tools recorded (%d non-hot), %d hot skills need auditing.",
        onet_code, len(skills), non_hot, len(needs_audit),
    )
    return needs_audit


def process_onet_code_ingestion(onet_code: str) -> dict:
    """
    Ingests one O*NET occupation code.

    Phase 1 records discovery and gathers candidates locally. Phase 2 spends Gemini
    quota only on candidates that cleared the relevance gate. Phase 3 promotes the
    auto-approved ones.
    """
    data = fetch_onet_tools_and_tech(onet_code)
    skills = data.get("skills", [])
    onet_title = data.get("onet_title", "Unknown")

    empty_result = {
        "processed": 0, "hot": 0, "non_hot": 0,
        "queued": 0, "auto_approved": 0, "onet_title": onet_title,
    }
    if not skills:
        logger.warning("No tools or technologies returned for %s.", onet_code)
        return empty_result

    today = datetime.date.today()
    stale_before = today - datetime.timedelta(days=RESUMMARIZE_AFTER_DAYS)
    hot_count = sum(1 for item in skills if item["is_hot_tech"])

    # --- Phase 1: discovery, then local candidate gathering (no API calls) ---
    needs_audit = _record_discovery(skills, onet_code, onet_title, today, stale_before)

    cache = load_local_cache()
    cache_size_at_start = len(cache)

    # --- Phase 2: audit the gathered candidates, then queue ---
    # scrape_and_validate_skill resolves candidates and cross-encodes them before it
    # spends a Gemini call, and only calls Gemini for candidates above the threshold.
    queued = 0
    auto_approved_ids: List[int] = []

    for skill_item in needs_audit:
        skill_name = skill_item["skill_name"]
        logger.info(
            "Resolving %r (last snapshot: %s).", skill_name, skill_item.get("_last_snapshot")
        )
        decision = scrape_and_validate_skill(skill_item, cache=cache)

        target_status = QUEUE_APPROVED if decision["auto_approve"] else QUEUE_PENDING
        with db_cursor() as cursor:
            queue_id = _insert_queue_item(
                cursor, skill_item, decision, onet_code, onet_title,
                target_status, skill_item["is_hot_tech"],
            )

        if decision["auto_approve"]:
            auto_approved_ids.append(queue_id)
        else:
            queued += 1
            logger.info(
                "Queued %r for review (score=%s, threshold=%.2f, reason=%s).",
                skill_name, decision["cross_score"], CROSS_ENCODER_THRESHOLD,
                decision["gate_reason"],
            )

    if len(cache) != cache_size_at_start:
        save_local_cache(cache)

    # --- Phase 3: promotion runs last, so a scoring failure cannot abort ingestion ---
    scored = 0
    for queue_id in auto_approved_ids:
        try:
            if auto_approve_and_score_skill(queue_id):
                scored += 1
        except Exception:
            logger.exception("Auto-approval scoring failed for queue item #%s.", queue_id)

    logger.info(
        "O*NET %s complete: %d tools seen (%d hot), %d queued for review, %d auto-approved.",
        onet_code, len(skills), hot_count, queued, scored,
    )
    return {
        "processed": len(skills),
        "hot": hot_count,
        "non_hot": len(skills) - hot_count,
        "queued": queued,
        "auto_approved": scored,
        "onet_title": onet_title,
    }


def run_full_onet_pipeline() -> dict:
    onet_codes = fetch_all_onet_codes()
    logger.info("Starting ingestion across %d O*NET codes.", len(onet_codes))

    # Keys must match those returned by process_onet_code_ingestion: the accumulation
    # below iterates this dict, so a key missing here is silently dropped.
    totals = {"processed": 0, "hot": 0, "non_hot": 0, "queued": 0, "auto_approved": 0}

    for index, (code, title) in enumerate(onet_codes.items(), start=1):
        logger.info("[%d/%d] O*NET %s (%s)", index, len(onet_codes), code, title)
        try:
            result = process_onet_code_ingestion(code)
        except Exception:
            logger.exception("Ingestion failed for O*NET code %s. Continuing.", code)
            continue
        for key in totals:
            totals[key] += result.get(key, 0)

    logger.info(
        "Full pipeline complete: %d tools seen (%d hot, %d non-hot), %d queued, %d auto-approved.",
        totals["processed"], totals["hot"], totals["non_hot"],
        totals["queued"], totals["auto_approved"],
    )
    return totals


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    # initialize_schema() otherwise runs only from the FastAPI lifespan, so a CLI run
    # against an un-migrated database would die on a missing is_hot_tech column.
    initialize_schema()
    run_full_onet_pipeline()

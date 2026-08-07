"""
One-time export of the SQL Server database into the two JSON store files.

    python3.11 migrate_sql_to_json.py            writes *.json.new, changes nothing
    python3.11 migrate_sql_to_json.py --apply    writes the real store files

Reads only. Nothing in SQL is modified or dropped, so the database remains the
fallback until you are satisfied with the JSON.

What comes across:
  - Every approved Skills_Master row, with its latest summary, its resolved reference
    page, and its occupation mappings aggregated into deduped lists.
  - Every pending queue row, as a pending master entry carrying its gate reason, so
    the items currently awaiting review survive the move.
  - Every Skills_Historical_Metrics row as a snapshot, RE-SCORED from its stored
    summary text with the new anchors.

Why re-score rather than copy the stored numbers: the three enabling sims do not
exist in SQL, so copying would leave them null for all history and the enabling
breakdown would only start working from the next run. ai_score itself is unaffected,
because sortingalgorithmnew keeps both AI poles verbatim -- the migration verifies
that on every row rather than assuming it, and reports any drift.
"""

import argparse
import datetime
import logging
import sys
from collections import defaultdict

from database import QUEUE_APPROVED, QUEUE_PENDING, check_db_health, db_cursor
from json_store import (
    MASTER_FILE,
    STATUS_APPROVED,
    STATUS_PENDING,
    TIMESERIES_FILE,
    atomic_write,
    quarter_for,
)
from sortingalgorithmnew import calculate_ai_correlation

logger = logging.getLogger(__name__)

# A recomputed ai_score should equal the stored one exactly, since the AI poles are
# unchanged. Allow a hair of float noise before calling it drift.
SCORE_DRIFT_TOLERANCE = 0.0002


def fetch_occupation_map(cursor) -> dict:
    """{skill_id: ([codes], [titles], hot_anywhere)} from Skill_Occupation_Map."""
    cursor.execute(
        """
        SELECT som.skill_id, som.onet_code, o.onet_title,
               CASE WHEN COALESCE(som.is_hot_tech, 1) = 1 THEN 1 ELSE 0 END
        FROM Skill_Occupation_Map AS som
        LEFT JOIN Occupations_Master AS o ON o.onet_code = som.onet_code
        ORDER BY som.skill_id, som.onet_code
        """
    )
    codes = defaultdict(list)
    titles = defaultdict(list)
    hot = defaultdict(bool)
    for skill_id, code, title, is_hot in cursor.fetchall():
        if code and code not in codes[skill_id]:
            codes[skill_id].append(code)
        if title and title not in titles[skill_id]:
            titles[skill_id].append(title)
        if is_hot:
            hot[skill_id] = True
    return {
        skill_id: (codes[skill_id], titles[skill_id], hot[skill_id])
        for skill_id in set(codes) | set(titles) | set(hot)
    }


def fetch_resolved_titles(cursor) -> dict:
    """
    {skill_name: wiki_title} from the newest approved queue row per skill.

    Same rule dashboard_data.EXPORT_SQL used, so the reference page that migrates is
    the one the old dashboard was displaying.
    """
    cursor.execute(
        f"""
        WITH ranked AS (
            SELECT skill_name, wiki_title,
                   ROW_NUMBER() OVER (
                       PARTITION BY skill_name
                       ORDER BY created_at DESC, QueueID DESC
                   ) AS rank
            FROM HITL_Validation_Queue
            WHERE is_approved = {QUEUE_APPROVED} AND wiki_title IS NOT NULL
        )
        SELECT skill_name, wiki_title FROM ranked WHERE rank = 1
        """
    )
    return {name: title for name, title in cursor.fetchall()}


def build_master(cursor) -> dict:
    """Approved skills plus pending queue rows, as a master store dict."""
    occupation_map = fetch_occupation_map(cursor)
    resolved = fetch_resolved_titles(cursor)

    cursor.execute(
        """
        WITH latest AS (
            SELECT skill_id, summary_text, snapshot_date,
                   ROW_NUMBER() OVER (
                       PARTITION BY skill_id
                       ORDER BY snapshot_date DESC, metric_id DESC
                   ) AS rank
            FROM Skills_Historical_Metrics
        )
        SELECT s.skill_id, s.skill_name, s.category, l.summary_text, l.snapshot_date
        FROM Skills_Master AS s
        LEFT JOIN latest AS l ON l.skill_id = s.skill_id AND l.rank = 1
        """
    )

    # EVERY Skills_Master row, not just the approved ones. The majority are non-hot
    # tools that were discovered and mapped to an occupation but deliberately never
    # scraped or audited -- they are what the "tools discovered" figure counts, and
    # dropping them would silently cut that KPI from 382 to 128. The new ingestion
    # records non-hot tools the same way, so carrying them keeps the two consistent.
    master = {}
    for skill_id, skill_name, category, summary, snapshot_date in cursor.fetchall():
        codes, titles, hot = occupation_map.get(skill_id, ([], [], False))
        master[skill_name] = {
            "skill_name": skill_name,
            "category": category or "",
            "resolved_title": resolved.get(skill_name),
            "wikipedia_summary": summary or "",
            "last_updated": snapshot_date.isoformat() if snapshot_date else None,
            "onet_codes": codes,
            "onet_titles": titles,
            # Approved only where there is a definition to stand on. A row with no
            # summary was never audited, so it is pending by definition -- but see
            # dashboardtables.build_review_rows: pending WITHOUT a summary is not
            # reviewable and is kept out of the review screen. That distinction is
            # what stops 254 unaudited tools flooding the queue, and it is the same
            # invariant that the 388-row backlog taught us.
            "status": STATUS_APPROVED if summary else STATUS_PENDING,
            "gate_reason": None if summary else "discovered_not_audited",
            "cross_score": None,
            "is_credible": None,
            "is_hot_tech_anywhere": bool(hot),
        }

    # Pending queue rows. Skipped where the skill is already approved in master: an
    # open report on an approved skill must not demote it, and the report itself has
    # no equivalent in the JSON model.
    cursor.execute(
        f"""
        SELECT skill_name, category, wiki_title, wiki_summary, wiki_score,
               is_credible, gate_reason, onet_code, onet_title
        FROM HITL_Validation_Queue
        WHERE is_approved = {QUEUE_PENDING}
        ORDER BY created_at DESC, QueueID DESC
        """
    )
    pending_added = 0
    for (name, category, title, summary, score, credible,
         gate_reason, code, onet_title) in cursor.fetchall():
        existing = master.get(name)
        if existing and existing["status"] == STATUS_APPROVED:
            continue

        if existing:
            # The skill is already in master from Skills_Master, unaudited. Attach the
            # queue row's evidence so it becomes genuinely reviewable, rather than
            # inserting a second entry under the same name.
            if existing.get("gate_reason") == "discovered_not_audited":
                existing.update({
                    "resolved_title": title or existing.get("resolved_title"),
                    "wikipedia_summary": summary or existing.get("wikipedia_summary") or "",
                    "gate_reason": gate_reason,
                    "cross_score": float(score) if score is not None else None,
                    "is_credible": None if credible is None else bool(credible),
                    "is_hot_tech_anywhere": True,
                })
                pending_added += 1
            continue  # newest pending row wins; later ones are ignored

        master[name] = {
            "skill_name": name,
            "category": category or "",
            "resolved_title": title,
            "wikipedia_summary": summary or "",
            "last_updated": None,
            "onet_codes": [code] if code else [],
            "onet_titles": [onet_title] if onet_title else [],
            "status": STATUS_PENDING,
            "gate_reason": gate_reason,
            "cross_score": float(score) if score is not None else None,
            "is_credible": None if credible is None else bool(credible),
            "is_hot_tech_anywhere": True,
        }
        pending_added += 1

    logger.info("Built %d master entries (%d from the pending queue).", len(master), pending_added)
    return master


def build_timeseries(cursor, master: dict) -> tuple:
    """
    Every historical metrics row, re-scored with the new anchors.

    Returns (records, drift) where drift lists any skill whose recomputed ai_score
    differs from the stored one by more than the tolerance. Drift is not fatal -- a
    post-mortem edit legitimately changes the text a snapshot was scored from -- but
    it must be reported rather than absorbed silently.
    """
    cursor.execute(
        """
        SELECT s.skill_name, s.category, m.snapshot_date, m.summary_text,
               m.ai_correlation_score
        FROM Skills_Historical_Metrics AS m
        JOIN Skills_Master AS s ON s.skill_id = m.skill_id
        ORDER BY s.skill_name, m.snapshot_date, m.metric_id
        """
    )

    records = []
    drift = []
    seen = set()

    for skill_name, category, snapshot_date, summary, stored_score in cursor.fetchall():
        when = snapshot_date or datetime.date.today()
        quarter = quarter_for(when)

        # One record per (skill, quarter). SQL allowed one per DAY, so two snapshots
        # in the same quarter collapse; the later one wins, which matches how the
        # store behaves going forward.
        key = (skill_name, quarter)
        metrics = calculate_ai_correlation(skill_name, category or "", summary or "")

        if stored_score is not None:
            delta = abs(float(stored_score) - metrics["ai_score"])
            if delta > SCORE_DRIFT_TOLERANCE:
                drift.append((skill_name, float(stored_score), metrics["ai_score"], delta))

        entry = master.get(skill_name, {})
        record = {
            "skill_name": skill_name,
            "quarter": quarter,
            "snapshot_date": when.isoformat(),
            "ai_score": metrics["ai_score"],
            "tech_base_sim": metrics["tech_base_sim"],
            "ml_pipeline_sim": metrics["ml_pipeline_sim"],
            "embedded_ai_sim": metrics["embedded_ai_sim"],
            "category_bucket": metrics["category_bucket"],
            "onet_codes": list(entry.get("onet_codes") or []),
            "onet_titles": list(entry.get("onet_titles") or []),
        }

        if key in seen:
            for index, existing in enumerate(records):
                if (existing["skill_name"], existing["quarter"]) == key:
                    records[index] = record
                    break
        else:
            seen.add(key)
            records.append(record)

    logger.info("Built %d snapshots (one per skill per quarter).", len(records))
    return records, drift


def migrate(apply: bool) -> int:
    if not check_db_health():
        logger.error("Database unreachable. Nothing exported.")
        return 1

    with db_cursor(autocommit=True) as cursor:
        master = build_master(cursor)
        timeseries, drift = build_timeseries(cursor, master)

    if drift:
        logger.warning(
            "%d snapshots re-scored to a different ai_score than SQL stored. This is "
            "expected only where a summary was edited after scoring.", len(drift)
        )
        for name, stored, fresh, delta in drift[:10]:
            logger.warning("  %-32s stored %+.4f -> recomputed %+.4f (%.4f)",
                           name, stored, fresh, delta)
        if len(drift) > 10:
            logger.warning("  ... and %d more.", len(drift) - 10)
    else:
        logger.info("Every recomputed ai_score matched SQL exactly. Anchors verified unchanged.")

    pending = sum(1 for e in master.values() if e["status"] == STATUS_PENDING)
    approved = sum(1 for e in master.values() if e["status"] == STATUS_APPROVED)

    master_path = MASTER_FILE if apply else f"{MASTER_FILE}.new"
    series_path = TIMESERIES_FILE if apply else f"{TIMESERIES_FILE}.new"

    atomic_write(master_path, master)
    atomic_write(series_path, timeseries)

    print(
        f"\n{'APPLIED' if apply else 'DRY RUN'}\n"
        f"  master:     {len(master):5} skills ({approved} approved, {pending} pending) -> {master_path}\n"
        f"  timeseries: {len(timeseries):5} snapshots -> {series_path}\n"
        f"  score drift: {len(drift)}\n"
    )
    if not apply:
        print("  Re-run with --apply to write the real store files.\n")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Write skills_master.json and skills_timeseries.json for real. Without "
             "this, writes *.json.new so you can inspect first.",
    )
    sys.exit(migrate(parser.parse_args().apply))

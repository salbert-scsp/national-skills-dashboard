"""
Read model for the dashboard.

Assembles one record per approved, scored skill plus its per-occupation Hot Tech
mapping, and the snapshot history behind the trend view.

The record-building half is deliberately pure: build_skill_records() and
bucket_counts() take plain lists of dicts and return plain lists of dicts, so the
bucketing, the EXISTS aggregation, and the array assembly are all testable without a
database, an ODBC driver, or a .env file. Only fetch_* touches SQL.

Normalization is NOT done here. It belongs to the view, because every panel rescales
within its own filtered slice -- see static/dashboard.js. Raw signed scores flow
through this module untouched.
"""

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from database import db_cursor

logger = logging.getLogger(__name__)

# Bucket thresholds, applied to RAW cosine scores and never to normalized ones.
# Bucketing after a per-view rescale would make a skill's category change as filters
# change, which would be nonsense.
AI_SKILL_THRESHOLD = 0.30
AI_ENABLING_THRESHOLD = 0.15

CATEGORY_AI = "AI Skill"
CATEGORY_ENABLING = "AI Enabling Skill"
CATEGORY_NON_AI = "Not AI Skill"

# Ordered strongest to weakest. The UI relies on this order for legend and stack order.
CATEGORY_ORDER = (CATEGORY_AI, CATEGORY_ENABLING, CATEGORY_NON_AI)


# One row per (skill, occupation). Grouped into per-skill records in Python.
#
# ROW_NUMBER picks the newest snapshot rather than `WHERE snapshot_date = (SELECT
# MAX(...))`: on a same-day tie that subquery returns several rows, which then fan out
# across the occupation join and silently duplicate the skill. score.py guards
# same-day duplicates only on the promotion path, so pre-guard history and any manual
# insert are unguarded. The metric_id tiebreak makes the winner unique, and taking all
# six metric columns from one ranked row stops them being spliced across snapshots.
#
# COALESCE(som.is_hot_tech, 1) because a NULL flag means the row predates
# per-relationship tracking, and every such row was written under hot-only ingestion.
# Defaulting to 0 instead would render every already-scored skill non-hot and empty the
# hot-filtered panels.
EXPORT_SQL = """
WITH latest_metrics AS (
    SELECT
        m.skill_id,
        m.snapshot_date,
        m.summary_text,
        m.best_source_name,
        m.ai_correlation_score,
        m.ai_sim,
        m.infra_sim,
        m.lang_sim,
        ROW_NUMBER() OVER (
            PARTITION BY m.skill_id
            ORDER BY m.snapshot_date DESC, m.metric_id DESC
        ) AS snapshot_rank
    FROM Skills_Historical_Metrics AS m
)
SELECT
    s.skill_id,
    s.skill_name,
    s.category,
    lm.snapshot_date,
    lm.summary_text,
    lm.best_source_name,
    lm.ai_correlation_score,
    lm.ai_sim,
    lm.infra_sim,
    lm.lang_sim,
    COALESCE(q.wiki_title, s.skill_name) AS resolved_title,
    som.onet_code,
    o.onet_title,
    CASE WHEN COALESCE(som.is_hot_tech, 1) = 1 THEN 1 ELSE 0 END AS is_hot_tech
FROM Skills_Master AS s
INNER JOIN latest_metrics AS lm
        ON lm.skill_id = s.skill_id AND lm.snapshot_rank = 1
LEFT JOIN Skill_Occupation_Map AS som
        ON som.skill_id = s.skill_id
LEFT JOIN Occupations_Master AS o
        ON o.onet_code = som.onet_code
OUTER APPLY (
    SELECT TOP 1 h.wiki_title
    FROM HITL_Validation_Queue AS h
    WHERE h.skill_name = s.skill_name
      AND h.is_approved = 1
      AND h.wiki_title IS NOT NULL
    ORDER BY h.created_at DESC, h.QueueID DESC
) AS q
WHERE s.is_approved = 1
ORDER BY s.skill_name, som.onet_code;
"""

# Every dated snapshot, for the trend view. Skills with a single snapshot are included
# so the UI can say "one snapshot so far" rather than drawing a misleading flat line.
TREND_SQL = """
SELECT m.skill_id, s.skill_name, m.snapshot_date, m.ai_correlation_score
FROM Skills_Historical_Metrics AS m
INNER JOIN Skills_Master AS s ON s.skill_id = m.skill_id
WHERE s.is_approved = 1
ORDER BY s.skill_name, m.snapshot_date, m.metric_id;
"""


def _to_float(value: Any) -> Optional[float]:
    """
    pyodbc hands back Decimal for DECIMAL(5,4), which json.dumps cannot serialize.

    Mandatory, not defensive. Rounding to 4 places matches the column precision and
    keeps float artifacts out of the page.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return round(float(value), 4)
    return round(float(value), 4)


def classify(raw_score: Optional[float]) -> str:
    """Buckets a RAW cosine score. Never call this with a normalized value."""
    if raw_score is None:
        return CATEGORY_NON_AI
    if raw_score >= AI_SKILL_THRESHOLD:
        return CATEGORY_AI
    if raw_score >= AI_ENABLING_THRESHOLD:
        return CATEGORY_ENABLING
    return CATEGORY_NON_AI


def build_skill_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups occupation-grain rows into one record per skill. Pure function.

    Keys are written out explicitly rather than splatted from the cursor description,
    so a column added to the metrics table later cannot leak into the page by accident.
    """
    records: Dict[int, Dict[str, Any]] = {}

    for row in rows:
        skill_id = row["skill_id"]
        record = records.get(skill_id)

        if record is None:
            raw_score = _to_float(row["ai_correlation_score"])
            snapshot = row["snapshot_date"]
            record = {
                "skill_id": skill_id,
                "skill_name": row["skill_name"],
                "category": row["category"] or "",
                "ai_score": raw_score,
                "ai_sim": _to_float(row["ai_sim"]),
                "infra_sim": _to_float(row["infra_sim"]),
                "lang_sim": _to_float(row["lang_sim"]),
                "ai_category": classify(raw_score),
                "summary": row["summary_text"] or "",
                "best_source_name": row["best_source_name"] or "",
                "resolved_title": row["resolved_title"] or row["skill_name"],
                "snapshot_date": snapshot.isoformat() if snapshot else None,
                "occupations": [],
            }
            records[skill_id] = record

        # A LEFT JOIN with no mapping yields a NULL onet_code. Keep the skill, skip the
        # phantom occupation: empty arrays are safe everywhere in the UI, whereas
        # dropping the skill would hide a scored result over a bookkeeping gap.
        if row["onet_code"] is None:
            continue

        record["occupations"].append({
            "onet_code": row["onet_code"],
            "onet_title": row["onet_title"] or "",
            "is_hot_tech": bool(row["is_hot_tech"]),
        })

    result = list(records.values())

    for record in result:
        # Hot for at least one occupation. This is the EXISTS semantics the macro views
        # filter on, computed here so it is unit-testable.
        record["is_hot_tech_anywhere"] = any(
            occ["is_hot_tech"] for occ in record["occupations"]
        )
        if not record["occupations"]:
            logger.warning(
                "Skill %r is scored but mapped to no occupation.", record["skill_name"]
            )

    result.sort(key=lambda r: (r["ai_score"] is None, -(r["ai_score"] or 0.0)))
    return result


def build_trend_series(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Groups snapshot rows into {skill_name: [{date, score}, ...]}. Pure function."""
    series: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        snapshot = row["snapshot_date"]
        if snapshot is None:
            continue
        series.setdefault(row["skill_name"], []).append({
            "date": snapshot.isoformat(),
            "score": _to_float(row["ai_correlation_score"]),
        })
    return series


def bucket_counts(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Counts records per AI category, in fixed strongest-to-weakest order."""
    counts = {name: 0 for name in CATEGORY_ORDER}
    for record in records:
        counts[record["ai_category"]] = counts.get(record["ai_category"], 0) + 1
    return [{"category": name, "count": counts[name]} for name in CATEGORY_ORDER]


def build_summary(records: List[Dict[str, Any]], tools_discovered: int = 0) -> Dict[str, Any]:
    """
    KPI figures for the header row.

    Deliberately does NOT report "skills hot in at least one job": only hot skills are
    ever promoted, so that count always equals skills_scored and the tile would be
    dead space. `tools_discovered` is the useful counterpart -- every tool O*NET listed,
    scored or not -- which makes the funnel visible.
    """
    occupations = {
        occ["onet_code"] for record in records for occ in record["occupations"]
    }
    at_threshold = sum(
        1 for record in records
        if record["ai_score"] is not None and record["ai_score"] >= AI_SKILL_THRESHOLD
    )
    return {
        "skills_scored": len(records),
        "tools_discovered": tools_discovered,
        "occupations_covered": len(occupations),
        "at_ai_threshold": at_threshold,
        "ai_threshold": AI_SKILL_THRESHOLD,
        "enabling_threshold": AI_ENABLING_THRESHOLD,
    }


def _rows_from_cursor(cursor) -> List[Dict[str, Any]]:
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def load_dashboard() -> Dict[str, Any]:
    """
    Reads everything the dashboard page needs.

    Both queries run inside one connection so the skill records and the trend series
    cannot disagree about what has been promoted.
    """
    try:
        with db_cursor(autocommit=True) as cursor:
            cursor.execute(EXPORT_SQL)
            skill_rows = _rows_from_cursor(cursor)
            cursor.execute(TREND_SQL)
            trend_rows = _rows_from_cursor(cursor)
            # Every tool ever discovered, hot or not. The gap between this and the
            # scored count is the review backlog plus the non-hot tail.
            cursor.execute("SELECT COUNT(*) FROM Skills_Master")
            tools_discovered = int(cursor.fetchone()[0])
    except Exception:
        logger.exception("Could not load dashboard data.")
        return {
            "skills": [],
            "trends": {},
            "buckets": bucket_counts([]),
            "occupations": [],
            "summary": build_summary([], 0),
            "categories": list(CATEGORY_ORDER),
            "failed": True,
        }

    records = build_skill_records(skill_rows)

    occupations = sorted(
        {
            (occ["onet_code"], occ["onet_title"])
            for record in records
            for occ in record["occupations"]
        },
        key=lambda pair: pair[1] or pair[0],
    )

    logger.info(
        "Dashboard loaded: %d skills, %d occupations, %d trend series.",
        len(records), len(occupations), len(set(row["skill_name"] for row in trend_rows)),
    )

    return {
        "skills": records,
        "trends": build_trend_series(trend_rows),
        "buckets": bucket_counts(records),
        "occupations": [{"onet_code": code, "onet_title": title} for code, title in occupations],
        "summary": build_summary(records, tools_discovered),
        "categories": list(CATEGORY_ORDER),
        "failed": False,
    }

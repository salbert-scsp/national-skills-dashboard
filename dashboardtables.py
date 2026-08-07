"""
Flattening and export layer between the JSON store and the UI.

Joins skills_master.json to skills_timeseries.json and produces plain lists of dicts
that drop straight into a DataFrame. The Streamlit app does no joining of its own.

Every builder here is a PURE FUNCTION over data passed in as arguments. Only
load_dashboard() and export_dashboard_json() touch the disk. That is deliberate: the
SQL version it replaces could only be exercised by standing up a database, and these
can be checked with two literal dicts.

Nothing in this module recomputes a category bucket. The bucket is stored on each
snapshot, computed from the raw score at scoring time by sortingalgorithmnew. If it
were re-derived here from anything the UI had already normalized, a skill's class
would change as the user filtered.
"""

import logging
from typing import Any, Dict, List

import storage
from json_store import (
    STATUS_APPROVED,
    STATUS_PENDING,
    STATUS_REJECTED,
    latest_snapshots,
    load_master,
    load_timeseries,
)
from sortingalgorithmnew import (
    AI_ENABLING_THRESHOLD,
    AI_SKILL_THRESHOLD,
    BUCKET_AI,
    BUCKET_ENABLING,
    BUCKET_NOT_AI,
)

logger = logging.getLogger(__name__)

EXPORT_FILE = storage.EXPORT_FILE

# Fixed display order. Counting with a dict built from the data would order buckets
# by whichever happened to appear first, and the legend would reshuffle between runs.
BUCKET_ORDER = (BUCKET_AI, BUCKET_ENABLING, BUCKET_NOT_AI)

# Columns the public table may show. The reference page and its URL are deliberately
# absent: raw source links are for the review screen, not the summary table.
PUBLIC_COLUMNS = (
    "skill_name",
    "category",
    "ai_score",
    "category_bucket",
    "tech_base_sim",
    "ml_pipeline_sim",
    "embedded_ai_sim",
    "occupation_count",
)


def build_dashboard_rows(
    master: Dict[str, Any],
    timeseries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    One row per APPROVED skill that has been scored, newest snapshot only.

    A skill with no snapshot is excluded rather than emitted with a null score. A row
    reading "n/a" in every metric column is noise in a table people sort by score, and
    an unscored skill is not a finding.

    `wikipedia_summary` and `resolved_title` are included for the detail view. The UI
    is responsible for keeping them out of the summary table -- see PUBLIC_COLUMNS.
    """
    newest = latest_snapshots(timeseries)
    rows: List[Dict[str, Any]] = []

    for skill_name, entry in master.items():
        if entry.get("status") != STATUS_APPROVED:
            continue
        snapshot = newest.get(skill_name)
        if not snapshot:
            continue

        codes = entry.get("onet_codes") or []
        rows.append({
            "skill_name": skill_name,
            "category": entry.get("category") or "",
            "ai_score": snapshot.get("ai_score"),
            "category_bucket": snapshot.get("category_bucket"),
            "tech_base_sim": snapshot.get("tech_base_sim"),
            "ml_pipeline_sim": snapshot.get("ml_pipeline_sim"),
            "embedded_ai_sim": snapshot.get("embedded_ai_sim"),
            "wikipedia_summary": entry.get("wikipedia_summary") or "",
            "resolved_title": entry.get("resolved_title") or "",
            # Absent on entries migrated from SQL, which never recorded it. The
            # drawer renders "n/a" rather than guessing "Wikipedia", since Wikidata
            # is also a possible source.
            "best_source_name": entry.get("best_source_name") or "",
            # Object form for the drawer, which badges each occupation with its OWN
            # hot status. The flat lists stay for the by-job filter and the snapshot
            # records, which only ever need codes.
            "occupations": entry.get("occupations") or [],
            "onet_codes": codes,
            "onet_titles": entry.get("onet_titles") or [],
            "occupation_count": len(codes),
            "quarter": snapshot.get("quarter"),
            "snapshot_date": snapshot.get("snapshot_date"),
            "is_hot_tech_anywhere": bool(entry.get("is_hot_tech_anywhere")),
        })

    # Highest AI relevance first. Ties break on name so the order is total and does
    # not shuffle between renders. None sorts last -- it cannot occur here, since
    # unscored skills are excluded above, but the guard costs nothing.
    rows.sort(key=lambda row: (row["ai_score"] is None, -(row["ai_score"] or 0.0), row["skill_name"]))
    return rows


def bucket_counts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Counts per category bucket, in fixed display order.

    Reads the STORED bucket on each row. Buckets with a zero count are still emitted,
    so the legend does not lose an entry when a filter empties it.
    """
    counts = {bucket: 0 for bucket in BUCKET_ORDER}
    for row in rows:
        bucket = row.get("category_bucket")
        if bucket in counts:
            counts[bucket] += 1
        elif bucket:
            # An unrecognized bucket means someone wrote a value outside
            # sortingalgorithmnew's vocabulary. Loud, because it would otherwise
            # vanish from the composition chart while still inflating the table.
            logger.error("Unrecognized category bucket %r on %r.", bucket, row.get("skill_name"))

    total = sum(counts.values()) or 1
    return [
        {"category": bucket, "count": counts[bucket], "share": counts[bucket] / total}
        for bucket in BUCKET_ORDER
    ]


def build_trend_series(timeseries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Groups snapshots as {skill_name: [{quarter, date, score}, ...]}, oldest first.

    This is the shape the dashboard's trend chart consumes: it picks one skill from a
    selector and plots its points, so a per-skill index beats a flat list it would have
    to filter on every redraw.
    """
    series: Dict[str, List[Dict[str, Any]]] = {}
    for record in timeseries:
        name = record.get("skill_name")
        if not name or record.get("ai_score") is None:
            continue
        series.setdefault(name, []).append({
            "quarter": record.get("quarter"),
            "date": record.get("snapshot_date"),
            "score": record.get("ai_score"),
        })

    # Sorted per skill rather than trusting file order: the store is appended to over
    # many runs, and a line chart fed unsorted points zigzags.
    for points in series.values():
        points.sort(key=lambda point: (point["quarter"] or "", point["date"] or ""))
    return series


def build_occupation_index(master: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Deduped [{onet_code, onet_title}] across every skill, for the by-job selector.

    Built from master rather than from the dashboard rows so an occupation whose
    skills are all still pending review still appears, rather than the selector
    silently losing a job.
    """
    seen: Dict[str, str] = {}
    for entry in master.values():
        for occupation in entry.get("occupations") or []:
            code = occupation.get("onet_code")
            if code and code not in seen:
                seen[code] = occupation.get("onet_title") or code
    return [
        {"onet_code": code, "onet_title": title}
        for code, title in sorted(seen.items(), key=lambda item: item[1])
    ]


def build_trend_frame(timeseries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    The full time series, long-form and sorted, ready to chart.

    Sorted by (skill, quarter) rather than trusting file order: the store is appended
    to across many runs, and a line chart fed unsorted points draws them in file
    order, which zigzags.
    """
    frame = [
        {
            "skill_name": record.get("skill_name"),
            "quarter": record.get("quarter"),
            "snapshot_date": record.get("snapshot_date"),
            "ai_score": record.get("ai_score"),
            "tech_base_sim": record.get("tech_base_sim"),
            "ml_pipeline_sim": record.get("ml_pipeline_sim"),
            "embedded_ai_sim": record.get("embedded_ai_sim"),
            "category_bucket": record.get("category_bucket"),
        }
        for record in timeseries
        if record.get("skill_name")
    ]
    frame.sort(key=lambda item: (item["skill_name"], item["quarter"] or "", item["snapshot_date"] or ""))
    return frame


def build_review_rows(master: Dict[str, Any], status: str = STATUS_PENDING) -> List[Dict[str, Any]]:
    """
    Master entries in a given review state that a human can ACT on.

    Unlike the dashboard rows, these deliberately DO carry the reference page and the
    gate reason. A reviewer judging a definition needs to see where it came from and
    why the pipeline was unsure; a dashboard reader does not.

    Pending entries with no definition text are EXCLUDED. Most of them are non-hot
    tools that were recorded for the occupation map and deliberately never audited --
    hundreds of them -- and there is nothing for a reviewer to approve or reject. The
    invariant is the one the 388-row backlog taught: the rows that appear in the queue
    must be a subset of the rows a human can actually approve. Use
    build_discovered_rows() to see the unaudited ones.
    """
    rows = [
        {
            "skill_name": name,
            "category": entry.get("category") or "",
            "resolved_title": entry.get("resolved_title") or "",
            "wikipedia_summary": entry.get("wikipedia_summary") or "",
            "gate_reason": entry.get("gate_reason") or "",
            "cross_score": entry.get("cross_score"),
            "is_credible": entry.get("is_credible"),
            "best_source_name": entry.get("best_source_name") or "",
            "report_note": entry.get("report_note"),
            # The automated re-resolution's finding, when one exists. Carried so the
            # card can offer its suggestion as a single button instead of leaving the
            # reviewer to work out which article the skill actually names.
            "second_pass": entry.get("second_pass") or None,
            # Object form, so the card can name the occupation this skill came from.
            # Without it the template has nothing to render and every card reads
            # "Not mapped" even when the skill is mapped to five jobs.
            "occupations": entry.get("occupations") or [],
            "onet_codes": entry.get("onet_codes") or [],
            "onet_titles": entry.get("onet_titles") or [],
            "last_updated": entry.get("last_updated"),
        }
        for name, entry in master.items()
        if entry.get("status") == status
        and (entry.get("wikipedia_summary") or "").strip()
    ]
    rows.sort(key=lambda row: row["skill_name"])
    return rows


def build_discovered_rows(master: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Tools that were discovered and mapped but never audited.

    These are overwhelmingly non-hot tools, which the pipeline records for the
    occupation map and then skips before any network call. They are counted in
    "tools discovered" and are legitimately absent from both the dashboard and the
    review queue, so this exists purely so the gap is inspectable rather than
    mysterious.
    """
    rows = [
        {
            "skill_name": name,
            "category": entry.get("category") or "",
            "onet_codes": entry.get("onet_codes") or [],
            "onet_titles": entry.get("onet_titles") or [],
            "is_hot_tech_anywhere": bool(entry.get("is_hot_tech_anywhere")),
        }
        for name, entry in master.items()
        if entry.get("status") == STATUS_PENDING
        and not (entry.get("wikipedia_summary") or "").strip()
    ]
    rows.sort(key=lambda row: row["skill_name"])
    return rows


def build_summary(master: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Headline counts for the KPI row."""
    occupations = set()
    for entry in master.values():
        occupations.update(entry.get("onet_codes") or [])

    statuses = {STATUS_APPROVED: 0, STATUS_PENDING: 0, STATUS_REJECTED: 0}
    for entry in master.values():
        status = entry.get("status")
        if status in statuses:
            statuses[status] += 1

    # pending_review counts only entries a human can act on, so it agrees with the
    # length of the review table. The raw pending status count includes hundreds of
    # never-audited tools and would make the tab badge lie.
    reviewable = sum(
        1 for entry in master.values()
        if entry.get("status") == STATUS_PENDING
        and (entry.get("wikipedia_summary") or "").strip()
    )

    return {
        "skills_scored": len(rows),
        "tools_discovered": len(master),
        "occupations_covered": len(occupations),
        "at_ai_threshold": sum(1 for row in rows if row.get("category_bucket") == BUCKET_AI),
        "pending_review": reviewable,
        "not_audited": statuses[STATUS_PENDING] - reviewable,
        "approved": statuses[STATUS_APPROVED],
        "rejected": statuses[STATUS_REJECTED],
        # Surfaced so the templates state the thresholds without hardcoding them in
        # two places. They come from the scoring engine, which is the single source.
        "ai_threshold": AI_SKILL_THRESHOLD,
        "enabling_threshold": AI_ENABLING_THRESHOLD,
    }


def load_dashboard() -> Dict[str, Any]:
    """
    Reads both store files and assembles everything the UI needs in one payload.

    The only disk-touching function here besides the exporter. On a corrupted store it
    lets json_store.StoreCorrupted propagate: the UI must show that loudly rather than
    render an empty dashboard that looks like a pipeline that found nothing.
    """
    master = load_master()
    timeseries = load_timeseries()

    rows = build_dashboard_rows(master, timeseries)
    return {
        "skills": rows,
        "buckets": bucket_counts(rows),
        # Per-skill index for the trend chart; the flat frame is kept alongside for
        # anything that wants to tabulate the whole series.
        "trends": build_trend_series(timeseries),
        "trend_rows": build_trend_frame(timeseries),
        "occupations": build_occupation_index(master),
        "categories": list(BUCKET_ORDER),
        "pending": build_review_rows(master, STATUS_PENDING),
        "discovered": build_discovered_rows(master),
        "summary": build_summary(master, rows),
        "failed": False,
    }


def export_dashboard_json(path: str = None) -> str:
    """
    Writes the flattened payload to a standalone file.

    Not needed by the web app, which calls load_dashboard() directly. This exists for
    handing a single self-contained artifact to something else.

    Written atomically, unlike the raw open() this replaced: a crash part-way through
    used to leave a truncated export behind that still looked like a valid file.
    """
    path = path or EXPORT_FILE
    payload = load_dashboard()
    storage.save_dashboard_export(path, payload)
    logger.info("Exported %d skills to %s.", len(payload["skills"]), path)
    return path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    data = load_dashboard()
    print(f"\nskills scored:       {data['summary']['skills_scored']}")
    print(f"tools discovered:    {data['summary']['tools_discovered']}")
    print(f"occupations covered: {data['summary']['occupations_covered']}")
    print(f"pending review:      {data['summary']['pending_review']}")
    print("\ncomposition:")
    for bucket in data["buckets"]:
        print(f"  {bucket['category']:20} {bucket['count']:5}  {bucket['share']:.1%}")
    export_dashboard_json()

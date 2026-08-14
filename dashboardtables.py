"""
STAGE: Scoring

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
    REPORT_GATE_REASON,
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
    "sub_category",
    "tech_base_sim",
    "ml_pipeline_sim",
    "embedded_ai_sim",
    "embeds_ai",
    "classification_confidence",
    "occupation_count",
)

# The three states, spelled the way the dashboard says them. "Not checked" is a first
# class answer and not a blank: a skill nobody has searched for and a skill searched and
# found to have no AI features are different facts, and a reader deciding whether to
# trust the class needs to be able to tell them apart.
EMBEDS_AI_LABELS = {
    True: "AI Embedding",
    False: "No AI Embedding",
    None: "Embedding not checked",
}


def embeds_ai_label(value) -> str:
    return EMBEDS_AI_LABELS.get(value if isinstance(value, bool) else None)


# --- The reader-facing score --------------------------------------------------
#
# ai_score is a RAW COSINE. It runs negative, it is not a percentage, and its absolute
# value means nothing to anyone who has not read sortingalgorithmnew.py -- so the card
# could print "AI Enabling Skill, 0.3549" directly above "AI Skill, 0.2952" and look
# broken. It is not broken: those two are in different buckets because the top bucket
# also asks the engineering pole. But no reader can see that from the number.
#
# display_ai_score maps each bucket onto its own band so visual rank always matches tier.
#
#   (raw_low, raw_high, band_low, band_high)
#
# The raw bounds are OBSERVED over the live store and then FROZEN as constants. Computing
# them from the current rows instead would mean a skill's displayed score changes when an
# unrelated skill is added -- which is precisely the defect in the view-relative min-max
# normalizer this replaces.
#
# The AI Enabling band starts at -0.05 and not at 0.12. Measured: 216 of 400 AI Enabling
# skills score BELOW 0.12, which is not an anomaly -- rule 1 promotes on tech_base_sim or
# ml_pipeline_sim and never reads ai_score, so C++ earns the bucket at ai_score 0.083. A
# 0.12 floor would collapse over half the bucket onto a single tied value.
#
# The band edges are exact at the TWO DECIMALS the value is displayed to, which is why
# they read 0.69 and 0.29 rather than 0.699 and 0.299. Rounding after clamping to 0.299
# produces "0.30" -- numerically inside the band and visually identical to the bottom of
# the band above it, which defeats the point. The bands are separated where the reader
# actually sees them.
DISPLAY_BANDS = {
    #                raw_low  raw_high  band_low  band_high
    BUCKET_AI:       (0.29,   0.75,     0.70,     1.00),
    BUCKET_ENABLING: (-0.05,  0.36,     0.30,     0.69),
    BUCKET_NOT_AI:   (0.00,   0.29,     0.00,     0.29),
}


def compute_display_ai_score(raw_ai_score, category_bucket):
    """
    The banded score the dashboard shows. Presentation only -- NOTHING may decide on it.

    A banded score is discontinuous at the boundaries by construction: raw 0.289 and
    0.291 display 0.30 and 0.70. That is the correct trade for a reader-facing view and a
    disqualifying one for a rule, so this lives in the presentation layer and no engine
    module imports it. Feeding it back into a classification would turn a rounding
    difference into a tier difference.

    Every branch CLAMPS to its own band. Two real skills escape their band without it:
    Strategic Reporting Systems (raw -0.0185) would display 0.060, inside the Not AI band,
    and Transcription system software (raw 0.3549) would display 0.708, inside the AI
    Skill band -- inverting the one property this function exists to guarantee.

    None returns None rather than 0.0. An unscored skill has no display score, and 0.0
    would sort it above every measured skill in an ascending sort as though it had been
    measured and found to be nothing.
    """
    if raw_ai_score is None:
        return None

    band = DISPLAY_BANDS.get(category_bucket)
    if band is None:
        # An unrecognised bucket is a data problem, not a rendering problem. Returning
        # None shows "n/a" rather than silently filing it at the bottom of the Not AI
        # band, where it would look like a real measurement.
        return None

    raw_low, raw_high, band_low, band_high = band
    raw = float(raw_ai_score)

    if category_bucket == BUCKET_NOT_AI:
        # Pass-through: a raw 0.28 displays 0.28. These scores are already in the band's
        # numeric range, so stretching them would inflate them for no gain, and the clamp
        # is what handles the 31 skills measuring below zero.
        scaled = raw
    else:
        span = raw_high - raw_low
        scaled = band_low + (band_high - band_low) * ((raw - raw_low) / span)

    # Rounded BEFORE clamping, so the clamp binds on the value the reader sees. Clamping
    # first lets rounding carry a value back out of its band -- 0.299 clamps fine and then
    # displays as "0.30", the bottom of the band above.
    return min(band_high, max(band_low, round(scaled, 2)))


def is_published(entry: Dict[str, Any]) -> bool:
    """
    Whether a master entry belongs on the reader-facing dashboard.

    Approved, or reported-and-awaiting-review. A report is a request, not a verdict, so
    it must not remove data from the charts on one unverified click -- see
    build_dashboard_rows and review_actions.report_skill, which both promise this.

    REJECTED IS NEVER PUBLISHED, including a rejected skill that was once reported: the
    gate reason survives the rejection, so this tests the status first and asks about the
    report only for a PENDING entry. Testing the gate reason alone would republish
    everything a reviewer has thrown out.
    """
    status = entry.get("status")
    if status == STATUS_APPROVED:
        return True
    return status == STATUS_PENDING and entry.get("gate_reason") == REPORT_GATE_REASON


def build_dashboard_rows(
    master: Dict[str, Any],
    timeseries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    One row per PUBLISHED skill that has been scored, newest snapshot only.

    Published means approved, OR reported by a viewer and awaiting review. That second
    case is the whole reason this is not simply a status check.

    A REPORT IS A REQUEST FOR REVIEW, NOT A VERDICT. report_skill() moves a skill to
    pending so it surfaces in the queue, and its docstring promises the skill "stays on
    the dashboard" and that "the existing snapshot means the dashboard row survives until
    someone decides". The reader is told the same thing in the report form: "The score
    stays as it is until someone reviews the report."

    None of that was true. This filter tested status alone, so a single unverified click
    by any viewer pulled the skill straight out of the charts -- exactly the behaviour
    both messages promise it will not have. The skill is still pending, still in the
    queue, still first in the priority order; it is only its DISAPPEARANCE that was wrong.

    A skill with no snapshot is excluded rather than emitted with a null score. A row
    reading "n/a" in every metric column is noise in a table people sort by score, and
    an unscored skill is not a finding. That applies to reported skills too: the snapshot
    is what survives a report, so a report cannot publish something that was never scored.

    `wikipedia_summary` and `resolved_title` are included for the detail view. The UI
    is responsible for keeping them out of the summary table -- see PUBLIC_COLUMNS.
    """
    newest = latest_snapshots(timeseries)
    rows: List[Dict[str, Any]] = []

    for skill_name, entry in master.items():
        if not is_published(entry):
            continue
        snapshot = newest.get(skill_name)
        if not snapshot:
            continue

        codes = entry.get("onet_codes") or []
        rows.append({
            "skill_name": skill_name,
            "category": entry.get("category") or "",
            "ai_score": snapshot.get("ai_score"),
            # What the reader sees. Derived here rather than in the browser so the table,
            # the tooltip, the drawer and the CSV cannot drift apart, and so the banding
            # rule has exactly one definition. See compute_display_ai_score.
            "display_ai_score": compute_display_ai_score(
                snapshot.get("ai_score"), snapshot.get("category_bucket")),
            # Falls back to ai_score with a zero boost on a snapshot written before the
            # split, which is what it was. The drawer shows the two whenever they differ.
            "ai_score_base": snapshot.get("ai_score_base", snapshot.get("ai_score")),
            "embedded_ai_boost": snapshot.get("embedded_ai_boost") or 0.0,
            "lexical_ai_boost": snapshot.get("lexical_ai_boost") or 0.0,
            "lexical_ai_terms": snapshot.get("lexical_ai_terms") or [],
            "contrast_sim": snapshot.get("contrast_sim"),
            "legacy_sim": snapshot.get("legacy_sim"),
            "ai_engineering_sim": snapshot.get("ai_engineering_sim"),
            "ai_generative_sim": snapshot.get("ai_generative_sim"),
            "category_bucket": snapshot.get("category_bucket"),
            "tech_base_sim": snapshot.get("tech_base_sim"),
            "ml_pipeline_sim": snapshot.get("ml_pipeline_sim"),
            "embedded_ai_sim": snapshot.get("embedded_ai_sim"),
            # Read from the SNAPSHOT, not the entry, so the tag and the class it produced
            # always come from the same moment. Reading the entry would show today's
            # verdict beside a bucket decided by last quarter's.
            "embeds_ai": snapshot.get("embeds_ai"),
            "embeds_ai_label": embeds_ai_label(snapshot.get("embeds_ai")),
            "embeds_ai_evidence": entry.get("embeds_ai_evidence") or "",
            "embeds_ai_evidence_url": entry.get("embeds_ai_evidence_url"),
            "embeds_ai_checked_at": entry.get("embeds_ai_checked_at"),
            # Rules-engine output. None on snapshots written before the engine landed,
            # which the UI renders as "n/a" rather than inventing a sub-category.
            "sub_category": snapshot.get("sub_category"),
            "is_flagship_version_evaluated": bool(snapshot.get("is_flagship_version_evaluated")),
            "is_generic_category": bool(snapshot.get("is_generic_category")),
            "flagship_version": snapshot.get("flagship_version"),
            "flagship_source": snapshot.get("flagship_source"),
            "decision_metric": snapshot.get("decision_metric"),
            "decision_threshold": snapshot.get("decision_threshold"),
            "decision_margin": snapshot.get("decision_margin"),
            "in_semantic_variance_band": bool(snapshot.get("in_semantic_variance_band")),
            "classification_confidence": snapshot.get("classification_confidence"),
            "wikipedia_summary": entry.get("wikipedia_summary") or "",
            "resolved_title": entry.get("resolved_title") or "",
            # Only set for a reference outside Wikipedia. The drawer links to it as
            # given rather than deriving an article URL from a non-article title.
            "reference_url": entry.get("reference_url") or "",
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
    Deduped [{onet_code, onet_title, major_group}] across every skill, for the by-job
    selector.

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
        {"onet_code": code, "onet_title": title, "major_group": major_group_of(code)}
        for code, title in sorted(seen.items(), key=lambda item: item[1])
    ]


# The SOC major groups, keyed by the two digits before the dash in an O*NET code.
# Only the groups that can plausibly appear in a hot-tech ingest are named; anything
# else falls back to the bare code, which is still a usable slice.
SOC_MAJOR_GROUPS = {
    "11": "Management",
    "13": "Business and Financial Operations",
    "15": "Computer and Mathematical",
    "17": "Architecture and Engineering",
    "19": "Life, Physical, and Social Science",
    "21": "Community and Social Service",
    "23": "Legal",
    "25": "Educational Instruction and Library",
    "27": "Arts, Design, Entertainment, Sports, and Media",
    "29": "Healthcare Practitioners and Technical",
    "31": "Healthcare Support",
    "33": "Protective Service",
    "35": "Food Preparation and Serving Related",
    "37": "Building and Grounds Cleaning and Maintenance",
    "39": "Personal Care and Service",
    "41": "Sales and Related",
    "43": "Office and Administrative Support",
    "45": "Farming, Fishing, and Forestry",
    "47": "Construction and Extraction",
    "49": "Installation, Maintenance, and Repair",
    "51": "Production",
    "53": "Transportation and Material Moving",
    "55": "Military Specific",
}


def major_group_of(onet_code: str) -> str:
    """The two-digit SOC major group prefix, e.g. '15-1252.00' -> '15'."""
    code = str(onet_code or "").strip()
    return code[:2] if len(code) >= 2 and code[:2].isdigit() else ""


def build_major_group_index(occupations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    The distinct SOC major groups present, with how many occupations each covers.

    Derived from the occupation index rather than from master a second time, so the
    groups offered in the selector can never name a slice the selector cannot fill.
    """
    counts: Dict[str, int] = {}
    for occupation in occupations:
        group = occupation.get("major_group")
        if group:
            counts[group] = counts.get(group, 0) + 1
    return [
        {
            "code": group,
            "title": SOC_MAJOR_GROUPS.get(group, f"SOC {group}"),
            "occupation_count": counts[group],
        }
        for group in sorted(counts)
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
            "sub_category": record.get("sub_category"),
            "classification_confidence": record.get("classification_confidence"),
        }
        for record in timeseries
        if record.get("skill_name")
    ]
    frame.sort(key=lambda item: (item["skill_name"], item["quarter"] or "", item["snapshot_date"] or ""))
    return frame


# Where a pending item sits in the queue. Lower sorts first.
#
# The ordering is by WHAT THE REVIEWER CAN DO, not by what the pipeline thought. An item
# carrying an accepted-in-one-click answer is a few seconds of work; a viewer's report is
# a live dashboard row somebody has already said is wrong; everything else is the long
# tail of judgement calls. Alphabetical order mixed all three together, so the cheap wins
# and the urgent corrections were spread evenly across forty pages of hard ones.
PRIORITY_ACTIONABLE = 0     # the second pass has an answer waiting behind one button
PRIORITY_REPORTED = 1       # a dashboard viewer reported it; it is live and wrong
PRIORITY_NEEDS_HUMAN = 2    # the model gave up; only a person can move this one
PRIORITY_REST = 3


def suggestion_ready(entry: Dict[str, Any]) -> bool:
    """
    True when the second pass has a page to offer that is NOT the page already on file.

    The differing-title test is the same one the card renders on. A proposal can resolve
    back onto the current page -- "AutoCAD Civil 3D" redirects to "AutoCAD" -- and an
    offer to replace a page with itself is not an action, so it must not sort as one.
    """
    block = entry.get("second_pass")
    if not isinstance(block, dict):
        return False
    if block.get("outcome") not in ("suggested", "suggested_strong", "suggested_weak"):
        return False
    proposed = (block.get("proposed_title") or "").strip()
    return bool(proposed) and proposed != (entry.get("resolved_title") or "")


def draft_ready(entry: Dict[str, Any]) -> bool:
    """True when a written definition is waiting to be accepted or rejected."""
    block = entry.get("second_pass")
    if not isinstance(block, dict):
        return False
    return block.get("outcome") == "drafted" and bool(
        (block.get("drafted_definition") or "").strip()
    )


def review_priority(entry: Dict[str, Any]) -> int:
    """
    Which band a pending item belongs to. Lower sorts first.

    NEEDS_HUMAN sits below REPORTED and above the rest. An item the model has given up
    on is not urgent the way a live wrong page is, but it is the only band that will
    never move on its own, so burying it among items still awaiting automation would
    hide exactly the work a person has to do.
    """
    if suggestion_ready(entry) or draft_ready(entry):
        return PRIORITY_ACTIONABLE
    if entry.get("gate_reason") == REPORT_GATE_REASON:
        return PRIORITY_REPORTED
    if entry.get("ai_status") == "cannot_determine":
        return PRIORITY_NEEDS_HUMAN
    return PRIORITY_REST


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
            # Set only for a reference that is not on Wikipedia. The card links to it
            # directly rather than building an en.wikipedia.org URL out of a title that
            # is not a Wikipedia title.
            "reference_url": entry.get("reference_url") or "",
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
            # Computed once, here, and used by BOTH the sort below and the card. The
            # card used to work the differing-title test out for itself in Jinja, which
            # is exactly how an ordering and the thing it orders drift apart.
            "suggestion_ready": suggestion_ready(entry),
            "draft_ready": draft_ready(entry),
            # Why the second pass will skip this entry, or None if it will try again.
            # Surfaced as a badge so a queue the automation has stopped working on is
            # visibly different from one it simply has not reached.
            "ai_status": entry.get("ai_status"),
            "ai_status_detail": entry.get("ai_status_detail"),
            "priority": review_priority(entry),
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
    # Band first, name second. The name is kept as the tie-break because it is unique,
    # which makes the ordering TOTAL: without it two items in the same band could be
    # ordered differently on two renders and a row could appear on two pages, or on
    # none. That was the reason this sorted by name in the first place.
    rows.sort(key=lambda row: (row["priority"], row["skill_name"]))
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
    occupations = build_occupation_index(master)
    return {
        "skills": rows,
        "buckets": bucket_counts(rows),
        # Per-skill index for the trend chart; the flat frame is kept alongside for
        # anything that wants to tabulate the whole series.
        "trends": build_trend_series(timeseries),
        "trend_rows": build_trend_frame(timeseries),
        "occupations": occupations,
        "major_groups": build_major_group_index(occupations),
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

"""
Promotion and scoring.

This module is the ONLY writer of Skills_Historical_Metrics. pipeline.py discovers
skills and queues candidates; nothing is scored until it is promoted here, either by
a human reviewer or by the auto-approval gate. That keeps exactly one metrics row per
promotion and means a rejected skill leaves no scored snapshot behind.
"""

import logging
from typing import Optional, Tuple

from database import QUEUE_APPROVED as APPROVED
from database import QUEUE_PENDING as PENDING
from database import QUEUE_REJECTED as REJECTED
from database import db_cursor
from scraping import resolve_reviewer_url, validate_wikipedia_reference
from Scoring_Algorithm import calculate_ai_correlation

logger = logging.getLogger(__name__)


def promote_queue_item(
    queue_id: int,
    override_summary: Optional[str] = None,
    auto_approved: bool = False,
) -> bool:
    """
    Scores a queued candidate and promotes it into the master tables.

    Runs as a single transaction: either the metrics row, the Skills_Master approval
    flag, and the queue status all land, or none of them do.

    `auto_approved` only affects logging and the is_override flag; the write path is
    identical, so human and automatic promotions produce structurally identical rows.
    """
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT skill_name, category, wiki_summary, best_source_name, is_credible
            FROM HITL_Validation_Queue
            WHERE QueueID = ?
            """,
            (queue_id,),
        )
        row = cursor.fetchone()

        if not row:
            logger.error("Queue item #%s not found.", queue_id)
            return False

        skill_name, category, wiki_summary, best_source_name, is_credible = row

        summary_to_score = (
            override_summary.strip() if override_summary and override_summary.strip() else wiki_summary
        )
        if not summary_to_score:
            logger.error(
                "Queue item #%s (%r) has no summary text to score.", queue_id, skill_name
            )
            return False

        is_override = bool(override_summary and override_summary.strip() != (wiki_summary or "").strip())

        cursor.execute("SELECT skill_id FROM Skills_Master WHERE skill_name = ?", (skill_name,))
        skill_row = cursor.fetchone()

        if skill_row:
            skill_id = skill_row[0]
            cursor.execute(
                "UPDATE Skills_Master SET category = ?, is_approved = 1 WHERE skill_id = ?",
                (category, skill_id),
            )
        else:
            # Discovery normally inserts this in pipeline.py; cover the case where a
            # queue row outlived its Skills_Master entry.
            cursor.execute(
                "INSERT INTO Skills_Master (skill_name, category, is_approved) VALUES (?, ?, 1)",
                (skill_name, category),
            )
            cursor.execute("SELECT skill_id FROM Skills_Master WHERE skill_name = ?", (skill_name,))
            skill_id = cursor.fetchone()[0]

        # Guard against a second promotion of the same skill on the same day, which
        # would reintroduce the duplicate-snapshot problem from the other direction.
        cursor.execute(
            """
            SELECT COUNT(*) FROM Skills_Historical_Metrics
            WHERE skill_id = ? AND snapshot_date = CAST(GETDATE() AS DATE)
            """,
            (skill_id,),
        )
        if cursor.fetchone()[0] > 0:
            logger.warning(
                "Skill %r already has a snapshot dated today. Marking queue item #%s "
                "approved without writing a duplicate metrics row.",
                skill_name, queue_id,
            )
            cursor.execute(
                "UPDATE HITL_Validation_Queue SET is_approved = ? WHERE QueueID = ?",
                (APPROVED, queue_id),
            )
            return True

        metrics = calculate_ai_correlation(skill_name, category, summary_to_score)

        cursor.execute(
            """
            INSERT INTO Skills_Historical_Metrics (
                skill_id, snapshot_date, summary_text, best_source_name, is_credible,
                ai_correlation_score, ai_sim, infra_sim, lang_sim, is_override
            ) VALUES (?, CAST(GETDATE() AS DATE), ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                skill_id,
                summary_to_score,
                best_source_name,
                is_credible,
                metrics["ai_correlation_score"],
                metrics["ai_sim"],
                metrics["infra_sim"],
                metrics["lang_sim"],
                1 if is_override else 0,
            ),
        )

        cursor.execute(
            "UPDATE HITL_Validation_Queue SET is_approved = ? WHERE QueueID = ?",
            (APPROVED, queue_id),
        )

        logger.info(
            "%s %r (ai=%+.4f infra=%+.4f lang=%+.4f, override=%s).",
            "Auto-approved and scored" if auto_approved else "Approved and scored",
            skill_name,
            metrics["ai_sim"],
            metrics["infra_sim"],
            metrics["lang_sim"],
            is_override,
        )
        return True


def approve_and_score_skill(queue_id: int, override_summary: Optional[str] = None) -> bool:
    """Human approval path, called from the review form."""
    return promote_queue_item(queue_id, override_summary=override_summary, auto_approved=False)


def auto_approve_and_score_skill(queue_id: int) -> bool:
    """Automatic approval path for items that cleared the relevance and credibility gates."""
    return promote_queue_item(queue_id, override_summary=None, auto_approved=True)


# Fixed reasons a viewer can pick when reporting a skill. Validated server side so the
# stored value is always one of these, never whatever was posted.
REPORT_REASONS = {
    "wrong_page": "The reference page is not about this skill",
    "wrong_definition": "The definition text is wrong or misleading",
    "outdated": "The page or definition is out of date",
    "other": "Something else",
}

REPORT_GATE_REASON = "reported_by_viewer"


def _latest_skill_evidence(cursor, skill_name: str) -> dict:
    """
    Reads the current summary, title and score for an already-approved skill.

    The summary comes from the newest Skills_Historical_Metrics row and the title from the
    newest APPROVED queue row, which is exactly the pairing dashboard_data.EXPORT_SQL
    renders. Reading them the same way here is what keeps a report or an edit describing
    the same thing the viewer was looking at.
    """
    cursor.execute(
        """
        SELECT TOP 1 m.summary_text, m.best_source_name, m.metric_id
        FROM Skills_Historical_Metrics AS m
        JOIN Skills_Master AS s ON s.skill_id = m.skill_id
        WHERE s.skill_name = ?
        ORDER BY m.snapshot_date DESC, m.metric_id DESC
        """,
        (skill_name,),
    )
    metric = cursor.fetchone()

    cursor.execute(
        f"""
        SELECT TOP 1 QueueID, wiki_title, wiki_score, onet_code, onet_title, category
        FROM HITL_Validation_Queue
        WHERE skill_name = ? AND is_approved = {APPROVED} AND wiki_title IS NOT NULL
        ORDER BY created_at DESC, QueueID DESC
        """,
        (skill_name,),
    )
    queued = cursor.fetchone()

    return {
        "summary": metric[0] if metric else None,
        "best_source_name": metric[1] if metric else None,
        "metric_id": metric[2] if metric else None,
        "queue_id": queued[0] if queued else None,
        "wiki_title": queued[1] if queued else None,
        "wiki_score": queued[2] if queued else None,
        "onet_code": queued[3] if queued else None,
        "onet_title": queued[4] if queued else None,
        "category": queued[5] if queued else None,
    }


def report_skill(skill_id: int, reason: str, note: str = "") -> Tuple[bool, str]:
    """
    Files a viewer's report that a skill's page or definition is wrong.

    Returns (ok, code). Codes: "ok", "already_pending", "not_found", "no_evidence",
    "bad_reason", "error".

    The report becomes a PENDING queue row carrying the skill's CURRENT summary and title,
    so it lands in the same HITL queue as everything else and the tools already there --
    the remediation field, the editable textarea, Approve, Reject -- are exactly what
    fixing it requires. Copying the current summary in is not cosmetic: a pending row with
    no summary text is unapprovable, and `_has_pending_review` deliberately ignores such
    rows, so a summary-less report would neither block re-ingestion nor be actionable.

    The skill's score and its place on the dashboard are NOT touched. A single unverified
    click by any viewer must not be able to remove data from the charts or change the
    counts; the report is a request for review, not a verdict.

    If the skill already has a pending row -- from the gate or an earlier report -- the
    note is appended to that row instead of inserting a second one. Duplicate pending
    rows for one skill are the exact failure mode that produced the 388-row backlog.
    """
    if reason not in REPORT_REASONS:
        logger.warning("Rejected report for skill #%s: unknown reason %r.", skill_id, reason)
        return False, "bad_reason"

    stamp = f"[{reason}] {note.strip()}" if note and note.strip() else f"[{reason}]"

    try:
        with db_cursor() as cursor:
            cursor.execute(
                "SELECT skill_name, category FROM Skills_Master WHERE skill_id = ?",
                (skill_id,),
            )
            row = cursor.fetchone()
            if not row:
                return False, "not_found"
            skill_name, category = row

            # An existing pending row is updated rather than duplicated.
            cursor.execute(
                f"""
                SELECT TOP 1 QueueID, report_note FROM HITL_Validation_Queue
                WHERE skill_name = ? AND is_approved = {PENDING}
                ORDER BY created_at DESC, QueueID DESC
                """,
                (skill_name,),
            )
            pending = cursor.fetchone()
            if pending:
                queue_id, existing_note = pending
                merged = f"{existing_note}\n{stamp}" if existing_note else stamp
                cursor.execute(
                    "UPDATE HITL_Validation_Queue SET report_note = ? WHERE QueueID = ?",
                    (merged[:500], queue_id),
                )
                logger.info(
                    "Report on %r appended to existing pending queue item #%s.",
                    skill_name, queue_id,
                )
                return True, "already_pending"

            evidence = _latest_skill_evidence(cursor, skill_name)
            if not evidence["summary"]:
                # Nothing to put in the row, so it would be unapprovable. Refuse loudly
                # rather than filing a row a human cannot act on.
                logger.error(
                    "Cannot file report for %r: no summary text on its latest snapshot.",
                    skill_name,
                )
                return False, "no_evidence"

            cursor.execute(
                f"""
                INSERT INTO HITL_Validation_Queue (
                    skill_name, category, wiki_title, wiki_summary, wiki_score,
                    best_source_name, is_credible, gate_reason, report_note,
                    onet_code, onet_title, is_approved
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, {PENDING})
                """,
                (
                    skill_name,
                    evidence["category"] or category,
                    evidence["wiki_title"],
                    evidence["summary"],
                    evidence["wiki_score"],
                    evidence["best_source_name"],
                    REPORT_GATE_REASON,
                    stamp[:500],
                    evidence["onet_code"],
                    evidence["onet_title"],
                ),
            )

        logger.warning(
            "Skill %r reported by a viewer (%s). Queued for review; its score is unchanged.",
            skill_name, reason,
        )
        return True, "ok"

    except Exception:
        logger.exception("Could not file report for skill #%s.", skill_id)
        return False, "error"


def edit_approved_skill(
    skill_id: int, new_summary: str, new_reference: str = ""
) -> Tuple[bool, str]:
    """
    Post-mortem correction of an already-approved skill's definition and reference page.

    Returns (ok, code). Codes: "ok", "not_found", "not_approved", "no_summary",
    "no_snapshot", "bad_reference", "error".

    WHAT YOU SEE IS WHAT IS SAVED. The summary box is authoritative for the text and the
    reference box only records the page it came from -- supplying a reference does NOT
    refetch and overwrite the text. That is the opposite of the pending-item remediation
    field, and deliberately so: here the editor has already typed the wording they want,
    and silently replacing it with a fresh extract would discard their work. The reference
    is still validated as a real Wikipedia page, so the stored link cannot be a dead one.

    The LATEST metrics row is corrected in place with is_override = 1, rather than a new
    row being appended. A manual fix is a correction to an existing observation, not a new
    observation: two rows on the same snapshot_date would make the trend chart draw a
    vertical jump that reads as real drift in the skill.

    Open report rows are left pending on purpose. Closing them here would mean deciding on
    someone else's behalf that their report is resolved; the caller is told how many are
    open so it can say so.
    """
    summary = (new_summary or "").strip()
    if not summary:
        return False, "no_summary"

    reference = (new_reference or "").strip()
    resolved_title = None
    if reference:
        # Validated but NOT fetched. See the docstring: the text is the editor's.
        resolved_title, error = validate_wikipedia_reference(reference)
        if error:
            logger.info("Rejected post-mortem reference for skill #%s: %s.", skill_id, error)
            return False, "bad_reference"

    try:
        with db_cursor() as cursor:
            cursor.execute(
                "SELECT skill_name, category, is_approved FROM Skills_Master WHERE skill_id = ?",
                (skill_id,),
            )
            row = cursor.fetchone()
            if not row:
                return False, "not_found"
            skill_name, category, is_approved = row
            if not is_approved:
                return False, "not_approved"

            evidence = _latest_skill_evidence(cursor, skill_name)
            if not evidence["metric_id"]:
                # Nothing to correct. Writing a first snapshot here would bypass the
                # promotion path that is meant to be the only writer of new metrics rows.
                return False, "no_snapshot"

            metrics = calculate_ai_correlation(skill_name, category, summary)

            cursor.execute(
                """
                UPDATE Skills_Historical_Metrics
                SET summary_text = ?,
                    ai_correlation_score = ?,
                    ai_sim = ?,
                    infra_sim = ?,
                    lang_sim = ?,
                    is_override = 1,
                    last_modified = GETDATE()
                WHERE metric_id = ?
                """,
                (
                    summary,
                    metrics["ai_correlation_score"],
                    metrics["ai_sim"],
                    metrics["infra_sim"],
                    metrics["lang_sim"],
                    evidence["metric_id"],
                ),
            )

            # Keep the approved queue row in step. dashboard_data reads resolved_title
            # from it, so leaving it stale would show the old page beside the new text.
            if evidence["queue_id"]:
                if resolved_title:
                    cursor.execute(
                        """
                        UPDATE HITL_Validation_Queue
                        SET wiki_title = ?, wiki_summary = ?
                        WHERE QueueID = ?
                        """,
                        (resolved_title, summary, evidence["queue_id"]),
                    )
                else:
                    cursor.execute(
                        "UPDATE HITL_Validation_Queue SET wiki_summary = ? WHERE QueueID = ?",
                        (summary, evidence["queue_id"]),
                    )

            cursor.execute(
                f"""
                SELECT COUNT(*) FROM HITL_Validation_Queue
                WHERE skill_name = ? AND is_approved = {PENDING}
                  AND gate_reason = '{REPORT_GATE_REASON}'
                """,
                (skill_name,),
            )
            open_reports = int(cursor.fetchone()[0])

        logger.warning(
            "Post-mortem edit of %r: metric #%s corrected in place (ai=%+.4f, was rescored "
            "from edited text). Reference %s. %d open report(s) left pending.",
            skill_name, evidence["metric_id"], metrics["ai_sim"],
            resolved_title or "unchanged", open_reports,
        )
        return True, "ok"

    except Exception:
        logger.exception("Post-mortem edit failed for skill #%s.", skill_id)
        return False, "error"


def remediate_queue_item(queue_id: int, raw_reference: str) -> Tuple[bool, str]:
    """
    Repoints a pending queue item at a reference page the reviewer supplied.

    Returns (ok, code). On success the code is "ok"; on failure it is one of
    scraping.REMEDIATION_ERROR_CODES, "not_pending", "not_found", or "error", all of
    which main.py maps to a sentence for the reviewer.

    The item stays PENDING. This only replaces the evidence on the card -- title,
    summary text and cross-encoder score -- so the reviewer can read the right page and
    then approve or reject it. Nothing is scored here; promotion remains the one path
    that writes Skills_Historical_Metrics.

    No credibility audit is run. The reviewer is personally vouching for the page, and
    spending a Gemini call to second-guess them would burn quota on the one case where
    a human already has better information than the model.

    is_credible is reset to NULL rather than being carried forward. The stored verdict
    was about the text this update is discarding, and leaving it in place would attribute
    a credibility judgement to text the auditor never saw.

    The network calls happen OUTSIDE any transaction, and the pending check is repeated
    as a predicate on the UPDATE itself. That keeps a multi-second Wikipedia round trip
    from holding a write transaction open, while still making "only pending rows are
    rewritten" atomic rather than a check that could go stale mid-fetch.
    """
    try:
        with db_cursor(autocommit=True) as cursor:
            cursor.execute(
                """
                SELECT skill_name, category, is_approved
                FROM HITL_Validation_Queue
                WHERE QueueID = ?
                """,
                (queue_id,),
            )
            row = cursor.fetchone()

        if not row:
            logger.error("Queue item #%s not found; nothing remediated.", queue_id)
            return False, "not_found"

        skill_name, category, is_approved = row
        if is_approved != PENDING:
            logger.warning(
                "Refusing to remediate queue item #%s (%r): status is %s, not pending.",
                queue_id, skill_name, is_approved,
            )
            return False, "not_pending"

        resolved = resolve_reviewer_url(skill_name, category or "", raw_reference)
        if resolved.get("error"):
            logger.info(
                "Reviewer remediation for queue item #%s (%r) rejected: %s.",
                queue_id, skill_name, resolved["error"],
            )
            return False, str(resolved["error"])

        with db_cursor() as cursor:
            cursor.execute(
                """
                UPDATE HITL_Validation_Queue
                SET wiki_title = ?,
                    wiki_summary = ?,
                    wiki_score = ?,
                    best_source_name = ?,
                    is_credible = NULL,
                    gate_reason = 'reviewer_remediated'
                WHERE QueueID = ? AND is_approved = ?
                """,
                (
                    resolved["title"],
                    resolved["extract"],
                    resolved["score"],
                    resolved["source_name"],
                    queue_id,
                    PENDING,
                ),
            )
            if cursor.rowcount == 0:
                # Decided by someone else while the fetch was in flight.
                logger.warning(
                    "Queue item #%s stopped being pending during remediation; not rewritten.",
                    queue_id,
                )
                return False, "not_pending"

        logger.info(
            "Remediated queue item #%s (%r) to page %r.",
            queue_id, skill_name, resolved["title"],
        )
        return True, "ok"

    except Exception:
        logger.exception("Remediation failed for queue item #%s.", queue_id)
        return False, "error"


def reject_skill(queue_id: int) -> bool:
    """
    Marks a queue item rejected.

    No metrics row exists to clean up, because nothing is scored before promotion.
    """
    with db_cursor() as cursor:
        cursor.execute(
            "UPDATE HITL_Validation_Queue SET is_approved = ? WHERE QueueID = ?",
            (REJECTED, queue_id),
        )
        if cursor.rowcount == 0:
            logger.error("Queue item #%s not found; nothing rejected.", queue_id)
            return False
        logger.info("Queue item #%s marked rejected.", queue_id)
        return True

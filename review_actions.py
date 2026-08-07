"""
Human-in-the-loop write actions against the JSON store.

The JSON equivalent of what score.py did for SQL, and the ONLY module that changes a
skill's review status. Ingestion writes discoveries; this writes decisions.

Every function returns (ok, code) where code is drawn from a small fixed set that
main.py maps to reviewer-facing sentences. Returning a code rather than raising keeps
the failure modes enumerable: a route can render every one of them without a try block
around the whole request.

Two rules hold throughout:

  1. Reload before writing. Never write back a payload the caller has been holding
     while a human read the page, or anything changed since the render is silently
     discarded. json_store.atomic_write then makes the write itself all-or-nothing.
  2. Only approved skills are scored. A snapshot is written exactly when a skill
     reaches approved status, so nothing unreviewed can reach the dashboard.
"""

import logging
from typing import Tuple

import json_store
from json_store import (
    STATUS_APPROVED,
    STATUS_PENDING,
    STATUS_REJECTED,
    load_master,
    load_timeseries,
    save_master,
    save_timeseries,
)

logger = logging.getLogger(__name__)

# Fixed reasons a viewer can pick when reporting a skill from the dashboard.
REPORT_REASONS = {
    "wrong_page": "The reference page is not about this skill",
    "wrong_definition": "The definition text is wrong or misleading",
    "outdated": "The page or definition is out of date",
    "other": "Something else",
}

REPORT_GATE_REASON = "reported_by_viewer"

# Set when an automated second pass repoints a card at a page it proposed itself.
# Deliberately distinct from reviewer_remediated: both mean "the cross-encoder score
# below is recorded, not enforced", but only one of them has a human behind it, and the
# card must not imply otherwise.
SECOND_PASS_GATE_REASON = "machine_remediated"

# Recorded in best_source_name when a definition was written by the model because no
# encyclopedia article covers the product. Owned here, where the writes happen, and
# imported by second_pass, so the provenance marker cannot come out two ways.
MODEL_AUTHORED_SOURCE_NAME = "Model-authored, no encyclopedic source"


def _load() -> Tuple[dict, list]:
    return load_master(), load_timeseries()


def approve_skill(skill_name: str, summary: str, reference: str = "") -> Tuple[bool, str]:
    """
    Approves a pending skill, scoring the (possibly edited) definition.

    The text in the box is what gets scored, so a reviewer's correction is the thing
    measured rather than the text the scraper originally pulled.

    Codes: ok, not_found, no_summary, error.
    """
    text = (summary or "").strip()
    if not text:
        return False, "no_summary"

    try:
        # Imported here, not at module scope: definitions_algorithm imports scraping,
        # which loads the cross-encoder ONNX session. main.py imports this module at
        # startup and should not pay for that until a review action actually happens.
        from definitions_algorithm import record_snapshot

        master, timeseries = _load()
        entry = master.get(skill_name)
        if entry is None:
            return False, "not_found"

        entry["wikipedia_summary"] = text
        if reference and reference.strip():
            entry["resolved_title"] = reference.strip()
        entry["last_updated"] = json_store.today().isoformat()
        entry["status"] = STATUS_APPROVED
        # The report is resolved by the act of deciding on it.
        entry["report_note"] = None

        metrics = record_snapshot(entry, timeseries)

        save_timeseries(timeseries)
        save_master(master)
        logger.info(
            "Approved %r: ai=%+.4f (%s).",
            skill_name, metrics["ai_score"], metrics["category_bucket"],
        )
        return True, "ok"
    except Exception:
        logger.exception("Approval failed for %r.", skill_name)
        return False, "error"


def reject_skill(skill_name: str) -> Tuple[bool, str]:
    """
    Rejects a skill. No snapshot is written, and any existing one is left alone.

    Leaving an old snapshot in place is deliberate: rejecting a re-reviewed skill
    should not silently erase the history of what it scored before. The dashboard
    filters on status, so a rejected skill disappears from view either way.

    Codes: ok, not_found, error.
    """
    try:
        master = load_master()
        entry = master.get(skill_name)
        if entry is None:
            return False, "not_found"

        entry["status"] = STATUS_REJECTED
        entry["report_note"] = None
        save_master(master)
        logger.info("Rejected %r.", skill_name)
        return True, "ok"
    except Exception:
        logger.exception("Rejection failed for %r.", skill_name)
        return False, "error"


def remediate_skill(skill_name: str, raw_reference: str) -> Tuple[bool, str]:
    """
    Repoints a pending skill at a reference page the reviewer supplied.

    The item stays PENDING. This replaces the evidence on the card -- title, text and
    cross-encoder score -- so the reviewer can read the right page and then decide.

    The score is RECORDED, NOT ENFORCED. A reviewer-supplied page that scores 0.0 is
    still accepted, because a human asserting the page is exactly the thing the
    cross-encoder exists to guess at. Re-applying the 0.90 gate here would reject the
    hard-mapped aliases this field was added for: PySpark against the Apache Spark
    article measures 0.0 and is nonetheless correct.

    No Gemini call. The reviewer is vouching for the page; spending audit quota to
    second-guess them contradicts both the quota rule and the point of the field.

    Codes: ok, not_found, not_pending, plus scraping's own error codes.
    """
    try:
        from scraping import resolve_reviewer_url

        master = load_master()
        entry = master.get(skill_name)
        if entry is None:
            return False, "not_found"
        if entry.get("status") != STATUS_PENDING:
            return False, "not_pending"

        resolved = resolve_reviewer_url(
            skill_name, entry.get("category", ""), raw_reference
        )
        if resolved.get("error"):
            logger.info("Remediation of %r rejected: %s.", skill_name, resolved["error"])
            return False, str(resolved["error"])

        entry["resolved_title"] = resolved["title"]
        entry["wikipedia_summary"] = resolved["extract"]
        entry["cross_score"] = resolved["score"]
        # The audit has not run against THIS text. Carrying the old verdict forward
        # would attribute a credibility judgement to text it never saw.
        entry["is_credible"] = None
        entry["gate_reason"] = "reviewer_remediated"
        entry["last_updated"] = json_store.today().isoformat()

        save_master(master)
        logger.info("Remediated %r to page %r.", skill_name, resolved["title"])
        return True, "ok"
    except Exception:
        logger.exception("Remediation failed for %r.", skill_name)
        return False, "error"


def _second_pass_block(entry: dict, result: dict) -> dict:
    """Builds the audit trail recorded on every second-passed entry."""
    prior = entry.get("second_pass")
    attempts = int(prior.get("attempts") or 0) if isinstance(prior, dict) else 0

    return {
        "version": 1,
        "attempted_on": json_store.today().isoformat(),
        "model": "gemini-flash-lite-latest",
        "outcome": result.get("outcome"),
        "proposed_title": result.get("proposed_title"),
        "prior_title": result.get("prior_title") or entry.get("resolved_title"),
        "measured_score": result.get("measured_score"),
        "confidence": result.get("confidence"),
        "rationale": result.get("rationale"),
        "detail": result.get("detail"),
        "suggested_action": result.get("suggested_action"),
        # The ONLY text this block ever stores, and it has to be stored. A page
        # suggestion is re-fetchable from its title, so it is refetched; a written
        # definition is not, and re-deriving it would mean paying Gemini a second time
        # for the same sentence the reviewer is looking at.
        "drafted_definition": result.get("drafted_definition"),
        "attempts": attempts + 1,
    }


def _write_model_authored(entry: dict, definition: str, source_name: str) -> None:
    """
    Puts a sourceless definition on an entry.

    Shared by the second pass writing one directly and by a reviewer accepting one that
    was proposed, so the two cannot drift into recording provenance differently. The
    entry stays PENDING and is_credible stays None in both: there is no source here for
    the audit layer to have judged, which is the whole point of marking it.
    """
    entry["wikipedia_summary"] = definition
    entry["best_source_name"] = source_name
    entry["resolved_title"] = None
    entry["is_credible"] = None
    entry["last_updated"] = json_store.today().isoformat()


def apply_second_pass_result(skill_name: str, result: dict) -> Tuple[bool, str]:
    """
    Records one second-pass outcome. The ONLY function that writes second_pass state.

    Most outcomes write nothing but the audit block: a finding about a card is not a
    change to it. Two outcomes do more, and both stay PENDING:

      applied         repoints the card at a page the model proposed and a separate
                      audit call graded. is_credible is reset to None, because the
                      passing audit graded the NEW text while the 0.90 corroboration
                      the store's True normally implies is absent. Same invariant as
                      remediate_skill: never carry a verdict onto text it never saw.
      model_authored  writes a definition with no source behind it, and says so in
                      best_source_name. Never approves, by construction -- there is
                      nothing here for the audit layer to have corroborated.

    auto_approved is the only outcome that changes status, and only on the unchanged
    bar the first-pass auto-approval already uses: a new page over the cross-encoder
    threshold AND a passing credibility audit.

    Codes: ok, not_found, not_pending, bad_result, error.
    """
    outcome = (result or {}).get("outcome")
    if not outcome:
        return False, "bad_result"

    try:
        from definitions_algorithm import record_snapshot

        master, timeseries = _load()
        entry = master.get(skill_name)
        if entry is None:
            return False, "not_found"
        if entry.get("status") != STATUS_PENDING:
            # The second pass may only ever settle a card that is already in the queue.
            # Refusing here is what makes it structurally incapable of enqueueing work
            # or of overwriting a decision a human made while the run was in flight.
            return False, "not_pending"

        entry["second_pass"] = _second_pass_block(entry, result)

        if outcome == "model_authored":
            _write_model_authored(entry, result["summary"], result.get("source_name"))
            save_master(master)
            logger.warning(
                "Second pass wrote a MODEL-AUTHORED definition for %r. It has no "
                "external source and cannot auto-approve.",
                skill_name,
            )
            return True, "ok"

        if outcome in ("applied", "auto_approved"):
            entry["resolved_title"] = result["proposed_title"]
            entry["cross_score"] = result.get("measured_score")
            entry["best_source_name"] = result.get("source_name")
            entry["gate_reason"] = SECOND_PASS_GATE_REASON
            entry["last_updated"] = json_store.today().isoformat()
            entry["report_note"] = None

        if outcome == "applied":
            entry["wikipedia_summary"] = result.get("summary") or result.get("extract") or ""
            entry["is_credible"] = None
            save_master(master)
            logger.info(
                "Second pass repointed %r from %r to %r (cross=%s). Still pending.",
                skill_name, result.get("prior_title"), result["proposed_title"],
                result.get("measured_score"),
            )
            return True, "ok"

        if outcome == "auto_approved":
            entry["wikipedia_summary"] = result["summary"]
            entry["is_credible"] = True
            entry["status"] = STATUS_APPROVED

            metrics = record_snapshot(entry, timeseries)

            save_timeseries(timeseries)
            save_master(master)
            # WARNING, not INFO: this is the one path where a machine puts a row on the
            # dashboard, and it must be greppable after the fact.
            logger.warning(
                "Second pass AUTO-APPROVED %r: %r -> %r, cross=%.4f, ai=%+.4f (%s).",
                skill_name, result.get("prior_title"), result["proposed_title"],
                result.get("measured_score") or 0.0,
                metrics["ai_score"], metrics["category_bucket"],
            )
            return True, "ok"

        # Every other outcome is a finding, not a change.
        save_master(master)
        return True, "ok"

    except Exception:
        logger.exception("Could not record second-pass result for %r.", skill_name)
        return False, "error"


def apply_machine_suggestion(skill_name: str) -> Tuple[bool, str]:
    """
    Accepts a stored second-pass suggestion on the reviewer's behalf, in one click.

    remediate_skill with the title read from the stored suggestion instead of a form
    field, and the same consequences: the item stays PENDING, the score is recorded but
    not enforced, and is_credible resets because no audit has run against the text this
    fetch returns.

    The page is REFETCHED rather than replaying text stored at proposal time. That is
    why the second_pass block holds no extract -- the master file stays small, there is
    no stale-text failure mode, and what a reviewer reads is always current.

    Codes: ok, not_found, not_pending, no_suggestion, plus scraping's own error codes.
    """
    try:
        from scraping import resolve_reviewer_url

        master = load_master()
        entry = master.get(skill_name)
        if entry is None:
            return False, "not_found"
        if entry.get("status") != STATUS_PENDING:
            return False, "not_pending"

        block = entry.get("second_pass")
        proposed = (block or {}).get("proposed_title") if isinstance(block, dict) else None
        if not proposed:
            return False, "no_suggestion"

        resolved = resolve_reviewer_url(skill_name, entry.get("category", ""), proposed)
        if resolved.get("error"):
            logger.info(
                "Suggested page %r for %r did not resolve: %s.",
                proposed, skill_name, resolved["error"],
            )
            return False, str(resolved["error"])

        entry["resolved_title"] = resolved["title"]
        entry["wikipedia_summary"] = resolved["extract"]
        entry["cross_score"] = resolved["score"]
        entry["is_credible"] = None
        entry["gate_reason"] = SECOND_PASS_GATE_REASON
        entry["last_updated"] = json_store.today().isoformat()

        save_master(master)
        logger.info("Applied machine suggestion for %r: page %r.", skill_name, resolved["title"])
        return True, "ok"
    except Exception:
        logger.exception("Could not apply machine suggestion for %r.", skill_name)
        return False, "error"


def apply_machine_draft(skill_name: str) -> Tuple[bool, str]:
    """
    Accepts a stored model-written definition, in one click.

    The counterpart to apply_machine_suggestion for products that have no encyclopedia
    article at all. Unlike that function there is nothing to refetch, so the text comes
    from the stored proposal -- which is why the second_pass block carries it.

    No Gemini call. The sentence was already written and paid for; asking again would
    spend quota to produce a different wording of the same claim.

    The item stays PENDING and keeps its no-source marker. Accepting a draft is not a
    judgement that it is right, only a decision to put it on the card and read it.

    Codes: ok, not_found, not_pending, no_draft, error.
    """
    try:
        master = load_master()
        entry = master.get(skill_name)
        if entry is None:
            return False, "not_found"
        if entry.get("status") != STATUS_PENDING:
            return False, "not_pending"

        block = entry.get("second_pass")
        draft = (block or {}).get("drafted_definition") if isinstance(block, dict) else None
        if not (draft or "").strip():
            return False, "no_draft"

        _write_model_authored(
            entry, draft.strip(), MODEL_AUTHORED_SOURCE_NAME
        )
        # The proposal has become the card, so record that rather than leaving an
        # outcome that still reads as "not yet applied".
        block["outcome"] = "model_authored"
        entry["second_pass"] = block

        save_master(master)
        logger.warning(
            "Reviewer accepted a MODEL-AUTHORED definition for %r. It has no external "
            "source and was never credibility-audited.",
            skill_name,
        )
        return True, "ok"
    except Exception:
        logger.exception("Could not apply machine draft for %r.", skill_name)
        return False, "error"


def report_skill(skill_name: str, reason: str, note: str = "") -> Tuple[bool, str]:
    """
    Files a viewer's report that an approved skill's page or definition is wrong.

    The skill KEEPS its snapshot and stays on the dashboard. A single unverified click
    by any viewer must not remove data from the charts; the report is a request for
    review, not a verdict. Status moves to pending so it surfaces in the queue, and the
    existing snapshot means the dashboard row survives until someone decides.

    A second report on the same skill appends to the note rather than overwriting it.

    Codes: ok, already_pending, not_found, bad_reason, no_evidence, error.
    """
    if reason not in REPORT_REASONS:
        return False, "bad_reason"

    stamp = f"[{reason}] {note.strip()}" if note and note.strip() else f"[{reason}]"

    try:
        master = load_master()
        entry = master.get(skill_name)
        if entry is None:
            return False, "not_found"
        if not (entry.get("wikipedia_summary") or "").strip():
            # Nothing to review. Queueing it would produce a card a human cannot act
            # on, which is the invariant the old 388-row backlog violated.
            return False, "no_evidence"

        already_pending = entry.get("status") == STATUS_PENDING
        existing = entry.get("report_note")
        entry["report_note"] = (f"{existing}\n{stamp}" if existing else stamp)[:500]
        entry["status"] = STATUS_PENDING
        entry["gate_reason"] = REPORT_GATE_REASON

        save_master(master)
        logger.warning(
            "%r reported by a viewer (%s). Queued for review; its score is unchanged.",
            skill_name, reason,
        )
        return True, "already_pending" if already_pending else "ok"
    except Exception:
        logger.exception("Could not file report for %r.", skill_name)
        return False, "error"


def edit_approved_skill(
    skill_name: str, summary: str, reference: str = ""
) -> Tuple[bool, str]:
    """
    Post-mortem correction of an already-approved skill.

    WHAT YOU SEE IS WHAT IS SAVED. The summary box is authoritative and the reference
    box only records the page -- supplying a reference does NOT refetch and overwrite
    the text. That is the opposite of remediate_skill, and deliberately so: here the
    editor has already typed the wording they want.

    The CURRENT quarter's snapshot is corrected in place by upsert_snapshot rather than
    a second point being appended, so the trend chart shows no false movement.

    Codes: ok, not_found, not_approved, no_summary, bad_reference, error.
    """
    text = (summary or "").strip()
    if not text:
        return False, "no_summary"

    try:
        from definitions_algorithm import record_snapshot
        from scraping import validate_wikipedia_reference

        resolved_title = None
        if reference and reference.strip():
            # Validated but NOT fetched.
            resolved_title, error = validate_wikipedia_reference(reference.strip())
            if error:
                return False, "bad_reference"

        master, timeseries = _load()
        entry = master.get(skill_name)
        if entry is None:
            return False, "not_found"
        if entry.get("status") != STATUS_APPROVED:
            return False, "not_approved"

        entry["wikipedia_summary"] = text
        if resolved_title:
            entry["resolved_title"] = resolved_title
        entry["last_updated"] = json_store.today().isoformat()

        metrics = record_snapshot(entry, timeseries)

        save_timeseries(timeseries)
        save_master(master)
        logger.warning(
            "Post-mortem edit of %r: rescored in place to %+.4f (%s).",
            skill_name, metrics["ai_score"], metrics["category_bucket"],
        )
        return True, "ok"
    except Exception:
        logger.exception("Post-mortem edit failed for %r.", skill_name)
        return False, "error"

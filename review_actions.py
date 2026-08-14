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

Every write is split in two: an `_apply_*(master, timeseries, ...)` function that mutates
in memory and returns (ok, code), and a thin public wrapper that loads, calls it and
saves. The split exists so apply_review_batch can run a whole page of reviewer decisions
against ONE load and ONE save instead of rewriting the 2 MB store per click. Rule 1 is
unaffected: the load still happens at commit time, never at render time, which is the
staleness the rule is about.
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

# Re-exported from json_store, where it sits beside the statuses it qualifies. Imported
# rather than redefined because build_dashboard_rows now READS it to decide whether a
# pending skill is still published, and two spellings of it would silently unpublish
# every reported skill.
REPORT_GATE_REASON = json_store.REPORT_GATE_REASON

# Set when an automated second pass repoints a card at a page it proposed itself.
# Deliberately distinct from reviewer_remediated: both mean "the cross-encoder score
# below is recorded, not enforced", but only one of them has a human behind it, and the
# card must not imply otherwise.
SECOND_PASS_GATE_REASON = "machine_remediated"

# Recorded in best_source_name when a definition was written by the model because no
# encyclopedia article covers the product. Owned here, where the writes happen, and
# imported by second_pass, so the provenance marker cannot come out two ways.
MODEL_AUTHORED_SOURCE_NAME = "Model-authored, no encyclopedic source"

# Set when a written definition reached the dashboard on two agreeing calls rather than a
# reviewer's click. Distinct from the reviewer-accepted case ON PURPOSE: both put text
# with no external source on the dashboard, and only one of them had a person read it.
# Keeping them apart is what makes "how much unsourced text is live, and who let it
# through" a question the store can answer.
MODEL_AUTHORED_VERIFIED_GATE_REASON = "model_authored_verified"


def _load() -> Tuple[dict, list]:
    return load_master(), load_timeseries()


def _apply_approve(
    master: dict, timeseries: list, skill_name: str, summary: str, reference: str = ""
) -> Tuple[bool, str]:
    """Approves and scores, in memory. Codes: ok, not_found, no_summary."""
    text = (summary or "").strip()
    if not text:
        return False, "no_summary"

    # Imported here, not at module scope: definitions_algorithm imports scraping, which
    # loads the cross-encoder ONNX session. main.py imports this module at startup and
    # should not pay for that until a review action actually happens.
    from definitions_algorithm import record_snapshot

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
    logger.info(
        "Approved %r: ai=%+.4f (%s).",
        skill_name, metrics["ai_score"], metrics["category_bucket"],
    )
    return True, "ok"


def approve_skill(skill_name: str, summary: str, reference: str = "") -> Tuple[bool, str]:
    """
    Approves a pending skill, scoring the (possibly edited) definition.

    The text in the box is what gets scored, so a reviewer's correction is the thing
    measured rather than the text the scraper originally pulled.

    Codes: ok, not_found, no_summary, error.
    """
    try:
        master, timeseries = _load()
        ok, code = _apply_approve(master, timeseries, skill_name, summary, reference)
        if not ok:
            return ok, code

        save_timeseries(timeseries)
        save_master(master)
        return True, "ok"
    except Exception:
        logger.exception("Approval failed for %r.", skill_name)
        return False, "error"


def _apply_reject(master: dict, skill_name: str) -> Tuple[bool, str]:
    """Rejects, in memory. Codes: ok, not_found."""
    entry = master.get(skill_name)
    if entry is None:
        return False, "not_found"

    entry["status"] = STATUS_REJECTED
    entry["report_note"] = None
    logger.info("Rejected %r.", skill_name)
    return True, "ok"


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
        ok, code = _apply_reject(master, skill_name)
        if not ok:
            return ok, code

        save_master(master)
        return True, "ok"
    except Exception:
        logger.exception("Rejection failed for %r.", skill_name)
        return False, "error"


def _merge_occupations(survivor: dict, duplicate: dict) -> int:
    """
    Folds the duplicate's occupation mappings into the survivor. Returns how many are new.

    This is the whole point of merging rather than rejecting. O*NET lists the same
    product under different spellings against DIFFERENT occupations -- "MicroSurvey
    Software MicroSurvey CAD" is recorded against 17-1022.01 while "MicroSurveyCAD" is
    recorded against 17-1022.00 and 17-3031.00. Rejecting the duplicate outright would
    silently drop whichever occupations only it carried, and the survivor would then
    understate how widely the product is used.

    Hot-tech status is OR'd per occupation, never overwritten. A skill hot for one
    occupation and not another is the normal case the by-job view depends on, so the
    merged record has to keep each occupation's own flag rather than flatten them.
    """
    merged = {
        occupation.get("onet_code"): dict(occupation)
        for occupation in (survivor.get("occupations") or [])
        if occupation.get("onet_code")
    }

    added = 0
    for occupation in duplicate.get("occupations") or []:
        code = occupation.get("onet_code")
        if not code:
            continue
        existing = merged.get(code)
        if existing is None:
            merged[code] = dict(occupation)
            added += 1
            continue
        existing["is_hot_tech"] = bool(existing.get("is_hot_tech")) or bool(
            occupation.get("is_hot_tech")
        )
        if not existing.get("onet_title"):
            existing["onet_title"] = occupation.get("onet_title")

    survivor["occupations"] = [merged[code] for code in sorted(merged)]

    # The flat lists are derived from the same source and are what the by-job filter and
    # the snapshot records read, so they are rebuilt here rather than merged separately.
    # Deriving them keeps the three from disagreeing about which occupations exist.
    survivor["onet_codes"] = [code for code in sorted(merged)]
    survivor["onet_titles"] = [merged[code].get("onet_title") or code for code in sorted(merged)]
    survivor["is_hot_tech_anywhere"] = any(
        occupation.get("is_hot_tech") for occupation in survivor["occupations"]
    )
    return added


def _apply_mark_duplicate(master: dict, skill_name: str, target: str) -> Tuple[bool, str]:
    """
    Marks skill_name as a duplicate of target, in memory.

    The duplicate's occupation mappings move to the survivor, then the duplicate is
    retired with STATUS_REJECTED and a `duplicate_of` pointer. Rejected rather than a
    new status on purpose: every status check in the dashboard, the queue and the
    counters already handles the three in VALID_STATUSES, and a fourth would have to be
    taught to each of them one at a time. `duplicate_of` is what makes it auditable and
    reversible, and what distinguishes "this is not a real skill" from "this is the same
    real skill under another spelling".

    The SURVIVOR is not otherwise touched. It keeps its own status, definition and
    reference page, and if it is still pending it stays in the queue to be decided on
    its own merits.

    Codes: ok, not_found, unknown_target, duplicate_self, target_rejected.
    """
    entry = master.get(skill_name)
    if entry is None:
        return False, "not_found"

    target = (target or "").strip()
    survivor = master.get(target)
    if not target or survivor is None:
        return False, "unknown_target"
    if target == skill_name:
        return False, "duplicate_self"
    if survivor.get("status") == STATUS_REJECTED:
        # Merging into a record that has already been thrown out would move the
        # occupations somewhere nothing reads them, which loses them just as surely as
        # rejecting the duplicate outright would have.
        return False, "target_rejected"

    added = _merge_occupations(survivor, entry)

    entry["status"] = STATUS_REJECTED
    entry["duplicate_of"] = target
    entry["report_note"] = None

    logger.info(
        "Marked %r as a duplicate of %r; %d occupation mapping(s) moved across, "
        "survivor now covers %d.",
        skill_name, target, added, len(survivor.get("onet_codes") or []),
    )
    return True, "ok"


def mark_duplicate(skill_name: str, target: str) -> Tuple[bool, str]:
    """
    Marks one queued skill as a duplicate of another and merges their occupations.

    Codes: ok, not_found, unknown_target, duplicate_self, target_rejected, error.
    """
    try:
        master = load_master()
        ok, code = _apply_mark_duplicate(master, skill_name, target)
        if not ok:
            return ok, code

        save_master(master)
        return True, "ok"
    except Exception:
        logger.exception("Marking %r as a duplicate of %r failed.", skill_name, target)
        return False, "error"


def _apply_remediate(master: dict, skill_name: str, raw_reference: str) -> Tuple[bool, str]:
    """
    Repoints at a reviewer-supplied page, in memory. Fetches over the network.

    Routed through resolve_external_url, so the page can be anywhere on the web and not
    only on Wikipedia -- a discontinued vendor tool often has exactly one authoritative
    page and it is the vendor's. Wikipedia links are still handled by the Wikipedia
    path inside it.

    Codes: ok, not_found, not_pending, plus scraping's own error codes.
    """
    from scraping import resolve_external_url

    entry = master.get(skill_name)
    if entry is None:
        return False, "not_found"
    if entry.get("status") != STATUS_PENDING:
        return False, "not_pending"

    resolved = resolve_external_url(skill_name, entry.get("category", ""), raw_reference)
    if resolved.get("error"):
        logger.info("Remediation of %r rejected: %s.", skill_name, resolved["error"])
        return False, str(resolved["error"])

    entry["resolved_title"] = resolved["title"]
    # None for a Wikipedia page, whose link is built from the title. Assigned either way
    # so a card repointed from a vendor site back to Wikipedia does not keep the old URL
    # and render a link to a page it no longer cites.
    entry["reference_url"] = resolved.get("url")
    entry["best_source_name"] = resolved.get("source_name")
    entry["wikipedia_summary"] = resolved["extract"]
    entry["cross_score"] = resolved["score"]
    # The audit has not run against THIS text. Carrying the old verdict forward
    # would attribute a credibility judgement to text it never saw.
    entry["is_credible"] = None
    entry["gate_reason"] = "reviewer_remediated"
    entry["last_updated"] = json_store.today().isoformat()
    # The card has new evidence on it, so a previous "cannot determine" no longer
    # describes the entry.
    entry.pop("ai_status", None)
    entry.pop("ai_status_detail", None)

    logger.info("Remediated %r to page %r.", skill_name, resolved["title"])
    return True, "ok"


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
        master = load_master()
        ok, code = _apply_remediate(master, skill_name, raw_reference)
        if not ok:
            return ok, code

        save_master(master)
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


def _tag_ai_status(entry: dict, outcome: str) -> None:
    """
    Records WHY the second pass will skip this entry next time, visibly, or clears it.

    The tag is the only thing select_candidates reads, so this function is the whole
    stop condition. Three cases:

      - the outcome earns a tag: written, and the item is skipped from now on
      - the outcome is transient (a failed call, an unreachable audit): no tag, and the
        item is retried next run. Any tag already on it is REMOVED, because a fresh
        finding supersedes an older decision to stop.
      - the attempt cap has been reached: tagged AI_CANNOT_DETERMINE regardless, so an
        item that keeps failing still ends up visible rather than quietly dropping out
        of every future run.

    Kept out of _second_pass_block because that function builds an audit record and this
    one makes a decision; conflating them is how the old skip became invisible.
    """
    from second_pass import AI_CANNOT_DETERMINE, MAX_SECOND_PASS_ATTEMPTS, ai_status_for

    status = ai_status_for(outcome)
    attempts = int((entry.get("second_pass") or {}).get("attempts") or 0)

    if status is None and attempts >= MAX_SECOND_PASS_ATTEMPTS:
        status = AI_CANNOT_DETERMINE
        entry["ai_status_detail"] = f"no usable answer after {attempts} attempts"
    elif status is not None:
        entry["ai_status_detail"] = outcome

    if status is None:
        entry.pop("ai_status", None)
        entry.pop("ai_status_detail", None)
        return

    entry["ai_status"] = status
    if status == AI_CANNOT_DETERMINE:
        logger.info(
            "Tagged %r as AI cannot determine (%s). It will not be re-attempted until "
            "a human clears the tag.",
            entry.get("skill_name") or "?", entry.get("ai_status_detail"),
        )


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
    entry["reference_url"] = None
    entry["is_credible"] = None
    entry["last_updated"] = json_store.today().isoformat()


def onet_boilerplate(skill_name: str, category: str) -> str:
    """The sentence a skill carries when nothing better is known about it."""
    return (
        f"{skill_name} is a hot technology asset categorized under "
        f"{category or 'Technology'} in the O*NET framework."
    )


# Non-terminal on purpose. second_pass.TERMINAL_OUTCOMES is what makes a later run skip
# an item without spending a call, so an outcome absent from that set is exactly what
# sends a card back to be redrafted.
DRAFT_REJECTED_OUTCOME = "draft_rejected"


def _apply_reject_draft(master: dict, skill_name: str) -> Tuple[bool, str]:
    """
    Sends a model-written definition back to be redrafted, in memory.

    Written for the case where the DRAFTING RULES changed rather than where one draft
    came out badly: definitions written before DEFINITION_SPEC are around 150 characters
    and name a category, a vendor and an audience, where the spec now asks for 200-400
    covering the specific technical category, vendor lineage and concrete use cases.
    Every draft written under the old rules is thin, and rejecting one is how it gets
    asked again.

    Handles both states a draft can be in:
      - WAITING on the card (outcome "drafted"): the offer is withdrawn.
      - ALREADY ACCEPTED (best_source_name is the model-authored marker): the card's text
        is that draft, so it is reverted to the O*NET boilerplate. The original
        encyclopedia text is long gone by then -- accepting a draft overwrote it -- so
        the boilerplate is the honest resting state, and it is non-empty, which is what
        keeps the card in the queue and the entry selectable by the second pass.

    The rejected text is KEPT, moved to `rejected_definition`. The second_pass block is
    an audit trail of what was asked and answered, and a reviewer comparing the redraft
    against what it replaced is the main way anyone will judge whether the new spec
    actually helped.

    The entry stays PENDING either way, and its gate_reason is left alone: it is already
    one of second_pass.ELIGIBLE_GATE_REASONS, which is how it came to be drafted at all.

    Codes: ok, not_found, not_pending, no_draft.
    """
    entry = master.get(skill_name)
    if entry is None:
        return False, "not_found"
    if entry.get("status") != STATUS_PENDING:
        return False, "not_pending"

    block = entry.get("second_pass")
    if not isinstance(block, dict):
        return False, "no_draft"

    draft = (block.get("drafted_definition") or "").strip()
    was_applied = entry.get("best_source_name") == MODEL_AUTHORED_SOURCE_NAME
    if not draft and not was_applied:
        return False, "no_draft"

    if draft:
        block["rejected_definition"] = draft
    block["drafted_definition"] = None
    block["outcome"] = DRAFT_REJECTED_OUTCOME
    block["terminal"] = False
    # Reset rather than decrement. The attempt cap exists to stop a run retrying an item
    # that keeps failing; a reviewer asking for a redraft is a new intention, not a
    # retry, and leaving attempts at the cap would silently refuse it.
    block["attempts"] = 0
    entry["second_pass"] = block
    # A reviewer asking for a rewrite is a decision to let the model look again, so the
    # tag that would otherwise skip this entry forever comes off.
    entry.pop("ai_status", None)
    entry.pop("ai_status_detail", None)

    if was_applied:
        entry["wikipedia_summary"] = onet_boilerplate(skill_name, entry.get("category") or "")
        entry["best_source_name"] = None
        entry["resolved_title"] = None
        entry["reference_url"] = None
        entry["is_credible"] = None
        entry["cross_score"] = None

    entry["last_updated"] = json_store.today().isoformat()
    logger.info(
        "Draft rejected for %r; it will be redrafted on the next second pass.%s",
        skill_name, " Its accepted text was reverted to boilerplate." if was_applied else "",
    )
    return True, "ok"


def reject_draft(skill_name: str) -> Tuple[bool, str]:
    """
    Sends a model-written definition back to be redrafted.

    Codes: ok, not_found, not_pending, no_draft, error.
    """
    try:
        master = load_master()
        ok, code = _apply_reject_draft(master, skill_name)
        if not ok:
            return ok, code

        save_master(master)
        return True, "ok"
    except Exception:
        logger.exception("Rejecting the draft for %r failed.", skill_name)
        return False, "error"


# Outcomes meaning "there is no encyclopedia article for this product". Both must clear
# whatever page the card is still citing; see _clear_contradicted_page.
NO_ARTICLE_OUTCOMES = ("no_article", "declined_to_define")


def _clear_contradicted_page(entry: dict, skill_name: str) -> bool:
    """
    Drops a reference page a lookup has just said does not exist. Returns whether it did.

    Without this, `no_article` was the one terminal outcome that contradicted its own
    card: 95 entries asserted "a lookup found no encyclopedia article for this product"
    while displaying another article's text as the definition. IEA Software Emerald cited
    "Foreign relations of India"; K2 Business Process Automation cited "K9 Thunder", a
    Korean self-propelled howitzer.

    The page goes REGARDLESS of its cross-encoder score, not only below RESOLUTION_FLOOR.
    A model that has looked at the product and reported no article exists is contradicting
    that page directly, which is stronger evidence than the similarity number that let it
    through in the first place.

    A REVIEWER-REMEDIATED page is left alone. A person asserting a page is exactly the
    thing the automation is guessing at -- the same principle that makes the score
    recorded rather than enforced on that path -- and a model finding no encyclopedia
    article is not grounds to overwrite them.

    The entry stays PENDING and keeps a card: the boilerplate written here is non-empty,
    which is what build_review_rows requires, and gate_reason becomes the bucket the
    second pass drafts definitions for.
    """
    if entry.get("gate_reason") == "reviewer_remediated":
        logger.info(
            "Leaving %r pointed at %r: a reviewer supplied that page.",
            skill_name, entry.get("resolved_title"),
        )
        return False

    had = entry.get("resolved_title")
    entry["resolved_title"] = None
    entry["reference_url"] = None
    entry["cross_score"] = None
    entry["is_credible"] = None
    entry["best_source_name"] = None
    entry["wikipedia_summary"] = onet_boilerplate(skill_name, entry.get("category") or "")
    entry["gate_reason"] = "no_candidate_found"
    entry["last_updated"] = json_store.today().isoformat()

    if had:
        logger.info(
            "No article exists for %r, so its citation of %r was cleared.", skill_name, had
        )
    return True


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
        _tag_ai_status(entry, outcome)

        if outcome == "model_authored":
            _write_model_authored(entry, result["summary"], result.get("source_name"))
            save_master(master)
            logger.warning(
                "Second pass wrote a MODEL-AUTHORED definition for %r. It has no "
                "external source and cannot auto-approve.",
                skill_name,
            )
            return True, "ok"

        if outcome == "model_authored_verified":
            # The one path on which text with no external source reaches the dashboard
            # without a human. It is permitted because it clears the SAME bar a page
            # does: one call wrote the definition, a separate call graded it, and the two
            # agreed. A page auto-approves on the cross-encoder plus the audit; this
            # auto-approves on the writer plus the verifier.
            #
            # The provenance is deliberately loud. best_source_name still says the text
            # is model-authored, and the gate reason distinguishes it from the case where
            # a reviewer accepted one by hand, so the volume of unsourced text on the
            # dashboard stays countable.
            _write_model_authored(entry, result["summary"], result.get("source_name"))
            entry["gate_reason"] = MODEL_AUTHORED_VERIFIED_GATE_REASON
            entry["status"] = STATUS_APPROVED
            entry["report_note"] = None

            metrics = record_snapshot(entry, timeseries)
            save_timeseries(timeseries)
            save_master(master)
            logger.warning(
                "Second pass APPROVED a MODEL-AUTHORED definition for %r on two agreeing "
                "calls: ai=%+.4f (%s). No external source; verifier confidence %s.",
                skill_name, metrics["ai_score"], metrics["category_bucket"],
                result.get("verifier_confidence"),
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

        if outcome in NO_ARTICLE_OUTCOMES:
            _clear_contradicted_page(entry, skill_name)

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


def _apply_machine_suggestion(master: dict, skill_name: str) -> Tuple[bool, str]:
    """
    Accepts a stored suggestion, in memory. Refetches the proposed page.

    Codes: ok, not_found, not_pending, no_suggestion, plus scraping's own error codes.
    """
    from scraping import resolve_reviewer_url

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
    # resolve_reviewer_url only ever returns a Wikipedia page, so any URL left from an
    # earlier reviewer-supplied source no longer describes this card.
    entry["reference_url"] = None
    entry["wikipedia_summary"] = resolved["extract"]
    entry["cross_score"] = resolved["score"]
    entry["is_credible"] = None
    entry["gate_reason"] = SECOND_PASS_GATE_REASON
    entry["last_updated"] = json_store.today().isoformat()

    logger.info("Applied machine suggestion for %r: page %r.", skill_name, resolved["title"])
    return True, "ok"


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
        master = load_master()
        ok, code = _apply_machine_suggestion(master, skill_name)
        if not ok:
            return ok, code

        save_master(master)
        return True, "ok"
    except Exception:
        logger.exception("Could not apply machine suggestion for %r.", skill_name)
        return False, "error"


def _apply_machine_draft(master: dict, skill_name: str) -> Tuple[bool, str]:
    """
    Accepts a stored model-written definition, in memory. No network, no Gemini.

    Codes: ok, not_found, not_pending, no_draft.
    """
    entry = master.get(skill_name)
    if entry is None:
        return False, "not_found"
    if entry.get("status") != STATUS_PENDING:
        return False, "not_pending"

    block = entry.get("second_pass")
    draft = (block or {}).get("drafted_definition") if isinstance(block, dict) else None
    if not (draft or "").strip():
        return False, "no_draft"

    _write_model_authored(entry, draft.strip(), MODEL_AUTHORED_SOURCE_NAME)
    # The proposal has become the card, so record that rather than leaving an
    # outcome that still reads as "not yet applied".
    block["outcome"] = "model_authored"
    entry["second_pass"] = block

    logger.warning(
        "Reviewer accepted a MODEL-AUTHORED definition for %r. It has no external "
        "source and was never credibility-audited.",
        skill_name,
    )
    return True, "ok"


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
        ok, code = _apply_machine_draft(master, skill_name)
        if not ok:
            return ok, code

        save_master(master)
        return True, "ok"
    except Exception:
        logger.exception("Could not apply machine draft for %r.", skill_name)
        return False, "error"


# Actions a staged reviewer decision can carry. Anything not in this table is rejected
# before the store is touched, so a hand-crafted POST cannot reach a writer the review
# page does not offer.
#
# They come in two kinds, and one skill may stage one of each. A SOURCE action changes
# what the card cites -- a page the reviewer supplied, a page the second pass proposed, a
# definition it drafted -- and leaves the item pending. A FINAL action decides it. Being
# able to chain them is the difference between "accept this definition" then reloading,
# finding the card again and approving it, and simply saying "accept it and approve it".
SOURCE_ACTIONS = ("remediate", "apply-suggestion", "apply-draft")
# mark-duplicate is FINAL because it retires the card, and it is meaningful to chain a
# source action before it only in the sense that the source would be wasted work. The
# server allows the chain rather than special-casing it; the merge does not read the
# text either way.
FINAL_ACTIONS = ("approve", "reject", "mark-duplicate", "reject-draft")
BATCH_ACTIONS = SOURCE_ACTIONS + FINAL_ACTIONS

# A commit covers one page of the queue (PAGE_SIZE is 25). The cap is well above that
# and exists only so a malformed or hostile payload cannot make the server hold an
# unbounded list of network fetches open while the store lock is held.
MAX_BATCH_DECISIONS = 200

# Approve scores locally, and the two refetching actions each make a Wikipedia round
# trip, so a commit heavy with those is slow by nature. Worth a line in the log when it
# happens, because the reviewer is sitting in front of a spinner for the duration.
SLOW_ACTION_WARN_THRESHOLD = 10


def apply_review_batch(decisions: list) -> Tuple[int, list, list]:
    """
    Applies a page of staged reviewer decisions in ONE load and ONE save.

    Each decision is {"skill": str, "action": str, "summary": str|None,
    "reference": str|None}. Returns (applied_count, applied_items, failures).

    A skill may carry TWO decisions: one SOURCE action and one FINAL action. They are
    always applied source-first regardless of the order they arrive in, so "use the
    suggested page, then approve it" scores the suggested page rather than the text it
    replaced. If the source fails, the final is not attempted and is reported as
    `source_failed` -- deciding on a page that never landed would be deciding about text
    the reviewer never saw.

    `applied_items` is [{"skill", "action", "status"}] where status is READ BACK from
    the saved store rather than assumed from the action. That is the difference between
    reporting what was asked for and reporting what happened, and it is the whole answer
    to "it said it committed and nothing changed": three of the five actions leave an
    item pending on purpose, and the caller can now say so.

    Each failure is {"skill", "action", "code"} carrying the same codes the
    single-action routes return, so main.py renders them through the tables it has.

    THE BATCH IS NOT A TRANSACTION over individual decisions: a decision that fails is
    recorded and the rest still apply. That is deliberate. The common failure is one
    stale item -- approved or rejected by someone else while this page was open -- and
    discarding twenty-four good decisions because of it would be a worse answer than
    reporting the one that did not land.

    It IS atomic as a write: every applier mutates memory only, and json_store's
    atomic_write means the single save at the end either lands whole or not at all.
    The caller is expected to be inside run_state.store_writer, which is what keeps a
    background scrape from writing over the top of this.
    """
    if not isinstance(decisions, list):
        return 0, [], []
    if len(decisions) > MAX_BATCH_DECISIONS:
        return 0, [], [{"skill": "", "action": "", "code": "too_many"}]

    master, timeseries = _load()

    applied = 0
    scored = False
    applied_items = []
    failures = []
    seen = set()

    slow = sum(
        1 for d in decisions
        if isinstance(d, dict)
        and d.get("action") in ("approve", "remediate", "apply-suggestion")
    )
    if slow > SLOW_ACTION_WARN_THRESHOLD:
        logger.info("Commit carries %d slow decision(s); this will take a moment.", slow)

    # Grouped by skill, in the order the skills were first seen, so a chain runs SOURCE
    # then FINAL whichever order the two were clicked in. Approving before repointing
    # would score the text the reviewer was trying to replace.
    chains = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            failures.append({"skill": "", "action": "", "code": "bad_decision"})
            continue

        skill_name = (decision.get("skill") or "").strip()
        action = (decision.get("action") or "").strip()

        if not skill_name or action not in BATCH_ACTIONS:
            failures.append({"skill": skill_name, "action": action, "code": "bad_decision"})
            continue

        chain = chains.setdefault(skill_name, {"source": None, "final": None})
        slot = "source" if action in SOURCE_ACTIONS else "final"
        if chain[slot] is not None:
            # One of each is the most a card can express. Two sources or two finals is a
            # malformed payload, not a reviewer changing their mind -- the page replaces
            # rather than appends.
            failures.append({"skill": skill_name, "action": action, "code": "duplicate"})
            continue
        chain[slot] = decision

    def _run(skill_name, decision, chained_summary=None):
        """Applies one staged decision in memory. Returns (ok, code)."""
        action = decision["action"]
        if action == "approve":
            # A chained approve scores what the SOURCE action just put on the card,
            # unless the reviewer typed something of their own. The client sends no
            # summary when it did not touch the textarea, precisely so the refetched or
            # accepted text is what gets measured rather than the stale text that was on
            # screen when the button was clicked.
            summary = (decision.get("summary") or "").strip() or (chained_summary or "")
            ok, code = _apply_approve(
                master, timeseries, skill_name, summary, decision.get("reference") or ""
            )
            return ok, code
        if action == "reject":
            return _apply_reject(master, skill_name)
        if action == "mark-duplicate":
            return _apply_mark_duplicate(master, skill_name, decision.get("target") or "")
        if action == "reject-draft":
            return _apply_reject_draft(master, skill_name)
        if action == "remediate":
            return _apply_remediate(master, skill_name, decision.get("reference") or "")
        if action == "apply-suggestion":
            return _apply_machine_suggestion(master, skill_name)
        return _apply_machine_draft(master, skill_name)

    for skill_name, chain in chains.items():
        seen.add(skill_name)
        source_ok = True

        for slot in ("source", "final"):
            decision = chain[slot]
            if decision is None:
                continue

            action = decision["action"]
            if slot == "final" and not source_ok:
                # The page or definition this decision was about never landed, so
                # deciding on it would decide about text the reviewer never saw.
                failures.append(
                    {"skill": skill_name, "action": action, "code": "source_failed"}
                )
                continue

            try:
                chained = None
                if slot == "final" and chain["source"] is not None:
                    chained = (master.get(skill_name) or {}).get("wikipedia_summary")
                ok, code = _run(skill_name, decision, chained_summary=chained)
                if action == "approve":
                    scored = scored or ok
            except Exception:
                logger.exception("Staged %s of %r failed.", action, skill_name)
                ok, code = False, "error"

            if ok:
                applied += 1
                applied_items.append({"skill": skill_name, "action": action})
            else:
                failures.append({"skill": skill_name, "action": action, "code": code})
                if slot == "source":
                    source_ok = False

        if chain["source"] is not None and chain["final"] is not None and source_ok:
            entry = master.get(skill_name) or {}
            if entry.get("best_source_name") == MODEL_AUTHORED_SOURCE_NAME:
                # WARNING, not INFO: a chain is the only way text with no source behind
                # it reaches the dashboard in a single action. Two human clicks, not a
                # machine decision, but it must be greppable after the fact.
                logger.warning(
                    "Chained commit put a MODEL-AUTHORED definition for %r straight to "
                    "%s. It has no external source and was never credibility-audited.",
                    skill_name, entry.get("status"),
                )

    if applied:
        # Timeseries first, master second, matching approve_skill. If the process dies
        # between them the store shows an unapproved skill with a snapshot, which the
        # dashboard filters out by status; the reverse would show an approved skill with
        # no score.
        if scored:
            save_timeseries(timeseries)
        save_master(master)

        # Read back from disk, not from the dict just written. Reporting the in-memory
        # copy would say "approved" even if the save had silently not landed, which is
        # exactly the failure this reporting exists to make visible.
        saved = load_master()
        for item in applied_items:
            entry = saved.get(item["skill"]) or {}
            item["status"] = entry.get("status") or "unknown"

    logger.info(
        "Committed %d staged decision(s), %d failed.", applied, len(failures)
    )
    return applied, applied_items, failures


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

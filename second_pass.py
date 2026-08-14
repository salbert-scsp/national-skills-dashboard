"""
Second-pass triage over the pending review queue.

The queue's dominant failure is entity resolution, not judgement. O*NET names products
the way a purchasing department does -- "Oracle Essbase", "Blackboard software",
"Balsamiq Studios Balsamiq Mockups" -- and the resolver matches those strings
lexically, so it lands on the Oracle Database article, the chalkboard article, and
nothing at all. The credibility audit then correctly says "not credible" without ever
being asked the question a human would ask: which article is this actually about?

So this module does not re-run the audit. It asks the new question, then feeds the
answer back through the EXISTING resolve, score and audit path. Re-asking the old
question on the old text returns the old answer forever, which is the whole reason the
48 failed_credibility_audit items have been sitting there.

Two hard boundaries:

  1. NOTHING HERE WRITES. Every function returns a decision; review_actions.py applies
     it. That is what makes the quota wall safe -- a run that dies between the proposal
     call and the audit call leaves the entry byte-identical, with no marker, no partial
     evidence and no verdict attributed to a model that never spoke.
  2. NOTHING HERE CAN ENQUEUE. Candidates are drawn only from entries that are already
     pending with text a human could act on. The pass can settle a card or improve one;
     it cannot create one.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import json_store
from agentic_source_check import BATCH_SIZE, AuditUnavailable, get_source_checker
from gemini_keys import (
    ROLE_AUDIT,
    ROLE_DRAFT,
    ROLE_PROPOSAL,
    ROLE_VERIFY,
    DailyQuotaExhausted,
)
from json_store import STATUS_PENDING
from review_actions import MODEL_AUTHORED_SOURCE_NAME

logger = logging.getLogger(__name__)

SECOND_PASS_VERSION = 1

# Transient outcomes get one retry. Terminal ones never re-spend quota, which is what
# bounds a run: re-running over the same queue converges on zero calls rather than
# re-deciding everything at full price.
MAX_SECOND_PASS_ATTEMPTS = 2

# reviewer_remediated is deliberately absent. A human supplied that page, and a machine
# must not second-guess them -- the same reasoning that keeps remediate_skill from
# spending a Gemini call to check a reviewer's work.
ELIGIBLE_GATE_REASONS = (
    "below_relevance_threshold",
    "failed_credibility_audit",
    "no_candidate_found",
)

# Outcomes that settle an item. A later run skips these without a call.
#
# "applied" is legacy and no longer produced: decide() used to let a credible-but-
# below-threshold resolution rewrite the card in apply/auto mode, and now stages it as
# "suggested" instead. It stays listed, and stays handled in review_actions, because
# entries settled by earlier runs still carry it and must keep rendering and skipping.
TERMINAL_OUTCOMES = frozenset({
    "auto_approved",
    "applied",
    "model_authored",
    "drafted",
    "no_article",
    "confirmed_original",
    "confirmed_original_failed_audit",
    "suggested",
    "suggested_strong",
    "suggested_weak",
    "not_a_product",
    "no_distinct_article",
    "model_authored_verified",
    # "no article exists AND the model would not describe the product either". Split from
    # no_article because the two ask different things of the reviewer: one has a written
    # definition behind it, the other means nobody here knows what this product is and a
    # human has to say so.
    "declined_to_define",
})

# Both mean "there is no encyclopedia article". review_actions clears the page the card
# is still citing for either of them.
NO_ARTICLE_OUTCOMES = ("no_article", "declined_to_define")


# --- Why an item is skipped, recorded ON THE ENTRY ----------------------------
#
# The second pass used to skip anything whose outcome was in TERMINAL_OUTCOMES. That was
# invisible: nothing on the card said the model had stopped looking, so a queue of items
# the automation had quietly given up on was indistinguishable from one it had not
# reached yet, and "why is the second pass instant" had no answer short of reading the
# store.
#
# Skipping is now driven by an EXPLICIT tag written onto the entry. Two consequences that
# are the whole point:
#
#   1. Nothing is skipped without a tag a human can see. An entry with no ai_status is
#      always re-attempted, whatever its history.
#   2. Every skip is legible on the card, and AI_CANNOT_DETERMINE in particular is the
#      answer to "which of these actually need a person".
AI_CANNOT_DETERMINE = "cannot_determine"   # the model looked and gave up
AI_AWAITING_REVIEW = "awaiting_review"     # the model has an answer; a human must act

# Outcome -> tag. An outcome ABSENT from this map is transient (a failed call, an
# unreachable audit, a reviewer-requested redraft) and leaves the entry untagged, which
# means it is picked up again on the next run. That absence is the retry mechanism.
OUTCOME_AI_STATUS = {
    # The model gave up. These are what a person has to resolve.
    "no_article": AI_CANNOT_DETERMINE,
    "declined_to_define": AI_CANNOT_DETERMINE,
    "not_a_product": AI_CANNOT_DETERMINE,
    "unresolvable": AI_CANNOT_DETERMINE,
    # The model produced an answer and is waiting on a click. Re-asking would spend a
    # request to overwrite a proposal the reviewer has not looked at yet.
    "drafted": AI_AWAITING_REVIEW,
    "model_authored": AI_AWAITING_REVIEW,
    # Approved outright by two agreeing calls; it leaves the queue entirely.
    "model_authored_verified": AI_AWAITING_REVIEW,
    "suggested": AI_AWAITING_REVIEW,
    "suggested_strong": AI_AWAITING_REVIEW,
    "suggested_weak": AI_AWAITING_REVIEW,
    "confirmed_original": AI_AWAITING_REVIEW,
    "confirmed_original_failed_audit": AI_AWAITING_REVIEW,
    "no_distinct_article": AI_AWAITING_REVIEW,
    "applied": AI_AWAITING_REVIEW,
    "auto_approved": AI_AWAITING_REVIEW,
}


def ai_status_for(outcome: Optional[str]) -> Optional[str]:
    """The tag an outcome earns, or None when the item should be tried again."""
    return OUTCOME_AI_STATUS.get(outcome or "")

# The model's own confidence in its own proposal, which until now was recorded on every
# entry and compared to nothing.
#
# Below this, the page is NOT chased. Fetching, rescoring and auditing a title the model
# was unsure of costs a request to end up arguing for an article nobody stood behind, and
# the same call that produced the shaky title can produce a definition instead. The
# number matches CROSS_ENCODER_THRESHOLD for familiarity, but it measures something
# softer: a self-report, not a measurement. It is a spending rule, not a guarantee, which
# is why the guessed title is still recorded and shown to the reviewer.
PROPOSAL_CONFIDENCE_THRESHOLD = 0.90

APPLY_MODES = ("suggest", "apply", "auto")

# Provenance marker for a sourceless definition. Defined in review_actions, which owns
# every write, and re-exported here so callers of this module do not need both imports.
MODEL_AUTHORED_SOURCE = MODEL_AUTHORED_SOURCE_NAME


def _norm_title(title: Optional[str]) -> str:
    return (title or "").strip().lower().replace("_", " ")


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def _already_settled(entry: Dict[str, Any]) -> bool:
    """
    Whether the second pass should skip this entry. Driven ONLY by the visible tag.

    The outcome recorded in the second_pass block is deliberately NOT consulted. An
    outcome is a finding; ai_status is a decision to stop, and only the second can
    silence an item. That separation is what makes "the second pass is skipping things"
    something a reviewer can see on a card rather than infer from an empty run.

    MAX_SECOND_PASS_ATTEMPTS is not read here either. The attempt cap is enforced where
    the outcome is recorded, by writing AI_CANNOT_DETERMINE once it trips -- so an item
    that keeps failing still ends up tagged rather than silently dropping out.
    """
    return bool(entry.get("ai_status"))


def select_candidates(
    master: Dict[str, Any],
    *,
    buckets: Optional[Tuple[str, ...]] = None,
    limit: Optional[int] = None,
    redo: bool = False,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Pending entries a second pass could usefully re-resolve.

    The filter mirrors build_review_rows' invariant exactly: pending AND carrying text
    a human could act on. An entry outside that set is one of the ~1,120 non-hot tools
    recorded for the occupation map, and pulling one in would materialize a card nobody
    asked for -- the failure the 388-row backlog taught.
    """
    wanted = tuple(buckets) if buckets else ELIGIBLE_GATE_REASONS

    rows = [
        (name, entry)
        for name, entry in master.items()
        if entry.get("status") == STATUS_PENDING
        and (entry.get("wikipedia_summary") or "").strip()
        and entry.get("gate_reason") in wanted
        and (redo or not _already_settled(entry))
    ]
    rows.sort(key=lambda row: row[0])
    return rows[:limit] if limit else rows


# --------------------------------------------------------------------------
# The three calls
#
# Two of them -- propose and audit -- are issued in batches from process_chunk, which is
# where their prompts and their item_id mapping live. Only the local rescore and the
# rare sourceless draft remain per-item here.
# --------------------------------------------------------------------------

def _occupation_titles(entry: Dict[str, Any]) -> List[str]:
    titles = [
        occ.get("onet_title")
        for occ in (entry.get("occupations") or [])
        if isinstance(occ, dict) and occ.get("onet_title")
    ]
    return titles or [t for t in (entry.get("onet_titles") or []) if t]


def verify_proposal(skill_name: str, category: str, proposed_title: str) -> Dict[str, Any]:
    """
    Fetches and rescores a proposed title. No Gemini call; the cross-encoder is local.

    Routed through resolve_reviewer_url rather than fetch_wikipedia_extract for three
    reasons, all load-bearing. It is the existing "someone handed us a title, check it"
    path, so no fetch or scoring logic is duplicated. It applies the same title
    validation as human input, so a hallucinated reference cannot reach a fetch -- model
    output must never be handed to the fetcher directly. And it reports disambiguation
    and missing pages as distinct errors instead of as a generic scoring failure.
    """
    from scraping import resolve_reviewer_url

    return resolve_reviewer_url(skill_name, category or "", proposed_title)


def draft_for_entry(skill_name: str, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One Gemini call, last resort for products with no article at all."""
    return get_source_checker(ROLE_DRAFT).draft_definition(
        skill_name=skill_name,
        category=entry.get("category") or "",
        occupation_titles=_occupation_titles(entry),
    )


# --------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------

def _result(outcome: str, **fields: Any) -> Dict[str, Any]:
    result = {"outcome": outcome, "terminal": outcome in TERMINAL_OUTCOMES}
    result.update(fields)
    return result


def decide(
    entry: Dict[str, Any],
    proposal: Optional[Dict[str, Any]],
    verified: Optional[Dict[str, Any]],
    audit: Optional[Dict[str, Any]],
    *,
    apply_mode: str,
    threshold: float,
) -> Dict[str, Any]:
    """
    Turns the three call results into one outcome. PURE: no IO, no clock, no network.

    Separated from process_one so the whole decision surface can be exercised against
    every real pending entry offline, with stubbed call results, before a single token
    is spent. `threshold` is injected for the same reason.
    """
    if proposal is None:
        return _result("proposal_failed")

    if not proposal.get("article_exists"):
        # Never auto-reject. "I could not find an article" is a research result, not a
        # verdict on whether the skill belongs in the taxonomy, and rejecting drops the
        # skill out of the queue where nobody would ever see the claim.
        return _result(
            "no_article",
            rationale=proposal.get("rationale"),
            confidence=proposal.get("confidence"),
            suggested_action="reject",
        )

    proposed_title = (proposal.get("proposed_title") or "").strip()
    if not proposed_title:
        # article_exists true with no title is a malformed answer, not a finding.
        return _result("proposal_failed")

    if proposal.get("same_as_current") or (
        _norm_title(proposed_title) == _norm_title(entry.get("resolved_title"))
        and entry.get("resolved_title")
    ):
        if entry.get("gate_reason") == "failed_credibility_audit":
            # The audit already ran against this exact text and rejected it. Spending a
            # second call to ask again is precisely the waste this module exists to stop.
            return _result(
                "confirmed_original_failed_audit",
                proposed_title=proposed_title,
                rationale=proposal.get("rationale"),
                confidence=proposal.get("confidence"),
            )
        return _result(
            "confirmed_original",
            proposed_title=proposed_title,
            measured_score=entry.get("cross_score"),
            rationale=proposal.get("rationale"),
            confidence=proposal.get("confidence"),
        )

    if verified is None:
        return _result("unresolvable", proposed_title=proposed_title, detail="not_attempted")
    if verified.get("error"):
        return _result(
            "unresolvable", proposed_title=proposed_title, detail=verified["error"]
        )

    # The model named a DIFFERENT article and it redirected onto the one already on file.
    # "AutoCAD Civil 3D" and "Civil 3D" are both redirects to "AutoCAD", so the check
    # above -- which runs on the raw title, before resolution -- lets them through.
    #
    # This is not a suggestion. It is evidence that the specific product has no article
    # of its own, and offering "this is really AutoCAD, not AutoCAD" wastes a reviewer's
    # attention. Placed before the audit test on purpose: reachable with audit=None, so
    # the caller can settle it without spending a credibility call on a page the card is
    # already pointed at.
    resolved_title = (verified.get("title") or proposed_title).strip()
    if entry.get("resolved_title") and _norm_title(resolved_title) == _norm_title(
        entry.get("resolved_title")
    ):
        return _result(
            "no_distinct_article",
            proposed_title=resolved_title,
            prior_title=entry.get("resolved_title"),
            measured_score=entry.get("cross_score"),
            rationale=proposal.get("rationale"),
            confidence=proposal.get("confidence"),
            # The name that redirected, so the card can show what was actually proposed.
            detail=proposed_title,
        )

    if audit is None:
        return _result("audit_unavailable", proposed_title=proposed_title)

    score = verified.get("score")
    clean_summary = audit.get("clean_summary")
    credible = bool(audit.get("is_credible")) and bool(clean_summary)

    common = {
        "proposed_title": verified.get("title") or proposed_title,
        "measured_score": score,
        "prior_title": entry.get("resolved_title"),
        "rationale": proposal.get("rationale"),
        "confidence": proposal.get("confidence"),
        "extract": verified.get("extract") or "",
        "source_name": audit.get("best_source_name") or verified.get("source_name"),
    }

    # ---- Tier A: a clean re-resolve ------------------------------------
    # Exactly the bar scraping.py already uses to auto-approve on the first pass:
    # the cross-encoder cleared the threshold AND the audit passed. No new privilege
    # is granted here, the same evidence is simply gathered a second time.
    if score is not None and score >= threshold and credible:
        if apply_mode == "auto":
            return _result(
                "auto_approved", summary=clean_summary, is_credible=True, **common
            )
        return _result("suggested_strong", summary=clean_summary, **common)

    # ---- Tier B: the alias trap ----------------------------------------
    # PySpark against the Apache Spark article scores 0.0 and is correct. The audit
    # passing on a page the cross-encoder rejects is the signature of that case.
    #
    # It is STAGED, never applied and never approved, in every mode. resolve_reviewer_url
    # drops the threshold for human input because a reviewer supplying a page is
    # asserting the very thing the cross-encoder exists to guess; a model has no such
    # standing. Two model opinions agreeing -- one call chose the article, a different
    # call graded it -- is enough to hand a human a finished answer to accept in one
    # click, and not enough to rewrite the card underneath them before they look.
    if credible:
        return _result("suggested", summary=clean_summary, **common)

    # A better page than the one on file, but the audit did not clear it. Worth showing;
    # not worth applying over text a human may already have started reading.
    return _result("suggested_weak", summary=None, **common)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _draft_or_settle(
    skill_name: str,
    entry: Dict[str, Any],
    proposal: Optional[Dict[str, Any]],
    *,
    low_confidence: bool = False,
) -> Dict[str, Any]:
    """
    Settles an item with a written definition instead of a page.

    Reached two ways: no article exists at all, or the model's confidence in the article
    it named is below PROPOSAL_CONFIDENCE_THRESHOLD. Both mean the same thing in spending
    terms -- there is no page here worth fetching, rescoring and auditing.

    NO CALL IS MADE when the proposal already carries a definition, which it does
    whenever the prompt's own rule fired. The separate draft_definition request is a
    fallback for a proposal that came back with neither a title nor a definition, not
    the normal path: paying twice to ask one model what one product is was the waste
    this exists to remove.
    """
    draft = proposal if (proposal or {}).get("definition") else None
    if draft is None:
        draft = draft_for_entry(skill_name, entry)
    if draft is None:
        return _result("draft_failed")
    if not draft.get("is_software_product"):
        return _result(
            "not_a_product",
            rationale=(proposal or {}).get("rationale"),
            suggested_action="reject",
        )

    definition = (draft.get("definition") or "").strip()
    if not definition:
        # The model declined to guess, which is the behaviour the prompt asks for and a
        # useful signal in itself: it looked for an article, found none, and would not
        # describe the product from memory either. Distinct from no_article, where a
        # definition WAS written -- here there is nothing at all, and the card should ask
        # a human for one rather than reading as though a lookup merely came up short.
        return _result(
            "declined_to_define",
            rationale=(proposal or {}).get("rationale"),
            suggested_action="reject",
        )

    # Staged, pending verification. A definition reaches the dashboard only when a
    # SEPARATE call has graded it -- see verify_drafts, which upgrades this outcome to
    # model_authored_verified when the two agree. Until that runs, and whenever it
    # disagrees or cannot answer, this stays a proposal a reviewer accepts by hand.
    #
    # On the low-confidence path the guessed title rides along. It cost nothing -- the
    # call was made anyway -- and it is the reviewer's cheapest next move if they think
    # the guess was right: paste it into the reference field. It is recorded as a guess,
    # never as a proposal, because the whole reason no page was fetched is that the model
    # did not stand behind it.
    return _result(
        "drafted",
        drafted_definition=definition,
        source_name=MODEL_AUTHORED_SOURCE,
        rationale=(proposal or {}).get("rationale"),
        confidence=(proposal or {}).get("confidence"),
        proposed_title=(
            ((proposal or {}).get("proposed_title") or "").strip() or None
            if low_confidence else None
        ),
        detail="low_confidence" if low_confidence else None,
    )


def verify_drafts(
    results: Dict[str, Dict[str, Any]], entries: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Grades every drafted definition in a chunk, in one batched call, and upgrades the
    ones a second opinion agrees with.

    This is the second of the two opinions, and the reason a written definition may now
    auto-approve at all. One call wrote it; a different call, shown the definition and
    NOT the reasoning behind it, judges whether it describes the right product and
    whether it invents specifics. Only agreement promotes "drafted" to
    "model_authored_verified".

    Everything else stays exactly as it was:

      - a verdict that fails either test keeps the item staged, with the stated problem
        recorded so the reviewer sees WHY rather than a bare flag
      - an id the verifier did not answer for stays staged. A missing verdict is not
        agreement, and a quota wall must never read as a pass
      - AuditUnavailable leaves the whole chunk staged, which is the same per-batch
        retryable failure the audit leg already has

    Mutates and returns `results`, matching how process_chunk assembles its outcomes.
    """
    drafted = {
        skill_name: result for skill_name, result in results.items()
        if result.get("outcome") == "drafted" and (result.get("drafted_definition") or "").strip()
    }
    if not drafted:
        return results

    items = [
        {
            "item_id": skill_name,
            "skill_name": skill_name,
            "category": (entries.get(skill_name) or {}).get("category") or "",
            "occupation_titles": _occupation_titles(entries.get(skill_name) or {}),
            "definition": result["drafted_definition"],
        }
        for skill_name, result in drafted.items()
    ]

    try:
        verdicts = get_source_checker(ROLE_VERIFY).verify_definitions_batch(items)
    except AuditUnavailable:
        logger.warning(
            "Could not verify %d written definition(s); they stay staged for a human.",
            len(items),
        )
        return results

    promoted = 0
    for skill_name, result in drafted.items():
        verdict = verdicts.get(skill_name)
        if verdict is None:
            logger.info("No verdict for %r; it stays staged.", skill_name)
            continue

        result["verified_by"] = "definition_check"
        result["verifier_confidence"] = verdict.get("confidence")
        result["is_generic_category"] = bool(verdict.get("is_generic_category"))

        if not verdict.get("describes_this_skill"):
            result["detail"] = "verifier_wrong_subject"
            result["verifier_problem"] = verdict.get("problem")
            logger.info(
                "Definition for %r describes the wrong subject: %s",
                skill_name, verdict.get("problem"),
            )
            continue
        if verdict.get("contains_invented_specifics"):
            result["detail"] = "verifier_invented_specifics"
            result["verifier_problem"] = verdict.get("problem")
            logger.info(
                "Definition for %r invents specifics: %s", skill_name, verdict.get("problem")
            )
            continue

        result["outcome"] = "model_authored_verified"
        result["summary"] = result["drafted_definition"]
        promoted += 1

    if promoted:
        logger.info(
            "%d of %d written definition(s) were verified and will be approved without a "
            "human. The rest stay staged.", promoted, len(drafted),
        )
    return results


def process_chunk(
    candidates: List[Tuple[str, Dict[str, Any]]], *, apply_mode: str
) -> Dict[str, Dict[str, Any]]:
    """
    Runs up to BATCH_SIZE items end to end and returns {skill_name: outcome}. Writes nothing.

    At most two batched Gemini requests for the whole chunk, and often one: the first
    asks which article each skill names, the second audits only the pages worth auditing.
    Between them sits verify_proposal, which is Wikipedia plus the local cross-encoder and
    costs no quota at all.

    Three kinds of item never reach the second request, because there is nothing there
    worth grading: a proposal the model is not confident in, a proposal that redirects
    onto the page already on the card, and a product with no article at all. The first
    call is asked to write a definition for exactly those cases, so they settle on text
    that has already been paid for rather than on a page nobody stood behind.

    The propose and audit legs stay SEPARATE requests, which is the invariant the whole
    module rests on -- one call chooses the article, a different call grades it, and the
    auto-approval bar is those two agreeing. Batching packs several skills into each leg;
    it never merges the legs.

    DailyQuotaExhausted is allowed to propagate: it is not a per-item failure but the end
    of the run, and the caller stops on it. AuditUnavailable is per-batch, so the items in
    that batch become retryable outcomes and the run continues.
    """
    from scraping import CROSS_ENCODER_THRESHOLD

    results: Dict[str, Dict[str, Any]] = {}
    work: List[Tuple[str, Dict[str, Any]]] = []

    for skill_name, entry in candidates:
        if entry.get("status") != STATUS_PENDING:
            # Belt and braces against a caller that built its own candidate list. The
            # same check exists in review_actions; neither is sufficient alone, because
            # this one guards the quota spend and that one guards the write.
            results[skill_name] = _result("skipped_not_pending")
        else:
            work.append((skill_name, entry))

    if not work:
        return results

    # ---- Leg 1: which article is each of these? ------------------------
    #
    # The confidence bar goes to the model so it knows the rule it is being judged by:
    # below it, no page is fetched and its written definition is what gets used, so a
    # guessed title helps nobody and an honest low score costs it nothing.
    try:
        proposals = get_source_checker(ROLE_PROPOSAL).propose_reference_pages_batch(
            [
                {
                    "item_id": skill_name,
                    "skill_name": skill_name,
                    "category": entry.get("category") or "",
                    "occupation_titles": _occupation_titles(entry),
                    "current_title": entry.get("resolved_title"),
                    # The card's stored summary IS the extract for these buckets, so no
                    # cache read is needed to show the model what the pipeline believes.
                    "current_extract_head": (
                        (entry.get("wikipedia_summary") or "") if entry.get("resolved_title") else ""
                    ),
                }
                for skill_name, entry in work
            ],
            confidence_bar=PROPOSAL_CONFIDENCE_THRESHOLD,
        )
    except AuditUnavailable:
        logger.warning(
            "Second pass could not reach Gemini for a batch of %d item(s). Will retry.",
            len(work),
        )
        for skill_name, _ in work:
            results[skill_name] = _result("audit_unavailable")
        return results

    # ---- Between the legs: local resolve and rescore, no quota ---------
    awaiting_audit: List[Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []

    for skill_name, entry in work:
        proposal = proposals.get(skill_name)

        # No article exists for this product. The only path that produces text with no
        # source behind it, and it is taken on two different terms:
        #
        #   - the proposal already carries a definition, written in the call that has
        #     just been paid for. Free, so it is offered whatever bucket the entry is
        #     in: a card whose page is wrong and whose product has no article is exactly
        #     as stuck as one that never resolved at all.
        #   - it does not, and only the no_candidate_found bucket is worth a SEPARATE
        #     draft request. That is the old cost discipline, kept for the case where
        #     the definition has to be bought.
        if proposal is None or not proposal.get("article_exists"):
            has_definition = bool((proposal or {}).get("definition"))
            if has_definition or entry.get("gate_reason") == "no_candidate_found":
                try:
                    results[skill_name] = _draft_or_settle(skill_name, entry, proposal)
                except AuditUnavailable:
                    logger.warning(
                        "Second pass could not draft a definition for %r. Will retry.",
                        skill_name,
                    )
                    results[skill_name] = _result("audit_unavailable")
                continue

            results[skill_name] = decide(entry, proposal, None, None,
                                         apply_mode=apply_mode,
                                         threshold=CROSS_ENCODER_THRESHOLD)
            continue

        # The model named an article but does not stand behind it. Stop here: fetching,
        # rescoring and auditing it would spend a request to argue for a page chosen on
        # a coin flip. The definition it wrote in this same call is the better answer and
        # is already paid for.
        confidence = proposal.get("confidence")
        if confidence is not None and confidence < PROPOSAL_CONFIDENCE_THRESHOLD:
            logger.info(
                "Proposal for %r is %.2f confident, below %.2f. Taking its definition "
                "instead of chasing %r.",
                skill_name, confidence, PROPOSAL_CONFIDENCE_THRESHOLD,
                proposal.get("proposed_title"),
            )
            try:
                results[skill_name] = _draft_or_settle(
                    skill_name, entry, proposal, low_confidence=True
                )
            except AuditUnavailable:
                logger.warning(
                    "Second pass could not draft a definition for %r. Will retry.", skill_name
                )
                results[skill_name] = _result("audit_unavailable")
            continue

        proposed_title = (proposal.get("proposed_title") or "").strip()
        same = proposal.get("same_as_current") or (
            entry.get("resolved_title")
            and _norm_title(proposed_title) == _norm_title(entry.get("resolved_title"))
        )
        if not proposed_title or same:
            results[skill_name] = decide(entry, proposal, None, None,
                                         apply_mode=apply_mode,
                                         threshold=CROSS_ENCODER_THRESHOLD)
            continue

        verified = verify_proposal(skill_name, entry.get("category") or "", proposed_title)
        if verified.get("error"):
            results[skill_name] = decide(entry, proposal, verified, None,
                                         apply_mode=apply_mode,
                                         threshold=CROSS_ENCODER_THRESHOLD)
            continue

        # A redirect can land the proposal back on the page already on the card, which
        # the raw-title check above cannot see. decide() settles that as
        # no_distinct_article without an audit result, so pass it through here rather
        # than queueing a credibility call on a page the card already carries.
        if entry.get("resolved_title") and _norm_title(
            verified.get("title") or proposed_title
        ) == _norm_title(entry.get("resolved_title")):
            logger.info(
                "Proposal %r for %r redirects to %r, already on the card. No audit spent.",
                proposed_title, skill_name, entry.get("resolved_title"),
            )
            results[skill_name] = decide(entry, proposal, verified, None,
                                         apply_mode=apply_mode,
                                         threshold=CROSS_ENCODER_THRESHOLD)
            continue

        awaiting_audit.append((skill_name, entry, proposal, verified))

    if not awaiting_audit:
        # Straight to leg 3, NOT straight out. A chunk where every item ended up with a
        # written definition has no page to audit, which is precisely the case the
        # verification leg exists for -- returning here skipped it for the common case.
        return verify_drafts(results, {name: entry for name, entry in work})

    # ---- Leg 2: is the page that came back authoritative? --------------
    try:
        audits = get_source_checker(ROLE_AUDIT).evaluate_candidates_batch([
            {
                "item_id": skill_name,
                "skill_name": skill_name,
                "source_name": verified.get("source_name") or "Wikipedia",
                "raw_text": verified.get("extract") or "",
            }
            for skill_name, _, _, verified in awaiting_audit
        ])
    except AuditUnavailable:
        audits = {}

    for skill_name, entry, proposal, verified in awaiting_audit:
        # A missing id is "the audit did not run", which decide() reads from a None
        # audit and records as the retryable audit_unavailable outcome. It must never
        # be read as a failed audit.
        results[skill_name] = decide(
            entry, proposal, verified, audits.get(skill_name),
            apply_mode=apply_mode, threshold=CROSS_ENCODER_THRESHOLD,
        )

    # ---- Leg 3: a second opinion on anything written from memory -------
    #
    # One batched call for the whole chunk, and only for items that ended up with a
    # written definition. A page-backed item never reaches it: those already have two
    # opinions behind them, the cross-encoder and the audit.
    return verify_drafts(results, {name: entry for name, entry in work})


def process_one(
    skill_name: str, entry: Dict[str, Any], *, apply_mode: str
) -> Dict[str, Any]:
    """
    Runs one item end to end and returns its outcome. Writes nothing.

    A chunk of one. Kept as the single-item entry point so the behaviour of one skill
    can be exercised on its own, and so there is only one implementation of the ladder.
    """
    outcomes = process_chunk([(skill_name, entry)], apply_mode=apply_mode)
    return outcomes.get(skill_name, _result("audit_unavailable"))


def run_second_pass(
    *,
    buckets: Optional[Tuple[str, ...]] = None,
    limit: Optional[int] = None,
    apply_mode: str = "suggest",
    redo: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Sweeps the pending queue. Returns a summary; the review UI reads it back.

    apply_mode:
      suggest  record findings only. NO card field is touched -- not the status, text,
               title, gate reason, score or credibility -- so every change a reviewer
               ends up seeing is one they clicked for. The default, deliberately.
      apply    accepted for compatibility, and now identical to suggest. It used to
               repoint cards on a below-threshold resolution and write sourceless
               definitions; both are staged for a click instead, because neither cleared
               the high-confidence bar that would justify a machine editing a card.
      auto     additionally approve Tier A re-resolves outright -- score at or above the
               cross-encoder threshold AND a passing credibility audit. The only
               machine write that remains.

    On the daily quota wall the loop stops where it stands. Nothing is queued to
    backlog.py: that store tracks occupation-granular ingestion work, and second-pass
    work is re-derived for free by rescanning the queue, so queueing it there would
    make the next ingestion re-scrape occupations that need nothing. Resumability comes
    from the absence of a marker on the untouched entries.
    """
    if apply_mode not in APPLY_MODES:
        raise ValueError(f"Unknown apply_mode {apply_mode!r}; expected one of {APPLY_MODES}.")

    import review_actions

    master = json_store.load_master()
    candidates = select_candidates(master, buckets=buckets, limit=limit, redo=redo)

    summary: Dict[str, Any] = {
        "selected": len(candidates),
        "processed": 0,
        "remaining": len(candidates),
        "quota_exhausted": False,
        "apply_mode": apply_mode,
        "dry_run": dry_run,
        "outcomes": {},
    }

    if not candidates:
        logger.info("Second pass found nothing to do.")
        return summary

    logger.info(
        "Second pass starting over %d pending item(s) in %s mode%s.",
        len(candidates), apply_mode, " (dry run)" if dry_run else "",
    )

    for start in range(0, len(candidates), BATCH_SIZE):
        chunk = candidates[start:start + BATCH_SIZE]
        try:
            outcomes = process_chunk(chunk, apply_mode=apply_mode)
        except DailyQuotaExhausted:
            # The entries are untouched: process_chunk wrote nothing, and the writes
            # below never ran for this chunk. Re-running tomorrow picks them up
            # unchanged, including the ones whose proposal had already come back --
            # a repeated question costs a request, an entry marked from half an answer
            # would cost correctness.
            #
            # The exception's own message says remaining work went to the backlog,
            # which is true of ingestion and not of this. Nothing is queued here
            # because nothing needs to be, so the message is not repeated.
            logger.error(
                "All Gemini keys have hit their daily quota. Second pass stopping after "
                "%d item(s); %d were never reached, starting at %r. Nothing was written "
                "for them and re-running tomorrow picks them up unchanged.",
                summary["processed"], len(candidates) - summary["processed"], chunk[0][0],
            )
            summary["quota_exhausted"] = True
            break

        for skill_name, _ in chunk:
            result = outcomes.get(skill_name)
            if result is None:
                # process_chunk answers for every item it is given, so this is a bug
                # rather than an outcome. Left unrecorded, which means the next run
                # re-asks: the one safe response to not knowing what happened.
                logger.error("Second pass produced no outcome for %r.", skill_name)
                continue

            outcome = result["outcome"]
            summary["outcomes"][outcome] = summary["outcomes"].get(outcome, 0) + 1
            summary["processed"] += 1
            summary["remaining"] = len(candidates) - summary["processed"]

            if dry_run:
                logger.info("[dry run] %r -> %s", skill_name, outcome)
                continue

            ok, code = review_actions.apply_second_pass_result(skill_name, result)
            if not ok:
                logger.warning(
                    "Could not record second-pass result for %r: %s.", skill_name, code
                )

    logger.info(
        "Second pass finished: %d processed, %d left. %s",
        summary["processed"], summary["remaining"], summary["outcomes"],
    )
    return summary

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
from agentic_source_check import AuditUnavailable, get_source_checker
from gemini_keys import DailyQuotaExhausted
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
})

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
    block = entry.get("second_pass")
    if not isinstance(block, dict):
        return False
    if block.get("outcome") in TERMINAL_OUTCOMES:
        return True
    return int(block.get("attempts") or 0) >= MAX_SECOND_PASS_ATTEMPTS


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
# --------------------------------------------------------------------------

def _occupation_titles(entry: Dict[str, Any]) -> List[str]:
    titles = [
        occ.get("onet_title")
        for occ in (entry.get("occupations") or [])
        if isinstance(occ, dict) and occ.get("onet_title")
    ]
    return titles or [t for t in (entry.get("onet_titles") or []) if t]


def propose_for_entry(skill_name: str, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One Gemini call: which article is this? Returns None when no usable answer came back."""
    current_title = entry.get("resolved_title")
    return get_source_checker().propose_reference_page(
        skill_name=skill_name,
        category=entry.get("category") or "",
        occupation_titles=_occupation_titles(entry),
        current_title=current_title,
        # The card's stored summary IS the extract for these buckets, so no cache read
        # is needed to show the model what the pipeline currently believes.
        current_extract_head=(entry.get("wikipedia_summary") or "") if current_title else "",
    )


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


def audit_proposal(skill_name: str, verified: Dict[str, Any]) -> Dict[str, Any]:
    """
    One Gemini call: is the newly resolved text authoritative and about this skill?

    Kept as a SEPARATE call from propose_for_entry on purpose. Merging them would halve
    the cost and quietly destroy the design: the moment the model that picks the article
    also grades it, the auto-approval bar below stops being two independent signals and
    becomes one opinion agreeing with itself, while the code still reads as though two
    things agreed.
    """
    return get_source_checker().evaluate_candidates(
        skill_name=skill_name,
        candidates=[{
            "source_name": verified.get("source_name") or "Wikipedia",
            "raw_text": verified.get("extract") or "",
        }],
    )


def draft_for_entry(skill_name: str, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One Gemini call, last resort for products with no article at all."""
    return get_source_checker().draft_definition(
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
    # It gets APPLIED but never APPROVED. resolve_reviewer_url drops the threshold for
    # human input because a reviewer supplying a page is asserting the very thing the
    # cross-encoder exists to guess; a model has no such standing. What substitutes here
    # is two independent signals agreeing -- one call chose the article, a different call
    # graded it -- and two model opinions are enough to hand a human a finished answer,
    # not enough to skip them.
    if credible:
        if apply_mode in ("apply", "auto"):
            return _result("applied", summary=clean_summary, **common)
        return _result("suggested", summary=clean_summary, **common)

    # A better page than the one on file, but the audit did not clear it. Worth showing;
    # not worth applying over text a human may already have started reading.
    return _result("suggested_weak", summary=None, **common)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def process_one(
    skill_name: str, entry: Dict[str, Any], *, apply_mode: str
) -> Dict[str, Any]:
    """
    Runs one item end to end and returns its outcome. Writes nothing.

    DailyQuotaExhausted is allowed to propagate: it is not a per-item failure but the
    end of the run, and the caller stops on it. AuditUnavailable is per-item, so it
    becomes a retryable outcome and the loop continues.
    """
    from scraping import CROSS_ENCODER_THRESHOLD

    if entry.get("status") != STATUS_PENDING:
        # Belt and braces against a caller that built its own candidate list. The same
        # check exists in review_actions; neither is sufficient alone, because this one
        # guards the quota spend and that one guards the write.
        return _result("skipped_not_pending")

    try:
        proposal = propose_for_entry(skill_name, entry)

        # No article, and nothing on file to fall back to. The only path that produces
        # text with no source behind it, in either of its two forms below.
        if (
            entry.get("gate_reason") == "no_candidate_found"
            and (proposal is None or not proposal.get("article_exists"))
        ):
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
                # The model declined to guess, which is the behaviour the prompt asks
                # for and a useful signal in itself. Terminal via no_article below.
                return _result(
                    "no_article",
                    rationale=(proposal or {}).get("rationale"),
                    suggested_action="reject",
                )
            if apply_mode == "suggest":
                # Sourceless text is the last thing that should appear on a card
                # unasked. Carried as a proposal the reviewer accepts, so the only way
                # a definition with nothing behind it reaches the queue is a click.
                return _result(
                    "drafted",
                    drafted_definition=definition,
                    source_name=MODEL_AUTHORED_SOURCE,
                    rationale=(proposal or {}).get("rationale"),
                )

            return _result(
                "model_authored",
                summary=definition,
                source_name=MODEL_AUTHORED_SOURCE,
                rationale=(proposal or {}).get("rationale"),
            )

        if proposal is None or not proposal.get("article_exists"):
            return decide(entry, proposal, None, None,
                          apply_mode=apply_mode, threshold=CROSS_ENCODER_THRESHOLD)

        proposed_title = (proposal.get("proposed_title") or "").strip()
        same = proposal.get("same_as_current") or (
            entry.get("resolved_title")
            and _norm_title(proposed_title) == _norm_title(entry.get("resolved_title"))
        )
        if not proposed_title or same:
            return decide(entry, proposal, None, None,
                          apply_mode=apply_mode, threshold=CROSS_ENCODER_THRESHOLD)

        verified = verify_proposal(skill_name, entry.get("category") or "", proposed_title)
        if verified.get("error"):
            return decide(entry, proposal, verified, None,
                          apply_mode=apply_mode, threshold=CROSS_ENCODER_THRESHOLD)

        audit = audit_proposal(skill_name, verified)
        return decide(entry, proposal, verified, audit,
                      apply_mode=apply_mode, threshold=CROSS_ENCODER_THRESHOLD)

    except AuditUnavailable:
        # An outage or exhausted attempts on THIS item. Not a verdict on anything.
        logger.warning("Second pass could not reach Gemini for %r. Will retry.", skill_name)
        return _result("audit_unavailable")


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
      apply    additionally repoint cards at better pages and write drafted
               definitions, still approving nothing
      auto     additionally approve Tier A re-resolves outright

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

    for skill_name, entry in candidates:
        try:
            result = process_one(skill_name, entry, apply_mode=apply_mode)
        except DailyQuotaExhausted:
            # The entry is untouched: process_one wrote nothing, and the write below
            # never ran. Re-running tomorrow picks it up unchanged.
            #
            # The exception's own message says remaining work went to the backlog,
            # which is true of ingestion and not of this. Nothing is queued here
            # because nothing needs to be, so the message is not repeated.
            logger.error(
                "All Gemini keys have hit their daily quota. Second pass stopping after "
                "%d item(s); %d were never reached, starting at %r. Nothing was written "
                "for them and re-running tomorrow picks them up unchanged.",
                summary["processed"], len(candidates) - summary["processed"], skill_name,
            )
            summary["quota_exhausted"] = True
            break

        outcome = result["outcome"]
        summary["outcomes"][outcome] = summary["outcomes"].get(outcome, 0) + 1
        summary["processed"] += 1
        summary["remaining"] = len(candidates) - summary["processed"]

        if dry_run:
            logger.info("[dry run] %r -> %s", skill_name, outcome)
            continue

        ok, code = review_actions.apply_second_pass_result(skill_name, result)
        if not ok:
            logger.warning("Could not record second-pass result for %r: %s.", skill_name, code)

    logger.info(
        "Second pass finished: %d processed, %d left. %s",
        summary["processed"], summary["remaining"], summary["outcomes"],
    )
    return summary

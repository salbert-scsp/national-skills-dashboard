"""
Tests for sending a model-written definition back to be rewritten.

    python3.11 -m pytest test_redraft.py -q

Pure in-memory: every case calls _apply_reject_draft, which mutates a dict and writes
nothing, so none of this touches the real store.

The point of the feature is that the DRAFTING RULES changed, not that one draft came out
badly. Definitions written before agentic_source_check.DEFINITION_SPEC run about 150
characters; the spec now asks for 200-400 with the specific technical category, vendor
lineage and concrete use cases. Every old draft is thin, and this is how they get asked
again.
"""

import pytest

from json_store import STATUS_APPROVED, STATUS_PENDING
from review_actions import (
    DRAFT_REJECTED_OUTCOME,
    MODEL_AUTHORED_SOURCE_NAME,
    _apply_reject_draft,
    onet_boilerplate,
)
from second_pass import MAX_SECOND_PASS_ATTEMPTS, TERMINAL_OUTCOMES

OLD_DRAFT = (
    "DynaSCAPE Design is a computer-aided design software developed by DynaSCAPE "
    "Software specifically for landscape architects and professional designers."
)


def waiting_entry():
    """A card offering a draft nobody has accepted yet."""
    return {
        "status": STATUS_PENDING,
        "category": "Computer aided design CAD software",
        "gate_reason": "no_candidate_found",
        "wikipedia_summary": "DynaSCAPE Design is a hot technology asset categorized under CAD.",
        "best_source_name": None,
        "second_pass": {
            "outcome": "drafted",
            "terminal": True,
            "attempts": 1,
            "drafted_definition": OLD_DRAFT,
        },
    }


def accepted_entry():
    """A card whose text IS the draft, because a reviewer accepted it."""
    return {
        "status": STATUS_PENDING,
        "category": "Computer aided design CAD software",
        "gate_reason": "no_candidate_found",
        "wikipedia_summary": OLD_DRAFT,
        "best_source_name": MODEL_AUTHORED_SOURCE_NAME,
        "resolved_title": None,
        "is_credible": None,
        "cross_score": None,
        "second_pass": {
            "outcome": "model_authored",
            "terminal": True,
            "attempts": 1,
            "drafted_definition": OLD_DRAFT,
        },
    }


# --- the outcome must stop settling the item ----------------------------------

def test_the_outcome_becomes_non_terminal():
    """
    This is the mechanism. second_pass skips anything whose outcome is in
    TERMINAL_OUTCOMES without spending a call, so the redraft only happens if the
    outcome leaves that set.
    """
    master = {"DynaSCAPE": waiting_entry()}
    ok, code = _apply_reject_draft(master, "DynaSCAPE")
    assert (ok, code) == (True, "ok")

    block = master["DynaSCAPE"]["second_pass"]
    assert block["outcome"] == DRAFT_REJECTED_OUTCOME
    assert block["outcome"] not in TERMINAL_OUTCOMES
    assert block["terminal"] is False


def test_attempts_are_reset_not_left_at_the_cap():
    """
    _already_settled ALSO skips on attempts >= MAX_SECOND_PASS_ATTEMPTS, so clearing the
    outcome alone would silently refuse a redraft on an item that had already been tried.
    """
    master = {"DynaSCAPE": waiting_entry()}
    master["DynaSCAPE"]["second_pass"]["attempts"] = MAX_SECOND_PASS_ATTEMPTS

    _apply_reject_draft(master, "DynaSCAPE")
    assert master["DynaSCAPE"]["second_pass"]["attempts"] == 0


def test_the_item_is_selectable_again_by_the_second_pass():
    """The end-to-end property, asserted through second_pass's own selector."""
    from second_pass import AI_AWAITING_REVIEW, select_candidates

    master = {"DynaSCAPE": waiting_entry()}
    # A drafted item carries this tag in the store, and the TAG is what silences it now
    # -- the outcome alone no longer does. Set explicitly so the test proves rejecting
    # clears it rather than relying on an outcome list that no longer gates selection.
    master["DynaSCAPE"]["ai_status"] = AI_AWAITING_REVIEW
    assert select_candidates(master) == [], "a tagged draft must not be selectable"

    _apply_reject_draft(master, "DynaSCAPE")
    assert [name for name, _ in select_candidates(master)] == ["DynaSCAPE"]


# --- the offer is withdrawn ---------------------------------------------------

def test_the_card_stops_offering_the_draft():
    from dashboardtables import draft_ready

    master = {"DynaSCAPE": waiting_entry()}
    assert draft_ready(master["DynaSCAPE"]) is True

    _apply_reject_draft(master, "DynaSCAPE")
    assert draft_ready(master["DynaSCAPE"]) is False


def test_the_rejected_text_is_kept_for_comparison():
    """
    The second_pass block is an audit trail, and comparing the redraft against what it
    replaced is the main way anyone judges whether the new spec helped.
    """
    master = {"DynaSCAPE": waiting_entry()}
    _apply_reject_draft(master, "DynaSCAPE")

    block = master["DynaSCAPE"]["second_pass"]
    assert block["rejected_definition"] == OLD_DRAFT
    assert not block["drafted_definition"]


# --- an accepted draft is reverted --------------------------------------------

def test_an_accepted_draft_is_reverted_to_boilerplate():
    """
    The card's text IS the draft by then, and the encyclopedia text it overwrote is
    gone, so boilerplate is the honest resting state.
    """
    master = {"DynaSCAPE": accepted_entry()}
    ok, _ = _apply_reject_draft(master, "DynaSCAPE")
    assert ok

    entry = master["DynaSCAPE"]
    assert entry["wikipedia_summary"] == onet_boilerplate(
        "DynaSCAPE", "Computer aided design CAD software"
    )
    assert entry["best_source_name"] is None
    assert OLD_DRAFT not in entry["wikipedia_summary"]


def test_the_reverted_text_is_never_empty():
    """
    An entry with empty text drops out of build_review_rows, which would delete the card
    rather than requeue it -- the opposite of the intent.
    """
    master = {"DynaSCAPE": accepted_entry()}
    _apply_reject_draft(master, "DynaSCAPE")
    assert master["DynaSCAPE"]["wikipedia_summary"].strip()


def test_an_accepted_draft_still_renders_a_card():
    from dashboardtables import build_review_rows

    master = {"DynaSCAPE": accepted_entry()}
    _apply_reject_draft(master, "DynaSCAPE")
    rows = build_review_rows(master, STATUS_PENDING)
    assert [row["skill_name"] for row in rows] == ["DynaSCAPE"]


def test_a_waiting_draft_leaves_the_card_text_alone():
    """Only an ACCEPTED draft is on the card, so only that one gets reverted."""
    master = {"DynaSCAPE": waiting_entry()}
    before = master["DynaSCAPE"]["wikipedia_summary"]
    _apply_reject_draft(master, "DynaSCAPE")
    assert master["DynaSCAPE"]["wikipedia_summary"] == before


# --- what must not change -----------------------------------------------------

def test_the_skill_stays_pending():
    """Rejecting a DRAFT is not rejecting the SKILL."""
    master = {"DynaSCAPE": waiting_entry()}
    _apply_reject_draft(master, "DynaSCAPE")
    assert master["DynaSCAPE"]["status"] == STATUS_PENDING


def test_the_gate_reason_is_left_alone():
    """It is already eligible; that is how the item came to be drafted in the first place."""
    master = {"DynaSCAPE": waiting_entry()}
    _apply_reject_draft(master, "DynaSCAPE")
    assert master["DynaSCAPE"]["gate_reason"] == "no_candidate_found"


# --- refusals -----------------------------------------------------------------

def test_an_unknown_skill_is_refused():
    assert _apply_reject_draft({}, "Nope") == (False, "not_found")


def test_a_decided_skill_is_refused():
    master = {"DynaSCAPE": waiting_entry()}
    master["DynaSCAPE"]["status"] = STATUS_APPROVED
    assert _apply_reject_draft(master, "DynaSCAPE") == (False, "not_pending")


def test_a_card_with_no_draft_is_refused():
    master = {"DynaSCAPE": {"status": STATUS_PENDING, "second_pass": {"outcome": "suggested"}}}
    assert _apply_reject_draft(master, "DynaSCAPE") == (False, "no_draft")


def test_a_card_with_no_second_pass_block_is_refused():
    master = {"DynaSCAPE": {"status": STATUS_PENDING}}
    assert _apply_reject_draft(master, "DynaSCAPE") == (False, "no_draft")


def test_rejecting_twice_is_refused_rather_than_looping():
    master = {"DynaSCAPE": waiting_entry()}
    assert _apply_reject_draft(master, "DynaSCAPE")[0] is True
    assert _apply_reject_draft(master, "DynaSCAPE") == (False, "no_draft")


# --- wiring -------------------------------------------------------------------

def test_reject_draft_is_a_final_action():
    """So it cannot be staged alongside an approve on the same card."""
    from review_actions import BATCH_ACTIONS, FINAL_ACTIONS, SOURCE_ACTIONS

    assert "reject-draft" in FINAL_ACTIONS
    assert "reject-draft" not in SOURCE_ACTIONS
    assert "reject-draft" in BATCH_ACTIONS


def test_every_refusal_code_has_a_sentence_for_the_reviewer():
    from main import COMMIT_ERRORS

    for code in ("no_draft", "not_pending", "not_found"):
        assert COMMIT_ERRORS.get(code), f"{code} has no reviewer-facing sentence"


def test_the_batch_path_applies_it():
    import unittest.mock as mk

    import review_actions as ra

    master = {"DynaSCAPE": waiting_entry()}
    with mk.patch.object(ra, "_load", lambda: (master, [])), \
         mk.patch.object(ra, "save_master", lambda x: None), \
         mk.patch.object(ra, "save_timeseries", lambda x: None), \
         mk.patch.object(ra, "load_master", lambda: master):
        applied, items, failures = ra.apply_review_batch(
            [{"skill": "DynaSCAPE", "action": "reject-draft"}]
        )

    assert applied == 1
    assert failures == []
    assert items[0]["status"] == STATUS_PENDING
    assert master["DynaSCAPE"]["second_pass"]["outcome"] == DRAFT_REJECTED_OUTCOME

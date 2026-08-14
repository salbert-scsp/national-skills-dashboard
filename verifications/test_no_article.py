"""
Tests for what happens when a lookup finds no encyclopedia article.

    python3.11 -m pytest test_no_article.py -q

No network, no Gemini, no store writes: the clearing helper is exercised directly and
the integration goes through apply_second_pass_result with the store functions patched.

THE DEFECT: no_article was the one terminal outcome that contradicted its own card. 95
entries asserted "a lookup found no encyclopedia article for this product" while showing
another article's text as the definition -- IEA Software Emerald citing "Foreign relations
of India", K2 Business Process Automation citing "K9 Thunder", a Korean howitzer. Because
the outcome is terminal, nothing would ever revisit them.
"""

import pytest

from json_store import STATUS_APPROVED, STATUS_PENDING
from review_actions import (
    NO_ARTICLE_OUTCOMES,
    _clear_contradicted_page,
    onet_boilerplate,
)

WRONG_PAGE_TEXT = (
    "India, officially the Republic of India, has full diplomatic relations with 201 "
    "states, and maintains a network of embassies and high commissions."
)


def cited_entry(gate_reason="below_relevance_threshold"):
    """A card citing a page that has nothing to do with the skill."""
    return {
        "status": STATUS_PENDING,
        "category": "Enterprise resource planning ERP software",
        "gate_reason": gate_reason,
        "resolved_title": "Foreign relations of India",
        "reference_url": None,
        "wikipedia_summary": WRONG_PAGE_TEXT,
        "cross_score": 0.41,
        "is_credible": True,
        "best_source_name": "Wikipedia",
    }


# --- the clearing -------------------------------------------------------------

def test_the_contradicted_page_is_cleared():
    entry = cited_entry()
    assert _clear_contradicted_page(entry, "IEA Software Emerald") is True

    assert entry["resolved_title"] is None
    assert entry["reference_url"] is None
    assert entry["cross_score"] is None
    assert entry["is_credible"] is None
    assert entry["best_source_name"] is None
    assert "Republic of India" not in entry["wikipedia_summary"]


def test_the_card_falls_back_to_boilerplate():
    entry = cited_entry()
    _clear_contradicted_page(entry, "IEA Software Emerald")
    assert entry["wikipedia_summary"] == onet_boilerplate(
        "IEA Software Emerald", "Enterprise resource planning ERP software"
    )


def test_the_gate_reason_becomes_the_bucket_that_gets_drafted():
    """
    no_candidate_found is in second_pass.ELIGIBLE_GATE_REASONS, so the entry lands where
    a definition will actually be written for it.
    """
    from second_pass import ELIGIBLE_GATE_REASONS

    entry = cited_entry()
    _clear_contradicted_page(entry, "IEA Software Emerald")
    assert entry["gate_reason"] == "no_candidate_found"
    assert entry["gate_reason"] in ELIGIBLE_GATE_REASONS


def test_the_page_goes_even_when_it_scored_well():
    """
    Not only below RESOLUTION_FLOOR. A model that looked at the product and reported no
    article contradicts that page directly, which outweighs the similarity number that
    let it through.
    """
    from scraping import RESOLUTION_FLOOR

    entry = cited_entry()
    entry["cross_score"] = 0.87
    assert entry["cross_score"] > RESOLUTION_FLOOR

    _clear_contradicted_page(entry, "IBM Content Manager")
    assert entry["resolved_title"] is None


def test_the_skill_stays_pending_and_keeps_a_card():
    """Boilerplate is non-empty, which is what build_review_rows requires."""
    from dashboardtables import build_review_rows

    entry = cited_entry()
    _clear_contradicted_page(entry, "IEA Software Emerald")

    assert entry["status"] == STATUS_PENDING
    assert entry["wikipedia_summary"].strip()
    rows = build_review_rows({"IEA Software Emerald": entry}, STATUS_PENDING)
    assert [row["skill_name"] for row in rows] == ["IEA Software Emerald"]


def test_clearing_an_entry_with_no_page_is_harmless():
    entry = cited_entry()
    entry["resolved_title"] = None
    entry["best_source_name"] = None
    assert _clear_contradicted_page(entry, "Some Skill") is True
    assert entry["gate_reason"] == "no_candidate_found"


# --- the human exception ------------------------------------------------------

def test_a_reviewer_supplied_page_is_never_cleared():
    """
    A person asserting a page is the thing the automation is guessing at -- the same
    principle that makes the cross-encoder score recorded rather than enforced on that
    path. A model finding no article is not grounds to overwrite them.
    """
    entry = cited_entry(gate_reason="reviewer_remediated")
    before = dict(entry)

    assert _clear_contradicted_page(entry, "PySpark") is False
    assert entry["resolved_title"] == before["resolved_title"]
    assert entry["wikipedia_summary"] == before["wikipedia_summary"]
    assert entry["gate_reason"] == "reviewer_remediated"


def test_a_machine_remediated_page_is_cleared():
    """Only a HUMAN gets the exemption; the second pass has no such standing."""
    entry = cited_entry(gate_reason="machine_remediated")
    assert _clear_contradicted_page(entry, "Some Skill") is True
    assert entry["resolved_title"] is None


# --- the split ----------------------------------------------------------------

def test_declined_to_define_is_terminal_and_distinct():
    from second_pass import TERMINAL_OUTCOMES

    assert "declined_to_define" in TERMINAL_OUTCOMES
    assert "no_article" in TERMINAL_OUTCOMES
    assert "declined_to_define" != "no_article"


def test_both_no_article_outcomes_clear_the_page():
    assert set(NO_ARTICLE_OUTCOMES) == {"no_article", "declined_to_define"}


def test_the_two_modules_agree_on_the_outcome_list():
    """A drift here would clear the page for one outcome and not the other."""
    import review_actions
    import second_pass

    assert review_actions.NO_ARTICLE_OUTCOMES == second_pass.NO_ARTICLE_OUTCOMES


def test_declining_to_define_is_reported_separately():
    """
    The model looked, found no article, and would not describe the product either. That
    asks something different of the reviewer than "no page, here is a definition".

    draft_for_entry is stubbed rather than called: an empty definition on the proposal is
    exactly the condition that sends _draft_or_settle to the fallback request, and a test
    must not spend quota to reach a branch.
    """
    import unittest.mock as mk

    import second_pass as sp

    with mk.patch.object(
        sp, "draft_for_entry",
        lambda name, entry: {"definition": "", "is_software_product": True},
    ):
        result = sp._draft_or_settle(
            "Unknown Product",
            {"category": "software", "wikipedia_summary": "x"},
            {"definition": "", "rationale": "never heard of it"},
        )

    assert result["outcome"] == "declined_to_define"
    assert result["suggested_action"] == "reject"


def test_a_definition_on_the_proposal_costs_no_extra_call():
    """The normal path: the proposal already carries the definition, so nothing is asked."""
    import unittest.mock as mk

    import second_pass as sp

    def explode(name, entry):
        raise AssertionError("draft_for_entry must not be called when a definition exists")

    with mk.patch.object(sp, "draft_for_entry", explode):
        result = sp._draft_or_settle(
            "Known Product",
            {"category": "software", "wikipedia_summary": "x"},
            {"definition": "A real definition of the product.", "is_software_product": True},
        )

    assert result["outcome"] == "drafted"


# --- integration through the single write point -------------------------------

@pytest.mark.parametrize("outcome", ["no_article", "declined_to_define"])
def test_apply_second_pass_result_clears_the_page(outcome):
    import unittest.mock as mk

    import review_actions as ra

    master = {"IEA Software Emerald": cited_entry()}
    with mk.patch.object(ra, "_load", lambda: (master, [])), \
         mk.patch.object(ra, "save_master", lambda x: None), \
         mk.patch.object(ra, "save_timeseries", lambda x: None):
        ok, code = ra.apply_second_pass_result(
            "IEA Software Emerald",
            {"outcome": outcome, "rationale": "no article exists", "suggested_action": "reject"},
        )

    assert (ok, code) == (True, "ok")
    entry = master["IEA Software Emerald"]
    assert entry["resolved_title"] is None
    assert entry["status"] == STATUS_PENDING
    assert entry["second_pass"]["outcome"] == outcome


def test_an_approved_skill_is_refused_by_the_write_point():
    """The second pass may only settle a card already in the queue."""
    import unittest.mock as mk

    import review_actions as ra

    entry = cited_entry()
    entry["status"] = STATUS_APPROVED
    master = {"X": entry}
    with mk.patch.object(ra, "_load", lambda: (master, [])), \
         mk.patch.object(ra, "save_master", lambda x: None):
        ok, code = ra.apply_second_pass_result("X", {"outcome": "no_article"})

    assert (ok, code) == (False, "not_pending")
    assert entry["resolved_title"] == "Foreign relations of India"


# --- reopening ----------------------------------------------------------------

def test_reopening_makes_an_item_selectable_again():
    from redraft_definitions import find_no_article, reopen
    from second_pass import select_candidates

    entry = cited_entry(gate_reason="no_candidate_found")
    entry["second_pass"] = {"outcome": "no_article", "terminal": True, "attempts": 1}
    # The TAG is what silences an entry now, not the outcome. Without it this item would
    # already be selectable, and the test would prove nothing about reopening.
    entry["ai_status"] = "cannot_determine"
    master = {"IEA Software Emerald": entry}

    assert select_candidates(master) == [], "a tagged item must not be selectable"
    assert [name for name, _ in find_no_article(master)] == ["IEA Software Emerald"]

    reopen("IEA Software Emerald", entry)
    assert entry["second_pass"]["attempts"] == 0
    assert [name for name, _ in select_candidates(master)] == ["IEA Software Emerald"]


def test_declined_to_define_is_not_reopened():
    """
    The model was asked and said it does not know the product. Asking the same question
    again is exactly the waste the terminal rule exists to prevent.
    """
    from redraft_definitions import find_no_article

    entry = cited_entry()
    entry["second_pass"] = {"outcome": "declined_to_define", "terminal": True, "attempts": 1}
    assert find_no_article({"X": entry}) == []

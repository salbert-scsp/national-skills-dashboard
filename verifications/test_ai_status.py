"""
Tests for the explicit ai_status tag that decides what the second pass skips.

    python3.11 -m pytest test_ai_status.py -q

No network, no store writes.

THE RULE: the second pass skips an entry if and only if that entry carries an ai_status
tag. It used to skip anything whose outcome was in TERMINAL_OUTCOMES, which was invisible
-- a queue the automation had given up on looked identical to one it had not reached, and
"why is the second pass instant" could only be answered by reading the store. Now every
skip corresponds to a badge on a card, and an untagged entry is always tried again.
"""

import pytest

from json_store import STATUS_PENDING
from second_pass import (
    AI_AWAITING_REVIEW,
    AI_CANNOT_DETERMINE,
    MAX_SECOND_PASS_ATTEMPTS,
    _already_settled,
    ai_status_for,
    select_candidates,
)


def entry(ai_status=None, outcome=None, attempts=1, gate_reason="no_candidate_found"):
    record = {
        "status": STATUS_PENDING,
        "category": "software",
        "gate_reason": gate_reason,
        "wikipedia_summary": "Some text a human could act on.",
    }
    if outcome is not None:
        record["second_pass"] = {"outcome": outcome, "attempts": attempts}
    if ai_status is not None:
        record["ai_status"] = ai_status
    return record


# --- the skip rule ------------------------------------------------------------

def test_only_the_tag_silences_an_entry():
    assert _already_settled(entry(ai_status=AI_CANNOT_DETERMINE)) is True
    assert _already_settled(entry(ai_status=AI_AWAITING_REVIEW)) is True
    assert _already_settled(entry()) is False


def test_a_terminal_outcome_alone_no_longer_skips():
    """
    The inversion. Every one of these outcomes used to silence an entry on its own; now
    only the tag does, so an entry settled under the old rule is tried again.
    """
    for outcome in ("no_article", "drafted", "not_a_product", "suggested", "auto_approved"):
        assert _already_settled(entry(outcome=outcome)) is False, outcome


def test_the_attempt_cap_alone_no_longer_skips():
    """
    Enforced where the outcome is recorded instead, by writing a tag, so an item that
    keeps failing ends up visible rather than quietly dropping out of every run.
    """
    assert _already_settled(entry(outcome="proposal_failed", attempts=99)) is False


def test_an_untagged_entry_is_selected():
    master = {"A": entry(outcome="no_article")}
    assert [name for name, _ in select_candidates(master)] == ["A"]


def test_a_tagged_entry_is_not_selected():
    master = {"A": entry(ai_status=AI_CANNOT_DETERMINE, outcome="no_article")}
    assert select_candidates(master) == []


# --- which outcome earns which tag --------------------------------------------

@pytest.mark.parametrize("outcome", ["no_article", "declined_to_define", "not_a_product",
                                     "unresolvable"])
def test_giving_up_earns_the_cannot_determine_tag(outcome):
    assert ai_status_for(outcome) == AI_CANNOT_DETERMINE


@pytest.mark.parametrize("outcome", ["drafted", "model_authored", "suggested",
                                     "suggested_strong", "suggested_weak",
                                     "confirmed_original", "no_distinct_article"])
def test_an_answer_earns_the_awaiting_review_tag(outcome):
    """
    Tagged so it is NOT re-asked: the model has produced something and re-asking would
    spend a request to overwrite a proposal the reviewer has not looked at yet.
    """
    assert ai_status_for(outcome) == AI_AWAITING_REVIEW


@pytest.mark.parametrize("outcome", ["proposal_failed", "draft_failed",
                                     "audit_unavailable", "draft_rejected", "", None])
def test_a_transient_outcome_earns_no_tag(outcome):
    """Absence of a tag IS the retry mechanism."""
    assert ai_status_for(outcome) is None


# --- writing the tag ----------------------------------------------------------

def test_the_tag_is_written_when_the_outcome_earns_one():
    from review_actions import _tag_ai_status

    record = entry(outcome="no_article")
    _tag_ai_status(record, "no_article")
    assert record["ai_status"] == AI_CANNOT_DETERMINE
    assert record["ai_status_detail"] == "no_article"


def test_a_transient_outcome_removes_a_stale_tag():
    """A fresh finding supersedes an older decision to stop."""
    from review_actions import _tag_ai_status

    record = entry(ai_status=AI_CANNOT_DETERMINE, outcome="proposal_failed")
    _tag_ai_status(record, "proposal_failed")
    assert "ai_status" not in record
    assert "ai_status_detail" not in record


def test_hitting_the_attempt_cap_tags_rather_than_silently_dropping():
    from review_actions import _tag_ai_status

    record = entry(outcome="proposal_failed", attempts=MAX_SECOND_PASS_ATTEMPTS)
    _tag_ai_status(record, "proposal_failed")
    assert record["ai_status"] == AI_CANNOT_DETERMINE
    assert "attempts" in record["ai_status_detail"]


def test_below_the_cap_a_failure_stays_untagged():
    from review_actions import _tag_ai_status

    record = entry(outcome="proposal_failed", attempts=MAX_SECOND_PASS_ATTEMPTS - 1)
    _tag_ai_status(record, "proposal_failed")
    assert "ai_status" not in record


# --- a human clears the tag ---------------------------------------------------

def test_rejecting_a_draft_clears_the_tag():
    """Asking for a rewrite is a decision to let the model look again."""
    from review_actions import _apply_reject_draft

    record = entry(ai_status=AI_AWAITING_REVIEW, outcome="drafted")
    record["second_pass"]["drafted_definition"] = "A thin old definition."
    master = {"X": record}

    assert _apply_reject_draft(master, "X")[0] is True
    assert "ai_status" not in record
    assert [name for name, _ in select_candidates(master)] == ["X"]


def test_reopening_clears_the_tag():
    from redraft_definitions import reopen

    record = entry(ai_status=AI_CANNOT_DETERMINE, outcome="no_article")
    reopen("X", record)
    assert "ai_status" not in record
    assert _already_settled(record) is False


# --- what a reviewer sees -----------------------------------------------------

def test_the_tag_reaches_the_review_row():
    from dashboardtables import build_review_rows

    master = {"X": entry(ai_status=AI_CANNOT_DETERMINE, outcome="no_article")}
    master["X"]["ai_status_detail"] = "no_article"
    rows = build_review_rows(master, STATUS_PENDING)
    assert rows[0]["ai_status"] == AI_CANNOT_DETERMINE
    assert rows[0]["ai_status_detail"] == "no_article"


def test_given_up_items_sort_above_the_untouched_rest():
    """
    They are the only band that will never move on its own, so burying them among items
    still awaiting automation would hide the work a person has to do.
    """
    from dashboardtables import PRIORITY_NEEDS_HUMAN, PRIORITY_REST, review_priority

    assert review_priority(entry(ai_status=AI_CANNOT_DETERMINE)) == PRIORITY_NEEDS_HUMAN
    assert review_priority(entry()) == PRIORITY_REST
    assert PRIORITY_NEEDS_HUMAN < PRIORITY_REST


def test_awaiting_review_does_not_claim_a_human_is_needed():
    """Only giving up earns the badge; an answer waiting on a click is not stuck."""
    from dashboardtables import PRIORITY_NEEDS_HUMAN, review_priority

    assert review_priority(entry(ai_status=AI_AWAITING_REVIEW)) != PRIORITY_NEEDS_HUMAN


# --- the backfill -------------------------------------------------------------

def test_backfill_leaves_the_no_article_bucket_untagged_for_one_retry():
    from backfill_ai_status import plan

    master = {
        "gave_up": entry(outcome="no_article"),
        "declined": entry(outcome="declined_to_define"),
        "has_answer": entry(outcome="drafted"),
        "not_a_product": entry(outcome="not_a_product"),
    }
    grouped = plan(master)

    assert [n for n, _ in grouped["retry"]] == ["declined", "gave_up"]
    assert [n for n, _ in grouped["awaiting"]] == ["has_answer"]
    assert [n for n, _ in grouped["cannot"]] == ["not_a_product"]


def test_backfill_does_not_retag_an_already_tagged_entry():
    from backfill_ai_status import plan

    master = {"X": entry(ai_status=AI_AWAITING_REVIEW, outcome="drafted")}
    grouped = plan(master)
    assert [n for n, _ in grouped["already_tagged"]] == ["X"]
    assert grouped["awaiting"] == []

"""
Tests for auto-approving a written definition on two agreeing calls.

    python3.11 -m pytest test_verified_definitions.py -q

No network: verify_definitions_batch is stubbed everywhere. The live calibration that
justified this change is recorded in the plan, not re-run here -- it agreed with 23 of 24
human approvals and caught one the human had missed.

WHAT THIS GUARDS: this is the only path on which text with no external source reaches the
dashboard without a person reading it. Every test below is about the conditions under
which that is allowed, and the failure modes that must keep it staged instead.
"""

import unittest.mock as mk

import pytest

import second_pass as sp
from json_store import STATUS_APPROVED, STATUS_PENDING
from review_actions import (
    MODEL_AUTHORED_SOURCE_NAME,
    MODEL_AUTHORED_VERIFIED_GATE_REASON,
)

GOOD = (
    "GeoPak Bridge is a civil engineering software add-on developed by Bentley Systems "
    "used for modeling, designing and analyzing bridge structures."
)


def drafted(skill="Bentley GeoPak Bridge", definition=GOOD):
    return {skill: {"outcome": "drafted", "drafted_definition": definition}}


def entries(skill="Bentley GeoPak Bridge"):
    return {skill: {"category": "CAD software", "occupations": [], "onet_titles": []}}


def verdict(subject=True, invented=False, category=False, confidence=0.9, problem=None):
    return {
        "describes_this_skill": subject,
        "contains_invented_specifics": invented,
        "is_generic_category": category,
        "confidence": confidence,
        "problem": problem,
    }


def run(results, verdicts, raises=None):
    checker = mk.Mock()
    if raises is not None:
        checker.verify_definitions_batch.side_effect = raises
    else:
        checker.verify_definitions_batch.return_value = verdicts
    with mk.patch.object(sp, "get_source_checker", lambda *_role: checker):
        return sp.verify_drafts(results, entries())


# --- the one condition that permits auto-approval -----------------------------

def test_agreement_promotes_the_draft():
    results = run(drafted(), {"Bentley GeoPak Bridge": verdict()})
    row = results["Bentley GeoPak Bridge"]
    assert row["outcome"] == "model_authored_verified"
    assert row["summary"] == GOOD
    assert row["verified_by"] == "definition_check"


@pytest.mark.parametrize("bad", [
    verdict(subject=False, problem="describes a video game"),
    verdict(invented=True, problem="asserts a version number and a customer count"),
    verdict(subject=False, invented=True),
])
def test_any_disagreement_keeps_it_staged(bad):
    results = run(drafted(), {"Bentley GeoPak Bridge": bad})
    assert results["Bentley GeoPak Bridge"]["outcome"] == "drafted"


def test_the_stated_problem_is_recorded_for_the_reviewer():
    """A bare flag tells a reviewer nothing about what to do next."""
    results = run(drafted(), {"Bentley GeoPak Bridge": verdict(
        subject=False, problem="describes a video game, not this product")})
    row = results["Bentley GeoPak Bridge"]
    assert row["verifier_problem"] == "describes a video game, not this product"
    assert row["detail"] == "verifier_wrong_subject"


def test_invented_specifics_are_reported_distinctly():
    results = run(drafted(), {"Bentley GeoPak Bridge": verdict(invented=True)})
    assert results["Bentley GeoPak Bridge"]["detail"] == "verifier_invented_specifics"


# --- silence is never agreement -----------------------------------------------

def test_a_missing_verdict_keeps_it_staged():
    """
    The failure that matters most. A truncated response or a quota wall must never read
    as a pass, or the two-opinion bar collapses into one whenever the second call fails.
    """
    results = run(drafted(), {})
    assert results["Bentley GeoPak Bridge"]["outcome"] == "drafted"
    assert "verified_by" not in results["Bentley GeoPak Bridge"]


def test_an_unreachable_verifier_keeps_the_whole_chunk_staged():
    from agentic_source_check import AuditUnavailable

    results = run(drafted(), None, raises=AuditUnavailable("down"))
    assert results["Bentley GeoPak Bridge"]["outcome"] == "drafted"


def test_a_verdict_for_an_unrelated_skill_does_not_promote_anything():
    results = run(drafted(), {"Some Other Skill": verdict()})
    assert results["Bentley GeoPak Bridge"]["outcome"] == "drafted"


# --- scope --------------------------------------------------------------------

def test_only_drafted_items_are_verified():
    """A page-backed item already has two opinions: the cross-encoder and the audit."""
    results = {
        "PageBacked": {"outcome": "auto_approved", "summary": "from an article"},
        "Suggested": {"outcome": "suggested", "proposed_title": "Something"},
    }
    checker = mk.Mock()
    with mk.patch.object(sp, "get_source_checker", lambda *_role: checker):
        out = sp.verify_drafts(results, {})

    checker.verify_definitions_batch.assert_not_called()
    assert out["PageBacked"]["outcome"] == "auto_approved"
    assert out["Suggested"]["outcome"] == "suggested"


def test_an_empty_draft_is_not_sent_for_verification():
    results = {"X": {"outcome": "drafted", "drafted_definition": "   "}}
    checker = mk.Mock()
    with mk.patch.object(sp, "get_source_checker", lambda *_role: checker):
        sp.verify_drafts(results, {})
    checker.verify_definitions_batch.assert_not_called()


def test_a_category_verdict_still_promotes():
    """A class-level definition is a correct answer, not a weak one."""
    results = run(drafted("Tariff databases", "Tariff databases are reference systems..."),
                  {"Tariff databases": verdict(category=True)})
    row = results["Tariff databases"]
    assert row["outcome"] == "model_authored_verified"
    assert row["is_generic_category"] is True


# --- the write -----------------------------------------------------------------

def test_the_write_approves_scores_and_marks_provenance():
    import review_actions as ra

    entry = {
        "skill_name": "Bentley GeoPak Bridge",
        "status": STATUS_PENDING,
        "category": "CAD software",
        "wikipedia_summary": "old boilerplate",
        "gate_reason": "no_candidate_found",
        "onet_codes": [], "onet_titles": [], "occupations": [],
    }
    master, timeseries = {"Bentley GeoPak Bridge": entry}, []

    with mk.patch.object(ra, "_load", lambda: (master, timeseries)), \
         mk.patch.object(ra, "save_master", lambda x: None), \
         mk.patch.object(ra, "save_timeseries", lambda x: None):
        ok, code = ra.apply_second_pass_result("Bentley GeoPak Bridge", {
            "outcome": "model_authored_verified",
            "summary": GOOD,
            "source_name": MODEL_AUTHORED_SOURCE_NAME,
            "verifier_confidence": 0.92,
        })

    assert (ok, code) == (True, "ok")
    assert entry["status"] == STATUS_APPROVED
    assert entry["wikipedia_summary"] == GOOD
    # Provenance must stay loud: the text has no source behind it either way.
    assert entry["best_source_name"] == MODEL_AUTHORED_SOURCE_NAME
    assert entry["gate_reason"] == MODEL_AUTHORED_VERIFIED_GATE_REASON
    assert entry["is_credible"] is None, "no audit graded this text"
    assert len(timeseries) == 1, "an approved skill must be scored"


def test_it_is_distinguishable_from_a_reviewer_accepting_one():
    """
    Both put unsourced text on the dashboard and only one had a person read it. Telling
    them apart is what makes 'how much unsourced text is live, and who let it through' a
    question the store can answer.
    """
    assert MODEL_AUTHORED_VERIFIED_GATE_REASON != "reviewer_remediated"

    from main import GATE_REASON_LABELS

    label = GATE_REASON_LABELS[MODEL_AUTHORED_VERIFIED_GATE_REASON]
    assert "no source" in label.lower()


def test_the_outcome_is_registered_everywhere_it_has_to_be():
    assert "model_authored_verified" in sp.TERMINAL_OUTCOMES
    assert sp.ai_status_for("model_authored_verified") == sp.AI_AWAITING_REVIEW


# --- the probe fixes ------------------------------------------------------------

def test_a_vendor_prefix_is_dropped_for_the_probe():
    """PTC Windchill has no page; Windchill (software) resolves at 1.000."""
    from scraping import _disambiguated_seeds

    seeds = _disambiguated_seeds("PTC Windchill", "software", "PTC Windchill")
    assert "Windchill (software)" in seeds


def test_the_last_token_is_probed_with_the_language_suffix():
    """
    NOISE_CLEANER strips both "Tool" and "language" from "Tool command language Tcl",
    leaving the unmatchable "command Tcl". The raw name is the only place the language
    hint survives, and Tcl (programming language) resolves at 0.999.
    """
    from scraping import _disambiguated_seeds

    seeds = _disambiguated_seeds("command Tcl", "software", "Tool command language Tcl")
    assert "Tcl (programming language)" in seeds


def test_a_single_word_name_gets_no_extra_probes():
    from scraping import _disambiguated_seeds

    assert _disambiguated_seeds("Photoshop", "software", "Photoshop") == []


def test_seeds_are_deduplicated():
    from scraping import _disambiguated_seeds

    seeds = _disambiguated_seeds("PTC Windchill", "software", "PTC Windchill")
    assert len(seeds) == len(set(seeds))

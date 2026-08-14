"""
Tests for merging one queued skill into another as a duplicate.

    python3.11 -m pytest test_duplicates.py -q

Pure in-memory: every case calls the _apply_ form, which mutates a dict and writes
nothing, so none of this touches the real store.

The case that motivated the feature is real. O*NET carries the same product twice:

    MicroSurvey Software MicroSurvey CAD   17-1022.01
    MicroSurveyCAD                         17-1022.00, 17-3031.00

and the two resolved to two DIFFERENT wrong Wikipedia pages. Rejecting either one loses
the occupations only it carried, which is what these tests exist to prevent.
"""

import pytest

from json_store import STATUS_APPROVED, STATUS_PENDING, STATUS_REJECTED
from review_actions import _apply_mark_duplicate, _merge_occupations


def occupation(code, title="Some Occupation", hot=False):
    return {"onet_code": code, "onet_title": title, "is_hot_tech": hot}


def entry(codes, status=STATUS_PENDING, hot_codes=()):
    occupations = [occupation(code, f"Occ {code}", code in hot_codes) for code in codes]
    return {
        "status": status,
        "occupations": occupations,
        "onet_codes": list(codes),
        "onet_titles": [f"Occ {code}" for code in codes],
        "is_hot_tech_anywhere": bool(hot_codes),
    }


def store():
    """The real MicroSurvey pair, shaped as the store holds it."""
    return {
        "MicroSurvey Software MicroSurvey CAD": entry(["17-1022.01"]),
        "MicroSurveyCAD": entry(["17-1022.00", "17-3031.00"], hot_codes=["17-3031.00"]),
    }


# --- the point of the feature -------------------------------------------------

def test_occupations_are_united_not_replaced():
    master = store()
    ok, code = _apply_mark_duplicate(
        master, "MicroSurvey Software MicroSurvey CAD", "MicroSurveyCAD"
    )
    assert (ok, code) == (True, "ok")

    survivor = master["MicroSurveyCAD"]
    assert survivor["onet_codes"] == ["17-1022.00", "17-1022.01", "17-3031.00"]
    assert len(survivor["occupations"]) == 3
    # The three views of the same fact must not disagree.
    assert len(survivor["onet_titles"]) == 3
    assert [o["onet_code"] for o in survivor["occupations"]] == survivor["onet_codes"]


def test_the_duplicate_is_retired_and_says_what_it_duplicates():
    master = store()
    _apply_mark_duplicate(master, "MicroSurvey Software MicroSurvey CAD", "MicroSurveyCAD")

    duplicate = master["MicroSurvey Software MicroSurvey CAD"]
    assert duplicate["status"] == STATUS_REJECTED
    assert duplicate["duplicate_of"] == "MicroSurveyCAD"


def test_the_survivor_keeps_its_own_status():
    """Merging decides nothing about the survivor; it stays in the queue on its merits."""
    master = store()
    _apply_mark_duplicate(master, "MicroSurvey Software MicroSurvey CAD", "MicroSurveyCAD")
    assert master["MicroSurveyCAD"]["status"] == STATUS_PENDING


def test_nothing_is_lost_when_merging_the_other_way_round():
    """The reviewer picks which one survives, and either direction must be lossless."""
    master = store()
    ok, _ = _apply_mark_duplicate(
        master, "MicroSurveyCAD", "MicroSurvey Software MicroSurvey CAD"
    )
    assert ok
    assert master["MicroSurvey Software MicroSurvey CAD"]["onet_codes"] == [
        "17-1022.00", "17-1022.01", "17-3031.00"
    ]


# --- hot-tech handling --------------------------------------------------------

def test_hot_tech_is_ord_per_occupation_never_flattened():
    """
    A skill hot for one occupation and not another is the normal case the by-job view
    depends on, so each occupation has to keep its own flag.
    """
    survivor = entry(["15-1252.00"], hot_codes=["15-1252.00"])
    duplicate = entry(["15-1252.00", "15-2051.00"])  # same code, NOT hot; plus a new one

    _merge_occupations(survivor, duplicate)

    by_code = {o["onet_code"]: o for o in survivor["occupations"]}
    assert by_code["15-1252.00"]["is_hot_tech"] is True, "a true must not be overwritten"
    assert by_code["15-2051.00"]["is_hot_tech"] is False


def test_hot_tech_is_ord_in_the_other_direction_too():
    survivor = entry(["15-1252.00"])
    duplicate = entry(["15-1252.00"], hot_codes=["15-1252.00"])

    _merge_occupations(survivor, duplicate)

    assert survivor["occupations"][0]["is_hot_tech"] is True
    assert survivor["is_hot_tech_anywhere"] is True


def test_a_missing_title_is_filled_from_the_duplicate():
    survivor = {"occupations": [{"onet_code": "15-1252.00", "onet_title": ""}]}
    duplicate = {"occupations": [occupation("15-1252.00", "Software Developers")]}

    _merge_occupations(survivor, duplicate)
    assert survivor["occupations"][0]["onet_title"] == "Software Developers"


def test_merging_reports_how_many_mappings_moved():
    survivor = entry(["17-1022.00"])
    duplicate = entry(["17-1022.00", "17-1022.01", "17-3031.00"])
    assert _merge_occupations(survivor, duplicate) == 2


def test_merging_an_overlapping_pair_adds_nothing_and_loses_nothing():
    survivor = entry(["17-1022.00"])
    duplicate = entry(["17-1022.00"])
    assert _merge_occupations(survivor, duplicate) == 0
    assert survivor["onet_codes"] == ["17-1022.00"]


def test_occupations_without_a_code_are_skipped_rather_than_crashing():
    survivor = entry(["17-1022.00"])
    duplicate = {"occupations": [{"onet_title": "No code here"}, occupation("17-3031.00")]}
    _merge_occupations(survivor, duplicate)
    assert survivor["onet_codes"] == ["17-1022.00", "17-3031.00"]


# --- refusals -----------------------------------------------------------------

def test_a_skill_cannot_duplicate_itself():
    master = store()
    ok, code = _apply_mark_duplicate(master, "MicroSurveyCAD", "MicroSurveyCAD")
    assert (ok, code) == (False, "duplicate_self")


def test_an_unknown_target_changes_nothing():
    master = store()
    before = dict(master["MicroSurvey Software MicroSurvey CAD"])
    ok, code = _apply_mark_duplicate(
        master, "MicroSurvey Software MicroSurvey CAD", "Not In The Store"
    )
    assert (ok, code) == (False, "unknown_target")
    assert master["MicroSurvey Software MicroSurvey CAD"]["status"] == before["status"]
    assert "duplicate_of" not in master["MicroSurvey Software MicroSurvey CAD"]


@pytest.mark.parametrize("target", ["", "   ", None])
def test_an_empty_target_is_refused(target):
    master = store()
    ok, code = _apply_mark_duplicate(
        master, "MicroSurvey Software MicroSurvey CAD", target
    )
    assert (ok, code) == (False, "unknown_target")


def test_an_unknown_skill_is_refused():
    master = store()
    ok, code = _apply_mark_duplicate(master, "Never Heard Of It", "MicroSurveyCAD")
    assert (ok, code) == (False, "not_found")


def test_merging_into_a_rejected_skill_is_refused():
    """
    Its occupations would land somewhere nothing reads, which loses them exactly as
    surely as rejecting the duplicate outright would have.
    """
    master = store()
    master["MicroSurveyCAD"]["status"] = STATUS_REJECTED
    ok, code = _apply_mark_duplicate(
        master, "MicroSurvey Software MicroSurvey CAD", "MicroSurveyCAD"
    )
    assert (ok, code) == (False, "target_rejected")
    assert master["MicroSurvey Software MicroSurvey CAD"]["status"] == STATUS_PENDING


def test_merging_into_an_approved_skill_is_allowed():
    """The common real case: the good record was approved before the duplicate surfaced."""
    master = store()
    master["MicroSurveyCAD"]["status"] = STATUS_APPROVED
    ok, code = _apply_mark_duplicate(
        master, "MicroSurvey Software MicroSurvey CAD", "MicroSurveyCAD"
    )
    assert (ok, code) == (True, "ok")
    assert master["MicroSurveyCAD"]["onet_codes"] == [
        "17-1022.00", "17-1022.01", "17-3031.00"
    ]


def test_the_target_name_is_trimmed():
    """A name pasted out of the queue often carries whitespace."""
    master = store()
    ok, _ = _apply_mark_duplicate(
        master, "MicroSurvey Software MicroSurvey CAD", "  MicroSurveyCAD  "
    )
    assert ok


# --- the batch path -----------------------------------------------------------

def test_mark_duplicate_is_a_final_action():
    """
    So it cannot be staged alongside an approve or reject on the same card, and so the
    server applies any source action before it.
    """
    from review_actions import BATCH_ACTIONS, FINAL_ACTIONS, SOURCE_ACTIONS

    assert "mark-duplicate" in FINAL_ACTIONS
    assert "mark-duplicate" not in SOURCE_ACTIONS
    assert "mark-duplicate" in BATCH_ACTIONS


def test_every_refusal_code_has_a_sentence_for_the_reviewer():
    """A code with no entry renders as the generic fallback, which explains nothing."""
    from main import COMMIT_ERRORS

    for code in ("unknown_target", "duplicate_self", "target_rejected", "not_found"):
        assert COMMIT_ERRORS.get(code), f"{code} has no reviewer-facing sentence"

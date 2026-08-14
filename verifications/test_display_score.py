"""
Tests for the reader-facing banded AI Score.

    python3.11 -m pytest verifications/test_display_score.py -q

The property this file exists to protect is that DISPLAY RANK MATCHES TIER: every AI Skill
displays above every AI Enabling skill, which displays above every Not AI skill. The raw
cosine does not have that property and cannot be given it -- the top bucket is decided by
two metrics, not one -- so the band is a presentation device and these tests are what stop
it drifting back into being a measurement.

Most of these run against the LIVE STORE rather than against constructed values. A band
that holds on three hand-picked examples and breaks on the 400 skills in the middle bucket
would be worse than no band at all, because it would look correct.
"""

import pytest

from dashboardtables import (
    DISPLAY_BANDS,
    build_dashboard_rows,
    compute_display_ai_score,
)
from json_store import load_master, load_timeseries
from sortingalgorithmnew import BUCKET_AI, BUCKET_ENABLING, BUCKET_NOT_AI

AI_BAND = (0.70, 1.00)
ENABLING_BAND = (0.30, 0.69)
NOT_AI_BAND = (0.00, 0.29)

BANDS = {
    BUCKET_AI: AI_BAND,
    BUCKET_ENABLING: ENABLING_BAND,
    BUCKET_NOT_AI: NOT_AI_BAND,
}


@pytest.fixture(scope="module")
def live_rows():
    return build_dashboard_rows(load_master(), load_timeseries())


# --- The property, over real data ---------------------------------------------

def test_every_skill_in_the_store_displays_inside_its_own_band(live_rows):
    """
    THE WHOLE POINT, asserted over all 1051 approved skills rather than on examples.

    Checked exhaustively because the failure mode is a handful of skills at the extremes
    escaping into a neighbouring band -- which is exactly what the spec's original bounds
    did, and which no sample of three would have caught.
    """
    assert live_rows, "the live store must not be empty, or this asserts nothing"

    for row in live_rows:
        bucket = row["category_bucket"]
        score = row["display_ai_score"]
        low, high = BANDS[bucket]
        assert low <= score <= high, (
            f"{row['skill_name']!r} is {bucket} at raw {row['ai_score']} but displays "
            f"{score}, outside [{low}, {high}]")


def test_the_bands_do_not_touch_at_display_precision(live_rows):
    """
    Separation is asserted on the DISPLAYED value, not on the internal one.

    The bands were written 0.699 and 0.299 first. Both round to 0.70 and 0.30 at two
    decimals -- so the top of one band rendered identically to the bottom of the next, and
    a reader sorting the table would have seen an AI Enabling skill and an AI Skill both
    reading 0.70. Numerically fine, visually the exact failure the band exists to prevent.
    """
    top = {bucket: [] for bucket in BANDS}
    for row in live_rows:
        top[row["category_bucket"]].append(row["display_ai_score"])

    assert max(top[BUCKET_NOT_AI]) < min(top[BUCKET_ENABLING])
    assert max(top[BUCKET_ENABLING]) < min(top[BUCKET_AI])


def test_the_named_false_positives_display_below_the_genuine_tools(live_rows):
    """
    The complaint that prompted this, in the reader's terms: the word-processing tools
    used to print a HIGHER number than spaCy and XGBoost, because their raw generative
    score is genuinely higher and the engineering floor -- not the score -- is what kept
    them out of the top bucket.
    """
    scored = {row["skill_name"]: row for row in live_rows}
    genuine = ["spaCy", "XGBoost"]
    misleading = ["Word processing software", "Transcription system software",
                  "Telluride Software Classic Trak-It"]

    for name in genuine + misleading:
        assert name in scored, f"{name!r} left the store; this test needs re-anchoring"

    floor = min(scored[name]["display_ai_score"] for name in genuine)
    for name in misleading:
        row = scored[name]
        assert row["display_ai_score"] < floor, (
            f"{name!r} displays {row['display_ai_score']} against a genuine floor of "
            f"{floor}")
        # And the raw score confirms this is a real inversion being corrected, not a
        # coincidence: on the raw scale the ordering is the wrong way round.
        assert row["ai_score"] > 0.20


def test_ordering_inside_a_band_still_follows_the_measurement(live_rows):
    """
    The band groups tiers; it must not flatten the ranking inside one. Two skills in the
    same bucket keep their measured order.
    """
    enabling = sorted(
        (row for row in live_rows if row["category_bucket"] == BUCKET_ENABLING),
        key=lambda row: row["ai_score"])

    displays = [row["display_ai_score"] for row in enabling]
    assert displays == sorted(displays), "a higher raw score must never display lower"
    assert len(set(displays)) > 10, \
        "the band must not collapse the bucket onto a handful of tied values"


# --- The cases that needed a clamp --------------------------------------------

@pytest.mark.parametrize("name,bucket", [
    # Raw -0.0185. Rule 1 promotes on tech_base/ml_pipeline and never reads ai_score, so
    # an AI Enabling skill can measure below zero. Unclamped this displayed 0.060.
    ("Strategic Reporting Systems ReportSmith", BUCKET_ENABLING),
    # Raw 0.3549, the highest in its bucket and above the top of the AI Skill bucket's
    # own minimum. Unclamped this displayed 0.708.
    ("Transcription system software", BUCKET_ENABLING),
])
def test_the_real_out_of_range_skills_are_clamped_into_their_band(live_rows, name, bucket):
    scored = {row["skill_name"]: row for row in live_rows}
    row = scored[name]
    low, high = BANDS[bucket]
    assert row["category_bucket"] == bucket
    assert low <= row["display_ai_score"] <= high


def test_a_score_far_outside_every_observed_bound_still_lands_in_band():
    """Direct on the function, so the clamp is asserted rather than inferred from data."""
    assert compute_display_ai_score(5.0, BUCKET_ENABLING) == ENABLING_BAND[1]
    assert compute_display_ai_score(-5.0, BUCKET_ENABLING) == ENABLING_BAND[0]
    assert compute_display_ai_score(5.0, BUCKET_NOT_AI) == NOT_AI_BAND[1]
    assert compute_display_ai_score(-5.0, BUCKET_AI) == AI_BAND[0]


# --- The things it must not do ------------------------------------------------

def test_an_unscored_skill_has_no_display_score():
    """
    None, not 0.0. An unscored skill has not been measured and found to be nothing; 0.0
    would sort it above every measured skill in an ascending sort as though it had.
    """
    assert compute_display_ai_score(None, BUCKET_AI) is None
    assert compute_display_ai_score(None, None) is None


def test_an_unrecognised_bucket_returns_none_rather_than_guessing():
    """A bad bucket is a data problem. Filing it at the bottom of the Not AI band would
    render it as a real measurement of a real skill."""
    assert compute_display_ai_score(0.5, "Something new") is None


def test_the_display_score_is_not_view_dependent(live_rows):
    """
    The defect in the min-max normalizer this replaced: it scaled against whatever was
    filtered, so the top skill of any slice always read 1.00 and the same skill showed
    different numbers in different views.
    """
    subset = [row for row in live_rows if row["category_bucket"] == BUCKET_NOT_AI][:25]
    assert subset, "fixture needs Not AI rows"

    for row in subset:
        assert compute_display_ai_score(row["ai_score"], row["category_bucket"]) \
            == row["display_ai_score"]


def test_no_scoring_module_imports_the_display_helper():
    """
    THE STRUCTURAL GUARD. A banded score is discontinuous at its boundaries -- raw 0.289
    and 0.291 display 0.30 and 0.70 -- so anything that decided on it would turn a
    rounding difference into a tier difference. It stays in the presentation layer.

    Asserted on the source rather than on behaviour, because a rule that reads it but
    happens not to fire on today's data would pass a behavioural test.
    """
    import inspect

    import definitions_algorithm
    import json_store
    import sortingalgorithmnew

    for module in (sortingalgorithmnew, json_store, definitions_algorithm):
        source = inspect.getsource(module)
        assert "compute_display_ai_score" not in source, (
            f"{module.__name__} references the display score. It is for rendering only; "
            "classification must read ai_score.")
        assert "display_ai_score" not in source, (
            f"{module.__name__} references display_ai_score.")


def test_the_bands_are_declared_for_every_bucket():
    """A bucket added to the engine without a band here would silently render as n/a."""
    assert set(DISPLAY_BANDS) == {BUCKET_AI, BUCKET_ENABLING, BUCKET_NOT_AI}


# --- The markup the script depends on -----------------------------------------

def test_every_element_dashboard_js_looks_up_exists_in_the_template():
    """
    THE OUTAGE GUARD, and it is here because this failed silently in production markup.

    templates/dashboard.html was missing #group-select, #role-bucket-stack and
    #role-bucket-legend. dashboard.js looks all three up at start-up and calls appendChild
    on the result, so getElementById returned null and the script threw -- during the
    straight-line set-up, before it drew anything.

    The symptom did not look like a crash. The category chips had already been given their
    listeners, so the page loaded blank and only filled in when the reader clicked a chip,
    which called drawRanked() again outside the aborted run. The bucket bar rendered as an
    empty grey track. Nothing in the Python suite could see any of it.

    Static because it is cheap and total: it reads every getElementById in the script and
    every id in the template, so a node deleted from the markup fails here rather than in
    a browser.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    script = (root / "static" / "dashboard.js").read_text()
    template = (root / "templates" / "dashboard.html").read_text()

    looked_up = set(re.findall(r'getElementById\("([^"]+)"\)', script))
    present = set(re.findall(r'id="([^"]+)"', template))

    assert looked_up, "the regex stopped matching; this test is no longer checking anything"

    missing = sorted(looked_up - present)
    assert not missing, (
        "dashboard.js looks up "
        + ", ".join("#" + name for name in missing)
        + " but the template has no such element. getElementById returns null and the "
          "script throws at start-up, blanking the whole dashboard.")

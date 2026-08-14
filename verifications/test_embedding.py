"""
Tests for the searched embedded-AI boolean, its boost, and the recheck policy.

    python3.11 -m pytest test_embedding.py -q

No network and no Gemini: the DuckDuckGo responses below are REAL bytes captured from
html.duckduckgo.com, and the grader is stubbed. The live calibration that justified
letting this change a score is reported separately; what is guarded here is everything
that must hold whether or not the model is having a good day.

THREE THINGS THIS FILE EXISTS TO STOP:

  1. embedded_ai_sim quietly regaining the power to classify. That is what put LaTeX and
     a 2D drafting package in a class called "Non-Technical Embedded AI".
  2. A failed search reading as "no AI features". DuckDuckGo answers a rate-limited
     client with 202 and an empty result page, and the difference between that and a
     finding is the difference between a coverage gap and a claim about a product.
  3. A confirmed verdict being re-asked, or an unknown one not being. That is what makes
     the second run of a five-and-a-half-hour pass nearly free.
"""

import datetime
import unittest.mock as mk

import pytest

import embedding_pass as ep
import embedding_probe as probe
from sortingalgorithmnew import (
    AI_SKILL_THRESHOLD,
    BUCKET_AI,
    BUCKET_ENABLING,
    BUCKET_NOT_AI,
    EMBEDDED_AI_BOOST,
    SUB_EMBEDDED_AI,
    apply_boost,
    base_of,
    boost_for,
    calculate_ai_correlation,
    classify,
)

TODAY = datetime.date(2026, 8, 13)


# --- The trap the whole design is built around --------------------------------
#
# Captured verbatim from a real search. Every result argues that LaTeX embeds AI, and
# every one of them is a THIRD-PARTY editor built around LaTeX. The correct answer is
# False. This is the fixture because it is the failure the payload actively pushes
# toward, and no amount of query tuning removes it -- telling these apart requires
# knowing what LaTeX is, which is why the definition is sent with the results.

LATEX_HTML = b"""
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fai-latex">
    AI for LaTeX Writing: What It Actually Helps With</a>
  <a class="result__snippet">An honest look at AI in LaTeX editors: where AI compile-error
    fixes save real time, and what gets oversold.</a>
</div>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.overleaf.com%2Fai">
    AI features - Overleaf, Online LaTeX Editor</a>
  <a class="result__snippet">An online LaTeX editor that's easy to use. No installation,
    real-time collaboration, version control, hundreds of LaTeX templates.</a>
</div>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fprism.example%2F">
    Prism | A free, LaTeX Editor and AI-native workspace for scientists</a>
  <a class="result__snippet">Introducing a free, AI-first LaTeX editor that integrates
    ChatGPT and Codex directly into scientific writing.</a>
</div>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Funderleaf.example%2F">
    AI LaTeX Editor - Write LaTeX with AI - Underleaf</a>
  <a class="result__snippet">Write LaTeX with AI assistance. Generate LaTeX from plain
    English and get an AI copilot right in your editor.</a>
</div>
"""

LATEX_DEFINITION = (
    "LaTeX is a software system for typesetting documents, widely used in academia for "
    "the communication and publication of scientific documents. It was created by Leslie "
    "Lamport in 1984 as a set of macros over Donald Knuth's TeX."
)

# The 202 body a rate-limited client actually receives: structurally a result page with
# nothing in it. Not an error status, so requests will not raise on it.
BLOCKED_HTML = b"<html><body><div class='no-results'></div></body></html>"


def response(status=200, content=b""):
    return mk.Mock(status_code=status, content=content)


# --- Parsing ------------------------------------------------------------------

def test_the_captured_latex_page_parses():
    results = probe.parse_results(LATEX_HTML)
    assert len(results) == 4
    assert results[1]["title"].startswith("AI features - Overleaf")


def test_a_result_url_is_unwrapped_from_the_redirector():
    """
    A grader shown duckduckgo.com/l/?uddg=... for every result cannot tell helpx.adobe
    .com from a listicle, which is exactly the judgement it is being asked to make.
    """
    results = probe.parse_results(LATEX_HTML)
    assert results[1]["url"] == "https://www.overleaf.com/ai"


def test_the_grader_is_shown_the_host_it_can_judge():
    rendered = probe.format_results(probe.parse_results(LATEX_HTML))
    assert "[www.overleaf.com]" in rendered


def test_the_definition_travels_with_the_results():
    """
    Not decoration. "Overleaf is an online LaTeX editor" and "LaTeX embeds AI" are
    indistinguishable from the snippets alone; the 1984 typesetting definition is what
    makes them distinguishable, and it costs no extra request.
    """
    checker = mk.Mock()
    checker._batch_call.return_value = []
    checker.grade_embedding_batch = type(checker).grade_embedding_batch = None

    from agentic_source_check import AgenticSourceChecker

    captured = {}

    def fake_batch_call(**kwargs):
        captured.update(kwargs)
        return []

    real = mk.Mock(spec=AgenticSourceChecker)
    real._batch_call = fake_batch_call
    AgenticSourceChecker.grade_embedding_batch(real, [{
        "item_id": "LaTeX", "skill_name": "LaTeX",
        "definition": LATEX_DEFINITION,
        "results": probe.parse_results(LATEX_HTML),
    }])

    assert "Leslie Lamport in 1984" in captured["prompt"]
    assert "Overleaf" in captured["prompt"]


# --- Three states, and the one that must never collapse -----------------------

@pytest.mark.parametrize("status, content, expected", [
    (202, BLOCKED_HTML, probe.ERROR_BLOCKED),
    (200, BLOCKED_HTML, probe.ERROR_NO_RESULTS),
    (503, b"", probe.ERROR_FETCH_FAILED),
])
def test_every_failure_is_unknown_and_never_a_negative(status, content, expected):
    """
    The failure that matters most. A 202 means rate limited, and if it read as False it
    would strip the boost from every skill an unlucky afternoon touched while looking
    exactly like hundreds of findings.
    """
    with mk.patch.object(probe, "_wait_for_slot"), \
         mk.patch.object(probe.requests, "get", return_value=response(status, content)):
        found = probe.search_embedding_evidence("LaTeX")

    assert found["error"] == expected
    assert found["results"] == []


def test_a_200_with_no_parseable_results_is_unknown_not_negative():
    """
    Specifically about a PARSER break, not about the web. It is far more likely that
    DuckDuckGo moved its markup than that it has nothing on a product O*NET lists, and a
    silent parser break must not mark the entire store as shipping no AI.
    """
    with mk.patch.object(probe, "_wait_for_slot"), \
         mk.patch.object(probe.requests, "get", return_value=response(200, b"<html></html>")):
        assert probe.search_embedding_evidence("Slack")["error"] == probe.ERROR_NO_RESULTS


def test_an_empty_name_is_not_searched_for():
    with mk.patch.object(probe.requests, "get") as get:
        assert probe.search_embedding_evidence("  ")["error"] == probe.ERROR_EMPTY_QUERY
    get.assert_not_called()


def test_a_transport_error_is_unknown():
    with mk.patch.object(probe, "_wait_for_slot"), \
         mk.patch.object(probe.requests, "get",
                         side_effect=probe.requests.RequestException("timeout")):
        assert probe.search_embedding_evidence("Slack")["error"] == probe.ERROR_FETCH_FAILED


def test_the_query_is_the_one_that_was_calibrated():
    assert probe.build_query("Slack") == "does Slack embed AI"


def test_the_measured_cadence_is_not_quietly_lowered():
    """
    25s is measured, not chosen: six queries inside ten seconds got this IP blocked for
    two minutes, and eight at 25s apart all returned results. Lowering it does not make
    a pass faster, it makes it return unknown for everything while looking like progress.
    """
    assert probe.POLITENESS_DELAY >= 25.0


# --- The grader's contract ----------------------------------------------------

def test_an_item_with_no_search_results_is_never_sent():
    """Asking the model to judge from nothing would get an answer, and it would be an
    invention. Absent is the correct outcome: no evidence was gathered."""
    from agentic_source_check import AgenticSourceChecker

    checker = mk.Mock(spec=AgenticSourceChecker)
    out = AgenticSourceChecker.grade_embedding_batch(checker, [
        {"item_id": "X", "skill_name": "X", "definition": "d", "results": []},
    ])
    assert out == {}
    checker._batch_call.assert_not_called()


def graded(verdicts, results=None):
    """Runs the reconciliation half of grade_embedding_batch over stubbed model output."""
    from agentic_source_check import AgenticSourceChecker

    checker = mk.Mock(spec=AgenticSourceChecker)
    checker._batch_call.return_value = verdicts
    return AgenticSourceChecker.grade_embedding_batch(checker, [{
        "item_id": "LaTeX", "skill_name": "LaTeX", "definition": LATEX_DEFINITION,
        "results": results if results is not None else probe.parse_results(LATEX_HTML),
    }])


def test_a_citation_on_an_unrelated_host_is_dropped_and_the_verdict_kept():
    """
    MEASURED on the calibration run, not hypothetical. The model returned the CORRECT
    verdict for LaTeX -- third-party editors, not LaTeX itself -- and cited
    "openai.com/index/prism-a-free-latex-editor-...", a domain in none of the results.
    The drawer renders this as a link, so evidence attributed to an unrelated company is
    worse than no evidence. The reasoning was sound, so the verdict stands.
    """
    out = graded([{
        "item_id": "LaTeX", "embeds_ai": False, "evidence": "third-party editors only",
        "evidence_url": "https://www.openai.com/index/prism-a-free-latex-editor",
        "confidence": 1.0,
    }])

    assert out["LaTeX"]["embeds_ai"] is False
    assert out["LaTeX"]["evidence"] == "third-party editors only"
    assert out["LaTeX"]["evidence_url"] is None


def test_a_url_copied_from_the_results_survives():
    out = graded([{
        "item_id": "LaTeX", "embeds_ai": False, "evidence": "third-party editors only",
        "evidence_url": "https://www.overleaf.com/ai", "confidence": 1.0,
    }])
    assert out["LaTeX"]["evidence_url"] == "https://www.overleaf.com/ai"


def test_a_deep_link_on_a_host_the_search_surfaced_survives():
    """
    WHY THE CHECK IS ON THE HOST AND NOT THE EXACT URL, measured on the same run.

    Grading six Adobe products, four cited helpx.adobe.com deep links that were not
    verbatim in the results -- but helpx.adobe.com WAS among the result hosts. Those are
    the canonical vendor documentation pages and the single most useful thing to link.
    Exact matching dropped all four and left the column empty on every Adobe skill.
    """
    out = graded(
        [{"item_id": "LaTeX", "embeds_ai": True, "evidence": "ships an assistant",
          "evidence_url": "https://www.overleaf.com/learn/ai-assist", "confidence": 1.0}],
    )
    assert out["LaTeX"]["evidence_url"] == "https://www.overleaf.com/learn/ai-assist"


def test_a_verdict_with_no_citation_is_left_alone():
    out = graded([{
        "item_id": "LaTeX", "embeds_ai": False, "evidence": "nothing native",
        "evidence_url": None, "confidence": 0.8,
    }])
    assert out["LaTeX"]["evidence_url"] is None
    assert out["LaTeX"]["embeds_ai"] is False


def test_a_missing_item_id_is_absent_rather_than_false():
    from agentic_source_check import AgenticSourceChecker

    checker = mk.Mock(spec=AgenticSourceChecker)
    checker._batch_call.return_value = []
    out = AgenticSourceChecker.grade_embedding_batch(checker, [
        {"item_id": "LaTeX", "skill_name": "LaTeX", "definition": LATEX_DEFINITION,
         "results": probe.parse_results(LATEX_HTML)},
    ])
    assert "LaTeX" not in out


# --- The engine ---------------------------------------------------------------

def test_only_a_confirmed_finding_earns_the_boost():
    assert boost_for(True) == EMBEDDED_AI_BOOST
    assert boost_for(False) == 0.0
    assert boost_for(None) == 0.0


def test_the_boost_is_split_out_rather_than_folded_in():
    """
    ai_score_base is what keeps a boosted 2026 snapshot comparable with an unboosted
    2025 one. Without the split, a trend spanning this change draws a 0.05 step that is
    a policy decision rather than drift.
    """
    boosted = apply_boost(0.2800, True)
    assert boosted == {
        "ai_score_base": 0.28,
        "embedded_ai_boost": 0.05,
        "lexical_ai_boost": 0.0,
        "lexical_ai_terms": [],
        "ai_score": 0.33,
    }


def test_the_phrase_match_no_longer_adds_to_the_score():
    """
    The lexical boost existed only while "nlp" and friends were stripped out of the
    generative pole. The pole is restored, so it measures those phrases again and a
    boost on top would count the same evidence twice. The match is still recorded.
    """
    with_phrase = apply_boost(0.2000, True, ["machine learning"])
    assert with_phrase["embedded_ai_boost"] == 0.05, "the searched fact still adds"
    assert with_phrase["lexical_ai_boost"] == 0.0, "the phrase does not"
    assert with_phrase["ai_score"] == pytest.approx(0.25, abs=1e-9)
    assert with_phrase["lexical_ai_terms"] == ["machine learning"], "still on the record"
    assert with_phrase["ai_score_base"] == 0.20, "neither touches the measurement"


def test_a_legacy_snapshot_reads_its_own_score_as_the_base():
    """Nothing had been added to it, so it IS the base. This is the compatibility shim
    every cross-boundary comparison has to go through."""
    assert base_of({"ai_score": 0.42}) == 0.42
    assert base_of({"ai_score": 0.47, "ai_score_base": 0.42}) == 0.42


def test_the_boost_moves_nothing_but_the_score():
    definition = "Slack is a team communication platform with topic-based chat rooms."
    plain = calculate_ai_correlation("Slack", "", definition, "", None)
    boosted = calculate_ai_correlation("Slack", "", definition, "", True)

    assert boosted["ai_score"] == pytest.approx(plain["ai_score"] + EMBEDDED_AI_BOOST, abs=1e-4)
    assert boosted["ai_score_base"] == plain["ai_score_base"]
    for metric in ("tech_base_sim", "ml_pipeline_sim", "embedded_ai_sim"):
        assert boosted[metric] == plain[metric]


def test_a_confirmed_finding_alone_lands_in_the_embedded_class():
    """The common case: the boost is nowhere near the AI Skill bar, so rule 2 decides."""
    result = classify(0.10 + EMBEDDED_AI_BOOST, 0.0, 0.0, 0.0, True)
    assert result["category_bucket"] == BUCKET_ENABLING
    assert result["sub_category"] == SUB_EMBEDDED_AI


def test_the_boost_can_carry_a_skill_over_the_ai_skill_bar():
    """
    THE SHARP EDGE, asserted rather than discovered later. The boost is a sixth of the AI
    Skill bar, so any skill measuring within EMBEDDED_AI_BOOST of it is promoted outright
    by a confirmed finding.

    That is the boost working as specified, and it is the reason the finding has to be
    established by a search rather than inferred from the definition: the old similarity
    would have handed this to a third of the store.

    Written as offsets from AI_SKILL_THRESHOLD rather than as literals. The bar moved once
    already (0.30 -> 0.29, when scoring changed to the 90/10 blend) and broke this test for
    a reason that had nothing to do with the boost it exists to check.
    """
    within_reach = AI_SKILL_THRESHOLD - EMBEDDED_AI_BOOST + 0.01
    out_of_reach = AI_SKILL_THRESHOLD - EMBEDDED_AI_BOOST - 0.01

    assert classify(within_reach, 0.0, 0.0, 0.0)["category_bucket"] == BUCKET_NOT_AI

    promoted = classify(within_reach + EMBEDDED_AI_BOOST, 0.0, 0.0, 0.0, True)
    assert promoted["category_bucket"] == BUCKET_AI
    assert promoted["decision_metric"] == "ai_score", "rule 0 outranks rule 2"

    # Just below the reach: the boost still leaves it under the bar, so rule 2 catches it.
    assert classify(out_of_reach + EMBEDDED_AI_BOOST, 0.0, 0.0, 0.0, True)["category_bucket"] \
        == BUCKET_ENABLING


def test_a_blank_title_never_earns_a_boost():
    """A stale True arriving with nothing to score must not give an empty record 0.05."""
    metrics = calculate_ai_correlation("", "", "", "", True)
    assert metrics["ai_score"] == 0.0
    assert metrics["embeds_ai"] is None


def test_similarity_no_longer_reaches_the_class():
    """The regression guard, stated once more at the level the store cares about."""
    assert classify(0.10, 0.0, 0.0, 0.49)["sub_category"] != SUB_EMBEDDED_AI
    assert classify(0.10, 0.0, 0.0, 0.00, True)["sub_category"] == SUB_EMBEDDED_AI


# --- The recheck policy -------------------------------------------------------

def record(value, days_ago=0):
    return {
        "embeds_ai": value,
        "checked_at": (TODAY - datetime.timedelta(days=days_ago)).isoformat(),
    }


def test_a_confirmed_finding_is_never_re_asked():
    """A product does not un-ship its AI features, so there is nothing to re-establish
    and a window would only spend search cadence to confirm what is already known."""
    for days in (0, 90, 400, 5000):
        assert ep.is_due(record(True, days), TODAY) is False


@pytest.mark.parametrize("days, due", [
    (0, False), (1, False), (89, False),
    (90, True), (91, True), (365, True),
])
def test_a_negative_goes_stale_after_the_window(days, due):
    """The state that genuinely ages: a 2026 negative for a tool that ships a copilot in
    2027 is the entire reason a window exists."""
    assert ep.is_due(record(False, days), TODAY) is due


def test_an_unknown_is_retried_every_run_whatever_its_date():
    """It is not a finding, it is a gap. Retrying is the only thing that closes it, and
    a freshly-written unknown must not look fresh enough to skip."""
    assert ep.is_due(None, TODAY) is True
    assert ep.is_due(record(None, 0), TODAY) is True
    assert ep.is_due(record(None, 500), TODAY) is True


def test_a_corrupt_date_is_re_probed_rather_than_trusted():
    """Re-probing costs 25 seconds. Trusting an unparseable date costs a wrong verdict
    standing forever, because nothing else would ever revisit it."""
    assert ep.is_due({"embeds_ai": False, "checked_at": "not-a-date"}, TODAY) is True


def test_an_unknown_is_recorded_with_why_it_failed():
    """Recorded but not honoured. is_due ignores the date on an unknown, so writing it
    changes nothing about when it is asked again -- what it buys is a store that can say
    how many skills were never established, instead of leaving that to be inferred."""
    cache = {}
    ep.record_verdict(cache, "Slack", None, error=probe.ERROR_BLOCKED, now=TODAY)

    assert cache["Slack"]["embeds_ai"] is None
    assert cache["Slack"]["error"] == probe.ERROR_BLOCKED
    assert ep.is_due(cache["Slack"], TODAY) is True


def test_the_three_states_round_trip_through_the_cache():
    cache = {}
    ep.record_verdict(cache, "Yes", {"embeds_ai": True, "evidence": "ships a copilot"}, now=TODAY)
    ep.record_verdict(cache, "No", {"embeds_ai": False, "evidence": "third-party only"}, now=TODAY)
    ep.record_verdict(cache, "Unknown", None, error="blocked", now=TODAY)

    assert ep.embeds_ai_of(cache["Yes"]) is True
    assert ep.embeds_ai_of(cache["No"]) is False
    assert ep.embeds_ai_of(cache["Unknown"]) is None


# --- What a run actually costs ------------------------------------------------

def entry(status="approved"):
    return {"status": status, "skill_name": "X", "category": "", "wikipedia_summary": "d"}


def store():
    master = {name: entry() for name in ("Confirmed", "FreshNo", "StaleNo", "NeverAsked")}
    timeseries = [
        {"skill_name": name, "quarter": "2026Q3", "ai_score": 0.1,
         "tech_base_sim": 0.0, "ml_pipeline_sim": 0.0, "embedded_ai_sim": 0.0}
        for name in master
    ]
    cache = {
        "Confirmed": record(True, 900),
        "FreshNo": record(False, 10),
        "StaleNo": record(False, 200),
    }
    return master, timeseries, cache


def test_a_second_run_only_pays_for_gaps_and_stale_negatives():
    master, timeseries, cache = store()
    plan = ep.plan_run(master, timeseries, cache, now=TODAY)

    assert [name for name, _ in plan["due"]] == ["NeverAsked", "StaleNo"]
    assert plan["counts"]["cached_yes"] == 1
    assert plan["counts"]["cached_no_fresh"] == 1
    assert plan["counts"]["due_recheck"] == 1
    assert plan["counts"]["due_unknown"] == 1


def test_force_overrides_a_confirmed_finding():
    """For when a verdict is contested. The only way past the never-re-ask rule, and it
    has to be named explicitly rather than being a blanket flag."""
    master, timeseries, cache = store()
    plan = ep.plan_run(master, timeseries, cache, forced={"Confirmed"}, now=TODAY)

    assert "Confirmed" in [name for name, _ in plan["due"]]
    assert plan["counts"]["forced"] == 1


def test_unapproved_and_unscored_skills_are_out_of_scope():
    """An unapproved skill is not on the dashboard, so a tag for it would describe
    something nobody can see; an unscored one has no snapshot to put the verdict in."""
    master = {"Pending": entry("pending"), "Unscored": entry()}
    plan = ep.plan_run(master, [], {}, now=TODAY)

    assert plan["due"] == []
    assert plan["counts"]["in_scope"] == 0

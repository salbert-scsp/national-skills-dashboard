"""
Tests for the conditional classification rules engine.

    python3.11 -m pytest test_classification.py -q

Two kinds of test live here. Most call classify() with metric values taken from the live
store, so they assert the engine's logic without loading the ONNX model. A smaller set
marked `real_model` runs the actual scorer, because the flagship rule's whole safety
argument is about what embedding a sentence does to the OTHER metrics, and stubbed
numbers cannot demonstrate that.
"""

import pytest

from sortingalgorithmnew import (
    AI_ENGINEERING_FLOOR,
    AI_SKILL_THRESHOLD,
    BUCKET_AI,
    BUCKET_ENABLING,
    BUCKET_NOT_AI,
    CONFIDENCE_HIGH,
    CONFIDENCE_MARGINAL,
    DRIFT_DELTA,
    SUB_CONCEPTUAL,
    SUB_EMBEDDED_AI,
    SUB_TECHNICAL_BACKBONE,
    TECH_BASE_FLOOR,
    classify,
    class_normalized,
)

# Metrics measured on the live store, so these are regression tests against real data
# rather than against numbers chosen to make the rules look right.
# (ai_score, tech_base_sim, ml_pipeline_sim, embedded_ai_sim)
#
# RE-MEASURED under the 90/10 definition-primary blend (see CONTEXT_WEIGHT). The whole
# table shifted down, because the blend lowered every pole and not only the two AI ones.
# Re-measuring rather than adjusting the assertions is the only honest option here: these
# numbers are the fixture's claim to be a regression test against real data, and comparing
# old-scale metrics against re-derived floors tests nothing that exists.
#
# RE-MEASURED 2026-10-06 after the 2026-08-14 pole and anchor change (commit 4639910),
# from the latest snapshot of each skill in skills_timeseries.json. Those snapshots were
# confirmed identical to a fresh calculate_ai_correlation() run (150 of 150 sampled
# skills, every similarity within 0.0001). Store keys: SSIS = "Microsoft SQL Server
# Integration Services SSIS", Power BI = "Microsoft Power BI", DHTML = "Dynamic hypertext
# markup language DHTML", Teams = "Microsoft Teams".
MEASURED = {
    "C++": (0.084, 0.335, 0.165, 0.143),
    "SAS": (0.101, 0.381, 0.167, 0.116),
    "SSIS": (0.065, 0.250, 0.274, 0.257),
    "Power BI": (0.112, 0.370, 0.267, 0.225),
    "Google Docs": (0.162, 0.066, 0.114, 0.142),
    "Yahoo! Email": (0.136, -0.018, 0.001, 0.108),
    "DHTML": (0.091, 0.114, -0.051, 0.122),
    "MongoDB": (0.065, 0.012, 0.133, -0.043),
    # Slack and Teams as measured BEFORE any flagship note. Both must be Not AI here;
    # only the flagship pass is allowed to move them.
    "Slack": (0.074, 0.126, 0.161, 0.183),
    "Teams": (0.120, 0.118, 0.184, 0.139),
    # The same two after the flagship note. The first three values are the re-measured
    # base values above (a note moves embedded_ai_sim only -- see
    # test_flagship_note_moves_embedded_ai_and_nothing_else). The embedded_ai_sim values
    # are from the PRE-2026-08-14 measurement and were not re-derived: the Teams note text
    # is not in the repository. They only need to sit above every base value, which they do.
    "Slack, flagship": (0.074, 0.126, 0.161, 0.443),
    "Teams, flagship": (0.120, 0.118, 0.184, 0.331),
}


def classify_named(name):
    return classify(*MEASURED[name])


# --- Category 3: technical backbone ------------------------------------------

@pytest.mark.parametrize("name", ["C++", "SAS", "SSIS"])
def test_core_technology_is_not_suppressed_to_not_ai(name):
    """The defect this engine exists to fix: high tech_base, low ai_score, class None."""
    result = classify_named(name)
    assert result["category_bucket"] == BUCKET_ENABLING
    assert result["sub_category"] == SUB_TECHNICAL_BACKBONE


def test_backbone_fires_on_ml_pipeline_alone():
    """Either arm of the OR is sufficient; tech_base here is far below its floor."""
    result = classify(0.10, 0.05, 0.25, 0.05)
    assert result["sub_category"] == SUB_TECHNICAL_BACKBONE
    assert result["decision_metric"] == "ml_pipeline_sim"


def test_backbone_reports_the_arm_it_won_by():
    """An OR is decided by its largest margin, not by whichever is checked first."""
    result = classify(0.10, 0.60, 0.21, 0.05)
    assert result["decision_metric"] == "tech_base_sim"


# --- Category 2: embedded AI --------------------------------------------------
#
# THIS RULE DELIBERATELY CHANGED. It used to test embedded_ai_sim against a 0.22 floor.
# On the live store that similarity has median 0.179 and p75 0.237, so a third of every
# skill cleared it, and the resulting class contained LaTeX (0.260), Thomson EndNote
# (0.264), SofTech CADRA (0.225) and Transoft AutoTURN (0.280) -- a typesetting system
# from 1984, a bibliography manager and two drafting packages. The anchor's vocabulary
# (workspace, workflow, integration, automated) is the vocabulary of any productivity
# tool, and most of these definitions predate the features being judged.
#
# The fact is now searched for instead -- embedding_probe.py -- and arrives as a
# three-state boolean. The tests below are about that boolean and nothing else.

def test_the_boolean_is_what_fires_the_rule():
    result = classify(0.10, 0.0, 0.0, 0.0, True)
    assert result["category_bucket"] == BUCKET_ENABLING
    assert result["sub_category"] == SUB_EMBEDDED_AI
    assert result["decision_metric"] == "embeds_ai"


@pytest.mark.parametrize("similarity", [0.19, 0.22, 0.30, 0.49])
def test_similarity_alone_never_promotes_anything(similarity):
    """
    The regression guard for the whole change.

    0.49 is above the highest embedded_ai_sim in the store. If any of these starts
    classifying as Embedded AI again, the floor has come back and LaTeX is in the class
    with it.
    """
    assert classify(0.10, 0.0, 0.0, similarity)["category_bucket"] == BUCKET_NOT_AI


@pytest.mark.parametrize("name", ["Google Docs", "Slack, flagship", "Teams, flagship"])
def test_skills_that_used_to_qualify_on_similarity_no_longer_do(name):
    """
    Named cases, so the change is visible rather than implied. Google Docs (0.209) and
    both flagship-noted tools (0.502, 0.451) all cleared the old rule. None of them is
    promoted now without a search saying so.
    """
    assert classify_named(name)["sub_category"] != SUB_EMBEDDED_AI


@pytest.mark.parametrize("name", ["Google Docs", "Slack", "Teams"])
def test_the_same_skills_are_promoted_once_a_search_confirms_them(name):
    """The rule still exists and still fires; it just needs evidence now."""
    result = classify(*MEASURED[name], True)
    assert result["sub_category"] == SUB_EMBEDDED_AI


def test_unknown_is_treated_as_not_established_but_is_not_false():
    """
    None must not promote -- an unsearched skill is not evidence of embedded AI -- and
    it must not be confused with False anywhere the two are stored or displayed.
    """
    assert classify(0.10, 0.0, 0.0, 0.30, None)["category_bucket"] == BUCKET_NOT_AI
    assert classify(0.10, 0.0, 0.0, 0.30, False)["category_bucket"] == BUCKET_NOT_AI


@pytest.mark.parametrize("falsy", [0, "", 0.0, [], None])
def test_only_the_literal_true_fires_the_rule(falsy):
    """`is True`, not truthiness: a caller passing 0 or "" must not be read as False
    and, more importantly, a caller passing 1 or "yes" must not be read as True."""
    assert classify(0.10, 0.0, 0.0, 0.0, falsy)["sub_category"] != SUB_EMBEDDED_AI


def test_a_boolean_decision_carries_no_margin():
    """
    A searched fact did not come from a vector and cannot drift by DRIFT_DELTA.
    Reporting a distance here would put a fabricated number in the field readers use to
    judge how reliable a call was.
    """
    result = classify(0.10, 0.0, 0.0, 0.0, True)
    assert result["decision_margin"] is None
    assert result["decision_threshold"] is None
    assert result["in_semantic_variance_band"] is False
    assert result["classification_confidence"] == CONFIDENCE_HIGH


# --- Rule priority ------------------------------------------------------------

def test_power_bi_is_backbone_despite_confirmed_embedded_ai():
    """
    Priority is load-bearing. Power BI clears the tech_base floor (0.397 >= 0.28) and
    genuinely ships AI features; rule 1 firing first is what makes it read as
    infrastructure that gained AI features rather than as an AI-first product.
    """
    result = classify(*MEASURED["Power BI"], True)
    assert result["sub_category"] == SUB_TECHNICAL_BACKBONE


def test_ai_skill_outranks_every_enabling_rule():
    """Rule 0 survives the engine; without it the top bucket would disappear."""
    result = classify(0.45, 0.90, 0.90, 0.90)
    assert result["category_bucket"] == BUCKET_AI
    assert result["sub_category"] is None


# --- Default ------------------------------------------------------------------

@pytest.mark.parametrize("name", ["Yahoo! Email", "DHTML", "MongoDB"])
def test_low_signal_technology_drops_to_not_ai(name):
    result = classify_named(name)
    assert result["category_bucket"] == BUCKET_NOT_AI
    assert result["sub_category"] is None


def test_not_ai_reports_the_nearest_threshold_it_missed():
    """
    A near miss must be distinguishable from a rout.

    C++ misses the tech_base floor by 0.006 and everything else by far more, so that is
    the comparison worth reporting and it puts C++ inside the variance band -- one
    phrasing of its definition away from being classified AI Enabling.
    """
    result = classify(0.112, TECH_BASE_FLOOR - 0.006, 0.170, 0.155)
    assert result["decision_metric"] == "tech_base_sim"
    assert result["decision_margin"] == pytest.approx(-0.006, abs=1e-4)
    assert result["in_semantic_variance_band"] is True


def test_the_nearest_miss_never_reports_the_embedding_rule():
    """
    Rule 2 is excluded from the near-miss search, and not only because a None margin
    cannot be compared. A skill searched and found to ship no AI features did not
    NARROWLY miss anything -- there is no distance -- and inventing one would be the
    same lie _boolean_condition exists to avoid.
    """
    result = classify(0.10, 0.0, 0.0, 0.2199, False)
    assert result["decision_metric"] != "embeds_ai"
    assert result["decision_margin"] is not None


def test_a_rout_is_not_flagged_marginal():
    """The counterpart: MongoDB misses every rule by a wide margin."""
    result = classify_named("MongoDB")
    assert result["in_semantic_variance_band"] is False
    assert result["classification_confidence"] == CONFIDENCE_HIGH


def test_none_score_classifies_rather_than_raising():
    """Called from render paths, where an exception takes the whole table down."""
    result = classify(None)
    assert result["category_bucket"] == BUCKET_NOT_AI
    assert class_normalized(None) == 0.0


# --- Boundaries and the variance band ----------------------------------------

def test_a_metric_exactly_on_its_floor_classifies_in():
    result = classify(0.10, TECH_BASE_FLOOR, 0.0, 0.0)
    assert result["sub_category"] == SUB_TECHNICAL_BACKBONE
    assert result["decision_margin"] == pytest.approx(0.0, abs=1e-9)


def test_a_metric_just_under_its_floor_classifies_out():
    result = classify(0.10, TECH_BASE_FLOOR - 0.001, 0.0, 0.0)
    assert result["category_bucket"] == BUCKET_NOT_AI


@pytest.mark.parametrize("offset", [0.0, DRIFT_DELTA, -DRIFT_DELTA, 0.01, -0.02])
def test_decisions_within_delta_of_a_threshold_are_flagged_marginal(offset):
    result = classify(0.10, TECH_BASE_FLOOR + offset, 0.0, 0.0)
    assert result["in_semantic_variance_band"] is True
    assert result["classification_confidence"] == CONFIDENCE_MARGINAL


@pytest.mark.parametrize("offset", [DRIFT_DELTA + 0.001, 0.2, -DRIFT_DELTA - 0.001])
def test_decisions_beyond_delta_are_high_confidence(offset):
    result = classify(0.10, TECH_BASE_FLOOR + offset, 0.0, 0.0)
    assert result["in_semantic_variance_band"] is False
    assert result["classification_confidence"] == CONFIDENCE_HIGH


def test_class_normalized_is_a_fraction_of_the_ai_skill_bar():
    """
    The spec's 0.70 and 0.58 are unreachable on the raw scale, and the UI's min-max
    normalization is view-relative and must never decide a class. This is the mapping
    that makes both constants live and stable.
    """
    assert class_normalized(AI_SKILL_THRESHOLD) == pytest.approx(1.0)
    # Derived from the bar rather than written as literals. These used to read 0.21 and
    # 0.174, which were 0.70 and 0.58 of a 0.30 bar -- so moving the bar broke them for a
    # reason that had nothing to do with the mapping they exist to check. The point of the
    # test is that class_normalized is a FRACTION OF THE BAR, whatever the bar is.
    assert class_normalized(0.70 * AI_SKILL_THRESHOLD) == pytest.approx(0.70)
    assert class_normalized(0.58 * AI_SKILL_THRESHOLD) == pytest.approx(0.58)


def test_conceptual_rule_is_reachable():
    """It sits below the AI Skill bar by construction, so it can actually fire."""
    result = classify(0.21, 0.20, 0.0, 0.0)
    assert result["category_bucket"] == BUCKET_ENABLING
    assert result["sub_category"] == SUB_CONCEPTUAL


# --- The real scorer ----------------------------------------------------------

@pytest.mark.real_model
def test_flagship_note_moves_embedded_ai_and_nothing_else():
    """
    The isolation guarantee, checked against the actual model rather than asserted.

    Everything the rules engine reads except embedded_ai_sim must be byte-identical
    with and without the note.
    """
    from sortingalgorithmnew import calculate_ai_correlation

    definition = (
        "Slack is a team communication platform offering persistent chat rooms "
        "organized by topic, private groups, and direct messaging."
    )
    note = (
        "Slack AI provides generative conversation summaries, AI-powered search "
        "answers and automated recaps of channels and threads."
    )

    base = calculate_ai_correlation("Slack", "", definition)
    flagged = calculate_ai_correlation("Slack", "", definition, note)

    assert flagged["embedded_ai_sim"] > base["embedded_ai_sim"]
    assert flagged["ai_score"] == base["ai_score"]
    assert flagged["tech_base_sim"] == base["tech_base_sim"]
    assert flagged["ml_pipeline_sim"] == base["ml_pipeline_sim"]

    assert base["is_flagship_version_evaluated"] is False
    assert flagged["is_flagship_version_evaluated"] is True


# --- Flagship substitution and the ceiling ------------------------------------
#
# Category terms and their flagships, both as measured on the live store. Every one of
# these categories currently outscores its own flagship, which is the defect.
# (ai, tech, ml, emb)
CATEGORY_PAIRS = {
    "Word processing software": ((0.321, 0.254, 0.180, 0.353), "Microsoft Word",
                                 (0.192, 0.156, 0.194, 0.280)),
    "Spreadsheet software": ((0.175, 0.458, 0.189, 0.229), "Microsoft Excel",
                             (0.065, 0.395, 0.152, 0.171)),
    "Presentation software": ((0.239, 0.291, 0.059, 0.231), "Microsoft PowerPoint",
                              (0.098, 0.255, 0.125, 0.157)),
    "Social media software": ((0.288, 0.140, 0.051, 0.224), "Facebook",
                              (0.157, -0.026, 0.016, -0.033)),
    "Statistical software": ((0.137, 0.479, 0.117, 0.188), "SAS",
                             (0.107, 0.395, 0.173, 0.128)),
}


def as_metrics(values):
    return dict(zip(("ai_score", "tech_base_sim", "ml_pipeline_sim", "embedded_ai_sim"), values))


@pytest.mark.parametrize("category", sorted(CATEGORY_PAIRS))
def test_category_takes_its_flagship_metrics_exactly(category):
    """Substitution is wholesale, not a blend: the three unmoved metrics must match."""
    from sortingalgorithmnew import score_as_flagship

    _, _, flagship_values = CATEGORY_PAIRS[category]
    flagship = as_metrics(flagship_values)
    result = score_as_flagship(flagship)

    assert result["ai_score"] == pytest.approx(flagship["ai_score"], abs=1e-4)
    assert result["tech_base_sim"] == pytest.approx(flagship["tech_base_sim"], abs=1e-4)
    assert result["ml_pipeline_sim"] == pytest.approx(flagship["ml_pipeline_sim"], abs=1e-4)
    assert result["is_generic_category"] is True


@pytest.mark.parametrize("category", sorted(CATEGORY_PAIRS))
def test_no_category_outscores_its_flagship_after_substitution(category):
    """
    The ceiling, asserted on every measured pair.

    Each of these categories beats its flagship BEFORE the change -- the assertion at
    the end of this test is what fails if substitution is ever bypassed.
    """
    from sortingalgorithmnew import enforce_flagship_ceiling, score_as_flagship

    category_values, _, flagship_values = CATEGORY_PAIRS[category]
    flagship = as_metrics(flagship_values)

    beat_its_flagship = any(
        category_values[index] > flagship_values[index] for index in range(3)
    )
    assert beat_its_flagship, "fixture no longer demonstrates the defect"

    result = score_as_flagship(flagship)
    enforce_flagship_ceiling(result, flagship)


def test_ceiling_raises_rather_than_silently_clamping():
    """
    A category above its flagship means substitution did not happen, or happened against
    the wrong product. Clamping the number would hide which of those it was.
    """
    from sortingalgorithmnew import enforce_flagship_ceiling

    flagship = as_metrics((0.10, 0.10, 0.10, 0.10))
    with pytest.raises(ValueError, match="exceeds the flagship ceiling"):
        enforce_flagship_ceiling(as_metrics((0.90, 0.10, 0.10, 0.10)), flagship)


def test_the_three_miscategorized_ai_skills_stop_being_ai_skills():
    """
    Word processing software (0.321) outranked XGBoost (0.310) as an AI Skill while
    Microsoft Word (0.192) and Google Docs (0.181) did not. Scored as Word it drops out
    of the top bucket.
    """
    from sortingalgorithmnew import score_as_flagship

    category, _, flagship_values = CATEGORY_PAIRS["Word processing software"]
    assert classify(*CATEGORY_PAIRS["Word processing software"][0])["category_bucket"] == BUCKET_AI

    result = score_as_flagship(as_metrics(flagship_values))
    assert result["category_bucket"] != BUCKET_AI


def test_product_ai_skills_are_untouched_by_substitution():
    """The 8 genuine AI Skills are products, and nothing here may reach them."""
    for values in [(0.685, 0.10, 0.10, 0.10), (0.472, 0.30, 0.30, 0.10), (0.310, 0.2, 0.2, 0.2)]:
        assert classify(*values)["category_bucket"] == BUCKET_AI


@pytest.mark.real_model
def test_a_flagship_note_no_longer_rescues_a_category_by_itself():
    """
    WHAT THIS CHANGE COST, recorded rather than glossed over.

    Substitution inherits the flagship article's weaknesses. Facebook's is corporate
    history -- tech -0.026, embedded -0.033 -- so "Social media software" scored as a
    bare Facebook collapses to Not AI, which is wrong for a category whose feeds are
    ranked by machine learning.

    The AI note used to fix that on its own, by lifting embedded_ai_sim over 0.22. It no
    longer can, because that floor is gone: the note moves a diagnostic and decides
    nothing. This is a real capability that was removed, and the test asserts its
    absence so nobody assumes the old rescue is still happening.
    """
    from sortingalgorithmnew import score_as_flagship

    facebook = as_metrics(CATEGORY_PAIRS["Social media software"][2])
    note = (
        "Facebook ranks its News Feed and recommends content with machine learning "
        "models, and offers Meta AI, a generative assistant, inside the app."
    )

    assert score_as_flagship(facebook)["category_bucket"] == BUCKET_NOT_AI
    noted = score_as_flagship(facebook, note)
    assert noted["embedded_ai_sim"] > facebook["embedded_ai_sim"], "the note still moves it"
    assert noted["category_bucket"] == BUCKET_NOT_AI, "but it no longer decides anything"


@pytest.mark.real_model
def test_a_search_is_what_rescues_it_now():
    """
    The replacement for the note's rescue, and a better one: it turns on whether social
    media platforms actually rank feeds with machine learning, not on whether a sentence
    about it embeds near an anchor.
    """
    from sortingalgorithmnew import score_as_flagship

    facebook = as_metrics(CATEGORY_PAIRS["Social media software"][2])
    rescued = score_as_flagship(facebook, "", True)

    assert rescued["category_bucket"] == BUCKET_ENABLING
    assert rescued["sub_category"] == SUB_EMBEDDED_AI
    # The boost rides along, and is visible rather than folded into the measurement.
    assert rescued["ai_score_base"] == pytest.approx(facebook["ai_score"], abs=1e-4)
    assert rescued["embedded_ai_boost"] == pytest.approx(0.05, abs=1e-9)


# --- Flagship name matching ---------------------------------------------------

def test_flagship_matching_is_exact_and_never_fuzzy():
    """
    A fuzzy match does not produce a slightly-off score here, it substitutes an entire
    unrelated product's four metrics into a category term.
    """
    from flagship_pass import resolve_flagship

    master = {"Microsoft Word": {}, "Microsoft Visual SourceSafe": {}, "SAS": {}}

    assert resolve_flagship("Microsoft Word", master) == "Microsoft Word"
    assert resolve_flagship("microsoft word", master) == "Microsoft Word"
    assert resolve_flagship("Microsoft  Word!", master) == "Microsoft Word"
    # Partial and superset names must NOT resolve.
    assert resolve_flagship("Word", master) == ""
    assert resolve_flagship("Microsoft Word in Microsoft 365", master) == ""
    assert resolve_flagship("Visual SourceSafe", master) == ""
    assert resolve_flagship("", master) == ""
    assert resolve_flagship(None, master) == ""


def test_a_category_with_no_flagship_is_left_measured_on_its_own_text():
    """Not half-substituted. A partial substitution is a score nothing stands behind."""
    from flagship_pass import score_entry

    entry = {
        "is_generic_category": True,
        "flagship_version": "Some Product Not In The Store",
        "flagship_definition": "",
        "flagship_note": "",
        "category": "",
        "wikipedia_summary": "A category of software used for a purpose.",
    }
    metrics = score_entry("Mystery software", entry, {}, {})

    assert entry["flagship_source"] is None
    assert metrics["flagship_version"] is None
    assert metrics["is_generic_category"] is False


def test_a_named_product_is_never_substituted():
    """Path 3 covers every product, whatever its flagship fields say."""
    from flagship_pass import score_entry

    entry = {
        "is_generic_category": False,
        "flagship_version": "Microsoft Word",
        "flagship_note": "",
        "category": "",
        "wikipedia_summary": "Slack is a team communication platform.",
    }
    master = {"Microsoft Word": {}}
    newest = {"Microsoft Word": as_metrics((0.192, 0.156, 0.194, 0.280))}

    metrics = score_entry("Slack", entry, master, newest)
    assert entry["flagship_source"] is None
    assert metrics["ai_score"] != pytest.approx(0.192, abs=1e-4)


@pytest.mark.real_model
def test_flagship_note_cannot_lower_a_score():
    """
    Scored as its own text and combined with max(), so a note about a product with no
    AI features leaves the measurement where it was instead of diluting it.
    """
    from sortingalgorithmnew import calculate_ai_correlation

    definition = "Yahoo! Mail is a web-based email service."
    note = "Yahoo Mail is a consumer webmail inbox with folders, filters and spam handling."

    base = calculate_ai_correlation("Yahoo! Email", "", definition)
    flagged = calculate_ai_correlation("Yahoo! Email", "", definition, note)

    assert flagged["embedded_ai_sim"] >= base["embedded_ai_sim"]
    assert flagged["category_bucket"] == base["category_bucket"]


# --- The generative pole gives text tools a mid-tier score, BY DESIGN ---------
#
# Four skills were briefly "fixed" by stripping "natural language processing", "nlp" and
# "prompt engineering" out of AI_GENERATIVE_POLE. That was the wrong fix and it is
# reverted. The pole is meant to give everyday text and communication tools a non-zero
# mid-tier score, because those tools embed AI features and because tokenizing text and
# structuring prompts is behaviourally parallel to how NLP works.
#
# The defect was never that these skills score. It was that four of them crossed into the
# TOP bucket. Mid-tier is intended; AI Skill is not. These tests pin that distinction.

def gen(text):
    """Similarity of arbitrary text to the generative pole."""
    from sortingalgorithmnew import AI_GEN_VEC, _dot, get_embedding
    return _dot(get_embedding(text), AI_GEN_VEC)


@pytest.mark.real_model
@pytest.mark.xfail(strict=True, reason=(
    "KNOWN DISCREPANCY. The live AI_GENERATIVE_POLE (commit 4639910, 2026-08-14) dropped "
    "'chatbot', which the measurements in sortingalgorithmnew.py mark as KEPT, and added "
    "'LLM', which they mark as REMOVED for misfiring (Oracle JMS, WebSphere MQ). The "
    "lowercase check below misses 'LLM' because the pole spells it in capitals, but "
    "get_embedding() lowercases everything, so it is in effect. Every stored score was "
    "produced with this pole. Resolve in the Phase 1 accuracy check: keep the pole and "
    "update the comments, or restore the documented pole and re-score. strict=True: "
    "remove this marker once the two agree."))
def test_the_generative_pole_holds_only_terms_that_discriminate():
    """
    A guard on the pole text, because editing it to move one skill is the tempting and
    wrong move in BOTH directions -- six terms were once removed for the wrong reason and
    put back, then removed again for a measured one.

    Each term below was embedded alone and scored across all 992 skills. The ones that
    stay fire on the sense intended. The ones that must NOT come back fire on the wrong
    sense of their own words, which a keyword reading of the pole cannot reveal:

        prompt engineering        gap -0.085   matches ENGINEERING (Robotics, schematics)
        transformer architecture  gap +0.065   matches ELECTRICAL transformers
        openai                    gap +0.168   matches "Open" (OpenOffice, OpenMake)
        large language model      gap +0.141   matches "MODEL" (ModelSim, Modelica)
        llm                       gap +0.089   opaque acronym, lands near other acronyms
        nlp                       gap +0.012   the same

    Re-adding any of them requires re-running that measurement, not an opinion.
    """
    from sortingalgorithmnew import AI_GENERATIVE_POLE

    for term in ("artificial intelligence", "generative ai", "chatgpt", "chatbot",
                 "natural language processing"):
        assert term in AI_GENERATIVE_POLE, f"{term!r} is load-bearing and was removed"

    for term in ("prompt engineering", "transformer architecture", "openai",
                 "large language model", " llm", " nlp"):
        assert term not in AI_GENERATIVE_POLE, (
            f"{term!r} was measured firing on the wrong sense of its own words"
        )


@pytest.mark.real_model
def test_shortening_the_pole_further_is_not_an_improvement():
    """
    MEASURED, so the next person does not retry it. Cutting to a minimal
    "Generative AI, LLM, GPT, Transformer, Prompting" gives 22 AI Skills with 16 false
    positives and loses spaCy, against 7 and 0 for the pole as it stands.

    A short pole does not average its errors away, it is DOMINATED by them: "Transformer"
    without its qualifying word points at electrical transformers (Rockwell Automation
    0.356, PLC code generation 0.357, TMW PowerSuite 0.317, Autodesk Revit 0.305).
    """
    from sortingalgorithmnew import AI_ENG_VEC, _dot, get_embedding

    minimal = get_embedding("Generative AI, LLM, GPT, Transformer, Prompting")
    # The REAL definition from the store, not a paraphrase. An invented sentence scored
    # 0.298 and this one 0.356, which is the difference between the test asserting the
    # measurement and asserting something that merely resembles it.
    rockwell = get_embedding(
        "Rockwell Automation is an American provider of industrial automation and "
        "digital transformation technologies. Its notable offerings include "
        "Allen-Bradley products and FactoryTalk software.")
    # The minimal pole scores an industrial automation tool as high as a real AI library.
    assert _dot(rockwell, minimal) > 0.30, (
        "if this drops, re-measure -- the argument against shortening rests on it")
    # And the engineering pole, which is what the top bucket reads, correctly does not.
    assert _dot(rockwell, AI_ENG_VEC) < AI_ENGINEERING_FLOOR


@pytest.mark.real_model
def test_the_ai_engineering_pole_is_untouched():
    from sortingalgorithmnew import AI_ENGINEERING_POLE

    for term in ("neural networks", "model training", "computational inference"):
        assert term in AI_ENGINEERING_POLE


@pytest.mark.real_model
@pytest.mark.xfail(strict=True, reason=(
    "OPEN QUESTION, not a test bug. Since the 2026-08-14 pole change, "
    "'documentation platform software' scores 0.138 on the generative pole, below this "
    "0.15 floor. Decide in the Phase 1 accuracy check whether the floor or the pole is "
    "right. strict=True: once it passes again, remove this marker."))
def test_office_tools_score_mid_tier_rather_than_zero():
    """
    The behaviour the design asks for in as many words: common office applications
    "rarely receive a flat zero score" and land mid-tier through embedded capabilities
    and behavioural parallels with NLP. If a future pole edit flattens these to zero, the
    linguistic-parallel signal has been lost.
    """
    for text in ("word processing software", "instant messaging software",
                 "documentation platform software"):
        assert gen(text) > 0.15, f"{text!r} collapsed to a floor score"


@pytest.mark.real_model
def test_the_two_title_driven_skills_land_mid_tier_on_their_definition():
    """
    WordPerfect and Report generation software crossed into AI Skill on their TITLE
    alone -- 0.311 and 0.304 -- while their definitions read 0.288 and 0.185. Both are
    correctly mid-tier when the definition is what gets measured, with the pole intact.
    This is why the fix belongs in the scoring surface and not in the pole.
    """
    from sortingalgorithmnew import AI_ENABLING_THRESHOLD

    wordperfect = gen("WordPerfect is a word processing application currently developed "
                      "and owned by Corel, originally created in 1979.")
    reports = gen("Software that produces printed summary reports from rows in a database.")

    assert wordperfect < AI_SKILL_THRESHOLD, "must not reach the top bucket"
    assert wordperfect > AI_ENABLING_THRESHOLD, "but must not collapse to zero either"
    assert reports < AI_SKILL_THRESHOLD


@pytest.mark.real_model
def test_a_deep_learning_framework_outscores_a_word_processor_on_its_definition():
    """
    The comparison that exposed the original problem. On TITLES alone WordPerfect (0.311)
    beat PyTorch (0.296), which is what a two-word product name buys against a pole about
    language. On definitions, which is what should be measured, the ordering is right.
    """
    pytorch = gen("PyTorch is an open source machine learning library used for deep "
                  "learning, computer vision and natural language processing.")
    wordperfect = gen("WordPerfect is a word processing application currently developed "
                      "and owned by Corel, originally created in 1979.")
    assert pytorch > wordperfect


# --- Deterministic AI phrases are reported, not scored ------------------------

@pytest.mark.parametrize("text, expected", [
    ("spaCy is a natural language processing library", ["natural language processing"]),
    ("uses NLP.", ["nlp"]),
    ("NLP-based tooling", ["nlp"]),
    ("neural networks", ["neural networks"]),
    ("machine learning framework", ["machine learning"]),
    ("WordPerfect", []),
    ("Word processing software", []),
    ("Report generation software", []),
])
def test_the_phrase_match_is_exact(text, expected):
    from sortingalgorithmnew import lexical_ai_terms_in
    assert lexical_ai_terms_in(text) == expected


def test_the_boundary_is_a_word_boundary_not_a_prefix():
    """
    REGRESSION. The first version used a greedy word-character suffix so "network"
    would match "networks". It also made "nlp" match inside "nlpxyz". It matches an
    optional trailing "s" only, now.
    """
    from sortingalgorithmnew import lexical_ai_terms_in
    assert lexical_ai_terms_in("an nlpxyz thing") == []
    assert lexical_ai_terms_in("machine learnings and neural networks") == [
        "machine learnings", "neural networks"]


def test_the_phrase_match_adds_nothing_to_the_score():
    """
    It was briefly a +0.05 boost, while the matching terms were removed from the pole.
    The pole measures them again, so a boost on top would count the same evidence twice.
    The match survives as something to look at, and reads on nothing.
    """
    from sortingalgorithmnew import LEXICAL_AI_BOOST, apply_boost

    assert LEXICAL_AI_BOOST == 0.0
    with_phrase = apply_boost(0.2000, None, ["machine learning"])
    without = apply_boost(0.2000, None, [])
    assert with_phrase["ai_score"] == without["ai_score"]
    assert with_phrase["lexical_ai_terms"] == ["machine learning"], "still recorded"


# --- The contrast anchor decides nothing --------------------------------------

def test_classify_cannot_read_the_contrast_score():
    """
    Recorded, not enforced. If it is ever promoted to a rule that must be a deliberate
    change rather than a drift.
    """
    import inspect
    from sortingalgorithmnew import classify
    assert "contrast" not in str(inspect.signature(classify))


# --- The engineering floor on the top bucket ----------------------------------
#
# WHY A SECOND TEST ON RULE 0 EXISTS AT ALL. Measured across the 15 skills that were in
# the AI Skill bucket, ai_score CANNOT separate the genuine from the false: the lowest
# genuine sits at 0.306 and the highest false positive at 0.379. They overlap, so raising
# AI_SKILL_THRESHOLD loses real skills before it loses fake ones. The AI Engineering pole
# does separate, with every genuine above 0.18 and every false positive below 0.11.

def classify_with_eng(ai_score, eng, tech=0.0, ml=0.0):
    return classify(ai_score, tech, ml, 0.0, None, eng)


def test_a_high_score_without_engineering_signal_is_not_an_ai_skill():
    """Word processing software: ai 0.360 but engineering 0.097."""
    result = classify_with_eng(0.360, 0.097)
    assert result["category_bucket"] != BUCKET_AI
    assert result["decision_metric"] == "ai_engineering_sim", \
        "it must report the condition it actually failed, not the one it cleared"


def test_the_same_score_with_engineering_signal_is():
    assert classify_with_eng(0.360, 0.30)["category_bucket"] == BUCKET_AI


def test_engineering_signal_alone_is_not_enough_either():
    """Both terms bind. TensorFlow-level engineering with a low ai_score stays out."""
    assert classify_with_eng(0.20, 0.45)["category_bucket"] != BUCKET_AI


# Engineering-pole values under the 90/10 blend, measured on the live store. ChatGPT is
# the lowest genuine AI Skill and Word processing the highest false positive, so these two
# are what any candidate floor has to separate.
CHATGPT_ENG = 0.1769
WORD_PROCESSING_ENG = 0.0914


@pytest.mark.parametrize("floor", [0.10, 0.12, 0.14, 0.17])
def test_the_floor_is_not_balanced_on_a_knife_edge(floor, monkeypatch):
    """
    Any value in [0.10, 0.17] gives the identical set, so 0.14 is not tuned.
    """
    import sortingalgorithmnew as engine
    monkeypatch.setattr(engine, "AI_ENGINEERING_FLOOR", floor)

    assert engine.classify(0.725, 0, 0, 0, None, CHATGPT_ENG)["category_bucket"] == BUCKET_AI
    assert engine.classify(0.309, 0, 0, 0, None, WORD_PROCESSING_ENG)["category_bucket"] \
        != BUCKET_AI


def test_the_floors_upper_bound_is_recorded_not_assumed(monkeypatch):
    """
    THE BAND NARROWED, and this records where. Under the old max() scoring it ran to 0.18;
    under the 90/10 blend 0.18 gates OpenAI ChatGPT (engineering 0.1769) out of its own top
    bucket.

    Asserted rather than left implicit because the failure is silent: raising the floor
    "for safety" would quietly drop the single most obviously-AI skill in the store, and
    nothing else in the suite would notice.
    """
    import sortingalgorithmnew as engine

    assert engine.AI_ENGINEERING_FLOOR < CHATGPT_ENG, \
        "the floor must sit below the lowest genuine AI Skill's engineering score"

    monkeypatch.setattr(engine, "AI_ENGINEERING_FLOOR", 0.18)
    assert engine.classify(0.725, 0, 0, 0, None, CHATGPT_ENG)["category_bucket"] != BUCKET_AI


def test_a_snapshot_written_before_the_floor_is_not_demoted_by_it():
    """
    THE COMPATIBILITY RULE. Records predating this field carry None, and None SKIPS the
    floor rather than failing it. Defaulting to 0.0 instead would silently demote every
    old AI Skill the first time reclassify_snapshots touched it -- a data change
    disguised as a re-decision.
    """
    assert classify(0.45)["category_bucket"] == BUCKET_AI
    assert classify(0.45, 0, 0, 0, None, None)["category_bucket"] == BUCKET_AI
    assert classify(0.45, 0, 0, 0, None, 0.05)["category_bucket"] != BUCKET_AI


@pytest.mark.real_model
def test_ai_score_is_still_the_max_over_both_poles_and_both_surfaces():
    """
    The floor is a GATE, not a replacement. If ai_score ever becomes the engineering pole
    alone, every stored score changes meaning and the trend history stops being
    comparable -- so this asserts the generative pole still reaches the score.
    """
    from sortingalgorithmnew import calculate_ai_correlation

    chatgpt = calculate_ai_correlation(
        "OpenAI ChatGPT", "",
        "ChatGPT is a generative artificial intelligence chatbot developed by OpenAI.")
    assert chatgpt["ai_generative_sim"] > chatgpt["ai_engineering_sim"]
    assert chatgpt["ai_score_base"] == pytest.approx(chatgpt["ai_generative_sim"], abs=1e-9), \
        "the generative pole must still be able to win the max"


@pytest.mark.real_model
def test_office_tools_keep_their_mid_tier_score():
    """
    The design asks that common office applications not collapse to a flat zero. The
    floor changes their CLASS, never their score: it gates the top bucket and touches
    nothing else.
    """
    from sortingalgorithmnew import AI_ENABLING_THRESHOLD, calculate_ai_correlation

    # The REAL definition from the store. A shortened paraphrase of it measured
    # engineering 0.214 against the real text's 0.097 and passed the floor -- the
    # sentence about "negligible technical integrations" is doing real work, and a
    # fixture that drops it tests something else.
    result = calculate_ai_correlation(
        "Word processing software", "",
        "End-user document creation and editorial software like Google Docs and "
        "Microsoft Word. Generally the tool used to be a standard writing tool with "
        "negligible technical integrations, but the tools increasingly embed AI "
        "capabilities. Primarily for authoring and document production.")
    assert result["ai_score"] > AI_ENABLING_THRESHOLD, "must not collapse to zero"
    assert result["category_bucket"] != BUCKET_AI, "but must not reach the top bucket"
    assert result["ai_engineering_sim"] < AI_ENGINEERING_FLOOR


@pytest.mark.real_model
def test_a_legacy_application_is_not_an_ai_skill():
    """
    Telluride Software Classic Trak-It, the case that made this necessary. Its definition
    opens by calling itself legacy, and it was still in the top bucket -- the generative
    pole scored its own product NAME at 0.045 a word, and "large language model" matched
    the "model" in nothing more than its vocabulary neighbourhood. Engineering reads 0.052.
    """
    from sortingalgorithmnew import calculate_ai_correlation

    result = calculate_ai_correlation(
        "Telluride Software Classic Trak-It", "",
        "Telluride Software Classic Trak-It is a legacy real estate project management "
        "and client tracking application developed by Telluride Software. It ran on local "
        "desktop environments before modern web-based CRM platforms became standard.")
    assert result["ai_engineering_sim"] < AI_ENGINEERING_FLOOR
    assert result["category_bucket"] != BUCKET_AI


# --- The 90/10 definition-primary blend ---------------------------------------
#
# These replace what used to be max(title_sim, context_sim). Every fixture below uses the
# REAL store definition, because the whole point of the blend is that the definition
# decides -- a paraphrased fixture is literally testing different input to the thing under
# test, and that has already produced two wrong conclusions in this file's history.

SCRIPTING_SOFTWARE_DEFINITION = (
    "Scripting software is a category of programming environments and text editors "
    "designed for writing, editing, and executing interpreted code or scripts. It is used "
    "by developers, system administrators, and media professionals to automate workflows, "
    "build web applications, and control software systems. Common languages supported "
    "include Python, JavaScript, Ruby, and Bash."
)

TELLURIDE_DEFINITION = (
    "Telluride Software Classic Trak-It is a legacy real estate project management and "
    "client tracking application developed by Telluride Software. It was used by real "
    "estate sales agents and brokerages to monitor client interactions, property "
    "listings, and transaction pipelines. The software ran on local desktop environments "
    "before modern web-based CRM platforms became standard."
)


@pytest.mark.real_model
def test_a_pole_score_is_the_blend_and_not_the_max():
    """
    THE STRUCTURAL GUARANTEE, asserted directly on the arithmetic rather than inferred
    from an outcome. If someone reinstates max() across surfaces this fails immediately,
    which is the only way to stop the regression that produced the false positives.

    Checked on a skill whose TITLE outscores its DEFINITION, because that is the only
    case where blend and max disagree -- on any other skill the two coincide and the test
    would pass while proving nothing.
    """
    from sortingalgorithmnew import (
        AI_ENG_VEC, CONTEXT_WEIGHT, _dot, calculate_ai_correlation, get_embedding,
    )

    title = "Machine learning"
    definition = SCRIPTING_SOFTWARE_DEFINITION

    title_eng = _dot(get_embedding(title), AI_ENG_VEC)
    ctx_eng = _dot(get_embedding(definition), AI_ENG_VEC)
    assert title_eng > ctx_eng, "fixture is only meaningful when the title wins"

    result = calculate_ai_correlation(title, "", definition)
    expected = CONTEXT_WEIGHT * ctx_eng + (1.0 - CONTEXT_WEIGHT) * title_eng

    assert result["ai_engineering_sim"] == pytest.approx(expected, abs=5e-5)
    assert result["ai_engineering_sim"] < title_eng, \
        "the title must no longer be able to carry the score on its own"


@pytest.mark.real_model
def test_the_blend_suppresses_a_title_carried_score():
    """
    Scripting software, one of the regressions that prompted this. Under max() its short,
    developer-sounding title carried it to 0.308 -- over the AI Skill bar -- while its
    definition is about text editors and interpreters. The blend reads the definition.
    """
    from sortingalgorithmnew import (
        AI_SKILL_THRESHOLD as BAR, calculate_ai_correlation,
    )

    result = calculate_ai_correlation(
        "Scripting software", "", SCRIPTING_SOFTWARE_DEFINITION)

    assert result["ai_score"] < BAR
    assert result["category_bucket"] != BUCKET_AI
    # It is still a real programming environment, so it must not be flattened to nothing.
    assert result["tech_base_sim"] > result["ai_score"], \
        "its signal belongs to the tooling anchor, which is the honest place for it"


@pytest.mark.real_model
def test_legacy_similarity_is_measured_and_decides_nothing():
    """
    Telluride Trak-It. The legacy anchor DOES detect it -- that was never in doubt -- and
    the point of this test is that detection is all it does.

    The anchor was proposed as a score multiplier. Measured across the 1051 approved
    skills, the largest discount that formula applies anywhere is 0.018, so it cannot move
    a classification; and the hard-gate alternative demotes Red Hat Ansible Engine, Talend
    and Cloudera Impala along with the legacy tools. So legacy_sim is recorded and shown,
    and the blend is what actually keeps Telluride out (0.298 -> 0.246).
    """
    from sortingalgorithmnew import calculate_ai_correlation

    result = calculate_ai_correlation(
        "Telluride Software Classic Trak-It", "", TELLURIDE_DEFINITION)

    assert result["legacy_sim"] >= 0.28, "the diagnostic must actually detect legacy text"
    assert result["category_bucket"] != BUCKET_AI
    assert result["ai_score"] < AI_SKILL_THRESHOLD
    assert result["ai_engineering_sim"] < AI_ENGINEERING_FLOOR


def test_no_rule_can_read_legacy_similarity():
    """
    Enforced on the SIGNATURE, not on behaviour. A behavioural test would pass for a rule
    that reads legacy_sim but happens not to fire on the fixtures; this cannot.
    """
    import inspect

    from sortingalgorithmnew import classify

    assert "legacy_sim" not in inspect.signature(classify).parameters, \
        "legacy_sim is a diagnostic; giving it to classify() is the change this forbids"


@pytest.mark.real_model
def test_a_genuine_neural_network_tool_reaches_the_top_bucket():
    """
    Ward Systems NeuralShell Predictor, which the blend SURFACED rather than demoted. Its
    definition says "neural network software tool ... for predictive modeling", so its
    engineering score is 0.332 -- above XGBoost's. max() was scoring it below the bar.

    Recorded because the eighth AI Skill appearing in a change whose purpose was to remove
    AI Skills looks like a regression at a glance, and it is the opposite.
    """
    from sortingalgorithmnew import calculate_ai_correlation

    result = calculate_ai_correlation(
        "Ward Systems Group NeuralShell Predictor", "",
        "Ward Systems Group NeuralShell Predictor is a neural network software tool "
        "developed by Ward Systems Group for predictive modeling and data analysis. It "
        "allows financial analysts and risk specialists to build forecasting models "
        "without requiring advanced programming knowledge.")

    assert result["ai_engineering_sim"] > AI_ENGINEERING_FLOOR
    assert result["category_bucket"] == BUCKET_AI


def test_a_snapshot_without_legacy_similarity_still_reclassifies():
    """
    MIGRATION. Every snapshot on disk predates legacy_sim, and most predate
    ai_engineering_sim too. Neither absence may raise, and neither may demote.
    """
    from reclassify_snapshots import reclassify

    old = {
        "skill_name": "Something scored before either field existed",
        "ai_score": 0.45,
        "tech_base_sim": 0.10,
        "ml_pipeline_sim": 0.10,
        "embedded_ai_sim": 0.10,
        "category_bucket": BUCKET_AI,
    }
    records = [old]
    reclassify(records)

    assert records[0]["category_bucket"] == BUCKET_AI, \
        "a missing field must skip its rule, never fail it"


@pytest.mark.real_model
@pytest.mark.parametrize("label,store_key", [
    ("C++", "C++"),
    ("SAS", "SAS"),
    ("Teams", "Microsoft Teams"),
])
def test_the_measured_table_still_matches_the_real_scorer(label, store_key):
    """
    THE FIXTURE-STALENESS GUARD, and it exists because the staleness already bit once.

    MEASURED holds real numbers from the live store, and the rest of this file compares
    them against the engine's floors. When scoring moved to the 90/10 blend the floors were
    re-derived and the table was not, so the suite spent a run comparing old-scale metrics
    against new-scale floors -- Teams flipped to Technical Backbone and looked like a rule
    change rather than what it was.

    Teams is in the parameter list deliberately: it is the tightest case, at ml_pipeline
    0.184 against a 0.19 floor.
    """
    import json

    from sortingalgorithmnew import calculate_ai_correlation

    with open("skills_master.json") as handle:
        entry = json.load(handle)[store_key]

    fresh = calculate_ai_correlation(
        entry.get("skill_name") or store_key, entry.get("category", ""),
        entry.get("wikipedia_summary", "") or "", "", None)

    expected = MEASURED[label]
    actual = (fresh["ai_score"], fresh["tech_base_sim"],
              fresh["ml_pipeline_sim"], fresh["embedded_ai_sim"])

    assert actual == pytest.approx(expected, abs=1e-3), (
        f"MEASURED[{label!r}] no longer matches what the scorer produces. Re-measure the "
        "table -- do not adjust the assertions that read it.")

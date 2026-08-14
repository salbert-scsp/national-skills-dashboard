"""
Offline vector engine: embeddings, anchor poles, scoring, and bucketing.

Replaces Scoring_Algorithm.py. Same offline guarantees, different anchor set.

Data flow:
    text -> get_embedding() -> 384-d unit vector
         -> dot product against five pre-computed anchor vectors
         -> calculate_ai_correlation() -> {ai_score_base, three enabling sims, bucket}
         -> apply_boost() adds EMBEDDED_AI_BOOST when a SEARCH established that this
            product ships AI features -> ai_score
         -> classify() applies the conditional rules engine and reports how close the
            call was (see DRIFT_DELTA)

Not everything here is measured. `embeds_ai` is a three-state fact established outside
this module by embedding_probe.py, and it is the only input that adds to a score. See
EMBEDDED_AI_BOOST for the store measurements that made it a searched fact rather than a
sixth anchor.

bucket_for() is the older flat ladder on ai_score alone. It is retained because the
thresholds it names are the single source of truth and the UI states them, but it is
no longer what decides a skill's class -- classify() is.

Constraints enforced here, all carried over from the module this replaces:
  - Zero Hugging Face network calls. The tokenizer loads only via
    Tokenizer.from_file("tokenizer.json"). Never from_pretrained().
  - Truncation at 256 tokens, padding disabled (CPU inference is single-item).
  - Dot products returned UNCLAMPED across [-1.0, +1.0]. Anti-correlation is real
    signal: Amazon Redshift measured -0.0058 against a pole on live data, and a
    max(0.0, ...) clamp would silently erase it.
  - Missing model or tokenizer raises at import. A zero vector scores 0.0 against
    every pole, which is indistinguishable from a genuine low-relevance result and
    would quietly corrupt the time series.
"""

import logging
import os
import re

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

import storage

logger = logging.getLogger(__name__)

BI_MODEL_PATH = storage.model_path(storage.BI_MODEL_NAME)
TOKENIZER_PATH = storage.model_path(storage.TOKENIZER_NAME)

EMBEDDING_DIM = 384
# cross_encoder.py carries its own 256 for the cross-encoder. See the note there: two
# independent session configs that happen to agree, deliberately not shared.
MAX_SEQUENCE_LENGTH = 256

# --- Absolute bucket thresholds, applied to the RAW cosine score only ---------
# These live here and nowhere else. Any other module that needs a bucket calls
# bucket_for(); none of them re-derive it, and in particular none of them derive it
# from a normalized value, which would make a skill's class change as the UI filters.
#
# RE-DERIVED when scoring moved from max(title, definition) to the 90/10 blend -- see
# CONTEXT_WEIGHT. The blend is not a bug fix applied to a few skills, it lowers EVERY
# score by roughly 10-20%, so holding the old numbers would have silently tightened both
# bars. Measured over the 1051 approved skills:
#
#   AI_SKILL_THRESHOLD    0.30 -> 0.29   at 0.30 the blend loses spaCy (0.2952) and
#                                        XGBoost (0.2955), both genuine. The next
#                                        candidate below 0.29 is Jupyter at 0.252, so
#                                        this is a wide gap, not a value tuned to squeak
#                                        two skills back in.
#   AI_ENABLING_THRESHOLD 0.15 -> 0.12   at 0.15 the middle bucket falls 389 -> 174 and
#                                        at 0.20 it falls to 40. 0.12 holds it at 343,
#                                        which preserves the design intent that everyday
#                                        office tools sit mid-tier and non-zero.
#
# These are a rescale, NOT a loosening. Both bars moved down because the scale beneath
# them moved down.
AI_SKILL_THRESHOLD = 0.29
AI_ENABLING_THRESHOLD = 0.12

BUCKET_AI = "AI Skill"
BUCKET_ENABLING = "AI Enabling Skill"
BUCKET_NOT_AI = "Not AI Skill"

# --- Conditional rules engine -------------------------------------------------
#
# classify() replaces the flat "ai_score >= threshold" ladder with a priority
# evaluation over all four metrics. The three enabling sims were always computed and
# stored; until now nothing read them to decide a class, which is why core
# infrastructure was suppressed to Not AI (C++ ai=0.112 with tech_base=0.374) while
# workflow tools with real embedded AI scored low on every pole that counted.
#
# ClassNormalized is ai_score expressed as a FRACTION OF THE AI SKILL BAR, i.e.
# ai_score / AI_SKILL_THRESHOLD. It is deliberately NOT the UI's min-max
# normalization: that is relative to whatever is currently filtered, so a skill's
# class would change as the reader changes the view. This form is view-independent,
# and it keeps the conceptual constant below the AI Skill bar so its rule can actually
# fire -- on the raw scale 0.70 is unreachable (the whole store tops out at 0.685).
CLASS_NORMALIZED_BASE = AI_SKILL_THRESHOLD

CONCEPTUAL_AI_NORM = 0.70       # -> raw 0.2100

# --- What the TOP bucket additionally requires ---------------------------------
#
# A skill reaches AI Skill only if it also clears this on the AI ENGINEERING pole.
#
# WHY A SECOND TEST IS NEEDED AT ALL: no threshold on ai_score alone can separate genuine
# AI tools from text software, because the two sets OVERLAP on it -- measured, the lowest
# genuine sits at 0.306 and the highest false positive at 0.379. Raising the bar loses
# real skills before it loses fake ones.
#
# The engineering pole is the one anchor that does separate, and by a wide margin. Values
# below are the 90/10 blend (see CONTEXT_WEIGHT), which is what the rule now reads:
#
#     TensorFlow    0.458   |   Word processing software      0.091
#     PyTorch       0.385   |   Transcription software        0.078
#     Scikit-learn  0.383   |   Transcription system software 0.078
#     Keras         0.357   |   Scripting software            0.067
#     NeuralShell   0.332   |   Telluride Trak-It             0.051
#     XGBoost       0.296   |   WordPerfect                   0.042
#     spaCy         0.219   |   Report generation software    0.009
#     ChatGPT       0.177   |
#     ------------- gap 0.0855 -------------
#
# ANY VALUE IN [0.10, 0.17] PRODUCES THE IDENTICAL SET, so this is not balanced on a knife
# edge. Leave-one-out cross-validation over the 15 contested skills: this pole 15/15,
# max(eng, gen) 11/15, fitted weights over all six anchors 9/15 -- the last being what
# seven free parameters on fifteen examples buys.
#
# THE BAND NARROWED at the top when scoring moved to the blend. It used to be [0.12, 0.18];
# 0.18 now fails, because ChatGPT itself sits at 0.1769 and would be gated out of its own
# top bucket. 0.14 is still comfortably mid-band, but the headroom above it is 0.03, not
# 0.04, and a test records that explicitly so the next person does not rediscover it.
#
# It is also the interpretable statement of what an AI skill is: a real AI technology
# talks about neural networks, model training and inference. A word processor does not,
# however much it talks about language.
#
# THIS IS A GATE, NOT A REPLACEMENT. ai_score is still the max over BOTH poles, so the
# generative pole still contributes to the number and everyday text tools keep their
# mid-tier score. It only stops that number, on its own, opening the top bucket.
AI_ENGINEERING_FLOOR = 0.14

# --- What the ENABLING bucket requires ----------------------------------------
#
# ALSO RE-DERIVED FOR THE 90/10 BLEND, and this is the part that is easy to miss. The blend
# lowered EVERY pole, not just the two AI ones, so leaving these at their old values would
# have silently tightened the middle bucket -- which is exactly what happened on the first
# run of this change: AI Enabling fell 410 -> 312 while nothing about the rules had
# supposedly changed. The middle bucket is decided by THESE floors, not by
# AI_ENABLING_THRESHOLD, which classify() does not read.
#
# Measured over the 1051 approved skills, mean similarity under blend vs max:
#
#   tech_base    0.2166 -> 0.1972   ratio 0.911    245 skills over 0.28 -> 160
#   ml_pipeline  0.1501 -> 0.1399   ratio 0.932    237 skills over 0.20 -> 198
#
# Each floor is scaled by its own pole's measured ratio and then checked against the
# population it actually produces: tech_base 245 -> 251, ml_pipeline 237 -> 234. Scaling
# rather than refitting is deliberate -- these floors were argued for on their own
# evidence, and the blend is a change of measurement, not of what they mean.
#
# ML_PIPELINE_FLOOR is 0.19 and not the 0.186 the ratio gives directly. 0.186 is a better
# number in the abstract and a worse one in practice: Microsoft Teams measures ml_pipeline
# 0.184 under the blend, so it would sit 0.001 under the floor and any re-measurement could
# flip it. 0.19 restores a real margin AND lands closer to the old population (234 vs 253).
TECH_BASE_FLOOR = 0.255          # was 0.28
# KNOWN EXCEPTION, do not "fix" by lowering this floor. MongoDB measures tech_base 0.082
# and ml_pipeline 0.140 and therefore classifies Not AI, which looks wrong for a database.
# Dropping ML_PIPELINE_FLOOR to 0.14 to rescue it admits 100+ unrelated skills to
# Technical Backbone -- measured on the live store, not estimated. The defect is in
# MongoDB's definition text, which describes the company and the licence rather than the
# data infrastructure, so the fix is that text and not this constant.
ML_PIPELINE_FLOOR = 0.19         # was 0.20
CONCEPTUAL_TECH_FLOOR = 0.18     # was 0.20, scaled by the tech_base ratio

# --- Embedded AI is established, not measured ---------------------------------
#
# There is no EMBEDDED_AI_FLOOR any more, and its absence is the point.
#
# embedded_ai_sim is a cosine similarity against EMBEDDED_AI_ANCHOR, half of whose
# vocabulary -- workspace, workflow, integration, automated, interface -- appears in the
# prose of any design or productivity tool whether or not it embeds AI. Measured across
# the live store: median 0.179 against a floor of 0.22 and p75 0.237, so a THIRD of every
# skill cleared it, and the class it produced contained LaTeX (0.260, a typesetting
# system from 1984), Thomson EndNote (0.264, a bibliography manager), SofTech CADRA
# (0.225, 2D drafting) and Transoft AutoTURN (0.280, vehicle swept-path analysis). None
# of them embed AI. The definition text simply does not carry the fact, largely because
# most of these definitions were written before the features existed.
#
# So the fact is established by SEARCHING for it -- see embedding_probe.py -- and enters
# here as a three-state boolean:
#
#     True     a search plus a grader confirmed the product itself ships AI features
#     False    they looked and it does not
#     None     nobody has established it yet, or the search was blocked
#
# None behaves exactly as False for classification. The difference between "we checked
# and the answer is no" and "we never checked" is real and worth showing a reader, but it
# is not a difference the engine can act on: neither is evidence of embedded AI.
#
# embedded_ai_sim is still computed, still stored and still displayed. It is a genuine
# measurement of how a definition READS, which is worth seeing next to a verdict that
# contradicts it. It just no longer decides anything.
EMBEDDED_AI_BOOST = 0.05

# Half-width of the band around a decision threshold inside which the classification
# is reported as marginal. Embedding similarity drifts by more than this between model
# revisions and between two phrasings of the same definition, so a decision won by
# less than DRIFT_DELTA is not a decision anyone should build on silently.
DRIFT_DELTA = 0.03

SUB_TECHNICAL_BACKBONE = "Technical Backbone"
SUB_EMBEDDED_AI = "Non-Technical Embedded AI"
SUB_CONCEPTUAL = "Conceptual AI Framework"

CONFIDENCE_HIGH = "High"
CONFIDENCE_MARGINAL = "Marginal - Semantic Drift Potential"

if not os.path.exists(BI_MODEL_PATH):
    raise FileNotFoundError(
        f"Bi-encoder model '{BI_MODEL_PATH}' not found in the project root. "
        "model.onnx (all-MiniLM-L6-v2) is required for offline scoring."
    )

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(
        f"Tokenizer file '{TOKENIZER_PATH}' not found in the project root. "
        "The tokenizer must load from disk; remote downloads are prohibited."
    )

# Dedicated instance. Truncation and padding configured here cannot be mutated by
# another module loading the same file for the cross-encoder.
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH, direction="right")
tokenizer.no_padding()

session = ort.InferenceSession(BI_MODEL_PATH, providers=["CPUExecutionProvider"])

logger.info(
    "Bi-encoder ONNX session loaded offline (max_seq=%d, padding disabled).",
    MAX_SEQUENCE_LENGTH,
)


def get_embedding(text: str) -> np.ndarray:
    """Encodes text into an L2-normalized 384-dimensional dense vector."""
    clean_text = str(text).lower().strip() if text else ""
    if not clean_text:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    encoded = tokenizer.encode(clean_text)

    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    token_type_ids = np.array([encoded.type_ids], dtype=np.int64)

    token_embeddings = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )[0]

    # Attention-masked mean pooling, so description length does not skew the vector.
    mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
    sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    raw_vector = (sum_embeddings / sum_mask)[0]

    # L2 normalization, so cosine similarity reduces to a plain dot product.
    norm = np.linalg.norm(raw_vector)
    if norm <= 0:
        return raw_vector.astype(np.float32)
    return (raw_vector / norm).astype(np.float32)


# --- Anchor poles -------------------------------------------------------------
#
# The two AI poles are unchanged from the previous engine, on purpose: ai_score is
# derived from them alone, so every score already in the time series stays on the
# same scale and migrated history remains comparable to new runs.
AI_ENGINEERING_POLE = (
    "machine learning deep learning neural networks computational inference "
    "tokenization regression clustering predictive analytics tensor framework "
    "model training"
)
# SIX TERMS REMOVED, each because it was measured firing on the WRONG SENSE OF ITS OWN
# WORDS. This is not a trigger list -- the whole string is embedded ONCE into a single
# vector, and a term drags that vector wherever its strongest word points, whether or not
# that is the meaning intended.
#
# Each term embedded alone and scored across all 992 skills. "gap" is the mean over 8
# genuine AI tools minus the mean over 7 known false positives, so a large positive gap
# is a term that discriminates and a negative one actively misleads:
#
#   REMOVED                      gap     what it actually pulled in
#   prompt engineering        -0.085     Robotics ROBOTC .41, Schematic design .38
#                                        -> the strongest word in it is ENGINEERING
#   transformer architecture  +0.065     Rockwell Automation .28, mechanical and
#                                        electrical plumbing .27 -> ELECTRICAL transformers
#   openai                    +0.168     OpenMake Meister .41, OpenOffice.org .37
#                                        -> matches "Open"
#   large language model      +0.141     Xilinx ModelSim .45, Modelica .34
#                                        -> matches "MODEL"
#   llm                       +0.089     Oracle JMS .32, WebSphere MQ .32
#   nlp                       +0.012     HTML .33, WordWeb .32
#                                        -> both are opaque three-letter tokens, and they
#                                           land near other opaque three-letter tokens
#
#   KEPT                         gap     fires on the sense intended
#   natural language processing +0.030   spaCy .45      (the only term that finds spaCy)
#   chatgpt                     +0.048   ChatGPT .72
#   chatbot                     +0.037   ChatGPT .65
#   artificial intelligence     +0.062
#   generative ai               +0.064   PyTorch .34, ChatGPT .32, TensorFlow .30
#
# DO NOT SHORTEN THIS FURTHER. Measured: cutting to "Generative AI, LLM, GPT, Transformer,
# Prompting" is worse than either version -- 22 AI Skills with 16 false positives, and it
# loses spaCy. A short pole does not average its errors away, it is dominated by them, and
# "Transformer" without its qualifying word points straight at electrical transformers
# (Rockwell .356, PLC code generation .357, TMW PowerSuite .317, Autodesk Revit .305).
#
# The mid-tier behaviour the design asks for survives: everyday text tools still score
# 0.25-0.37 here and still land in AI Enabling. What changed is that they no longer reach
# the top bucket -- see AI_ENGINEERING_FLOOR.
AI_GENERATIVE_POLE = (
    "artificial intelligence generative ai chatgpt Natural Language Processing LLM"
)

# The three enabling anchors replace the old INFRA_CORE / DEV_LANG pair. They split
# "not AI itself, but adjacent to it" into three distinguishable kinds of adjacency,
# and are reported as DIAGNOSTICS alongside ai_score rather than feeding into it.
TECH_BASE_ANCHOR = (
    "programming data analysis matrix array visualization scripting runtime "
    "data manipulation statistical package numerical computing software development coding"
)
ML_PIPELINE_ANCHOR = (
    "data pipeline machine learning lifecycle orchestration distributed computing "
    "feature store model registry containerization acceleration compute cluster "
    "extract transform load workflow execution"
)
EMBEDDED_AI_ANCHOR = (
    "copilot embedded artificial intelligence generative AI assistant plugin automated "
    "agent smart workspace prompt interface auto-complete enterprise agentic "
)

# A CONTRAST anchor: conventional text and document software, which is the exact family
# the generative pole kept mistaking for AI. Reported as a diagnostic and read by NO
# RULE -- see contrast_sim in calculate_ai_correlation for why it is recorded rather than
# subtracted.
CONTRAST_TEXT_ANCHOR = (
    "word processor document editor spreadsheet formatting typesetting page layout "
    "dictation typing shorthand printing publishing office suite text editor "
    "conventional software without machine learning"
)

# A LEGACY anchor: desktop-era, offline, pre-web software. Like CONTRAST_TEXT_ANCHOR this
# is a DIAGNOSTIC read by NO RULE, and the reason is measured rather than cautious.
#
# It was proposed as a gated discount, ai_score * (1 - (legacy_sim - 0.28)). That formula
# cannot do anything: at the store's HIGHEST observed legacy_sim (0.567) the factor is
# still 0.71, and across all 1051 approved skills the largest discount it applies anywhere
# is 0.018 -- Telluride 0.246 -> 0.235. It would not move a single classification.
#
# The alternative, a hard gate (legacy_sim >= 0.28 and engineering below its floor -> Not
# AI), does bite: 26 skills. But it takes the wrong ones with the right ones. It correctly
# kills Telluride, Telemetry software and Telephone records software, and it also demotes
# Red Hat Ansible Engine, Talend Big Data Integration and Cloudera Impala, which are real
# data infrastructure and belong in AI Enabling.
#
# The blend made the argument moot in any case: it already drops Telluride 0.298 -> 0.246
# and Report generation 0.250 -> 0.131 without any negative anchor at all. So legacy_sim is
# stored and shown, costs one dot product, and earns the right to decide something when
# there is data showing it separates cleanly. Today there is not.
LEGACY_STATIC_ANCHOR = (
    "legacy desktop software offline executable application pre-web database "
    "local system utility standalone desktop tool legacy client server software"
)

# Computed once at import and held for the process lifetime. Seven embeddings, not
# seven per skill.
AI_ENG_VEC = get_embedding(AI_ENGINEERING_POLE)
AI_GEN_VEC = get_embedding(AI_GENERATIVE_POLE)
TECH_BASE_VEC = get_embedding(TECH_BASE_ANCHOR)
ML_PIPELINE_VEC = get_embedding(ML_PIPELINE_ANCHOR)
EMBEDDED_AI_VEC = get_embedding(EMBEDDED_AI_ANCHOR)
CONTRAST_VEC = get_embedding(CONTRAST_TEXT_ANCHOR)
LEGACY_VEC = get_embedding(LEGACY_STATIC_ANCHOR)

logger.info("Seven anchor pole vectors pre-computed and cached in memory.")


# --- Deterministic AI phrases, recorded and NOT scored -------------------------
#
# These were briefly used as a +0.05 boost while the matching terms were removed from
# AI_GENERATIVE_POLE. The pole is restored, so the boost is gone: "nlp" is measured by
# the pole again, and adding a deterministic bonus on top of it would count the same
# evidence twice.
#
# The match is kept because it is genuinely useful to SEE which skills state an AI
# technique outright -- it is reported on the record and nothing reads it to decide a
# class. \b is load-bearing: without word boundaries "nlp" matches inside other tokens,
# and the suffix is `s?` rather than `\w*` for the same reason, after `\w*` was found to
# match "nlp" inside "nlpxyz".
LEXICAL_AI_TERMS = (
    "nlp",
    "natural language processing",
    "natural language understanding",
    "prompt engineering",
    "large language model",
    "generative ai",
    "machine learning",
    "deep learning",
    "neural network",
)

LEXICAL_AI_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in LEXICAL_AI_TERMS) + r")s?\b",
    re.IGNORECASE,
)

# Zero. The pole measures these phrases; this constant exists so the field stays in the
# schema and any caller that reads it gets an honest 0.0 rather than a KeyError.
LEXICAL_AI_BOOST = 0.0


def lexical_ai_terms_in(text: str) -> list:
    """Which AI phrases literally appear. Reported only; reads on no rule."""
    if not text:
        return []
    return sorted({match.group(0).lower() for match in LEXICAL_AI_PATTERN.finditer(str(text))})


def _dot(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity via dot product on unit vectors. Sign is preserved."""
    return float(np.dot(vec_a, vec_b))


# How much of a pole score comes from the DEFINITION rather than the title.
#
# This replaced max(title_sim, context_sim), and the reason is that max() gave a 1-4 token
# title a veto over a whole paragraph. Titles are short and ambiguous, so their embeddings
# are high variance and land near whatever their strongest token points at; definitions are
# long and specific. Under max(), a title that happened to score high carried the skill
# regardless of what its definition said -- which is exactly how a legacy project-management
# tool reached 0.298 and a report writer reached 0.250.
#
# The blend removes that path without discarding the title, which still matters for skills
# whose definition is thin. Measured over the 1051 approved skills:
#
#                                 max()   90/10
#   Telluride Trak-It             0.298   0.246
#   Report generation software    0.250   0.131
#   Scripting software            0.308   0.261
#   WordPerfect                   0.262   0.221
#   Word processing software      0.309   0.309   <- unchanged, and correctly so: its
#                                                    definition really is about language
#
# It also SURFACED one skill that max() was scoring too low: Ward Systems NeuralShell
# Predictor, a genuine neural-network forecasting tool, at engineering 0.332.
#
# NOTE FOR ANYONE COMPARING HISTORY: this rescales every score in the store. A
# quarter-over-quarter comparison across the change measures the change in method, not a
# change in the world.
CONTEXT_WEIGHT = 0.90


def _blend(title_vec: np.ndarray, ctx_vec: np.ndarray, pole_vec: np.ndarray) -> float:
    """
    A pole score from both text surfaces: the definition decides, the title nudges.

    Applied to EVERY anchor, deliberately. A pole scored one way and another pole scored
    another way cannot be compared, and ai_score is a max across two of them.
    """
    return (CONTEXT_WEIGHT * _dot(ctx_vec, pole_vec)
            + (1.0 - CONTEXT_WEIGHT) * _dot(title_vec, pole_vec))


def bucket_for(ai_score) -> str:
    """
    Classifies a RAW cosine score into an absolute category bucket.

    The single source of truth for the thresholds. Never pass a normalized value:
    normalization is relative to whatever is currently filtered, so a normalized
    input would make a skill's category change as the user changes the view.

    A None score buckets as Not AI Skill rather than raising. An unscored skill is
    not an AI skill, and this is called from render paths where an exception would
    take the whole table down.
    """
    if ai_score is None:
        return BUCKET_NOT_AI
    score = float(ai_score)
    if score >= AI_SKILL_THRESHOLD:
        return BUCKET_AI
    if score >= AI_ENABLING_THRESHOLD:
        return BUCKET_ENABLING
    return BUCKET_NOT_AI


def class_normalized(ai_score) -> float:
    """
    ai_score as a fraction of the AI Skill bar. See CLASS_NORMALIZED_BASE.

    Uncapped on purpose: a skill at 1.4 is meaningfully further past the bar than one
    at 1.0, and clamping would throw that away for no benefit -- nothing downstream
    requires the value to sit inside [0, 1].
    """
    if ai_score is None:
        return 0.0
    return float(ai_score) / CLASS_NORMALIZED_BASE


def _condition(name: str, value: float, threshold: float) -> dict:
    """One threshold comparison, carrying the signed distance that decided it."""
    return {
        "metric": name,
        "value": float(value),
        "threshold": float(threshold),
        "margin": float(value) - float(threshold),
        "met": float(value) >= float(threshold),
    }


def _any_of(conditions):
    """
    The condition that decides an OR: the largest margin.

    Satisfied or not, the largest margin is the right one to report. When the OR
    fires it is the comparison won by the most; when it does not, it is the one that
    came closest, which is exactly what the drift band needs to know about.
    """
    return max(conditions, key=lambda c: c["margin"])


def _all_of(conditions):
    """
    The condition that decides an AND: the smallest margin, i.e. the binding one.

    An AND is only as satisfied as its weakest term, so a rule that fires with
    ai_score 0.20 clear but embedded_ai_sim 0.001 clear is a marginal decision, and
    reporting the comfortable term would hide that.
    """
    return min(conditions, key=lambda c: c["margin"])


def _boolean_condition(name: str, established: bool) -> dict:
    """
    An established FACT presented in the shape the rules engine reports.

    Deliberately carries margin None rather than 0.0 or 1.0. The drift band exists
    because embedding similarity moves between model revisions and between two
    phrasings of the same definition, so a threshold won by less than DRIFT_DELTA is
    not a decision anyone should build on. None of that applies to a searched fact:
    it did not come from a vector and it cannot drift by 0.03.

    Reporting a number here would put a fabricated distance in the one field readers
    use to judge how reliable a call was, and would let decided() band it as Marginal
    for a reason that does not exist.
    """
    return {
        "metric": name,
        "value": bool(established),
        "threshold": None,
        "margin": None,
        "met": bool(established),
    }


def classify(
    ai_score,
    tech_base_sim=0.0,
    ml_pipeline_sim=0.0,
    embedded_ai_sim=0.0,
    embeds_ai=None,
    ai_engineering_sim=None,
) -> dict:
    """
    Applies the conditional rules engine and reports how close the call was.

    Rules are evaluated in strict priority order and the FIRST match wins:

        0. ai_score >= AI_SKILL_THRESHOLD              -> AI Skill
        1. tech_base >= 0.28 OR ml_pipeline >= 0.20    -> Enabling, Technical Backbone
        2. embeds_ai is True                            -> Enabling, Embedded AI
        3. normalized >= 0.70 AND tech_base >= 0.20    -> Enabling, Conceptual
        4. otherwise                                    -> Not AI Skill

    Priority is load-bearing, not cosmetic. Power BI carries both a high tech_base
    (0.397) and confirmed embedded AI; rule 1 firing first is what makes it read as
    infrastructure that gained AI features rather than as an AI-first product.

    RULE 2 IS A FACT, NOT A MEASUREMENT. It used to test embedded_ai_sim against 0.22
    and it no longer reads that metric at all -- see EMBEDDED_AI_BOOST for the store
    measurements that killed it. `embeds_ai` is three-state and None behaves as False:
    "we checked and it does not" and "nobody has checked" are a real distinction to show
    a reader, but neither is evidence of embedded AI, so neither can promote a skill.

    `ai_score` is expected to ALREADY CARRY the boost, because rule 0 and the normalized
    fraction both have to see the same number the dashboard shows. calculate_ai_correlation
    adds it before calling here; a caller passing a raw pole measurement together with
    embeds_ai=True gets rule 2 without the boost, which is a coherent answer to a
    different question rather than a bug.

    Returns the bucket, the sub-category, and the comparison that decided it:

        category_bucket               one of the three BUCKET_* constants
        sub_category                  None for AI Skill and Not AI Skill
        decision_metric               which metric the call turned on
        decision_threshold            the bar it was measured against, None for a fact
        decision_margin               signed distance from that bar, None for a fact
        in_semantic_variance_band     abs(margin) <= DRIFT_DELTA, False for a fact
        classification_confidence     CONFIDENCE_HIGH or CONFIDENCE_MARGINAL

    `embedded_ai_sim` is accepted and unused, exactly as `onet_category_title` is in
    calculate_ai_correlation. Every caller passes the four metrics positionally and
    dropping the parameter would silently shift `embeds_ai` into its place at every one
    of those call sites -- a rename that type-checks and misclassifies the whole store.

    A None ai_score classifies as Not AI Skill rather than raising, matching
    bucket_for(): this is called from render paths where an exception would take the
    whole table down, and an unscored skill is not an AI skill.
    """
    normalized = class_normalized(ai_score)
    ai_value = 0.0 if ai_score is None else float(ai_score)

    def decided(bucket, sub_category, condition):
        # A fact carries margin None and is never banded: there is no threshold it came
        # close to, and CONFIDENCE_HIGH is the honest reading of "someone established
        # this" rather than a flattering one.
        if condition["margin"] is None:
            return {
                "category_bucket": bucket,
                "sub_category": sub_category,
                "decision_metric": condition["metric"],
                "decision_threshold": None,
                "decision_margin": None,
                "in_semantic_variance_band": False,
                "classification_confidence": CONFIDENCE_HIGH,
            }

        # Banded on the ROUNDED margin, which is also the one reported. Comparing the
        # raw float instead put a decision sitting exactly on the band edge outside it:
        # 0.28 + 0.03 - 0.28 evaluates to 0.030000000000000027, and a skill whose
        # displayed margin reads exactly +0.0300 would have been labelled High.
        margin = round(condition["margin"], 4)
        in_band = abs(margin) <= DRIFT_DELTA
        return {
            "category_bucket": bucket,
            "sub_category": sub_category,
            "decision_metric": condition["metric"],
            "decision_threshold": round(condition["threshold"], 4),
            "decision_margin": margin,
            "in_semantic_variance_band": in_band,
            "classification_confidence": (
                CONFIDENCE_MARGINAL if in_band else CONFIDENCE_HIGH
            ),
        }

    # Rule 0. Kept ahead of everything so the top bucket survives the rules engine:
    # every rule below returns AI Enabling or Not AI, and without this the AI Skill
    # class would silently disappear along with its colour in the UI.
    #
    # TWO conditions, not one. ai_score alone cannot separate genuine AI tools from text
    # software -- the two sets overlap on it -- so the top bucket also asks the AI
    # ENGINEERING pole, which does separate. See AI_ENGINEERING_FLOOR for the measurement.
    #
    # _all_of reports the BINDING term, so a skill that fails on engineering says so
    # rather than reporting the ai_score it comfortably cleared.
    #
    # A None ai_engineering_sim SKIPS the floor rather than failing it. Snapshots written
    # before this landed do not carry the value, and defaulting it to 0.0 would silently
    # demote every one of them the first time an old record was reclassified.
    rule_zero = _condition("ai_score", ai_value, AI_SKILL_THRESHOLD)
    if ai_engineering_sim is None:
        top = rule_zero
    else:
        top = _all_of([
            rule_zero,
            _condition("ai_engineering_sim", ai_engineering_sim, AI_ENGINEERING_FLOOR),
        ])
    if top["met"]:
        return decided(BUCKET_AI, None, top)

    # Rule 1. Prevents core languages and data infrastructure from being suppressed to
    # Not AI by a low ai_score they were never going to earn.
    backbone = [
        _condition("tech_base_sim", tech_base_sim, TECH_BASE_FLOOR),
        _condition("ml_pipeline_sim", ml_pipeline_sim, ML_PIPELINE_FLOOR),
    ]
    if any(c["met"] for c in backbone):
        return decided(BUCKET_ENABLING, SUB_TECHNICAL_BACKBONE, _any_of(backbone))

    # Rule 2. Non-technical applications that actually ship agentic or generative
    # features. `is True` and not a truth test: None must not be confused with False by
    # a caller passing 0 or "", and neither may fire this rule.
    embedded = _boolean_condition("embeds_ai", embeds_ai is True)
    if embedded["met"]:
        return decided(BUCKET_ENABLING, SUB_EMBEDDED_AI, embedded)

    # Rule 3. Conceptual AI frameworks: close to the AI poles and technical with it,
    # but not close enough to clear rule 0 outright.
    conceptual = _all_of([
        _condition("class_normalized", normalized, CONCEPTUAL_AI_NORM),
        _condition("tech_base_sim", tech_base_sim, CONCEPTUAL_TECH_FLOOR),
    ])
    if conceptual["met"]:
        return decided(BUCKET_ENABLING, SUB_CONCEPTUAL, conceptual)

    # Default. The reported margin is the nearest threshold this skill MISSED across
    # every rule, so a Not AI sitting 0.005 under a floor is visibly a near miss
    # rather than being indistinguishable from one that missed by 0.3.
    #
    # Rule 2 is excluded, and not only because _any_of cannot compare a None margin. A
    # skill that was checked and found to embed no AI did not NARROWLY miss anything;
    # there is no distance to report, and inventing one would be the same lie
    # _boolean_condition exists to avoid.
    # `top` rather than `rule_zero`, so a skill held out of the AI Skill bucket by the
    # engineering floor reports THAT as its nearest miss. Reporting the ai_score it
    # comfortably cleared would be the most misleading thing on the card.
    nearest = _any_of(backbone + [conceptual, top])
    return decided(BUCKET_NOT_AI, None, nearest)


FLAGSHIP_METRICS = ("ai_score", "tech_base_sim", "ml_pipeline_sim", "embedded_ai_sim")


def boost_for(embeds_ai) -> float:
    """
    The raw addition a confirmed embedded-AI finding earns, or 0.0.

    Only True earns it. None -- never checked, or the search was blocked -- earns
    nothing, which is what keeps a DuckDuckGo outage from quietly inflating the store.
    """
    return EMBEDDED_AI_BOOST if embeds_ai is True else 0.0


def base_of(metrics: dict):
    """
    The unboosted pole measurement from any metrics dict or stored snapshot.

    THE compatibility shim for the whole change. Snapshots written before the boost
    existed have no ai_score_base, and for those ai_score IS the base -- nothing had
    been added to it. Anything comparing scores across that boundary has to read them
    through here, or it compares a boosted 2026 number against an unboosted 2025 one
    and reports drift that never happened.
    """
    base = metrics.get("ai_score_base")
    return metrics.get("ai_score") if base is None else base


def apply_boost(ai_score_base, embeds_ai, lexical_terms=()) -> dict:
    """
    Splits a boosted score into the fields every caller stores.

    The split is not bookkeeping, it is what keeps the module's own guarantee true.
    ai_score_base is still derived from the two AI poles and nothing else, so a 2026
    snapshot stays on the same scale as migrated history; ai_score is what the rules
    engine and the dashboard read. A snapshot written before this landed carries no
    ai_score_base, and readers take base == ai_score with a zero boost, which is exactly
    what it was.

    TWO ADDITIONS, KEPT SEPARATE, because they answer different questions and can be
    wrong independently:

        embedded_ai_boost   a SEARCH established that this product ships AI features
        lexical_ai_boost    the text literally SAYS "machine learning", "nlp", and so on

    They stack. A skill can both name a technique and ship a copilot, and there is no
    reason one should mask the other. Both are capped at one application each, so the
    most any skill earns is 0.10 -- a third of the AI Skill bar, which is deliberately
    not enough to carry a skill there from nothing.
    """
    base = 0.0 if ai_score_base is None else float(ai_score_base)
    embedded = boost_for(embeds_ai)
    lexical = LEXICAL_AI_BOOST if lexical_terms else 0.0
    return {
        "ai_score_base": round(base, 4),
        "embedded_ai_boost": round(embedded, 4),
        "lexical_ai_boost": round(lexical, 4),
        "lexical_ai_terms": list(lexical_terms),
        "ai_score": round(base + embedded + lexical, 4),
    }


def embedded_with_note(base_embedded: float, flagship_note: str = "") -> float:
    """
    Raises an embedded-AI similarity to account for a flagship AI note.

    The note is scored as its OWN text and combined with max(), never concatenated onto
    the definition. Concatenation would dilute a two-line sentence inside a 1,500
    character article, and it would move every other metric too -- see
    calculate_ai_correlation for the measurement that ruled it out. Combining with max()
    also makes the note strictly additive: it can raise the similarity, never lower it,
    so a note about a product with no real AI features leaves the measurement alone.

    DIAGNOSTIC ONLY since embedded_ai_sim stopped deciding anything. This still moves
    the number a reader sees, and it can no longer move a classification. Kept rather
    than deleted because the note is a real measurement of text describing the shipped
    product, and it is the most useful thing to show beside a searched verdict.
    """
    if not (flagship_note and str(flagship_note).strip()):
        return float(base_embedded)
    note_vec = get_embedding(flagship_note)
    return max(float(base_embedded), _dot(note_vec, EMBEDDED_AI_VEC))


def score_as_flagship(
    flagship_metrics: dict, flagship_note: str = "", embeds_ai=None
) -> dict:
    """
    Scores a CATEGORY term as the flagship product it denotes.

    A category article outscores every product in its category, because it is written in
    abstract taxonomy-dense prose that matches the anchor vocabulary directly while
    product articles are corporate and historical. Measured across the store: "Word
    processing software" reads ai 0.321 against Microsoft Word's 0.192 and was
    classified an AI Skill outright; "Spreadsheet software" reads tech_base 0.458
    against Excel's 0.395. The engine was scoring the vocabulary of the category rather
    than the software anyone uses.

    So the category takes the flagship's metrics wholesale. It is then EQUAL to its
    flagship by construction and can never sit above it, which is the invariant this
    function exists to guarantee and which enforce_flagship_ceiling asserts.

    `flagship_note` still applies to embedded_ai_sim, and for a category it is close to
    mandatory rather than opportunistic: substitution inherits the flagship article's
    weaknesses along with its scores. Facebook's article is corporate history, reading
    tech_base -0.026 and embedded -0.033, so "Social media software" scored as a bare
    Facebook collapses to Not AI -- wrong for a category whose feeds are ranked by
    machine learning. With the note it reads embedded 0.270 and classifies correctly.

    `flagship_metrics` is whatever measured the flagship: an existing store record when
    the flagship is itself a scored skill, otherwise a fresh calculate_ai_correlation
    over a supplied definition.

    `embeds_ai` is the CATEGORY's own searched verdict, not the flagship's. Substitution
    borrows the flagship's measurements because a category article scores its own
    vocabulary rather than the software; it does not borrow its facts. "Word processing
    software" and "Microsoft Word" are asked separately and may honestly disagree.

    The flagship's metrics are taken as the BASE, so the boost lands on top of them. A
    flagship record already carrying a boost of its own would have it applied twice, and
    enforce_flagship_ceiling compares base against base for exactly that reason.
    """
    metrics = {name: float(flagship_metrics[name]) for name in FLAGSHIP_METRICS}
    metrics["embedded_ai_sim"] = embedded_with_note(
        metrics["embedded_ai_sim"], flagship_note
    )

    # base_of(), not metrics["ai_score"]: a stored flagship snapshot may already carry a
    # boost, and inheriting it here would compound it with this category's own.
    #
    # No lexical terms are passed. A substituted category is measured on the FLAGSHIP's
    # text, and the deterministic boost is a statement about the text that was actually
    # read -- inheriting it from a flagship record would attribute the flagship's words
    # to the category, which is exactly the conflation substitution already risks.
    boosted = apply_boost(base_of(flagship_metrics), embeds_ai)

    return {
        **boosted,
        "tech_base_sim": round(metrics["tech_base_sim"], 4),
        "ml_pipeline_sim": round(metrics["ml_pipeline_sim"], 4),
        "embedded_ai_sim": round(metrics["embedded_ai_sim"], 4),
        "contrast_sim": float(flagship_metrics.get("contrast_sim") or 0.0),
        # None, not 0.0, when the flagship record predates the field. Nothing reads it, so
        # nothing can raise on it, and 0.0 would be a measurement claim about a pole that
        # was never scored.
        "legacy_sim": flagship_metrics.get("legacy_sim"),
        # Inherited from the flagship along with every other metric, because that is what
        # substitution means: the category is measured AS the product. A category whose
        # flagship record predates this field gets None, which classify() reads as "skip
        # the floor" rather than "fail it" -- see rule 0.
        "ai_engineering_sim": flagship_metrics.get("ai_engineering_sim"),
        "ai_generative_sim": flagship_metrics.get("ai_generative_sim"),
        "embeds_ai": embeds_ai,
        "is_flagship_version_evaluated": bool(flagship_note and str(flagship_note).strip()),
        "is_generic_category": True,
        "pathway": "Flagship Substitution",
        # Unrounded, matching calculate_ai_correlation.
        **classify(
            float(base_of(flagship_metrics) or 0.0) + boost_for(embeds_ai),
            metrics["tech_base_sim"],
            metrics["ml_pipeline_sim"],
            metrics["embedded_ai_sim"],
            embeds_ai,
            flagship_metrics.get("ai_engineering_sim"),
        ),
    }


def enforce_flagship_ceiling(metrics: dict, flagship_metrics: dict) -> None:
    """
    Asserts that a substituted category did not end up above its flagship.

    Raises ValueError rather than silently clamping. A category scoring above its
    flagship means the substitution did not happen or happened against the wrong
    product, and quietly capping the number would hide which. The one deliberate
    exception is embedded_ai_sim, which the flagship note may legitimately raise --
    the note describes the flagship itself, so the result is still the flagship's
    score, measured with text its encyclopedia article predates.

    THE AI SCORE IS COMPARED BASE TO BASE. A category asked about its own embedded AI
    may earn the boost where its flagship did not, and then a correct substitution reads
    0.242 against a ceiling of 0.192 and this would raise on a right answer. The
    invariant being defended is that the category is MEASURED as its flagship, and that
    is a statement about the pole measurement. The boost is not a measurement, it is a
    fact about the category itself, and the two are asked separately on purpose --
    see score_as_flagship.
    """
    measured_ai = float(base_of(metrics))
    ceiling_ai = round(float(base_of(flagship_metrics)), 4)
    if measured_ai > ceiling_ai + 1e-9:
        raise ValueError(
            f"ai_score_base {measured_ai:.4f} exceeds the flagship ceiling "
            f"{ceiling_ai:.4f}. A category term must never be measured above the "
            "product it stands for."
        )

    for name in ("tech_base_sim", "ml_pipeline_sim"):
        measured = float(metrics[name])
        ceiling = round(float(flagship_metrics[name]), 4)
        if measured > ceiling + 1e-9:
            raise ValueError(
                f"{name} {measured:.4f} exceeds the flagship ceiling {ceiling:.4f}. "
                "A category term must never score above the product it stands for."
            )


def calculate_ai_correlation(
    skill_title: str,
    onet_category_title: str = "",
    context_summary: str = "",
    flagship_note: str = "",
    embeds_ai=None,
) -> dict:
    """
    Scores a skill against the AI poles and the three enabling anchors.

    Returns unclamped scalars in [-1.0, +1.0] plus the category bucket:

    Every pole is scored by the 90/10 definition-primary blend (see _blend), then
    ai_score_base takes the max across the two AI poles:

        ai_score_base     max over {AI_ENG, AI_GEN} of the blended score
        embedded_ai_boost EMBEDDED_AI_BOOST when embeds_ai is True, else 0.0
        ai_score          ai_score_base + embedded_ai_boost, what classify() reads
        tech_base_sim     blended against TECH_BASE_ANCHOR
        ml_pipeline_sim   blended against ML_PIPELINE_ANCHOR
        embedded_ai_sim   blended against EMBEDDED_AI_ANCHOR, then max'd with the
                          flagship note if one exists
        embeds_ai         the three-state fact, passed straight through
        category_bucket   from classify(), plus its sub-category and drift fields

    The three enabling sims do NOT contribute to ai_score_base. Blending five poles into
    one number would change every score already recorded and make the migrated
    history incomparable with new runs, so the enabling anchors describe HOW a skill
    is adjacent to AI while ai_score_base alone decides how close it is.

    `embeds_ai` is the one thing that adds to the score, and it is not a pole. It is a
    searched fact -- see EMBEDDED_AI_BOOST -- and it is added in a SEPARATE field so the
    guarantee above survives intact: ai_score_base is still two poles and nothing else,
    and any comparison across the boundary reads it through base_of(). A True adds
    exactly EMBEDDED_AI_BOOST and touches no other metric, which is the same discipline
    flagship_note follows for the same reason.

    That guarantee is about this function only, and one path deliberately breaks it:
    score_as_flagship replaces all four metrics for a CATEGORY term, so up to 116 of
    the store's skills get a different ai_score than they carried before. Old snapshots
    are not rewritten -- the time series records what was measured at the time, and the
    substitution appears as a genuine step in the quarter it lands. A named product's
    ai_score still comes from the two original poles and nothing else.

    Two text surfaces are read: the skill title and its definition. Titles are short
    and often ambiguous ("R", "Go"), definitions are long and specific, so the definition
    carries 90% of every pole score and the title 10%. This used to be a max over the two,
    which let a 1-4 token title outvote a whole paragraph -- see _blend for the measurement
    that ended that.

    `flagship_note` describes the AI features of the version of this product people
    actually deploy -- Copilot in Teams, Gemini in Docs -- for generic terms and
    enterprise platforms whose Wikipedia definition predates those features.

    It feeds embedded_ai_sim ONLY, and that restriction is the whole design. Measured:
    appending Slack's flagship sentence to the scoring text lifts embedded_ai_sim
    0.190 -> 0.367, which is the intended effect, but it also drags ai_score 0.139 ->
    0.326, across the AI Skill bar. A chat application would be classified an AI Skill
    on the strength of a sentence about its assistant. The note answers exactly one
    question -- does the flagship deployment embed AI -- so it may move exactly one
    metric. This also keeps ai_score derived from the two original poles alone, which
    is what makes the migrated history comparable with new runs.

    `onet_category_title` is accepted and unused. It is part of the call signature
    the pipeline already uses, and dropping it would mean touching every call site
    for no gain; keeping it leaves the door open to a taxonomy-aware pole later.
    """
    if not skill_title:
        # Not an error. O*NET occasionally yields a blank example name, and a zeroed
        # record is the honest representation of "nothing to score".
        #
        # embeds_ai is forced to None here rather than passed through. There is no skill
        # to have searched for, so "nobody established it" is the only true answer, and
        # a stale True arriving with a blank title must not earn a blank record 0.05.
        return {
            "ai_score_base": 0.0,
            "embedded_ai_boost": 0.0,
            "lexical_ai_boost": 0.0,
            "lexical_ai_terms": [],
            "ai_score": 0.0,
            "tech_base_sim": 0.0,
            "ml_pipeline_sim": 0.0,
            "embedded_ai_sim": 0.0,
            "contrast_sim": 0.0,
            "legacy_sim": 0.0,
            "ai_engineering_sim": 0.0,
            "ai_generative_sim": 0.0,
            "embeds_ai": None,
            "pathway": "Scalar Winner-Take-All Mapping",
            **classify(0.0, 0.0, 0.0, 0.0, None, 0.0),
        }

    title_vec = get_embedding(skill_title)

    # Fall back to the title vector when there is no definition, rather than to a
    # zero vector, which would drag every max() down to 0.0 and read as low relevance.
    has_context = bool(context_summary and str(context_summary).strip())
    ctx_vec = get_embedding(context_summary) if has_context else title_vec

    # Broken out rather than folded into the max(), because the top bucket reads it on its
    # own -- see AI_ENGINEERING_FLOOR.
    #
    # The max() across POLES stays -- highest-match-wins is the design, and a skill should
    # be scored by whichever anchor it is genuinely closest to. What went is the max()
    # across SURFACES, which let a short title outvote its own definition. See _blend.
    ai_engineering_sim = _blend(title_vec, ctx_vec, AI_ENG_VEC)
    ai_generative_sim = _blend(title_vec, ctx_vec, AI_GEN_VEC)

    ai_score_base = max(ai_engineering_sim, ai_generative_sim)
    tech_base_sim = _blend(title_vec, ctx_vec, TECH_BASE_VEC)
    ml_pipeline_sim = _blend(title_vec, ctx_vec, ML_PIPELINE_VEC)
    embedded_ai_sim = _blend(title_vec, ctx_vec, EMBEDDED_AI_VEC)
    contrast_sim = _blend(title_vec, ctx_vec, CONTRAST_VEC)
    legacy_sim = _blend(title_vec, ctx_vec, LEGACY_VEC)

    # The deterministic half, read from the RAW TEXT rather than from a vector. Title and
    # definition are both scanned: "Machine learning software" carries the phrase in its
    # name and nowhere else.
    lexical_terms = lexical_ai_terms_in(f"{skill_title} {context_summary or ''}")

    has_flagship = bool(flagship_note and str(flagship_note).strip())
    embedded_ai_sim = embedded_with_note(embedded_ai_sim, flagship_note)

    boosted = apply_boost(ai_score_base, embeds_ai, lexical_terms)
    lexical_boost = LEXICAL_AI_BOOST if lexical_terms else 0.0

    return {
        **boosted,
        "tech_base_sim": round(tech_base_sim, 4),
        "ml_pipeline_sim": round(ml_pipeline_sim, 4),
        "embedded_ai_sim": round(embedded_ai_sim, 4),
        # DIAGNOSTIC, read by no rule. See classify(): it does not take this argument.
        # Recorded rather than subtracted because subtracting would move every score in
        # the store and force AI_SKILL_THRESHOLD to be re-derived from 0.30 to about
        # 0.08, which breaks comparability with migrated history. Stored, it costs one
        # dot product and buys a quarter of real data before anyone argues it should
        # decide anything -- the same discipline the three enabling sims already follow.
        "contrast_sim": round(contrast_sim, 4),
        # DIAGNOSTIC, read by no rule, for the same reason and with the same discipline as
        # contrast_sim above. See LEGACY_STATIC_ANCHOR for the measurement showing why
        # neither of the two ways of wiring it into a rule survived contact with the store.
        "legacy_sim": round(legacy_sim, 4),
        # The two poles behind ai_score, reported separately. The engineering one is read
        # by the top bucket; both are worth showing, because "0.31" means something very
        # different depending on which pole produced it.
        "ai_engineering_sim": round(ai_engineering_sim, 4),
        "ai_generative_sim": round(ai_generative_sim, 4),
        "embeds_ai": embeds_ai,
        "is_flagship_version_evaluated": has_flagship,
        "is_generic_category": False,
        "pathway": "Conditional Rules Engine",
        # Classified from the UNROUNDED values so a metric sitting exactly on a
        # threshold cannot be rounded across it -- including the boosts, which are added
        # here rather than read back from `boosted` for exactly that reason.
        **classify(
            ai_score_base + boost_for(embeds_ai) + lexical_boost,
            tech_base_sim,
            ml_pipeline_sim,
            embedded_ai_sim,
            embeds_ai,
            ai_engineering_sim,
        ),
    }

"""
Offline vector engine: embeddings, anchor poles, scoring, and bucketing.

Replaces Scoring_Algorithm.py. Same offline guarantees, different anchor set.

Data flow:
    text -> get_embedding() -> 384-d unit vector
         -> dot product against five pre-computed anchor vectors
         -> calculate_ai_correlation() -> {ai_score, three enabling sims, bucket}
         -> bucket_for() classifies on the RAW score

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

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

BI_MODEL_PATH = "model.onnx"
TOKENIZER_PATH = "tokenizer.json"

EMBEDDING_DIM = 384
MAX_SEQUENCE_LENGTH = 256

# --- Absolute bucket thresholds, applied to the RAW cosine score only ---------
# These live here and nowhere else. Any other module that needs a bucket calls
# bucket_for(); none of them re-derive it, and in particular none of them derive it
# from a normalized value, which would make a skill's class change as the UI filters.
AI_SKILL_THRESHOLD = 0.30
AI_ENABLING_THRESHOLD = 0.15

BUCKET_AI = "AI Skill"
BUCKET_ENABLING = "AI Enabling Skill"
BUCKET_NOT_AI = "Not AI Skill"

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
AI_GENERATIVE_POLE = (
    "artificial intelligence generative ai large language model llm chatgpt openai "
    "transformer architecture chatbot natural language processing nlp prompt engineering"
)

# The three enabling anchors replace the old INFRA_CORE / DEV_LANG pair. They split
# "not AI itself, but adjacent to it" into three distinguishable kinds of adjacency,
# and are reported as DIAGNOSTICS alongside ai_score rather than feeding into it.
TECH_BASE_ANCHOR = (
    "programming language data analysis matrix array visualization scripting runtime "
    "data manipulation statistical package numerical computing software development library"
)
ML_PIPELINE_ANCHOR = (
    "data pipeline machine learning lifecycle orchestration distributed computing "
    "feature store model registry containerization acceleration compute cluster "
    "extract transform load workflow execution"
)
EMBEDDED_AI_ANCHOR = (
    "copilot embedded artificial intelligence generative AI assistant plugin automated "
    "agent smart workspace prompt interface text generation auto-complete enterprise "
    "workflow integration"
)

# Computed once at import and held for the process lifetime. Five embeddings, not
# five per skill.
AI_ENG_VEC = get_embedding(AI_ENGINEERING_POLE)
AI_GEN_VEC = get_embedding(AI_GENERATIVE_POLE)
TECH_BASE_VEC = get_embedding(TECH_BASE_ANCHOR)
ML_PIPELINE_VEC = get_embedding(ML_PIPELINE_ANCHOR)
EMBEDDED_AI_VEC = get_embedding(EMBEDDED_AI_ANCHOR)

logger.info("Five anchor pole vectors pre-computed and cached in memory.")


def _dot(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity via dot product on unit vectors. Sign is preserved."""
    return float(np.dot(vec_a, vec_b))


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


def calculate_ai_correlation(
    skill_title: str,
    onet_category_title: str = "",
    context_summary: str = "",
) -> dict:
    """
    Scores a skill against the AI poles and the three enabling anchors.

    Returns unclamped scalars in [-1.0, +1.0] plus the category bucket:

        ai_score          winner-take-all max over {title, summary} x {AI_ENG, AI_GEN}
        tech_base_sim     max over {title, summary} against TECH_BASE_ANCHOR
        ml_pipeline_sim   max over {title, summary} against ML_PIPELINE_ANCHOR
        embedded_ai_sim   max over {title, summary} against EMBEDDED_AI_ANCHOR
        category_bucket   bucket_for(ai_score)

    The three enabling sims do NOT contribute to ai_score. Blending five poles into
    one number would change every score already recorded and make the migrated
    history incomparable with new runs, so the enabling anchors describe HOW a skill
    is adjacent to AI while ai_score alone decides how close it is.

    Two text surfaces are read: the skill title and its definition. Titles are short
    and often ambiguous ("R", "Go"), definitions are long and specific, and taking
    the max means a strong signal in either one survives.

    `onet_category_title` is accepted and unused. It is part of the call signature
    the pipeline already uses, and dropping it would mean touching every call site
    for no gain; keeping it leaves the door open to a taxonomy-aware pole later.
    """
    if not skill_title:
        # Not an error. O*NET occasionally yields a blank example name, and a zeroed
        # record is the honest representation of "nothing to score".
        return {
            "ai_score": 0.0,
            "tech_base_sim": 0.0,
            "ml_pipeline_sim": 0.0,
            "embedded_ai_sim": 0.0,
            "category_bucket": BUCKET_NOT_AI,
            "pathway": "Scalar Winner-Take-All Mapping",
        }

    title_vec = get_embedding(skill_title)

    # Fall back to the title vector when there is no definition, rather than to a
    # zero vector, which would drag every max() down to 0.0 and read as low relevance.
    has_context = bool(context_summary and str(context_summary).strip())
    ctx_vec = get_embedding(context_summary) if has_context else title_vec

    ai_score = max(
        _dot(title_vec, AI_ENG_VEC),
        _dot(title_vec, AI_GEN_VEC),
        _dot(ctx_vec, AI_ENG_VEC),
        _dot(ctx_vec, AI_GEN_VEC),
    )
    tech_base_sim = max(_dot(title_vec, TECH_BASE_VEC), _dot(ctx_vec, TECH_BASE_VEC))
    ml_pipeline_sim = max(_dot(title_vec, ML_PIPELINE_VEC), _dot(ctx_vec, ML_PIPELINE_VEC))
    embedded_ai_sim = max(_dot(title_vec, EMBEDDED_AI_VEC), _dot(ctx_vec, EMBEDDED_AI_VEC))

    return {
        "ai_score": round(ai_score, 4),
        "tech_base_sim": round(tech_base_sim, 4),
        "ml_pipeline_sim": round(ml_pipeline_sim, 4),
        "embedded_ai_sim": round(embedded_ai_sim, 4),
        # Bucketed from the unrounded value so a score sitting exactly on a threshold
        # cannot be rounded across it.
        "category_bucket": bucket_for(ai_score),
        "pathway": "Scalar Winner-Take-All Mapping",
    }

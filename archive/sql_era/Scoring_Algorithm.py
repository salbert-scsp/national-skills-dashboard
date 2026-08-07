"""
Standalone offline vector engine.

Loads the all-MiniLM-L6-v2 bi-encoder (model.onnx) and the local tokenizer file,
projects text into 384-dimensional L2-normalized space, and scores each skill
against four fixed anchor poles using a winner-take-all maximum.

Constraints enforced here:
  - Zero Hugging Face network calls. The tokenizer is loaded exclusively via
    Tokenizer.from_file("tokenizer.json").
  - Truncation at 256 tokens, padding disabled (CPU inference is single-item).
  - Dot products are returned unclamped across [-1.0, +1.0] so anti-correlation
    signal is preserved in the database.
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

# Fail at import rather than silently returning zero vectors. A zero vector scores
# 0.0 against every pole, which is indistinguishable from a real low-relevance
# result and would corrupt the historical metrics table.
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

# Dedicated instance, so truncation and padding configured here cannot be mutated
# by another module that loads the same tokenizer file for the cross-encoder.
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


# --- Anchor poles (reference definitions) ---
AI_ENGINEERING_POLE = (
    "machine learning deep learning neural networks computational inference "
    "tokenization regression clustering predictive analytics tensor framework "
    "model training"
)
AI_GENERATIVE_POLE = (
    "artificial intelligence generative ai large language model llm chatgpt openai "
    "transformer architecture chatbot natural language processing nlp prompt engineering"
)
INFRA_CORE_ANCHOR = (
    "database management relational schema structured query data pipeline backend "
    "architecture version control data warehouse distributed compute virtual computer "
    "data manipulation analytics system software"
)
DEV_LANG_ANCHOR = (
    "programming language development environment source code runtime engine compiler "
    "written in code execution script development syntax object oriented"
)

# Computed once at import and held in memory for the process lifetime.
AI_ENG_VEC = get_embedding(AI_ENGINEERING_POLE)
AI_GEN_VEC = get_embedding(AI_GENERATIVE_POLE)
INFRA_VEC = get_embedding(INFRA_CORE_ANCHOR)
LANG_VEC = get_embedding(DEV_LANG_ANCHOR)

logger.info("Four anchor pole vectors pre-computed and cached in memory.")


def _dot(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity via dot product on unit vectors. Sign is preserved."""
    return float(np.dot(vec_a, vec_b))


def calculate_ai_correlation(
    skill_title: str,
    onet_category_title: str,
    context_summary: str,
) -> dict:
    """
    Scores a skill against the four anchor poles.

    Returns unclamped scalars in [-1.0, +1.0]. Negative values are meaningful and
    are written to the database as-is; DECIMAL(5,4) spans -9.9999 to 9.9999, so the
    full cosine range fits without a schema change.
    """
    if not skill_title:
        return {
            "ai_correlation_score": 0.0,
            "pathway": "Scalar Winner-Take-All Mapping",
            "ai_sim": 0.0,
            "infra_sim": 0.0,
            "lang_sim": 0.0,
        }

    title_vec = get_embedding(skill_title)

    has_context = bool(context_summary and str(context_summary).strip())
    ctx_vec = get_embedding(context_summary) if has_context else title_vec

    # Winner-take-all across both AI poles and both text surfaces.
    ai_sim = max(
        _dot(title_vec, AI_ENG_VEC),
        _dot(title_vec, AI_GEN_VEC),
        _dot(ctx_vec, AI_ENG_VEC),
        _dot(ctx_vec, AI_GEN_VEC),
    )
    infra_sim = max(_dot(title_vec, INFRA_VEC), _dot(ctx_vec, INFRA_VEC))
    lang_sim = max(_dot(title_vec, LANG_VEC), _dot(ctx_vec, LANG_VEC))

    return {
        "ai_correlation_score": round(ai_sim, 4),
        "pathway": "Scalar Winner-Take-All Mapping",
        "ai_sim": round(ai_sim, 4),
        "infra_sim": round(infra_sim, 4),
        "lang_sim": round(lang_sim, 4),
    }

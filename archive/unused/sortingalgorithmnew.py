"""
DEAD, AND UNSAFE TO REVIVE. Kept only as a record of what this used to do.

Line ~7 of the original body calls Tokenizer.from_pretrained(), which DOWNLOADS A
TOKENIZER FROM HUGGING FACE at import time. That breaks the offline guarantee the live
scorer is built on.

The live scorer is the root sortingalgorithmnew.py. It loads tokenizer.json from disk
with Tokenizer.from_file() and makes no network call for models at all -- which is why
its scores are reproducible, why it needs no credentials, and why a network outage
cannot silently change a number that ends up in the time series.

If anything here is ever wanted, port the idea into the live module. Do not import this.
"""

import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# Load Bi-Encoder Tokenizer
tokenizer = Tokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer.enable_truncation(max_length=256, direction="right")

# Initialize ONNX Bi-Encoder Session (model.onnx)
MODEL_PATH = "model.onnx"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Embedding model file '{MODEL_PATH}' was not found in the root directory! "
        "Please ensure model.onnx is present."
    )

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


def get_embedding(text: str) -> np.ndarray:
    """Encodes input text into an L2-normalized 384-dimensional dense vector space."""
    clean_text = str(text).lower().strip() if text else ""
    if not clean_text:
        return np.zeros(384, dtype=np.float32)

    encoded = tokenizer.encode(clean_text)

    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

    onnx_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
    }

    token_embeddings = session.run(None, onnx_inputs)[0]

    # Mean Pooling with Attention Mask
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

    raw_vector = (sum_embeddings / sum_mask)[0]

    # L2 Normalization
    norm = np.linalg.norm(raw_vector)
    return raw_vector / norm if norm > 0 else raw_vector


# --- DEFINITION ANCHORS (Reference Poles) ---
AI_ENGINEERING_POLE = "machine learning deep learning neural networks computational inference tokenization regression clustering predictive analytics tensor framework model training"
AI_GENERATIVE_POLE = "artificial intelligence generative ai large language model llm chatgpt openai transformer architecture chatbot natural language processing nlp prompt engineering"

INFRA_CORE_ANCHOR = "database management relational schema structured query data pipeline backend architecture version control data warehouse distributed compute virtual computer data manipulation analytics system software"
DEV_LANG_ANCHOR = "programming language development environment source code runtime engine compiler written in code execution script development syntax object oriented"

# Compute static reference vectors once at module load time
AI_ENG_VEC = get_embedding(AI_ENGINEERING_POLE)
AI_GEN_VEC = get_embedding(AI_GENERATIVE_POLE)
INFRA_VEC = get_embedding(INFRA_CORE_ANCHOR)
LANG_VEC = get_embedding(DEV_LANG_ANCHOR)


def calculate_ai_correlation(skill_title: str, onet_category_title: str, wikipedia_summary: str) -> dict:
    """Calculates AI correlation and dimensional similarity metrics for a skill."""
    if not skill_title:
        return {
            "ai_correlation_score": 0.0,
            "pathway": "Scalar Winner-Take-All Mapping",
            "ai_sim": 0.0,
            "infra_sim": 0.0,
            "lang_sim": 0.0
        }

    title_vec = get_embedding(skill_title)
    ctx_vec = get_embedding(wikipedia_summary) if wikipedia_summary and str(wikipedia_summary).strip() else title_vec

    # Calculate Cosine Similarities via Dot Products (since vectors are L2 normalized)
    title_ai_eng = float(np.dot(title_vec, AI_ENG_VEC))
    title_ai_gen = float(np.dot(title_vec, AI_GEN_VEC))
    ctx_ai_eng = float(np.dot(ctx_vec, AI_ENG_VEC))
    ctx_ai_gen = float(np.dot(ctx_vec, AI_GEN_VEC))

    ai_sim = max(title_ai_eng, title_ai_gen, ctx_ai_eng, ctx_ai_gen)
    infra_sim = max(float(np.dot(title_vec, INFRA_VEC)), float(np.dot(ctx_vec, INFRA_VEC)))
    lang_sim = max(float(np.dot(title_vec, LANG_VEC)), float(np.dot(ctx_vec, LANG_VEC)))

    return {
        "ai_correlation_score": round(max(0.0, ai_sim), 4),
        "pathway": "Scalar Winner-Take-All Mapping",
        "ai_sim": round(max(0.0, ai_sim), 4),
        "infra_sim": round(max(0.0, infra_sim), 4),
        "lang_sim": round(max(0.0, lang_sim), 4)
    }
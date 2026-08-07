"""
Offline cross-encoder relevance scorer (ms-marco-MiniLM-L6-v2).

Answers one question: is this candidate page actually about this skill? It runs
BEFORE the Gemini audit, so a wrong page is rejected without spending API quota.
Gemini then judges credibility of a page we already believe is correct.

Lives in its own module because both scraping.py (candidate reranking) and the
pipeline need it, and scraping.py cannot import pipeline.py without a cycle.
"""

import logging
import os
from typing import Optional

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

import storage

logger = logging.getLogger(__name__)

CROSS_MODEL_PATH = storage.model_path(storage.CROSS_MODEL_NAME)
TOKENIZER_PATH = storage.model_path(storage.TOKENIZER_NAME)

# Paired with strategy="longest_first" below, which trims the long candidate page and
# never the short skill name. sortingalgorithmnew.py carries its own 256 for the
# bi-encoder; the two agree today but are independent session configs, and coupling
# them would mean swapping one model silently retunes the other.
MAX_SEQUENCE_LENGTH = 256

if not os.path.exists(CROSS_MODEL_PATH):
    raise FileNotFoundError(
        f"Cross-encoder model '{CROSS_MODEL_PATH}' not found in the project root. "
        "cross_encoder_model.onnx (ms-marco-MiniLM-L6-v2) is required to gate page pulls."
    )

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(
        f"Tokenizer file '{TOKENIZER_PATH}' not found in the project root. "
        "The tokenizer must load from disk; remote downloads are prohibited."
    )

# Dedicated instance. Shares the vocabulary file with the bi-encoder (both are
# BERT-uncased, 30522 tokens) but keeps its own truncation and padding settings.
# LongestFirst truncation matters here: it trims the long candidate text rather
# than the short skill name when the pair exceeds the limit.
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
tokenizer.enable_truncation(
    max_length=MAX_SEQUENCE_LENGTH,
    direction="right",
    strategy="longest_first",
)
tokenizer.no_padding()

session = ort.InferenceSession(CROSS_MODEL_PATH, providers=["CPUExecutionProvider"])

logger.info(
    "Cross-encoder ONNX session loaded offline (max_seq=%d, padding disabled).",
    MAX_SEQUENCE_LENGTH,
)


def score_pair(query: str, candidate_title: str, candidate_text: str) -> Optional[float]:
    """
    Scores how well a candidate page matches the query skill.

    Returns a sigmoid-squashed logit in (0.0, 1.0), or None when the pair cannot be
    scored. None means "unknown", never "confident" -- callers must route None to
    manual review rather than substituting a passing score.
    """
    if not query or not candidate_title or not candidate_text:
        return None

    try:
        text_a = str(query).lower().strip()
        text_b = f"{candidate_title}. {candidate_text}".lower().strip()
        encoded = tokenizer.encode(text_a, text_b)

        raw_logits = session.run(
            None,
            {
                "input_ids": np.array([encoded.ids], dtype=np.int64),
                "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
                "token_type_ids": np.array([encoded.type_ids], dtype=np.int64),
            },
        )[0]

        logit_val = float(raw_logits[0][0])
        return round(float(1.0 / (1.0 + np.exp(-logit_val))), 4)
    except Exception:
        logger.exception("Cross-encoder inference failed for query %r", query)
        return None

"""
Multilingual sentence embeddings for blocking and pair features.

Optional: only used when config.USE_EMBEDDINGS is True (or --embeddings).
Needs sentence-transformers and a one-time model download. At full scale
(~12M records per split) a GPU is required — use Kaggle / Colab. The model
must be MIT or Apache-2.0 and ≤ 8B params, and be listed in MODELS.md.

Two vectors per record, both L2-normalized so a dot product is the cosine:
- "name": the raw business name
- "full": "<business name>, <business address>"
Raw text (not our normalized text) is used because the model handles casing,
accents and scripts itself — including Devanagari names in S3, which our
string normalization cannot compare with Latin spellings.
"""

import numpy as np
import pandas as pd

import config

_MODEL = None


def load_model():
    """
    Load the sentence-transformers model once and cache it.

    Returns:
        SentenceTransformer model on GPU if available, else CPU.
    """
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        if _MODEL.device.type == "cuda":
            _MODEL.half()
    return _MODEL


def encode_texts(texts: list, verbose: bool = True) -> np.ndarray:
    """
    Encode texts into L2-normalized float16 embeddings.

    Args:
        texts: List of strings.
        verbose: Whether to show a progress bar.

    Returns:
        Array of shape (len(texts), dim), float16 (half the memory of float32).
    """
    model = load_model()
    vectors = model.encode(
        [config.EMBEDDING_PREFIX + t for t in texts],
        batch_size=config.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=verbose,
    )
    return vectors.astype(np.float16)


def embed_frame(frame: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Embed every record of a source frame.

    Args:
        frame: Frame with business_name and business_address.
        verbose: Whether to print progress.

    Returns:
        Dict with "name" and "full" arrays, rows aligned with the frame.
    """
    if verbose:
        print(f"  Embedding {len(frame):,} records with {config.EMBEDDING_MODEL_NAME}...")
    names = frame["business_name"].tolist()
    full = (frame["business_name"] + ", " + frame["business_address"]).tolist()
    return {
        "name": encode_texts(names, verbose=verbose),
        "full": encode_texts(full, verbose=verbose),
    }

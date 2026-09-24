"""
Multilingual sentence embeddings for blocking and pair features.

Optional: only used when config.USE_EMBEDDINGS is True. Needs
sentence-transformers and a one-time model download; a GPU is recommended
(Kaggle / Colab). The model must be MIT or Apache-2.0 and ≤ 8B params, and be
listed in MODELS.md.

Two vectors per record, both L2-normalized so a dot product is the cosine:
- "name": the raw business name
- "full": "<business name>, <business address>"
Raw text (not our normalized text) is used because the model handles casing,
accents and scripts itself.
"""

import numpy as np

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
    return _MODEL


def encode_texts(texts: list, verbose: bool = True) -> np.ndarray:
    """
    Encode texts into L2-normalized float32 embeddings.

    Args:
        texts: List of strings.
        verbose: Whether to show a progress bar.

    Returns:
        Array of shape (len(texts), dim).
    """
    model = load_model()
    vectors = model.encode(
        [config.EMBEDDING_PREFIX + t for t in texts],
        batch_size=config.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=verbose,
    )
    return vectors.astype(np.float32)


def embed_records(record_lists: list, verbose: bool = True) -> dict:
    """
    Embed every record of one split (S1 + S2 + S3).

    Args:
        record_lists: List of record lists, e.g. [s1_records, s2_records, s3_records].
        verbose: Whether to print progress.

    Returns:
        Dict with:
        - "index": entity_id → row number
        - "name": (n, dim) name embeddings
        - "full": (n, dim) name + address embeddings
    """
    records = [rec for recs in record_lists for rec in recs]
    if verbose:
        print(f"  Embedding {len(records)} records with {config.EMBEDDING_MODEL_NAME}...")

    names = [rec.get("business_name", "") for rec in records]
    full = [
        f"{rec.get('business_name', '')}, {rec.get('business_address', '')}"
        for rec in records
    ]
    return {
        "index": {rec["entity_id"]: i for i, rec in enumerate(records)},
        "name": encode_texts(names, verbose=verbose),
        "full": encode_texts(full, verbose=verbose),
    }


def rows_for(embeddings: dict, entity_ids: list) -> np.ndarray:
    """
    Map entity IDs to their row numbers in the embedding arrays.

    Args:
        embeddings: Output of embed_records.
        entity_ids: List of entity IDs.

    Returns:
        Integer array of row numbers.
    """
    index = embeddings["index"]
    return np.array([index[e] for e in entity_ids], dtype=np.int64)


def pair_cosine(embeddings: dict, field: str, left_ids: list,
                right_ids: list) -> np.ndarray:
    """
    Cosine similarity of the given field for each (left, right) ID pair.

    Args:
        embeddings: Output of embed_records.
        field: "name" or "full".
        left_ids: Entity IDs of the left side of each pair.
        right_ids: Entity IDs of the right side of each pair.

    Returns:
        float32 array of cosines, one per pair.
    """
    vectors = embeddings[field]
    li = rows_for(embeddings, left_ids)
    ri = rows_for(embeddings, right_ids)
    out = np.empty(len(li), dtype=np.float32)
    step = config.FEATURE_CHUNK_SIZE
    for start in range(0, len(li), step):
        end = start + step
        out[start:end] = np.einsum(
            "ij,ij->i", vectors[li[start:end]], vectors[ri[start:end]]
        )
    return out

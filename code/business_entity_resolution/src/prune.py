"""
Stage 2 — pruning: shrink stage-1 candidates to a small set per S1.

The official update says the approach with the smaller candidate set per S1
entity ranks higher, and candidate_pairs.tsv must be exactly what the final
model scores. So between blocking (broad, ~tens of candidates per S1) and the
final matcher we rank each S1's candidates with a light LightGBM on cheap
features and keep only the best few:

    keep = top config.PRUNE_TOP_N per S1  AND  probability ≥ config.PRUNE_MIN_PROB

On train, the pruner scores every pair out-of-fold (grouped by S1), so the
final matcher is trained on pruned sets that look like the test-time ones.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.model_selection import GroupKFold

import config
from blocking import group_rank


def take_strings(frame: pd.DataFrame, column: str, rows: np.ndarray) -> list:
    """
    Values of one text column for the given rows, as a list of Python strings.

    Args:
        frame: Normalized frame.
        column: Column name.
        rows: Row indices.

    Returns:
        List of strings.
    """
    return frame[column].array.take(rows).tolist()


# rapidfuzz scorers used by the pruner: (feature name, field, scorer, scale)
PRUNER_STRING_FEATURES = [
    ("p_name_core_ratio", "name_core", fuzz.ratio, 100.0),
    ("p_name_core_token_set", "name_core", fuzz.token_set_ratio, 100.0),
    ("p_name_jw", "name_norm", JaroWinkler.similarity, 1.0),
    ("p_addr_token_set", "address_norm", fuzz.token_set_ratio, 100.0),
    ("p_addr_partial", "address_norm", fuzz.partial_ratio, 100.0),
]


def pruner_features(pairs: pd.DataFrame, s1: pd.DataFrame, tgt: pd.DataFrame,
                    verbose: bool = True) -> tuple:
    """
    Cheap features for every stage-1 pair.

    Blocker cosines / ranks (already computed in stage 1), the source flag,
    and five multi-threaded rapidfuzz similarities.

    Args:
        pairs: Stage-1 pairs (blocking.generate_candidates output).
        s1: Normalized S1 frame.
        tgt: Normalized S2 + S3 frame.
        verbose: Whether to print progress.

    Returns:
        Tuple (X float32 matrix, feature names).
    """
    cols = {}
    for c in pairs.columns:
        if c.startswith(("score_", "rank_")) or c == "is_s2":
            cols[c] = pairs[c].to_numpy(dtype=np.float32)
    cols["n_blockers"] = np.array(
        [bin(m).count("1") for m in range(16)], dtype=np.float32)[pairs["blockers"].to_numpy()]

    s1_rows, tgt_rows = pairs["s1_idx"].to_numpy(), pairs["tgt_idx"].to_numpy()
    for name, *_ in PRUNER_STRING_FEATURES:
        cols[name] = np.zeros(len(pairs), dtype=np.float32)
    step = config.FEATURE_CHUNK_SIZE
    for start in range(0, len(pairs), step):
        end = start + step
        for field in {f for _, f, _, _ in PRUNER_STRING_FEATURES}:
            left = take_strings(s1, field, s1_rows[start:end])
            right = take_strings(tgt, field, tgt_rows[start:end])
            for name, f, scorer, scale in PRUNER_STRING_FEATURES:
                if f == field:
                    cols[name][start:end] = process.cpdist(
                        left, right, scorer=scorer, dtype=np.float32, workers=-1) / scale

    names = list(cols)
    X = np.column_stack([cols[n] for n in names]) if len(pairs) else \
        np.empty((0, len(names)), dtype=np.float32)
    if verbose:
        print(f"  Pruner features: {len(pairs):,} pairs × {len(names)}")
    return X.astype(np.float32, copy=False), names


def _training_rows(groups: np.ndarray, rows: np.ndarray, max_pairs: int) -> np.ndarray:
    """
    Subsample training rows by whole S1 groups when there are too many.

    Args:
        groups: S1 index per pair.
        rows: Candidate training row indices.
        max_pairs: Upper bound on returned rows.

    Returns:
        Row indices to train on.
    """
    if len(rows) <= max_pairs:
        return rows
    uniq = np.unique(groups[rows])
    rng = np.random.default_rng(config.RANDOM_SEED)
    keep = rng.choice(uniq, size=int(len(uniq) * max_pairs / len(rows)), replace=False)
    return rows[np.isin(groups[rows], keep)]


def fit_pruner(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
               feature_names: list, rows: np.ndarray = None) -> lgb.Booster:
    """
    Train the pruner LightGBM (fixed number of rounds, no early stopping).

    Args:
        X: Pruner features.
        y: Labels.
        groups: S1 index per pair (for subsampling whole S1 entities).
        feature_names: Feature names.
        rows: Rows to train on (default all).

    Returns:
        Trained booster.
    """
    rows = np.arange(len(y)) if rows is None else rows
    rows = _training_rows(groups, rows, config.PRUNE_MAX_TRAIN_PAIRS)
    data = lgb.Dataset(X[rows], label=y[rows], feature_name=feature_names)
    return lgb.train(config.PRUNER_PARAMS, data, num_boost_round=config.PRUNE_ROUNDS)


def oof_pruner_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      feature_names: list, verbose: bool = True) -> np.ndarray:
    """
    Out-of-fold pruner probabilities, grouped by S1 entity.

    Args:
        X: Pruner features.
        y: Labels.
        groups: S1 index per pair.
        feature_names: Feature names.
        verbose: Whether to print progress.

    Returns:
        Probability per pair, each from a model that never saw its S1.
    """
    oof = np.zeros(len(y), dtype=np.float32)
    n_folds = min(config.PRUNE_FOLDS, len(np.unique(groups)))
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups), 1):
        model = fit_pruner(X, y, groups, feature_names, rows=tr)
        oof[va] = model.predict(X[va])
        if verbose:
            print(f"    pruner fold {fold}/{n_folds}: trained on {len(tr):,} pairs")
    return oof


def select(pairs: pd.DataFrame, prob: np.ndarray) -> np.ndarray:
    """
    Rows of `pairs` that survive pruning.

    Args:
        pairs: Stage-1 pairs (with s1_idx).
        prob: Pruner probability per pair.

    Returns:
        Sorted integer row indices to keep.
    """
    rank = group_rank(pairs["s1_idx"].to_numpy(), prob)
    keep = (rank <= config.PRUNE_TOP_N) & (prob >= config.PRUNE_MIN_PROB)
    return np.flatnonzero(keep)

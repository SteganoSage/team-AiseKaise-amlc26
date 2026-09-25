"""
Evaluation module for the entity resolution pipeline.

Implements macro-averaged F0.5 per S1 entity, plus blocking recall.

F0.5 = (1.25 * P * R) / (0.25 * P + R)

Edge cases (per S1 entity):
- true empty & pred empty → 1.0 (correctly predicted no match)
- true empty & pred non-empty → 0.0 (false merges)
- true non-empty & pred empty → 0.0 (missed everything)
- otherwise: standard P/R → F0.5 (0 if no overlap)

Precision is weighted 2× over recall — false merges are expensive.

The dict-based functions are the reference implementation; the pipeline uses
the vectorized per_s1_counts + macro_f05_from_counts (same edge cases) to
score millions of S1 entities quickly.
"""

import numpy as np


def f05_score(precision: float, recall: float) -> float:
    """
    Compute F0.5 from precision and recall.

    F0.5 weighs precision twice as much as recall.

    Args:
        precision: Precision value in [0, 1].
        recall: Recall value in [0, 1].

    Returns:
        F0.5 score in [0, 1].
    """
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def entity_f05(true_set: set, pred_set: set) -> float:
    """
    Compute F0.5 for a single S1 entity.

    Handles all edge cases as specified:
    - true empty & pred empty → 1.0
    - true empty & pred non-empty → 0.0
    - true non-empty & pred empty → 0.0
    - otherwise: P/R → F0.5

    Args:
        true_set: Set of true match IDs for this S1 entity.
        pred_set: Set of predicted match IDs for this S1 entity.

    Returns:
        F0.5 score for this entity.
    """
    true_empty = len(true_set) == 0
    pred_empty = len(pred_set) == 0

    if true_empty and pred_empty:
        return 1.0
    if true_empty and not pred_empty:
        return 0.0
    if not true_empty and pred_empty:
        return 0.0

    # Standard P/R
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = tp / len(true_set) if len(true_set) > 0 else 0.0

    return f05_score(precision, recall)


def macro_f05(predictions: dict, ground_truth: dict) -> float:
    """
    Compute macro-averaged F0.5 over all S1 entities.

    Args:
        predictions: Dict mapping s1_id → list of predicted match IDs.
        ground_truth: Dict mapping s1_id → list/set of true match IDs.

    Returns:
        Macro-averaged F0.5 score.
    """
    scores = []
    for s1_id in ground_truth:
        true_set = set(ground_truth[s1_id]) if not isinstance(ground_truth[s1_id], set) else ground_truth[s1_id]
        pred_set = set(predictions.get(s1_id, []))
        scores.append(entity_f05(true_set, pred_set))

    return sum(scores) / max(len(scores), 1)


def macro_f05_with_details(predictions: dict, ground_truth: dict) -> tuple:
    """
    Compute macro-averaged F0.5 along with macro precision and recall.

    Args:
        predictions: Dict mapping s1_id → list of predicted match IDs.
        ground_truth: Dict mapping s1_id → list/set of true match IDs.

    Returns:
        Tuple of (macro_f05, macro_precision, macro_recall).
    """
    f05_scores = []
    precisions = []
    recalls = []

    for s1_id in ground_truth:
        true_set = set(ground_truth[s1_id]) if not isinstance(ground_truth[s1_id], set) else ground_truth[s1_id]
        pred_set = set(predictions.get(s1_id, []))

        true_empty = len(true_set) == 0
        pred_empty = len(pred_set) == 0

        if true_empty and pred_empty:
            f05_scores.append(1.0)
            precisions.append(1.0)
            recalls.append(1.0)
        elif true_empty and not pred_empty:
            f05_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(1.0)  # nothing to recall
        elif not true_empty and pred_empty:
            f05_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
        else:
            tp = len(true_set & pred_set)
            p = tp / len(pred_set) if len(pred_set) > 0 else 0.0
            r = tp / len(true_set) if len(true_set) > 0 else 0.0
            f05_scores.append(f05_score(p, r))
            precisions.append(p)
            recalls.append(r)

    n = max(len(f05_scores), 1)
    return sum(f05_scores) / n, sum(precisions) / n, sum(recalls) / n


# ──────────────────────────────────────────────────────────────────────
# Vectorized version for millions of S1 entities (same edge cases)
# ──────────────────────────────────────────────────────────────────────

def per_s1_counts(universe: np.ndarray, n_true_by_row: np.ndarray,
                  kept_s1: np.ndarray, kept_label: np.ndarray) -> tuple:
    """
    Per-S1 counts of true links, predicted links and correct predictions.

    Args:
        universe: S1 row indices being evaluated (every one counts, including
            S1s with no candidates or no predictions).
        n_true_by_row: Number of true links per S1 row (all S1 rows).
        kept_s1: S1 row of each predicted pair.
        kept_label: 1 if the predicted pair is a true link, else 0.

    Returns:
        Tuple (n_true, n_pred, n_tp) arrays aligned with `universe`.
    """
    n_rows = len(n_true_by_row)
    n_pred = np.bincount(kept_s1, minlength=n_rows)
    n_tp = np.bincount(kept_s1, weights=kept_label, minlength=n_rows)
    return n_true_by_row[universe], n_pred[universe], n_tp[universe]


def macro_f05_from_counts(n_true: np.ndarray, n_pred: np.ndarray,
                          n_tp: np.ndarray) -> tuple:
    """
    Macro F0.5 / precision / recall from per-S1 counts.

    Same edge cases as macro_f05_with_details:
    - true empty & pred empty      → F0.5 1, P 1, R 1
    - true empty & pred non-empty  → F0.5 0, P 0, R 1
    - true non-empty & pred empty  → F0.5 0, P 0, R 0
    - otherwise standard P/R → F0.5 (0 if no overlap)

    Args:
        n_true: True links per S1.
        n_pred: Predicted links per S1.
        n_tp: Correct predicted links per S1.

    Returns:
        Tuple (macro_f05, macro_precision, macro_recall).
    """
    n_true = np.asarray(n_true, dtype=np.float64)
    n_pred = np.asarray(n_pred, dtype=np.float64)
    n_tp = np.asarray(n_tp, dtype=np.float64)
    if len(n_true) == 0:
        return 0.0, 0.0, 0.0

    both_empty = (n_true == 0) & (n_pred == 0)
    only_pred = (n_true == 0) & (n_pred > 0)
    normal = (n_true > 0) & (n_pred > 0)

    p = np.where(normal, n_tp / np.maximum(n_pred, 1), 0.0)
    r = np.where(normal, n_tp / np.maximum(n_true, 1), 0.0)
    f = np.where(normal & (n_tp > 0), 1.25 * p * r / np.maximum(0.25 * p + r, 1e-12), 0.0)

    f = np.where(both_empty, 1.0, f)
    p = np.where(both_empty, 1.0, p)
    r = np.where(both_empty | only_pred, 1.0, r)
    return float(f.mean()), float(p.mean()), float(r.mean())

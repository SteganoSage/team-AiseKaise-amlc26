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
"""


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

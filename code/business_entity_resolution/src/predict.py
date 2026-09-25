"""
Prediction module for entity resolution.

Takes a trained LightGBM model and candidate pairs, scores each pair,
and turns scores into final matches with a single decision rule
(decide). The same rule is used when tuning the threshold and when
writing predictions, so the tuned threshold is the one that is optimal for
the output we actually submit.

Decision rule:
1. Keep pairs with probability ≥ threshold (one shared threshold, or one
   per source when config.PER_SOURCE_THRESHOLD is on; each S1's best
   candidate gets its own "top1" threshold when config.TOP1_THRESHOLD is on).
2. One-to-one assignment: an S2/S3 record goes to at most one S1 (the
   highest-scoring one), since S1 is deduplicated.
3. Optional per-S1 top-N cap.

Everything works on integer index arrays, so millions of pairs take seconds.
"""

import numpy as np

import config
import evaluate as eval_module
from blocking import group_rank


def predict_scores(model, X: np.ndarray) -> np.ndarray:
    """
    Predict match probabilities for candidate pairs.

    Args:
        model: Trained LightGBM Booster.
        X: Feature matrix of shape (n_pairs, n_features).

    Returns:
        Array of match probabilities in [0, 1].
    """
    if len(X) == 0:
        return np.zeros(0)
    return model.predict(X, num_iteration=model.best_iteration)


def format_threshold(threshold) -> str:
    """
    Human-readable threshold: "0.870" or "S2 0.850 / S3 0.910 / top1 0.600".

    Args:
        threshold: Float, or dict with "S2"/"S3" or "base", and optionally "top1".

    Returns:
        Formatted string.
    """
    if isinstance(threshold, dict):
        return " / ".join(f"{src} {t:.3f}" for src, t in sorted(threshold.items()))
    return f"{threshold:.3f}"


def pair_thresholds(is_s2: np.ndarray, threshold, is_top1: np.ndarray = None):
    """
    Resolve the threshold that applies to each pair.

    Args:
        is_s2: 1 for S2 candidates, 0 for S3.
        threshold: Float (same for all pairs), or dict with {"S2": t, "S3": t}
            or {"base": t}, plus optionally {"top1": t} for each S1's best pair.
        is_top1: True for the best-scoring pair of its S1 (needed for "top1").

    Returns:
        The float itself, or an array with one threshold per pair.
    """
    if not isinstance(threshold, dict):
        return threshold
    if "S2" in threshold:
        per_pair = np.where(is_s2 == 1, threshold["S2"], threshold["S3"])
    else:
        per_pair = np.full(len(is_s2), threshold["base"])
    if "top1" in threshold and is_top1 is not None:
        per_pair = np.where(is_top1, threshold["top1"], per_pair)
    return per_pair


def decide(scores: np.ndarray, s1_idx: np.ndarray, tgt_idx: np.ndarray,
           is_s2: np.ndarray, threshold=None, resolve_conflicts: bool = None,
           max_per_s1: int = None) -> np.ndarray:
    """
    Turn pair scores into final matches.

    Args:
        scores: Match probability per pair.
        s1_idx: S1 row per pair.
        tgt_idx: Target row per pair.
        is_s2: 1 for S2 candidates, 0 for S3.
        threshold: Cutoff (float or {"S2": t, "S3": t}). Default config.MATCH_THRESHOLD.
        resolve_conflicts: Assign each S2/S3 record to at most one S1 (the
            highest-scoring). Default config.RESOLVE_CONFLICTS.
        max_per_s1: Keep at most this many matches per S1. Default
            config.MAX_MATCHES_PER_S1 (None = no cap).

    Returns:
        Sorted indices of the pairs predicted as matches.
    """
    if threshold is None:
        threshold = config.MATCH_THRESHOLD
    if resolve_conflicts is None:
        resolve_conflicts = config.RESOLVE_CONFLICTS
    if max_per_s1 is None:
        max_per_s1 = config.MAX_MATCHES_PER_S1

    is_top1 = None
    if isinstance(threshold, dict) and "top1" in threshold:
        is_top1 = group_rank(s1_idx, scores) == 1
    keep = np.flatnonzero(scores >= pair_thresholds(is_s2, threshold, is_top1))
    if resolve_conflicts and len(keep):
        order = keep[np.argsort(-scores[keep], kind="stable")]
        _, first = np.unique(tgt_idx[order], return_index=True)
        keep = order[first]  # best-scoring S1 for each target
    if max_per_s1 and len(keep):
        rank = group_rank(s1_idx[keep], scores[keep])
        keep = keep[rank <= max_per_s1]
    return np.sort(keep)


def evaluate_decision(keep: np.ndarray, s1_idx: np.ndarray, labels: np.ndarray,
                      n_true_by_row: np.ndarray, universe: np.ndarray) -> tuple:
    """
    Macro F0.5 / precision / recall of a set of predicted pairs.

    Args:
        keep: Indices of predicted pairs.
        s1_idx: S1 row per pair.
        labels: 1 for true pairs.
        n_true_by_row: True links per S1 row (counts links blocking missed).
        universe: S1 rows being evaluated.

    Returns:
        Tuple (macro_f05, macro_precision, macro_recall).
    """
    counts = eval_module.per_s1_counts(universe, n_true_by_row, s1_idx[keep], labels[keep])
    return eval_module.macro_f05_from_counts(*counts)


def tune_threshold(scores: np.ndarray, s1_idx: np.ndarray, tgt_idx: np.ndarray,
                   is_s2: np.ndarray, labels: np.ndarray, n_true_by_row: np.ndarray,
                   universe: np.ndarray, thresholds: list = None,
                   verbose: bool = True) -> tuple:
    """
    Tune the decision threshold to maximize macro F0.5.

    Should be given out-of-fold scores, so the threshold is not fitted on the
    same predictions it is evaluated on. Every S1 in `universe` counts,
    including those with no candidates (they count as predicted-empty).

    Args:
        scores: Match probability per pair.
        s1_idx: S1 row per pair.
        tgt_idx: Target row per pair.
        is_s2: 1 for S2 candidates.
        labels: 1 for true pairs.
        n_true_by_row: True links per S1 row.
        universe: S1 rows to evaluate over.
        thresholds: Thresholds to try. Defaults to config.THRESHOLD_GRID.
        verbose: Whether to print progress.

    Returns:
        Tuple (best_threshold, best_f05). best_threshold is a dict
        {"S2": t, "S3": t} when config.PER_SOURCE_THRESHOLD is on.
    """
    if thresholds is None:
        start, stop, step = config.THRESHOLD_GRID
        thresholds = np.round(np.arange(start, stop, step), 4)

    # Pairs below the lowest threshold can never be predicted: drop them once
    sub = np.flatnonzero(scores >= min(thresholds))
    sub_scores, sub_s1, sub_tgt = scores[sub], s1_idx[sub], tgt_idx[sub]
    sub_s2, sub_labels = is_s2[sub], labels[sub]

    def score(threshold) -> tuple:
        """Macro F0.5, precision, recall of the decision rule at `threshold`."""
        keep = decide(sub_scores, sub_s1, sub_tgt, sub_s2, threshold)
        return evaluate_decision(keep, sub_s1, sub_labels, n_true_by_row, universe)

    rows = [(float(t), *score(float(t))) for t in thresholds]
    best_threshold, best_f05, _, _ = max(rows, key=lambda r: r[1])
    global_threshold = best_threshold

    if verbose:
        print("\n  Threshold tuning (macro over S1 entities):")
        print(f"  {'Threshold':>10s}  {'F0.5':>8s}  {'Precision':>10s}  {'Recall':>8s}")
        print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}")
        for i, (t, f05, p, r) in enumerate(rows):
            if i % 5 == 0 or t == best_threshold:
                marker = "  ←" if t == best_threshold else ""
                print(f"  {t:>10.3f}  {f05:>8.4f}  {p:>10.4f}  {r:>8.4f}{marker}")
        print(f"\n  → Best threshold: {best_threshold:.3f} (F0.5 = {best_f05:.4f})")
        if best_threshold in (rows[0][0], rows[-1][0]):
            print("  ⚠ Best threshold is at the edge of config.THRESHOLD_GRID — "
                  "widen the grid.")

    if config.PER_SOURCE_THRESHOLD:
        # Coordinate search: move one source's threshold at a time
        best = {"S2": global_threshold, "S3": global_threshold}
        for _ in range(2):
            for src in ("S2", "S3"):
                for t in thresholds:
                    trial = {**best, src: float(t)}
                    f05 = score(trial)[0]
                    if f05 > best_f05:
                        best, best_f05 = trial, f05
        best_threshold = best
        if verbose:
            print(f"  → Per-source thresholds: {format_threshold(best)} "
                  f"(F0.5 = {best_f05:.4f})")

    if config.TOP1_THRESHOLD:
        # Own threshold for each S1's best candidate; then re-check the others
        best = dict(best_threshold) if isinstance(best_threshold, dict) \
            else {"base": best_threshold}
        best["top1"] = best.get("S2", best.get("base"))
        for key in ("top1", *[k for k in best if k != "top1"], "top1"):
            for t in thresholds:
                trial = {**best, key: float(t)}
                f05 = score(trial)[0]
                if f05 > best_f05:
                    best, best_f05 = trial, f05
        best_threshold = best
        if verbose:
            print(f"  → With top-1 threshold: {format_threshold(best)} "
                  f"(F0.5 = {best_f05:.4f})")

    return best_threshold, best_f05

"""
Prediction module for entity resolution.

Takes a trained LightGBM model and candidate pairs, scores each pair,
and turns scores into final matches with a single decision rule
(decide_matches). The same rule is used when tuning the threshold and when
writing predictions, so the tuned threshold is the one that is optimal for
the output we actually submit.

Decision rule:
1. Keep pairs with probability ≥ threshold (one shared threshold, or one
   per source when config.PER_SOURCE_THRESHOLD is on).
2. One-to-one assignment: an S2/S3 record goes to at most one S1 (the
   highest-scoring one), since S1 is deduplicated.
3. Optional per-S1 top-N cap.
"""

import numpy as np
import config
import evaluate as eval_module


def predict_scores(model, X: np.ndarray) -> np.ndarray:
    """
    Predict match probabilities for candidate pairs.

    Args:
        model: Trained LightGBM Booster.
        X: Feature matrix of shape (n_pairs, n_features).

    Returns:
        Array of match probabilities in [0, 1].
    """
    return model.predict(X, num_iteration=model.best_iteration)


def format_threshold(threshold) -> str:
    """
    Human-readable threshold: "0.870" or "S2 0.850 / S3 0.910".

    Args:
        threshold: Float, or dict {"S2": float, "S3": float}.

    Returns:
        Formatted string.
    """
    if isinstance(threshold, dict):
        return " / ".join(f"{src} {t:.3f}" for src, t in sorted(threshold.items()))
    return f"{threshold:.3f}"


def pair_thresholds(pairs: list, threshold):
    """
    Resolve the threshold that applies to each pair.

    Args:
        pairs: List of (s1_id, cand_id) tuples.
        threshold: Float (same for all pairs), or dict {"S2": t, "S3": t}
            keyed by the candidate ID prefix.

    Returns:
        The float itself, or an array with one threshold per pair.
    """
    if isinstance(threshold, dict):
        return np.array([threshold[cand_id[:2]] for _, cand_id in pairs])
    return threshold


def decide_matches(scores: np.ndarray, pairs: list, threshold=None,
                   resolve_conflicts: bool = None,
                   max_per_s1: int = None) -> dict:
    """
    Turn pair scores into final matches.

    Pairs are processed from highest to lowest score, so conflict resolution
    and the per-S1 cap always keep the most confident pairs.

    Args:
        scores: Array of match probabilities (same order as pairs).
        pairs: List of (s1_id, cand_id) tuples.
        threshold: Probability cutoff, or dict {"S2": t, "S3": t}.
            Defaults to config.MATCH_THRESHOLD.
        resolve_conflicts: Assign each S2/S3 record to at most one S1.
            Defaults to config.RESOLVE_CONFLICTS.
        max_per_s1: Keep at most this many matches per S1 (None = no cap).
            Defaults to config.MAX_MATCHES_PER_S1.

    Returns:
        Dict mapping s1_id → list of matched candidate IDs (S1 entities with
        no match are absent).
    """
    if threshold is None:
        threshold = config.MATCH_THRESHOLD
    if resolve_conflicts is None:
        resolve_conflicts = config.RESOLVE_CONFLICTS
    if max_per_s1 is None:
        max_per_s1 = config.MAX_MATCHES_PER_S1

    scores = np.asarray(scores)
    keep = np.flatnonzero(scores >= pair_thresholds(pairs, threshold))
    keep = keep[np.argsort(-scores[keep], kind="stable")]

    matches = {}
    assigned = set()
    for i in keep:
        s1_id, cand_id = pairs[i]
        if resolve_conflicts and cand_id in assigned:
            continue
        if max_per_s1 and len(matches.get(s1_id, [])) >= max_per_s1:
            continue
        matches.setdefault(s1_id, []).append(cand_id)
        assigned.add(cand_id)

    return matches


def tune_threshold(scores: np.ndarray, pairs: list,
                   ground_truth: dict, s1_ids: list,
                   thresholds: list = None,
                   verbose: bool = True) -> tuple:
    """
    Tune the decision threshold to maximize macro F0.5.

    Should be given out-of-fold scores, so the threshold is not fitted on the
    same predictions it is evaluated on. Every S1 in s1_ids is scored,
    including those with no candidates (they count as predicted-empty).

    Args:
        scores: Array of match probabilities.
        pairs: List of (s1_id, cand_id) tuples.
        ground_truth: Dict mapping s1_id → set of true match IDs.
        s1_ids: List of all S1 entity IDs to evaluate over.
        thresholds: Thresholds to try. Defaults to config.THRESHOLD_GRID.
        verbose: Whether to print progress.

    Returns:
        Tuple of (best_threshold, best_f05_score). best_threshold is a dict
        {"S2": t, "S3": t} when config.PER_SOURCE_THRESHOLD is on.
    """
    if thresholds is None:
        start, stop, step = config.THRESHOLD_GRID
        thresholds = np.round(np.arange(start, stop, step), 4)

    # Pairs below the lowest threshold can never be predicted: drop them once
    scores = np.asarray(scores)
    idx = np.flatnonzero(scores >= min(thresholds))
    sub_scores = scores[idx]
    sub_pairs = [pairs[i] for i in idx]
    gt = {s1_id: ground_truth.get(s1_id, set()) for s1_id in s1_ids}

    def score(threshold) -> tuple:
        """Macro F0.5, precision, recall of the decision rule at `threshold`."""
        matches = decide_matches(sub_scores, sub_pairs, threshold=threshold)
        return eval_module.macro_f05_with_details(matches, gt)

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

    return best_threshold, best_f05

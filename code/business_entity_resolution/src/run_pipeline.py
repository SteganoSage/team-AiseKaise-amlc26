"""
End-to-end pipeline for Business Entity Resolution.

Usage (from the repo root):
    python code/business_entity_resolution/src/run_pipeline.py --mode validate
    python code/business_entity_resolution/src/run_pipeline.py --mode loco
    python code/business_entity_resolution/src/run_pipeline.py --mode test

    # Kaggle: dataset is read-only under /kaggle/input, outputs go to /kaggle/working
    python code/business_entity_resolution/src/run_pipeline.py --mode test \
        --data-dir /kaggle/input/<dataset>/dataset \
        --output-dir /kaggle/working/output --model-dir /kaggle/working/models

Modes:
- validate: Hold out 20% of train S1 entities. Fit on the rest (group K-fold
            out-of-fold scores pick the threshold and boosting rounds), then
            report blocking recall and macro F0.5 on the untouched holdout,
            per country, and dump the holdout errors.
- loco:     Leave one country out: for each train country, fit on the other
            countries and evaluate on it. Our stand-in for the unseen test
            country (France).
- test:     Fit the same way on all train data, predict on the test set,
            write output/*.tsv and run utils/validate_submission.py.

Blocking and features always run once over ALL S1 records of a split (the
rank features compare each pair with its competitors); modes then select
pairs by S1 entity. Labels are never used before fitting, so this is not
leakage — at test time all test S1 records are also scored together.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

# Ensure the src directory is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import io_utils
import normalize
import blocking
import features
import train
import predict
import evaluate


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

def load_and_normalize(source1_path: str, source2_path: str,
                       source3_path: str, verbose: bool = True) -> tuple:
    """
    Load source TSVs and normalize all records.

    Args:
        source1_path: Path to source 1 TSV.
        source2_path: Path to source 2 TSV.
        source3_path: Path to source 3 TSV.
        verbose: Whether to print progress.

    Returns:
        Tuple of (s1_records, s2_records, s3_records) as lists of dicts.
    """
    if verbose:
        print("Loading and normalizing data...")

    df1 = io_utils.read_source_tsv(source1_path)
    df2 = io_utils.read_source_tsv(source2_path)
    df3 = io_utils.read_source_tsv(source3_path)

    if verbose:
        print(f"  S1: {len(df1)} records, S2: {len(df2)} records, S3: {len(df3)} records")
        for name, df in [("S1", df1), ("S2", df2), ("S3", df3)]:
            counts = df["country"].value_counts().head(5).to_dict()
            print(f"  {name} countries: {counts}")

    s1_records = normalize.normalize_dataframe(df1)
    s2_records = normalize.normalize_dataframe(df2)
    s3_records = normalize.normalize_dataframe(df3)

    if verbose:
        print("  ✓ Normalization complete")

    return s1_records, s2_records, s3_records


def load_ground_truth(path: str, s1_ids: list = None) -> dict:
    """
    Load and parse the ground truth file.

    Args:
        path: Path to ground truth TSV.
        s1_ids: All S1 entity IDs. Any S1 missing from the file gets an empty
            set (a singleton), so every S1 entity is scored.

    Returns:
        Dict mapping s1_id → set of true match IDs.
    """
    df = io_utils.read_ground_truth(path)
    gt = {s1_id: set() for s1_id in (s1_ids or [])}
    for s1_id, matched in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        gt[s1_id] = set(io_utils.parse_id_list(matched))
    return gt


def records_to_map(records: list) -> dict:
    """
    Convert a list of record dicts to a dict keyed by entity_id.

    Args:
        records: List of normalized record dicts.

    Returns:
        Dict mapping entity_id → record dict.
    """
    return {rec["entity_id"]: rec for rec in records}


def load_split(split: str, verbose: bool = True) -> dict:
    """
    Load, normalize and (optionally) embed one split.

    Args:
        split: "train" or "test".
        verbose: Whether to print progress.

    Returns:
        Dict with s1 / s2 / s3 record lists, s1_ids, s1_map, target_map
        (S2 + S3 by ID), embeddings (or None), and ground_truth (train only).
    """
    if split == "train":
        paths = (config.TRAIN_SOURCE1, config.TRAIN_SOURCE2, config.TRAIN_SOURCE3)
    else:
        paths = (config.TEST_SOURCE1, config.TEST_SOURCE2, config.TEST_SOURCE3)
    s1, s2, s3 = load_and_normalize(*paths, verbose=verbose)

    data = {
        "s1": s1, "s2": s2, "s3": s3,
        "s1_ids": [r["entity_id"] for r in s1],
        "s1_map": records_to_map(s1),
        "target_map": {**records_to_map(s2), **records_to_map(s3)},
        "embeddings": None,
    }
    if config.USE_EMBEDDINGS:
        import embeddings
        data["embeddings"] = embeddings.embed_records([s1, s2, s3], verbose=verbose)
    if split == "train":
        data["ground_truth"] = load_ground_truth(config.TRAIN_GROUND_TRUTH, data["s1_ids"])
        gt = data["ground_truth"]
        n_with_matches = sum(1 for v in gt.values() if v)
        print(f"  Ground truth: {len(gt)} S1 entities ({n_with_matches} with matches, "
              f"{len(gt) - n_with_matches} singletons)")
    return data


# ──────────────────────────────────────────────────────────────────────
# Pipeline stages
# ──────────────────────────────────────────────────────────────────────

def build_pairs(data: dict, verbose: bool = True) -> dict:
    """
    Blocking + pair features for every S1 record of a split, in one pass.

    Args:
        data: Output of load_split.
        verbose: Whether to print progress.

    Returns:
        Dict with candidates, sources, X, pairs, feature_names, and
        pair_s1 (numpy array of the S1 ID of each pair, for selection).
    """
    candidates, sources = blocking.generate_candidates(
        data["s1"], data["s2"], data["s3"],
        embeddings=data["embeddings"], verbose=verbose,
    )
    X, pairs, feature_names = features.build_feature_matrix(
        data["s1_map"], data["target_map"], candidates,
        sources=sources, embeddings=data["embeddings"], verbose=verbose,
    )
    return {
        "candidates": candidates,
        "sources": sources,
        "X": X,
        "pairs": pairs,
        "feature_names": feature_names,
        "pair_s1": np.array([p[0] for p in pairs]),
    }


def select_pairs(built: dict, s1_ids: set) -> tuple:
    """
    Select the rows of the feature matrix that belong to the given S1 entities.

    Args:
        built: Output of build_pairs.
        s1_ids: Set of S1 entity IDs to keep.

    Returns:
        Tuple of (X subset, pairs subset).
    """
    mask = np.isin(built["pair_s1"], list(s1_ids))
    idx = np.flatnonzero(mask)
    return built["X"][idx], [built["pairs"][i] for i in idx]


def fit_matcher(X: np.ndarray, pairs: list, feature_names: list,
                ground_truth: dict, s1_ids: list, verbose: bool = True) -> dict:
    """
    Fit the matcher on labelled pairs.

    Group K-fold OOF scores → tune threshold on OOF (with the same decision
    rule used for final predictions) → train the final LightGBM on all pairs
    with the mean best iteration from the folds.

    Args:
        X: Feature matrix of the labelled pairs.
        pairs: List of (s1_id, cand_id), same order as X.
        feature_names: Feature names.
        ground_truth: Dict s1_id → set of true match IDs, for every S1 in s1_ids.
        s1_ids: All labelled S1 entity IDs (including ones with no candidates).
        verbose: Whether to print progress.

    Returns:
        Dict with model, threshold, num_boost_round, oof_f05, n_pairs,
        n_positives.
    """
    y = train.create_labels(pairs, ground_truth)
    if verbose:
        print(f"\n  Pairs: {len(y):,}, positives: {int(y.sum())} "
              f"({100 * y.mean():.2f}%)")

    print(f"\n[fit] {config.CV_FOLDS}-fold group CV (by S1 entity)...")
    groups = np.array([p[0] for p in pairs])
    oof, best_iters = train.cross_validate_oof(
        X, y, groups, feature_names=feature_names, verbose=verbose
    )
    threshold, oof_f05 = predict.tune_threshold(
        oof, pairs, ground_truth, s1_ids, verbose=verbose
    )

    num_boost_round = max(int(round(np.mean(best_iters))), 1)
    print(f"\n[fit] Final LightGBM on all pairs ({num_boost_round} rounds)...")
    model = train.train_model(
        X, y, feature_names=feature_names, num_boost_round=num_boost_round,
        verbose=verbose,
    )

    return {
        "model": model,
        "threshold": threshold,
        "num_boost_round": num_boost_round,
        "oof_f05": oof_f05,
        "n_pairs": len(y),
        "n_positives": int(y.sum()),
    }


def predict_matches(fitted: dict, X: np.ndarray, pairs: list,
                    s1_ids: list) -> tuple:
    """
    Score pairs with a fitted matcher and apply its decision rule.

    Args:
        fitted: Output of fit_matcher.
        X: Feature matrix.
        pairs: List of (s1_id, cand_id), same order as X.
        s1_ids: Every S1 ID that needs a prediction (empty list if no match).

    Returns:
        Tuple of (predictions dict s1_id → list of IDs, scores array).
    """
    scores = predict.predict_scores(fitted["model"], X) if len(pairs) else np.zeros(0)
    matches = predict.decide_matches(scores, pairs, fitted["threshold"])
    return {s1_id: matches.get(s1_id, []) for s1_id in s1_ids}, scores


# ──────────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────────

def report_by_country(predictions: dict, ground_truth: dict, s1_map: dict) -> dict:
    """
    Print macro F0.5 / precision / recall per S1 country.

    Args:
        predictions: Dict s1_id → predicted IDs.
        ground_truth: Dict s1_id → true IDs (the S1s to report on).
        s1_map: Dict s1_id → normalized S1 record.

    Returns:
        Dict country → {"n": ..., "f05": ..., "precision": ..., "recall": ...}.
    """
    by_country = {}
    for s1_id in ground_truth:
        by_country.setdefault(s1_map[s1_id].get("country_norm", ""), []).append(s1_id)

    report = {}
    print(f"\n  {'country':>12s}  {'S1':>6s}  {'singletons':>10s}  {'F0.5':>7s}  "
          f"{'P':>7s}  {'R':>7s}")
    for country, ids in sorted(by_country.items()):
        gt = {s: ground_truth[s] for s in ids}
        f05, p, r = evaluate.macro_f05_with_details(predictions, gt)
        singletons = sum(1 for s in ids if not gt[s]) / len(ids)
        report[country] = {"n": len(ids), "f05": f05, "precision": p, "recall": r}
        print(f"  {country or '(none)':>12s}  {len(ids):>6d}  {singletons:>10.1%}  "
              f"{f05:>7.4f}  {p:>7.4f}  {r:>7.4f}")
    return report


def write_error_dump(path: str, predictions: dict, ground_truth: dict,
                     candidates: dict, pairs: list, scores: np.ndarray,
                     s1_map: dict, target_map: dict) -> int:
    """
    Write every wrong decision on the holdout, with names and addresses.

    Error types:
    - FP: predicted but not a true match
    - FN_scored: true match that was a candidate but scored below threshold
      (or lost to another S1 in conflict resolution)
    - FN_not_shortlisted: true match that blocking never proposed

    Args:
        path: Output TSV path.
        predictions: Dict s1_id → predicted IDs.
        ground_truth: Dict s1_id → true IDs.
        candidates: Dict s1_id → candidate IDs.
        pairs: Scored pairs (s1_id, cand_id).
        scores: Score of each pair.
        s1_map: Dict S1 ID → record.
        target_map: Dict S2/S3 ID → record.

    Returns:
        Number of error rows written.
    """
    score_of = {pair: float(s) for pair, s in zip(pairs, scores)}
    rows = []
    for s1_id, true_ids in ground_truth.items():
        pred = set(predictions.get(s1_id, []))
        cands = set(candidates.get(s1_id, []))
        errors = [(c, "FP") for c in sorted(pred - true_ids)]
        errors += [(c, "FN_scored" if c in cands else "FN_not_shortlisted")
                   for c in sorted(true_ids - pred)]
        for cand_id, kind in errors:
            a, b = s1_map[s1_id], target_map.get(cand_id, {})
            score = score_of.get((s1_id, cand_id))
            rows.append([
                kind, s1_id, cand_id, "" if score is None else f"{score:.4f}",
                a.get("business_name", ""), b.get("business_name", ""),
                a.get("business_address", ""), b.get("business_address", ""),
                a.get("country", ""), b.get("country", ""),
            ])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("error\ts1_id\tcand_id\tscore\ts1_name\tcand_name\t"
                "s1_address\tcand_address\ts1_country\tcand_country\n")
        for row in rows:
            f.write("\t".join(str(x).replace("\t", " ") for x in row) + "\n")
    return len(rows)


def save_run_info(mode: str, info: dict) -> None:
    """
    Save the key numbers of a run to <model_dir>/run_info_<mode>.json.

    These are the numbers to copy into submissions/LOG.md.

    Args:
        mode: 'validate', 'loco' or 'test'.
        info: JSON-serializable dict of run results.
    """
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    path = os.path.join(config.MODEL_DIR, f"run_info_{mode}.json")
    info = {"mode": mode, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "use_embeddings": config.USE_EMBEDDINGS,
            "feature_groups": config.FEATURE_GROUPS, **info}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"  ✓ Run info saved to {path}")


# ──────────────────────────────────────────────────────────────────────
# Modes
# ──────────────────────────────────────────────────────────────────────

def run_validate(verbose: bool = True):
    """
    Run the validation pipeline and report macro F0.5 on a held-out split.

    Args:
        verbose: Whether to print progress.

    Returns:
        Holdout macro F0.5.
    """
    start_time = time.time()
    print("=" * 60)
    print("VALIDATION MODE")
    print("=" * 60)

    data = load_split("train", verbose=verbose)
    gt = data["ground_truth"]

    print("\n[pairs] Blocking + features for all train S1 entities...")
    built = build_pairs(data, verbose=verbose)
    blocking.evaluate_blocking_recall(built["candidates"], gt,
                                      sources=built["sources"], verbose=verbose)

    # Split S1 entities: dev (fit + CV) and an untouched holdout
    dev_ids, holdout_ids = train.holdout_split_s1(data["s1_ids"])
    print(f"\n  Dev S1: {len(dev_ids)}, Holdout S1: {len(holdout_ids)}")
    dev_gt = {k: gt[k] for k in dev_ids}
    holdout_gt = {k: gt[k] for k in holdout_ids}

    X_dev, pairs_dev = select_pairs(built, dev_ids)
    fitted = fit_matcher(X_dev, pairs_dev, built["feature_names"], dev_gt,
                         sorted(dev_ids), verbose=verbose)

    print("\n[holdout] Scoring held-out S1 entities...")
    holdout_candidates = {s: built["candidates"].get(s, []) for s in holdout_ids}
    holdout_recall = blocking.evaluate_blocking_recall(
        holdout_candidates, holdout_gt, verbose=verbose
    )
    X_hold, pairs_hold = select_pairs(built, holdout_ids)
    predictions, scores = predict_matches(fitted, X_hold, pairs_hold, sorted(holdout_ids))

    f05, precision, recall = evaluate.macro_f05_with_details(predictions, holdout_gt)
    empty_f05 = evaluate.macro_f05({}, holdout_gt)
    by_country = report_by_country(predictions, holdout_gt, data["s1_map"])

    if config.WRITE_ERROR_DUMP:
        path = os.path.join(config.MODEL_DIR, "holdout_errors.tsv")
        n_errors = write_error_dump(path, predictions, holdout_gt, holdout_candidates,
                                    pairs_hold, scores, data["s1_map"], data["target_map"])
        print(f"\n  ✓ {n_errors} holdout errors written to {path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  Holdout macro F0.5:    {f05:.4f}   (all-empty baseline {empty_f05:.4f})")
    print(f"  Holdout precision:     {precision:.4f}")
    print(f"  Holdout recall:        {recall:.4f}")
    print(f"  Holdout block recall:  {holdout_recall:.4f}")
    print(f"  OOF F0.5 (dev):        {fitted['oof_f05']:.4f}")
    print(f"  Threshold:             {predict.format_threshold(fitted['threshold'])}")
    print(f"  Time elapsed:          {elapsed:.1f}s")
    print("=" * 60)

    save_run_info("validate", {
        "holdout_f05": f05,
        "holdout_precision": precision,
        "holdout_recall": recall,
        "holdout_blocking_recall": holdout_recall,
        "all_empty_f05": empty_f05,
        "holdout_by_country": by_country,
        "oof_f05": fitted["oof_f05"],
        "threshold": fitted["threshold"],
        "num_boost_round": fitted["num_boost_round"],
        "n_features": len(built["feature_names"]),
        "elapsed_s": round(elapsed, 1),
    })

    return f05


def run_loco(verbose: bool = True) -> dict:
    """
    Leave-one-country-out: fit on all other train countries, evaluate on one.

    Reports the F0.5 at the transferred threshold next to the best F0.5
    achievable on that country ("oracle" threshold tuned on its own scores).
    A large gap means the threshold does not transfer to an unseen country,
    which is the risk for France.

    Args:
        verbose: Whether to print progress.

    Returns:
        Dict country → results.
    """
    start_time = time.time()
    print("=" * 60)
    print("LEAVE-ONE-COUNTRY-OUT MODE")
    print("=" * 60)

    data = load_split("train", verbose=verbose)
    gt = data["ground_truth"]
    built = build_pairs(data, verbose=verbose)

    by_country = {}
    for s1_id in data["s1_ids"]:
        by_country.setdefault(data["s1_map"][s1_id].get("country_norm", ""), set()).add(s1_id)

    results = {}
    for country, held_ids in sorted(by_country.items()):
        rest_ids = set(data["s1_ids"]) - held_ids
        if len(held_ids) < config.LOCO_MIN_S1 or not rest_ids:
            print(f"\n  Skipping '{country}' ({len(held_ids)} S1 entities)")
            continue
        print(f"\n[loco] Held-out country '{country}': fit on {len(rest_ids)} S1, "
              f"evaluate on {len(held_ids)} S1...")
        X_fit, pairs_fit = select_pairs(built, rest_ids)
        fitted = fit_matcher(X_fit, pairs_fit, built["feature_names"],
                             {k: gt[k] for k in rest_ids}, sorted(rest_ids),
                             verbose=False)
        X_held, pairs_held = select_pairs(built, held_ids)
        held_gt = {k: gt[k] for k in held_ids}
        predictions, scores = predict_matches(fitted, X_held, pairs_held, sorted(held_ids))
        f05, p, r = evaluate.macro_f05_with_details(predictions, held_gt)
        oracle_t, oracle_f05 = predict.tune_threshold(
            scores, pairs_held, held_gt, sorted(held_ids), verbose=False
        )
        results[country] = {
            "n_s1": len(held_ids), "f05": f05, "precision": p, "recall": r,
            "threshold": fitted["threshold"], "oracle_threshold": oracle_t,
            "oracle_f05": oracle_f05,
        }

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  {'held out':>10s}  {'S1':>6s}  {'F0.5':>7s}  {'P':>7s}  {'R':>7s}  "
          f"{'threshold':>18s}  {'oracle F0.5':>11s}  {'oracle thr':>18s}")
    for country, res in results.items():
        print(f"  {country or '(none)':>10s}  {res['n_s1']:>6d}  {res['f05']:>7.4f}  "
              f"{res['precision']:>7.4f}  {res['recall']:>7.4f}  "
              f"{predict.format_threshold(res['threshold']):>18s}  "
              f"{res['oracle_f05']:>11.4f}  "
              f"{predict.format_threshold(res['oracle_threshold']):>18s}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print("=" * 60)

    save_run_info("loco", {"results": results, "elapsed_s": round(elapsed, 1)})
    return results


def run_validator() -> None:
    """
    Run the organizers' utils/validate_submission.py on the written outputs.

    Skipped (with a message) if the validator is not present, e.g. when only
    the code folder was copied to Kaggle.
    """
    if not os.path.exists(config.VALIDATOR_SCRIPT):
        print(f"\n  ⚠ Validator not found at {config.VALIDATOR_SCRIPT}; run it manually.")
        return
    print("\nRunning submission validator...")
    result = subprocess.run([
        sys.executable, config.VALIDATOR_SCRIPT,
        "--matching", config.MATCHING_RESULTS,
        "--candidate", config.CANDIDATE_PAIRS,
        "--test-dir", config.TEST_DIR,
    ], capture_output=True, text=True)
    print("  " + (result.stdout + result.stderr).strip().replace("\n", "\n  "))
    if result.returncode != 0:
        print("  ✗ Validator FAILED — do not upload these files.")


def run_test(verbose: bool = True):
    """
    Run the test pipeline: fit on all train data → predict on test set →
    write output files → validate them.

    Args:
        verbose: Whether to print progress.
    """
    start_time = time.time()
    print("=" * 60)
    print("TEST MODE")
    print("=" * 60)

    train_data = load_split("train", verbose=verbose)
    gt = train_data["ground_truth"]

    print("\n[pairs] Blocking + features for all train S1 entities...")
    train_built = build_pairs(train_data, verbose=verbose)
    train_recall = blocking.evaluate_blocking_recall(
        train_built["candidates"], gt, sources=train_built["sources"], verbose=verbose
    )
    fitted = fit_matcher(train_built["X"], train_built["pairs"],
                         train_built["feature_names"], gt, train_data["s1_ids"],
                         verbose=verbose)
    train.save_model(fitted["model"])

    print("\n[test] Loading test data...")
    test_data = load_split("test", verbose=verbose)

    print("\n[test] Blocking + scoring...")
    test_built = build_pairs(test_data, verbose=verbose)
    if test_built["feature_names"] != train_built["feature_names"]:
        raise RuntimeError("Train and test feature columns differ — check config.")
    test_predictions, _ = predict_matches(
        fitted, test_built["X"], test_built["pairs"], test_data["s1_ids"]
    )

    # Write outputs; candidate_pairs = exactly the pairs the model scored
    print("\n[test] Writing output files...")
    io_utils.write_matching_results(test_predictions, config.MATCHING_RESULTS)
    io_utils.write_candidate_pairs(
        {s1_id: test_built["candidates"].get(s1_id, []) for s1_id in test_data["s1_ids"]},
        config.CANDIDATE_PAIRS,
    )

    test_s1_ids = test_data["s1_ids"]
    n_matched = sum(1 for v in test_predictions.values() if v)
    n_links = sum(len(v) for v in test_predictions.values())
    matched_by_country = {}
    for s1_id, matched in test_predictions.items():
        country = test_data["s1_map"][s1_id].get("country_norm", "")
        n, m = matched_by_country.get(country, (0, 0))
        matched_by_country[country] = (n + 1, m + bool(matched))

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  Test S1 entities:      {len(test_s1_ids)}")
    print(f"  With ≥1 match:         {n_matched} ({100 * n_matched / max(len(test_s1_ids), 1):.1f}%)")
    for country, (n, m) in sorted(matched_by_country.items()):
        print(f"    {country or '(none)':>12s}: {m}/{n} S1 matched ({100 * m / n:.1f}%)")
    print(f"  Total matched links:   {n_links}")
    print(f"  Threshold (from OOF):  {predict.format_threshold(fitted['threshold'])}")
    print(f"  OOF F0.5 (train):      {fitted['oof_f05']:.4f}")
    print(f"  Train blocking recall: {train_recall:.4f}")
    print(f"  Output written to      {config.OUTPUT_DIR}")
    print(f"  Time elapsed:          {elapsed:.1f}s")
    print("=" * 60)

    save_run_info("test", {
        "oof_f05": fitted["oof_f05"],
        "threshold": fitted["threshold"],
        "num_boost_round": fitted["num_boost_round"],
        "train_blocking_recall": train_recall,
        "test_s1": len(test_s1_ids),
        "test_s1_with_match": n_matched,
        "test_links": n_links,
        "test_matched_by_country": {c: {"s1": n, "matched": m}
                                    for c, (n, m) in matched_by_country.items()},
        "n_features": len(train_built["feature_names"]),
        "elapsed_s": round(elapsed, 1),
    })

    run_validator()


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["validate", "loco", "test"],
        required=True,
        help="'validate' = local F0.5 on a train holdout; 'loco' = leave one "
             "country out; 'test' = predict on test set",
    )
    parser.add_argument("--data-dir", help="Folder with train/ and test/ (default: <repo>/dataset)")
    parser.add_argument("--output-dir", help="Where to write the TSVs (default: <repo>/output)")
    parser.add_argument("--model-dir", help="Where to save the model, run info and "
                        "error dump (default: code/business_entity_resolution/models)")
    parser.add_argument("--embeddings", action="store_true",
                        help="Turn on config.USE_EMBEDDINGS for this run")
    args = parser.parse_args()

    config.set_paths(args.data_dir, args.output_dir, args.model_dir)
    if args.embeddings:
        config.USE_EMBEDDINGS = True
    np.random.seed(config.RANDOM_SEED)

    if args.mode == "validate":
        run_validate()
    elif args.mode == "loco":
        run_loco()
    elif args.mode == "test":
        run_test()


if __name__ == "__main__":
    main()

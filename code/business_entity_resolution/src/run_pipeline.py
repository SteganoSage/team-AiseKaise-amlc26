"""
End-to-end pipeline for Business Entity Resolution.

Usage (from the repo root):
    python code/business_entity_resolution/src/run_pipeline.py --mode validate --sample-s1 200000
    python code/business_entity_resolution/src/run_pipeline.py --mode loco --sample-s1 200000
    python code/business_entity_resolution/src/run_pipeline.py --mode test

    # Kaggle: dataset is read-only under /kaggle/input, outputs go to /kaggle/working
    python code/business_entity_resolution/src/run_pipeline.py --mode test \
        --data-dir /kaggle/input/<dataset>/dataset \
        --output-dir /kaggle/working/output --model-dir /kaggle/working/models

Pipeline (the real data is ~12M records per split, so every stage is built
for scale — compact string columns, sparse/parallel blocking, index arrays):

    load + normalize (cached)
      → stage 1 blocking      broad, high-recall candidates (blocking.py)
      → stage 2 pruning       light LightGBM keeps the best few per S1 (prune.py)
                              = candidate_pairs.tsv (what the final model scores)
      → stage 3 matching      full features + LightGBM + tuned threshold
      → matching_results.tsv  + organizers' validator

Modes:
- validate: Hold out 20% of (optionally sampled) train S1 entities. Fit on
            the rest (group K-fold OOF scores pick the threshold and boosting
            rounds) and report candidate-set size, recall and macro F0.5 on
            the untouched holdout, per country, plus an error dump.
- loco:     Leave one country out: fit on the other countries, evaluate on
            it. Our stand-in for the unseen test country (France).
- test:     Fit on all train data, predict the test set, write output/*.tsv
            and run utils/validate_submission.py.

Blocking and features run once over all S1 records of a split (rank features
compare each pair with its competitors); modes then select pairs by S1.
Labels are never used before fitting, and the pruner is out-of-fold on
train, so this is not leakage — at test time all test S1s are also scored
together.
"""

import argparse
import gc
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# Ensure the src directory is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import io_utils
import normalize
import blocking
import crossenc
import prune
import features
import train
import predict
import evaluate


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

def _cache_path(path: str) -> str:
    """
    Parquet cache location for the normalized version of a source file.

    The key changes whenever the raw file or normalize.py changes.

    Args:
        path: Raw TSV path.

    Returns:
        Cache file path under <model_dir>/cache.
    """
    stat = os.stat(path)
    with open(normalize.__file__, "rb") as f:
        code = f.read()
    key = hashlib.sha1(
        f"{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime_ns}".encode() + code
    ).hexdigest()[:12]
    return os.path.join(config.MODEL_DIR, "cache",
                        f"{os.path.basename(path)}.{key}.parquet")


def load_source(path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Read and normalize one source file (cached as parquet).

    Args:
        path: Raw TSV path.
        verbose: Whether to print progress.

    Returns:
        Normalized frame with compact string columns.
    """
    cache = _cache_path(path)
    if config.USE_NORMALIZE_CACHE and os.path.exists(cache):
        df = pd.read_parquet(cache, dtype_backend="pyarrow")
        df = df.astype({c: io_utils.STRING_DTYPE for c in df.columns})
        if verbose:
            print(f"  {os.path.basename(path)}: {len(df):,} records (normalized, from cache)")
        return df

    t0 = time.time()
    df = normalize.normalize_frame(io_utils.read_source_tsv(path), n_jobs=config.N_JOBS)
    if verbose:
        print(f"  {os.path.basename(path)}: {len(df):,} records "
              f"(normalized in {time.time() - t0:.0f}s)")
    if config.USE_NORMALIZE_CACHE:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def load_split(split: str, sample_s1: int = None, verbose: bool = True) -> dict:
    """
    Load, normalize and (optionally) embed one split.

    Args:
        split: "train" or "test".
        sample_s1: Use only this many randomly chosen S1 records (all S2/S3
            records are kept, so blocking difficulty stays realistic).
        verbose: Whether to print progress.

    Returns:
        Dict with:
        - s1: S1 frame (row = S1 index), tgt: S2 + S3 frame (row = target index)
        - emb: embeddings or None
        - train only: true_keys (sorted pair keys of true links),
          n_true_by_row (true links per S1 row), links (s1_row, tgt_row) arrays
    """
    if split == "train":
        paths = (config.TRAIN_SOURCE1, config.TRAIN_SOURCE2, config.TRAIN_SOURCE3)
    else:
        paths = (config.TEST_SOURCE1, config.TEST_SOURCE2, config.TEST_SOURCE3)
    print(f"\nLoading {split} split...")
    s1 = load_source(paths[0], verbose)
    s2 = load_source(paths[1], verbose)
    s3 = load_source(paths[2], verbose)
    tgt = pd.concat([s2.assign(source="S2"), s3.assign(source="S3")], ignore_index=True)
    del s2, s3
    gc.collect()
    if config.SNAP_TO_REFERENCE:
        # Uses all S1 records (before sampling) as the vocabulary
        tgt = normalize.snap_to_reference(tgt, s1, verbose)
    if config.FEATURE_GROUPS.get("frequency", True):
        # Whole split, before sampling: same counts in training and at test time
        features.add_frequency_columns(s1, tgt, verbose)

    if sample_s1 and sample_s1 < len(s1):
        rows = np.sort(np.random.default_rng(config.RANDOM_SEED)
                       .choice(len(s1), size=sample_s1, replace=False))
        s1 = s1.iloc[rows].reset_index(drop=True)
        print(f"  Sampled {len(s1):,} S1 records (all {len(tgt):,} S2/S3 records kept)")

    if verbose:
        for name, frame in (("S1", s1), ("S2+S3", tgt)):
            counts = frame["country"].value_counts().head(6).to_dict()
            print(f"  {name} countries: {counts}")

    data = {"s1": s1, "tgt": tgt, "emb": None}
    if split == "train":
        links = io_utils.ground_truth_links(io_utils.read_ground_truth(config.TRAIN_GROUND_TRUTH))
        s1_row = pd.Index(s1["entity_id"]).get_indexer(links["s1_id"])
        tgt_row = pd.Index(tgt["entity_id"]).get_indexer(links["cand_id"])
        ours = s1_row >= 0
        s1_row, tgt_row = s1_row[ours], tgt_row[ours]
        data["links"] = (s1_row, tgt_row)
        data["n_true_by_row"] = np.bincount(s1_row, minlength=len(s1))
        found = tgt_row >= 0
        data["true_keys"] = np.unique(train.pair_keys(s1_row[found], tgt_row[found], len(tgt)))
        singletons = (data["n_true_by_row"] == 0).mean()
        print(f"  Ground truth: {len(s1_row):,} links for {len(s1):,} S1 "
              f"({singletons:.1%} singletons); {int((~found).sum())} linked IDs missing "
              f"from S2/S3 files")

    if config.USE_EMBEDDINGS:
        import embeddings
        data["emb"] = {"s1": embeddings.embed_frame(s1, verbose),
                       "tgt": embeddings.embed_frame(tgt, verbose)}
    return data


# ──────────────────────────────────────────────────────────────────────
# Stages
# ──────────────────────────────────────────────────────────────────────

def build_candidates(data: dict, pruner=None, fit_final_pruner: bool = False,
                     verbose: bool = True) -> dict:
    """
    Stage 1 (blocking) + stage 2 (pruning) for one split.

    Args:
        data: Output of load_split.
        pruner: Trained pruner to apply (test). If None, labels are required
            and the pruner is fitted out-of-fold (train).
        fit_final_pruner: Also fit a pruner on all train pairs (to apply to test).
        verbose: Whether to print progress.

    Returns:
        Dict with pairs (pruned; with pruner_prob), labels (train) and
        pruner, plus stage-1/2 size and recall numbers.
    """
    s1, tgt = data["s1"], data["tgt"]
    labelled = "true_keys" in data
    t0 = time.time()
    block_emb = None
    if data["emb"] is not None:
        block_emb = {"s1": data["emb"]["s1"]["full"], "tgt": data["emb"]["tgt"]["full"]}

    print("\n[stage 1] Blocking...")
    pairs = blocking.generate_candidates(s1, tgt, embeddings=block_emb, verbose=verbose)
    out = {"stage1": blocking.print_candidate_stats(pairs, len(s1), len(tgt), "Stage 1 (blocking)")}
    labels = None
    if labelled:
        labels = train.make_labels(pairs["s1_idx"].to_numpy(), pairs["tgt_idx"].to_numpy(),
                                   data["true_keys"], len(tgt))
        out["stage1"]["recall"] = blocking.report_recall(
            pairs, labels, int(data["n_true_by_row"].sum()), "Stage 1 (blocking)")
    print(f"  stage 1 took {time.time() - t0:.0f}s")

    print("\n[stage 2] Pruning...")
    t0 = time.time()
    if pruner is None:
        Xp, pruner_names = prune.pruner_features(pairs, s1, tgt, verbose)
        groups = pairs["s1_idx"].to_numpy()
        prob = prune.oof_pruner_scores(Xp, labels, groups, pruner_names, verbose)
        if fit_final_pruner:
            pruner = prune.fit_pruner(Xp, labels, groups, pruner_names)
        del Xp
    else:
        # Test: the pruner features are per pair, so build and score them in
        # slices; the full matrix (~76M pairs x 13 on test) is never held at once
        prob = np.empty(len(pairs), dtype=np.float32)
        step = config.PRUNE_PREDICT_CHUNK
        for start in range(0, len(pairs), step):
            Xp, _ = prune.pruner_features(pairs.iloc[start:start + step], s1, tgt, verbose=False)
            prob[start:start + step] = pruner.predict(Xp)
            del Xp
        if verbose:
            print(f"  Pruner features + scores: {len(pairs):,} pairs, "
                  f"{-(-len(pairs) // step)} slices of {step:,}")
    gc.collect()

    keep = prune.select(pairs, prob)
    pairs = pairs.iloc[keep].reset_index(drop=True)
    pairs["pruner_prob"] = prob[keep]
    out["stage2"] = blocking.print_candidate_stats(
        pairs, len(s1), len(tgt), "Stage 2 (pruned) = candidate_pairs.tsv")
    if labelled:
        labels = labels[keep]
        out["stage2"]["recall"] = blocking.report_recall(
            pairs, labels, int(data["n_true_by_row"].sum()), "Stage 2 (pruned)")
    print(f"  stage 2 took {time.time() - t0:.0f}s")

    out.update(pairs=pairs, labels=labels, pruner=pruner)
    return out


def fit_matcher(X: np.ndarray, pairs: pd.DataFrame, labels: np.ndarray,
                n_true_by_row: np.ndarray, universe: np.ndarray,
                feature_names: list, verbose: bool = True) -> dict:
    """
    Stage 3: fit the final matcher on labelled candidate pairs.

    Group K-fold OOF scores → tune threshold on OOF (with the same decision
    rule used for final predictions) → train the final LightGBM on all pairs
    with the mean best iteration from the folds.

    Args:
        X: Stage-3 feature matrix of the pairs.
        pairs: Pairs (s1_idx, tgt_idx, is_s2), same order as X.
        labels: 1 for true pairs.
        n_true_by_row: True links per S1 row (counts links blocking missed).
        universe: S1 rows these pairs come from (every one counts in F0.5).
        feature_names: Feature names.
        verbose: Whether to print progress.

    Returns:
        Dict with model, threshold, num_boost_round, oof_f05.
    """
    s1_idx, tgt_idx = pairs["s1_idx"].to_numpy(), pairs["tgt_idx"].to_numpy()
    is_s2 = pairs["is_s2"].to_numpy()
    print(f"\n[stage 3] {config.CV_FOLDS}-fold group CV on {len(labels):,} pairs "
          f"({int(labels.sum()):,} positives)...")
    oof, best_iters = train.cross_validate_oof(X, labels, s1_idx, feature_names, verbose=verbose)
    threshold, oof_f05 = predict.tune_threshold(
        oof, s1_idx, tgt_idx, is_s2, labels, n_true_by_row, universe, verbose=verbose)

    num_boost_round = max(int(round(np.mean(best_iters))), 1)
    print(f"\n[stage 3] Final LightGBM on all pairs ({num_boost_round} rounds)...")
    model = train.train_model(X, labels, feature_names=feature_names,
                              num_boost_round=num_boost_round, verbose=verbose)
    return {"model": model, "threshold": threshold, "oof": oof,
            "num_boost_round": num_boost_round, "oof_f05": oof_f05}


def stage3_features(data: dict, pairs: pd.DataFrame, verbose: bool = True,
                    extra: dict = None) -> tuple:
    """
    Stage-3 feature matrix for a split's pruned candidate pairs.

    Args:
        data: Output of load_split.
        pairs: Pruned candidate pairs.
        verbose: Whether to print progress.
        extra: Optional extra feature columns aligned with pairs
            (the cross-encoder score).

    Returns:
        Tuple (X, feature_names).
    """
    t0 = time.time()
    X, names = features.build_feature_matrix(pairs, data["s1"], data["tgt"],
                                             embeddings=data["emb"], verbose=verbose,
                                             extra=extra)
    print(f"  stage-3 features took {time.time() - t0:.0f}s")
    return X, names


# ──────────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────────

def evaluate_rows(pairs: pd.DataFrame, labels: np.ndarray, keep: np.ndarray,
                  n_true_by_row: np.ndarray, rows: np.ndarray) -> dict:
    """
    Macro F0.5 / P / R, candidate recall and all-empty baseline for some S1 rows.

    Args:
        pairs: Candidate pairs of these rows.
        labels: Labels of those pairs.
        keep: Indices (into pairs) predicted as matches.
        n_true_by_row: True links per S1 row.
        rows: S1 rows to evaluate.

    Returns:
        Dict of metrics.
    """
    f05, p, r = predict.evaluate_decision(keep, pairs["s1_idx"].to_numpy(), labels,
                                          n_true_by_row, rows)
    empty = evaluate.macro_f05_from_counts(n_true_by_row[rows], np.zeros(len(rows)),
                                           np.zeros(len(rows)))[0]
    return {"n_s1": int(len(rows)), "f05": f05, "precision": p, "recall": r,
            "all_empty_f05": empty,
            "candidate_recall": float(labels.sum() / max(n_true_by_row[rows].sum(), 1)),
            "candidates_per_s1": float(len(pairs) / max(len(rows), 1))}


def report_by_country(data: dict, pairs: pd.DataFrame, labels: np.ndarray,
                      keep: np.ndarray, rows: np.ndarray) -> dict:
    """
    Print holdout metrics per S1 country.

    Args:
        data: Output of load_split (train).
        pairs: Holdout candidate pairs.
        labels: Their labels.
        keep: Indices predicted as matches.
        rows: Holdout S1 rows.

    Returns:
        Dict country → metrics.
    """
    country = data["s1"]["country_norm"].to_numpy(dtype=object)
    pair_country = country[pairs["s1_idx"].to_numpy()]
    kept_mask = np.zeros(len(pairs), dtype=bool)
    kept_mask[keep] = True
    report = {}
    print(f"\n  {'country':>10s}  {'S1':>8s}  {'singletons':>10s}  {'F0.5':>7s}  "
          f"{'P':>7s}  {'R':>7s}  {'cands/S1':>8s}  {'cand recall':>11s}")
    for c in sorted(set(country[rows])):
        c_rows = rows[country[rows] == c]
        sel = np.flatnonzero(pair_country == c)
        m = evaluate_rows(pairs.iloc[sel], labels[sel], np.flatnonzero(kept_mask[sel]),
                          data["n_true_by_row"], c_rows)
        report[c] = m
        singles = (data["n_true_by_row"][c_rows] == 0).mean()
        print(f"  {c or '(none)':>10s}  {m['n_s1']:>8,d}  {singles:>10.1%}  {m['f05']:>7.4f}  "
              f"{m['precision']:>7.4f}  {m['recall']:>7.4f}  {m['candidates_per_s1']:>8.2f}  "
              f"{m['candidate_recall']:>11.4f}")
    return report


def write_error_dump(path: str, data: dict, pairs: pd.DataFrame, labels: np.ndarray,
                     scores: np.ndarray, keep: np.ndarray, rows: np.ndarray) -> int:
    """
    Write every wrong decision for the given S1 rows, with names and addresses.

    Error types:
    - FP: predicted but not a true match
    - FN_scored: true match that was a candidate but was not predicted
    - FN_not_candidate: true match that never made it into candidate_pairs
      (missed by blocking or dropped by pruning)

    Args:
        path: Output TSV path.
        data: Output of load_split (train).
        pairs: Candidate pairs of these rows.
        labels: Their labels.
        scores: Their match probabilities.
        keep: Indices predicted as matches.
        rows: S1 rows the dump covers.

    Returns:
        Number of error rows written.
    """
    kept = np.zeros(len(pairs), dtype=bool)
    kept[keep] = True
    fp = np.flatnonzero(kept & (labels == 0))
    fn = np.flatnonzero(~kept & (labels == 1))
    link_s1, link_tgt = data["links"]
    in_rows = np.isin(link_s1, rows)
    cand_keys = train.pair_keys(pairs["s1_idx"].to_numpy(), pairs["tgt_idx"].to_numpy(),
                                len(data["tgt"]))
    link_keys = train.pair_keys(link_s1, np.maximum(link_tgt, 0), len(data["tgt"]))
    missed = np.flatnonzero(in_rows & ~np.isin(link_keys, cand_keys))

    s1_rows = np.concatenate([pairs["s1_idx"].to_numpy()[fp], pairs["s1_idx"].to_numpy()[fn],
                              link_s1[missed]])
    tgt_rows = np.concatenate([pairs["tgt_idx"].to_numpy()[fp], pairs["tgt_idx"].to_numpy()[fn],
                               link_tgt[missed]])
    kinds = ["FP"] * len(fp) + ["FN_scored"] * len(fn) + ["FN_not_candidate"] * len(missed)
    score_col = [f"{s:.4f}" for s in scores[fp]] + [f"{s:.4f}" for s in scores[fn]] \
        + [""] * len(missed)

    s1, tgt = data["s1"], data["tgt"]
    valid_tgt = np.maximum(tgt_rows, 0)
    cols = {
        "error": kinds,
        "s1_id": prune.take_strings(s1, "entity_id", s1_rows),
        "cand_id": prune.take_strings(tgt, "entity_id", valid_tgt),
        "score": score_col,
        "s1_name": prune.take_strings(s1, "business_name", s1_rows),
        "cand_name": prune.take_strings(tgt, "business_name", valid_tgt),
        "s1_address": prune.take_strings(s1, "business_address", s1_rows),
        "cand_address": prune.take_strings(tgt, "business_address", valid_tgt),
        "s1_country": prune.take_strings(s1, "country", s1_rows),
        "cand_country": prune.take_strings(tgt, "country", valid_tgt),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(cols).to_csv(path, sep="\t", index=False, quoting=3, escapechar="\\")
    return len(kinds)


def save_run_info(mode: str, info: dict) -> None:
    """
    Save the key numbers of a run to <model_dir>/run_info_<mode>.json.

    These are the numbers to copy into submissions/LOG.md and PR descriptions.

    Args:
        mode: 'validate', 'loco' or 'test'.
        info: JSON-serializable dict of run results.
    """
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    path = os.path.join(config.MODEL_DIR, f"run_info_{mode}.json")
    info = {"mode": mode, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "use_embeddings": config.USE_EMBEDDINGS,
            "use_crossenc_config": bool(config.USE_CROSSENC),
            "snap_to_reference": config.SNAP_TO_REFERENCE,
            "top1_threshold": config.TOP1_THRESHOLD,
            "feature_groups": config.FEATURE_GROUPS,
            "blocking_top_k": config.BLOCKING_TOP_K,
            "blocking_max_df": config.BLOCKING_MAX_DF,
            "prune_top_n": config.PRUNE_TOP_N, "prune_min_prob": config.PRUNE_MIN_PROB,
            **info}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=float)
    print(f"  ✓ Run info saved to {path}")


# ──────────────────────────────────────────────────────────────────────
# Modes
# ──────────────────────────────────────────────────────────────────────

def _prune_mask(pairs: pd.DataFrame, top_n: int, min_prob: float) -> np.ndarray:
    """
    Pairs that would survive a stricter pruning setting.

    The pairs are already pruned with config.PRUNE_TOP_N / PRUNE_MIN_PROB, so
    their pruner rank within each S1 equals the rank before pruning, and any
    stricter (smaller top-N, higher min-prob) setting is a subset of them.

    Args:
        pairs: Pruned pairs with s1_idx and pruner_prob.
        top_n: Keep at most this many per S1.
        min_prob: Keep only pairs with pruner probability ≥ this.

    Returns:
        Boolean mask over pairs.
    """
    prob = pairs["pruner_prob"].to_numpy()
    rank = blocking.group_rank(pairs["s1_idx"].to_numpy(), prob)
    return (rank <= top_n) & (prob >= min_prob)


def _sweep_settings() -> list:
    """
    (top-N, min-prob) settings of the pruning sweep that are at least as
    strict as the current config (looser ones would need the dropped pairs).

    Returns:
        List of (top_n, min_prob) tuples, current setting included.
    """
    return [(n, p) for n in config.PRUNE_SWEEP_TOP_N for p in config.PRUNE_SWEEP_MIN_PROB
            if n <= config.PRUNE_TOP_N and p >= config.PRUNE_MIN_PROB]


def _size_stats(s1_idx: np.ndarray, rows: np.ndarray) -> str:
    """
    Candidates-per-S1 distribution (mean / p90 / p99 / max) over some S1 rows.

    Args:
        s1_idx: S1 row of each kept pair.
        rows: S1 rows to describe (S1 with no pair count as 0).

    Returns:
        Formatted string.
    """
    counts = np.bincount(s1_idx, minlength=int(rows.max()) + 1 if len(rows) else 0)[rows] \
        if len(rows) else np.zeros(0)
    if not len(counts):
        return "-"
    return (f"{counts.mean():5.2f} {np.percentile(counts, 90):4.0f} "
            f"{np.percentile(counts, 99):4.0f} {counts.max():4.0f}")


def report_prune_sweep(data: dict, pairs: pd.DataFrame, labels: np.ndarray,
                       dev: np.ndarray, hold: np.ndarray, fitted: dict,
                       hold_scores: np.ndarray, dev_rows: np.ndarray,
                       hold_rows: np.ndarray) -> list:
    """
    Replay stricter pruning settings on this validate run's own scores.

    For each setting: drop the pairs it would prune, re-tune the threshold on
    the dev OOF scores of the remaining pairs, apply it to the holdout scores
    and report holdout F0.5 / candidate recall / candidates per S1 overall and
    per country. Approximate: stage-3 features that depend on the candidate
    set (ranks, candidate counts) keep their values from the full set, so
    confirm the chosen setting with a real run.

    Args:
        data: Output of load_split (train).
        pairs: All pruned pairs of the run (dev + holdout).
        labels: Their labels.
        dev, hold: Indices of dev / holdout pairs.
        fitted: fit_matcher output (needs "oof", aligned with dev).
        hold_scores: Final-model scores of the holdout pairs.
        dev_rows, hold_rows: Dev / holdout S1 rows.

    Returns:
        List of result dicts, one per setting.
    """
    s1_idx, tgt_idx = pairs["s1_idx"].to_numpy(), pairs["tgt_idx"].to_numpy()
    is_s2 = pairs["is_s2"].to_numpy()
    country = data["s1"]["country_norm"].to_numpy(dtype=object)
    countries = sorted(set(country[hold_rows]))
    grid = np.round(np.arange(0.40, 0.951, 0.02), 3)
    n_true = data["n_true_by_row"]
    results = []
    print("\n[prune sweep] Stricter pruning replayed on this run (approximate; "
          "threshold re-tuned on OOF)")
    print(f"  {'top-N':>5s} {'min-p':>5s} {'thr':>5s}  {'F0.5':>6s} {'P':>6s} {'R':>6s} "
          f"{'cand rec':>8s}  cands/S1: mean  p90  p99  max  | "
          + " | ".join(f"{c} F0.5 / cands" for c in countries))
    for top_n, min_prob in _sweep_settings():
        mask = _prune_mask(pairs, top_n, min_prob)
        dm, hm = mask[dev], mask[hold]
        d, h = dev[dm], hold[hm]
        threshold, _ = predict.tune_threshold(
            fitted["oof"][dm], s1_idx[d], tgt_idx[d], is_s2[d], labels[d], n_true,
            dev_rows, thresholds=grid, verbose=False)
        keep = predict.decide(hold_scores[hm], s1_idx[h], tgt_idx[h], is_s2[h], threshold)
        m = evaluate_rows(pairs.iloc[h], labels[h], keep, n_true, hold_rows)
        per_country = []
        kept = np.zeros(len(h), dtype=bool)
        kept[keep] = True
        for c in countries:
            c_rows = hold_rows[country[hold_rows] == c]
            sel = np.flatnonzero(country[s1_idx[h]] == c)
            mc = evaluate_rows(pairs.iloc[h[sel]], labels[h[sel]], np.flatnonzero(kept[sel]),
                               n_true, c_rows)
            per_country.append(f"{mc['f05']:.4f} / {mc['candidates_per_s1']:.2f}")
        print(f"  {top_n:>5d} {min_prob:>5.2f} {predict.format_threshold(threshold):>5s}  "
              f"{m['f05']:.4f} {m['precision']:.4f} {m['recall']:.4f} {m['candidate_recall']:>8.4f}"
              f"            {_size_stats(s1_idx[h], hold_rows)}  | " + " | ".join(per_country))
        results.append({"top_n": top_n, "min_prob": min_prob, "threshold": threshold,
                        **{k: m[k] for k in ("f05", "precision", "recall",
                                             "candidate_recall", "candidates_per_s1")}})
    return results


def report_test_prune_sizes(s1: pd.DataFrame, pairs: pd.DataFrame) -> None:
    """
    Test candidates per S1 (overall and per country) under each sweep setting.

    Args:
        s1: Test S1 frame.
        pairs: Pruned test pairs with s1_idx and pruner_prob.
    """
    country = s1["country_norm"].to_numpy(dtype=object)
    rows = np.arange(len(s1))
    countries = sorted(set(country))
    print("\n[prune sweep] Test candidates per S1 under stricter pruning "
          "(mean p90 p99 max)")
    print(f"  {'top-N':>5s} {'min-p':>5s}  {'all':>20s}  "
          + "  ".join(f"{c:>20s}" for c in countries))
    s1_idx = pairs["s1_idx"].to_numpy()
    for top_n, min_prob in _sweep_settings():
        kept = s1_idx[_prune_mask(pairs, top_n, min_prob)]
        print(f"  {top_n:>5d} {min_prob:>5.2f}  {_size_stats(kept, rows):>20s}  "
              + "  ".join(f"{_size_stats(kept, rows[country == c]):>20s}" for c in countries))


def run_validate(sample_s1: int = None, verbose: bool = True) -> float:
    """
    Validation: fit on 80% of train S1 entities, score the untouched 20%.

    Args:
        sample_s1: Optional number of train S1 records to use.
        verbose: Whether to print progress.

    Returns:
        Holdout macro F0.5.
    """
    start_time = time.time()
    print("=" * 70)
    print("VALIDATION MODE")
    print("=" * 70)

    data = load_split("train", sample_s1, verbose)
    cand = build_candidates(data, verbose=verbose)
    pairs, labels = cand["pairs"], cand["labels"]

    dev_rows, hold_rows = train.holdout_split_rows(len(data["s1"]))
    extra = None
    if crossenc.enabled():
        # Part of the dev S1 fine-tunes the cross-encoder and is then left out
        # of LightGBM training; the holdout S1 are the same as without it
        ce_rows, dev_rows = crossenc.split_rows(dev_rows)
        bundle = crossenc.fit(data, pairs, labels, ce_rows, verbose)
        extra = {"crossenc_prob": crossenc.score(bundle, data, pairs, verbose)}
        del bundle
    X, names = stage3_features(data, pairs, verbose, extra)

    s1_of_pair = pairs["s1_idx"].to_numpy()
    dev = np.flatnonzero(np.isin(s1_of_pair, dev_rows))
    hold = np.flatnonzero(np.isin(s1_of_pair, hold_rows))
    print(f"\n  Dev S1: {len(dev_rows):,} ({len(dev):,} pairs), "
          f"Holdout S1: {len(hold_rows):,} ({len(hold):,} pairs)")

    fitted = fit_matcher(X[dev], pairs.iloc[dev], labels[dev], data["n_true_by_row"],
                         dev_rows, names, verbose)

    print("\n[holdout] Scoring held-out S1 entities...")
    hold_pairs, hold_labels = pairs.iloc[hold].reset_index(drop=True), labels[hold]
    scores = predict.predict_scores(fitted["model"], X[hold])
    keep = predict.decide(scores, hold_pairs["s1_idx"].to_numpy(),
                          hold_pairs["tgt_idx"].to_numpy(), hold_pairs["is_s2"].to_numpy(),
                          fitted["threshold"])
    metrics = evaluate_rows(hold_pairs, hold_labels, keep, data["n_true_by_row"], hold_rows)
    by_country = report_by_country(data, hold_pairs, hold_labels, keep, hold_rows)
    sweep = None
    if config.PRUNE_SWEEP:
        sweep = report_prune_sweep(data, pairs, labels, dev, hold, fitted, scores,
                                   dev_rows, hold_rows)

    if config.WRITE_ERROR_DUMP:
        path = os.path.join(config.MODEL_DIR, "holdout_errors.tsv")
        n = write_error_dump(path, data, hold_pairs, hold_labels, scores, keep, hold_rows)
        print(f"\n  ✓ {n:,} holdout errors written to {path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  Holdout macro F0.5:      {metrics['f05']:.4f}   "
          f"(all-empty baseline {metrics['all_empty_f05']:.4f})")
    print(f"  Holdout precision:       {metrics['precision']:.4f}")
    print(f"  Holdout recall:          {metrics['recall']:.4f}")
    print(f"  Candidates per S1:       {metrics['candidates_per_s1']:.2f}   "
          f"(stage 1: {cand['stage1']['mean_per_s1']:.2f})")
    print(f"  Candidate recall:        {metrics['candidate_recall']:.4f}   "
          f"(stage 1: {cand['stage1']['recall']:.4f})")
    print(f"  OOF F0.5 (dev):          {fitted['oof_f05']:.4f}")
    print(f"  Threshold:               {predict.format_threshold(fitted['threshold'])}")
    print(f"  Time elapsed:            {elapsed:.0f}s")
    print("=" * 70)

    save_run_info("validate", {
        "sample_s1": sample_s1, "holdout": metrics, "holdout_by_country": by_country,
        "crossenc_used": extra is not None,
        "stage1": cand["stage1"], "stage2": cand["stage2"],
        "oof_f05": fitted["oof_f05"], "threshold": fitted["threshold"],
        "num_boost_round": fitted["num_boost_round"], "n_features": len(names),
        "prune_sweep": sweep, "elapsed_s": round(elapsed, 1),
    })
    return metrics["f05"]


def run_loco(sample_s1: int = None, verbose: bool = True) -> dict:
    """
    Leave-one-country-out: fit on all other train countries, evaluate on one.

    Reports the F0.5 at the transferred threshold next to the best F0.5
    achievable on that country ("oracle" threshold tuned on its own scores).
    A large gap means the threshold does not transfer to an unseen country,
    which is the risk for France.

    Args:
        sample_s1: Optional number of train S1 records to use.
        verbose: Whether to print progress.

    Returns:
        Dict country → results.
    """
    start_time = time.time()
    print("=" * 70)
    print("LEAVE-ONE-COUNTRY-OUT MODE")
    print("=" * 70)

    data = load_split("train", sample_s1, verbose)
    cand = build_candidates(data, verbose=verbose)
    pairs, labels = cand["pairs"], cand["labels"]
    X, names = stage3_features(data, pairs, verbose)
    country = data["s1"]["country_norm"].to_numpy(dtype=object)
    pair_country = country[pairs["s1_idx"].to_numpy()]

    results = {}
    for c in sorted(set(country)):
        held_rows = np.flatnonzero(country == c)
        rest_rows = np.flatnonzero(country != c)
        if len(held_rows) < config.LOCO_MIN_S1 or len(rest_rows) == 0:
            print(f"\n  Skipping '{c}' ({len(held_rows)} S1 entities)")
            continue
        print(f"\n[loco] Held-out country '{c}': fit on {len(rest_rows):,} S1, "
              f"evaluate on {len(held_rows):,} S1...")
        fit_idx, held_idx = np.flatnonzero(pair_country != c), np.flatnonzero(pair_country == c)
        fitted = fit_matcher(X[fit_idx], pairs.iloc[fit_idx], labels[fit_idx],
                             data["n_true_by_row"], rest_rows, names, verbose=False)
        held = pairs.iloc[held_idx].reset_index(drop=True)
        args = (held["s1_idx"].to_numpy(), held["tgt_idx"].to_numpy(), held["is_s2"].to_numpy())
        scores = predict.predict_scores(fitted["model"], X[held_idx])
        keep = predict.decide(scores, *args, fitted["threshold"])
        m = evaluate_rows(held, labels[held_idx], keep, data["n_true_by_row"], held_rows)
        oracle_t, oracle_f05 = predict.tune_threshold(
            scores, *args, labels[held_idx], data["n_true_by_row"], held_rows, verbose=False)
        results[c] = {**m, "threshold": fitted["threshold"],
                      "oracle_threshold": oracle_t, "oracle_f05": oracle_f05}

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  {'held out':>10s}  {'S1':>8s}  {'F0.5':>7s}  {'P':>7s}  {'R':>7s}  "
          f"{'threshold':>18s}  {'oracle F0.5':>11s}  {'oracle thr':>18s}")
    for c, res in results.items():
        print(f"  {c or '(none)':>10s}  {res['n_s1']:>8,d}  {res['f05']:>7.4f}  "
              f"{res['precision']:>7.4f}  {res['recall']:>7.4f}  "
              f"{predict.format_threshold(res['threshold']):>18s}  {res['oracle_f05']:>11.4f}  "
              f"{predict.format_threshold(res['oracle_threshold']):>18s}")
    print(f"  Time elapsed: {elapsed:.0f}s")
    print("=" * 70)

    save_run_info("loco", {"sample_s1": sample_s1, "results": results,
                           "elapsed_s": round(elapsed, 1)})
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


def _train_state_path() -> str:
    """Where the training half of --mode test leaves what the test half needs."""
    return os.path.join(config.MODEL_DIR, "test_train_state.pkl")


def _crossenc_path() -> str:
    """Where --mode test keeps the fine-tuned cross-encoder for the test half."""
    return os.path.join(config.MODEL_DIR, "crossenc")


def _exec_predict() -> None:
    """
    Replace this process with a fresh one running the test half (--mode predict).

    Memory the training half freed is often not handed back to the operating
    system (Python / Arrow allocators keep it), and on Kaggle the test split
    was then loaded on top of it and stalled out of memory. os.execv starts a
    new interpreter in the same process (same PID, same stdout), so the
    notebook keeps streaming the log and the test half starts with empty memory.
    """
    args = list(sys.argv[1:])
    for i, arg in enumerate(args):
        if arg == "--mode" and i + 1 < len(args):
            args[i + 1] = "predict"
        elif arg.startswith("--mode="):
            args[i] = "--mode=predict"
    cmd = [sys.executable, "-u", os.path.abspath(sys.argv[0])] + args
    print(f"\n[test] Training done; restarting as a fresh process for the test half "
          f"(frees the training memory): {' '.join(cmd[2:])}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, cmd)


def run_test(train_sample_s1: int = None, verbose: bool = True) -> None:
    """
    Fit on train data → predict the test set → write outputs → validate.

    The training half saves the models and a small state file, then (with
    config.TEST_FRESH_PROCESS) hands over to a fresh process for the test half
    (run_predict), so the test split does not share memory with training.

    Args:
        train_sample_s1: Train on this many random train S1 records (all
            S2/S3 kept) instead of all of them. Every test S1 is always scored.
        verbose: Whether to print progress.
    """
    start_time = time.time()
    print("=" * 70)
    print("TEST MODE")
    print("=" * 70)

    # ── Train ──
    data = load_split("train", train_sample_s1, verbose)
    cand = build_candidates(data, fit_final_pruner=True, verbose=verbose)
    fit_rows = np.arange(len(data["s1"]))
    bundle, extra = None, None
    if crossenc.enabled():
        # Reserved S1 fine-tune the cross-encoder and are left out of LightGBM
        ce_rows, fit_rows = crossenc.split_rows(fit_rows)
        bundle = crossenc.fit(data, cand["pairs"], cand["labels"], ce_rows, verbose)
        extra = {"crossenc_prob": crossenc.score(bundle, data, cand["pairs"], verbose)}
    X, names = stage3_features(data, cand["pairs"], verbose, extra)
    sel = np.flatnonzero(np.isin(cand["pairs"]["s1_idx"].to_numpy(), fit_rows))
    if len(sel) < len(X):
        X = X[sel]
    fitted = fit_matcher(X, cand["pairs"].iloc[sel], cand["labels"][sel], data["n_true_by_row"],
                         fit_rows, names, verbose)
    train.save_model(fitted["model"])
    train.save_model(cand["pruner"], os.path.join(config.MODEL_DIR, "pruner_model.pkl"))
    if bundle is not None:
        crossenc.save(bundle, _crossenc_path())
    state = {"names": names, "threshold": fitted["threshold"], "oof_f05": fitted["oof_f05"],
             "num_boost_round": fitted["num_boost_round"],
             "train_info": {"stage1": cand["stage1"], "stage2": cand["stage2"]},
             "train_sample_s1": train_sample_s1, "start_time": start_time,
             "crossenc": bundle is not None}
    with open(_train_state_path(), "wb") as f:
        pickle.dump(state, f)
    del data, cand, X, fitted, bundle, extra
    gc.collect()

    if config.TEST_FRESH_PROCESS:
        _exec_predict()   # does not return
    run_predict(verbose)


def run_predict(verbose: bool = True) -> None:
    """
    Test half of --mode test: score the test set with the saved models.

    Loads the LightGBM matcher, the pruner, the cross-encoder (when training
    used it) and the state saved by run_test, then predicts, writes both
    output files and runs the validator.

    Args:
        verbose: Whether to print progress.
    """
    with open(_train_state_path(), "rb") as f:
        state = pickle.load(f)
    start_time, names, train_info = state["start_time"], state["names"], state["train_info"]
    fitted = {"model": train.load_model(), "threshold": state["threshold"],
              "oof_f05": state["oof_f05"], "num_boost_round": state["num_boost_round"]}
    pruner = train.load_model(os.path.join(config.MODEL_DIR, "pruner_model.pkl"))
    train_sample_s1 = state["train_sample_s1"]
    bundle = crossenc.load(_crossenc_path()) if state.get("crossenc") else None

    # ── Test ──
    test = load_split("test", verbose=verbose)
    tcand = build_candidates(test, pruner=pruner, verbose=verbose)
    pairs = tcand["pairs"]
    if config.PRUNE_SWEEP:
        report_test_prune_sizes(test["s1"], pairs)
    test_extra = None
    if bundle is not None:
        test_extra = {"crossenc_prob": crossenc.score(bundle, test, pairs, verbose)}
        del bundle
    X_test, test_names = stage3_features(test, pairs, verbose, test_extra)
    if test_names != names:
        raise RuntimeError("Train and test feature columns differ — check config.")
    scores = predict.predict_scores(fitted["model"], X_test)
    keep = predict.decide(scores, pairs["s1_idx"].to_numpy(), pairs["tgt_idx"].to_numpy(),
                          pairs["is_s2"].to_numpy(), fitted["threshold"])

    print("\n[test] Writing output files...")
    s1, tgt = test["s1"], test["tgt"]
    s1_ids = s1["entity_id"].tolist()
    matches = io_utils.group_pairs(
        prune.take_strings(s1, "entity_id", pairs["s1_idx"].to_numpy()[keep]),
        prune.take_strings(tgt, "entity_id", pairs["tgt_idx"].to_numpy()[keep]))
    candidates = io_utils.group_pairs(
        prune.take_strings(s1, "entity_id", pairs["s1_idx"].to_numpy()),
        prune.take_strings(tgt, "entity_id", pairs["tgt_idx"].to_numpy()))
    io_utils.write_matching_results(matches, config.MATCHING_RESULTS, s1_ids)
    io_utils.write_candidate_pairs(candidates, config.CANDIDATE_PAIRS, s1_ids)

    country = s1["country_norm"].to_numpy(dtype=object)
    matched_rows = np.zeros(len(s1), dtype=bool)
    matched_rows[pairs["s1_idx"].to_numpy()[keep]] = True
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  Test S1 entities:      {len(s1):,}")
    print(f"  With ≥1 match:         {matched_rows.sum():,} ({matched_rows.mean():.1%})")
    for c in sorted(set(country)):
        sel = country == c
        print(f"    {c or '(none)':>10s}: {matched_rows[sel].mean():.1%} of {sel.sum():,} S1 matched")
    print(f"  Matched links:         {len(keep):,}")
    print(f"  Candidates per S1:     {tcand['stage2']['mean_per_s1']:.2f} "
          f"(stage 1: {tcand['stage1']['mean_per_s1']:.2f})")
    print(f"  Threshold (from OOF):  {predict.format_threshold(fitted['threshold'])}")
    print(f"  OOF F0.5 (train):      {fitted['oof_f05']:.4f}")
    print(f"  Train candidate recall: {train_info['stage2']['recall']:.4f} "
          f"(stage 1: {train_info['stage1']['recall']:.4f})")
    print(f"  Output written to      {config.OUTPUT_DIR}")
    print(f"  Time elapsed:          {elapsed:.0f}s")
    print("=" * 70)

    save_run_info("test", {
        "oof_f05": fitted["oof_f05"], "threshold": fitted["threshold"],
        "num_boost_round": fitted["num_boost_round"], "train": train_info,
        "crossenc_used": test_extra is not None,
        "test_stage1": tcand["stage1"], "test_stage2": tcand["stage2"],
        "test_s1": len(s1), "test_s1_with_match": int(matched_rows.sum()),
        "test_links": int(len(keep)), "n_features": len(names),
        "train_sample_s1": train_sample_s1,
        "elapsed_s": round(elapsed, 1),
    })
    run_validator()


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument(
        "--mode", choices=["validate", "loco", "test", "predict"], required=True,
        help="'validate' = local F0.5 on a train holdout; 'loco' = leave one "
             "country out; 'test' = train, then predict on the test set; "
             "'predict' = test half only, with the models a 'test' run saved",
    )
    parser.add_argument("--data-dir", help="Folder with train/ and test/ (default: <repo>/dataset)")
    parser.add_argument("--output-dir", help="Where to write the TSVs (default: <repo>/output)")
    parser.add_argument("--model-dir", help="Where to save models, cache, run info and "
                        "error dump (default: code/business_entity_resolution/models)")
    parser.add_argument("--sample-s1", type=int, default=config.SAMPLE_S1,
                        help="validate/loco: use this many random train S1 records "
                             "(all S2/S3 kept). Default: all")
    parser.add_argument("--train-sample-s1", type=int, default=config.TRAIN_SAMPLE_S1,
                        help="test: train on this many random train S1 records "
                             "(every test S1 is still scored). Default: all")
    parser.add_argument("--embeddings", action="store_true",
                        help="Turn on config.USE_EMBEDDINGS for this run (GPU recommended)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Do not read/write the normalized parquet cache")
    parser.add_argument("--n-jobs", type=int, help="Worker processes (default: all cores)")
    args = parser.parse_args()

    config.set_paths(args.data_dir, args.output_dir, args.model_dir)
    if args.embeddings:
        config.USE_EMBEDDINGS = True
    if args.no_cache:
        config.USE_NORMALIZE_CACHE = False
    if args.n_jobs:
        config.N_JOBS = args.n_jobs
        config.LGBM_PARAMS["n_jobs"] = args.n_jobs
        config.PRUNER_PARAMS["n_jobs"] = args.n_jobs
    np.random.seed(config.RANDOM_SEED)

    if args.mode == "validate":
        run_validate(args.sample_s1)
    elif args.mode == "loco":
        run_loco(args.sample_s1)
    elif args.mode == "test":
        run_test(args.train_sample_s1)
    elif args.mode == "predict":
        run_predict()


if __name__ == "__main__":
    main()

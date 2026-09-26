"""Stage 4/5: two-stage LightGBM, decision tuning and test prediction.

stage1   LightGBM on pair features, K folds grouped by S1 entity, trained on a
         deterministic subset of training entities (--frac). Then first-stage
         probabilities p1 are written for EVERY training pair (each scored by
         the fold model that never saw its entity) and every test pair (mean of
         the fold models); ber.stage2 turns them into context features.
stage2   LightGBM on pair features + ber.stage2 features, same folds and
         training subset. Out-of-fold probabilities go through ber.decide and are
         scored with the exact macro F0.5 (true matches missed by blocking count
         as misses) to choose the decision rule.
predict  averages the stage-2 fold models on test, applies the chosen rule and
         writes both submission files.

Run from src/:  python -m ber.model stage1
                python -m ber.stage2 --split train && python -m ber.stage2 --split test
                python -m ber.model stage2
                python -m ber.model predict
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from .config import THREADS, stage_dir, work
from .decide import decide
from .features import feature_columns
from .featurize import s1_subset
from .io import in_ce_pool, load_gt_pairs
from .metrics import blocking_report, f05_macro

PARAMS = dict(
    objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=255, num_threads=THREADS, verbose=-1, seed=7,
)
N_FOLDS = 3
MAX_ROUNDS = 4000
TRAIN_FRAC = float(os.environ.get("BER_TRAIN_FRAC", 0.6))  # share of (kept, non-cross-encoder) training S1 entities the LightGBMs are fitted on (memory: ~50M rows)
TAG = ""  # dev runs set this from --tag; selects feat_<tag>/ and models_<tag>/

# Optional trimming of the final candidate set (candidate_pairs.tsv size counts
# in the final ranking): keep, per S1 entity, only the CAND_TOP_N pairs with the
# highest first-stage probability p1 and p1 >= CAND_MIN_P1; the second-stage
# model then scores (and the decision/output uses) only those. 0 = off (all
# blocked pairs are candidates, as originally). stage2 always prints what each
# setting would cost on out-of-fold predictions (trim report).
CAND_TOP_N = int(os.environ.get("BER_CAND_TOP_N", 0))
CAND_MIN_P1 = float(os.environ.get("BER_CAND_MIN_P1", 0.0))
TRIM_REPORT = [(n, m) for n in (0, 15, 10, 8, 6) for m in (0.0, 0.01, 0.03)]


def trim_mask(df, top_n=None, min_p1=None):
    """Boolean mask of the pairs kept by candidate trimming (all True when off).

    Args:
        df: Pairs with s1_id and p1 (first-stage probability).
        top_n: Keep at most this many pairs per S1 by p1 (0 = no cap).
        min_p1: Keep only pairs with p1 >= this.

    Returns:
        numpy bool array aligned with df.
    """
    top_n = CAND_TOP_N if top_n is None else top_n
    min_p1 = CAND_MIN_P1 if min_p1 is None else min_p1
    keep = pl.col("p1") >= min_p1
    if top_n:
        keep = keep & (pl.col("p1").rank("ordinal", descending=True).over("s1_id") <= top_n)
    return df.select(keep.alias("k"))["k"].to_numpy()


def trim_report(df, oof, log):
    """Out-of-fold macro F0.5 and candidates per S1 for each trimming setting.

    For each (top-N, min p1): drop the trimmed pairs, re-pick the best decision
    rule on the remaining out-of-fold probabilities and score it (true matches
    that were trimmed count as misses).
    """
    ids = df["s1_id"].unique()
    truth = load_gt_pairs().join(ids.to_frame(), on="s1_id")
    pred_all = df.select("s1_id", "rec_id").with_columns(p=pl.Series(oof))
    rows = []
    for top_n, min_p1 in TRIM_REPORT:
        m = trim_mask(df, top_n, min_p1)
        pred = pred_all.filter(pl.Series(m))
        best = 0.0
        for t in (0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
            best = max(best, f05_macro(decide(pred, "threshold", t=t), truth, ids))
        for miss in (0.0, 0.1, 0.25):
            best = max(best, f05_macro(decide(pred, "expected_f", miss=miss), truth, ids))
        kept_true = pred.join(truth, on=["s1_id", "rec_id"]).height
        rows.append({"top_n": top_n or "all", "min_p1": min_p1, "f05": round(best, 5),
                     "cands_per_s1": round(len(pred) / max(len(ids), 1), 2),
                     "cand_recall": round(kept_true / max(len(truth), 1), 4)})
    with pl.Config(tbl_rows=30):
        log(f"trim report (out-of-fold, stage 2):\n{pl.DataFrame(rows, strict=False)}")


def feat_files(split):
    return sorted(work(stage_dir("feat", TAG), "x").parent.glob(f"{split}_*.parquet"))


def model_path(name):
    return work(stage_dir("models", TAG), name)


def to_matrix(df, cols):
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


def fold_of(ids, n=N_FOLDS):
    return (ids.hash(seed=3) % n).cast(pl.Int8)


def load_train(extra=None):
    """Training-subset rows of the train feature files, optionally + extra columns.

    Entities reserved for the cross-encoder (ber.io.in_ce_pool) are excluded.
    """
    parts = []
    for f in feat_files("train"):
        df = pl.read_parquet(f).filter(~in_ce_pool(pl.col("s1_id")))
        df = df.join(s1_subset(df["s1_id"].unique(), TRAIN_FRAC).to_frame(), on="s1_id")
        if extra is not None:
            df = df.join(extra, on=["s1_id", "rec_id"], how="left")
        parts.append(df)
    return pl.concat(parts, how="diagonal_relaxed")


def fit_kfold(df, cols, prefix, log):
    """K-fold LightGBM; saves fold models, returns out-of-fold probabilities."""
    folds = fold_of(df["s1_id"]).to_numpy()
    X, y = to_matrix(df, cols), df["label"].to_numpy()
    oof = np.zeros(len(df), dtype=np.float32)
    for k in range(N_FOLDS):
        tr, va = folds != k, folds == k
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=cols, free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        booster = lgb.train(PARAMS, dtr, MAX_ROUNDS, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = booster.predict(X[va], num_iteration=booster.best_iteration)
        booster.save_model(str(model_path(f"{prefix}_fold{k}.txt")),
                           num_iteration=booster.best_iteration)
        log(f"{prefix} fold {k}: best_iter {booster.best_iteration}  "
            f"val logloss {booster.best_score['valid_0']['binary_logloss']:.5f}")
        if k == 0:
            imp = sorted(zip(cols, booster.feature_importance("gain")), key=lambda x: -x[1])
            log("top features: " + ", ".join(f"{c}={g:.0f}" for c, g in imp[:25]))
    return oof


def boosters(prefix):
    return [lgb.Booster(model_file=str(model_path(f"{prefix}_fold{k}.txt"))) for k in range(N_FOLDS)]


def tune(oof, truth, ids, log):
    """Grid over decision rules on out-of-fold probabilities."""
    results = []
    for t in (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        f = f05_macro(decide(oof, "threshold", t=t), truth, ids)
        results.append({"method": "threshold", "t": t, "f05": f})
    for miss in (0.0, 0.1, 0.25, 0.5):
        f = f05_macro(decide(oof, "expected_f", miss=miss), truth, ids)
        results.append({"method": "expected_f", "miss": miss, "f05": f})
    res = pl.DataFrame(results, strict=False).sort("f05", descending=True)
    with pl.Config(tbl_rows=30):
        log(f"decision grid:\n{res}")
    best = res.row(0, named=True)
    return {k: v for k, v in best.items() if v is not None}


def evaluate(df, oof, log, name):
    ids = df["s1_id"].unique()
    truth = load_gt_pairs().join(ids.to_frame(), on="s1_id")
    pred = df.select("s1_id", "rec_id").with_columns(p=pl.Series(oof))
    pred.with_columns(label=df["label"]).write_parquet(model_path(f"oof_{name}.parquet"))
    best = tune(pred, truth, ids, log)
    log(f"{name} best decision: {best}")
    return best


def predict_p1(split, cols, log):
    """First-stage probabilities for every pair of a split."""
    files = feat_files(split)
    if not files:
        log(f"no {split} features; skipping p1")
        return
    bs = boosters("lgb1")
    out = []
    for f in files:
        df = pl.read_parquet(f, columns=["s1_id", "rec_id", *cols])
        X = to_matrix(df, cols)
        if split == "train":
            folds = fold_of(df["s1_id"]).to_numpy()
            p = np.empty(len(df), dtype=np.float32)
            for k, b in enumerate(bs):
                m = folds == k
                if m.any():
                    p[m] = b.predict(X[m], num_threads=THREADS)
        else:
            p = np.mean([b.predict(X, num_threads=THREADS) for b in bs], axis=0)
        out.append(df.select("s1_id", "rec_id").with_columns(p1=pl.Series(p.astype(np.float32))))
    pl.concat(out).write_parquet(model_path(f"p1_{split}.parquet"))
    log(f"p1 written for {split}: {sum(len(o) for o in out):,} pairs")


def stage1(log):
    df = load_train()
    cols = feature_columns(df)
    ids = df["s1_id"].unique()
    truth = load_gt_pairs().join(ids.to_frame(), on="s1_id")
    log(f"stage1 train pairs {len(df):,}  S1 {len(ids):,}  positives {df['label'].sum():,}  "
        f"features {len(cols)}")
    log(f"blocking on this subset: {blocking_report(df.select('s1_id', 'rec_id'), truth, ids)}")
    oof = fit_kfold(df, cols, "lgb1", log)
    evaluate(df, oof, log, "stage1")
    model_path("features1.json").write_text(json.dumps(cols))
    del df
    for split in ("train", "test"):
        predict_p1(split, cols, log)


def stage2(log):
    extra = pl.read_parquet(model_path("stage2_train.parquet"))
    df = load_train(extra)
    cols = feature_columns(df)
    log(f"stage2 train pairs {len(df):,}  features {len(cols)}")
    oof = fit_kfold(df, cols, "lgb2", log)
    trim_report(df, oof, log)
    if CAND_TOP_N or CAND_MIN_P1:
        m = trim_mask(df)
        log(f"candidate trimming ON: top {CAND_TOP_N or 'all'}, p1 >= {CAND_MIN_P1} "
            f"-> {m.mean():.1%} of pairs kept")
        df, oof = df.filter(pl.Series(m)), oof[m]
    best = evaluate(df, oof, log, "stage2")
    best["features"] = cols
    model_path("decision.json").write_text(json.dumps(best, indent=1))


def predict(log):
    cfg = json.loads(model_path("decision.json").read_text())
    cols = cfg["features"]
    extra = pl.read_parquet(model_path("stage2_test.parquet"))
    bs = boosters("lgb2")
    preds = []
    for f in feat_files("test"):
        df = pl.read_parquet(f).join(extra, on=["s1_id", "rec_id"], how="left")
        if CAND_TOP_N or CAND_MIN_P1:
            df = df.filter(pl.Series(trim_mask(df)))
        p = np.mean([b.predict(to_matrix(df, cols), num_threads=THREADS) for b in bs], axis=0)
        preds.append(df.select("s1_id", "rec_id").with_columns(p=pl.Series(p.astype(np.float32))))
        log(f"scored {f.name}: {len(df):,} pairs")
    pred = pl.concat(preds)
    pred.write_parquet(model_path("test_pred.parquet"))
    kw = {k: cfg[k] for k in ("t", "miss") if k in cfg}
    matches = decide(pred, cfg["method"], **kw)
    from .submit import write_submission
    write_submission(pred.select("s1_id", "rec_id"), matches, log)


def main():
    global TAG, TRAIN_FRAC
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage2", "predict"])
    ap.add_argument("--tag", default="", help="dev only: suffix of feat/models folders")
    ap.add_argument("--frac", type=float, default=TRAIN_FRAC)
    args = ap.parse_args()
    TAG, TRAIN_FRAC = args.tag, args.frac
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
    {"stage1": stage1, "stage2": stage2, "predict": predict}[args.stage](log)


if __name__ == "__main__":
    main()

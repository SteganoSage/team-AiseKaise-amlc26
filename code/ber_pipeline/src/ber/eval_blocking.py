"""Diagnostic: blocking recall on a sample of training S1 entities.

Queries a random fraction of training S1 entities against the FULL training
record pool (so the number of look-alike distractors is realistic) and reports,
per view and for the union, pair recall and the F0.5 ceiling at several K.

Run from src/:  python -m ber.eval_blocking --frac 0.1 --k 30
"""
import argparse
import time

import polars as pl

from .blocking import VIEWS, generate
from .config import work
from .io import load_gt_pairs
from .metrics import blocking_report


def main():
    """Command line: blocking recall report on a sample of training S1 entities."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--max-df", type=int, default=None, help="override every view's cap")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    s1 = pl.read_parquet(work("norm", "train_s1.parquet")).sample(fraction=args.frac, seed=0)
    rec = pl.read_parquet(work("norm", "train_rec.parquet"))
    gt = load_gt_pairs().join(s1.select(pl.col("entity_id").alias("s1_id")), on="s1_id")
    ids = s1["entity_id"]
    views = args.views.split(",")
    log(f"{len(s1):,} S1 queries, {len(rec):,} records, {len(gt):,} true pairs")

    caps = {v: args.max_df for v in views} if args.max_df else None
    cand = generate(s1, rec, views=views, k=args.k, max_df=caps, log=log)
    cand.write_parquet(work("dev", f"cand_frac{args.frac}_k{args.k}{args.tag}.parquet"))
    log(f"candidates: {len(cand):,}")

    rows = []
    for k in sorted({5, 10, 20, args.k}):
        if k > args.k:
            continue
        for v in views:
            sel = cand.filter(pl.col(f"{v}_rank") < k).select("s1_id", "rec_id")
            rows.append({"K": k, "view": v, **blocking_report(sel, gt, ids)})
        union = pl.any_horizontal([pl.col(f"{v}_rank") < k for v in views])
        sel = cand.filter(union).select("s1_id", "rec_id")
        rows.append({"K": k, "view": "UNION", **blocking_report(sel, gt, ids)})
    with pl.Config(tbl_rows=100, tbl_cols=20, float_precision=4):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()

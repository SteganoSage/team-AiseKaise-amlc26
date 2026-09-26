"""Stage 2: candidate table per split and country.

For every Source 1 entity: blocking (top-K per view, union), exact cosines per
view for every pair, and record-side / S1-side competition context. Training
pairs also get their label. Countries are processed one at a time (matches
never cross countries), which keeps memory bounded.

Output: WORK_DIR/cand/{split}_{country}.parquet
Run from src/:  python -m ber.candidates --split train
"""
import argparse
import time

import polars as pl

from . import translit
from .blocking import block_country
from .translit import translate_expr
from .config import stage_dir, work
from .featurize import s1_subset
from .io import load_norm_s1
from .features import COS_COLS, add_context, add_exact_cosines
from .io import load_gt_pairs

# Top-K per view (every view at the cheap 5k frequency cap). 15/10/10 gave
# 0.964 pair recall on training; the leaderboard leader scores above that
# set's F0.5 ceiling, so K is raised and a reverse search is added.
DEFAULT_K = {"words2": 20, "addr2": 12, "name4": 12, "comb4": 10}
# Reverse search: each record's top-K S1 entities with cosine >= min
# (words2: name+address; name4: name only, for records without an address or
# with compact names).
REVERSE_K = {"words2": (3, 0.0), "name4": (5, 0.3), "comb4": (3, 0.2)}


def build_country(s1c, recc, views, k, gt=None, log=print, reverse=REVERSE_K):
    cand = block_country(s1c, recc, views, k, log=log, reverse=reverse)
    cand = cand.with_columns(
        s1_id=s1c["entity_id"].gather(cand["s1_row"]),
        rec_id=recc["entity_id"].gather(cand["rec_row"]),
    ).drop("s1_row", "rec_row")
    cand = add_exact_cosines(cand, s1c, recc, log=log)
    cand = add_context(cand, COS_COLS, "rec_id")
    cand = add_context(cand, COS_COLS, "s1_id")
    if gt is not None:
        cand = cand.join(gt.with_columns(label=pl.lit(1, pl.Int8)), on=["s1_id", "rec_id"],
                         how="left").with_columns(pl.col("label").fill_null(0))
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--countries", default="")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="dev only: query a share of S1 entities against all records")
    ap.add_argument("--tag", default="", help="dev only: suffix for the output folder")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    s1 = load_norm_s1(args.split)
    if args.frac < 1:
        s1 = s1.join(s1_subset(s1["entity_id"], args.frac).to_frame(), on="entity_id")
    rec = pl.read_parquet(work("norm", f"{args.split}_rec.parquet"))
    # Transliterated record names ("skti bildrs") are mapped back to Latin
    # words before blocking so they can be found by name, not only address.
    s1 = s1.with_columns(n_core_tr=pl.col("n_core"))
    rec = rec.with_columns(n_core_tr=translate_expr("n_core", translit.table_for(args.split, log)))
    gt = load_gt_pairs() if args.split == "train" else None
    countries = args.countries.split(",") if args.countries else sorted(s1["country"].unique().to_list())
    for country in countries:
        s1c = s1.filter(pl.col("country") == country)
        recc = rec.filter(pl.col("country") == country)
        log(f"{args.split} {country}: {len(s1c):,} S1 x {len(recc):,} records")
        # A sampled S1 set would pull every record into a reverse search, so
        # dev runs (--frac < 1) skip it.
        cand = build_country(s1c, recc, list(DEFAULT_K), DEFAULT_K, gt, log,
                             reverse=REVERSE_K if args.frac >= 1 else None)
        cand.write_parquet(work(stage_dir("cand", args.tag), f"{args.split}_{country}.parquet"))
        log(f"{args.split} {country}: {len(cand):,} candidate pairs "
            f"({len(cand) / max(len(s1c), 1):.1f} per S1)")


if __name__ == "__main__":
    main()

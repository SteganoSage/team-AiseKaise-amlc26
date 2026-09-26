"""Stage 3: pair features for candidate tables.

Every S1 entity is featurised (training needs first-stage probabilities for
all of them, see ber.model); --frac can restrict it for quick experiments.
Work is chunked by S1 entity so that S1-side context stays exact within a chunk.

Output: WORK_DIR/feat/{split}_{country}_{chunk:03d}.parquet
Run from src/:  python -m ber.featurize --split train
"""
import argparse
import time

import polars as pl

from . import translit
from .config import stage_dir, work
from .features import add_record_extras, add_s1_extras, pair_features, word_idf
from .io import load_norm_s1

CHUNK_S1 = 150_000


STATE_BLANK_FRAC = 0.15


def blank_state(feats, frac=STATE_BLANK_FRAC):
    """Training augmentation: hide state agreement for a share of S1 entities.

    Unseen countries (France in test) have no state table, so their pairs
    always show "state unknown on both sides", a pattern that otherwise never
    occurs in US/India training data. Blanking it for some training entities
    teaches the model to fall back on the other evidence.
    """
    mask = (pl.col("s1_id").hash(seed=29) % 1000) < int(frac * 1000)
    return feats.with_columns(
        state_eq=pl.when(mask).then(False).otherwise(pl.col("state_eq")),
        state_conflict=pl.when(mask).then(False).otherwise(pl.col("state_conflict")),
    )


def s1_subset(ids, frac, seed=17):
    """Deterministic hash-based subset of S1 ids (same on every run/machine)."""
    if frac >= 1:
        return ids
    h = ids.hash(seed=seed) % 1_000_000
    return ids.filter(h < int(frac * 1_000_000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--frac", type=float, default=1.0, help="share of S1 entities (train)")
    ap.add_argument("--tag", default="", help="dev only: suffix of cand/feat folders")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    s1 = add_s1_extras(load_norm_s1(args.split))
    rec = pl.read_parquet(work("norm", f"{args.split}_rec.parquet"))
    rec = add_record_extras(rec, s1, translit.table_for(args.split, log))
    idf = word_idf(s1)
    log("record extras added")
    feat_dir = stage_dir("feat", args.tag)
    for old in work(feat_dir, "x").parent.glob(f"{args.split}_*.parquet"):
        old.unlink()
    for path in sorted(work(stage_dir("cand", args.tag), "x").parent.glob(f"{args.split}_*.parquet")):
        country = path.stem.split("_", 1)[1]
        cand = pl.read_parquet(path)
        ids = s1_subset(cand["s1_id"].unique().sort(), args.frac)
        for i in range(0, len(ids), CHUNK_S1):
            chunk_ids = ids.slice(i, CHUNK_S1).to_frame("s1_id")
            part = cand.join(chunk_ids, on="s1_id")
            s1c = s1.join(chunk_ids.rename({"s1_id": "entity_id"}), on="entity_id")
            recc = rec.join(part.select(pl.col("rec_id").unique().alias("entity_id")), on="entity_id")
            feats = pair_features(part, s1c, recc, idf=idf)
            if args.split == "train":
                feats = blank_state(feats)
            feats.write_parquet(work(feat_dir, f"{args.split}_{country}_{i // CHUNK_S1:03d}.parquet"))
            log(f"{args.split} {country} chunk {i // CHUNK_S1}: {len(feats):,} pairs")


if __name__ == "__main__":
    main()

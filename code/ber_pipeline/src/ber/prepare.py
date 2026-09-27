"""Stage 1: normalise every record of both splits and cache it as parquet.

Outputs (under WORK_DIR/norm):
  {split}_s1.parquet   Source 1 records with normalised name/address columns
  {split}_rec.parquet  Source 2+3 records, same columns plus `source`
  state_aliases.json   transliterated state names learned from training matches

Run from src/:  python -m ber.prepare
"""
import json
import os
import time
from multiprocessing import Pool

import polars as pl

from .config import work
from .io import load_gt_pairs, load_records, load_source
from .normalize import INDIC_RE, STATE_MAPS, norm_addr, norm_name, state_key

CHUNK = 100_000
KEEP = ["entity_id", "country"]


def _norm_chunk(args):
    """Worker: normalise a chunk of (name, address, country) triples into a DataFrame."""
    names, addrs, countries, aliases = args
    out = {}
    for n, a, c in zip(names, addrs, countries):
        r = norm_name(n)
        r.update(norm_addr(a, c, aliases.get(c)))
        for k, v in r.items():
            out.setdefault(k, []).append(v)
    return pl.DataFrame(out)


def normalize_frame(df, aliases):
    """Normalise every record of a frame in parallel worker processes.

    Args:
        df: Raw records (entity_id, business_name, business_address, country[, source]).
        aliases: {country: {address component -> state code}} learned from training.

    Returns:
        DataFrame with the id columns plus the normalised name/address columns.
    """
    tasks = [
        (part["business_name"].to_list(), part["business_address"].to_list(),
         part["country"].to_list(), aliases)
        for part in df.iter_slices(CHUNK)
    ]
    # A fresh pool per frame: polars crashed (segfault) when heavy frame
    # operations ran in the parent while worker processes were alive.
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        normed = pl.concat(pool.map(_norm_chunk, tasks))
    extra = ["source"] if "source" in df.columns else []
    return pl.concat([df.select(KEEP + extra), normed], how="horizontal").rechunk()


def learn_state_aliases(s1_norm, recs, gt, min_count=20, min_purity=0.9):
    """Map transliterated Indic address components to the matched S1 state."""
    comps = (
        recs.select("entity_id", "country",
                    pl.col("business_address").str.split(",").alias("comp"))
        .explode("comp")
        .filter(pl.col("comp").str.contains(INDIC_RE.pattern))
    )
    keys = {c: state_key(c) for c in comps["comp"].unique().to_list()}
    comps = comps.with_columns(pl.col("comp").replace_strict(keys).alias("key"))
    counts = (
        comps.join(gt, left_on="entity_id", right_on="rec_id")
        .join(s1_norm.select(pl.col("entity_id").alias("s1_id"), "a_state"), on="s1_id")
        .filter(pl.col("a_state") != "")
        .group_by("country", "key", "a_state").len()
    )
    best = (
        counts.with_columns(
            total=pl.col("len").sum().over("country", "key"))
        .sort("len", descending=True)
        .group_by("country", "key").first()
        .filter((pl.col("total") >= min_count)
                & (pl.col("len") / pl.col("total") >= min_purity))
    )
    aliases = {}
    for r in best.iter_rows(named=True):
        if r["key"] not in STATE_MAPS.get(r["country"], {}):
            aliases.setdefault(r["country"], {})[r["key"]] = r["a_state"]
    return aliases


def main():
    """Command line: normalise both splits and learn the state aliases."""
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    s1_path = work("norm", "train_s1.parquet")
    if not s1_path.exists():
        normalize_frame(load_source("train", 1), {}).write_parquet(s1_path)
    log("train S1 normalised")

    # Aliases come from a sample: only a few dozen distinct Indic components
    # exist, and the full-size exploded frame crashed polars on Windows.
    recs = load_records("train")
    aliases = learn_state_aliases(
        pl.read_parquet(s1_path), recs.sample(min(3_000_000, len(recs)), seed=0), load_gt_pairs())
    work("norm", "state_aliases.json").write_text(
        json.dumps(aliases, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"state aliases: { {c: len(a) for c, a in aliases.items()} }")

    normalize_frame(recs, aliases).write_parquet(work("norm", "train_rec.parquet"))
    log(f"train records normalised: {len(recs):,}")
    del recs

    for name, loader in (("test_s1", lambda: load_source("test", 1)),
                         ("test_rec", lambda: load_records("test"))):
        df = loader()
        normalize_frame(df, aliases).write_parquet(work("norm", f"{name}.parquet"))
        log(f"{name} normalised: {len(df):,}")
        del df


if __name__ == "__main__":
    main()

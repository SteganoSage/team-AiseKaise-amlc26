"""Second-stage inputs: first-stage probabilities in context.

For every candidate pair (S1 entity s, record r) with first-stage probability p1:

  record competition   best p1 of r with any OTHER S1 entity, margin over it,
                       rank of s among r's candidate entities, confident
                       entities of r. A record belongs to at most one entity.
  entity context       rank of r among s's candidates, gap to s's best, number
                       and total p1 of s's other confident records.
  cluster support      similarity of r to s's confident records (p1 >= 0.8,
                       r itself excluded): best name token-set and plain ratio,
                       best address token-set, exact-name agreement. A record
                       without an address whose name equals that of a
                       confidently matched sibling record is likely a match.

Needs WORK_DIR/models/p1_{split}.parquet from `ber.model stage1`.
Output: WORK_DIR/models/stage2_{split}.parquet
Run from src/:  python -m ber.stage2 --split train
"""
import argparse
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from . import translit
from .config import stage_dir, work
from .translit import translate_expr

CONF = 0.8
CHUNK_S1 = 200_000


def context(p):
    rec_top = p.group_by("rec_id").agg(
        p1_max_rec=pl.col("p1").max(),
        p1_2nd_rec=pl.col("p1").sort(descending=True).head(2).last(),
        n_rec=pl.len(),
        n_conf_rec=(pl.col("p1") >= 0.5).sum(),
    )
    s1_top = p.group_by("s1_id").agg(
        p1_max_s1=pl.col("p1").max(),
        p1_sum_s1=pl.col("p1").sum(),
        n_conf_s1=(pl.col("p1") >= 0.5).sum(),
    )
    is_best = pl.col("p1") >= pl.col("p1_max_rec")
    other = pl.when(is_best).then(
        pl.when(pl.col("n_rec") > 1).then(pl.col("p1_2nd_rec")).otherwise(0.0)
    ).otherwise(pl.col("p1_max_rec"))
    return (
        p.join(rec_top, on="rec_id").join(s1_top, on="s1_id")
         .with_columns(p1_other_rec=other.cast(pl.Float32))
         .with_columns(
             p1_margin_rec=(pl.col("p1") - pl.col("p1_other_rec")).cast(pl.Float32),
             p1_rk_rec=pl.col("p1").rank("ordinal", descending=True).over("rec_id").cast(pl.Int16),
             p1_rk_s1=pl.col("p1").rank("ordinal", descending=True).over("s1_id").cast(pl.Int16),
             p1_gap_s1=(pl.col("p1_max_s1") - pl.col("p1")).cast(pl.Float32),
             n_conf_s1_other=(pl.col("n_conf_s1") - (pl.col("p1") >= 0.5)).cast(pl.Int16),
             p1_sum_s1_other=(pl.col("p1_sum_s1") - pl.col("p1")).cast(pl.Float32),
             n_conf_rec=pl.col("n_conf_rec").cast(pl.Int16),
             n_rec=pl.col("n_rec").cast(pl.Int16),
         )
         .select("s1_id", "rec_id", "p1", "p1_other_rec", "p1_margin_rec", "p1_rk_rec",
                 "p1_rk_s1", "p1_gap_s1", "n_conf_s1_other", "p1_sum_s1_other",
                 "n_conf_rec", "n_rec")
    )


def cluster(p, names, log=print):
    """Best similarity of each candidate record to the entity's confident records."""
    conf = p.filter(pl.col("p1") >= CONF).select("s1_id", pl.col("rec_id").alias("c_id"))
    r_names = names.rename({"name": "r_name", "addr": "r_addr"})
    c_names = names.rename({"rec_id": "c_id", "name": "c_name", "addr": "c_addr"})
    ids = p["s1_id"].unique()
    out = []
    for i in range(0, len(ids), CHUNK_S1):
        chunk = ids.slice(i, CHUNK_S1).to_frame()
        j = (p.join(chunk, on="s1_id").select("s1_id", "rec_id")
              .join(conf.join(chunk, on="s1_id"), on="s1_id")
              .filter(pl.col("rec_id") != pl.col("c_id"))
              .join(r_names, on="rec_id").join(c_names, on="c_id"))
        if len(j) == 0:
            continue
        rn, cn = j["r_name"].to_list(), j["c_name"].to_list()
        ra, ca = j["r_addr"].to_list(), j["c_addr"].to_list()
        j = j.select("s1_id", "rec_id", "r_name", "c_name").with_columns(
            n_tset=cpdist(rn, cn, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32),
            n_ratio=cpdist(rn, cn, scorer=fuzz.ratio, workers=-1, dtype=np.float32),
            a_tset=cpdist(ra, ca, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32),
        )
        out.append(j.group_by("s1_id", "rec_id").agg(
            clu_name_tset=pl.col("n_tset").max(),
            clu_name_ratio=pl.col("n_ratio").max(),
            clu_addr_tset=pl.col("a_tset").max(),
            clu_name_exact=((pl.col("r_name") == pl.col("c_name")) & (pl.col("r_name") != "")).any(),
        ))
        log(f"  cluster chunk {i // CHUNK_S1}: {len(j):,} record-to-sibling comparisons")
    return pl.concat(out) if out else pl.DataFrame(schema={"s1_id": pl.String, "rec_id": pl.String})


def build(split, tag="", log=print):
    models = stage_dir("models", tag)
    p = pl.read_parquet(work(models, f"p1_{split}.parquet"))
    names = (pl.read_parquet(work("norm", f"{split}_rec.parquet"),
                             columns=["entity_id", "n_core", "a_norm"])
               .join(p.select(pl.col("rec_id").unique().alias("entity_id")), on="entity_id")
               .select(pl.col("entity_id").alias("rec_id"),
                       translate_expr("n_core", translit.table_for(split, log)).alias("name"),
                       pl.col("a_norm").alias("addr")))
    ctx = context(p)
    log(f"context features: {len(ctx):,} pairs")
    clu = cluster(p, names, log)
    out = ctx.join(clu, on=["s1_id", "rec_id"], how="left")
    ce_path = work(models, f"ce_{split}.parquet")
    if ce_path.exists():
        # Cross-encoder score (uncertain pairs only) and how it compares with
        # the record's / the entity's other scored candidates.
        ce = pl.read_parquet(ce_path).with_columns(
            ce_gap_rec=(pl.col("ce").max().over("rec_id") - pl.col("ce")).cast(pl.Float32),
            ce_gap_s1=(pl.col("ce").max().over("s1_id") - pl.col("ce")).cast(pl.Float32),
            ce_n_rec=pl.len().over("rec_id").cast(pl.Int16),
        )
        out = out.join(ce, on=["s1_id", "rec_id"], how="left")
        log(f"cross-encoder scores joined: {len(ce):,} pairs")
    out.write_parquet(work(models, f"stage2_{split}.parquet"))
    log(f"wrote stage2_{split}.parquet: {len(out):,} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    build(args.split, args.tag, lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True))


if __name__ == "__main__":
    main()

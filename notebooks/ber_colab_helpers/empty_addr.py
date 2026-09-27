"""Are rejected address-less true matches recoverable? (diagnostic, on the VM)"""
import json
import os
import sys

os.environ["BER_WORK_DIR"] = "/content/ber_work"
sys.path.insert(0, "/content/ber/src")
import polars as pl

from ber.config import work
from ber.decide import decide

cfg = json.loads(work("models", "decision.json").read_text())
kw = {k: cfg[k] for k in ("t", "miss") if k in cfg}
oof = pl.read_parquet(work("models", "oof_stage2.parquet"))
pred = decide(oof.select("s1_id", "rec_id", "p"), cfg["method"], **kw).with_columns(pred=pl.lit(1))
cols = ["s1_id", "rec_id", "r_addr_empty", "r_core_s1_count", "r_skel_s1_count", "n_core_tr_ratio",
        "n_core_tr_tset", "r_compact"]
feat = pl.concat([pl.read_parquet(p, columns=cols) for p in work("feat", "x").parent.glob("train_*.parquet")])
d = (oof.filter(pl.col("label") == 1).join(feat, on=["s1_id", "rec_id"])
        .join(pred, on=["s1_id", "rec_id"], how="left").with_columns(pl.col("pred").fill_null(0)))
e = d.filter(pl.col("r_addr_empty"))
print(f"address-less true candidate pairs: {len(e):,}; matched {e['pred'].mean():.2%}")
bucket = pl.when(pl.col("r_core_s1_count") == 0).then(pl.lit("0 (name differs)")) \
           .when(pl.col("r_core_s1_count") == 1).then(pl.lit("1 (unique)")) \
           .when(pl.col("r_core_s1_count") <= 3).then(pl.lit("2-3")).otherwise(pl.lit("4+"))
print(e.with_columns(same_name_S1=bucket).group_by("same_name_S1").agg(
    pairs=pl.len(), matched=pl.col("pred").mean(), mean_p=pl.col("p").mean(),
    exact_name=(pl.col("n_core_tr_ratio") == 100).mean()).sort("same_name_S1"))
# how often is an address-less record with a unique exact name NOT a true match? (precision risk)
allc = oof.join(feat, on=["s1_id", "rec_id"]).filter(
    pl.col("r_addr_empty") & (pl.col("r_core_s1_count") == 1) & (pl.col("n_core_tr_ratio") == 100))
print(f"address-less, name exactly equal and unique among S1: {len(allc):,} pairs, "
      f"true-match rate {allc['label'].mean():.3f}, currently predicted "
      f"{allc.join(pred, on=['s1_id', 'rec_id'], how='semi').height / max(len(allc), 1):.2%}")

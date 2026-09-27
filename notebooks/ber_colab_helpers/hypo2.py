"""(C) Do full names (incl. legal form) resolve 'ambiguous' address-less records?
(B) What do blocking misses with an address look like?  (diagnostic, on the VM)"""
import json
import os
import sys

os.environ["BER_WORK_DIR"] = "/content/ber_work"
sys.path.insert(0, "/content/ber/src")
import polars as pl

from ber.config import work
from ber.decide import decide
from ber.io import load_gt_pairs, load_norm_s1

cfg = json.loads(work("models", "decision.json").read_text())
kw = {k: cfg[k] for k in ("t", "miss") if k in cfg}
oof = pl.read_parquet(work("models", "oof_stage2.parquet"))
ids = oof["s1_id"].unique()
gt = load_gt_pairs().join(ids.to_frame(), on="s1_id")
pred = decide(oof.select("s1_id", "rec_id", "p"), cfg["method"], **kw)
fn = oof.filter(pl.col("label") == 1).select("s1_id", "rec_id").join(pred, on=["s1_id", "rec_id"], how="anti")

s1 = load_norm_s1("train").select("entity_id", "country", "n_core", "n_full")
rec = pl.read_parquet(work("norm", "train_rec.parquet"),
                      columns=["entity_id", "n_core", "n_full", "a_is_empty", "source"]).rename({"entity_id": "rec_id"})
core_cnt = s1.group_by("country", "n_core").len().rename({"len": "same_core"})
full_cnt = s1.group_by("country", "n_full").len().rename({"len": "same_full"})
s1c = s1.select(pl.col("entity_id").alias("s1_id"), "country")
d = (fn.join(s1c, on="s1_id").join(rec, on="rec_id").filter(pl.col("a_is_empty"))
       .join(core_cnt, on=["country", "n_core"], how="left")
       .join(full_cnt, on=["country", "n_full"], how="left")
       .with_columns(pl.col("same_core", "same_full").fill_null(0)))
amb = d.filter(pl.col("same_core") >= 2)
print(f"rejected address-less true pairs with core name shared by 2+ S1: {len(amb):,}")
print("  record FULL name (incl. legal form) matches exactly one S1:", (amb["same_full"] == 1).sum(),
      "| matches none:", (amb["same_full"] == 0).sum(), "| matches 2+:", (amb["same_full"] >= 2).sum())
print("  distribution of same_core:", amb.group_by("same_core").len().sort("same_core").head(8).to_dicts())
print("  source:", amb.group_by("source").len().to_dicts())

# (B) blocking misses with an address
cand = oof.select("s1_id", "rec_id")
miss = gt.join(cand, on=["s1_id", "rec_id"], how="anti")
raw = lambda n: pl.read_parquet(work("raw", f"train_source{n}.parquet"),
                                columns=["entity_id", "business_name", "business_address"])
s1raw = raw(1).rename({"entity_id": "s1_id", "business_name": "s1n", "business_address": "s1a"})
rraw = pl.concat([raw(2), raw(3)]).rename({"entity_id": "rec_id", "business_name": "rn", "business_address": "ra"})
recflags = pl.read_parquet(work("norm", "train_rec.parquet"),
                           columns=["entity_id", "a_is_empty", "n_is_indic", "n_is_compact"]).rename({"entity_id": "rec_id"})
m = (miss.join(recflags, on="rec_id")
         .filter(~pl.col("a_is_empty") & ~pl.col("n_is_indic") & ~pl.col("n_is_compact"))
         .sample(30, seed=2).join(s1raw, on="s1_id").join(rraw, on="rec_id"))
print("\n#### blocking misses with an address (not Indic, not compact)")
for x in m.iter_rows(named=True):
    print(f"S1: {x['s1n']} | {x['s1a']}\n R: {x['rn']} | {x['ra']}\n")

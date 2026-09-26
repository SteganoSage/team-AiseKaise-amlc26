"""Where is F0.5 lost? Out-of-fold breakdown of the final (stage-2) model, on the VM."""
import json
import os
import sys

os.environ["BER_WORK_DIR"] = "/content/ber_work"
sys.path.insert(0, "/content/ber/src")
import polars as pl

from ber.config import work
from ber.decide import decide
from ber.io import load_gt_pairs
from ber.metrics import f05_macro

cfg = json.loads(work("models", "decision.json").read_text())
kw = {k: cfg[k] for k in ("t", "miss") if k in cfg}
oof = pl.read_parquet(work("models", "oof_stage2.parquet"))
ids = oof["s1_id"].unique()
gt_all = load_gt_pairs()
gt = gt_all.join(ids.to_frame(), on="s1_id")
pred = decide(oof.select("s1_id", "rec_id", "p"), cfg["method"], **kw)
tp = pred.join(gt, on=["s1_id", "rec_id"], how="semi")
fp = pred.join(gt, on=["s1_id", "rec_id"], how="anti")
cand_true = oof.filter(pl.col("label") == 1).select("s1_id", "rec_id")
fn_cand = cand_true.join(pred, on=["s1_id", "rec_id"], how="anti")
fn_block = gt.join(oof.select("s1_id", "rec_id"), on=["s1_id", "rec_id"], how="anti")
f = lambda p: f05_macro(p, gt, ids)
base = f(pred)
print(f"current {base:.4f} | no FP {f(tp):.4f} | +cand FN {f(pl.concat([pred, fn_cand])):.4f} | "
      f"ceiling {f(cand_true):.4f}")
print(f"pairs: TP {len(tp):,}  FP {len(fp):,}  FN in candidates {len(fn_cand):,}  FN blocking {len(fn_block):,}")

s1_all = pl.read_parquet(work("norm", "train_s1.parquet"), columns=["entity_id", "country"])
country = s1_all.rename({"entity_id": "s1_id"})
for c in ("India", "US"):
    ci = country.filter(pl.col("country") == c).join(ids.to_frame(), on="s1_id")["s1_id"]
    sub = lambda df: df.join(ci.to_frame(), on="s1_id")
    print(f"{c}: F0.5 {f05_macro(sub(pred), sub(gt), ci):.4f}")

# FP owner: another S1 in the fitted set (sibling), an S1 removed to simulate
# test (orphan), or no S1 at all (original decoy)
kept = pl.read_parquet(work("feat", "x").parent.glob("train_*.parquet").__next__(), columns=["s1_id"]).head(0)
kept_ids = pl.concat([pl.read_parquet(p, columns=["s1_id"]).unique()
                      for p in work("feat", "x").parent.glob("train_*.parquet")])["s1_id"].unique()
owner = gt_all.rename({"s1_id": "true_s1"})
fo = fp.join(owner, on="rec_id", how="left").with_columns(
    kind=pl.when(pl.col("true_s1").is_null()).then(pl.lit("decoy"))
          .when(pl.col("true_s1").is_in(kept_ids.implode())).then(pl.lit("sibling"))
          .otherwise(pl.lit("orphan (entity removed)")))
print("FP by owner:", fo.group_by("kind").len().sort("kind").to_dicts())
print("S1 with >=1 FP:", fp["s1_id"].n_unique(), " of which singletons:",
      fp.join(gt.select("s1_id").unique(), on="s1_id", how="anti")["s1_id"].n_unique())

rec = pl.read_parquet(work("norm", "train_rec.parquet"),
                      columns=["entity_id", "a_is_empty", "n_is_indic", "n_is_compact"]).rename({"entity_id": "rec_id"})
def profile(name, df):
    d = df.join(rec, on="rec_id", how="left")
    print(f"{name:14s} n={len(d):>9,}  empty_addr {d['a_is_empty'].mean():.2f}  "
          f"indic_name {d['n_is_indic'].mean():.2f}  compact {d['n_is_compact'].mean():.2f}")
profile("all true pairs", gt)
profile("FN candidates", fn_cand)
profile("FN blocking", fn_block)
profile("FP", fp)
# how many S1 entities lose points, and how
e = (gt.group_by("s1_id").len().rename({"len": "n_true"})
       .join(tp.group_by("s1_id").len().rename({"len": "tp"}), on="s1_id", how="left")
       .join(pred.group_by("s1_id").len().rename({"len": "n_pred"}), on="s1_id", how="left").fill_null(0))
print("entities with true matches:", len(e), " perfect:", (e["tp"] == e["n_true"]).sum() - 0,
      " missing some:", (e["tp"] < e["n_true"]).sum(), " with an FP:", (e["n_pred"] > e["tp"]).sum())

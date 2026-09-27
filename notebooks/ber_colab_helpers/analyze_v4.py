"""v4 error analysis on the VM: F0.5 lost per error type, examples, test by country."""
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

pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(120); pl.Config.set_tbl_rows(40)
cfg = json.loads(work("models", "decision.json").read_text())
kw = {k: cfg[k] for k in ("t", "miss") if k in cfg}
oof = pl.read_parquet(work("models", "oof_stage2.parquet"))
ids = oof["s1_id"].unique()
gt_all = load_gt_pairs()
gt = gt_all.join(ids.to_frame(), on="s1_id")
pred = decide(oof.select("s1_id", "rec_id", "p"), cfg["method"], **kw)
f = lambda p: f05_macro(p, gt, ids)
base = f(pred)
tp = pred.join(gt, on=["s1_id", "rec_id"], how="semi")
fp = pred.join(gt, on=["s1_id", "rec_id"], how="anti")
cand_true = oof.filter(pl.col("label") == 1).select("s1_id", "rec_id", "p")
fn_cand = cand_true.join(pred, on=["s1_id", "rec_id"], how="anti")
fn_block = gt.join(oof.select("s1_id", "rec_id"), on=["s1_id", "rec_id"], how="anti")
print(f"v4 OOF {base:.4f} | no FP {f(tp):.4f} | +all cand FN {f(pl.concat([pred, fn_cand.select('s1_id','rec_id')])):.4f} "
      f"| +blocking FN too {f(pl.concat([pred, fn_cand.select('s1_id','rec_id'), fn_block])):.4f}")

cols = ["s1_id", "rec_id", "r_addr_empty", "r_core_s1_count", "r_name_indic", "r_compact", "n_core_tr_ratio"]
feat = pl.concat([pl.read_parquet(p, columns=cols) for p in work("feat", "x").parent.glob("train_*.parquet")])
rec = pl.read_parquet(work("norm", "train_rec.parquet"),
                      columns=["entity_id", "a_is_empty", "n_is_indic", "n_is_compact"]).rename({"entity_id": "rec_id"})
kind = (pl.when(pl.col("r_addr_empty") & (pl.col("r_core_s1_count") >= 2)).then(pl.lit("addr-less, name shared by 2+ S1"))
          .when(pl.col("r_addr_empty")).then(pl.lit("addr-less, other"))
          .when(pl.col("r_name_indic")).then(pl.lit("indic name"))
          .when(pl.col("r_compact")).then(pl.lit("compact/domain name"))
          .otherwise(pl.lit("other (has address)")))
fnc = fn_cand.join(feat, on=["s1_id", "rec_id"], how="left").with_columns(kind=kind)
print("\nrejected true candidates: F0.5 gained if each group were fixed")
rows = []
for k in fnc["kind"].unique().sort().to_list():
    add = fnc.filter(pl.col("kind") == k).select("s1_id", "rec_id")
    rows.append({"group": k, "pairs": len(add), "f05_gain": round(f(pl.concat([pred, add])) - base, 5)})
print(pl.DataFrame(rows).sort("f05_gain", descending=True))
fnb = fn_block.join(rec, on="rec_id", how="left").with_columns(
    kind=pl.when(pl.col("a_is_empty")).then(pl.lit("addr-less")).when(pl.col("n_is_indic")).then(pl.lit("indic name"))
           .when(pl.col("n_is_compact")).then(pl.lit("compact")).otherwise(pl.lit("other")))
rows = []
for k in fnb["kind"].unique().sort().to_list():
    add = fnb.filter(pl.col("kind") == k).select("s1_id", "rec_id")
    rows.append({"group": k, "pairs": len(add), "f05_gain": round(f(pl.concat([pred, add])) - base, 5)})
print("\nblocking misses: F0.5 gained if each group were retrieved and matched")
print(pl.DataFrame(rows).sort("f05_gain", descending=True))
owner = gt_all.rename({"s1_id": "true_s1"})
fo = fp.join(owner, on="rec_id", how="left").join(feat, on=["s1_id", "rec_id"], how="left")
print(f"\nFP {len(fp):,}: F0.5 gain if all removed {f(tp) - base:.5f}; address-less share "
      f"{fo['r_addr_empty'].mean():.2f}; owned by another kept S1 {fo['true_s1'].is_not_null().mean():.2f}")

raw = lambda n: pl.read_parquet(work("raw", f"train_source{n}.parquet"),
                                columns=["entity_id", "business_name", "business_address"])
s1raw = raw(1).rename({"entity_id": "s1_id", "business_name": "s1_name", "business_address": "s1_addr"})
rraw = pl.concat([raw(2), raw(3)]).rename({"entity_id": "rec_id", "business_name": "r_name", "business_address": "r_addr"})
def show(title, df, n):
    print(f"\n#### {title}")
    d = df.sample(min(n, len(df)), seed=3).join(s1raw, on="s1_id").join(rraw, on="rec_id")
    for r in d.iter_rows(named=True):
        print(f"S1: {r['s1_name']} | {r['s1_addr']}\n R: {r['r_name']} | {r['r_addr']}   p={r.get('p', 0) or 0:.2f}\n")
show("rejected true candidates WITH an address", fnc.filter(~pl.col("r_addr_empty").fill_null(False)), 25)
show("false matches", fp.join(oof.select("s1_id", "rec_id", "p"), on=["s1_id", "rec_id"]), 20)

# test predictions by country
tp_ = pl.read_parquet(work("models", "test_pred.parquet"))
c = pl.read_parquet(work("raw", "test_source1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
t = tp_.join(c, on="s1_id").group_by("country").agg(
    pairs=pl.len(), uncertain=pl.col("p").is_between(0.1, 0.9).mean(),
    confident=(pl.col("p") > 0.9).mean(), per_s1_conf=(pl.col("p") > 0.5).sum() / pl.col("s1_id").n_unique())
print("\ntest predictions by country\n", t.sort("country"))
o = oof.join(country := pl.read_parquet(work("raw", "train_source1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}), on="s1_id")
print("train OOF by country\n", o.group_by("country").agg(uncertain=pl.col("p").is_between(0.1, 0.9).mean(),
      per_s1_conf=(pl.col("p") > 0.5).sum() / pl.col("s1_id").n_unique()).sort("country"))

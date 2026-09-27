"""H1: are invented brand names tied to one entity (shared across its records)?
H2: what do uncertain French test pairs look like?  (diagnostic, on the VM)"""
import os
import sys

os.environ["BER_WORK_DIR"] = "/content/ber_work"
sys.path.insert(0, "/content/ber/src")
import polars as pl

from ber.config import work
from ber.io import load_gt_pairs

gt = load_gt_pairs()
rec = pl.read_parquet(work("norm", "train_rec.parquet"),
                      columns=["entity_id", "n_core", "n_alt", "n_has_dba", "country"])
s1_words = (pl.read_parquet(work("norm", "train_s1.parquet"), columns=["n_core"])
              .select(pl.col("n_core").str.split(" ").alias("w")).explode("w").unique())
# invented token: single-token core name never seen in any S1 name
inv = (rec.filter(pl.col("n_core").str.split(" ").list.len() == 1)
          .join(s1_words.rename({"w": "n_core"}), on="n_core", how="anti")
          .select("entity_id", pl.col("n_core").alias("brand")))
alt = (rec.filter(pl.col("n_alt") != "")
          .select("entity_id", pl.col("n_alt").str.split(" ").list.first().alias("brand")))
brands = pl.concat([inv.with_columns(kind=pl.lit("standalone")), alt.with_columns(kind=pl.lit("dba_prefix"))])
b = brands.join(gt.rename({"rec_id": "entity_id"}), on="entity_id", how="left")
print("brand records:", b.group_by("kind").agg(n=pl.len(), matched=pl.col("s1_id").is_not_null().mean()))
m = b.filter(pl.col("s1_id").is_not_null())
per_brand = m.group_by("brand").agg(n_rec=pl.len(), n_ent=pl.col("s1_id").n_unique())
print("brands used by >1 record:", (per_brand["n_rec"] > 1).sum(), "of", len(per_brand),
      "| of those, share tied to a single entity:",
      round(per_brand.filter(pl.col("n_rec") > 1)["n_ent"].eq(1).mean(), 3))
per_ent = m.group_by("s1_id").agg(n_rec=pl.len(), n_brand=pl.col("brand").n_unique())
print("entities with >=2 brand records:", (per_ent["n_rec"] >= 2).sum(),
      "| share where all their brand records use ONE brand:",
      round(per_ent.filter(pl.col("n_rec") >= 2)["n_brand"].eq(1).mean(), 3))
print("brand collisions across entities (brand used by >1 entity):", (per_brand["n_ent"] > 1).sum())
dec = b.filter(pl.col("s1_id").is_null()).select("brand").unique()
print("decoy brands also used by matched records:", dec.join(per_brand.select("brand"), on="brand").height,
      "of", len(dec))

# H2: uncertain French test pairs
tp = pl.read_parquet(work("models", "test_pred.parquet")).filter(pl.col("p").is_between(0.2, 0.8))
c = pl.read_parquet(work("raw", "test_source1.parquet"), columns=["entity_id", "country", "business_name", "business_address"])
r = pl.concat([pl.read_parquet(work("raw", f"test_source{n}.parquet"),
                               columns=["entity_id", "business_name", "business_address"]) for n in (2, 3)])
fr = (tp.join(c.rename({"entity_id": "s1_id"}), on="s1_id").filter(pl.col("country") == "France")
        .join(r.rename({"entity_id": "rec_id", "business_name": "rn", "business_address": "ra"}), on="rec_id")
        .sample(30, seed=1))
print("\n#### uncertain French test pairs (0.2 <= p <= 0.8)")
for x in fr.iter_rows(named=True):
    print(f"S1: {x['business_name']} | {x['business_address']}\n R: {x['rn']} | {x['ra']}   p={x['p']:.2f}\n")

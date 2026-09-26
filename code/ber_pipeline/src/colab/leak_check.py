"""Does row order or ID numbering leak matches? (diagnostic only, on the VM)"""
import os
import sys

os.environ["BER_WORK_DIR"] = "/content/ber_work"
sys.path.insert(0, "/content/ber/src")
import numpy as np
import polars as pl

from ber.config import work

gt = pl.read_parquet(work("raw", "train_gt_pairs.parquet"))
s1 = pl.read_parquet(work("raw", "train_source1.parquet"), columns=["entity_id"]).with_row_index("s1_row")
s2 = pl.read_parquet(work("raw", "train_source2.parquet"), columns=["entity_id"]).with_row_index("row")
j = (gt.join(s1.rename({"entity_id": "s1_id"}), on="s1_id")
       .join(s2.rename({"entity_id": "rec_id"}), on="rec_id"))
a, b = j["s1_row"].to_numpy().astype(float), j["row"].to_numpy().astype(float)
print("rank corr S1 row vs S2 row:", round(float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1]), 4))
spread = j.group_by("s1_id").agg(span=pl.col("row").max() - pl.col("row").min(), n=pl.len()).filter(pl.col("n") >= 2)
print("S2 records of one entity: median row span", spread["span"].median(), "of", len(s2), "rows;",
      "share within 100 rows:", round((spread["span"] < 100).mean(), 4))
num = lambda c: pl.col(c).str.extract(r"(\d+)$").cast(pl.Int64)
ids = j.select(num("s1_id").alias("a"), num("rec_id").alias("b"))
print("rank corr S1 id number vs S2 id number:",
      round(float(np.corrcoef(ids["a"].rank().to_numpy(), ids["b"].rank().to_numpy())[0, 1]), 4))

"""Decision layer: from pair probabilities to each S1 entity's match list.

1. One-owner assignment: a record matches at most one S1 entity (true in all
   7.6M training matches), so each record is kept only for its most probable
   S1 entity.
2. Per-S1 selection, either
   - threshold: keep pairs with p >= t, or
   - expected F0.5: sort by p, keep the top-k (k may be 0) maximising
       E[F] ~= 1.25 * sum_{i<=k} p_i / (0.25 * (sum_all p + miss) + k)
     against E[F | empty] = prod(1 - p_i). `miss` is the expected number of
     true matches blocking did not retrieve.
"""
import polars as pl


def assign_one_owner(pred):
    """Keep, for every record, only its highest-probability S1 entity."""
    return (pred.sort(["rec_id", "p", "s1_id"], descending=[False, True, False])
                .unique(subset="rec_id", keep="first", maintain_order=True))


def select_threshold(pred, t):
    return pred.filter(pl.col("p") >= t).select("s1_id", "rec_id")


def select_expected_f(pred, miss=0.0, p_floor=0.02):
    df = pred.filter(pl.col("p") >= p_floor).sort(["s1_id", "p"], descending=[False, True])
    df = df.with_columns(
        k=pl.col("p").cum_count().over("s1_id"),
        cs=pl.col("p").cum_sum().over("s1_id"),
        total=pl.col("p").sum().over("s1_id"),
        ef0=(1 - pl.col("p")).clip(1e-6, 1).log().sum().over("s1_id").exp(),
    ).with_columns(ef=1.25 * pl.col("cs") / (0.25 * (pl.col("total") + miss) + pl.col("k")))
    df = df.with_columns(best=pl.col("ef").max().over("s1_id"))
    kbest = pl.col("k").filter(pl.col("ef") == pl.col("best")).first().over("s1_id")
    return (df.filter((pl.col("best") > pl.col("ef0")) & (pl.col("k") <= kbest))
              .select("s1_id", "rec_id"))


def decide(pred, method="threshold", **kw):
    owned = assign_one_owner(pred)
    if method == "threshold":
        return select_threshold(owned, kw.get("t", 0.5))
    return select_expected_f(owned, kw.get("miss", 0.0), kw.get("p_floor", 0.02))

"""Loading the challenge TSVs, cached as parquet under WORK_DIR/raw."""
import polars as pl

from .config import DATA_DIR, work

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def _read_tsv(path):
    # quote_char=None: names contain stray quotes; every column is read as text.
    return pl.read_csv(
        path, separator="\t", quote_char=None, infer_schema=False,
        missing_utf8_is_empty_string=True,
    )


def load_source(split, n):
    """Records of one source file: entity_id, business_name, business_address, country."""
    cache = work("raw", f"{split}_source{n}.parquet")
    if not cache.exists():
        df = _read_tsv(DATA_DIR / split / f"{split}_source{n}.tsv")
        assert df.columns == SOURCE_COLS, df.columns
        df.write_parquet(cache)
    return pl.read_parquet(cache)


def load_records(split):
    """Source 2 and 3 records stacked, with a `source` column (2 or 3)."""
    return pl.concat([
        load_source(split, n).with_columns(pl.lit(n, pl.Int8).alias("source"))
        for n in (2, 3)
    ])


# Test has ~5.8 records per Source 1 entity against 4.7 in training, while the
# number of matches per entity looks the same: test contains a larger share of
# records whose entity is absent from Source 1 (~40% vs 26%). Dropping 19% of
# training S1 entities (keeping all their records) reproduces that, so the
# model, the competition features and the decision threshold are fitted under
# test-like conditions: 4.68 / (1 - 0.19) = 5.78 records per entity.
TRAIN_S1_DROP = 0.19


def drop_train_s1(s1, frac=TRAIN_S1_DROP, seed=41):
    """Deterministically remove `frac` of training S1 entities."""
    if frac <= 0:
        return s1
    return s1.filter((pl.col("entity_id").hash(seed=seed) % 1_000_000) >= int(frac * 1_000_000))


def load_norm_s1(split):
    """Normalised Source 1 of a split; training has TRAIN_S1_DROP entities removed."""
    s1 = pl.read_parquet(work("norm", f"{split}_s1.parquet"))
    return drop_train_s1(s1) if split == "train" else s1


# Share of kept training entities reserved to fit the cross-encoder (ber.crossenc);
# the LightGBM models never train on them, so cross-encoder scores are
# out-of-sample wherever LightGBM uses them.
CE_POOL = 0.2


def in_ce_pool(s1_ids):
    """Boolean expression/Series: entity belongs to the cross-encoder pool."""
    return (s1_ids.hash(seed=53) % 1_000_000) < int(CE_POOL * 1_000_000)


def load_gt_pairs():
    """Training ground truth as one row per (s1_id, rec_id) match."""
    cache = work("raw", "train_gt_pairs.parquet")
    if not cache.exists():
        gt = _read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
        pairs = (
            gt.filter(pl.col("matched_entity_ids") != "")
            .select(
                pl.col("source1_entity_id").alias("s1_id"),
                pl.col("matched_entity_ids").str.split(",").alias("rec_id"),
            )
            .explode("rec_id")
        )
        pairs.write_parquet(cache)
    return pl.read_parquet(cache)

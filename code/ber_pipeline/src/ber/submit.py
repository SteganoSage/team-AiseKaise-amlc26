"""Writes the two submission files (one row per test S1 entity, tab-separated).

  OUTPUT_DIR/matching_results.tsv  source1_entity_id, matched_entity_ids
  OUTPUT_DIR/candidate_pairs.tsv   source1_entity_id, candidate_entity_ids

OUTPUT_DIR defaults to WORK_DIR/output (override with BER_OUTPUT_DIR).
"""
import os
from pathlib import Path

import polars as pl

from .config import WORK_DIR
from .io import load_source

OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", WORK_DIR / "output"))


def _lists(pairs, all_s1, col):
    grouped = (pairs.unique(subset=["s1_id", "rec_id"])
                    .sort(["s1_id", "rec_id"])
                    .group_by("s1_id", maintain_order=True)
                    .agg(pl.col("rec_id").str.join(",").alias(col)))
    return (all_s1.join(grouped, on="s1_id", how="left")
                  .with_columns(pl.col(col).fill_null(""))
                  .rename({"s1_id": "source1_entity_id"}))


def write_submission(candidates, matches, log=print):
    all_s1 = load_source("test", 1).select(pl.col("entity_id").alias("s1_id"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Final matches must be a subset of the candidates fed to the model.
    matches = matches.join(candidates, on=["s1_id", "rec_id"], how="semi")
    for name, pairs, col in (("candidate_pairs.tsv", candidates, "candidate_entity_ids"),
                             ("matching_results.tsv", matches, "matched_entity_ids")):
        out = _lists(pairs, all_s1, col)
        out.write_csv(OUTPUT_DIR / name, separator="\t", quote_style="never")
        non_empty = (out[col] != "").sum()
        log(f"wrote {OUTPUT_DIR / name}: {len(out):,} rows, {non_empty:,} non-empty")

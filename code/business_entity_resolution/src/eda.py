#!/usr/bin/env python3
"""
Quick EDA script for the entity resolution dataset.

Prints dataset statistics that guide the pipeline configuration:
- File sizes and record counts per source
- Country strings in every file (critical: "India" vs "IN" breaks country blocking)
- S1 match distribution: % singletons, matches per S1 (0/1/2+), S2 vs S3 split
- Postal code format examples per country
- Empty name/address rates
- Name/address length distributions
- Sample records per country

Run from the repo root:
    python code/business_entity_resolution/src/eda.py
    python code/business_entity_resolution/src/eda.py --data-dir /path/to/dataset
"""

import argparse
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import io_utils
import normalize


def analyze_source(path: str, name: str) -> pd.DataFrame:
    """Load and print basic stats for one source file."""
    df = io_utils.read_source_tsv(path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"\n{'─' * 60}")
    print(f"  {name}: {len(df):,} records, {size_mb:.2f} MB")
    print(f"{'─' * 60}")

    # Country strings (exact values — crucial for blocking)
    print(f"\n  Country values:")
    for country, count in df["country"].value_counts().items():
        empty_name = (df[df["country"] == country]["business_name"] == "").sum()
        empty_addr = (df[df["country"] == country]["business_address"] == "").sum()
        print(f"    '{country}': {count:,} records "
              f"(empty name: {empty_name}, empty addr: {empty_addr})")
    empty_country = (df["country"] == "").sum()
    if empty_country:
        print(f"    (empty country): {empty_country}")

    # Name/address lengths
    name_lens = df["business_name"].str.len()
    addr_lens = df["business_address"].str.len()
    print(f"\n  Name length:    mean {name_lens.mean():.0f}, "
          f"median {name_lens.median():.0f}, max {name_lens.max()}")
    print(f"  Address length: mean {addr_lens.mean():.0f}, "
          f"median {addr_lens.median():.0f}, max {addr_lens.max()}")

    # ID prefix distribution
    prefixes = df["entity_id"].str[:2].value_counts()
    print(f"\n  ID prefixes: {dict(prefixes)}")

    # Postal code examples
    print(f"\n  Postal code examples (by country):")
    for country in sorted(df["country"].unique()):
        if not country:
            continue
        subset = df[df["country"] == country]["business_address"].head(20)
        codes = []
        for addr in subset:
            norm = normalize.normalize_text(addr)
            found = normalize.extract_postal_codes(norm)
            codes.extend(found)
        if codes:
            print(f"    {country}: {codes[:5]}")
        else:
            print(f"    {country}: (none found in first 20 records)")

    # Sample records
    print(f"\n  Sample records:")
    for _, row in df.head(3).iterrows():
        print(f"    {row['entity_id']}: {row['business_name'][:50]}")
        print(f"      addr: {row['business_address'][:60]}")
        print(f"      country: {row['country']}")

    return df


def analyze_ground_truth(path: str, s1_ids: list):
    """Analyze ground truth match distribution."""
    df = io_utils.read_ground_truth(path)
    print(f"\n{'═' * 60}")
    print(f"  GROUND TRUTH ANALYSIS")
    print(f"{'═' * 60}")

    match_counts = []
    s2_count = 0
    s3_count = 0
    for _, row in df.iterrows():
        matched = io_utils.parse_id_list(row.get("matched_entity_ids", ""))
        match_counts.append(len(matched))
        for m in matched:
            if m.startswith("S2-"):
                s2_count += 1
            elif m.startswith("S3-"):
                s3_count += 1

    total = len(match_counts)
    singletons = sum(1 for c in match_counts if c == 0)
    one_match = sum(1 for c in match_counts if c == 1)
    multi_match = sum(1 for c in match_counts if c >= 2)

    print(f"\n  Total S1 entities: {total:,}")
    print(f"  Singletons (no match): {singletons:,} ({100*singletons/total:.1f}%)")
    print(f"  Exactly 1 match: {one_match:,} ({100*one_match/total:.1f}%)")
    print(f"  2+ matches: {multi_match:,} ({100*multi_match/total:.1f}%)")
    print(f"\n  Total matched links: {sum(match_counts):,}")
    print(f"    S2 links: {s2_count:,}")
    print(f"    S3 links: {s3_count:,}")

    if multi_match > 0:
        max_matches = max(match_counts)
        dist = Counter(match_counts)
        print(f"\n  Match count distribution:")
        for k in sorted(dist.keys()):
            print(f"    {k} matches: {dist[k]:,} S1 entities")

    # All-empty baseline F0.5
    empty_f05 = singletons / total
    print(f"\n  All-empty baseline F0.5 ≈ {empty_f05:.4f} "
          f"(singleton rate = fraction of S1 with no true match)")


def main():
    parser = argparse.ArgumentParser(description="EDA for entity resolution dataset")
    parser.add_argument("--data-dir", help="Folder with train/ and test/")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both",
                        help="Which split to analyze")
    args = parser.parse_args()

    config.set_paths(args.data_dir)

    if args.split in ("train", "both"):
        print("\n" + "=" * 60)
        print("  TRAIN SET")
        print("=" * 60)
        s1_df = analyze_source(config.TRAIN_SOURCE1, "train_source1")
        analyze_source(config.TRAIN_SOURCE2, "train_source2")
        analyze_source(config.TRAIN_SOURCE3, "train_source3")
        analyze_ground_truth(config.TRAIN_GROUND_TRUTH, s1_df["entity_id"].tolist())

    if args.split in ("test", "both"):
        print("\n\n" + "=" * 60)
        print("  TEST SET")
        print("=" * 60)
        analyze_source(config.TEST_SOURCE1, "test_source1")
        analyze_source(config.TEST_SOURCE2, "test_source2")
        analyze_source(config.TEST_SOURCE3, "test_source3")
        print("\n  (No ground truth for test set)")


if __name__ == "__main__":
    main()

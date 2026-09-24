#!/usr/bin/env python3
"""
Generate an all-empty submission: every test S1 entity predicts zero matches.

This is the "first submission idea" from CLAUDE.md §9 and PLAN.md §6:
public score ≈ singleton rate of the test set. Useful calibration
baseline — tells us what fraction of S1 entities have no true matches,
and whether that distribution looks like train.

Usage:
    python code/business_entity_resolution/src/make_empty_submission.py
    python code/business_entity_resolution/src/make_empty_submission.py \\
        --data-dir /kaggle/input/.../dataset \\
        --output-dir /kaggle/working/output
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import io_utils


def main():
    parser = argparse.ArgumentParser(
        description="Generate an all-empty baseline submission"
    )
    parser.add_argument("--data-dir", help="Folder with train/ and test/")
    parser.add_argument("--output-dir", help="Where to write the TSVs")
    args = parser.parse_args()

    config.set_paths(args.data_dir, args.output_dir)

    print("Generating all-empty submission...\n")

    # Read test S1 entity IDs
    df = io_utils.read_source_tsv(config.TEST_SOURCE1)
    s1_ids = df["entity_id"].tolist()
    print(f"  Test S1 entities: {len(s1_ids)}")

    # All-empty predictions: every S1 gets zero matches
    predictions = {s1_id: [] for s1_id in s1_ids}

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    io_utils.write_matching_results(predictions, config.MATCHING_RESULTS)
    io_utils.write_candidate_pairs(predictions, config.CANDIDATE_PAIRS)

    print(f"\n  All-empty submission written to {config.OUTPUT_DIR}")
    print(f"  Expected public LB ≈ singleton rate of the test set.")
    print(f"\n  Validate with:")
    print(f"    python3 utils/validate_submission.py \\")
    print(f"      --matching {config.MATCHING_RESULTS} \\")
    print(f"      --candidate {config.CANDIDATE_PAIRS} \\")
    print(f"      --test-dir {config.TEST_DIR}")


if __name__ == "__main__":
    main()

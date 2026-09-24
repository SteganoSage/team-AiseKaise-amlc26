#!/usr/bin/env python3
"""
Validate submission files before uploading to the leaderboard.

Checks:
  - Both files are present and tab-separated with correct headers.
  - Exactly one row per test S1 entity, no duplicates.
  - Only S2/S3 IDs that exist in the test set.
  - No duplicate IDs within a list.
  - Every matched ID also appears in candidate_pairs.tsv (matches ⊆ candidates).

Usage:
    python3 utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test

Exit 0 + "PASS" = safe to upload.
Exit 1 + error message = fix before uploading.
"""

import argparse
import csv
import os
import sys


def read_tsv_ids(path, id_col, list_col):
    """Read a two-column TSV and return a dict {id: set of listed IDs}."""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # Validate header
        if id_col not in reader.fieldnames:
            raise ValueError(f"Missing column '{id_col}' in {path}. Found: {reader.fieldnames}")
        if list_col not in reader.fieldnames:
            raise ValueError(f"Missing column '{list_col}' in {path}. Found: {reader.fieldnames}")
        for row in reader:
            entity_id = row[id_col].strip()
            if entity_id in result:
                raise ValueError(f"Duplicate {id_col} '{entity_id}' in {path}")
            raw = row[list_col].strip()
            if raw == "":
                result[entity_id] = set()
            else:
                ids = [x.strip() for x in raw.split(",")]
                # Check for duplicates within this row
                if len(ids) != len(set(ids)):
                    dupes = [x for x in ids if ids.count(x) > 1]
                    raise ValueError(
                        f"Duplicate IDs in row '{entity_id}': {set(dupes)}"
                    )
                result[entity_id] = set(ids)
    return result


def load_test_entity_ids(test_dir):
    """Load all entity IDs from test source files."""
    s1_ids = set()
    s2_ids = set()
    s3_ids = set()

    for fname in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(test_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Test file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                eid = row["entity_id"].strip()
                if eid.startswith("S1-"):
                    s1_ids.add(eid)
                elif eid.startswith("S2-"):
                    s2_ids.add(eid)
                elif eid.startswith("S3-"):
                    s3_ids.add(eid)

    return s1_ids, s2_ids, s3_ids


def validate(matching_path, candidate_path, test_dir):
    """Run all validation checks. Returns list of error strings (empty = PASS)."""
    errors = []

    # Load test IDs
    try:
        s1_ids, s2_ids, s3_ids = load_test_entity_ids(test_dir)
    except Exception as e:
        errors.append(f"Cannot load test data: {e}")
        return errors

    valid_target_ids = s2_ids | s3_ids

    # Load matching results
    try:
        matching = read_tsv_ids(
            matching_path, "source1_entity_id", "matched_entity_ids"
        )
    except Exception as e:
        errors.append(f"matching_results.tsv error: {e}")
        return errors

    # Load candidate pairs
    try:
        candidates = read_tsv_ids(
            candidate_path, "source1_entity_id", "candidate_entity_ids"
        )
    except Exception as e:
        errors.append(f"candidate_pairs.tsv error: {e}")
        return errors

    # Check: exactly one row per test S1 entity in matching
    missing_s1 = s1_ids - set(matching.keys())
    extra_s1 = set(matching.keys()) - s1_ids
    if missing_s1:
        errors.append(
            f"matching_results.tsv missing {len(missing_s1)} S1 entities: "
            f"{sorted(missing_s1)[:5]}..."
        )
    if extra_s1:
        errors.append(
            f"matching_results.tsv has {len(extra_s1)} unknown S1 entities: "
            f"{sorted(extra_s1)[:5]}..."
        )

    # Check: only valid S2/S3 IDs in matching
    for s1, matched in matching.items():
        for mid in matched:
            if not (mid.startswith("S2-") or mid.startswith("S3-")):
                errors.append(
                    f"Invalid ID prefix in matching for {s1}: '{mid}' "
                    f"(must be S2- or S3-)"
                )
            elif mid not in valid_target_ids:
                errors.append(
                    f"Unknown ID in matching for {s1}: '{mid}' not in test set"
                )

    # Check: matches ⊆ candidates
    for s1, matched in matching.items():
        if not matched:
            continue
        cand = candidates.get(s1, set())
        not_in_cand = matched - cand
        if not_in_cand:
            errors.append(
                f"Matched IDs for {s1} not in candidates: {not_in_cand}"
            )

    # Check: only valid S2/S3 IDs in candidates
    for s1, cands in candidates.items():
        for cid in cands:
            if not (cid.startswith("S2-") or cid.startswith("S3-")):
                errors.append(
                    f"Invalid ID prefix in candidates for {s1}: '{cid}'"
                )
            elif cid not in valid_target_ids:
                errors.append(
                    f"Unknown ID in candidates for {s1}: '{cid}' not in test set"
                )

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate submission files for Amazon ML Challenge 2026"
    )
    parser.add_argument(
        "--matching",
        required=True,
        help="Path to matching_results.tsv",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to candidate_pairs.tsv",
    )
    parser.add_argument(
        "--test-dir",
        required=True,
        help="Path to directory containing test_source{1,2,3}.tsv",
    )
    args = parser.parse_args()

    for path in [args.matching, args.candidate]:
        if not os.path.exists(path):
            print(f"FAIL: File not found: {path}", file=sys.stderr)
            sys.exit(1)

    errors = validate(args.matching, args.candidate, args.test_dir)

    if errors:
        print("FAIL — fix these errors before uploading:\n", file=sys.stderr)
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}", file=sys.stderr)
        sys.exit(1)
    else:
        print("PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()

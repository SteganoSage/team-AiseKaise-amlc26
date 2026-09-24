"""
I/O utilities for reading and writing TSV files.

All data files are TAB-separated. IDs are always strings. Empty lists
stay as "" (not NaN). This module enforces those conventions.
"""

import os
import pandas as pd


def read_source_tsv(path: str) -> pd.DataFrame:
    """
    Read a source TSV file (train or test).

    Columns: entity_id, business_name, business_address, country.
    All values are read as strings; NaN is never introduced.

    Args:
        path: Absolute or relative path to the TSV file.

    Returns:
        DataFrame with string columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If expected columns are missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    expected_cols = {"entity_id", "business_name", "business_address", "country"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}. Found: {list(df.columns)}"
        )

    return df


def read_ground_truth(path: str) -> pd.DataFrame:
    """
    Read the ground truth TSV file.

    Columns: source1_entity_id, matched_entity_ids.
    matched_entity_ids is a comma-separated string of S2/S3 IDs (or empty).

    Args:
        path: Path to the ground truth TSV.

    Returns:
        DataFrame with source1_entity_id and matched_entity_ids as strings.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df


def parse_id_list(id_string: str) -> list:
    """
    Parse a comma-separated ID string into a list of IDs.

    Args:
        id_string: Comma-separated IDs, e.g. "S2-001,S3-002". May be empty.

    Returns:
        List of ID strings (empty list if input is empty).
    """
    if not id_string or id_string.strip() == "":
        return []
    return [x.strip() for x in id_string.split(",") if x.strip()]


def format_id_list(ids: list) -> str:
    """
    Format a list of IDs into a comma-separated string with no spaces.

    Args:
        ids: List of ID strings.

    Returns:
        Comma-joined string, or empty string if list is empty.
    """
    return ",".join(ids)


def write_matching_results(results: dict, output_path: str) -> None:
    """
    Write matching_results.tsv.

    Args:
        results: Dict mapping source1_entity_id → list of matched S2/S3 IDs.
        output_path: Where to write the TSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(results.keys()):
            matched = format_id_list(results[s1_id])
            f.write(f"{s1_id}\t{matched}\n")

    print(f"  ✓ Wrote {len(results)} rows to {output_path}")


def write_candidate_pairs(candidates: dict, output_path: str) -> None:
    """
    Write candidate_pairs.tsv.

    Args:
        candidates: Dict mapping source1_entity_id → list of candidate S2/S3 IDs.
        output_path: Where to write the TSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(candidates.keys()):
            cands = format_id_list(candidates[s1_id])
            f.write(f"{s1_id}\t{cands}\n")

    print(f"  ✓ Wrote {len(candidates)} rows to {output_path}")

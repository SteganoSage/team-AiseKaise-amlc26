"""
I/O utilities for reading and writing TSV files.

All data files are TAB-separated. IDs are always strings. Empty lists
stay as "" (not NaN). This module enforces those conventions.

Scale: a source file has up to ~5M rows, so text columns are stored as
pyarrow-backed strings (a few bytes per character instead of ~60 bytes per
Python str object), and quoting is disabled — a stray double quote in a
business name must never swallow tabs or following lines.
"""

import csv
import os

import pandas as pd

# Compact string dtype for millions of rows (falls back to object if pyarrow is missing)
try:
    import pyarrow  # noqa: F401
    STRING_DTYPE = "string[pyarrow]"
except ImportError:  # pragma: no cover
    STRING_DTYPE = str


def read_tsv(path: str) -> pd.DataFrame:
    """
    Read any challenge TSV with the safe settings.

    Args:
        path: Path to the TSV file.

    Returns:
        DataFrame with string columns; empty fields stay "".
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(
        path, sep="\t", dtype=STRING_DTYPE, keep_default_na=False,
        quoting=csv.QUOTE_NONE, engine="c",
    )


def read_source_tsv(path: str) -> pd.DataFrame:
    """
    Read a source TSV file (train or test).

    Columns: entity_id, business_name, business_address, country.

    Args:
        path: Absolute or relative path to the TSV file.

    Returns:
        DataFrame with string columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If expected columns are missing.
    """
    df = read_tsv(path)

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
    return read_tsv(path)


def ground_truth_links(gt: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the ground truth into one row per true (S1, candidate) link.

    Args:
        gt: Output of read_ground_truth.

    Returns:
        DataFrame with columns s1_id, cand_id (plain Python strings).
    """
    ids = gt["matched_entity_ids"].astype(object)
    links = pd.DataFrame({
        "s1_id": gt["source1_entity_id"].astype(object),
        "cand_id": ids.str.split(","),
    }).explode("cand_id")
    links["cand_id"] = links["cand_id"].str.strip()
    return links[links["cand_id"].notna() & (links["cand_id"] != "")].reset_index(drop=True)


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


def format_id_list(ids) -> str:
    """
    Format a list of IDs into a comma-separated string with no spaces.

    Args:
        ids: Iterable of ID strings.

    Returns:
        Comma-joined string, or empty string if list is empty.
    """
    return ",".join(ids)


def write_id_lists(path: str, header: tuple, s1_ids: list, lists: dict) -> None:
    """
    Write a two-column TSV: one row per S1 ID, IDs comma-joined, no quoting.

    Args:
        path: Output path.
        header: The two column names.
        s1_ids: Every S1 ID, in output order (each written exactly once).
        lists: Dict S1 ID → list of IDs (missing S1 IDs get an empty list).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{header[0]}\t{header[1]}\n")
        for s1_id in s1_ids:
            f.write(f"{s1_id}\t{format_id_list(lists.get(s1_id, ()))}\n")
    print(f"  ✓ Wrote {len(s1_ids):,} rows to {path}")


def write_matching_results(results: dict, output_path: str, s1_ids: list = None) -> None:
    """
    Write matching_results.tsv.

    Args:
        results: Dict mapping source1_entity_id → list of matched S2/S3 IDs.
        output_path: Where to write the TSV.
        s1_ids: All S1 IDs in output order. Defaults to sorted(results).
    """
    write_id_lists(output_path, ("source1_entity_id", "matched_entity_ids"),
                   s1_ids if s1_ids is not None else sorted(results), results)


def write_candidate_pairs(candidates: dict, output_path: str, s1_ids: list = None) -> None:
    """
    Write candidate_pairs.tsv.

    Args:
        candidates: Dict mapping source1_entity_id → list of candidate S2/S3 IDs.
        output_path: Where to write the TSV.
        s1_ids: All S1 IDs in output order. Defaults to sorted(candidates).
    """
    write_id_lists(output_path, ("source1_entity_id", "candidate_entity_ids"),
                   s1_ids if s1_ids is not None else sorted(candidates), candidates)


def group_pairs(s1_ids: list, tgt_ids: list) -> dict:
    """
    Turn aligned lists of pair IDs into {S1 ID: [candidate IDs]}.

    Args:
        s1_ids: S1 entity ID of each pair.
        tgt_ids: S2/S3 entity ID of each pair.

    Returns:
        Dict S1 ID → list of target IDs (deduplicated, in pair order).
    """
    out = {}
    for s, t in zip(s1_ids, tgt_ids):
        out.setdefault(s, []).append(t)
    return {s: list(dict.fromkeys(ts)) for s, ts in out.items()}

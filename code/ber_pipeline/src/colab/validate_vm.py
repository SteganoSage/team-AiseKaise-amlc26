"""Run the official validator on the VM (with --check-ids).

The validator only reads the first column of the test source TSVs, so id-only
TSVs are written from the cached parquet files.
"""
import os
import subprocess

import polars as pl

d = "/content/dataset/test"
os.makedirs(d, exist_ok=True)
for n in (1, 2, 3):
    (pl.read_parquet(f"/content/ber_work/raw/test_source{n}.parquet", columns=["entity_id"])
       .write_csv(f"{d}/test_source{n}.tsv", separator="\t"))
r = subprocess.run(
    ["python", "/content/utils/validate_submission.py",
     "--matching", "/content/ber_work/output/matching_results.tsv",
     "--candidate", "/content/ber_work/output/candidate_pairs.tsv",
     "--test-dir", d, "--check-ids"],
    capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-2000:], "exit", r.returncode)

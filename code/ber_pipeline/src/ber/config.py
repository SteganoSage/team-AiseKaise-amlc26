"""Paths shared by all pipeline stages.

Override with environment variables so the same code runs locally and on Colab:
  BER_DATA_DIR  folder containing train/ and test/ (the challenge `dataset/` folder)
  BER_WORK_DIR  folder for cached parquet files, candidates, features and models
"""
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get(
    "BER_DATA_DIR", _REPO.parent / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", "D:/ber_work"))

# Threads for native code (sparse top-k, LightGBM); override with BER_THREADS.
THREADS = int(os.environ.get("BER_THREADS", os.cpu_count() or 4))

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)


def stage_dir(name, tag=""):
    """Folder name of a pipeline stage; dev runs add a tag suffix."""
    return f"{name}_{tag}" if tag else name


def work(*parts):
    """Return a path under WORK_DIR, creating the parent folder."""
    p = WORK_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

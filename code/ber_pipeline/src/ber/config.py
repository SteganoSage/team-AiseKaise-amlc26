"""Paths and run settings shared by all pipeline stages.

Every setting can be overridden with an environment variable, so the same code
runs unchanged on a laptop, Kaggle or a cloud VM:

  BER_DATA_DIR    folder containing train/ and test/   (default: <root>/dataset)
  BER_WORK_DIR    scratch folder for parquet caches, candidates, features and
                  models, tens of GB on the full data   (default: <root>/work)
  BER_OUTPUT_DIR  where the two submission files go     (default: <root>/output)
  BER_THREADS     threads for sparse top-k, rapidfuzz and LightGBM (default: all cores)

<root> is the folder that contains code/ (the submission root), so with the
dataset unpacked into <root>/dataset the pipeline needs no settings at all.
"""
import os
from pathlib import Path

# <root>/code/business_entity_resolution/src/ber/config.py -> <root>
ROOT = Path(__file__).resolve().parents[4]

DATA_DIR = Path(os.environ.get("BER_DATA_DIR", ROOT / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", ROOT / "work"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", ROOT / "output"))

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

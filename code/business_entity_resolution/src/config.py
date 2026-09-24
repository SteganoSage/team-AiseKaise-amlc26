"""
Configuration module for the Business Entity Resolution pipeline.

Centralizes all paths, hyperparameters, thresholds, and random seeds
to ensure reproducibility and easy tuning.
"""

import os

# ──────────────────────────────────────────────────────────────────────
# Paths (relative to the repository root)
# ──────────────────────────────────────────────────────────────────────

# Repo root = three levels up from this file (src → business_entity_resolution → code → root)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))


def set_paths(data_dir: str = None, output_dir: str = None,
              model_dir: str = None) -> None:
    """
    (Re)compute every data/output path from three base directories.

    Called once at import with the repo defaults, and again by run_pipeline
    when --data-dir / --output-dir / --model-dir are given (e.g. on Kaggle,
    where the dataset lives in read-only /kaggle/input and outputs must go
    to /kaggle/working).

    Args:
        data_dir: Folder containing train/ and test/. Defaults to <repo>/dataset.
        output_dir: Folder for matching_results.tsv + candidate_pairs.tsv.
            Defaults to <repo>/output.
        model_dir: Folder for the trained model and run_info.json.
            Defaults to <repo>/code/business_entity_resolution/models.
    """
    global DATA_DIR, TRAIN_DIR, TEST_DIR, OUTPUT_DIR, MODEL_DIR
    global TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH
    global TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3
    global MATCHING_RESULTS, CANDIDATE_PAIRS

    DATA_DIR = os.path.abspath(data_dir or os.path.join(REPO_ROOT, "dataset"))
    OUTPUT_DIR = os.path.abspath(output_dir or os.path.join(REPO_ROOT, "output"))
    MODEL_DIR = os.path.abspath(
        model_dir or os.path.join(REPO_ROOT, "code", "business_entity_resolution", "models")
    )
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TEST_DIR = os.path.join(DATA_DIR, "test")

    TRAIN_SOURCE1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
    TRAIN_SOURCE2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
    TRAIN_SOURCE3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
    TRAIN_GROUND_TRUTH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

    TEST_SOURCE1 = os.path.join(TEST_DIR, "test_source1.tsv")
    TEST_SOURCE2 = os.path.join(TEST_DIR, "test_source2.tsv")
    TEST_SOURCE3 = os.path.join(TEST_DIR, "test_source3.tsv")

    MATCHING_RESULTS = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")


set_paths()

# Organizers' submission validator (run automatically at the end of --mode test)
VALIDATOR_SCRIPT = os.path.join(REPO_ROOT, "utils", "validate_submission.py")

# ──────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

# Validation split ratio (fraction of S1 entities held out, untouched until
# the final score in --mode validate)
VAL_SPLIT_RATIO = 0.2

# Folds for group-aware (by S1 entity) cross-validation. Out-of-fold scores
# pick the threshold and the number of boosting rounds.
CV_FOLDS = 5

# ──────────────────────────────────────────────────────────────────────
# Blocking parameters
# ──────────────────────────────────────────────────────────────────────

# Top-K candidates per blocker per S1 entity
BLOCKING_TOP_K_TFIDF_NAME = 50
BLOCKING_TOP_K_TFIDF_COMBINED = 30
BLOCKING_TOP_K_EMBEDDING = 20

# Exact-key blockers (postal code, first name token): a key shared by more
# than this many S2/S3 records (e.g. a busy PIN code, "sri", "the") is too
# unspecific and is skipped; the TF-IDF blockers still cover those records.
BLOCKING_EXACT_KEY_MAX_BLOCK = 50

# First-name-token keys shorter than this are skipped ("a", "m", "de", ...)
BLOCKING_FIRST_TOKEN_MIN_LEN = 3

# Only compare records with the same country string. S1 records with no
# country, and countries with no S2/S3 records at all (e.g. "India" vs "IN"
# spelling mismatch), fall back to searching all S2/S3 records. S2/S3 records
# with no country are searched from every country.
BLOCK_BY_COUNTRY = True

# TF-IDF char n-gram range
TFIDF_NGRAM_RANGE = (3, 4)

# ──────────────────────────────────────────────────────────────────────
# Embeddings (optional: needs sentence-transformers + model download,
# a GPU is recommended — use on Kaggle)
# ──────────────────────────────────────────────────────────────────────

# When True: an extra embedding kNN blocker (BLOCKING_TOP_K_EMBEDDING per
# source) and embedding-cosine pair features.
USE_EMBEDDINGS = False

# MIT license, 118M params — listed in MODELS.md
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# e5 models expect this prefix on every input text
EMBEDDING_PREFIX = "query: "
EMBEDDING_BATCH_SIZE = 256

# ──────────────────────────────────────────────────────────────────────
# Pair features — switch whole groups on/off for ablations
# ──────────────────────────────────────────────────────────────────────

FEATURE_GROUPS = {
    "string": True,     # rapidfuzz similarities on name / name_core / address
    "tokens": True,     # token Jaccard / overlap on name_core and address
    "tfidf": True,      # TF-IDF cosine: name chars, address chars, name+address words
    "numbers": True,    # overlap of all numbers in the address
    "structure": True,  # postal code / house number / country / lengths / missing / source
    "rank": True,       # rank + gap to best within the S1's candidates and the
                        # candidate's S1s, mutual best, candidate counts
    "blockers": True,   # which blockers produced the pair
    "embedding": True,  # embedding cosine (only when USE_EMBEDDINGS)
}

# Pairs per chunk for the TF-IDF / embedding row-wise cosine (memory bound)
FEATURE_CHUNK_SIZE = 200_000

# ──────────────────────────────────────────────────────────────────────
# Model (LightGBM) hyperparameters
# ──────────────────────────────────────────────────────────────────────

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
    "seed": RANDOM_SEED,
    # Physical cores ≈ logical / 2. All logical threads was ~2x slower on a
    # 16-thread laptop (thread contention on small data).
    "n_jobs": max(1, (os.cpu_count() or 2) // 2),
    # Bit-for-bit reproducible trees (organizers rerun top teams' code)
    "deterministic": True,
    "force_row_wise": True,
}

LGBM_NUM_BOOST_ROUND = 1000
LGBM_EARLY_STOPPING_ROUNDS = 50

# ──────────────────────────────────────────────────────────────────────
# Decision threshold
# ──────────────────────────────────────────────────────────────────────

# Fallback threshold only; every run re-tunes it on out-of-fold scores to
# maximize macro F0.5. Expect a high value (precision-heavy metric).
MATCH_THRESHOLD = 0.5

# Threshold search grid: np.arange(start, stop, step)
THRESHOLD_GRID = (0.30, 0.995, 0.01)

# S1 is deduplicated → an S2/S3 record belongs to at most one S1. When a
# record clears the threshold for several S1s, keep only the highest-scoring one.
RESOLVE_CONFLICTS = True

# Optional: cap on matches per S1 entity (None = no cap)
MAX_MATCHES_PER_S1 = None

# Tune separate thresholds for S2 and S3 candidates (coordinate search
# starting from the best single threshold) instead of one shared threshold.
PER_SOURCE_THRESHOLD = False

# ──────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────

# --mode validate writes holdout false positives / false negatives with
# names + addresses to <model_dir>/holdout_errors.tsv
WRITE_ERROR_DUMP = True

# --mode loco skips countries with fewer S1 entities than this
LOCO_MIN_S1 = 50

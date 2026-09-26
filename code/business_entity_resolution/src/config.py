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
# pick the threshold and the number of boosting rounds. (3, not 5: the real
# train set gives ~10M candidate pairs, so each fold is already large.)
CV_FOLDS = 3

# ──────────────────────────────────────────────────────────────────────
# Scale (the real data is ~12M records per split: S1 ~2M, S2/S3 ~5M each)
# ──────────────────────────────────────────────────────────────────────

# Worker processes for normalization and blocking (all cores)
N_JOBS = os.cpu_count() or 1

# --mode validate / loco: use a random sample of this many train S1 entities
# (all S2/S3 records stay in the pool, so difficulty is realistic).
# None = all ~2.2M. Override per run with --sample-s1.
SAMPLE_S1 = None

# --mode test: train on this many random train S1 entities instead of all
# ~2.2M (the test side always scores every test S1). 300k S1 ≈ 2M candidate
# pairs is plenty for LightGBM and cuts the training half of a full run ~7x.
# None = all. Override per run with --train-sample-s1.
TRAIN_SAMPLE_S1 = None

# Normalized source files are cached as parquet in <model_dir>/cache and
# reused until the raw file or normalize.py changes.
USE_NORMALIZE_CACHE = True

# Rewrite S2/S3 names/addresses written in Indic scripts (after
# transliteration) and website-style names ("anilandevelopers.com") in the
# words of the same split's S1 (normalize.snap_to_reference). ~49% of the
# India true matches that blocking missed had an Indic-script name.
SNAP_TO_REFERENCE = True

# ──────────────────────────────────────────────────────────────────────
# Stage 1 — blocking (broad, cheap, high recall)
# ──────────────────────────────────────────────────────────────────────

# Hashed TF-IDF over word 1-2 grams, per source, searched per country:
#   "name":     name_core
#   "combined": name_norm + address_norm
#   "address":  address_norm only (finds records whose name is garbled but
#               whose house number + street match; 0 = off)
# Top-K per blocker per source per S1 entity.
BLOCKING_TOP_K = {"name": 10, "combined": 10, "address": 0}
BLOCKING_TOP_K_EMBEDDING = 10

# Terms (words or word pairs) found in more than this many S2/S3 records of
# a source are dropped from blocking: they are too common to identify a
# business ("limited", "road", "delhi") and they make the sparse product huge.
# Word pairs ("newton road", "prime realty") are much rarer, so they survive.
BLOCKING_MAX_DF = 2000

# Hash space for the TF-IDF vectorizers (collisions are negligible at 2^22)
HASH_FEATURES = 2 ** 22

# S1 rows per sparse-product chunk (memory per worker ~ chunk x postings)
BLOCKING_CHUNK_ROWS = 2000

# Only compare records with the same country string. S1 records with no
# country, and countries with no S2/S3 records at all (e.g. "India" vs "IN"
# spelling mismatch), fall back to searching all S2/S3 records. S2/S3 records
# with no country are searched from every country.
BLOCK_BY_COUNTRY = True

# ──────────────────────────────────────────────────────────────────────
# Stage 2 — pruning (makes candidate_pairs.tsv small)
# ──────────────────────────────────────────────────────────────────────
# A light LightGBM on cheap features (blocker scores/ranks + a few rapidfuzz
# similarities) ranks each S1's stage-1 candidates. The kept pairs are the
# exact set the final model scores = candidate_pairs.tsv. Smaller candidate
# sets per S1 rank higher in the final evaluation, so this is tuned for
# "as few as possible without losing true matches".

# Sweep on 100k validate (7078791): 10/0.01 → F0.5 0.9562, 5.95 cands/S1;
# 8/0.03 → 0.9559, 4.92 cands/S1 (-17%). Fewer than ~7 per S1 costs F0.5.
# Confirmed by a real 100k validate on e6d2766: holdout 0.9559, OOF 0.9556,
# 4.92 cands/S1, India 0.9463 / US 0.9624.
PRUNE_TOP_N = 8           # keep at most this many candidates per S1 (both sources)
PRUNE_MIN_PROB = 0.03     # ...and only those with pruner probability ≥ this

# Report-only sweep of stricter pruning settings (candidate_pairs.tsv size
# counts for the final ranking). --mode validate replays each (top-N,
# min-prob) setting on the run's own scores: re-tunes the threshold on OOF and
# prints holdout F0.5 / candidates per S1 by country. --mode test prints the
# test candidates per S1 each setting would give. Never changes the outputs.
PRUNE_SWEEP = True
PRUNE_SWEEP_TOP_N = (6, 7, 8)
PRUNE_SWEEP_MIN_PROB = (0.03, 0.05, 0.1)
PRUNE_FOLDS = 2           # out-of-fold pruning on train (by S1 entity)
PRUNE_ROUNDS = 200        # boosting rounds for the pruner
PRUNE_MAX_TRAIN_PAIRS = 20_000_000  # subsample S1 entities above this

PRUNER_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_child_samples": 50,
    "feature_fraction": 0.9,
    "verbose": -1,
    "seed": RANDOM_SEED,
    "n_jobs": N_JOBS,
    "deterministic": True,
    "force_row_wise": True,
}

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
EMBEDDING_BATCH_SIZE = 512

# ──────────────────────────────────────────────────────────────────────
# Stage 3 — pair features for the final matcher. Switch whole groups on/off
# for ablations.
# ──────────────────────────────────────────────────────────────────────

FEATURE_GROUPS = {
    "string": True,     # rapidfuzz similarities on name / name_core / address
    "tokens": True,     # token Jaccard / overlap on name_core and address
    "tfidf": True,      # blocking TF-IDF cosines (name, name+address) + their ranks
    "numbers": True,    # overlap of all numbers in the address
    "structure": True,  # postal code / house number / country / lengths / missing / source
    "rank": True,       # rank + gap to best within the S1's candidates and the
                        # candidate's S1s, mutual best, candidate counts
    "blockers": True,   # which blockers produced the pair
    "pruner": True,     # stage-2 pruner probability
    "embedding": True,  # embedding cosine (only when USE_EMBEDDINGS)
    # Ideas from the record-linkage literature (Fellegi-Sunter, Ditto/DeepMatcher
    # feature lists), not yet measured on a real validate:
    "frequency": True,  # how many S1 / S2+S3 records share the name / address
                        # (whole split, before S1 sampling)
    "cross": True,      # name × address product / min / max, exact flags,
                        # first / last name word equal
    "char_ngram": True, # character 3-gram cosine of name_core (typos)
    # Similarity of a candidate to the OTHER candidates of the same S1 (true
    # matches of one business resemble each other; look-alikes do not), weighted
    # by pruner probability. Not yet measured (base: llm-advice 27ad884 = 0.9633).
    "sibling": True,
}

# Pairs per chunk when computing features / cosines (memory bound)
FEATURE_CHUNK_SIZE = 1_000_000

# ──────────────────────────────────────────────────────────────────────
# Model (LightGBM) hyperparameters
# ──────────────────────────────────────────────────────────────────────

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    # Regularization: without it single trees made outsized corrections (train
    # loss rose after ~300 rounds, CV folds stopped at 144-635 rounds). 100k
    # validate: holdout F0.5 0.9532 -> 0.9564, folds 588-752 rounds.
    "lambda_l2": 1.0,
    "min_sum_hessian_in_leaf": 1.0,
    "verbose": -1,
    "seed": RANDOM_SEED,
    "n_jobs": N_JOBS,
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

# Tune a separate threshold for the best-scoring candidate of each S1 ("top1")
# on top of the shared one. For the top candidate, predicting nothing costs as
# much as a wrong match (both score 0), while every extra match only moves
# precision/recall a little, so the best cutoffs usually differ.
TOP1_THRESHOLD = False

# ──────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────

# --mode validate writes holdout false positives / false negatives with
# names + addresses to <model_dir>/holdout_errors.tsv
WRITE_ERROR_DUMP = True

# --mode loco skips countries with fewer S1 entities than this
LOCO_MIN_S1 = 50

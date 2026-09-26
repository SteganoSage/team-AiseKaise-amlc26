# Business Entity Resolution — Run Instructions

## Prerequisites

- Python 3.10–3.12, Linux recommended (worker processes use `fork`)
- Dataset files in `dataset/train/` and `dataset/test/` (TSV), or pass `--data-dir`
- The real data is ~12M records per split. A full run needs **~30 GB RAM** (e.g. a SageMaker
  `ml.r5.2xlarge` / `ml.m5.4xlarge` notebook, or Kaggle). A laptop can run
  `--mode validate --sample-s1 20000` or smaller.

## Setup

```bash
pip install -r code/business_entity_resolution/requirements.txt
# only for --embeddings (GPU):
pip install -r code/business_entity_resolution/requirements-embeddings.txt
```

## Pipeline

```
load TSVs (compact pyarrow strings, quoting off)
  → normalize names/addresses in parallel            (cached as parquet)
  → stage 1  blocking: hashed TF-IDF on word 1-2 grams of name and of
             name+address, common terms dropped, sparse top-K per S1 per
             source (S2, S3) inside the same country     [+ optional embeddings]
  → stage 2  pruning: light LightGBM on cheap features keeps the best few
             candidates per S1  → output/candidate_pairs.tsv
  → stage 3  matching: ~70 pair features → LightGBM → threshold tuned for
             macro F0.5 on out-of-fold scores → one S1 per S2/S3 record
  → output/matching_results.tsv → organizers' validator
```

`candidate_pairs.tsv` is exactly the set the stage-3 model scores (the last
filtering stage), as the rules require. Smaller candidate sets per S1 rank higher,
so every run reports candidates per S1 and candidate recall.

## Running

### Local validation (train holdout → F0.5)

```bash
python code/business_entity_resolution/src/run_pipeline.py --mode validate --sample-s1 200000
```

Uses a random sample of train S1 entities (all S2/S3 records stay in the pool, so
difficulty is realistic); omit `--sample-s1` for all ~2.2M. Holds out 20% of those
S1 entities; fits on the rest (3-fold group CV by S1 entity; out-of-fold scores
pick the threshold and boosting rounds); reports on the untouched holdout:
macro F0.5 / P / R next to the all-empty baseline, candidates per S1 and candidate
recall after each stage, per-blocker recall, a per-country table, and writes
`models/holdout_errors.tsv` (false positives, missed matches, and true matches
that never became candidates — with names and addresses).

### Leave one country out (stand-in for the unseen France)

```bash
python code/business_entity_resolution/src/run_pipeline.py --mode loco --sample-s1 200000
```

Fits on one train country, evaluates on the other; shows the F0.5 at the
transferred threshold next to the best achievable ("oracle") threshold.

### Test prediction (writes output files)

```bash
python code/business_entity_resolution/src/run_pipeline.py --mode test
```

Fits on all training data, then writes:
- `output/matching_results.tsv` — upload this to the portal
- `output/candidate_pairs.tsv` — goes in the final zip (audited, counts for ranking)

and runs `utils/validate_submission.py` on them automatically (must print `PASS`).

For lower peak memory on Kaggle, run the two halves as separate processes (or
separate notebook runs) using the same `--model-dir`:

```bash
python code/business_entity_resolution/src/run_pipeline.py --mode train_only \
  --data-dir /kaggle/input/<dataset>/dataset \
  --output-dir /kaggle/working/output --model-dir /kaggle/working/models

python code/business_entity_resolution/src/run_pipeline.py --mode predict_only \
  --data-dir /kaggle/input/<dataset>/dataset \
  --output-dir /kaggle/working/output --model-dir /kaggle/working/models
```

`train_only` saves the matcher, pruner, threshold, and feature-state files;
`predict_only` loads them and writes the same two validated output files.

Every mode saves its numbers to `models/run_info_<mode>.json` — copy them into
`submissions/LOG.md` and PR descriptions.

### Options

| Flag | Meaning |
|---|---|
| `--data-dir DIR` | folder containing `train/` and `test/` (default `<repo>/dataset`) |
| `--output-dir DIR` | where the two TSVs go (default `<repo>/output`) |
| `--model-dir DIR` | models, normalized cache, run info, error dump (default `code/business_entity_resolution/models`) |
| `--sample-s1 N` | validate/loco: use N random train S1 records |
| `--embeddings` | add the multilingual-e5 embedding blocker + features (GPU strongly recommended) |
| `--no-cache` | don't read/write the normalized parquet cache |
| `--n-jobs N` | worker processes / LightGBM threads (default all cores) |

All other knobs (top-K per blocker, dropped-term threshold, pruning size, feature
groups, threshold grid, per-source thresholds) are in `src/config.py`.

### Kaggle

The input is read-only, so point outputs at `/kaggle/working`:

```bash
pip install -q rapidfuzz==3.10.1
python code/business_entity_resolution/src/run_pipeline.py --mode test \
  --data-dir /kaggle/input/<dataset>/dataset \
  --output-dir /kaggle/working/output --model-dir /kaggle/working/models
```

The normalized cache (`models/cache/`, ~1-2 GB per split) makes reruns much faster.

### Validate before upload

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Source files

| File | Role |
|---|---|
| `src/config.py` | paths, seeds, every tunable setting |
| `src/io_utils.py` | TSV read/write (tab-separated, no quoting, string IDs) |
| `src/normalize.py` | text normalization, abbreviations, legal suffixes, postal codes / numbers |
| `src/blocking.py` | stage 1 candidate generation + candidate/recall reports |
| `src/prune.py` | stage 2 pruner (candidate_pairs) |
| `src/features.py` | stage 3 pair features |
| `src/train.py` | labels, group CV, LightGBM training |
| `src/predict.py` | decision rule + threshold tuning |
| `src/evaluate.py` | macro F0.5 (reference + vectorized) |
| `src/embeddings.py` | optional multilingual-e5 embeddings |
| `src/run_pipeline.py` | end-to-end runner (`--mode validate / loco / test`) |
| `src/eda.py`, `src/make_empty_submission.py`, `src/archive_submission.py`, `src/make_test_data.py` | utilities |

See [CLAUDE.md](../../CLAUDE.md) §7 for the approach and [MODELS.md](MODELS.md) for pretrained models.

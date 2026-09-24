# Business Entity Resolution — Run Instructions

## Prerequisites

- Python 3.10+
- Dataset files in `dataset/train/` and `dataset/test/` (TSV format)

## Setup

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

## Running

### Local Validation (train split → held-out F0.5)

```bash
python code/business_entity_resolution/src/run_pipeline.py --mode validate
```

Holds out 20% of train S1 entities. On the other 80% it runs blocking, builds
features, and runs 5-fold group CV (by S1 entity); the out-of-fold scores pick
the threshold and the number of boosting rounds. It then reports blocking
recall and macro F0.5 on the untouched holdout, next to the all-empty baseline.

### Test Prediction (writes output files)

```bash
python code/business_entity_resolution/src/run_pipeline.py --mode test
```

Fits the same way on all training data (threshold from out-of-fold scores),
then produces:
- `output/matching_results.tsv` — final predictions for leaderboard upload
- `output/candidate_pairs.tsv` — candidate set the model scored

and runs `utils/validate_submission.py` on them automatically.

Both modes save their key numbers (F0.5, threshold, blocking recall) to
`code/business_entity_resolution/models/run_info_<mode>.json` for `submissions/LOG.md`.

### Custom paths (Kaggle / Colab)

`--data-dir` (folder containing `train/` and `test/`), `--output-dir` and
`--model-dir` override the defaults. On Kaggle the input is read-only:

```bash
pip install rapidfuzz==3.10.1   # other libraries are preinstalled on Kaggle
python code/business_entity_resolution/src/run_pipeline.py --mode validate \
  --data-dir /kaggle/input/<dataset>/dataset \
  --output-dir /kaggle/working/output --model-dir /kaggle/working/models
```

### Validate Before Upload

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Must print `PASS` before submitting.

## Pipeline Overview

```
load TSVs → normalize names/addresses → blocking (TF-IDF + exact keys)
  → pair features (string similarity, embedding cosine, etc.)
  → LightGBM classifier → threshold tuning → output
```

See [CLAUDE.md](../../CLAUDE.md) §7 for detailed approach description.

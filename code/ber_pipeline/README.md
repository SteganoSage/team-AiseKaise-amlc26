# ber pipeline (Saksham)

Two-stage entity-resolution pipeline, separate from `code/business_entity_resolution/`.
Written and run by Saksham on a Colab VM (see `src/colab/` for the VM helper
scripts); imported into branch `lavya` unchanged (commit 37f5118), then with the
small changes listed at the end.

## Stages (`src/run_all.py` runs them in order)

| Step | Module | What it does |
|---|---|---|
| prepare | `ber.prepare` | Normalise names/addresses of both splits (Indic → ASCII via anyascii, legal forms, state codes; state aliases learned from training matches) → parquet |
| cand_train / cand_test | `ber.candidates` | Blocking per country: TF-IDF views (char 3/4-grams of the name, word pairs of name+address, address words, char 4-grams of name+address), top-K per view + reverse search (each record's top S1s) |
| feat_train / feat_test | `ber.featurize` | ~100+ pair features (fuzzy ratios on several name forms, token sets, numbers, state/legal agreement, IDF of differing words, context ranks) |
| model_stage1 | `ber.model stage1` | LightGBM, 3 folds grouped by S1 → first-stage probability p1 for every pair |
| ce_train / ce_score | `ber.crossenc` | Fine-tune `intfloat/multilingual-e5-base` on raw texts of uncertain pairs (reserved 20 % of training entities) → score (GPU) |
| stage2_train / stage2_test | `ber.stage2` | p1 in context (record competition, entity context, cluster support = similarity to the entity's confident records) + cross-encoder score |
| model_stage2 / model_predict | `ber.model` | LightGBM on everything, decision rule tuned on out-of-fold macro F0.5 (threshold or expected-F0.5 top-k), one owner per record → `matching_results.tsv`, `candidate_pairs.tsv` |

## Run

```bash
pip install -r code/ber_pipeline/requirements.txt
cd code/ber_pipeline/src
export BER_DATA_DIR=/path/to/student_resource/dataset   # folder with train/ and test/
export BER_WORK_DIR=/path/to/scratch                     # parquet caches, features, models (large)
export BER_OUTPUT_DIR=/path/to/output                    # default: $BER_WORK_DIR/output
python run_all.py                    # all steps; --from <step> resumes
```
The official validator runs at the end when `BER_DATA_DIR/../utils/validate_submission.py` exists.

Settings (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `BER_THREADS` | all cores | threads for sparse top-k and LightGBM |
| `BER_TRAIN_FRAC` | 0.6 | share of training entities the LightGBMs are fitted on (memory: ~50M rows at 0.6) |
| `BER_CAND_TOP_N` | 0 (off) | keep only the top-N pairs per S1 by stage-1 probability as final candidates |
| `BER_CAND_MIN_P1` | 0 | ...and only pairs with stage-1 probability ≥ this |

Requirements: a CUDA GPU for the cross-encoder steps (bfloat16 on A100/L4, float16
on T4); a large-memory machine for the full data.

## Changes made in `lavya` on top of the imported code

1. `ber/prepare.py`: state-alias sample is `min(3M, rows)` (crashed on small data).
2. `ber/crossenc.py`: float16 + loss scaling when the GPU has no bfloat16 (Kaggle/Colab T4); unchanged on bfloat16 GPUs.
3. `ber/model.py`: `BER_TRAIN_FRAC` environment variable (run_all.py passes no `--frac`).
4. `ber/model.py`: optional candidate trimming (`BER_CAND_TOP_N`, `BER_CAND_MIN_P1`, off by default) and a report-only
   "trim report" in `stage2`: out-of-fold macro F0.5, candidates per S1 and candidate recall for top-N ∈ {all, 15, 10, 8, 6}
   × min p1 ∈ {0, 0.01, 0.03}. `candidate_pairs.tsv` size counts in the final ranking (portal update, 25 Sep); without
   trimming every blocked pair (top-K per view + reverse search) is a candidate.

Smoke-tested on the fake dataset (all steps except the cross-encoder, which needs a GPU and the model download):
official validator PASS with trimming off and on.

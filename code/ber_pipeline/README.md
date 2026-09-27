# Business Entity Resolution — Team AiseKaise

Two-stage LightGBM entity-resolution pipeline with a fine-tuned multilingual
cross-encoder. For every Source 1 business it predicts the Source 2 / Source 3
records of the same real-world business and writes the two challenge files.

In the submission zip this folder is `code/business_entity_resolution/`.

## Quick start

```bash
# from the submission root (the folder that contains code/ and output/)
pip install -r code/business_entity_resolution/requirements.txt
# put the organisers' data in dataset/train and dataset/test (or set BER_DATA_DIR)
cd code/business_entity_resolution/src
python run_all.py
```

This regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` at
the submission root. The organisers' validator runs at the end if it is found
(`<dataset>/../utils/validate_submission.py` or `<root>/utils/validate_submission.py`).
Resume a failed run with `python run_all.py --from <step>`.

### Settings (environment variables, all optional)

| Variable | Default | Meaning |
|---|---|---|
| `BER_DATA_DIR` | `<root>/dataset` | folder with `train/` and `test/` |
| `BER_WORK_DIR` | `<root>/work` | scratch: parquet caches, candidates, features, models (tens of GB) |
| `BER_OUTPUT_DIR` | `<root>/output` | the two submission files |
| `BER_THREADS` | all cores | threads for sparse top-k, rapidfuzz and LightGBM |
| `BER_TRAIN_FRAC` | 0.6 | share of training entities the LightGBMs are fitted on (memory) |
| `BER_CAND_TOP_N` | 0 (off) | keep only the top-N pairs per S1 by stage-1 probability as final candidates |
| `BER_CAND_MIN_P1` | 0 | ...and only pairs with stage-1 probability ≥ this |

### Hardware

- A CUDA GPU for the two cross-encoder steps (bfloat16 on A100/L4, float16 with loss
  scaling on T4). Without a GPU `run_all.py` skips them automatically (or pass
  `--skip-crossenc`); the pipeline still runs, a little weaker.
- A large-memory machine for the full data (~12M records per split; the reference
  run used a Colab VM). `BER_TRAIN_FRAC` lowers the memory of the model steps.

## Pipeline

| Step (`run_all.py`) | Module | What it does |
|---|---|---|
| `prepare` | `ber.prepare`, `ber.normalize` | Read the TSVs (no quoting, all text), normalise names/addresses of both splits: any script → ASCII (anyascii), legal forms, DBA wrappers, domain-style names, digit-for-letter typos, address abbreviations, state names/codes (+ transliterated state names learned from training matches) → parquet |
| `cand_train`, `cand_test` | `ber.candidates`, `ber.blocking`, `ber.tfidf` | Blocking per country (country is an open set; France is handled like any other): TF-IDF views — word 1-2 grams of name+address, word 1-2 grams of the address, char 4-grams of the name, char 4-grams of name+address — sparse top-K per S1 per view, plus a reverse search (each record's top S1s). Transliterated names are mapped back to Latin words first (`ber.translit`) |
| `feat_train`, `feat_test` | `ber.featurize`, `ber.features` | ~100+ pair features: exact TF-IDF cosines per view, fuzzy ratios on several name forms, token sets, numbers, state/legal agreement, IDF of differing words, rank/gap context within the S1 and within the record |
| `model_stage1` | `ber.model stage1` | LightGBM, 3 folds grouped by S1 → out-of-fold first-stage probability `p1` for every training pair, fold-mean for test |
| `ce_train`, `ce_score` | `ber.crossenc` | Fine-tune `intfloat/multilingual-e5-base` (MIT) on the raw "name \| address" texts of uncertain pairs of a reserved 20 % of training entities; score uncertain train/test pairs |
| `stage2_train`, `stage2_test` | `ber.stage2` | `p1` in context: competition for the record, entity context, cluster support (similarity to the entity's confident records), cross-encoder score and gaps |
| `model_stage2`, `model_predict` | `ber.model`, `ber.decide`, `ber.submit` | Second LightGBM on everything; decision rule tuned on out-of-fold macro F0.5 (threshold, or expected-F0.5 top-k per S1); each record kept for at most one S1; write both files (matches ⊆ candidates) |

`ber.metrics` implements the challenge metric (macro F0.5 per S1, singletons
included) and blocking recall; `ber.eval_blocking` is an optional blocking-recall
diagnostic on a sample of training entities.

## Leakage control and reproducibility

- Folds, training subsets and the cross-encoder pool are chosen by hashing the S1
  id with fixed seeds, so they are identical on every run and machine.
- The cross-encoder is trained only on its reserved pool, which the LightGBMs never
  see; stage-1 probabilities used as stage-2 inputs on training pairs are out-of-fold.
- No external data or services: only the provided files, hand-written normalisation
  dictionaries and one pretrained model (see `MODELS.md`).

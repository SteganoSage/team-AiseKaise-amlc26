## What changed
<!-- One idea per PR. What did you change and why? -->

## Local validation — `run_pipeline.py --mode validate`
<!-- Run on main and on this branch with the same data. Numbers are in models/run_info_validate.json. -->

| | main | this branch |
|---|---|---|
| Holdout macro F0.5 | | |
| Holdout precision / recall | | |
| OOF F0.5 | | |
| Holdout blocking recall | | |
| Candidates per S1 (mean) | | |
| Threshold | | |
| Runtime | | |

## Checklist
- [ ] Branched from latest `main` and merged `main` back in before the final run
- [ ] Did not change `RANDOM_SEED`, `VAL_SPLIT_RATIO`, `CV_FOLDS`, `evaluate.py` or `utils/validate_submission.py`
- [ ] No `dataset/`, `output/*.tsv` or model files committed
- [ ] No hard-coded country values; no external data
- [ ] Any new pretrained model is MIT/Apache-2.0, ≤ 8B params, and listed in `MODELS.md`
- [ ] Docstrings on every new function

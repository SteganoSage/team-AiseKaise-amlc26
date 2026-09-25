# Amazon ML Challenge 2026 — Business Entity Resolution

**Team AiseKaise**

## Problem

Given business records from 3 independent sources (S1, S2, S3), predict which S2/S3
records refer to the same real-world business as each S1 entity. Matching is based on
noisy business names and addresses — there are no shared IDs across sources.
 
- **Train countries:** US, India
- **Test adds:** France (unseen during training)
- **Metric:** Macro-averaged F0.5 per S1 entity (precision-heavy)

## Pipeline

```
load → normalize → blocking (candidates) → pair features → LightGBM classifier → threshold → output
```

## Quick Start

```bash
# Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# Run validation (local F0.5 on held-out train split)
python code/business_entity_resolution/src/run_pipeline.py --mode validate

# Run on test set (writes output/*.tsv)
python code/business_entity_resolution/src/run_pipeline.py --mode test

# Validate submission format before uploading
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Team Workflow

See [PLAN.md](PLAN.md): who owns what, branch/commit format, daily schedule and the upload checklist.

## Repo Structure

See [CLAUDE.md](CLAUDE.md) §8 for full details.

## Rules

- No external data or APIs
- Pretrained models must be MIT or Apache 2.0, ≤ 8B params
- Repo stays private until results are out
- Max 5 leaderboard submissions per day

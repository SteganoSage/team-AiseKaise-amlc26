# Documentation — Amazon ML Challenge 2026

## Team: AiseKaise

---

## 1. Methodology

### Overview
We approach the business entity resolution problem as a supervised binary classification task
on candidate pairs. The pipeline consists of:

1. **Text normalization** — Lowercase, Unicode NFKD accent stripping, abbreviation expansion,
   legal suffix removal, punctuation cleanup.
2. **Blocking (candidate generation)** — Union of TF-IDF char n-grams, TF-IDF word tokens,
   exact postal code keys, and first name token keys.
3. **Pair feature extraction** — Rich string similarity features (rapidfuzz, Jaro-Winkler,
   Jaccard), structural features (postal code match, house number match), and metadata features.
4. **LightGBM classifier** — Binary classifier trained on (S1, candidate) pairs.
5. **Threshold tuning** — Optimize decision threshold on validation data to maximize macro F0.5.
6. **Post-processing** — Resolve multi-assignment conflicts (S2/S3 → at most one S1).

---

## 2. Candidate Generation / Blocking Strategy

### Approach
We use a union of multiple blocking strategies to maximize recall while keeping
the candidate set manageable:

| Blocker | Key | Top-K | Purpose |
|---|---|---|---|
| TF-IDF char n-grams (3-4) | `name_core` | 50 | Catches typos, abbreviation variants |
| TF-IDF word tokens (1-2) | `name_norm + address_norm` | 30 | Catches address-level matches |
| Exact postal code | `postal_codes` | all | Exact geographic blocking |
| First name token | `name_core[0]` | all | Cheap high-recall filter |

### Metrics
- **Blocking recall:** _TODO (fill after validation)_
- **Reduction ratio:** _TODO_
- **Total candidate pairs:** _TODO_

---

## 3. Model Architecture & Feature Engineering

### Model
- **LightGBM** binary classifier (GBDT)
- Group-aware train/val split by S1 entity to prevent leakage

### Features (all country-agnostic)

| Group | Features |
|---|---|
| Name similarity | ratio, partial_ratio, token_sort_ratio, token_set_ratio, Jaro-Winkler (on both name_norm and name_core) |
| Name tokens | Jaccard, overlap count, overlap fraction (S1 / candidate) |
| Address similarity | Same string similarity suite as names |
| Address tokens | Jaccard, overlap count, overlap fraction |
| Postal code | Exact match flag, both present flag |
| House number | Exact match flag, both present flag |
| Country | Exact match flag, both present flag |
| Length | Name/address lengths, length difference, length ratio |
| Missing fields | Flags for empty name/address |
| Source | S2 vs S3 indicator |

---

## 4. Other Relevant Information

### Handling Unseen Countries (France)
- All normalization is language-agnostic: NFKD accent stripping, generic abbreviation
  dictionaries including French legal forms (SARL, SAS, SA, EURL) and address terms
  (bd, r→rue, ch→chemin).
- No country one-hot encoding or country-specific logic.
- Model features are purely string-similarity and structural — no country-dependent features.

### Normalization Details
- Unicode NFKD decomposition + accent mark removal
- `&` → `and`, punctuation stripping, whitespace collapsing
- Name abbreviations: pvt→private, ltd→limited, corp→corporation, etc.
- Address abbreviations: rd→road, st→street, bd→boulevard, r→rue, etc.
- Legal suffix removal for `name_core`: private limited, sarl, sas, gmbh, etc.
- Postal code extraction: 5-6 digit numbers (works for US ZIP, India PIN, France CP)

### Threshold Tuning
- Sweep thresholds from 0.1 to 0.95 in steps of 0.05
- Select threshold maximizing macro F0.5 on held-out validation S1 entities
- Precision-heavy metric → expect a high optimal threshold

### Models Used
See `MODELS.md` for full list with licenses and parameter counts.

---

## 5. Reproducibility

- Fixed random seed: 42
- Pinned `requirements.txt`
- Pipeline regenerates both output files from raw data:
  ```bash
  python code/business_entity_resolution/src/run_pipeline.py --mode test
  ```

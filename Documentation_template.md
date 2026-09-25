# Documentation — Amazon ML Challenge 2026

## Team: AiseKaise

---

## 1. Methodology

### Overview
We treat business entity resolution as supervised classification of (S1, candidate) pairs,
organized as a three-stage cascade so that it scales to ~12M records per split (test: 1.73M S1,
4.89M S2, 5.08M S3) and hands the final model only a small candidate set per S1 entity:

1. **Normalization** — lowercase, Unicode NFKD accent stripping, `&`→`and`, punctuation cleanup,
   abbreviation expansion, legal-suffix removal (`name_core`), placeholder removal (`null`),
   postal code / house number / address number extraction. Parallel over worker processes,
   stored as compact Arrow string columns.
2. **Stage 1 — blocking (candidate generation)** — hashed TF-IDF nearest neighbours on name and
   on name+address, per source, inside the same country (details in §2).
3. **Stage 2 — pruning** — a light LightGBM ranks each S1's stage-1 candidates on cheap features
   and keeps only the best few. This pruned set is `candidate_pairs.tsv`: exactly what the final
   model scores.
4. **Stage 3 — matching** — ~70 pair features, LightGBM binary classifier.
5. **Decision** — threshold tuned for macro F0.5 on out-of-fold predictions; each S2/S3 record is
   assigned to at most one S1 (the highest-scoring), since S1 is deduplicated.

---

## 2. Candidate Generation / Blocking Strategy

### Stage 1 — blocking
For each S1 record and each source (S2, S3) separately, within the same country string (S1
records with no country, or a country the source doesn't have, search all records):

| Blocker | Text | Top-K | Purpose |
|---|---|---|---|
| name | `name_core`, word 1-2 grams | 10 | same business name, other address format |
| combined | `name_norm + address_norm`, word 1-2 grams | 10 | name + street / number / city together |
| embedding (optional) | multilingual-e5-small on raw name + address | 10 | transliteration (Devanagari), reordering |

Why it scales: a HashingVectorizer (no vocabulary in memory); terms found in more than 2,000
target records get IDF weight 0 (they carry almost no identity and are what makes an all-vs-all
product explode — word pairs like "newton road" are rare and survive); the cosine is a sparse
matrix product computed in row chunks across worker processes, keeping only the top-K per row.
Single-key blockers (postal code, first word) were dropped: at 5M records their blocks are huge,
and US addresses rarely carry a ZIP.

### Stage 2 — pruning
Features: every blocker's cosine and rank, number of blockers, source, and five rapidfuzz
similarities (name_core ratio / token-set, name Jaro-Winkler, address token-set / partial).
Keep ≤ 10 candidates per S1 with pruner probability ≥ 0.01. On train, the pruner is applied
out-of-fold (2 folds by S1), so the final model is trained on candidate sets like the test ones.

### Metrics
Numbers from a 20k-S1 slice of train (full-data numbers: _TODO after the Kaggle run_):

| | Stage 1 (blocking) | Stage 2 (pruned = candidate_pairs.tsv) |
|---|---|---|
| Candidates per S1 (mean) | 34.7 | 4.3 |
| Pair recall | 0.987 | 0.983 |
| Reduction ratio (full test) | _TODO_ | _TODO_ |

---

## 3. Model Architecture & Feature Engineering

### Model
- **LightGBM** binary classifier (GBDT), deterministic, fixed seed.
- Group K-fold (3 folds) by S1 entity: out-of-fold predictions choose the threshold and the
  number of boosting rounds; the final model is trained on all pairs.

### Features (all country-agnostic; groups can be switched off for ablations)

| Group | Features |
|---|---|
| String similarity | ratio, partial_ratio, token_sort_ratio, token_set_ratio, Jaro-Winkler on name_norm, name_core, address_norm |
| Tokens | Jaccard, overlap count, overlap fraction per side (name_core, address) |
| Blocking TF-IDF | name and name+address cosines and ranks from stage 1 |
| Address numbers | Jaccard / overlap of all numbers, both-present, "both have numbers but none shared" |
| Structure | postal code match / presence, house number match, country match, lengths, missing fields, source |
| Competition | rank and gap to the best within the S1's candidates and within the candidate's S1s, mutual best, candidate counts |
| Blockers | which blockers proposed the pair |
| Pruner | stage-2 probability |
| Embeddings (optional) | name and name+address embedding cosine |

---

## 4. Other Relevant Information

### Handling Unseen Countries (France)
- Country is an open set of strings: no country is hard-coded, filtered or one-hot encoded.
  Blocking groups by whatever country strings appear, with fallbacks for unknown ones.
- Normalization is language-agnostic: NFKD accent stripping, generic abbreviation dictionaries
  including French legal forms (SARL, SAS, SASU, EURL, SCI) and address terms (bd, r→rue, ch→chemin).
- Features are purely similarity/structure-based.
- We check transfer with leave-one-country-out validation (fit on India, test on US and vice
  versa): _TODO results_.

### Threshold Tuning
- Grid 0.30 → 0.99 (step 0.01) on out-of-fold scores, using the same decision rule as the
  submission (threshold → one S1 per S2/S3 record), maximizing macro F0.5 over all S1 entities
  including singletons and S1s without candidates.

### Models Used
See `code/business_entity_resolution/MODELS.md` (MIT / Apache-2.0, ≤ 8B parameters). No external
data or services are used.

---

## 5. Reproducibility

- Fixed random seed (42), deterministic LightGBM, pinned `requirements.txt`.
- Pipeline regenerates both output files from raw data (≈30 GB RAM for the full data):
  ```bash
  python code/business_entity_resolution/src/run_pipeline.py --mode test --data-dir <dir with train/ and test/>
  ```

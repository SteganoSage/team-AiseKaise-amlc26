# Documentation — Amazon ML Challenge 2026

## Team: AiseKaise

---

## 1. Methodology

### Overview
We treat entity resolution as supervised classification of (Source 1 entity, Source 2/3 record)
pairs, organised as a cascade that scales to ~12M records per split
(test: 1.73M S1, 4.89M S2, 5.08M S3):

1. **Normalisation** — every name and address is mapped to canonical forms (any script → ASCII,
   legal forms, DBA wrappers, domain-style names, address abbreviations, state names).
2. **Candidate generation (blocking)** — several TF-IDF views, sparse top-K nearest neighbours per
   S1 entity inside its country, plus a reverse search from the records' side.
3. **Stage-1 model** — LightGBM on ~100+ pair features gives a first match probability `p1`.
4. **Cross-encoder** — a fine-tuned multilingual transformer reads the raw texts of the uncertain pairs.
5. **Stage-2 model** — LightGBM on the pair features + `p1` in context (competition between
   candidates, cluster support) + the cross-encoder score.
6. **Decision** — rule tuned for macro F0.5 on out-of-fold predictions; each record is assigned to at
   most one S1 entity (S1 is deduplicated; true for all 7.6M training matches).

All code is in `code/business_entity_resolution/src/` (`python run_all.py` runs every step and
regenerates both output files).

---

## 2. Candidate Generation / Blocking Strategy

Blocking runs **per country** (training matches never cross countries). The country is an open set
of strings taken from the data, so France (absent from training) is blocked exactly like US/India.

| View | Text | Analyzer | Top-K per S1 | Reverse (record → S1) |
|---|---|---|---|---|
| `words2` | core name (tokens sorted) + address + state | word 1-2 grams | 20 | top-3 |
| `addr2` | address | word 1-2 grams | 12 | — |
| `name4` | core name, spaces removed | char 4-grams | 12 | top-5, cos ≥ 0.3 |
| `comb4` | sorted core name + address | char 4-grams | 10 | top-3, cos ≥ 0.2 |

- **Why several views:** word pairs ("173 orchard") find the same business at the same address;
  character 4-grams survive typos inside words ("brrton st", "saint pual") and domain-style names;
  the address view finds records renamed to an unrelated brand.
- **Reverse search:** each record also retrieves its own top S1 entities, which recovers records
  whose S1's top-K list is crowded by look-alike records.
- **Transliteration:** names written in Indic scripts become ASCII via anyascii ("skti bildrs");
  a token table learned from training matches (+ a consonant-skeleton fallback) maps them back to
  Latin words ("shakti builders") before blocking.
- **Why it scales:** hashed TF-IDF (2²² buckets, no vocabulary in memory) tokenised in worker
  processes; features present in more than 3,000–5,000 documents are dropped from the product
  (they carry almost no identity and dominate its cost); the top-K product is a multithreaded
  sparse matrix product (`sparse_dot_topn`) in row chunks. Nothing compares all records with all records.
- **Final candidate set (`candidate_pairs.tsv`)** = exactly the pairs the stage-2 model scores
  (optionally trimmed to the top-N by `p1`, see `BER_CAND_TOP_N`; the trim report printed by
  `ber.model stage2` gives out-of-fold F0.5, candidates per S1 and candidate recall for each setting).

| Metric (full run) | Value |
|---|---|
| Candidates per S1, test (mean) | _from the run log_ |
| Pair recall of the candidate set (training, out-of-fold) | _from the run log_ |
| Reduction ratio (test) | 1 − candidates / (S1 × (S2 + S3)) ≈ 0.999999 |

---

## 3. Model Architecture & Feature Engineering

### Models
- **Stage 1 and stage 2:** LightGBM binary classifiers (GBDT, 127 leaves, learning rate 0.08,
  L2 1.0, early stopping), 3 folds grouped by S1 entity (fold = hash of the S1 id). Fitted on a
  deterministic hash-based subset of training entities (`BER_TRAIN_FRAC`, default 0.6) for memory.
- **Cross-encoder:** `intfloat/multilingual-e5-base` (MIT, 278M parameters) fine-tuned as a pair
  classifier on "name | address" of both sides (raw text, original scripts), 1 epoch, max length 96,
  on uncertain pairs (0.0005 ≤ `p1` ≤ 0.995) of a reserved pool of 20 % of training entities that
  the LightGBMs never see, so its scores are out-of-sample wherever they are used.

### Features (country is never a feature)

| Group | Features |
|---|---|
| TF-IDF | exact cosine per view (`name3`, `name4`, `skel3`, `words`, `words2`, `addr`) for every pair, blocking rank |
| Context | rank / gap to best / group size of each cosine and key similarity within the S1's candidates and within the record's S1s |
| Name | ratio, token-set, token-sort, partial, Jaro-Winkler on full / core / compact / consonant-skeleton / transliterated forms; token Jaccard and containment |
| Differing words | number and IDF (rarity) of words present on one side only — noise swaps common words, a different business differs in a rare word |
| Structure | legal-form agreement/conflict, state agreement/conflict, first house number equal / log difference, Roman-numeral / number markers ("project ii" vs "iii"), acronyms, filler words added |
| Record flags | compact/domain name, DBA, Indic script in name/address, empty address, source S3, share of name words unseen in Source 1, how many S1 entities share the name |
| Stage 2 only | `p1`, best `p1` of the record with any other S1 and the margin over it, ranks, confident siblings, cluster support (best name/address similarity to the entity's confident records), cross-encoder score and its gaps |

---

## 4. Other Relevant Information

### Handling the unseen country (France)
- No country is hard-coded, filtered or one-hot encoded; blocking groups by whatever country strings exist.
- Normalisation is language-agnostic (accent stripping, French legal forms SARL/SAS/SASU/EURL/SCI/SNC,
  French address words rue/allée/impasse/chemin/route/boulevard).
- France has no state table, so its pairs always show "state unknown". Training augmentation blanks
  the state features for 15 % of training entities so the model learns to rely on other evidence.
- Test contains more records without a Source 1 entity than training (~5.8 vs 4.7 records per S1);
  19 % of training S1 entities are dropped (their records kept) to reproduce test-like conditions for
  the competition features and the decision threshold.

### Decision rule
Chosen on out-of-fold stage-2 probabilities with the exact challenge metric (true matches missed by
blocking count as misses): thresholds 0.3–0.9, or per-S1 expected-F0.5 selection (keep the top-k pairs
maximising 1.25·Σp / (0.25·(Σp + miss) + k), k may be 0, against Π(1 − p) for predicting nothing).
Before either, every record is kept only for its most probable S1 entity.

### Models used
One pretrained model: `intfloat/multilingual-e5-base`, MIT licence, 278M parameters
(`code/business_entity_resolution/MODELS.md`). No external data, APIs or services.

---

## 5. Reproducibility

- Fixed seeds; folds, subsets and the cross-encoder pool are hash-based on entity ids.
- Pinned `requirements.txt`.
- From the submission root, with the data in `dataset/train` and `dataset/test`:
  ```bash
  pip install -r code/business_entity_resolution/requirements.txt
  cd code/business_entity_resolution/src && python run_all.py
  ```
  writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs the validator.
  Needs a CUDA GPU for the cross-encoder steps and a large-memory machine for the full data.

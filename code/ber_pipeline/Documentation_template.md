# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** AiseKaise
**Team Members:** [List all team members]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve Source 2/3 records to Source 1 entities with a five-step pipeline:

1. **Rule-based normalisation** that undoes the observed noise (script changes, legal-suffix moves,
   DBA wrappers, domain-style names, rotated addresses).
2. **Multi-view TF-IDF blocking** with word-pair and character 4-gram features, forward and reverse.
3. **A first LightGBM pair classifier** over ~110 string, overlap, name-uniqueness and competition
   features.
4. **A fine-tuned multilingual transformer cross-encoder** (`intfloat/multilingual-e5-base`, MIT,
   278M parameters) that scores the uncertain pairs from their raw text.
5. **A second LightGBM** that re-scores every pair using the first-stage probabilities in context
   (competition for the record, the entity's other confident records) and the cross-encoder score,
   followed by a **decision layer** that assigns every record to at most one Source 1 entity.

Key ideas:
- a transliteration token table learned from training matches ("praibhet limitet" → "private
  limited", "phuds" → "foods"), used in both blocking and features;
- word-pair TF-IDF that stays discriminative when every single word is common;
- exploiting "each record belongs to at most one Source 1 entity";
- training under test-like conditions: the test set has ~23% more records per Source 1 entity, so
  19% of training entities are removed (keeping their records) to reproduce the larger share of
  records without a true match.

**Result:** macro F0.5 **0.9868** out of fold under test-like conditions (Section 5).

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale:** train 2.2M S1 / 10.3M S2+S3 records; test 1.73M S1 / 9.97M records.
- **Source 1 is clean** (no accents, no other scripts, no empty addresses); all noise is in Sources 2/3.
- **No training match crosses countries** (0 of 7.64M), so all work is per country.
- **Every S2/S3 record matches at most one S1 entity** (0 of 7.64M matched IDs are shared). About 26%
  of S2/S3 records match nothing (decoys).
- Only 5.6% of S1 entities are singletons; the mean is ~3.5 matches per entity.
- **Train/test shift:** training has 4.7 S2/S3 records per S1 entity, test has 5.5–5.8 in every
  country, while predicted matches per entity stay at ~3.3. So test has a larger share of records whose
  entity is absent from Source 1 (~40% vs 26%). A model validated without this over-estimated the
  leaderboard (out-of-fold 0.9785 vs public leaderboard 0.971 for v2).
- **Name noise:** case, junk prefixes (`--`, `<<`, `***`, `@`), honorifics (Smt, Shri, Dr), legal
  suffixes moved / bracketed / swapped / duplicated, filler words appended (Center, Services, Group,
  Partners), DBA / f/k/a / "formerly" wrappers with an invented brand, names reduced to domains or
  handles (`premierprojects.com`, `@TRUSTEDPRIME`), digit-for-letter swaps (`Civic0n`), injected
  accents, word reordering and duplication, typos, and for India 18% of names written fully or partly
  in Devanagari, Telugu, Gujarati, Gurmukhi, Kannada, Bengali or Tamil script.
- **Address noise:** component rotation, state name ↔ code ↔ native-script name, null placeholders
  (NULL, N/A, `<NULL>`), added prefixes (H.NO, DOOR NO, ##), PO Box / PMB additions, house-number
  perturbations (0666, 3996C, 210-212, 656→56), inserted random localities, missing components, 3% empty.
- **Hard negatives:** decoys that copy an entity's address with a near-identical name ("frontier
  project ii" vs "iii", PC vs LLC, "group" inserted), and name collisions across different addresses
  (the name vocabulary is small).
- **France (test only)** shows the same noise families with French legal forms (SARL, SAS, SASU, EURL,
  SCI, EI), street abbreviations (R, AV, BD, ALL, IMP, CH), N°/BIS/TER numbering and region ↔
  département variation.

### 2.2 Solution Strategy

**Approach Type:** Hybrid — blocking + gradient-boosted pairwise classifier stacked with a transformer
cross-encoder + constrained assignment.
**Core Innovation:** learned transliteration table; word-pair TF-IDF blocking under a frequency cap;
one-owner-per-record assignment with competition features; country-agnostic features with
state-blanking augmentation for the unseen country.

```
raw TSVs → normalise → blocking (4 TF-IDF views, forward + reverse, per country)
  → pair features → LightGBM stage 1 (p1) → cross-encoder on uncertain pairs (GPU)
  → stage-2 context features → LightGBM stage 2 → one-owner assignment + decision rule
  → output/matching_results.tsv, output/candidate_pairs.tsv
```

---

## 3. Candidate Generation (Blocking)

Each country is blocked separately; the country is an open set of strings taken from the data, so
France is blocked exactly like US and India. Four TF-IDF views retrieve, for every S1 entity, its
top-K most similar records by cosine (multithreaded sparse top-k product, `sparse_dot_topn`); the
candidate set is the union of all views.

| View | Text | Features (frequency cap) | Top-K per S1 | Reverse search |
|---|---|---|---|---|
| `words2` | sorted core-name words + normalised address + state | word unigrams + bigrams (5,000) | 20 | top-3 |
| `addr2` | normalised address | word unigrams + bigrams (5,000) | 12 | — |
| `name4` | core name, spaces removed | character 4-grams (5,000) | 12 | top-5, cosine ≥ 0.3 |
| `comb4` | sorted core-name words + normalised address | character 4-grams (3,000) | 10 | top-3, cosine ≥ 0.2 |

- `comb4` tolerates typos inside words ("brrton st", "saint pual st", "asswaoman") that break the
  word-level views.
- The **reverse search** retrieves, for every record, its most similar Source 1 entities: by
  name+address (`words2`, `comb4`) and by name alone (`name4`, for records without an address or with
  compact / domain-style names). It finds records whose entity's own top-K list is crowded by
  look-alike records, and supplies hard negatives for records whose entity is absent.
- Record names pass through the learned **transliteration table** (Section 4) before blocking, so an
  Indic-script name ("skti bildrs" → "shakti builders") can be retrieved by name and not only by
  address. On a 3% sample this raised pair recall from 0.957 to 0.965 and the F0.5 ceiling from 0.984
  to 0.988 at the same number of candidates (a third of the pairs blocking still missed before were
  transliterated names).
- **Frequency cap:** features that occur in more than 5,000 records (3,000 for `comb4`) are dropped.
  This bounds the cost of the product (cost ∝ posting-list lengths; a 50k → 5k cap was ~25× faster)
  and loses little, because such features carry little identity.
- **Word pairs are the key:** "173 orchard", "dental preferred" remain rare when every individual word
  is common; name words are sorted first so pairs do not depend on word order. Word pairs raised
  single-view recall at K=10 from 0.886 (unigrams, 20k cap) to 0.929 (5k cap).
- TF-IDF uses a parallel hashing vectoriser (2^22 buckets, no vocabulary in memory) with smooth IDF,
  sublinear TF and L2 normalisation. Nothing is ever compared all-against-all.

**Blocking keys used:** TF-IDF over word unigrams+bigrams of name+address, of the address alone, and
character 4-grams of the compacted name and of name+address.

**Candidate pairs generated:**
- Train: 129,738,095 (India 51.5M, US 78.3M, after removing 19% of entities).
- Test: 122,089,836 (France 18.8M, India 57.1M, US 46.2M) — **~71 per S1 entity**. Every test entity
  has at least one candidate.
- Reduction ratio (test): 1 − 122.1M / (1.73M × 9.97M) ≈ 0.999993.
- `candidate_pairs.tsv` is exactly the set the stage-2 model scores. The pipeline can trim it to the
  top-N pairs per entity by stage-1 probability (`BER_CAND_TOP_N`, `BER_CAND_MIN_P1`); the trim report
  printed by `ber.model stage2` gives out-of-fold F0.5, candidates per S1 and candidate recall for each
  setting. The submitted run used no trimming.

**How true matches were kept:** views were chosen by measured recall on a held-out sample of training
entities queried against the FULL record pool (so distractor density is realistic). On the 858,726
training entities used for fitting:

| Version | Pair recall | Entities with all matches retrieved | F0.5 ceiling (perfect matcher) |
|---|---|---|---|
| v2 | 0.964 | 0.898 | 0.988 |
| v3 | 0.979 | 0.936 | 0.993 |
| **v5 (submitted)** | **0.982** | **0.947** | **0.994** |

---

## 4. Matching Model

### Normalisation (before everything)
- **Names:** `anyascii` transliteration; ID tags, junk punctuation and handles removed; DBA wrappers
  split (the name after "dba / f/k/a / formerly" is kept); initials joined (L.L.C. → llc); domain
  suffixes stripped; digit-for-letter repair inside words; legal forms canonicalised and separated
  (US, Indian and French forms); filler words and honorifics removed from a "core" name; a consonant
  skeleton per word (lines up transliterated and Latin spellings).
- **Addresses:** null placeholders and PO Box / PMB removed, street types canonicalised (English and
  French), ordinals and zero-padding removed, state names mapped to codes (US and India tables, plus
  native-script state names learned from training matches).

### Transliteration table
For each word of an Indic-script record name, the Source 1 name word it co-occurs with most across
training matches (cosine score, ≥ 20 occurrences, ≥ 50% purity): 563 mappings such as `phuds→foods`,
`tredimg→trading`, `pmpay→bombay`, `civm/sivm→shivam`, `praibhet→pvt`. Words not in the table (rarer
personal and place names) fall back to the Source 1 word with the same consonant skeleton when one word
clearly dominates it (`kubr→kuber`, `pratapgrh→pratapgarh`).

### Features used (~110; country is never a feature)
- **Name:** RapidFuzz ratio / token-set / token-sort / partial / Jaro-Winkler on full, core, compact,
  skeleton and translated names; token Jaccard and containment; legal-form agreement / conflict;
  roman-numeral and digit conflicts; filler words inserted; share of record-name words unseen in any
  Source 1 name (invented brands); compact / DBA / script flags.
- **Address:** fuzzy ratios, token Jaccard / containment, number-set overlap, first-number equality and
  log distance, state agreement / conflict, empty flags.
- **Initials:** record name equal to, or starting with, the initials of the Source 1 name (records
  reduced to `jcqdro`, `rp`).
- **Word substitution:** for the words present in only one of the two core names, their count and the
  max / sum of their IDF over the split's Source 1 names. The noise swaps a name word for another
  common word ("office" → "service", "sport" → "loisirs"); a different business differs in a rare
  word. Computed from each split's own Source 1 names, so it applies to French vocabulary without
  labels (`n_r_only` became a top-10 stage-1 feature).
- **Name uniqueness:** number of Source 1 entities in the country with the same core name / consonant
  skeleton as the record, and with the same core name as the S1 entity. A name unique among Source 1
  is safe to match on name alone (records without an address were 62% of the rejected true matches);
  a generic name like "tech foods" is not.
- **Other:** exact TF-IDF cosine for six views (IDF-weighted overlap) and blocking ranks;
  **competition context** — rank and gap-to-best of each score within the S1 entity's candidates and
  within the record's candidate S1 entities, and group sizes.

### Training data
19% of training S1 entities are removed (deterministic hash) while all their records are kept, so the
training candidate sets contain the same share of records without a true entity as the test set (5.8
records per entity). Their records become negatives for every entity they resemble, and all competition
features, the classifiers and the decision rule are fitted under these conditions. State agreement is
blanked for 15% of training entities, because the unseen country has no state table.

### Model type
Two-stage LightGBM plus a transformer cross-encoder; 3-fold cross-validation grouped by S1 entity,
early stopping on log-loss. Of the remaining training entities, 20% are reserved for the
cross-encoder; the LightGBM models are fitted on a deterministic 60% of the other 80% (858,726
entities, 49.4M candidate pairs).

- **Stage 1** uses the pair features above. Its probability p1 is then produced for every training
  pair by the fold model that never saw that entity, and for every test pair as the mean of the fold
  models.
- **Cross-encoder:** `intfloat/multilingual-e5-base` (MIT licence, 278M parameters, far below the 8B
  limit) fine-tuned as a pair classifier on the RAW texts "name | address" of both sides, so it reads
  the original scripts and learns typo, transliteration, abbreviation and decoy patterns directly.
  - Trained (1 epoch, bf16, max 96 tokens) only on candidate pairs of the reserved 20% of entities:
    uncertain pairs (0.0001 ≤ p1 ≤ 0.999) plus a 3% sample of the rest. Its scores wherever LightGBM
    uses them are therefore out-of-sample.
  - Scores only uncertain pairs (0.0005 ≤ p1 ≤ 0.995) of the LightGBM training entities and of the
    test set; elsewhere the feature is missing, identically for training and test.
  - In the submitted run it trained on 1,693,654 pairs (14.7% positive; held-out log-loss 0.058,
    accuracy 0.977) and scored 2.84M training and 3.80M test pairs. It is the third most important
    stage-2 feature after the stage-1 probability and its margin. (With the smaller
    `multilingual-e5-small` in v4: log-loss 0.077, accuracy 0.968.)
- **Stage 2** adds p1 in context: the record's best p1 with any other entity and the margin over it,
  ranks of the pair within the record's and the entity's candidates, the number and total p1 of the
  entity's other confident records, and **cluster support** — the best name / address similarity
  between the record and the entity's confident records (p1 ≥ 0.8) plus exact-name agreement, which
  helps address-less records that share a name with a confidently matched sibling. It also receives
  the cross-encoder score and its gap to the best cross-encoder score of the same record and of the
  same entity. Stage 2 uses the same folds.

The only pretrained model is the MIT-licensed cross-encoder backbone above; nothing is looked up
externally.

### Threshold selection method
Out-of-fold stage-2 probabilities → **one-owner assignment** (each record kept only for its most
probable S1 entity) → either keep pairs with p ≥ t, or keep, per S1 entity, the top-k pairs maximising
the expected F0.5,

  E[F0.5 | top k] ≈ 1.25 · Σ_{i≤k} p_i / (0.25 · (Σ_all p + miss) + k),

against E[F0.5 | predict nothing] = Π (1 − p_i) (k may be 0). The rule and its parameter (t, or
`miss` = expected true matches not retrieved by blocking) are chosen by maximising the exact macro
F0.5 on out-of-fold predictions (true matches missed by blocking count as misses). The submitted run
uses expected-F0.5 top-k selection.

---

## 5. Results & Error Analysis

**F0.5 score (macro): 0.9868** out-of-fold on 858,726 training entities under test-like conditions
(19% of entities removed; 62.3M candidate pairs; 3 folds grouped by entity; stage-2 validation
log-loss 0.0024). Decision: one-owner assignment + expected-F0.5 top-k selection.

### Progression (full scale)

| Version | Validation | Blocking pair recall | F0.5 ceiling | Macro F0.5 | Public LB |
|---|---|---|---|---|---|
| v1: single LightGBM | all entities | 0.956 | 0.984 | 0.9724 | – |
| v2: + transliteration in blocking, name uniqueness, stage 2 | all entities | 0.964 | 0.988 | 0.9785 | 0.971 |
| v3: + reverse search, larger K, test-like training (19% removed), 50% fit | 19% removed | 0.979 | 0.993 | 0.9823 (stage 1: 0.9807) | 0.9781 |
| v4: + cross-encoder, name-only reverse search, skeleton transliteration, initials features | 19% removed | 0.981 | 0.994 | 0.9865 (stage 1: 0.9811) | 0.983 |
| **v5: + typo-tolerant `comb4` view, word-substitution features, e5-base cross-encoder** | 19% removed | **0.982** | **0.994** | **0.9868** (stage 1: 0.9815) | [fill] |

v2's validation did not contain test's larger share of records without an entity, which is why it
over-estimated the leaderboard; from v3 on, validation uses test-like conditions and tracks the
leaderboard closely.

### Test output
1,630,721 of 1,732,544 entities matched; 101,823 (5.9%) predicted as singletons (5.6% singletons in
training). Official validator (with `--check-ids`): **PASS**.

### Error analysis
- **v1 (motivated v2):** blocking misses cost 0.016 F0.5 and a third of them were transliterated names
  (7% of true pairs); rejected true candidates cost 0.009 and 62% of them had no address; false merges
  cost 0.003 (43% were records of a sibling entity, 57% planted decoys). India 0.9625 vs US 0.9790.
- **Loss decomposition (development run):** blocking misses −0.016, rejected true candidates −0.012,
  false merges −0.006. Singletons: 105 of 3,728 predicted non-empty.
- **v4 (out of fold):** the largest remaining loss (0.0033 F0.5) is address-less records whose name is
  shared by 2+ Source 1 entities; the full name including the legal form resolves only 12% of them, so
  most are genuinely ambiguous and predicting them lowers expected F0.5. Blocking misses cost 0.006
  (typos inside words, address fragments, transliterated names), false merges 0.002.
- **Common false positives (wrong merges):** planted near-duplicates at the same address — roman
  numerals (ii/iii), legal form (PC/LLC), inserted "group/holdings"; records of a sibling Source 1
  entity (resolved at full scale by the one-owner rule, since the sibling competes for the record).
- **Common false negatives (missed matches):** records renamed to an invented brand ("quodelta") with a
  partial address; empty-address records with generic names; heavily transliterated names; house
  numbers perturbed (169 vs 171).

### Unseen country (France)
- No country is hard-coded, filtered or one-hot encoded; features are country-agnostic.
- Cross-country transfer as a proxy: train India → test US 0.961 vs 0.974 in-country.
- Hiding state agreement changes F0.5 by ~0.001, so the lack of a French region table is not a risk;
  training additionally blanks state agreement for 15% of entities.
- The word-substitution IDF and name-uniqueness counts are computed from each split's own Source 1
  names, so they cover French vocabulary without labels.

### Caveat
The transliteration table is learned from all training matches, so the out-of-fold scores for
Indic-script records may be slightly optimistic; the table only maps generic business-name words.

---

## 6. Conclusion

Careful normalisation plus IDF-weighted word-pair blocking (forward and reverse, word- and
character-level) over transliteration-corrected names recovers 98.2% of true pairs at ~71 candidates
per entity. A two-stage gradient-boosted classifier with competition and cluster-support features, a
fine-tuned multilingual cross-encoder for uncertain pairs and a one-owner constraint reaches 0.9868
macro F0.5 out of fold under test-like conditions. The biggest levers were data-driven undoing of the
noise (transliteration table, legal / filler handling), exploiting the one-record-one-owner structure,
and matching the training conditions to the test set's share of records without an entity.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/`:

| File | Role |
|---|---|
| `run_all.py` | runs every step in order (separate processes; `--from <step>` resumes) and the validator |
| `ber/config.py` | paths and settings (environment variables `BER_DATA_DIR`, `BER_WORK_DIR`, `BER_OUTPUT_DIR`, `BER_THREADS`) |
| `ber/io.py` | TSV reading (no quoting, all text), parquet caches, training-entity subsets |
| `ber/normalize.py`, `ber/prepare.py` | normalisation of names and addresses, state aliases |
| `ber/translit.py` | learned transliteration table + consonant-skeleton fallback |
| `ber/tfidf.py`, `ber/blocking.py`, `ber/candidates.py` | parallel hashed TF-IDF, top-K blocking, candidate tables |
| `ber/features.py`, `ber/featurize.py` | pair features |
| `ber/model.py` | stage-1 / stage-2 LightGBM, decision tuning, test prediction, trim report |
| `ber/crossenc.py` | cross-encoder fine-tuning and scoring (GPU) |
| `ber/stage2.py` | stage-2 context, cluster-support and cross-encoder features |
| `ber/decide.py`, `ber/metrics.py`, `ber/submit.py` | decision rules, exact macro F0.5, output files |
| `ber/eval_blocking.py` | optional blocking-recall diagnostic |

**Reproduce** (from the submission root, with the data in `dataset/train` and `dataset/test`):

```bash
pip install -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution/src
python run_all.py
```

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs the validator when
it is found. Fixed seeds; folds, subsets and the cross-encoder pool are hash-based on entity ids;
pinned `requirements.txt`. The cross-encoder steps need a CUDA GPU (without one they are skipped and
the pipeline still runs). The submitted outputs were produced on a Colab G4 runtime (48 vCPU, 176 GB
RAM; RTX PRO 6000 GPU for the cross-encoder) in about 2 h 35 min end to end.

**Models:** one pretrained model, `intfloat/multilingual-e5-base` (MIT, 278M parameters); see
`code/business_entity_resolution/MODELS.md`. Libraries: polars, scikit-learn, sparse_dot_topn,
rapidfuzz, anyascii, LightGBM, PyTorch, transformers. No external data, APIs or services.

### B. Additional Results

Blocking recall by view and K on a 2% training sample (5k cap, word pairs):

| Views | K | Candidates / S1 | Pair recall | F0.5 ceiling |
|---|---|---|---|---|
| words2 | 10 | 10.0 | 0.929 | 0.976 |
| words2 | 20 | 20.0 | 0.953 | 0.984 |
| words2 + addr2 + name4 | 10 | 20.8 | 0.950 | 0.983 |
| words2 + addr2 + name4 | 15 | 32.4 | 0.960 | 0.986 |

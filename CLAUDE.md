# CLAUDE.md — Amazon ML Challenge 2026: Business Entity Resolution

This file gives any AI assistant (or new teammate) the full context for this repo.
**Read it fully before writing or changing code.**

---

## 1. The task (one paragraph)

We receive business records from **3 independent sources**. Each record has
`entity_id`, `business_name`, `business_address`, `country`. **Source 1 (S1) is the
deduplicated reference list.** For **every S1 entity**, predict the list of **Source 2 (S2)
and Source 3 (S3)** records that refer to the same real-world business.
An S1 entity may match **zero, one, or many** S2/S3 records. There are no shared IDs —
matching must come from noisy names/addresses.

- Source is given by the ID prefix: `S1-`, `S2-`, `S3-` (no separate source column).
- **Train countries: US, India. Test adds France (unseen in training).**
  Treat `country` as an **open set of strings**. Never hard-code, filter, or one-hot to
  `{US, India}`. Every test S1 entity (France included) must appear in output.

### Noise to expect
- **Names:** abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited), legal suffix
  differences, DBA/trade names, `&` vs `and`, punctuation, word-order swaps, typos,
  transliterations.
- **Addresses:** abbreviations (Rd/Road, St/Street), transliteration variants, missing
  components (no PIN/state), landmark references ("Near SBI ATM"), municipal numbering
  formats, reordered components.

---

## 2. Data

**All files are TAB-separated (.tsv). Always read with `sep="\t"`.** Without it pandas
silently produces one column. Addresses and ID lists contain commas.

```python
df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
```

| File | Contents |
|---|---|
| `dataset/train/train_source{1,2,3}.tsv` | training records |
| `dataset/train/train_ground_truth.tsv` | `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 IDs, empty if no match) |
| `dataset/test/test_source{1,2,3}.tsv` | test records — predict for every row of `test_source1.tsv` |

No test ground truth. Use a held-out validation split of train.

**`dataset/` is never committed to git** (see `.gitignore`).

---

## 3. Evaluation — F0.5, macro-averaged per S1 entity

```
F_0.5 = (1.25 * P * R) / (0.25 * P + R)
```
- Computed **per S1 entity**, then averaged over all S1 entities.
- **Singletons count:** true-empty + predicted-empty → **1.0**; true-empty + any prediction → **0.0**.
- Precision is weighted 2× over recall → **false merges are expensive. When unsure, predict nothing.**
- Public leaderboard = subset of test; **final ranking = private leaderboard** (rest of test).
  Do not overfit to public LB; trust local validation.

Local scorer lives in `src/evaluate.py`. Edge cases to implement exactly:
- true empty & pred empty → 1.0
- true empty & pred non-empty → 0.0
- true non-empty & pred empty → 0.0
- otherwise standard P/R → F0.5 (0 if no overlap)

---

## 4. Required outputs

Both go in `output/`, tab-separated, header row exactly as below, IDs comma-joined with
**no spaces and no quoting**.

### `matching_results.tsv` (scored; uploaded to the leaderboard)
```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003
```

### `candidate_pairs.tsv` (not scored; audited for blocking recall / reduction ratio)
```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
```
Must be the **final candidate set the model actually scores** (the last filtering stage),
not an earlier raw blocking pass.

### Hard rules (violations = rejection)
- Exactly **one row per test S1 entity**, no duplicate `source1_entity_id` rows.
- Empty list for no matches.
- Only **S2-/S3-** IDs that **exist in the test set**. No S1 IDs, no self-matches.
- No duplicate IDs within a list.
- Every matched ID must also appear in `candidate_pairs.tsv` (matches ⊆ candidates).

### Always validate before uploading
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
`PASS` (exit 0) = safe. Never spend a submission on an unvalidated file.

---

## 5. Rules that can disqualify us — NEVER violate

1. **No external data lookup.** No geocoding APIs, business-registry/government DBs,
   commercial ER APIs, web scraping, or internet data augmentation. Only the provided data.
   (Hand-written normalization dictionaries — e.g. `rd → road`, `sarl` legal suffix — are fine.)
2. **Model license:** any pretrained model in the final pipeline must be **MIT or Apache 2.0**
   and **≤ 8B parameters**. Check the license on the Hugging Face card before using it and
   record it in `MODELS.md`.
   - OK examples: `intfloat/multilingual-e5-small|base` (MIT), `sentence-transformers/all-MiniLM-L6-v2` (Apache 2.0),
     `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache 2.0).
   - Avoid anything CC-BY-NC, Llama-license, Gemma-license, or unknown license.
   - Libraries (LightGBM, scikit-learn, rapidfuzz, faiss) are fine.
3. **No cheating / plagiarism.** Repo stays **private** until results are out.
4. Only the **team leader** can open the Unstop assessment and upload. One device, one login —
   never log in to Unstop from two devices simultaneously.

---

## 6. Timeline & submission budget

- Window: **25 Sep 2026 12:00 AM IST → 27 Sep 2026 11:59 PM IST**.
- **Max 5 leaderboard submissions per day** (3 days). Use them daily; keep 1–2 for the final model on day 3.
- Queries → Google Form linked in the guidelines. Tech issues → support@unstop.com.
- Results / Top-100 announcement later; top teams' packages are reproduced and audited.

Log **every** submission in `submissions/LOG.md` (see §9).

---

## 7. Approach (pipeline)

```
load → normalize → blocking (candidates) → pair features → classifier → threshold → write outputs → validate
```

1. **Normalize** (`src/normalize.py`)
   - lowercase, unicode NFKD + strip accents (important for French), `&` → `and`, strip punctuation, collapse spaces.
   - expand/canonicalize abbreviations: names (pvt→private, ltd→limited, corp→corporation, co→company, inc, llc…)
     and addresses (rd→road, st→street, ave/av→avenue, blvd→boulevard, nagar, marg…, French: bd, av, r→rue, ch→chemin).
   - produce a `name_core` with legal suffixes removed (pvt ltd, llc, inc, corp, sarl, sas, sa, eurl, gmbh, etc.).
   - extract structured bits from addresses: postal codes (India 6-digit PIN, US 5-digit ZIP, France 5-digit CP),
     house/street numbers, tokens.
   - Keep dictionaries generic and multilingual; don't special-case by country name in ways that break on unseen countries.

2. **Blocking** (`src/blocking.py`) — determines the **recall ceiling**.
   - Union of several cheap blockers, top-K each, per S1 record against S2 and S3 separately:
     - TF-IDF char n-grams (3–4) on `name_core` → nearest neighbours
     - TF-IDF on name+address
     - (optional) multilingual sentence-embedding kNN
     - exact postal-code / first-token keys
   - Block **within the same country string** when country is present; fall back to global if country missing.
   - Track on validation: **pair recall** (true pairs kept / all true pairs) and **reduction ratio**.
     Target recall ≥ 0.97 with a manageable candidate count (tune K).

3. **Pair features** (`src/features.py`) — for each (S1, candidate):
   - name: rapidfuzz `ratio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio`, Jaro-Winkler, TF-IDF cosine, Jaccard on tokens, on both raw-normalized and `name_core`.
   - address: same similarity set, postal-code equal / both-present flags, number overlap, token overlap.
   - embedding cosine (name, address) if embeddings used.
   - blocking rank / score, candidate source (S2 vs S3), length features, missing-field flags.
   - **Country-agnostic features only** (no one-hot country) so France generalizes.

4. **Model** (`src/train.py`) — LightGBM binary classifier on pairs (label = pair in ground truth).
   - Group-aware split: split by **S1 entity**, not by pair.
   - Keep all S2/S3 records in the candidate pool for validation (realistic difficulty).

5. **Decision** (`src/predict.py`)
   - Keep candidates with prob ≥ threshold; **tune threshold to maximize local macro F0.5** (expect a high value).
   - Optional post-processing to test on validation: per-S1 top-N cap, requiring a minimum name similarity,
     resolving an S2/S3 record assigned to several S1s (S1 is deduplicated → an S2/S3 record should usually
     belong to at most one S1; keep the highest-prob S1).

6. **Write outputs** + run validator.

### Ideas to try once baseline works
- Better abbreviation/transliteration normalization; phonetic keys (Double Metaphone) for Indian names.
- Cross-encoder reranker on top candidates (license-checked, ≤8B).
- Per-source thresholds (S2 vs S3) if validation supports it.
- Check calibration on a pseudo-"unseen country" by holding out one train country.

---

## 8. Repo structure

The repo mirrors the **final submission zip**, so packaging is trivial.

```
amazon-ml-challenge-2026/
├── CLAUDE.md                      # this file
├── PLAN.md                        # team plan: roles, git workflow, schedule, upload checklist
├── README.md                      # short overview for humans
├── .github/pull_request_template.md  # every PR reports main vs branch validation numbers
├── .gitignore                     # dataset/, *.zip, __pycache__, .ipynb_checkpoints, models cache
├── dataset/                       # NOT committed — train/ and test/ TSVs from student_resource
├── utils/
│   └── validate_submission.py     # provided by organizers (do not modify)
├── notebooks/                     # EDA / experiments (Kaggle/Colab), prefixed with name: 01_eda_<name>.ipynb
├── output/                        # latest matching_results.tsv + candidate_pairs.tsv
├── submissions/
│   ├── LOG.md                     # every leaderboard upload (see §9)
│   └── <tag>/                     # copy of each uploaded matching_results.tsv + candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── README.md              # exact end-to-end run instructions
│       ├── requirements.txt       # pinned versions
│       ├── MODELS.md              # every pretrained model used + license + param count
│       └── src/
│           ├── config.py          # paths, K values, thresholds, seeds
│           ├── io_utils.py        # TSV read/write (sep="\t", dtype=str)
│           ├── normalize.py
│           ├── blocking.py
│           ├── features.py
│           ├── train.py
│           ├── predict.py
│           ├── evaluate.py        # macro F0.5 + blocking recall
│           └── run_pipeline.py    # data → blocking → matching → output/
└── Documentation_template.md      # organizers' template, filled in as we go
```

Packaging the final zip:
```bash
zip -r <team_name>_submission.zip output code Documentation_template.md
```
(Exclude `dataset/`, `notebooks/`, `submissions/`.)

### Running
```bash
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/run_pipeline.py --mode validate   # local F0.5 on held-out train
python code/business_entity_resolution/src/run_pipeline.py --mode test       # writes output/*.tsv
python3 utils/validate_submission.py --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

---

## 9. Conventions

- **Reproducibility:** fixed seeds in `config.py`; pinned `requirements.txt`; the pipeline must regenerate
  both output files from raw data using only `code/business_entity_resolution/`.
- **Code quality:** docstrings/comments on every function (explicitly required by organizers).
- **IDs are strings.** Read with `dtype=str, keep_default_na=False` so empty lists stay `""`, not NaN.
- **Git:** commit after every experiment; one commit + tag per leaderboard submission:
  `git tag sub-d<day>-<n>` (e.g. `sub-d1-2`), commit message includes local val F0.5 and LB score.
- `submissions/LOG.md` format:

  | tag | date/time IST | commit | local val F0.5 | blocking recall | public LB | notes |
  |---|---|---|---|---|---|---|

- **First submission idea:** an all-empty `matching_results.tsv` → public score ≈ singleton rate. Useful calibration.
- Before any upload: run local validation → run `validate_submission.py` → log it → leader uploads.

---

## 10. Methodology doc (fill `Documentation_template.md` as we go)

Must cover: methodology; candidate generation / blocking strategy (with recall + reduction ratio numbers);
model architecture & feature engineering; other relevant info (normalization, threshold choice,
handling unseen country France, models used + licenses). No page limit — clarity and depth.

---

## 11. Team

- Team size 3. Only the **team leader** submits on Unstop.
- Work is split by idea, not by file: see PLAN.md §5 (shared idea list, twice-daily merges, stacking rule —
  every merge is re-validated on top of the current `main`). Leader = merges, submissions, repo, documentation.
- Compute: Kaggle Notebooks / Google Colab (free GPU). No paid or external data services.

# Team Plan — Amazon ML Challenge 2026 (Team AiseKaise)

Read this first, then `CLAUDE.md` (full task details) and the official guidelines PDF.
Window: **25 Sep 00:00 IST → 27 Sep 23:59 IST**. Max **5 leaderboard uploads per day** (15 total).

---

## 1. Where we are

The full baseline pipeline is in `code/business_entity_resolution/src/`, rebuilt for the real data size
(**~12M records per split**: S1 ~2M, S2 ~5M, S3 ~5M — see CLAUDE.md §2):

```
normalize (parallel, cached)
  → stage 1 blocking   hashed TF-IDF on name / name+address, top-K per S1, same country   (~20-40 per S1)
  → stage 2 pruning    light LightGBM keeps the best few per S1 → candidate_pairs.tsv      (~1-10 per S1)
  → stage 3 matching   ~70 features → LightGBM → threshold tuned for macro F0.5 → matching_results.tsv
```

- **Two numbers matter now** (official portal update): macro F0.5 on the leaderboard **and** how small
  `candidate_pairs.tsv` is per S1 — smaller candidate sets rank higher in the final evaluation.
  Every run prints both, plus candidate recall after each stage.
- Runs end to end on fake data and on a small slice of the real train data; the output passes the
  organizers' validator. **Full-size runs need ~20-30 GB RAM → Kaggle** (free CPU session: 30 GB, 4 cores) with
  `notebooks/00_run_pipeline_kaggle.ipynb` (§2). A laptop or free Colab (~12 GB) can't hold the ~10M S2/S3 records.
- `--mode validate` → local score on a 20% holdout of train S1s (use `--sample-s1 200000` for quick runs).
- `--mode loco` → leave one country out (train on India, test on US and vice versa) = our France check.
- `--mode test` → writes `output/matching_results.tsv` + `output/candidate_pairs.tsv` and runs the validator.
- Holdout errors (false positives, misses, true matches that never became candidates) are written to
  `models/holdout_errors.tsv` — the best place to find the next idea.

## 2. How to run

**Local** (small runs only — a laptop can't hold the full data)
```bash
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/run_pipeline.py --mode validate --sample-s1 20000
```
The dataset goes in `dataset/train/` and `dataset/test/` (unzip the organizers' zip, move
`student_resource/dataset/*` there). Git ignores that folder, so it is never committed.
`make_test_data.py` writes fake data to `dataset_fake/` and never touches `dataset/`.

**Kaggle — where all full runs happen** (free CPU session: ~30 GB RAM, 4 cores, up to 12 h).
Easiest: import **`notebooks/00_run_pipeline_kaggle.ipynb`** (File → Import Notebook), set `BRANCH` / `MODE` in the
first cell, and for long runs use **Save Version → Save & Run All** (keeps running with the browser closed).
One-time: upload the organizers' zip as a **private** Kaggle dataset (kaggle.com/datasets → New Dataset;
the notebook finds the files wherever Kaggle unpacks them), add it via *Add Input*, turn *Internet* on,
and add a *Secret* `GITHUB_TOKEN` (GitHub token that can read this private repo). Manual version:
```python
# One-time: add a GitHub token (read-only, this repo only) as a Kaggle Secret named GITHUB_TOKEN
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone -b <your-branch> https://{token}@github.com/SteganoSage/team-AiseKaise-amlc26.git
!pip install -q rapidfuzz==3.10.1
!python team-AiseKaise-amlc26/code/business_entity_resolution/src/run_pipeline.py --mode validate \
    --data-dir /kaggle/input/<our-private-dataset>/dataset \
    --output-dir /kaggle/working/output --model-dir /kaggle/working/models
```
To import our modules inside a notebook:
`sys.path.insert(0, ".../code/business_entity_resolution/src")`. Put it at **position 0**, because our `evaluate.py` has the same name as a preinstalled Kaggle package.

**Colab:** free tier (~12 GB RAM) is too small for the real data — only for code tests on fake data
(`USE_FAKE_DATA = True` in the same notebook).

**AWS SageMaker (optional, not needed)** — needs a separate AWS account (not the Builder ID) with a card;
paid from the **$200 AWS credits each participant gets**:
$100 at signup + 5 × $20 console activities). From the organizers' prep guide (Jatin Mehrotra, AWS).

The free tier is too small for our data: the free notebook `ml.t3.medium` has 2 vCPU / **4 GB RAM**,
the free training hours are `ml.m5.xlarge` (16 GB). Use them only for editing and tiny tests. Full
runs need ~30 GB, so pay for a bigger **notebook instance** from the credits:

| Instance | vCPU / RAM | Use |
|---|---|---|
| `ml.r5.2xlarge` | 8 / 64 GB | full validate / test runs (cheapest with enough RAM) |
| `ml.m5.4xlarge` | 16 / 64 GB | same, faster blocking + normalization |
| `ml.g4dn.4xlarge` | 16 / 64 GB + T4 GPU | `--embeddings` runs |

Check the hourly price on aws.amazon.com/sagemaker/pricing (notebook instances, us-east-1) before
creating one. A full test run should take about 1–1.5 h, so the credits last the whole challenge
**if the instance is stopped whenever nothing is running**.

One-time setup (in **us-east-1**, as the guide recommends):
1. Create a budget / billing alert first (AWS Budgets — it's also one of the $20 credit activities).
2. SageMaker AI → Applications and IDEs → Notebooks → **Create notebook instance**: pick a type from the
   table; under **Additional configuration set Volume size to 50 GB** (the 5 GB default can't hold the
   2.5 GB data + cache); platform Amazon Linux 2023 / JupyterLab 4; IAM role: create new, defaults.
   If it fails with a quota/limit error: Service Quotas → Amazon SageMaker → "`<type>` for notebook
   instance usage" → request an increase (approval can take hours).
3. Get the data onto the instance once: upload the organizers' zip to S3 (S3 console upload, 1.1 GB;
   5 GB of S3 is free), then in a JupyterLab **Terminal**:
   ```bash
   cd ~/SageMaker        # the persistent disk — survives stop/start
   git clone https://<github-token>@github.com/SteganoSage/team-AiseKaise-amlc26.git
   cd team-AiseKaise-amlc26
   source activate python3
   pip install -q -r code/business_entity_resolution/requirements.txt
   aws s3 cp s3://<your-bucket>/<zip-name>.zip .
   unzip -q <zip-name>.zip 'student_resource/dataset/*' -x '__MACOSX/*'
   mv student_resource/dataset dataset && rm -rf student_resource <zip-name>.zip
   ```
4. Run in the background so a closed browser tab doesn't kill it:
   ```bash
   nohup python code/business_entity_resolution/src/run_pipeline.py --mode test > test.log 2>&1 &
   tail -f test.log
   ```
5. Download `output/matching_results.tsv` from the JupyterLab file browser (right-click → Download)
   and hand it to the leader for the Unstop upload.
6. **Stop the notebook instance** when done. We never need endpoints, training jobs or Bedrock.
   The Bedrock-playground credit activity is fine to click through, but **no AWS AI service
   (Bedrock, Comprehend, JumpStart models, …) may be called from our pipeline**: models must be
   MIT/Apache ≤ 8B, and external services for resolving entities are banned.

## 3. Rules we cannot break (any one = disqualification)

1. **No external data.** No APIs, geocoding, business registries, scraping or extra datasets. Hand-written dictionaries (`rd → road`, `sarl`) are fine.
2. **Pretrained models must be MIT or Apache-2.0 AND ≤ 8B parameters.** Check the Hugging Face card and add the model to `MODELS.md`.
   - OK: `intfloat/multilingual-e5-small/base`, `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`, `Qwen2.5-7B-Instruct`, `Phi-3.5-mini`.
   - **Not OK:** `Qwen2.5-3B` (Qwen license), any Llama or Gemma, and `Qwen3-8B` (8.19B, just over the limit).
3. **The repo stays private.** Any copy of the dataset (S3 bucket, Kaggle dataset) must also be **private**. Never share code or data outside the team.
4. **Only the team leader logs in to Unstop and uploads**, from one device. Never log in from two devices at once.
5. **Train countries are US + India; test adds France.** Never hard-code or filter country values, and no country one-hot features.

## 4. Git workflow (please follow exactly — this is how we keep track)

- `main` must always run. **Only the leader merges into `main`**, through Pull Requests.
- **Branch name:** `<yourname>/<topic>`, created from the latest main:
  ```bash
  git checkout main && git pull
  git checkout -b rahul/blocking-k
  ```
- **One idea per branch.** Keep it small so it's easy to compare and merge.
- **Commit message:** `[area] what changed | holdout F0.5 0.xxxx (main 0.yyyy)`
  - Areas: `normalize`, `blocking`, `features`, `model`, `predict`, `eval`, `docs`, `infra`.
- **Before opening a PR:**
  1. `git merge main` so your branch is up to date.
  2. Run `--mode validate`.
  3. Fill in the PR template (it opens automatically) with your numbers and main's numbers.
- **Merge rule:** compare **current `main` + your idea** against **current `main`**, with the same
  `--sample-s1` value. Your standalone score doesn't count; see the stacking rule in §5. Merge if:
  - holdout F0.5 goes up clearly (not +0.001 noise; OOF F0.5 should move the same way) **and**
    candidates per S1 does not go up, **or**
  - candidates per S1 goes down with the same F0.5 (smaller candidate sets rank higher), **or**
  - same numbers, but the code is faster or simpler.
- **Do not change** in experiment branches (otherwise scores are not comparable):
  - `RANDOM_SEED`, `VAL_SPLIT_RATIO`, `CV_FOLDS`
  - `evaluate.py`
  - `utils/validate_submission.py`
- **Never commit:** `dataset/`, `output/*.tsv` (only copies under `submissions/<tag>/`), or model files.
- **Notebooks:** name them `notebooks/NN_<topic>_<yourname>.ipynb`, e.g. `01_eda_dhruv.ipynb`. They are for exploring only. Anything that should affect our score must end up in `code/.../src/`, because the final zip is judged by rerunning that code.
- **Every new function gets a docstring.** The organizers explicitly ask for this.

## 5. How we split the work

Nobody owns files: anyone can change anything. **We split by idea.** Each person claims an idea from the list below, tries it on their own branch, reports the score, and picks the next one.

### The loop
1. **Claim an idea.** Assign yourself to its GitHub Issue. That is the claim, and it's how we keep track. Until the issues exist, say it in the group chat.
2. **Build it** on branch `<yourname>/<topic>` from the latest `main`, then run `--mode validate`.
3. **Helped** → merge `main` into your branch, run validate again, and open a PR (stacking rule below).
   **Didn't help** → close the issue with your numbers ("#4: 0.812 vs main 0.815, dropped"), so nobody repeats it.
4. Pick the next idea.

**Two syncs a day, around 13:00 and 21:00 IST.** The leader merges ready PRs **one at a time**. After that, everyone runs `git pull` on `main` before starting their next idea, so everyone always builds on everyone else's wins.

Two people editing the same file is fine. Git merges it automatically unless you both change the same lines. If that happens, the second person merges `main` into their branch and fixes the conflict. Small, single-idea branches keep conflicts tiny.

### Stacking rule: two good ideas don't automatically add up

Each idea is tested alone, but there's no guarantee two good ideas help together. So **every merge is re-tested on top of whatever was merged before it:**

```
main scores 0.80
│
├── #1 blocking (tested vs 0.80)      → 0.83 ✓ merged        → main is now 0.83
│
└── #8 rank features (tested alone)   → 0.84
        git merge main into the #8 branch, run validate again
        → 0.86  ✓ merge                (they stack)
        → 0.82  ✗ don't merge; find out why, or drop it
```

- Merge one idea at a time. Each merge raises the bar.
- Every other open branch must `git merge main` and re-run validate before it can be merged. The PR template asks for this.
- So `main` only ever goes up, one measured step at a time, and every combination in it has actually been scored together.

Why two good ideas might not add up:
- **They fix the same mistakes.** French normalization and the embedding shortlist might both rescue the same transliterated names, so together they give less than the sum.
- **One changes what the other was tuned for.** A bigger blocking K adds more hard-to-tell-apart candidates, so a feature that helped on the old candidate list may help less on the new one.
- **Rarely, they clash** and the combination scores worse than either alone. The re-test catches this before it reaches `main`.

The threshold is re-tuned automatically on every run, so a combination never gets stuck with a threshold picked for only one of the ideas.

### Idea list

**Baseline to beat** (Kaggle, `--mode validate --sample-s1 20000`, 20k train S1 vs **all 10.3M S2/S3**, 12.5 min):

| Holdout F0.5 | P / R | Cands per S1 (stage 2 / stage 1) | Candidate recall (stage 2 / stage 1) | India / US F0.5 | Threshold |
|---|---|---|---|---|---|
| **0.9343** (all-empty 0.055) | 0.964 / 0.877 | 6.16 / 35.4 | 0.935 / 0.941 | 0.905 / 0.952 | 0.69 |

What it says: **blocking is the bottleneck** — 5.9% of true matches never become candidates (the model
can't recover those), pruning only loses 0.6% more. The `name` blocker barely helps at full scale
(recall 0.56, only 440 pairs found by it alone) because `BLOCKING_MAX_DF = 2000` drops most name terms
when there are 5M records. **India is weaker at every stage** (Devanagari script is a likely cause).
Compare every experiment against this row with the same `--sample-s1 20000`.

**Day 1 first moves:** leader → private Kaggle dataset + `00_run_pipeline_kaggle.ipynb` with `MODE="test"`
(Save & Run All) for the first real upload; one person → #A (full-scale validate on Kaggle, runtime/memory); one person → #2 (read
`holdout_errors.tsv`, India first).

"Built" = code exists behind a switch in `config.py` — the job is to **measure it on real data**
(A/B with the switch) and keep or drop it.

| # | Idea | Status | Details |
|---|---|---|---|
| A | Full-scale run on Kaggle | in progress | 20k-S1 validate done (baseline above). Next: `--mode test --train-sample-s1 300000` for the first upload. Note runtime per stage — the stage-1 lines split hashing vs search time. If stage 1 is slow: lower `BLOCKING_MAX_DF`, raise `BLOCKING_CHUNK_ROWS`. |
| 1 | **Blocking recall (top priority)** | todo | Stage-1 recall is 0.941. Try `BLOCKING_TOP_K` 10→20, `BLOCKING_MAX_DF` 2000→10000 (slower — watch the new per-blocker timing lines), a name blocker that keeps common terms. Then `PRUNE_TOP_N` / `PRUNE_MIN_PROB` to keep candidates per S1 small. |
| 2 | Fix what we miss | todo | `holdout_errors.tsv`: `FN_not_candidate` rows = blocking/pruning misses, `FN_scored` = model misses, `FP` = false merges. Fix normalization/features for the biggest patterns. **India first.** |
| 3 | Indian spellings + **Devanagari** | todo | S3 has Hindi-script names/addresses (`मॉडर्न फाइनेंस`). Try transliteration to Latin in `normalize.py`; PIN `411 001`; Shri/Sri/Shree. |
| 4 | French normalization | todo | `sarl/sas/eurl`, `R.`/`rue`, `AV`, `bd`, "St" = saint vs street, accents. Check against real French test records (`eda.py --split test`). |
| 5 | Embedding shortlist | built (`--embeddings`) | multilingual-e5-small on a Kaggle GPU session (`EMBEDDINGS = True` in the notebook). Should help Devanagari + reordering. Measure recall gain vs runtime (~12M texts/split). |
| 6 | Sound-alike / typo blocker | todo | Word TF-IDF misses typos inside rare words. Options: Double Metaphone key, char n-grams restricted to rare tokens. |
| 7 | ~~Speed up features~~ | done | Vectorized (rapidfuzz `cpdist`, index arrays). |
| 8 | Rank features | built (`rank`) | Rank / gap to best within the S1's and the candidate's candidates. Ablate with `FEATURE_GROUPS["rank"] = False`. |
| 9 | Mutual-best feature | built (`rank`) | Part of the rank group. |
| 10 | More similarity features | built (`tfidf`, `numbers`) | Blocking cosines reused; address-number overlap. |
| 11 | Embedding similarity features | built (`--embeddings`) | After #5. |
| 12 | Decision settings | built (switches) | `RESOLVE_CONFLICTS`, `MAX_MATCHES_PER_S1`, `PER_SOURCE_THRESHOLD`. |
| 13 | LightGBM tuning | todo, Day 2+ | Only after the features settle. Pruner params too (`PRUNER_PARAMS`). |
| 14 | ~~EDA~~ | done | `src/eda.py`; real-data facts in CLAUDE.md §2. |
| 15 | ~~Error dump~~ | done | `models/holdout_errors.tsv` after every validate run. |
| 16 | Leave-one-country-out | built (`--mode loco`) | Run it on Kaggle; tells us how far the threshold drifts on an unseen country (France risk). |
| 17 | Cross-encoder reranker | todo, stretch | `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M) on the hardest pairs. Day 3 only if everything else is done. |

New ideas are welcome. Open an issue for them first.

### Leader-only jobs
- **Day 1, first thing:** upload the dataset to Kaggle as a **private** dataset and add teammates as collaborators (§2).
- Merge PRs at the two daily syncs, one at a time (stacking rule).
- All leaderboard uploads (§7) and `submissions/LOG.md`.
- Documentation (with blocking recall + reduction ratio numbers), `MODELS.md`, and the final zip.

## 6. Schedule

| When | Goal | Uploads |
|---|---|---|
| **Day 1 — 25 Sep** | Data in → pipeline rebuilt for 12M records → first full Kaggle run → first real upload. Then #A, #1, #2. Syncs ~13:00 and ~21:00. | #1 all-empty file (expect ≈ 0.05: only ~5% of S1s have no match — tells us if test looks like train). #2 baseline. #3–#5 after improvements are merged. |
| **Day 2 — 26 Sep** | Group features, embeddings, French normalization, leave-one-country-out check, error analysis. | Up to 5, spread through the day, each one = `main` after an improving merge. |
| **Day 3 — 27 Sep** | **No new ideas after 18:00 IST.** Final model, documentation, MODELS.md, zip, full rerun from a fresh clone to prove it reproduces. | Keep 1–2 for the final model. **Last upload by ~21:00 IST**, not 23:50. |

**Spread uploads through the day. Don't save them all for the evening.** Each upload gives feedback we can act on, and unused slots are lost at the daily reset.

**Pick what to upload by local holdout F0.5, not public LB.** The public LB is only part of the test set, and the final ranking uses the private part.

## 7. Upload checklist (leader)

1. `main` is up to date → `--mode validate` → note the holdout F0.5.
2. `--mode test` → the run must end with **`PASS`** from the validator. No PASS = no upload.
3. Copy the outputs: `mkdir submissions/sub-d1-2 && cp output/*.tsv submissions/sub-d1-2/`
4. Add a row to `submissions/LOG.md`. The numbers are in `code/business_entity_resolution/models/run_info_*.json`.
5. Commit and tag:
   ```bash
   git commit -m "sub-d1-2 | holdout F0.5 0.xxxx"
   git tag sub-d1-2
   git push && git push --tags
   ```
6. Upload **only `matching_results.tsv`** on the portal; it must show status `SCORED`. Then add the
   public LB score to `LOG.md` and commit.

**Final zip (once, Day 3):** `AiseKaise_submission.zip` = `output/` (both TSVs of the chosen run) +
`code/business_entity_resolution/` (`src/`, `README.md`, `requirements.txt`, `MODELS.md`) +
filled-in `Documentation_template.md` (`.md` or `.pdf`). No `dataset/`, `models/`, notebooks or
submissions. Rerun it from a fresh clone first — organizers will reproduce it and audit
`candidate_pairs.tsv` (smaller candidate sets rank higher).

## 8. Still to confirm (guidelines / live demo)

- Which upload counts for the final private leaderboard: the last one, the best one, or one we choose?
- When does the 5-per-day counter reset (midnight IST)?
- Deadline and upload place for the final zip.

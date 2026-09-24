# Team Plan — Amazon ML Challenge 2026 (Team AiseKaise)

Read this first, then `CLAUDE.md` (full task details) and the official guidelines PDF.
Window: **25 Sep 00:00 IST → 27 Sep 23:59 IST**. Max **5 leaderboard uploads per day** (15 total).

---

## 1. Where we are

The full baseline pipeline is in `code/business_entity_resolution/src/`:

```
normalize → blocking (shortlist candidates) → pair features → LightGBM → threshold → output TSVs
```

- It runs end to end, and its output passes the organizers' validator (`utils/validate_submission.py`).
- **So far it has only been tested on a small fake dataset.** We have no real scores yet. The first real numbers come on Day 1, once the dataset is in.
- `--mode validate` gives our local score: macro F0.5 on 20% of train S1 entities that the model never sees.
- `--mode test` writes `output/matching_results.tsv` + `output/candidate_pairs.tsv` and runs the validator on them.

## 2. How to run

**Local**
```bash
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/run_pipeline.py --mode validate   # local score
python code/business_entity_resolution/src/run_pipeline.py --mode test       # writes output/*.tsv
```
The dataset goes in `dataset/train/` and `dataset/test/`. Git ignores that folder, so it is never committed.

**Kaggle** (turn Internet ON in notebook settings; GPU only needed for embedding models)
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

## 3. Rules we cannot break (any one = disqualification)

1. **No external data.** No APIs, geocoding, business registries, scraping or extra datasets. Hand-written dictionaries (`rd → road`, `sarl`) are fine.
2. **Pretrained models must be MIT or Apache-2.0 AND ≤ 8B parameters.** Check the Hugging Face card and add the model to `MODELS.md`.
   - OK: `intfloat/multilingual-e5-small/base`, `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`, `Qwen2.5-7B-Instruct`, `Phi-3.5-mini`.
   - **Not OK:** `Qwen2.5-3B` (Qwen license), any Llama or Gemma, and `Qwen3-8B` (8.19B, just over the limit).
3. **The repo stays private.** The Kaggle dataset must also be **private**. Never share code or data outside the team.
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
- **Merge rule:** compare **current `main` + your idea** against **current `main`**. Your standalone score doesn't count; see the stacking rule in §5. Merge if holdout F0.5 goes up clearly (not +0.001 noise; OOF F0.5 should move the same way) **and** blocking recall does not drop. The other acceptable case is the same score with the code faster or simpler.
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

**Day 1, first hour:** three people, three ideas that unblock everything else: **#14** (EDA), **#7** (speed), **#1** (blocking).

| # | Idea | Details | After |
|---|---|---|---|
| 1 | Tune blocking | Measure blocking recall and mean candidates per S1 on real data, then tune `BLOCKING_TOP_K_*` and `BLOCKING_EXACT_KEY_MAX_BLOCK`. Target recall **≥ 0.97** with as few candidates as possible. Blocking sets our ceiling: a true match that isn't shortlisted can never be predicted. | |
| 2 | Fix what blocking misses | Look at the true pairs no blocker catches, and fix normalization for them (missing abbreviations etc.). | #1 |
| 3 | Indian spellings | PIN codes written `411 001`, Shri / Sri / Shree, other transliterations. | |
| 4 | French normalization | `sarl/sas/eurl`, `rue/bd/av/ch`, "St" = saint (French) vs street, accents. Check against the real French test records. | #14 |
| 5 | Embedding shortlist | kNN search with `intfloat/multilingual-e5-small` (MIT, 118M) on a Kaggle GPU. Helps with transliterations and reordered words. | |
| 6 | Sound-alike name key | Double Metaphone key as an extra blocker. | |
| 7 | **Speed up features** | Features are computed in a Python loop, at ~60 µs per pair. Vectorize with `rapidfuzz.process.cpdist(..., workers=-1)`. Makes everyone's experiments faster. | |
| 8 | Rank features | Each candidate's rank among its S1's candidates, and its score gap to that S1's best candidate. **Likely the biggest gain.** | |
| 9 | Mutual-best feature | Is this S1 also the candidate's best S1? | |
| 10 | More similarity features | TF-IDF cosine on name and address; overlap of numbers in the address. | |
| 11 | Embedding similarity features | Cosine of name/address embeddings. | #5 |
| 12 | Decision settings | `RESOLVE_CONFLICTS` on/off, `MAX_MATCHES_PER_S1`, separate thresholds for S2 vs S3. | |
| 13 | LightGBM tuning | Only after the features settle. | Day 2+ |
| 14 | **EDA notebook** | File sizes; % of S1 with no match; matches per S1 (0 / 1 / 2+); S2 vs S3 split; **exact country strings in every file** ("India" vs "IN" breaks country blocking); what French test records look like; postal-code formats; empty name/address rate. Share findings in the group. | |
| 15 | Error dump | Write holdout false positives / false negatives (with names + addresses) to a TSV so everyone can see what's failing. | |
| 16 | Leave-one-country-out check | Train on India, validate on US, then the reverse. Our best stand-in for how the model behaves on France, and whether the threshold holds up. | |
| 17 | Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M) on the hardest pairs. **Stretch goal, Day 3 only if everything else is done.** | |

New ideas are welcome. Open an issue for them first.

### Leader-only jobs
- **Day 1, first thing:** upload the dataset to Kaggle as a **private** dataset and add teammates as collaborators.
- Merge PRs at the two daily syncs, one at a time (stacking rule).
- All leaderboard uploads (§7) and `submissions/LOG.md`.
- Documentation (with blocking recall + reduction ratio numbers), `MODELS.md`, and the final zip.

## 6. Schedule

| When | Goal | Uploads |
|---|---|---|
| **Day 1 — 25 Sep** | Data in → baseline running on real data → first real numbers (recall, holdout F0.5, runtime). First ideas: #14 EDA (shared by noon), #7 speed, #1 blocking. Syncs ~13:00 and ~21:00. | #1 all-empty file (public score ≈ share of no-match S1s; tells us if test looks like train). #2 baseline. #3–#5 after improvements are merged. |
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
6. Upload on Unstop, then add the public LB score to `LOG.md` and commit.

## 8. Still to confirm (guidelines / live demo)

- Which upload counts for the final private leaderboard: the last one, the best one, or one we choose?
- When does the 5-per-day counter reset (midnight IST)?
- When exactly is the dataset released, and how big is the test set?
- The final zip format and deadline for the code + documentation package.

# Organizer video notes (Amazon ML Challenge 2026 intro, 6 slides, ~6 min)

These are notes from the official intro video. It matches `CLAUDE.md`; nothing in it contradicts our plan.

1. **Business Entity Resolution.** Records come from 3 independent sources with no shared ID.
   Decide which ones describe the same real-world business.
2. **The problem.** A business signs up (e.g. Amazon Business) and only **name + address** are used.
   Phone and email exist in reality but are not in this challenge.
   - **Blocking:** bucket records by a cheap key built from name + address, so similar names or
     shared addresses land in the same group. This catches true matches *and* look-alikes. Example:
     `S1-732914 Acme Robotics Inc, 500 Market St, San Jose` → candidates
     `S2-118820 Acme Robotics Inc` (match), `S2-540221 Acme Robotix, 12 Elm Rd` (look-alike),
     `S3-905477 Acme Robotics, Nr City Hall` (match), `S3-063118 Acme Bakery, 500 Market St` (same address, different business).
     This candidate list is what goes into `candidate_pairs.tsv` (portal update 25 Sep: smaller candidate sets per S1 rank higher in the final evaluation, see CLAUDE.md §4).
   - **Matching model:** filters the candidates and keeps only the true matches →
     `matching_results.tsv` (`S1-732914 → S2-118820, S3-905477`), one row per S1 entity. This file is the one the leaderboard scores.
3. **Dataset.** Train has all 3 sources + `train_ground_truth.tsv` (one row per S1, comma-separated
   S2/S3 IDs, empty = no match). Test has the same 3 sources with no labels; predict for every S1.
   All files are TSV, so read with `sep="\t"`.
4. **What you submit.** During the challenge: `matching_results.tsv` only (the only scored file).
   Final zip: `output/matching_results.tsv`, `output/candidate_pairs.tsv` (audited),
   `code/…` (runnable pipeline), `Documentation_template.md`. Run `utils/validate_submission.py` first.
5. **Judging.** Macro F0.5 = (1.25·P·R)/(0.25·P + R).
   - A false merge costs about 2× a miss, so when unsure, don't merge.
   - Singletons: predict empty → 1.0, predict anything → 0.
   - Blocking sets the recall ceiling, so invest there first.
   - Account for region-specific patterns in names and addresses.
   - **No external data lookup** (databases, APIs, geocoding).
6. All the best.

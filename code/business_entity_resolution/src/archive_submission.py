#!/usr/bin/env python3
"""
Copy submission outputs for archiving under submissions/<tag>/.

Usage after a successful --mode test run:
    python code/business_entity_resolution/src/archive_submission.py sub-d1-1 \\
        --notes "Baseline with TF-IDF blocking, LightGBM, threshold 0.85"

This:
1. Copies output/matching_results.tsv and output/candidate_pairs.tsv
   into submissions/<tag>/
2. Copies the run_info_test.json for reference
3. Reminds you to fill in submissions/LOG.md and tag the commit
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def main():
    parser = argparse.ArgumentParser(
        description="Archive a submission under submissions/<tag>/"
    )
    parser.add_argument("tag", help="Submission tag, e.g. sub-d1-1")
    parser.add_argument("--notes", default="", help="Notes for LOG.md")
    args = parser.parse_args()

    tag = args.tag
    sub_dir = os.path.join(config.REPO_ROOT, "submissions", tag)

    if os.path.exists(sub_dir):
        print(f"  ✗ Directory already exists: {sub_dir}")
        sys.exit(1)

    os.makedirs(sub_dir)

    # Copy output files
    for fname in ["matching_results.tsv", "candidate_pairs.tsv"]:
        src = os.path.join(config.OUTPUT_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(sub_dir, fname))
            print(f"  ✓ Copied {fname}")
        else:
            print(f"  ⚠ Not found: {src}")

    # Copy run info
    run_info = os.path.join(config.MODEL_DIR, "run_info_test.json")
    if os.path.exists(run_info):
        shutil.copy2(run_info, os.path.join(sub_dir, "run_info.json"))
        print(f"  ✓ Copied run_info.json")

    print(f"\n  Archived to {sub_dir}")
    print(f"\n  Next steps:")
    print(f"    1. Add a row to submissions/LOG.md")
    print(f"    2. Commit:")
    print(f"       git add submissions/{tag}/ submissions/LOG.md")
    print(f"       git commit -m \"{tag} | holdout F0.5 0.xxxx\"")
    print(f"       git tag {tag}")
    print(f"       git push && git push --tags")
    print(f"    3. Upload on Unstop, then add the public LB score to LOG.md")


if __name__ == "__main__":
    main()

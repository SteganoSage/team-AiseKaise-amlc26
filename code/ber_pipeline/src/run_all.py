"""Run the whole pipeline end to end: data -> normalise -> blocking -> features
-> model -> output files (+ the challenge validator when it is available).

Usage (from src/):
  python run_all.py                  # every stage
  python run_all.py --from model_train   # resume from a stage
Paths: BER_DATA_DIR (challenge dataset/ folder), BER_WORK_DIR (scratch),
BER_OUTPUT_DIR (defaults to BER_WORK_DIR/output). See ber/config.py.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("prepare", ["-m", "ber.prepare"]),
    ("cand_train", ["-m", "ber.candidates", "--split", "train"]),
    ("cand_test", ["-m", "ber.candidates", "--split", "test"]),
    ("feat_train", ["-m", "ber.featurize", "--split", "train"]),
    ("feat_test", ["-m", "ber.featurize", "--split", "test"]),
    ("model_stage1", ["-m", "ber.model", "stage1"]),
    ("ce_train", ["-m", "ber.crossenc", "train"]),
    ("ce_score", ["-m", "ber.crossenc", "score"]),
    ("stage2_train", ["-m", "ber.stage2", "--split", "train"]),
    ("stage2_test", ["-m", "ber.stage2", "--split", "test"]),
    ("model_stage2", ["-m", "ber.model", "stage2"]),
    ("model_predict", ["-m", "ber.model", "predict"]),
]


def validate():
    from ber.config import DATA_DIR
    from ber.submit import OUTPUT_DIR
    script = DATA_DIR.parent / "utils" / "validate_submission.py"
    if not script.exists():
        print(f"validator not found at {script}; skipping", flush=True)
        return 0
    return subprocess.call([sys.executable, str(script),
                            "--matching", str(OUTPUT_DIR / "matching_results.tsv"),
                            "--candidate", str(OUTPUT_DIR / "candidate_pairs.tsv"),
                            "--test-dir", str(DATA_DIR / "test")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STEPS[0][0], choices=[s for s, _ in STEPS])
    args = ap.parse_args()
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    here = Path(__file__).resolve().parent
    names = [s for s, _ in STEPS]
    for name, cmd in STEPS[names.index(args.start):]:
        t = time.time()
        print(f"===== {name} =====", flush=True)
        rc = subprocess.call([sys.executable, "-W", "ignore", "-X", "faulthandler", *cmd], cwd=here)
        print(f"===== {name} finished rc={rc} in {time.time() - t:.0f}s =====", flush=True)
        if rc != 0:
            sys.exit(rc)
    sys.exit(validate())


if __name__ == "__main__":
    main()

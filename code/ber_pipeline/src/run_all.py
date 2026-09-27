"""Run the whole pipeline end to end and check the result.

    raw TSVs -> normalise -> blocking (candidates) -> pair features
      -> LightGBM stage 1 -> cross-encoder (GPU) -> stage-2 context features
      -> LightGBM stage 2 -> decision rule -> output/matching_results.tsv
                                              output/candidate_pairs.tsv
      -> challenge validator (when found)

Every step is a separate process (`python -m ber.<module>`), so memory is fully
released between steps and a failed run can be resumed with --from.

Usage (from this folder):
  python run_all.py                      # every step
  python run_all.py --from model_stage2  # resume from a step
  python run_all.py --skip-crossenc      # CPU only: no cross-encoder feature
Paths and settings: environment variables, see ber/config.py.
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
CE_STEPS = {"ce_train", "ce_score"}


def has_gpu():
    """Whether PyTorch is installed and sees a CUDA GPU (needed by the cross-encoder)."""
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def validate():
    """Run the challenge validator on the two output files, if it can be found.

    Looks for utils/validate_submission.py next to the dataset folder (the
    organisers' student_resource layout) and at the submission root.

    Returns:
        The validator's exit code (0 = PASS), or 0 when it is not found.
    """
    from ber.config import DATA_DIR, OUTPUT_DIR, ROOT
    candidates = [DATA_DIR.parent / "utils" / "validate_submission.py",
                  ROOT / "utils" / "validate_submission.py"]
    script = next((p for p in candidates if p.exists()), None)
    if script is None:
        print("validator not found; skipping", flush=True)
        return 0
    return subprocess.call([sys.executable, str(script),
                            "--matching", str(OUTPUT_DIR / "matching_results.tsv"),
                            "--candidate", str(OUTPUT_DIR / "candidate_pairs.tsv"),
                            "--test-dir", str(DATA_DIR / "test")])


def main():
    """Run the steps in order (from --from on), stop at the first failure, then validate."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STEPS[0][0], choices=[s for s, _ in STEPS])
    ap.add_argument("--skip-crossenc", action="store_true",
                    help="leave out the cross-encoder steps (automatic when no CUDA GPU is found)")
    args = ap.parse_args()
    skip_ce = args.skip_crossenc or not has_gpu()
    if skip_ce:
        print("cross-encoder steps skipped (no CUDA GPU or --skip-crossenc): the stage-2 model "
              "is trained and applied without the cross-encoder score", flush=True)
    env = dict(os.environ, PYTHONIOENCODING=os.environ.get("PYTHONIOENCODING", "utf-8"))
    here = Path(__file__).resolve().parent
    names = [s for s, _ in STEPS]
    for name, cmd in STEPS[names.index(args.start):]:
        if skip_ce and name in CE_STEPS:
            continue
        t = time.time()
        print(f"===== {name} =====", flush=True)
        rc = subprocess.call([sys.executable, "-W", "ignore", "-X", "faulthandler", *cmd],
                             cwd=here, env=env)
        print(f"===== {name} finished rc={rc} in {time.time() - t:.0f}s =====", flush=True)
        if rc != 0:
            sys.exit(rc)
    sys.exit(validate())


if __name__ == "__main__":
    main()

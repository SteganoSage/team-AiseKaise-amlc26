"""Run the business-entity-resolution pipeline in SageMaker Processing.

The Processing job receives the complete ``student_resource/dataset`` prefix at
``/opt/ml/processing/input/dataset``. The existing pipeline is used unchanged
so normalization, blocking, pruning, feature construction, threshold tuning,
submission writing, and validation remain identical to local runs.
"""

import argparse
import os
import subprocess
import sys


def install_requirements(requirements_path: str) -> None:
    """Install the pinned runtime dependencies into the Processing container."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", requirements_path],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BER on student_resource in SageMaker")
    parser.add_argument("--mode", choices=("validate", "loco", "test"), default="test")
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    code_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = "/opt/ml/processing/input/dataset"
    output_dir = "/opt/ml/processing/output"
    model_dir = "/opt/ml/processing/model"
    requirements_path = os.path.join(code_dir, "requirements.txt")
    pipeline_path = os.path.join(code_dir, "src", "run_pipeline.py")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.isdir(os.path.join(dataset_dir, "train")):
        raise FileNotFoundError(
            f"Expected dataset/train under {dataset_dir}. "
            "Upload the student_resource/dataset directory as the Processing input."
        )

    if not args.skip_install:
        install_requirements(requirements_path)

    command = [
        sys.executable,
        pipeline_path,
        "--mode",
        args.mode,
        "--data-dir",
        dataset_dir,
        "--output-dir",
        output_dir,
        "--model-dir",
        model_dir,
    ]
    if args.embeddings:
        command.append("--embeddings")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, check=True, cwd=code_dir, env=env)


if __name__ == "__main__":
    main()
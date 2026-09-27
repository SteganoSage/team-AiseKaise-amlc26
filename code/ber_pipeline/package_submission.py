"""Build <team>_submission.zip in the layout the challenge asks for.

  output/matching_results.tsv, output/candidate_pairs.tsv   (from --outputs, same run)
  code/business_entity_resolution/{src/, README.md, requirements.txt, MODELS.md}
  Documentation_template.md

Usage:
  python code/ber_pipeline/package_submission.py --outputs <folder with both TSVs>
  python code/ber_pipeline/package_submission.py --code-only   # no output/ (add it later)
"""
import argparse
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent            # code/ber_pipeline
CODE_FILES = ("README.md", "requirements.txt", "MODELS.md")
OUTPUT_FILES = ("matching_results.tsv", "candidate_pairs.tsv")


def main():
    """Write the zip and list its contents."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="AiseKaise")
    ap.add_argument("--outputs", help="folder containing both output TSVs of the final run")
    ap.add_argument("--code-only", action="store_true", help="leave output/ out of the zip")
    ap.add_argument("--dest", default=".")
    args = ap.parse_args()
    if not args.code_only and not args.outputs:
        ap.error("pass --outputs <folder> (or --code-only)")
    zpath = Path(args.dest) / f"{args.team}_submission.zip"
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        if not args.code_only:
            for name in OUTPUT_FILES:
                z.write(Path(args.outputs) / name, f"output/{name}")
        for p in sorted((HERE / "src").rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                z.write(p, f"code/business_entity_resolution/{p.relative_to(HERE).as_posix()}")
        for name in CODE_FILES:
            z.write(HERE / name, f"code/business_entity_resolution/{name}")
        z.write(HERE / "Documentation_template.md", "Documentation_template.md")
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    for n in zipfile.ZipFile(zpath).namelist():
        print("  ", n)


if __name__ == "__main__":
    main()

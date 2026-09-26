"""Build <team>_submission.zip in the structure the challenge asks for.

  output/matching_results.tsv, output/candidate_pairs.tsv
  code/business_entity_resolution/{src/, README.md, requirements.txt}
  Documentation_template.md

Usage: python package.py --team NAME --outputs D:/ber_work/output --dest D:/ber_submission
"""
import argparse
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # business_entity_resolution/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--doc", default=str(REPO.parent / "Documentation_template.md"))
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / f"{args.team}_submission.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(Path(args.outputs) / name, f"output/{name}")
        for p in sorted((REPO / "src").rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, f"code/business_entity_resolution/{p.relative_to(REPO).as_posix()}")
        for name in ("README.md", "requirements.txt"):
            z.write(REPO / name, f"code/business_entity_resolution/{name}")
        z.write(args.doc, "Documentation_template.md")
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    for n in zipfile.ZipFile(zpath).namelist():
        print("  ", n)


if __name__ == "__main__":
    main()

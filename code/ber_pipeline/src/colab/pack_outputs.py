"""On the VM: gzip the submission files (+ run log, decision), split into 32 MB parts."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path

out = Path("/content/ber_work/output")
dl = Path("/content/ber_work/download")
shutil.rmtree(dl, ignore_errors=True)
dl.mkdir(parents=True)
files = {
    "matching_results.tsv": out / "matching_results.tsv",
    "candidate_pairs.tsv": out / "candidate_pairs.tsv",
    "full_run.log": Path("/content/ber_work/full_run.log"),
    "decision.json": Path("/content/ber_work/models/decision.json"),
}
CH = 32 * 1024 * 1024
manifest = {}
for name, path in files.items():
    raw = path.read_bytes()
    data = gzip.compress(raw, compresslevel=6)
    parts = []
    for i in range(0, len(data), CH):
        p = dl / f"{name}.gz.part{i // CH:03d}"
        p.write_bytes(data[i:i + CH])
        parts.append(p.name)
    manifest[name] = {"md5": hashlib.md5(raw).hexdigest(), "size": len(raw), "parts": parts}
    print(f"{name}: {len(raw) / 1e6:.1f} MB -> {len(data) / 1e6:.1f} MB gz, {len(parts)} parts")
(dl / "manifest.json").write_text(json.dumps(manifest))

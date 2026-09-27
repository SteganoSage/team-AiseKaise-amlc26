"""Rejoin uploaded 32 MB chunks into the raw parquet cache and verify checksums."""
import hashlib
import json
import os
from pathlib import Path

chunks = Path("/content/ber_work/chunks")
raw = Path("/content/ber_work/raw")
manifest = json.loads((chunks / "manifest.json").read_text())
ok = True
for name, info in manifest.items():
    missing = [p for p in info["parts"] if not (chunks / p).exists()]
    if missing:
        print(f"{name}: MISSING parts {missing}")
        ok = False
        continue
    data = b"".join((chunks / p).read_bytes() for p in info["parts"])
    good = len(data) == info["size"] and hashlib.md5(data).hexdigest() == info["md5"]
    print(f"{name}: {len(data):,} bytes, checksum {'OK' if good else 'MISMATCH'}")
    if good:
        (raw / f"{name}.parquet").write_bytes(data)
        for p in info["parts"]:
            os.remove(chunks / p)
    ok &= good
print("ALL OK" if ok else "PROBLEMS", sorted(os.listdir(raw)))

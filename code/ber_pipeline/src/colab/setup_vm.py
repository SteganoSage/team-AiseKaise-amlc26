"""Colab VM setup: unpack code, install pinned packages, create work folders."""
import os
import subprocess
import zipfile

os.makedirs("/content/ber_work/raw", exist_ok=True)
os.makedirs("/content/ber", exist_ok=True)
zipfile.ZipFile("/content/code.zip").extractall("/content/ber")
r = subprocess.run("pip install -q -r /content/ber/requirements.txt", shell=True,
                   capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:])
print(subprocess.run(
    "pip list 2>/dev/null | grep -iE '^(polars|numpy|scipy|scikit-learn|rapidfuzz|lightgbm|anyascii|sparse.dot.topn) '; "
    "ls -la /content/ber/src /content/ber_work", shell=True, capture_output=True, text=True).stdout)

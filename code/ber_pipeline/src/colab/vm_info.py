import subprocess

print(subprocess.run(
    "nproc; free -g | head -2; df -h /content | tail -1; "
    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; python --version; "
    "pip list 2>/dev/null | grep -iE '^(polars|pyarrow|numpy|scipy|scikit-learn|rapidfuzz|lightgbm|anyascii|sparse.dot.topn) '",
    shell=True, capture_output=True, text=True).stdout)

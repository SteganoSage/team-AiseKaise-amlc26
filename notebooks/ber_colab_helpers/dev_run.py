"""On the VM: quick end-to-end check of the pipeline on 3% of training entities."""
import os
import shutil
import subprocess

for d in ("cand_dev", "feat_dev", "models_dev"):
    shutil.rmtree(f"/content/ber_work/{d}", ignore_errors=True)
env = dict(os.environ, BER_WORK_DIR="/content/ber_work", BER_DATA_DIR="/content/dataset",
           PYTHONIOENCODING="utf-8")
steps = " && ".join(f"python -W ignore -m {s}" for s in (
    "ber.candidates --split train --frac 0.03 --tag dev",
    "ber.featurize --split train --tag dev",
    "ber.model stage1 --tag dev --frac 1.0",
    "ber.crossenc train --tag dev",
    "ber.crossenc score --tag dev",
    "ber.stage2 --split train --tag dev",
    "ber.model stage2 --tag dev --frac 1.0",
))
log = open("/content/ber_work/dev_run.log", "w")
p = subprocess.Popen(f"{steps}; echo EXIT $?", shell=True, cwd="/content/ber/src", env=env,
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("dev run started, pid", p.pid)

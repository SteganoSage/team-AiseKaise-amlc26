FROM = "cand_train"
import os
if os.path.exists("/content/ber_work/full_run.log"):
    os.replace("/content/ber_work/full_run.log", "/content/ber_work/full_run_v1.log")
"""Start the full pipeline on the VM as a background process (survives this exec call)."""
import os
import subprocess

FROM = globals().get("FROM", "prepare")
env = dict(os.environ,
           BER_WORK_DIR="/content/ber_work",
           BER_DATA_DIR="/content/dataset",
           BER_OUTPUT_DIR="/content/ber_work/output",
           PYTHONIOENCODING="utf-8")
log = open("/content/ber_work/full_run.log", "a")
proc = subprocess.Popen(["python", "-u", "run_all.py", "--from", FROM], cwd="/content/ber/src",
                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("started pid", proc.pid, "from", FROM)

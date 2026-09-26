LOG = "/content/ber_work/dev_run.log"
"""On the VM: tail a log file (set LOG before running, default full_run.log)."""
import subprocess

LOG = globals().get("LOG", "/content/ber_work/full_run.log")
print(subprocess.run(f"tail -n 30 {LOG}; echo; free -g | head -2", shell=True,
                     capture_output=True, text=True).stdout)

"""On the VM: stop the dev run (its process group) if it is still running."""
import os
import signal
import subprocess

out = subprocess.run("pgrep -f 'tag dev'", shell=True, capture_output=True, text=True).stdout.split()
for pid in out:
    try:
        os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
print("stopped", len(out), "dev processes")

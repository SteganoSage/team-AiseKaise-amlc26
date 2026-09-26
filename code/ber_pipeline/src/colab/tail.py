"""On the VM: tail the full-run log.

The Colab kernel keeps variables between exec calls, so the path is set
explicitly here rather than read from globals.
"""
import subprocess

print(subprocess.run("tail -n 30 /content/ber_work/full_run.log; echo; free -g | head -2",
                     shell=True, capture_output=True, text=True).stdout)

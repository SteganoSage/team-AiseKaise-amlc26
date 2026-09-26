"""After `colab new -s ber`: save the session's endpoint for restore_ber.py / poll.ps1."""
from pathlib import Path

from colab_cli.common import State

s = State().store.get("ber")
Path("D:/ber_work/ber_endpoint.txt").write_text(s.endpoint)
print("ber endpoint:", s.endpoint)

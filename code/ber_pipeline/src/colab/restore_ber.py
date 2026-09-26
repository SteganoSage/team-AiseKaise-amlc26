"""Re-register the `ber` VM assignment in the Colab CLI's local state.

The CLI's runtime-proxy token expires after ~1 h; the CLI then gets 404s and
drops the session from sessions.json even though the VM still exists. This
rebuilds the record with a fresh token from the server's assignment list (no
new VM is allocated). The endpoint is read from D:/ber_work/ber_endpoint.txt,
written by new_session.py. Run with the Colab CLI's own Python.
"""
from pathlib import Path

from colab_cli.common import State
from colab_cli.state import SessionState

ENDPOINT = Path("D:/ber_work/ber_endpoint.txt").read_text().strip()

st = State()
for a in st.client.list_assignments():
    if a.endpoint == ENDPOINT:
        st.store.add(SessionState(
            name="ber", token=a.runtime_proxy_info.token, url=a.runtime_proxy_info.url,
            endpoint=a.endpoint, variant=a.variant.name, accelerator=str(a.accelerator.value),
            machine_shape=a.machine_shape.name))
        print("restored ber ->", a.endpoint)
        break
else:
    print("assignment not found; VM is gone")

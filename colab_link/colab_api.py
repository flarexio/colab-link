"""Minimal Colab session API (stdlib only): runs locally and ON the Colab VM.

    python3 colab_api.py list            token from $COLAB_TOKEN_FILE (default ./.drive_token)
    python3 colab_api.py unassign ENDPOINT

The access token the colab CLI mints (see tools/drive_token.py) carries the
`colaboratory` scope as well as drive.file, so the same token that workers get
for Drive lets them release their own VM. That is the last line of defence
against a VM that keeps billing after everything supervising it has gone.
Mirrors colab_cli/client.py (google-colab-cli 0.7.4): /tun/m/assignments and
/tun/m/unassign/<endpoint> (GET for an XSRF token, then POST with it).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

DOMAIN = "https://colab.research.google.com"
XSSI = ")]}'\n"


def _call(token: str, path: str, method: str = "GET", headers: dict | None = None) -> dict:
    req = urllib.request.Request(f"{DOMAIN}/tun/m/{path}?authuser=0", method=method, data=b"" if method == "POST" else None,
                                 headers={"Authorization": f"Bearer {token}", "Accept": "application/json",
                                          "X-Colab-Client-Agent": "colab-cli", **(headers or {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode()
    body = body[len(XSSI):] if body.startswith(XSSI) else body
    return json.loads(body) if body.strip() else {}


def assignments(token: str) -> list[dict]:
    """Live sessions: [{endpoint, accelerator, machineShape, ...}]."""
    return _call(token, "assignments").get("assignments", [])


def unassign(token: str, endpoint: str) -> None:
    xsrf = _call(token, f"unassign/{endpoint}")["token"]
    _call(token, f"unassign/{endpoint}", "POST", {"X-Goog-Colab-Token": xsrf})


if __name__ == "__main__":
    tok = open(os.environ.get("COLAB_TOKEN_FILE", ".drive_token")).read().strip()
    if sys.argv[1] == "list":
        for a in assignments(tok):
            print(a.get("endpoint"), a.get("accelerator"), a.get("machineShape"))
    elif sys.argv[1] == "unassign":
        unassign(tok, sys.argv[2])

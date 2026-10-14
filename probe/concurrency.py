"""How many sessions can be open at once? Opens sessions of one kind until the
backend refuses (or --cap is hit), then stops them all.

    python3 -m probe.concurrency --gpu T4 --cap 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from colab_link.session import ColabError, ColabSession  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="")
    ap.add_argument("--cap", type=int, default=6)
    a = ap.parse_args()
    opened = []
    try:
        for i in range(a.cap):
            s = ColabSession(f"conc-{(a.gpu or 'cpu').lower()}-{i}", gpu=a.gpu or None)
            try:
                s.start()
            except ColabError as e:
                print(f"#{i + 1} REFUSED: {str(e).strip().splitlines()[-1][:300]}", flush=True)
                break
            opened.append(s)
            print(f"#{i + 1} opened", flush=True)
    finally:
        for s in opened:
            s.stop()
    print(f"RESULT {a.gpu or 'CPU'}: opened {len(opened)} (cap {a.cap})", flush=True)


if __name__ == "__main__":
    main()

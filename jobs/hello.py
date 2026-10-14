"""Smallest possible job: exercises the job contract end to end on a CPU VM.

    python job.py IN_DIR OUT_DIR
Lists the input files, reports progress a few times, writes out/summary.json.
On a warm worker handle() is called once per job in the same process; `calls`
shows that state survives between jobs.
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

calls = 0


def handle(in_dir: Path, out_dir: Path):
    global calls
    calls += 1
    files = sorted(p for p in in_dir.rglob("*") if p.is_file())
    for i in range(3):
        (out_dir / "progress.json").write_text(json.dumps({"stage": "working", "step": i + 1, "of": 3}))
        print(f"step {i + 1}/3: {len(files)} input files", flush=True)
        time.sleep(2)
    summary = {"host": platform.node(), "python": platform.python_version(), "files": len(files),
               "bytes": sum(p.stat().st_size for p in files), "pid": os.getpid(), "calls": calls}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "progress.json").write_text(json.dumps({"stage": "done"}))
    print("done", summary, flush=True)


if __name__ == "__main__":
    handle(Path(sys.argv[1]), Path(sys.argv[2]))

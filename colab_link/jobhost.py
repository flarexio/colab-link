"""Keep a job module loaded between jobs. Runs ON the Colab VM, started by worker.py.

    python jobhost.py job.py

Imports job.py once (module name "job"), then reads one JSON request per line on
stdin, {"in": IN_DIR, "out": OUT_DIR, "cwd": DIR}, calls job.handle(Path(in), Path(out))
and prints a DONE marker line with {"ok": bool}. Everything else on stdout/stderr
is the job's own output. State in module globals (a loaded model, say) survives
from one job to the next; that is the point of a warm worker.

The work sits behind the __main__ guard because vLLM may spawn processes that
re-import the main module.
"""
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

DONE = "\x1eCOLAB_LINK_DONE "


def main() -> None:
    spec = importlib.util.spec_from_file_location("job", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["job"] = mod
    sys.path.insert(0, str(Path(sys.argv[1]).resolve().parent))
    spec.loader.exec_module(mod)
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        ok = True
        try:
            os.chdir(req["cwd"])
            mod.handle(Path(req["in"]), Path(req["out"]))
        except BaseException:  # noqa: BLE001 - report SystemExit from the job as a failure too
            traceback.print_exc()
            ok = False
        sys.stderr.flush()
        print(DONE + json.dumps({"ok": ok}), flush=True)


if __name__ == "__main__":
    main()

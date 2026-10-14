"""Run one batch job on a fresh Colab VM and always release the VM afterwards.

Flow: new session -> upload job script + input tarball -> launch the job
detached on the VM -> poll a status file -> download output tarball -> stop.

The job is launched with Popen(start_new_session=True) rather than run via
`colab exec`, because exec's websocket times out on long work and the VM
must not depend on this host staying connected.

Usage:
  python3 -m colab_link.runner --gpu L4 --job jobs/vlm_batch.py \
      --input examples/ocr_demo --out results/demo \
      --pip vllm --env MODEL=Qwen/Qwen3-VL-4B-Instruct
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

from .session import ColabSession

REMOTE = "/content/run"

# Guarded by a marker file so a retried exec never starts the job twice.
LAUNCH = r"""
import json, os, subprocess
if not os.path.exists({remote!r} + "/launched"):
    open({remote!r} + "/launched", "w").close()
    os.makedirs({remote!r} + "/in", exist_ok=True)
    os.makedirs({remote!r} + "/out", exist_ok=True)
    subprocess.run(["tar", "xzf", {remote!r} + "/in.tar.gz", "-C", {remote!r} + "/in"], check=True)
    env = dict(os.environ, **{env!r})
    subprocess.Popen(["bash", "-c", {cmd!r}], cwd={remote!r}, env=env, start_new_session=True,
                     stdout=open({remote!r} + "/log.txt", "w"), stderr=subprocess.STDOUT)
print("LAUNCHED")
"""

POLL = r"""
import json, os
r = {remote!r}
done = os.path.exists(r + "/exitcode")
st = {{"done": done, "exitcode": int(open(r + "/exitcode").read().strip() or 1) if done else None}}
if os.path.exists(r + "/out/progress.json"):
    try: st["progress"] = json.load(open(r + "/out/progress.json"))
    except Exception: pass
lines = open(r + "/log.txt", errors="replace").read().splitlines() if os.path.exists(r + "/log.txt") else []
st["tail"] = lines[-{tail}:]
print("STATUS=" + json.dumps(st))
"""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def run_job(job: Path, input_dir: Path, out_dir: Path, gpu: str | None,
            pip: list[str], env: dict[str, str], poll_s: float = 20,
            max_hours: float = 6, keep_log_lines: int = 5) -> int:
    name = f"cl-{uuid.uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timings: dict[str, float] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tar_in = Path(tmp) / "in.tar.gz"
        with tarfile.open(tar_in, "w:gz") as tf:
            for p in sorted(input_dir.iterdir()):
                tf.add(p, arcname=p.name)

        sess = ColabSession(name, gpu=gpu)
        try:
            log(f"starting {name} gpu={gpu or 'CPU'}")
            sess.start()
            sess.wait_ready()
            timings["provision_s"] = time.time() - t0

            # The contents API behind `colab upload` 500s if the parent dir is missing.
            sess.exec(f"import os; os.makedirs({REMOTE!r}, exist_ok=True)", timeout=60)
            sess.upload(job, f"{REMOTE}/job.py")
            sess.upload(tar_in, f"{REMOTE}/in.tar.gz")
            timings["upload_s"] = time.time() - t0 - timings["provision_s"]

            steps = []
            if pip:
                steps.append("uv pip install --system -q " + " ".join(map(shlex.quote, pip)))
            steps.append("python job.py in out")
            # Exit code of the first failing step lands in ./exitcode.
            cmd = "( set -e; " + "; ".join(steps) + " ); echo $? > exitcode"
            sess.exec(LAUNCH.format(remote=REMOTE, env=env, cmd=cmd), timeout=120)
            log("job launched; polling")

            deadline = time.time() + max_hours * 3600
            status: dict = {}
            last_tail: list[str] = []
            poll_failures = 0
            while time.time() < deadline:
                time.sleep(poll_s)
                try:
                    raw = sess.exec(POLL.format(remote=REMOTE, tail=keep_log_lines), timeout=60)
                    poll_failures = 0
                except Exception as e:  # transient websocket hiccup: keep polling
                    poll_failures += 1
                    log(f"poll error {poll_failures}/10: {str(e)[-200:]}")
                    if poll_failures >= 10:  # each exec already retried 3x: VM is gone
                        raise
                    continue
                line = next((l for l in raw.splitlines() if l.startswith("STATUS=")), None)
                if not line:
                    continue
                status = json.loads(line[len("STATUS="):])
                if status["tail"] != last_tail:
                    for l in status["tail"]:
                        if l not in last_tail:
                            log(f"  vm| {l[:200]}")
                    last_tail = status["tail"]
                if status["done"]:
                    break
            else:
                log("max_hours reached; collecting whatever exists")
            timings["job_s"] = time.time() - t0 - timings["provision_s"] - timings["upload_s"]

            sess.exec(f"import subprocess; subprocess.run('cp log.txt out/ && tar czf out.tar.gz -C out .', "
                      f"shell=True, cwd={REMOTE!r}, check=True)", timeout=300)
            tar_out = Path(tmp) / "out.tar.gz"
            sess.download(f"{REMOTE}/out.tar.gz", tar_out)
            with tarfile.open(tar_out) as tf:
                tf.extractall(out_dir, filter="data")
        finally:
            log(f"stopping {name}")
            sess.stop()

    timings["total_s"] = time.time() - t0
    exitcode = status.get("exitcode")
    summary = {"session": name, "gpu": gpu, "exitcode": exitcode,
               **{k: round(v, 1) for k, v in timings.items()}}
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2))
    log(f"finished: {summary}")
    return exitcode if exitcode is not None else 124


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True, help="directory shipped to the VM as ./in")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--pip", action="append", default=[])
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the job")
    ap.add_argument("--poll", type=float, default=20)
    ap.add_argument("--max-hours", type=float, default=6)
    a = ap.parse_args()
    env = dict(kv.split("=", 1) for kv in a.env)
    sys.exit(run_job(a.job, a.input, a.out, a.gpu, a.pip, env, a.poll, a.max_hours))


if __name__ == "__main__":
    main()

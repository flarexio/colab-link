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

from . import guard
from .session import ColabSession, drive_token

REMOTE = "/content/run"
LEASE_S = 20 * 60  # renewed every poll; see guard.py

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
            max_hours: float = 6, keep_log_lines: int = 5,
            setup: list[str] | None = None, python: str = "python",
            drive_project: str | None = None, post: list[str] | None = None,
            drive_root: str = "colab-link-cache") -> int:
    name = f"cl-{uuid.uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timings: dict[str, float] = {}
    status: dict = {}

    with tempfile.TemporaryDirectory() as tmp:
        tar_in = Path(tmp) / "in.tar.gz"
        with tarfile.open(tar_in, "w:gz") as tf:
            for p in sorted(input_dir.iterdir()):
                tf.add(p, arcname=p.name)

        sess = ColabSession(name, gpu=gpu)
        try:
            log(f"starting {name} gpu={gpu or 'CPU'}")
            sess.start()
            # This transport has no worker heartbeat: a short lease renewed while we
            # poll keeps the reaper away, and lapses soon after this process dies.
            guard.lease(sess.endpoint or name, LEASE_S, f"colab-link run --transport exec ({name})")
            sess.wait_ready()
            timings["provision_s"] = time.time() - t0

            # The contents API behind `colab upload` 500s if the parent dir is missing.
            sess.exec(f"import os; os.makedirs({REMOTE!r}, exist_ok=True)", timeout=60)
            def upload(local: Path, remote: str) -> None:
                t = time.time()
                sess.upload(local, remote)
                log(f"uploaded {remote} ({local.stat().st_size / 1e6:.1f} MB) in {time.time() - t:.1f}s")

            def push_token() -> None:
                # Token goes up as a file, never inside exec code (the CLI logs code).
                tok = Path(tmp) / "drive_token"
                tok.write_text(drive_token())
                tok.chmod(0o600)
                upload(tok, f"{REMOTE}/.drive_token")

            upload(job, f"{REMOTE}/job.py")
            upload(tar_in, f"{REMOTE}/in.tar.gz")
            for helper in ("drive_vm.py", "vm_env.py"):
                upload(Path(__file__).with_name(helper), f"{REMOTE}/{helper}")
            token_t = 0.0
            if drive_project:
                push_token()
                token_t = time.time()
                env = {**env, "DRIVE_QUOTA_PROJECT": drive_project, "DRIVE_ROOT": drive_root,
                       "DRIVE_TOKEN_FILE": f"{REMOTE}/.drive_token"}
            timings["upload_s"] = time.time() - t0 - timings["provision_s"]

            steps = list(setup or [])
            if "/" in python:
                # A venv interpreter needs its bin/ on PATH too (vLLM shells out to ninja etc.).
                steps.insert(0, f"export PATH={shlex.quote(python.rsplit('/', 1)[0])}:$PATH")
            if pip:
                steps.append("uv pip install --system -q " + " ".join(map(shlex.quote, pip)))
            steps.append(f"{python} job.py in out")
            steps += list(post or [])
            # Exit code of the first failing step lands in ./exitcode.
            cmd = "( set -e; " + "; ".join(steps) + " ); echo $? > exitcode"
            sess.exec(LAUNCH.format(remote=REMOTE, env=env, cmd=cmd), timeout=120)
            log("job launched; polling")

            deadline = time.time() + max_hours * 3600
            last_tail: list[str] = []
            poll_failures = 0
            while time.time() < deadline:
                time.sleep(poll_s)
                guard.lease(sess.endpoint or name, LEASE_S, f"colab-link run --transport exec ({name})")
                if drive_project and time.time() - token_t > 40 * 60:  # tokens last ~1 h
                    try:
                        push_token()
                        token_t = time.time()
                    except Exception as e:
                        log(f"token refresh failed (will retry): {str(e)[-200:]}")
                try:
                    # Short timeout: a lost kernel reply otherwise stalls the poll for a full
                    # minute (seen ~3x per run); the next poll simply asks again.
                    raw = sess.exec(POLL.format(remote=REMOTE, tail=keep_log_lines),
                                    timeout=20, retries=1)
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

            sess.exec(f"import subprocess; subprocess.run('cp log.txt out/; cp env_state.json out/ 2>/dev/null; "
                      f"tar czf out.tar.gz -C out .', "
                      f"shell=True, cwd={REMOTE!r}, check=True)", timeout=300)
            tar_out = Path(tmp) / "out.tar.gz"
            sess.download(f"{REMOTE}/out.tar.gz", tar_out)
            with tarfile.open(tar_out) as tf:
                tf.extractall(out_dir, filter="data")
        finally:
            log(f"stopping {name}")
            ep = sess.endpoint
            sess.stop()
            sess.forget()
            if ep:
                guard.drop_lease(ep)

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
    ap.add_argument("--setup", action="append", default=[],
                    help="shell command run on the VM (in /content/run) before pip/job")
    ap.add_argument("--python", default="python", help="interpreter for job.py on the VM")
    ap.add_argument("--post", action="append", default=[],
                    help="shell command run on the VM after the job succeeds")
    ap.add_argument("--drive", metavar="GCP_PROJECT", default=None,
                    help="ship a Drive token + drive_vm.py; project is the Drive API quota project")
    a = ap.parse_args()
    env = dict(kv.split("=", 1) for kv in a.env)
    sys.exit(run_job(a.job, a.input, a.out, a.gpu, a.pip, env, a.poll, a.max_hours,
                     setup=a.setup, python=a.python, drive_project=a.drive, post=a.post))


if __name__ == "__main__":
    main()

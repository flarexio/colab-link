"""Run one job on a fresh Colab VM, coordinated over NATS (FlareX edge convention).

Control plane: colab CLI (new VM, ship the agent + secrets, stop).
Data plane:    Google Drive for files, NATS for control messages; see worker.py
               for the subject list.

Flow: tar input -> Drive  ->  new VM + launch worker  ->  wait .ready  ->
      request .job  ->  stream .log/.progress, watch .heartbeat  ->  .result
      ->  fetch output from Drive  ->  stop VM (always).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

from .session import ColabSession, drive_token

FLAREX = Path(os.environ.get("FLAREX_DIR", Path.home() / ".flarex" / "colab-link"))
NATS_URL = os.environ.get("NATS_URL", "wss://nats.flarex.io")
AGENT = "/content/agent"
HEARTBEAT_TIMEOUT_S = 90
READY_TIMEOUT_S = 300

LAUNCH = r"""
import os, subprocess
os.chmod({agent!r} + "/.nats.creds", 0o600)
os.chmod({agent!r} + "/.drive_token", 0o600)
if not os.path.exists({agent!r} + "/launched"):
    open({agent!r} + "/launched", "w").close()
    subprocess.Popen(["bash", "-c", "uv pip install --system -q 'nats-py[nkeys]' aiohttp && exec python3 worker.py"],
                     cwd={agent!r}, env=dict(os.environ, **{env!r}), start_new_session=True,
                     stdout=open({agent!r} + "/worker.log", "w"), stderr=subprocess.STDOUT)
print("LAUNCHED")
"""


def launch_agent(sess: ColabSession, vm: str, gpu: str | None, drive_project: str, cache_root: str,
                 jobs_root: str, token: str, extra_env: dict | None = None, flarex: Path = FLAREX) -> None:
    """Ship worker.py + helpers + secrets to a READY VM and start the worker detached."""
    sess.exec(f"import os; os.makedirs({AGENT!r}, exist_ok=True)")
    here = Path(__file__).parent
    with tempfile.TemporaryDirectory() as tmp:
        tok = Path(tmp) / "drive_token"
        tok.write_text(token)
        # Secrets go up as files, never inside exec code (the CLI logs executed code).
        for local, remote in [(flarex / "user.creds", ".nats.creds"), (flarex / "id", "id"),
                              (tok, ".drive_token"), (here / "worker.py", "worker.py"),
                              (here / "jobhost.py", "jobhost.py"), (here / "drive_vm.py", "drive_vm.py"),
                              (here / "vm_env.py", "vm_env.py")]:
            sess.upload(local, f"{AGENT}/{remote}")
    agent_env = {"VM_ID": vm, "NATS_URL": NATS_URL, "GPU": gpu or "",
                 "DRIVE_QUOTA_PROJECT": drive_project, "DRIVE_ROOT": cache_root,
                 "DRIVE_JOBS_ROOT": jobs_root, "DRIVE_TOKEN_FILE": f"{AGENT}/.drive_token",
                 **(extra_env or {})}
    sess.exec(LAUNCH.format(agent=AGENT, env=agent_env), 120)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _drive(project: str | None, cache_root: str):
    if not project:
        sys.exit("no GCP quota project for the Drive API: set COLAB_LINK_DRIVE_PROJECT "
                 "(or [drive] project in profiles.toml); see README")
    os.environ["DRIVE_TOKEN"] = drive_token()
    os.environ["DRIVE_QUOTA_PROJECT"] = project
    os.environ["DRIVE_ROOT"] = cache_root
    from . import drive_vm
    return drive_vm


async def run_job_nats(job: Path, input_dir: Path, out_dir: Path, gpu: str | None, env: dict,
                       venv: dict | None, pip: list[str], drive_project: str,
                       cache_root: str = "colab-link-cache", jobs_root: str = "colab-link-jobs",
                       max_hours: float = 6, keep_files: bool = False, cache: bool = True) -> int:
    import nats

    edge = (FLAREX / "id").read_text().strip()
    vm = f"cl-{uuid.uuid4().hex[:8]}"
    job_id = f"{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    base = f"edges.{edge}.colab-link.workers.{vm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timings: dict[str, float] = {}
    result: dict = {}

    d = _drive(drive_project, cache_root)
    in_file = f"job-{job_id}-in.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        tar_in = Path(tmp) / "in.tar.gz"
        with tarfile.open(tar_in, "w:gz") as tf:
            for p in sorted(input_dir.iterdir()):
                tf.add(p, arcname=p.name)
        await asyncio.to_thread(d.put, str(tar_in), in_file, jobs_root)
        timings["input_upload_s"] = round(time.time() - t0, 1)
        log(f"job {job_id}: input {tar_in.stat().st_size / 1e6:.1f} MB -> Drive:{jobs_root}/{in_file}")

        nc = await nats.connect(NATS_URL, user_credentials=str(FLAREX / "user.creds"),
                                name=f"Colab Link - {edge}", max_reconnect_attempts=-1)
        ready, done = asyncio.Event(), asyncio.Event()
        last_beat = [0.0]

        async def on_msg(msg):
            kind = msg.subject.rsplit(".", 1)[-1]
            data = json.loads(msg.data)
            if kind == "ready":
                log(f"worker ready ({data.get('gpu') or 'CPU'}, NATS connect {data['connect_s']}s)")
                last_beat[0] = time.time()
                ready.set()
            elif kind == "heartbeat":
                last_beat[0] = time.time()
            elif kind == "log" and data.get("job_id") == job_id:
                for line in data["lines"]:
                    log(f"  vm| {line[:200]}")
            elif kind == "progress" and data.get("job_id") == job_id:
                log(f"  progress: {json.dumps({k: v for k, v in data.items() if k != 'job_id'})[:200]}")
            elif kind == "result" and data.get("job_id") == job_id:
                result.update(data)
                done.set()

        sub = await nc.subscribe(f"{base}.*", cb=on_msg)
        sess = ColabSession(vm, gpu=gpu)
        try:
            log(f"starting {vm} gpu={gpu or 'CPU'}")
            await asyncio.to_thread(sess.start)
            await asyncio.to_thread(sess.wait_ready)
            timings["provision_s"] = round(time.time() - t0 - timings["input_upload_s"], 1)

            await asyncio.to_thread(launch_agent, sess, vm, gpu, drive_project, cache_root, jobs_root,
                                    os.environ["DRIVE_TOKEN"])
            try:
                await asyncio.wait_for(ready.wait(), READY_TIMEOUT_S)
            except asyncio.TimeoutError:
                tail = await asyncio.to_thread(
                    sess.exec, f"print(open({AGENT!r} + '/worker.log').read()[-2000:])")
                raise RuntimeError(f"worker never reported ready:\n{tail}")
            timings["agent_s"] = round(time.time() - t0 - sum(timings.values()), 1)

            spec = {"job_id": job_id, "script": job.read_text(), "in_file": in_file,
                    "env": env, "venv": venv, "pip": pip, "cache": cache}
            ack = json.loads((await nc.request(f"{base}.job", json.dumps(spec).encode(), timeout=30)).data)
            if not ack.get("accepted"):
                raise RuntimeError(f"job rejected: {ack}")
            log(f"job {job_id} accepted")

            deadline = time.time() + max_hours * 3600
            token_t = time.time()
            while not done.is_set():
                try:
                    await asyncio.wait_for(done.wait(), 10)
                except asyncio.TimeoutError:
                    pass
                now = time.time()
                if now - last_beat[0] > HEARTBEAT_TIMEOUT_S:
                    raise RuntimeError(f"no heartbeat for {now - last_beat[0]:.0f}s; VM presumed dead")
                if now > deadline:
                    raise RuntimeError(f"max_hours ({max_hours}) reached")
                if now - token_t > 40 * 60:  # Drive tokens last ~1 h
                    await nc.publish(f"{base}.ctl", json.dumps(
                        {"op": "drive_token", "token": await asyncio.to_thread(drive_token)}).encode())
                    token_t = now

            if result.get("out_file"):
                t = time.time()
                tar_out = Path(tmp) / "out.tar.gz"
                await asyncio.to_thread(d.get, result["out_file"], str(tar_out), 16, jobs_root)
                with tarfile.open(tar_out) as tf:
                    tf.extractall(out_dir, filter="data")
                timings["output_download_s"] = round(time.time() - t, 1)
        finally:
            try:
                await nc.publish(f"{base}.ctl", json.dumps({"op": "shutdown"}).encode())
            except Exception:
                pass
            log(f"stopping {vm}")
            await asyncio.to_thread(sess.stop)
            sess.forget()
            await sub.unsubscribe()
            await nc.drain()
            if not keep_files:
                for name in (in_file, result.get("out_file")):
                    if name:
                        try:
                            await asyncio.to_thread(d.rm, name, jobs_root)
                        except Exception as e:
                            log(f"could not remove Drive:{jobs_root}/{name}: {e}")

    timings["total_s"] = round(time.time() - t0, 1)
    exitcode = result.get("exitcode")
    summary = {"job_id": job_id, "vm": vm, "gpu": gpu, "exitcode": exitcode,
               "vm_timings": result.get("timings"), **timings}
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2))
    log(f"finished: {json.dumps(summary)}")
    return exitcode if exitcode is not None else 124

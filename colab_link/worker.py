"""colab-link edge worker. Runs ON the Colab VM, talks to colab-link over NATS.

Files never travel over NATS: inputs and outputs go through Google Drive
(folder DRIVE_JOBS_ROOT); NATS carries only control messages.

Subjects (FlareX edge convention), base = edges.<edge_id>.colab-link.workers.<vm_id>:
  .ready      pub   once connected                      {vm_id, gpu, connect_s}
  .heartbeat  pub   every HEARTBEAT_S                   {t, busy, job_id}
  .job        req   colab-link submits a job -> ack     {accepted, reason?}
  .log        pub   batched job output lines            {job_id, lines}
  .progress   pub   out/progress.json whenever it changes
  .result     pub   job finished                        {job_id, exitcode, out_file, timings}
  .status     req   current state + last result
  .ctl        sub   {"op": "drive_token", "token"} | {"op": "shutdown"}

Job spec (the .job payload):
  {job_id, script: <job.py source>, in_file: <Drive name>, env: {...},
   venv: {name, packages, jit} | null, pip: [...], cache: bool}

Launched by colab-link with: python3 worker.py   (cwd = /content/agent)
Env: VM_ID, NATS_URL, DRIVE_QUOTA_PROJECT, DRIVE_ROOT, DRIVE_JOBS_ROOT, GPU.
"""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import tarfile
import time
import traceback
from pathlib import Path

AGENT = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT))
import drive_vm  # noqa: E402

HEARTBEAT_S = 5
JOBS_ROOT = os.environ.get("DRIVE_JOBS_ROOT", "colab-link-jobs")
VENV_PY = "/content/venv/bin/python"


class Worker:
    def __init__(self):
        self.edge = (AGENT / "id").read_text().strip()
        self.vm = os.environ["VM_ID"]
        self.base = f"edges.{self.edge}.colab-link.workers.{self.vm}"
        self.nc = None
        self.job_id: str | None = None
        self.last_result: dict | None = None
        self.stop = asyncio.Event()

    def subj(self, kind: str) -> str:
        return f"{self.base}.{kind}"

    async def pub(self, kind: str, payload: dict) -> None:
        await self.nc.publish(self.subj(kind), json.dumps(payload).encode())

    # -- NATS wiring ---------------------------------------------------------
    async def run(self) -> None:
        import nats

        t = time.time()
        self.nc = await nats.connect(
            os.environ.get("NATS_URL", "wss://nats.flarex.io"),
            user_credentials=str(AGENT / ".nats.creds"),
            name=f"Colab Link VM - {self.edge}",
            max_reconnect_attempts=-1, reconnect_time_wait=2)
        await self.nc.subscribe(self.subj("job"), cb=self.on_job)
        await self.nc.subscribe(self.subj("status"), cb=self.on_status)
        await self.nc.subscribe(self.subj("ctl"), cb=self.on_ctl)
        await self.pub("ready", {"vm_id": self.vm, "gpu": os.environ.get("GPU"),
                                 "connect_s": round(time.time() - t, 2)})
        print(f"[worker] ready on {self.base}", flush=True)
        beat = asyncio.create_task(self.heartbeat())
        await self.stop.wait()
        beat.cancel()
        await self.nc.drain()

    async def heartbeat(self) -> None:
        while True:
            await self.pub("heartbeat", {"t": time.time(), "busy": self.job_id is not None,
                                         "job_id": self.job_id})
            await asyncio.sleep(HEARTBEAT_S)

    async def on_status(self, msg) -> None:
        await msg.respond(json.dumps({"vm_id": self.vm, "busy": self.job_id is not None,
                                      "job_id": self.job_id, "last_result": self.last_result}).encode())

    async def on_ctl(self, msg) -> None:
        cmd = json.loads(msg.data)
        if cmd.get("op") == "drive_token":
            tok = AGENT / ".drive_token"
            tok.write_text(cmd["token"])
            tok.chmod(0o600)
        elif cmd.get("op") == "shutdown":
            self.stop.set()

    async def on_job(self, msg) -> None:
        spec = json.loads(msg.data)
        if self.job_id:
            await msg.respond(json.dumps({"accepted": False, "reason": f"busy with {self.job_id}"}).encode())
            return
        self.job_id = spec["job_id"]
        await msg.respond(json.dumps({"accepted": True}).encode())
        asyncio.create_task(self.run_job(spec))

    # -- job execution -------------------------------------------------------
    async def run_job(self, spec: dict) -> None:
        jid = spec["job_id"]
        work = Path(f"/content/jobs/{jid}")
        (work / "in").mkdir(parents=True, exist_ok=True)
        (work / "out").mkdir(exist_ok=True)
        log_file = open(work / "out" / "log.txt", "w")
        timings: dict[str, float] = {}
        exitcode = 1
        t0 = time.time()
        try:
            env = {**os.environ, **spec.get("env", {}),
                   "VM_ENV_STATE": str(work / "out" / "env_state.json")}
            if not spec.get("cache", True):
                env.pop("DRIVE_QUOTA_PROJECT", None)  # vm_env skips Drive caches without it
            venv = spec.get("venv")
            python = "python3"
            if venv:
                cmd = (f"python3 {AGENT}/vm_env.py prepare --venv {shlex.quote(venv['name'])} "
                       f"--packages {shlex.quote(' '.join(venv['packages']))}"
                       + (f" --jit {shlex.quote(venv['jit'])}" if venv.get("jit") else ""))
                if await self.stream(jid, cmd, work, env, log_file):
                    raise RuntimeError("environment prepare failed")
                python = VENV_PY
                env["PATH"] = f"/content/venv/bin:{env['PATH']}"
            if spec.get("pip"):
                pip_target = f"--python {VENV_PY}" if venv else "--system"
                cmd = f"uv pip install -q {pip_target} " + " ".join(map(shlex.quote, spec["pip"]))
                if await self.stream(jid, cmd, work, env, log_file):
                    raise RuntimeError("pip install failed")
            timings["env_s"] = round(time.time() - t0, 1)

            t = time.time()
            await asyncio.to_thread(drive_vm.get, spec["in_file"], str(work / "in.tar.gz"), 16, JOBS_ROOT)
            with tarfile.open(work / "in.tar.gz") as tf:
                tf.extractall(work / "in", filter="data")
            (work / "job.py").write_text(spec["script"])
            timings["input_s"] = round(time.time() - t, 1)

            t = time.time()
            watcher = asyncio.create_task(self.watch_progress(jid, work / "out" / "progress.json"))
            exitcode = await self.stream(jid, f"{python} job.py in out", work, env, log_file)
            watcher.cancel()
            timings["job_s"] = round(time.time() - t, 1)

            if venv and exitcode == 0:
                t = time.time()
                await self.stream(jid, f"python3 {AGENT}/vm_env.py finalize", work, env, log_file)
                timings["finalize_s"] = round(time.time() - t, 1)
        except Exception:
            err = traceback.format_exc()
            log_file.write(err)
            await self.pub("log", {"job_id": jid, "lines": err.splitlines()})
        finally:
            log_file.close()
            out_file = None
            try:
                t = time.time()
                out_tar = work / "out.tar.gz"
                with tarfile.open(out_tar, "w:gz") as tf:
                    tf.add(work / "out", arcname=".")
                out_file = f"job-{jid}-out.tar.gz"
                await asyncio.to_thread(drive_vm.put, str(out_tar), out_file, JOBS_ROOT)
                timings["output_s"] = round(time.time() - t, 1)
            except Exception:
                await self.pub("log", {"job_id": jid, "lines": traceback.format_exc().splitlines()})
            timings["total_s"] = round(time.time() - t0, 1)
            self.last_result = {"job_id": jid, "exitcode": exitcode, "out_file": out_file,
                                "timings": timings}
            self.job_id = None
            await self.pub("result", self.last_result)

    async def stream(self, jid: str, cmd: str, cwd: Path, env: dict, log_file) -> int:
        """Run a shell command, tee output to log.txt and to .log in ~1 s batches."""
        proc = await asyncio.create_subprocess_shell(
            cmd, cwd=cwd, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            limit=1 << 20)
        batch: list[str] = []
        last = time.time()
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip("\n")
            log_file.write(line + "\n")
            batch.append(line[:500])
            if time.time() - last > 1 or len(batch) >= 200:
                await self.pub("log", {"job_id": jid, "lines": batch})
                batch, last = [], time.time()
        if batch:
            await self.pub("log", {"job_id": jid, "lines": batch})
        log_file.flush()
        return await proc.wait()

    async def watch_progress(self, jid: str, path: Path) -> None:
        seen = None
        while True:
            await asyncio.sleep(1)
            try:
                data = path.read_text()
            except OSError:
                continue
            if data != seen:
                seen = data
                try:
                    await self.pub("progress", {"job_id": jid, **json.loads(data)})
                except ValueError:
                    pass


if __name__ == "__main__":
    asyncio.run(Worker().run())

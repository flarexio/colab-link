"""colab-link edge worker. Runs ON the Colab VM, talks to colab-link over NATS.

Files never travel over NATS: inputs and outputs go through Google Drive
(folder DRIVE_JOBS_ROOT); NATS carries only control messages.

Two modes:
  one-shot  (colab-link run)    colab-link sends one job with a .job request.
  warm      (colab-link serve)  POOL_ID is set: the worker reads its pool spec from
            KV, prepares the environment once, then pulls jobs from the JetStream
            work queue until it has been idle IDLE_MINUTES, says .bye and exits
            (colab-link then stops the VM).
  LLM       (colab-link llm)    a pool with mode "llm": the worker runs `vllm serve`
            on localhost and answers OpenAI requests from NATS (see llm_loop) until
            no request has come for IDLE_MINUTES.

Subjects (FlareX edge convention), base = edges.<edge_id>.colab-link.workers.<vm_id>:
  .ready      pub   once connected                      {vm_id, gpu, pool, connect_s}
  .heartbeat  pub   every HEARTBEAT_S                   {vm_id, pool, gpu, busy, job_id, t}
  .job        req   one-shot job -> ack                 {accepted, reason?}
  .log        pub   batched job output lines            {job_id, lines}
  .progress   pub   out/progress.json whenever it changes
  .result     pub   job finished                        {job_id, exitcode, out_file, timings}
  .status     req   current state + last result
  .ctl        sub   {"op": "drive_token", "token"} | {"op": "cancel", "job_id"} | {"op": "shutdown"}
                    | {"op": "ping"}  (the supervisor is alive; it sends one every 30 s)
  .bye        pub   warm worker leaving (idle or shutdown)

LLM requests (queue group llm-<model>, so several VMs share the load):
  edges.<edge_id>.colab-link.llm.<model>.{chat.completions,completions,models}
  The payload is the OpenAI request body. Without "stream" the reply is the OpenAI
  response; with "stream": true each chunk is one message to the reply subject
  and an empty message ends the stream. Errors carry the header
  Nats-Service-Error-Code (the HTTP status) and an OpenAI error body.

Job spec (one-shot .job payload; in warm mode the pool spec plus {job_id, in_file}):
  {job_id, script: <job.py source>, in_file: <Drive name>, env: {...},
   venv: {name, packages, jit} | null, pip: [...], cache: bool}

A script that defines handle(in_dir, out_dir) is kept loaded by jobhost.py between
jobs (the model loads once); other scripts run as `python job.py in out` per job.

Self-stop: a VM must never outlive whoever supervises it. If no .ctl message has
come for SUPERVISOR_TIMEOUT_S (the service or the `run` client died, or NATS is
unreachable), or the access token is about to expire with no fresh one, the
worker releases its own VM through the Colab API (colab_api.py; the token has
the colaboratory scope). colab-link's reaper (guard.py) is the second line.

Launched by colab-link with: python3 worker.py   (cwd = /content/agent)
Env: VM_ID, NATS_URL, DRIVE_QUOTA_PROJECT, DRIVE_ROOT, DRIVE_JOBS_ROOT, GPU, COLAB_ENDPOINT,
     and in warm mode POOL_ID, IDLE_MINUTES.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import sys
import tarfile
import time
import traceback
from pathlib import Path

AGENT = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT))
import colab_api  # noqa: E402
import drive_vm  # noqa: E402

HEARTBEAT_S = 5
SUPERVISOR_TIMEOUT_S = 600     # no .ctl for this long -> release the VM
TOKEN_MARGIN_S = 300           # release the VM while the token can still do it
JOBS_ROOT = os.environ.get("DRIVE_JOBS_ROOT", "colab-link-jobs")
VENV_PY = "/content/venv/bin/python"
DONE = "\x1eCOLAB_LINK_DONE "  # jobhost.py marker
# Torch inductor defaults to /tmp/torchinductor_<user>; keep it under the JIT cache dirs.
INDUCTOR_DIR = "/root/.cache/torchinductor"
LLM_PORT = 8000
LLM_ROUTES = {"chat.completions": ("POST", "/v1/chat/completions"),
              "completions": ("POST", "/v1/completions"),
              "models": ("GET", "/v1/models")}


def now() -> float:
    return round(time.time(), 1)


class Worker:
    def __init__(self):
        self.edge = (AGENT / "id").read_text().strip()
        self.vm = os.environ["VM_ID"]
        self.pool = os.environ.get("POOL_ID") or None
        self.base = f"edges.{self.edge}.colab-link.workers.{self.vm}"
        self.nc = None
        self.js = None
        self.kv = None
        self.job_id: str | None = None
        self.last_result: dict | None = None
        self.stop = asyncio.Event()
        # Environment, prepared once per worker.
        self.python: str | None = None
        self.env: dict | None = None
        self.env_built = False      # a fresh venv/JIT cache still needs uploading
        # Output sink of the current job.
        self.log_file = None
        self.proc: asyncio.subprocess.Process | None = None
        self.host: asyncio.subprocess.Process | None = None
        self.host_reader: asyncio.Task | None = None
        self.host_done: asyncio.Future | None = None
        self.cancelled: set[str] = set()
        self.batch: list[str] = []
        self.batch_t = time.time()
        # LLM mode.
        self.llm: dict | None = None     # {model, state, inflight, requests}, sent with heartbeats
        self.log_tag: str | None = None  # .log job_id while no job runs (LLM server output)
        self.server: asyncio.subprocess.Process | None = None
        # Self-stop (see module docstring) and the reaper's view of us.
        self.endpoint = os.environ.get("COLAB_ENDPOINT") or None
        self.sup_t = time.time()
        self.token_exp: float | None = None
        self.idle_since = time.time()
        self.idle_limit: float | None = None

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
        await self.nc.subscribe(self.subj("status"), cb=self.on_status)
        await self.nc.subscribe(self.subj("ctl"), cb=self.on_ctl)
        if not self.pool:
            await self.nc.subscribe(self.subj("job"), cb=self.on_job)
        await self.pub("ready", {"vm_id": self.vm, "gpu": os.environ.get("GPU"), "pool": self.pool,
                                 "connect_s": round(time.time() - t, 2)})
        print(f"[worker] ready on {self.base} pool={self.pool}", flush=True)
        beat = asyncio.create_task(self.heartbeat())
        watchdog = asyncio.create_task(self.watchdog())
        asyncio.create_task(self.read_token_expiry())
        if self.pool:
            try:
                await self.pool_loop()
            except Exception:
                traceback.print_exc()
            await self.pub("bye", {"vm_id": self.vm, "pool": self.pool})
            if not self.stop.is_set():  # the service answers .bye with shutdown + `colab stop`
                try:
                    await asyncio.wait_for(self.stop.wait(), 120)
                except asyncio.TimeoutError:
                    await self.self_stop("said bye but nobody stopped the VM")
        else:
            await self.stop.wait()
        beat.cancel()
        watchdog.cancel()
        await self.kill_host()
        if self.server and self.server.returncode is None:
            self.server.kill()
        await self.nc.drain()

    async def heartbeat(self) -> None:
        while True:
            await self.pub("heartbeat", {"vm_id": self.vm, "pool": self.pool, "gpu": os.environ.get("GPU"),
                                         "busy": self.job_id is not None, "job_id": self.job_id,
                                         "llm": self.llm, "endpoint": self.endpoint,
                                         "supervised_s": round(time.time() - self.sup_t),
                                         "idle_s": None if self.job_id or (self.llm and self.llm["state"] != "serving")
                                         else round(time.time() - self.idle_since),
                                         "idle_limit_s": self.idle_limit, "t": time.time()})
            await asyncio.sleep(HEARTBEAT_S)

    async def on_status(self, msg) -> None:
        await msg.respond(json.dumps({"vm_id": self.vm, "pool": self.pool, "busy": self.job_id is not None,
                                      "job_id": self.job_id, "last_result": self.last_result}).encode())

    async def on_ctl(self, msg) -> None:
        cmd = json.loads(msg.data)
        self.sup_t = time.time()
        if cmd.get("op") == "drive_token":
            tok = AGENT / ".drive_token"
            tok.write_text(cmd["token"])
            tok.chmod(0o600)
            asyncio.create_task(self.read_token_expiry())
        elif cmd.get("op") == "cancel" and cmd.get("job_id") and cmd["job_id"] == self.job_id:
            self.cancelled.add(cmd["job_id"])
            print(f"[worker] cancelling {cmd['job_id']}", flush=True)
            if self.proc and self.proc.returncode is None:
                self.proc.kill()
            if self.host_done and not self.host_done.done():
                await self.kill_host()  # the loaded model goes too; the next job reloads it
        elif cmd.get("op") == "shutdown":
            self.stop.set()

    async def on_job(self, msg) -> None:
        spec = json.loads(msg.data)
        if self.job_id:
            await msg.respond(json.dumps({"accepted": False, "reason": f"busy with {self.job_id}"}).encode())
            return
        self.job_id = spec["job_id"]
        await msg.respond(json.dumps({"accepted": True}).encode())

        async def one_shot():
            self.last_result = await self.run_job(spec, spec)
            await self.pub("result", self.last_result)

        asyncio.create_task(one_shot())

    # -- self-stop ---------------------------------------------------------------
    async def read_token_expiry(self) -> None:
        import urllib.parse
        import urllib.request

        def expiry() -> float:
            data = urllib.parse.urlencode({"access_token": (AGENT / ".drive_token").read_text().strip()})
            with urllib.request.urlopen("https://oauth2.googleapis.com/tokeninfo", data.encode(), timeout=30) as r:
                return time.time() + int(json.loads(r.read())["expires_in"])

        try:
            self.token_exp = await asyncio.to_thread(expiry)
        except Exception as e:
            print(f"[worker] could not read token expiry: {e}", flush=True)

    async def watchdog(self) -> None:
        while True:
            await asyncio.sleep(15)
            quiet = time.time() - self.sup_t
            if quiet > SUPERVISOR_TIMEOUT_S:
                return await self.self_stop(f"no word from the supervisor for {quiet:.0f}s")
            if self.token_exp and self.token_exp - time.time() < TOKEN_MARGIN_S:
                return await self.self_stop("access token about to expire and no fresh one came")

    async def self_stop(self, reason: str) -> None:
        print(f"[worker] self-stop: {reason}", flush=True)
        try:
            await self.pub("bye", {"vm_id": self.vm, "pool": self.pool, "reason": f"self-stop: {reason}"})
            await self.nc.flush(timeout=5)
        except Exception:
            pass
        if self.endpoint:
            try:
                token = (AGENT / ".drive_token").read_text().strip()
                await asyncio.to_thread(colab_api.unassign, token, self.endpoint)
                print("[worker] VM released", flush=True)
            except Exception as e:
                print(f"[worker] could not release the VM: {e}", flush=True)
        self.stop.set()

    # -- warm mode -------------------------------------------------------------
    async def pool_loop(self) -> None:
        import nats.errors

        self.js = self.nc.jetstream()
        self.kv = await self.js.key_value(f"colab-link-{self.edge}")
        pool = json.loads((await self.kv.get(f"pools.{self.pool}")).value)
        if pool.get("mode") == "llm":
            return await self.llm_loop(pool)
        sub = await self.js.pull_subscribe_bind(durable=f"pool-{self.pool}",
                                                stream=f"COLAB_LINK_{self.edge}")
        idle_s = self.idle_limit = float(os.environ.get("IDLE_MINUTES", pool.get("idle_minutes", 10))) * 60
        self.idle_since = time.time()
        while not self.stop.is_set():
            try:
                msgs = await sub.fetch(1, timeout=15)
            except (nats.errors.TimeoutError, asyncio.TimeoutError):
                if time.time() - self.idle_since > idle_s:
                    print(f"[worker] idle for {idle_s / 60:.0f} min, leaving", flush=True)
                    return
                continue
            msg = msgs[0]
            job = json.loads(msg.data)
            jid = job["job_id"]
            rec = await self.job_get(jid)
            if rec.get("status") == "cancelled":
                await msg.ack()
                continue
            self.job_id = jid
            await self.job_put(jid, status="running", vm=self.vm, started_at=now(),
                               attempt=msg.metadata.num_delivered)

            async def keep_alive():  # JetStream redelivers after ack_wait unless told otherwise
                while True:
                    await asyncio.sleep(60)
                    await msg.in_progress()

            ka = asyncio.create_task(keep_alive())
            try:
                result = await self.run_job({**pool, **job}, job)
            finally:
                ka.cancel()
            status = ("cancelled" if jid in self.cancelled
                      else "done" if result["exitcode"] == 0 else "failed")
            await self.job_put(jid, status=status, finished_at=now(), exitcode=result["exitcode"],
                               out_file=result["out_file"], timings=result["timings"],
                               error=result.get("error"))
            await msg.ack()
            self.last_result = result
            await self.pub("result", result)
            self.idle_since = time.time()

    # -- LLM mode ----------------------------------------------------------------
    async def llm_loop(self, pool: dict) -> None:
        """Serve the pool's model with `vllm serve`; answer OpenAI requests from NATS."""
        import aiohttp

        model, env = pool["profile"], pool["env"]
        idle_s = self.idle_limit = float(os.environ.get("IDLE_MINUTES", pool.get("idle_minutes", 10))) * 60
        self.log_tag = f"llm:{model}"
        self.llm = {"model": model, "state": "preparing", "inflight": 0, "requests": 0}
        t0 = time.time()
        await self.prepare(pool)
        self.llm["state"] = "loading"
        # Chat is text only (images go through batch jobs), so skip the vision encoder.
        args = ["vllm", "serve", env["MODEL"], "--served-model-name", model,
                "--host", "127.0.0.1", "--port", str(LLM_PORT),
                "--max-model-len", env.get("MAX_MODEL_LEN", "8192"),
                "--gpu-memory-utilization", env.get("GPU_MEM_UTIL", "0.90"),
                "--max-num-seqs", env.get("MAX_NUM_SEQS", "256"),
                "--language-model-only",
                "--default-chat-template-kwargs",
                json.dumps({"enable_thinking": env.get("ENABLE_THINKING", "0") == "1"})]
        if env.get("QUANTIZATION"):
            args += ["--quantization", env["QUANTIZATION"]]
        if env.get("ENFORCE_EAGER") == "1":
            args.append("--enforce-eager")
        self.server = await asyncio.create_subprocess_exec(
            *args, cwd=AGENT, env=self.env, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, limit=1 << 20)

        async def pump():
            async for raw in self.server.stdout:
                await self.out_line(raw.decode(errors="replace").rstrip("\n"))
            await self.flush_log()

        pumping = asyncio.create_task(pump())
        base = f"http://127.0.0.1:{LLM_PORT}"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as http:
            while True:
                if self.stop.is_set():
                    return
                if self.server.returncode is not None:
                    raise RuntimeError(f"vllm serve exited with {self.server.returncode}")
                try:
                    async with http.get(f"{base}/health") as r:
                        if r.status == 200:
                            break
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(2)
            self.llm.update(state="serving", load_s=round(time.time() - t0, 1))
            await self.out_line(f"[worker] {model} serving after {time.time() - t0:.0f}s")
            await self.flush_log()
            if self.env_built:  # the caches it just built (JIT kernels, venv) go to Drive
                self.env_built = False
                asyncio.create_task(self.stream(f"python3 {AGENT}/vm_env.py finalize", AGENT, self.env))

            self.idle_since = time.time()
            tasks: set[asyncio.Task] = set()

            async def on_req(msg):  # callbacks run one at a time; vLLM batches concurrent requests
                self.idle_since = time.time()
                t = asyncio.create_task(self.llm_request(http, base, model, msg))
                tasks.add(t)
                t.add_done_callback(tasks.discard)

            sub = await self.nc.subscribe(f"edges.{self.edge}.colab-link.llm.{model}.>",
                                          queue=f"llm-{model}", cb=on_req)
            while not self.stop.is_set():
                await asyncio.sleep(5)
                if self.server.returncode is not None:
                    raise RuntimeError(f"vllm serve exited with {self.server.returncode}")
                if tasks:
                    self.idle_since = time.time()
                elif time.time() - self.idle_since > idle_s:
                    print(f"[worker] no requests for {idle_s / 60:.0f} min, leaving", flush=True)
                    break
            await sub.unsubscribe()
            self.llm["state"] = "leaving"
            if tasks:
                await asyncio.wait(tasks, timeout=30)
        self.server.terminate()
        await asyncio.wait([pumping], timeout=10)

    async def llm_request(self, http, base: str, model: str, msg) -> None:
        if not msg.reply:
            return
        # edges.<edge>.colab-link.llm.<model>.<route>
        route = LLM_ROUTES.get(msg.subject.split(".", 5)[-1])
        self.llm["inflight"] += 1
        self.llm["requests"] += 1
        try:
            if route is None:
                raise LookupError(f"unknown endpoint {msg.subject}")
            method, path = route
            body = json.loads(msg.data) if msg.data else {}
            body["model"] = model
            async with http.request(method, base + path, json=body if method == "POST" else None) as r:
                if r.status == 200 and body.get("stream"):
                    async for line in r.content:  # server-sent events: "data: {...}"
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            break
                        await self.nc.publish(msg.reply, data)
                    await self.nc.publish(msg.reply, b"")
                else:
                    data = await r.read()
                    hdr = None if r.status == 200 else {"Nats-Service-Error-Code": str(r.status),
                                                        "Nats-Service-Error": _hdr(data.decode(errors="replace"))}
                    await self.nc.publish(msg.reply, data, headers=hdr)
        except Exception as e:
            code = "404" if isinstance(e, LookupError) else "400" if isinstance(e, ValueError) else "502"
            await self.nc.publish(msg.reply, json.dumps({"error": {"message": str(e), "type": "colab_link",
                                                                   "code": int(code)}}).encode(),
                                  headers={"Nats-Service-Error-Code": code, "Nats-Service-Error": _hdr(str(e))})
        finally:
            self.llm["inflight"] -= 1

    async def job_get(self, jid: str) -> dict:
        try:
            return json.loads((await self.kv.get(f"jobs.{jid}")).value)
        except Exception:
            return {}

    async def job_put(self, jid: str, **fields) -> None:
        rec = await self.job_get(jid)
        rec.update(fields)
        await self.kv.put(f"jobs.{jid}", json.dumps(rec).encode())

    # -- job execution -------------------------------------------------------
    async def prepare(self, spec: dict) -> None:
        """venv/JIT caches + pip, once per worker."""
        env = {**os.environ, **spec.get("env", {}), "TORCHINDUCTOR_CACHE_DIR": INDUCTOR_DIR,
               "VM_ENV_STATE": str(AGENT / "env_state.json")}
        if not spec.get("cache", True):
            env.pop("DRIVE_QUOTA_PROJECT", None)  # vm_env skips Drive caches without it
        venv = spec.get("venv")
        python = "python3"
        if venv:
            cmd = (f"python3 {AGENT}/vm_env.py prepare --venv {shlex.quote(venv['name'])} "
                   f"--packages {shlex.quote(' '.join(venv['packages']))}"
                   + (f" --jit {shlex.quote(venv['jit'])}" if venv.get("jit") else ""))
            if await self.stream(cmd, AGENT, env):
                raise RuntimeError("environment prepare failed")
            python = VENV_PY
            env["PATH"] = f"/content/venv/bin:{env['PATH']}"
            st = json.loads((AGENT / "env_state.json").read_text())
            self.env_built = not st["venv_restored"] or (st["jit_key"] and not st["jit_restored"])
        if spec.get("pip"):
            pip_target = f"--python {VENV_PY}" if venv else "--system"
            cmd = f"uv pip install -q {pip_target} " + " ".join(map(shlex.quote, spec["pip"]))
            if await self.stream(cmd, AGENT, env):
                raise RuntimeError("pip install failed")
        self.python, self.env = python, env

    async def run_job(self, spec: dict, job: dict) -> dict:
        jid = job["job_id"]
        work = Path(f"/content/jobs/{jid}")
        (work / "in").mkdir(parents=True, exist_ok=True)
        (work / "out").mkdir(exist_ok=True)
        self.log_file = open(work / "out" / "log.txt", "w")
        timings: dict[str, float] = {}
        exitcode, error = 1, None
        t0 = time.time()
        try:
            if self.env is None:
                await self.prepare(spec)
                timings["env_s"] = round(time.time() - t0, 1)

            t = time.time()
            await asyncio.to_thread(drive_vm.get, job["in_file"], str(work / "in.tar.gz"), 16, JOBS_ROOT)
            with tarfile.open(work / "in.tar.gz") as tf:
                tf.extractall(work / "in", filter="data")
            (work / "in.tar.gz").unlink()
            timings["input_s"] = round(time.time() - t, 1)

            t = time.time()
            watcher = asyncio.create_task(self.watch_progress(jid, work / "out" / "progress.json"))
            if self.pool and re.search(r"^def handle\(", spec["script"], re.M):
                exitcode = await self.run_hosted(spec, work)
            else:
                (work / "job.py").write_text(spec["script"])
                exitcode = await self.stream(f"{self.python} job.py in out", work, self.env)
            watcher.cancel()
            timings["job_s"] = round(time.time() - t, 1)

            if self.env_built and exitcode == 0:
                t = time.time()
                await self.stream(f"python3 {AGENT}/vm_env.py finalize", AGENT, self.env)
                self.env_built = False
                timings["finalize_s"] = round(time.time() - t, 1)
        except Exception:
            error = traceback.format_exc()
            self.log_file.write(error)
            await self.pub("log", {"job_id": jid, "lines": error.splitlines()})
            if self.env is None:
                self.stop.set()  # a worker that cannot prepare its environment is useless
        finally:
            await self.flush_log()
            self.log_file.close()
            self.log_file = None
            out_file = None
            try:
                t = time.time()
                if (AGENT / "env_state.json").exists():
                    (work / "out" / "env_state.json").write_text((AGENT / "env_state.json").read_text())
                out_tar = work / "out.tar.gz"
                with tarfile.open(out_tar, "w:gz") as tf:
                    tf.add(work / "out", arcname=".")
                out_file = f"job-{jid}-out.tar.gz"
                await asyncio.to_thread(drive_vm.put, str(out_tar), out_file, JOBS_ROOT)
                timings["output_s"] = round(time.time() - t, 1)
            except Exception:
                await self.pub("log", {"job_id": jid, "lines": traceback.format_exc().splitlines()})
            timings["total_s"] = round(time.time() - t0, 1)
            self.job_id = None
            await asyncio.to_thread(_rmtree, work)
        if jid in self.cancelled:
            exitcode = 130
        return {"job_id": jid, "exitcode": exitcode, "out_file": out_file, "timings": timings,
                "error": error[-2000:] if error else None}

    async def run_hosted(self, spec: dict, work: Path) -> int:
        """Run the job through jobhost.py, starting it on first use."""
        if not self.host or self.host.returncode is not None:
            script = AGENT / "job.py"
            script.write_text(spec["script"])
            self.host = await asyncio.create_subprocess_exec(
                self.python, str(AGENT / "jobhost.py"), str(script), cwd=AGENT, env=self.env,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT, limit=1 << 20)
            self.host_reader = asyncio.create_task(self.read_host(self.host))
        self.host_done = asyncio.get_running_loop().create_future()
        self.host.stdin.write((json.dumps({"in": str(work / "in"), "out": str(work / "out"),
                                           "cwd": str(work)}) + "\n").encode())
        await self.host.stdin.drain()
        return await self.host_done

    async def read_host(self, host) -> None:
        async for raw in host.stdout:
            line = raw.decode(errors="replace").rstrip("\n")
            if line.startswith(DONE):
                ok = json.loads(line[len(DONE):]).get("ok")
                if self.host_done and not self.host_done.done():
                    self.host_done.set_result(0 if ok else 1)
                continue
            await self.out_line(line)
        code = await host.wait()
        if self.host_done and not self.host_done.done():
            self.host_done.set_result(code or 1)

    async def kill_host(self) -> None:
        if self.host and self.host.returncode is None:
            self.host.kill()
            await self.host.wait()
        if self.host_reader:
            await asyncio.wait([self.host_reader], timeout=5)

    async def out_line(self, line: str) -> None:
        """Tee one output line to log.txt and to .log in ~1 s batches."""
        if self.log_file:
            self.log_file.write(line + "\n")
        print(line, flush=True)
        self.batch.append(line[:500])
        if time.time() - self.batch_t > 1 or len(self.batch) >= 200:
            await self.flush_log()

    async def flush_log(self) -> None:
        if self.batch and (self.job_id or self.log_tag):
            await self.pub("log", {"job_id": self.job_id or self.log_tag, "lines": self.batch})
        self.batch, self.batch_t = [], time.time()
        if self.log_file:
            self.log_file.flush()

    async def stream(self, cmd: str, cwd: Path, env: dict) -> int:
        """Run a shell command with its output going to the current job's log."""
        self.proc = await asyncio.create_subprocess_shell(
            cmd, cwd=cwd, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            limit=1 << 20)
        async for raw in self.proc.stdout:
            await self.out_line(raw.decode(errors="replace").rstrip("\n"))
        await self.flush_log()
        return await self.proc.wait()

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


def _hdr(s: str) -> str:
    """A NATS header value: one line, short."""
    return " ".join(s.split())[:200]


def _rmtree(p: Path) -> None:
    import shutil

    shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(Worker().run())

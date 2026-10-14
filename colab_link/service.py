"""colab-link service: NATS micro service + JetStream work queue + warm Colab workers.

    colab-link serve

Endpoints (micro service "colab-link", group edges.<edge>.colab-link, JSON in/out):
  run      {profile? | gpu?, script?, env?, pip?, cache?, idle_minutes?, in_file}  -> job record
           in_file is a tar.gz already in Drive:<jobs_root> (put there by the caller;
           files never travel over NATS).
  status   {job_id} -> job record   |   {} -> recent job records
  cancel   {job_id} -> job record
  workers  {} -> live workers

A pool is everything that must match for a worker to take a job: GPU, script,
env, venv, pip. Each pool has a durable pull consumer on the work queue; this
service starts workers (Colab VMs running worker.py) for pools with pending
jobs, within Colab's limits, and stops a VM when its worker says .bye (idle),
stops sending heartbeats, or the service shuts down.

JetStream (per edge):
  stream COLAB_LINK_<edge>   subjects edges.<edge>.colab-link.queue.<pool>  (work queue)
  KV     colab-link-<edge>   jobs.<job_id> (job records), pools.<pool_id> (pool specs)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import config
from .nats_runner import FLAREX, NATS_URL, launch_agent, log
from .session import ColabSession, drive_token

STATE_DIR = Path(os.environ.get("COLAB_LINK_STATE", Path.home() / ".cache" / "colab-link" / "service-state"))
HEARTBEAT_TIMEOUT_S = 90
READY_TIMEOUT_S = 600          # agent start incl. pip install of nats-py
ACK_WAIT_S = 180               # workers send in_progress every 60 s; a dead VM's job is redelivered after this
MAX_DELIVER = 3
MAX_PER_GPU = 2                # Colab: at most 2 sessions per GPU type ...
MAX_TOTAL = 5                  # ... and 5 in total (measured 2026-10-14)
START_BACKOFF_S = 120          # after a failed VM start for a pool
TOKEN_REFRESH_S = 40 * 60      # Drive access tokens last ~1 h
RECORD_TTL_S = 7 * 86400       # job records and queued messages
DRIVE_RETENTION_S = 2 * 86400  # job files left in Drive:<jobs_root> (normally the caller removes them)
TERMINAL = {"done", "failed", "cancelled"}


def now() -> float:
    return round(time.time(), 1)


@dataclass
class Worker:
    vm: str
    pool: str
    gpu: str | None
    session: ColabSession
    state: str = "starting"    # starting | idle | busy | stopping
    started_at: float = field(default_factory=time.time)
    last_beat: float = 0.0
    job_id: str | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)

    def view(self) -> dict:
        return {"vm": self.vm, "pool": self.pool, "gpu": self.gpu, "state": self.state,
                "job_id": self.job_id, "up_s": round(time.time() - self.started_at),
                "last_beat_s": round(time.time() - self.last_beat, 1) if self.last_beat else None}


class Service:
    def __init__(self, cfg: config.Config, flarex: Path = FLAREX):
        self.cfg = cfg
        self.flarex = flarex
        self.edge = (flarex / "id").read_text().strip()
        self.topic = f"edges.{self.edge}.colab-link"
        self.stream = f"COLAB_LINK_{self.edge}"
        self.bucket = f"colab-link-{self.edge}"
        self.workers: dict[str, Worker] = {}
        self.pools: dict[str, dict] = {}
        self.backoff: dict[str, float] = {}
        self.wake = asyncio.Event()
        self.stopping = asyncio.Event()
        self.token = ""
        self.token_t = 0.0
        self.nc = self.js = self.kv = None
        STATE_DIR.mkdir(parents=True, exist_ok=True)

    # -- setup -----------------------------------------------------------------
    async def run(self) -> None:
        import nats
        from nats.js import api
        from nats.js.errors import BucketNotFoundError, NotFoundError
        from nats.micro import add_service

        await self.refresh_token()
        self.nc = await nats.connect(NATS_URL, user_credentials=str(self.flarex / "user.creds"),
                                     name=f"Colab Link - {self.edge}", max_reconnect_attempts=-1)
        self.js = self.nc.jetstream()
        try:
            await self.js.stream_info(self.stream)
        except NotFoundError:
            await self.js.add_stream(name=self.stream, subjects=[f"{self.topic}.queue.>"],
                                     retention=api.RetentionPolicy.WORK_QUEUE,
                                     storage=api.StorageType.FILE, max_age=RECORD_TTL_S)
        try:
            self.kv = await self.js.key_value(self.bucket)
        except BucketNotFoundError:
            self.kv = await self.js.create_key_value(bucket=self.bucket, history=1, ttl=RECORD_TTL_S)
        for key in await self.keys("pools."):
            try:
                spec = json.loads((await self.kv.get(key)).value)
                self.pools[spec["pool_id"]] = spec
            except Exception:
                pass

        for kind, cb in [("ready", self.on_ready), ("heartbeat", self.on_heartbeat), ("bye", self.on_bye)]:
            await self.nc.subscribe(f"{self.topic}.workers.*.{kind}", cb=cb)

        svc = await add_service(self.nc, name="colab-link", version="0.2.0",
                                description="Run Python jobs on rented Colab GPUs")
        group = svc.add_group(name=self.topic)
        for name, handler in [("run", self.ep_run), ("status", self.ep_status),
                              ("cancel", self.ep_cancel), ("workers", self.ep_workers)]:
            await group.add_endpoint(name=name, handler=handler)

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.stopping.set)
        log(f"serving on {self.topic}.{{run,status,cancel,workers}} ({len(self.pools)} known pools)")
        tasks = [asyncio.create_task(t) for t in (self.scale_loop(), self.monitor_loop(),
                                                    self.cleanup_loop(), self.reap_orphans())]
        await self.stopping.wait()

        log("shutting down: stopping all workers")
        for t in tasks:
            t.cancel()
        await svc.stop()
        await asyncio.gather(*(self.stop_worker(w, "service shutdown") for w in list(self.workers.values())),
                             return_exceptions=True)
        await self.nc.drain()

    async def keys(self, prefix: str) -> list[str]:
        # kv.keys(filters=["jobs.*"]) finds nothing with nats-py 2.16 + server 2.11; filter here.
        try:
            return [k for k in await self.kv.keys() if k.startswith(prefix)]
        except Exception:  # NoKeysError on an empty bucket
            return []

    async def refresh_token(self) -> None:
        self.token = await asyncio.to_thread(drive_token)
        self.token_t = time.time()
        os.environ.update(DRIVE_TOKEN=self.token, DRIVE_QUOTA_PROJECT=self.cfg.drive_project or "",
                          DRIVE_ROOT=self.cfg.drive_root)

    # -- pools -----------------------------------------------------------------
    def pool_for(self, req: dict) -> dict:
        name = req.get("profile")
        prof = self.cfg.profiles.get(name) if name else None
        if name and not prof:
            raise ValueError(f"unknown profile {name!r}")
        script = req.get("script")
        if not script and prof and prof.job:
            script = (config.ROOT / prof.job).read_text()
        if not script:
            raise ValueError("no job script: pass `script` or a profile that has one")
        gpu = req.get("gpu") or (prof.gpu if prof else None)
        env = {**(prof.env if prof else {}), **{k: str(v) for k, v in (req.get("env") or {}).items()}}
        venv = ({"name": prof.venv, "packages": prof.packages,
                 "jit": prof.jit_id({**env, "_gpu": gpu or "CPU"})} if prof and prof.venv else None)
        spec = {"profile": name, "gpu": gpu, "script": script, "env": env, "venv": venv,
                "pip": list(req.get("pip") or []), "cache": bool(req.get("cache", True))}
        digest = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
        spec["pool_id"] = f"{name or 'custom'}-{digest}"
        spec["idle_minutes"] = float(req.get("idle_minutes") or (prof.idle_minutes if prof else 10))
        spec["max_workers"] = int(prof.max_workers if prof else 1)
        return spec

    async def ensure_pool(self, spec: dict) -> None:
        from nats.js import api

        pid = spec["pool_id"]
        await self.kv.put(f"pools.{pid}", json.dumps(spec).encode())
        if pid not in self.pools:
            try:
                await self.js.consumer_info(self.stream, f"pool-{pid}")
            except Exception:
                await self.js.add_consumer(self.stream, durable_name=f"pool-{pid}",
                                           filter_subject=f"{self.topic}.queue.{pid}",
                                           ack_policy=api.AckPolicy.EXPLICIT, ack_wait=ACK_WAIT_S,
                                           max_deliver=MAX_DELIVER, inactive_threshold=RECORD_TTL_S)
        self.pools[pid] = spec

    # -- endpoints ---------------------------------------------------------------
    async def ep_run(self, req) -> None:
        try:
            body = json.loads(req.data or b"{}")
            if not body.get("in_file"):
                raise ValueError("in_file (a tar.gz in Drive:" + self.cfg.drive_jobs_root + ") is required")
            spec = self.pool_for(body)
        except Exception as e:
            await req.respond_error("400", str(e))
            return
        await self.ensure_pool(spec)
        jid = f"{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        rec = {"job_id": jid, "pool": spec["pool_id"], "profile": spec["profile"], "gpu": spec["gpu"],
               "status": "queued", "submitted_at": now(), "in_file": body["in_file"], "attempt": 0}
        await self.kv.put(f"jobs.{jid}", json.dumps(rec).encode())
        await self.js.publish(f"{self.topic}.queue.{spec['pool_id']}",
                              json.dumps({"job_id": jid, "in_file": body["in_file"]}).encode(),
                              headers={"Nats-Msg-Id": jid})
        log(f"queued {jid} -> pool {spec['pool_id']}")
        self.wake.set()
        await req.respond(json.dumps(rec).encode())

    async def get_job(self, jid: str) -> dict | None:
        try:
            return json.loads((await self.kv.get(f"jobs.{jid}")).value)
        except Exception:
            return None

    async def ep_status(self, req) -> None:
        body = json.loads(req.data or b"{}")
        if body.get("job_id"):
            rec = await self.get_job(body["job_id"])
            if rec is None:
                await req.respond_error("404", f"no job {body['job_id']}")
            else:
                await req.respond(json.dumps(rec).encode())
            return
        recs = [r for r in [await self.get_job(k.split(".", 1)[1]) for k in await self.keys("jobs.")] if r]
        recs.sort(key=lambda r: r.get("submitted_at", 0), reverse=True)
        await req.respond(json.dumps(recs[: int(body.get("limit", 50))]).encode())

    async def ep_cancel(self, req) -> None:
        jid = json.loads(req.data or b"{}").get("job_id", "")
        rec = await self.get_job(jid)
        if rec is None:
            await req.respond_error("404", f"no job {jid}")
            return
        if rec["status"] == "queued":
            rec.update(status="cancelled", finished_at=now())  # the worker acks and skips it
            await self.kv.put(f"jobs.{jid}", json.dumps(rec).encode())
        elif rec["status"] == "running" and rec.get("vm"):
            await self.nc.publish(f"{self.topic}.workers.{rec['vm']}.ctl",
                                  json.dumps({"op": "cancel", "job_id": jid}).encode())
            rec["cancel_requested"] = True
        await req.respond(json.dumps(rec).encode())

    async def ep_workers(self, req) -> None:
        await req.respond(json.dumps([w.view() for w in self.workers.values()]).encode())

    # -- worker events -----------------------------------------------------------
    @staticmethod
    def vm_of(msg) -> str:
        return msg.subject.split(".")[-2]

    async def on_ready(self, msg) -> None:
        w = self.workers.get(self.vm_of(msg))
        if w:
            w.last_beat = time.time()
            w.state = "idle"
            w.ready.set()

    async def on_heartbeat(self, msg) -> None:
        data = json.loads(msg.data)
        if not data.get("pool"):
            return  # a one-shot worker (colab-link run); not ours
        vm = self.vm_of(msg)
        w = self.workers.get(vm)
        if w is None:
            if not (STATE_DIR / f"{vm}.json").exists():
                return  # started by another colab-link instance
            # We restarted while this VM kept running: adopt it.
            sess = ColabSession(vm, gpu=data.get("gpu") or None, state_dir=STATE_DIR)
            sess.started = True
            w = self.workers[vm] = Worker(vm, data["pool"], data.get("gpu") or None, sess)
            w.ready.set()
            log(f"adopted running worker {vm} (pool {w.pool})")
        if w.state == "stopping":
            return
        w.last_beat = time.time()
        w.state = "busy" if data.get("busy") else "idle"
        w.job_id = data.get("job_id")

    async def on_bye(self, msg) -> None:
        w = self.workers.get(self.vm_of(msg))
        if w and w.state != "stopping":
            asyncio.create_task(self.stop_worker(w, "idle"))

    # -- scaling -----------------------------------------------------------------
    def capacity(self, gpu: str | None) -> bool:
        live = [w for w in self.workers.values() if w.state != "stopping"]
        return len(live) < MAX_TOTAL and sum(w.gpu == gpu for w in live) < MAX_PER_GPU

    async def scale_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self.wake.wait(), 5)
            except asyncio.TimeoutError:
                pass
            self.wake.clear()
            for pid, spec in list(self.pools.items()):
                try:
                    pending = (await self.js.consumer_info(self.stream, f"pool-{pid}")).num_pending
                except Exception:
                    continue
                ws = [w for w in self.workers.values() if w.pool == pid and w.state != "stopping"]
                free = [w for w in ws if w.state in ("starting", "idle")]
                if pending <= len(free) or len(ws) >= spec["max_workers"]:
                    continue
                if time.time() < self.backoff.get(pid, 0):
                    continue
                if not self.capacity(spec["gpu"]):
                    # Free a slot held by an idle worker of another pool on the same GPU type.
                    idle = [w for w in self.workers.values()
                            if w.state == "idle" and w.pool != pid and w.gpu == spec["gpu"]]
                    if idle:
                        log(f"pool {pid} waiting: asking idle {idle[0].vm} ({idle[0].pool}) to leave")
                        await self.nc.publish(f"{self.topic}.workers.{idle[0].vm}.ctl",
                                              json.dumps({"op": "shutdown"}).encode())
                        self.backoff[pid] = time.time() + 30
                    continue
                vm = f"cl-{uuid.uuid4().hex[:8]}"  # registered now so capacity() counts it
                w = self.workers[vm] = Worker(vm, pid, spec["gpu"],
                                              ColabSession(vm, gpu=spec["gpu"], state_dir=STATE_DIR))
                asyncio.create_task(self.start_worker(w, spec))

    async def start_worker(self, w: Worker, spec: dict) -> None:
        vm, pid, sess = w.vm, spec["pool_id"], w.session
        log(f"starting {vm} gpu={spec['gpu'] or 'CPU'} for pool {pid}")
        try:
            await asyncio.to_thread(sess.start)
            await asyncio.to_thread(sess.wait_ready)
            await asyncio.to_thread(
                launch_agent, sess, vm, spec["gpu"], self.cfg.drive_project, self.cfg.drive_root,
                self.cfg.drive_jobs_root, self.token,
                {"POOL_ID": pid, "IDLE_MINUTES": str(spec["idle_minutes"])}, self.flarex)
            await asyncio.wait_for(w.ready.wait(), READY_TIMEOUT_S)
            log(f"{vm} ready in {time.time() - w.started_at:.0f}s")
        except Exception as e:
            log(f"{vm} failed to start: {str(e)[:500]}")
            self.backoff[pid] = time.time() + START_BACKOFF_S
            await self.stop_worker(w, "start failed")

    async def stop_worker(self, w: Worker, reason: str) -> None:
        if w.state == "stopping":
            return
        w.state = "stopping"
        log(f"stopping {w.vm} ({reason})")
        try:
            await self.nc.publish(f"{self.topic}.workers.{w.vm}.ctl", json.dumps({"op": "shutdown"}).encode())
        except Exception:
            pass
        try:
            await asyncio.to_thread(w.session.stop)
        finally:
            w.session.forget()
            self.workers.pop(w.vm, None)
        if w.job_id:
            rec = await self.get_job(w.job_id)
            if rec and rec.get("status") == "running" and rec.get("vm") == w.vm:
                rec.update(status="queued", vm=None,
                           note=f"worker {w.vm} lost ({reason}); redelivered after {ACK_WAIT_S}s")
                await self.kv.put(f"jobs.{w.job_id}", json.dumps(rec).encode())
        self.wake.set()

    # -- housekeeping --------------------------------------------------------------
    async def monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(10)
            t = time.time()
            for w in list(self.workers.values()):
                if w.state in ("idle", "busy") and t - w.last_beat > HEARTBEAT_TIMEOUT_S:
                    asyncio.create_task(self.stop_worker(w, f"no heartbeat for {t - w.last_beat:.0f}s"))
            if t - self.token_t > TOKEN_REFRESH_S:
                try:
                    await self.refresh_token()
                    for w in self.workers.values():
                        if w.state in ("idle", "busy"):
                            await self.nc.publish(f"{self.topic}.workers.{w.vm}.ctl", json.dumps(
                                {"op": "drive_token", "token": self.token}).encode())
                except Exception as e:
                    log(f"Drive token refresh failed: {e}")

    async def reap_orphans(self) -> None:
        """VMs we started before a restart whose worker no longer heartbeats."""
        await asyncio.sleep(3 * 5 + 15)
        for f in STATE_DIR.glob("*.json"):
            if f.stem in self.workers:
                continue
            sess = ColabSession(f.stem, state_dir=STATE_DIR)
            try:
                live = json.loads(f.read_text() or "{}") != {}
            except ValueError:
                live = False
            if live:
                log(f"stopping orphaned VM {f.stem}")
                sess.started = True
                await asyncio.to_thread(sess.stop)
            sess.forget()

    async def cleanup_loop(self) -> None:
        """Drop job files the caller never collected."""
        from . import drive_vm

        while True:
            try:
                for f in await asyncio.to_thread(drive_vm.ls, self.cfg.drive_jobs_root):
                    age = time.time() - datetime.fromisoformat(f["modifiedTime"].replace("Z", "+00:00")).timestamp()
                    if f["name"].startswith("job-") and age > DRIVE_RETENTION_S:
                        await asyncio.to_thread(drive_vm.rm, f["name"], self.cfg.drive_jobs_root)
                        log(f"removed stale Drive:{self.cfg.drive_jobs_root}/{f['name']}")
            except Exception as e:
                log(f"Drive cleanup failed: {e}")
            await asyncio.sleep(3600)


def main(cfg: config.Config) -> int:
    asyncio.run(Service(cfg).run())
    return 0

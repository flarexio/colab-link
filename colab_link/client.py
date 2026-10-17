"""Talk to a running `colab-link serve` over NATS (what the HTTP API will do later).

submit: tar the input -> Drive:<jobs_root> -> request `run`; with --wait, stream
logs and status, fetch the output from Drive and remove the job files.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

from . import config
from .nats_runner import FLAREX, NATS_URL, _drive, log

TERMINAL = {"done", "failed", "cancelled"}


class ServiceError(Exception):
    """The colab-link service is not there or refused the request."""


async def _connect():
    import nats

    edge = (FLAREX / "id").read_text().strip()
    nc = await nats.connect(NATS_URL, user_credentials=str(FLAREX / "user.creds"),
                            name=f"Colab Link CLI - {edge}")
    return nc, f"edges.{edge}.colab-link"


async def call(endpoint: str, body: dict, timeout: float = 30):
    nc, topic = await _connect()
    try:
        return await _request(nc, f"{topic}.{endpoint}", body, timeout)
    finally:
        await nc.drain()


async def _request(nc, subject: str, body: dict, timeout: float = 30):
    from nats.errors import NoRespondersError, TimeoutError

    try:
        msg = await nc.request(subject, json.dumps(body).encode(), timeout=timeout)
    except (NoRespondersError, TimeoutError):
        raise ServiceError(f"no colab-link service answering on {subject}; is `colab-link serve` running?")
    if msg.headers and msg.headers.get("Nats-Service-Error-Code"):
        raise ServiceError(f"error {msg.headers['Nats-Service-Error-Code']}: {msg.headers.get('Nats-Service-Error')}")
    return json.loads(msg.data)


async def submit(cfg: config.Config, req: dict, input_dir: Path, out: Path | None, wait: bool) -> int:
    d = _drive(cfg.drive_project, cfg.drive_root)
    in_file = f"job-{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}-in.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        tar_in = Path(tmp) / "in.tar.gz"
        with tarfile.open(tar_in, "w:gz") as tf:
            for p in sorted(input_dir.iterdir()):
                tf.add(p, arcname=p.name)
        await asyncio.to_thread(d.put, str(tar_in), in_file, cfg.drive_jobs_root)
    log(f"input {input_dir} -> Drive:{cfg.drive_jobs_root}/{in_file}")

    nc, topic = await _connect()
    try:
        rec = await _request(nc, f"{topic}.run", {**req, "in_file": in_file})
        jid = rec["job_id"]
        log(f"job {jid} queued on pool {rec['pool']}")
        print(jid, flush=True)
        if not wait:
            return 0

        async def on_msg(msg):
            data = json.loads(msg.data)
            if data.get("job_id") != jid:
                return
            if msg.subject.endswith(".log"):
                for line in data["lines"]:
                    log(f"  vm| {line[:200]}")
            else:
                log(f"  progress: {json.dumps({k: v for k, v in data.items() if k != 'job_id'})[:200]}")

        await nc.subscribe(f"{topic}.workers.*.log", cb=on_msg)
        await nc.subscribe(f"{topic}.workers.*.progress", cb=on_msg)
        last = None
        while True:
            rec = await _request(nc, f"{topic}.status", {"job_id": jid})
            state = (rec["status"], rec.get("vm"))
            if state != last:
                log(f"job {jid}: {rec['status']}" + (f" on {rec['vm']}" if rec.get("vm") else "")
                    + (f" ({rec['note']})" if rec.get("note") else ""))
                last = state
            if rec["status"] in TERMINAL:
                break
            await asyncio.sleep(3)
    finally:
        await nc.drain()

    return await collect(cfg, rec, out)


async def fetch(cfg: config.Config, jid: str, out: Path | None) -> int:
    """Collect a finished job submitted without --wait (or whose client went away)."""
    rec = await call("status", {"job_id": jid})
    if rec["status"] not in TERMINAL:
        sys.exit(f"job {jid} is {rec['status']}; try again later")
    _drive(cfg.drive_project, cfg.drive_root)
    return await collect(cfg, rec, out)


async def collect(cfg: config.Config, rec: dict, out: Path | None) -> int:
    """Download a finished job's output from Drive, then remove its Drive files."""
    from . import drive_vm as d

    jid = rec["job_id"]
    out = out or config.ROOT / "results" / f"{rec.get('profile') or 'job'}-{jid}"
    out.mkdir(parents=True, exist_ok=True)
    if rec.get("out_file"):
        with tempfile.TemporaryDirectory() as tmp:
            tar_out = Path(tmp) / "out.tar.gz"
            await asyncio.to_thread(d.get, rec["out_file"], str(tar_out), 16, cfg.drive_jobs_root)
            with tarfile.open(tar_out) as tf:
                tf.extractall(out, filter="data")
    for name in (rec.get("in_file"), rec.get("out_file")):
        if name:
            try:
                await asyncio.to_thread(d.rm, name, cfg.drive_jobs_root)
            except Exception as e:
                log(f"could not remove Drive:{cfg.drive_jobs_root}/{name}: {e}")
    (out / "job.json").write_text(json.dumps(rec, indent=2))
    log(f"job {jid} {rec['status']} (exit {rec.get('exitcode')}); output in {out}")
    log(f"timings: {json.dumps(rec.get('timings'))}")
    return rec.get("exitcode") if rec.get("exitcode") is not None else 1


# -- LLM (OpenAI bodies over NATS; see worker.py llm_loop) ---------------------------
class NoWorker(Exception):
    """Nobody serves this model right now (NATS: no responders)."""


class LLMError(Exception):
    def __init__(self, status: int, body: bytes):
        self.status, self.body = status, body
        super().__init__(f"{status}: {body[:300].decode(errors='replace')}")


async def llm_request(nc, topic: str, model: str, route: str, body: dict, timeout: float = 600):
    """Yield OpenAI objects: the response, or with body["stream"] each chunk."""
    data = json.dumps(body).encode()
    if len(data) > nc.max_payload - 4096:
        raise LLMError(413, f"request is {len(data)} bytes; NATS allows {nc.max_payload}".encode())
    inbox = nc.new_inbox()
    sub = await nc.subscribe(inbox)
    try:
        await nc.publish(f"{topic}.llm.{model}.{route}", data, reply=inbox)
        while True:
            msg = await sub.next_msg(timeout=timeout)
            hdr = msg.headers or {}
            if hdr.get("Status") == "503":
                raise NoWorker(model)
            if hdr.get("Nats-Service-Error-Code"):
                raise LLMError(int(hdr["Nats-Service-Error-Code"]), msg.data)
            if not msg.data:
                return  # end of stream
            yield json.loads(msg.data)
            if not body.get("stream"):
                return
    finally:
        await sub.unsubscribe()


async def ensure_llm(nc, topic: str, model: str, wait_s: float = 1200, idle_minutes: float | None = None,
                     say=log) -> dict:
    """Ask the service for a worker serving `model` and wait until it serves."""
    view = await _request(nc, f"{topic}.llm_start", {"model": model, "idle_minutes": idle_minutes})
    t0, last = time.time(), None
    while True:
        states = [w["state"] for w in view["workers"]]
        if "serving" in states:
            return view
        if states != last:
            say(f"{model}: {', '.join(states) or 'waiting for a VM'} ({time.time() - t0:.0f}s)")
            last = states
        if not view["wanted"] and not view["workers"]:
            raise RuntimeError(f"{model} failed to start; see the colab-link serve log")
        if time.time() - t0 > wait_s:
            raise TimeoutError(f"{model} not serving after {wait_s:.0f}s")
        await asyncio.sleep(5)
        view = next(v for v in await _request(nc, f"{topic}.llm_list", {}) if v["model"] == model)


async def chat(model: str, messages: list[dict], start: bool = True, **params) -> int:
    """Stream one chat completion to stdout (colab-link chat)."""
    nc, topic = await _connect()
    body = {"messages": messages, "stream": True, "stream_options": {"include_usage": True}, **params}
    try:
        for attempt in range(2):
            try:
                usage, t0, first = None, time.time(), None
                async for chunk in llm_request(nc, topic, model, "chat.completions", body):
                    for c in chunk.get("choices", []):
                        text = (c.get("delta") or {}).get("content")
                        if text:
                            first = first or time.time()
                            sys.stdout.write(text)
                            sys.stdout.flush()
                    usage = chunk.get("usage") or usage
                print()
                if usage:
                    gen = time.time() - (first or t0)
                    log(f"{usage.get('prompt_tokens')} prompt + {usage.get('completion_tokens')} output tokens; "
                        f"first token {(first or t0) - t0:.1f}s, "
                        f"{(usage.get('completion_tokens') or 0) / max(gen, 1e-3):.0f} tok/s")
                return 0
            except NoWorker:
                if not start or attempt:
                    sys.exit(f"no worker serves {model}; run `colab-link llm start {model}`")
                log(f"{model} is not running; starting it on Colab (cold start takes several minutes)")
                await ensure_llm(nc, topic, model)
    except LLMError as e:
        sys.exit(f"error {e}")
    finally:
        await nc.drain()
    return 1

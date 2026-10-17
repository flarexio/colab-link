"""HTTP API in front of `colab-link serve`: upload files, run jobs, get results.

    colab-link api [--host 127.0.0.1] [--port 8080] [--api-key KEY]

Uploads land in Drive:<jobs_root> as the job's in.tar.gz, then the job is queued
over NATS (`run`); outputs come back from Drive on first download and are kept
under --data for RETENTION_S. Files never travel over NATS. The same server also
answers the OpenAI-compatible /v1 routes (see openai_gateway.py).

All routes except /api/health want `Authorization: Bearer <key>` when a key is
set; binding anything but loopback needs one.

    GET    /api/health
    GET    /api/profiles
    POST   /api/jobs                 multipart: profile | gpu, script, env, pip, cache,
                                     idle_minutes, files (repeatable), archive (.tar.*/.zip);
                                     or JSON {profile, ...} for a job without input
    GET    /api/jobs                 ?limit=50
    GET    /api/jobs/{id}            ?wait=SECONDS: hold until the job ends (max 600)
    DELETE /api/jobs/{id}            cancel
    GET    /api/jobs/{id}/events     SSE: status / log / progress, then end
    GET    /api/jobs/{id}/output     out.tar.gz
    GET    /api/jobs/{id}/files[/{path}]
    GET    /api/jobs/{id}/results    results.jsonl as a JSON list (vlm_batch, ocr)
    POST   /api/ocr                  multipart: images (repeatable), profile, prompt,
                                     max_tokens, idle_minutes -> a batch job
    GET    /api/workers | /api/usage | /api/llm
    POST   /api/llm/{model}/start    ?wait=1 holds until it serves | /api/llm/{model}/stop
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tarfile
import tempfile
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from aiohttp import web

from . import config
from .client import (IMAGE_EXT, TERMINAL, ServiceError, _connect, _request, ensure_llm,
                     write_ocr_input)
from .nats_runner import log
from .openai_gateway import Gateway, _error
from .session import _cli_env, drive_token

DATA_DIR = Path(os.environ.get("COLAB_LINK_API_DATA", Path.home() / ".cache" / "colab-link" / "api"))
RETENTION_S = 2 * 86400          # same as the service keeps uncollected files on Drive
TOKEN_REFRESH_S = 40 * 60        # Drive tokens last ~1 h
SSE = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}


def _rel(name: str) -> Path:
    """An uploaded filename as a safe relative path (sub/dirs allowed, no '..')."""
    parts = [p for p in PurePosixPath(name.replace("\\", "/")).parts if p not in ("/", ".", "")]
    if not parts or ".." in parts:
        raise ValueError(f"bad filename {name!r}")
    return Path(*parts)


def _form_bool(v: str | None, default: bool) -> bool:
    return default if v is None else v.strip().lower() not in ("0", "false", "no", "off", "")


class Api:
    def __init__(self, cfg: config.Config, api_key: str | None, start: bool, max_upload: int):
        self.cfg, self.api_key, self.max_upload = cfg, api_key, max_upload
        self.gw = Gateway(api_key, start, 1200)
        self.nc = self.topic = None
        self.token_t = 0.0
        self.collecting: dict[str, asyncio.Lock] = {}
        (DATA_DIR / "jobs").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "tmp").mkdir(parents=True, exist_ok=True)

    # -- lifecycle ---------------------------------------------------------------
    async def setup(self, app) -> None:
        self.nc, self.topic = await _connect()
        self.gw.nc, self.gw.topic = self.nc, self.topic
        os.environ.update(DRIVE_QUOTA_PROJECT=self.cfg.drive_project or "", DRIVE_ROOT=self.cfg.drive_root)
        await self.drive()
        app["prune"] = asyncio.create_task(self.prune_loop())

    async def cleanup(self, app) -> None:
        app["prune"].cancel()
        await self.nc.drain()

    async def drive(self):
        if time.time() - self.token_t > TOKEN_REFRESH_S:
            os.environ["DRIVE_TOKEN"] = await asyncio.to_thread(drive_token)
            self.token_t = time.time()
        from . import drive_vm
        return drive_vm

    async def prune_loop(self) -> None:
        while True:
            for d in [*(DATA_DIR / "jobs").iterdir(), *(DATA_DIR / "tmp").iterdir()]:
                if time.time() - d.stat().st_mtime > RETENTION_S:
                    shutil.rmtree(d, ignore_errors=True)
            await asyncio.sleep(3600)

    @web.middleware
    async def auth(self, request, handler):
        if request.path != "/api/health" and self.api_key \
                and request.headers.get("Authorization") != f"Bearer {self.api_key}":
            return _error(401, "invalid API key")
        try:
            return await handler(request)
        except ServiceError as e:
            return _error(e.code, str(e))
        except web.HTTPException:
            raise
        except Exception as e:  # Drive, the colab CLI: report, keep serving
            log(f"api: {request.method} {request.path}: {type(e).__name__}: {e}")
            return _error(502, f"{type(e).__name__}: {e}")

    async def call(self, endpoint: str, body: dict, timeout: float = 30):
        return await _request(self.nc, f"{self.topic}.{endpoint}", body, timeout)

    # -- uploads -----------------------------------------------------------------
    async def read_multipart(self, request, staging: Path) -> tuple[dict[str, list[str]], list[tuple[str, str, Path]]]:
        """Stream a multipart body to disk: ({field: [values]}, [(field, filename, path)])."""
        fields: dict[str, list[str]] = {}
        files: list[tuple[str, str, Path]] = []
        total = 0
        reader = await request.multipart()
        while (part := await reader.next()) is not None:
            if part.filename is None:
                buf = b""
                while chunk := await part.read_chunk():
                    buf += chunk
                    if len(buf) > 1 << 20:
                        raise ValueError(f"form field {part.name!r} is over 1 MB")
                fields.setdefault(part.name, []).append(buf.decode())
                continue
            dst = staging / "parts" / str(len(files))
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as f:
                while chunk := await part.read_chunk(1 << 20):
                    total += len(chunk)
                    if total > self.max_upload:
                        raise ValueError(f"upload is over {self.max_upload >> 20} MB")
                    f.write(chunk)
            files.append((part.name, part.filename, dst))
        return fields, files

    async def queue(self, req: dict, in_dir: Path) -> dict:
        """in_dir -> Drive in.tar.gz -> `run`. Returns the job record."""
        d = await self.drive()
        in_file = f"job-{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}-in.tar.gz"
        tar_in = in_dir.parent / "in.tar.gz"

        def pack():
            with tarfile.open(tar_in, "w:gz") as tf:
                for p in sorted(in_dir.iterdir()):
                    tf.add(p, arcname=p.name)

        await asyncio.to_thread(pack)
        await asyncio.to_thread(d.put, str(tar_in), in_file, self.cfg.drive_jobs_root)
        try:
            rec = await self.call("run", {**req, "in_file": in_file})
        except ServiceError:
            await asyncio.to_thread(d.rm, in_file, self.cfg.drive_jobs_root)
            raise
        log(f"api: job {rec['job_id']} queued on {rec['pool']} ({tar_in.stat().st_size / 1e6:.1f} MB in)")
        return rec

    async def staged(self, request, build, json_ok: bool = True) -> web.Response:
        """Run `build(fields, files, in_dir) -> run request` on an upload, then queue it."""
        staging = Path(tempfile.mkdtemp(dir=DATA_DIR / "tmp"))
        try:
            in_dir = staging / "in"
            in_dir.mkdir()
            try:
                if request.content_type == "application/json":
                    if not json_ok:
                        raise ValueError("send multipart/form-data")
                    body = await request.json()
                    if not isinstance(body, dict):
                        raise ValueError("JSON body must be an object")
                    req = {k: body.get(k) for k in ("profile", "gpu", "script", "env", "pip", "cache",
                                                    "idle_minutes") if body.get(k) is not None}
                else:
                    fields, files = await self.read_multipart(request, staging)
                    req = await asyncio.to_thread(build, fields, files, in_dir)
                if req.get("profile") and req["profile"] not in self.cfg.profiles:
                    raise ValueError(f"unknown profile {req['profile']!r}; see GET /api/profiles")
                if not req.get("profile") and not req.get("script"):
                    raise ValueError("pass a profile or a script")
            except (ValueError, tarfile.TarError, zipfile.BadZipFile) as e:
                return _error(400, str(e))
            rec = await self.queue(req, in_dir)
            return web.json_response(self.links(rec), status=202)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def job_request(fields: dict[str, list[str]], files: list[tuple[str, str, Path]], in_dir: Path) -> dict:
        one = lambda k: (fields.get(k) or [None])[-1]  # noqa: E731
        req: dict = {"profile": one("profile"), "gpu": one("gpu"), "script": one("script"),
                     "pip": fields.get("pip", []), "cache": _form_bool(one("cache"), True)}
        if one("idle_minutes"):
            req["idle_minutes"] = float(one("idle_minutes"))
        env: dict[str, str] = {}
        for v in fields.get("env", []):
            if v.lstrip().startswith("{"):
                env.update({k: str(x) for k, x in json.loads(v).items()})
            elif "=" in v:
                k, x = v.split("=", 1)
                env[k] = x
            else:
                raise ValueError(f"env must be KEY=VALUE or a JSON object, got {v!r}")
        req["env"] = env
        for name, filename, path in files:
            if name == "script":
                req["script"] = path.read_text()
            elif name == "files":
                dst = in_dir / _rel(filename)
                if dst.exists():
                    raise ValueError(f"duplicate file {filename!r}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(path, dst)
            elif name == "archive":
                if zipfile.is_zipfile(path):
                    with zipfile.ZipFile(path) as zf:
                        zf.extractall(in_dir)  # zipfile drops '..' and absolute parts
                else:
                    with tarfile.open(path, "r:*") as tf:
                        tf.extractall(in_dir, filter="data")
            else:
                raise ValueError(f"unexpected file field {name!r} (use files, archive or script)")
        return {k: v for k, v in req.items() if v not in (None, [], {})}

    def ocr_request(self, fields: dict[str, list[str]], files: list[tuple[str, str, Path]], in_dir: Path) -> dict:
        one = lambda k: (fields.get(k) or [None])[-1]  # noqa: E731
        images = []
        for name, filename, path in files:
            if name != "images":
                raise ValueError(f"unexpected file field {name!r} (use images)")
            rel = _rel(filename)
            if rel.suffix.lower() not in IMAGE_EXT:
                raise ValueError(f"not an image: {filename!r}")
            images.append((str(rel), path.rename(path.with_suffix(rel.suffix.lower()))))
        if not images:
            raise ValueError("no images (multipart field `images`)")
        if len({i for i, _ in images}) != len(images):
            raise ValueError("duplicate image filenames")
        write_ocr_input(images, in_dir, one("prompt"), int(one("max_tokens") or 4096))
        req = {"profile": one("profile") or "qwen35-9b-l4"}
        if one("idle_minutes"):
            req["idle_minutes"] = float(one("idle_minutes"))
        return req

    # -- jobs ----------------------------------------------------------------------
    @staticmethod
    def links(rec: dict) -> dict:
        base = f"/api/jobs/{rec['job_id']}"
        return {**rec, "links": {"self": base, "events": f"{base}/events", "output": f"{base}/output",
                                 "files": f"{base}/files", "results": f"{base}/results"}}

    async def post_job(self, request):
        return await self.staged(request, self.job_request)

    async def post_ocr(self, request):
        return await self.staged(request, self.ocr_request, json_ok=False)

    async def list_jobs(self, request):
        recs = await self.call("status", {"limit": int(request.query.get("limit", 50))})
        return web.json_response([self.links(r) for r in recs])

    async def get_job(self, request):
        jid = request.match_info["id"]
        rec = await self.call("status", {"job_id": jid})
        deadline = time.monotonic() + min(float(request.query.get("wait", 0)), 600)
        while rec["status"] not in TERMINAL and time.monotonic() < deadline:
            await asyncio.sleep(min(2, max(0.1, deadline - time.monotonic())))
            rec = await self.call("status", {"job_id": jid})
        return web.json_response(self.links(rec))

    async def cancel_job(self, request):
        return web.json_response(self.links(await self.call("cancel", {"job_id": request.match_info["id"]})))

    async def events(self, request):
        jid = request.match_info["id"]
        rec = await self.call("status", {"job_id": jid})
        resp = web.StreamResponse(headers=SSE)
        await resp.prepare(request)
        q: asyncio.Queue = asyncio.Queue()

        async def on_msg(msg):
            data = json.loads(msg.data)
            if data.get("job_id") == jid:
                q.put_nowait(("log" if msg.subject.endswith(".log") else "progress", data))

        async def send(event: str, data) -> None:
            await resp.write(f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode())

        subs = [await self.nc.subscribe(f"{self.topic}.workers.*.{s}", cb=on_msg) for s in ("log", "progress")]
        try:
            last, ping = None, time.monotonic()
            while True:
                if (rec["status"], rec.get("vm")) != last:
                    last = (rec["status"], rec.get("vm"))
                    await send("status", rec)
                if rec["status"] in TERMINAL:
                    break
                deadline = time.monotonic() + 3
                while (left := deadline - time.monotonic()) > 0:
                    try:
                        kind, data = await asyncio.wait_for(q.get(), left)
                    except asyncio.TimeoutError:
                        break
                    await send(kind, data)
                if time.monotonic() - ping > 15:
                    await resp.write(b": ping\n\n")
                    ping = time.monotonic()
                rec = await self.call("status", {"job_id": jid})
            await send("end", self.links(rec))
        except ConnectionResetError:  # the client went away
            pass
        finally:
            for s in subs:
                await s.unsubscribe()
        return resp

    async def collected(self, jid: str) -> Path:
        """The job's local output dir, fetched from Drive on first use (then removed there)."""
        jdir = DATA_DIR / "jobs" / jid
        lock = self.collecting.setdefault(jid, asyncio.Lock())
        async with lock:
            if (jdir / "job.json").exists():
                return jdir
            rec = await self.call("status", {"job_id": jid})
            if rec["status"] not in TERMINAL:
                raise web.HTTPConflict(text=json.dumps({"error": {"message": f"job {jid} is {rec['status']}",
                                                                  "code": 409}}),
                                       content_type="application/json")
            if not rec.get("out_file"):
                raise web.HTTPNotFound(text=json.dumps({"error": {"message": f"job {jid} has no output "
                                                                  f"({rec['status']}: {rec.get('note')})",
                                                                  "code": 404}}),
                                       content_type="application/json")
            d = await self.drive()
            tmp = DATA_DIR / "tmp" / f"{jid}-{uuid.uuid4().hex[:6]}"
            (tmp / "out").mkdir(parents=True)
            try:
                await asyncio.to_thread(d.get, rec["out_file"], str(tmp / "out.tar.gz"), 16,
                                        self.cfg.drive_jobs_root)

                def unpack():
                    with tarfile.open(tmp / "out.tar.gz") as tf:
                        tf.extractall(tmp / "out", filter="data")

                await asyncio.to_thread(unpack)
                (tmp / "job.json").write_text(json.dumps(rec, indent=2))
                shutil.rmtree(jdir, ignore_errors=True)
                tmp.rename(jdir)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            for name in (rec.get("in_file"), rec.get("out_file")):
                try:
                    await asyncio.to_thread(d.rm, name, self.cfg.drive_jobs_root)
                except Exception as e:
                    log(f"api: could not remove Drive:{self.cfg.drive_jobs_root}/{name}: {e}")
            log(f"api: collected {jid} ({rec['status']})")
        self.collecting.pop(jid, None)
        return jdir

    async def output(self, request):
        jdir = await self.collected(request.match_info["id"])
        return web.FileResponse(jdir / "out.tar.gz", headers={
            "Content-Disposition": f'attachment; filename="{request.match_info["id"]}-out.tar.gz"'})

    async def files(self, request):
        out = (await self.collected(request.match_info["id"])) / "out"
        rel = request.match_info.get("path", "")
        if not rel:
            return web.json_response([{"path": str(p.relative_to(out)), "size": p.stat().st_size}
                                      for p in sorted(out.rglob("*")) if p.is_file()])
        p = (out / rel).resolve()
        if not p.is_relative_to(out.resolve()) or not p.is_file():
            return _error(404, f"no file {rel!r}")
        return web.FileResponse(p)

    async def results(self, request):
        p = (await self.collected(request.match_info["id"])) / "out" / "results.jsonl"
        if not p.exists():
            return _error(404, "this job wrote no results.jsonl")
        return web.json_response([json.loads(line) for line in p.read_text().splitlines() if line.strip()])

    # -- everything else -----------------------------------------------------------
    async def health(self, request):
        return web.json_response({"ok": True, "nats": self.nc.is_connected})

    async def profiles(self, request):
        return web.json_response([{"name": p.name, "gpu": p.gpu, "venv": p.venv, "job": p.job,
                                   "model": p.env.get("MODEL"), "servable": p.servable,
                                   "idle_minutes": p.idle_minutes} for p in self.cfg.profiles.values()])

    async def workers(self, request):
        return web.json_response(await self.call("workers", {}))

    async def usage(self, request):
        proc = await asyncio.create_subprocess_exec("colab", "usage", env=_cli_env(),
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        text = (await asyncio.wait_for(proc.communicate(), 60))[0].decode()
        res = {"text": text.strip()}
        for line in text.splitlines():
            if line.startswith("Current balance:"):
                res["balance_cu"] = float(line.split()[2])
            elif line.startswith("Usage rate:"):
                res["rate_cu_per_hour"] = float(line.split()[2].split("/")[0])
        return web.json_response(res)

    async def llm_list(self, request):
        return web.json_response(await self.call("llm_list", {}))

    async def llm_start(self, request):
        model = request.match_info["model"]
        body = await request.json() if request.can_read_body else {}
        if _form_bool(request.query.get("wait"), False):
            return web.json_response(await ensure_llm(self.nc, self.topic, model,
                                                      idle_minutes=body.get("idle_minutes")))
        return web.json_response(await self.call("llm_start", {"model": model,
                                                               "idle_minutes": body.get("idle_minutes")}))

    async def llm_stop(self, request):
        return web.json_response(await self.call("llm_stop", {"model": request.match_info["model"]}))


def main(cfg: config.Config, host: str, port: int, api_key: str | None, start: bool = True,
         max_upload_mb: int = 2048) -> int:
    if not cfg.drive_project:
        raise SystemExit("no GCP quota project for the Drive API: set COLAB_LINK_DRIVE_PROJECT")
    if not api_key and host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit(f"refusing to listen on {host} without --api-key (or COLAB_LINK_API_KEY)")
    api = Api(cfg, api_key, start, max_upload_mb << 20)
    # client_max_size bounds JSON bodies (OpenAI routes); multipart uploads stream to disk.
    app = web.Application(middlewares=[api.auth], client_max_size=1 << 20)
    app.on_startup.append(api.setup)
    app.on_cleanup.append(api.cleanup)
    r = app.router
    r.add_get("/api/health", api.health)
    r.add_get("/api/profiles", api.profiles)
    r.add_post("/api/jobs", api.post_job)
    r.add_get("/api/jobs", api.list_jobs)
    r.add_get("/api/jobs/{id}", api.get_job)
    r.add_delete("/api/jobs/{id}", api.cancel_job)
    r.add_get("/api/jobs/{id}/events", api.events)
    r.add_get("/api/jobs/{id}/output", api.output)
    r.add_get("/api/jobs/{id}/files", api.files)
    r.add_get("/api/jobs/{id}/files/{path:.+}", api.files)
    r.add_get("/api/jobs/{id}/results", api.results)
    r.add_post("/api/ocr", api.post_ocr)
    r.add_get("/api/workers", api.workers)
    r.add_get("/api/usage", api.usage)
    r.add_get("/api/llm", api.llm_list)
    r.add_post("/api/llm/{model}/start", api.llm_start)
    r.add_post("/api/llm/{model}/stop", api.llm_stop)
    api.gw.add_routes(app)
    log(f"colab-link API on http://{host}:{port} (/api, OpenAI at /v1)"
        + ("" if api_key else "; no API key"))
    web.run_app(app, host=host, port=port, print=None)
    return 0

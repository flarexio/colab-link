"""Local OpenAI-compatible HTTP endpoint in front of colab-link's NATS LLM workers.

    colab-link openai [--host 127.0.0.1] [--port 8000] [--api-key KEY]

    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="...")
    client.chat.completions.create(model="qwen35-9b-l4", messages=[...], stream=True)

`model` is a servable profile name (GET /v1/models lists them). A request for a
model nobody serves asks the service to start it and waits for the cold start
(several minutes) unless --no-start; the VM leaves on its own after the
profile's idle_minutes without requests. Text only: images go through batch jobs.
"""
from __future__ import annotations

import asyncio
import json
import time

from aiohttp import web

from .client import LLMError, NoWorker, ServiceError, _connect, _request, ensure_llm, llm_request
from .nats_runner import log

ROUTES = {"/v1/chat/completions": "chat.completions", "/v1/completions": "completions"}


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": {"message": message, "type": "colab_link", "code": status}}, status=status)


class Gateway:
    def __init__(self, api_key: str | None, start: bool, start_timeout: float):
        self.api_key, self.start, self.start_timeout = api_key, start, start_timeout
        self.nc = self.topic = None
        self.starting: dict[str, asyncio.Task] = {}

    async def setup(self, app) -> None:
        self.nc, self.topic = await _connect()

    async def cleanup(self, app) -> None:
        await self.nc.drain()

    @web.middleware
    async def auth(self, request, handler):
        if self.api_key and request.headers.get("Authorization") != f"Bearer {self.api_key}":
            return _error(401, "invalid API key")
        return await handler(request)

    async def models(self, request) -> web.Response:
        try:
            views = await _request(self.nc, f"{self.topic}.llm_list", {})
        except ServiceError as e:
            return _error(503, str(e))
        return web.json_response({"object": "list", "data": [
            {"id": v["model"], "object": "model", "created": 0, "owned_by": "colab-link",
             "gpu": v["gpu"], "hf_model": v["hf_model"],
             "running": any(w["state"] == "serving" for w in v["workers"])} for v in views]})

    async def ensure(self, model: str) -> None:
        # One start per model even when many requests arrive during a cold start.
        task = self.starting.get(model)
        if task is None or task.done():
            log(f"{model} is not running; starting it on Colab")
            task = self.starting[model] = asyncio.create_task(
                ensure_llm(self.nc, self.topic, model, self.start_timeout))
        await asyncio.shield(task)

    async def complete(self, request) -> web.StreamResponse:
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "body is not JSON")
        model = body.get("model")
        if not model:
            return _error(400, "model is required")
        route = ROUTES[request.path]
        for attempt in range(2):
            stream = None
            try:
                async for obj in llm_request(self.nc, self.topic, model, route, body):
                    if not body.get("stream"):
                        return web.json_response(obj)
                    if stream is None:
                        stream = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                                             "Cache-Control": "no-cache"})
                        await stream.prepare(request)
                    await stream.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                if stream is None:  # a stream that ended before its first chunk
                    stream = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                    await stream.prepare(request)
                await stream.write(b"data: [DONE]\n\n")
                await stream.write_eof()
                return stream
            except NoWorker:
                if not self.start or attempt:
                    return _error(503, f"no worker serves {model}")
                try:
                    await self.ensure(model)
                except Exception as e:
                    return _error(503, f"could not start {model}: {e}")
            except LLMError as e:
                if stream is not None:  # headers already sent: report in-band
                    await stream.write(b"data: " + e.body + b"\n\n")
                    await stream.write_eof()
                    return stream
                try:
                    return web.json_response(json.loads(e.body), status=e.status)
                except ValueError:
                    return _error(e.status, e.body.decode(errors="replace"))
            except asyncio.TimeoutError:
                return _error(504, f"{model} did not answer in time")
        return _error(503, f"no worker serves {model}")

    def add_routes(self, app) -> None:
        app.router.add_get("/v1/models", self.models)
        for path in ROUTES:
            app.router.add_post(path, self.complete)


def main(host: str, port: int, api_key: str | None, start: bool = True, start_timeout: float = 1200) -> int:
    gw = Gateway(api_key, start, start_timeout)
    app = web.Application(middlewares=[gw.auth], client_max_size=1 << 20)
    app.on_startup.append(gw.setup)
    app.on_cleanup.append(gw.cleanup)
    gw.add_routes(app)
    log(f"OpenAI-compatible endpoint on http://{host}:{port}/v1 (started {time.strftime('%H:%M:%S')})")
    web.run_app(app, host=host, port=port, print=None)
    return 0

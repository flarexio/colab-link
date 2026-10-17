"""colab-link: run jobs on rented Colab GPUs with cached environments.

    colab-link profiles
    colab-link run qwen35-9b-l4 --input DIR [--out DIR] [--env KEY=VALUE ...]
    colab-link run --gpu T4 --job my.py --input DIR --pip pandas      (no profile)
    colab-link cache ls | cache rm NAME ...
    colab-link sessions | stop-all | usage

Service mode (warm workers fed from a JetStream queue):
    colab-link serve
    colab-link submit qwen35-9b-l4 --input DIR [--wait] [--out DIR]
    colab-link jobs [JOB_ID] | fetch JOB_ID | cancel JOB_ID | workers
    colab-link ocr IMAGES_OR_DIRS... [--profile P] [--out DIR]     (batch OCR job)

LLMs served on Colab, OpenAI request bodies over NATS (needs `serve`):
    colab-link llm ls | llm start MODEL [--wait] | llm stop MODEL
    colab-link chat MODEL "prompt"        (starts the model if needed)
    colab-link openai [--port 8000]       (local OpenAI-compatible HTTP endpoint)

Idle-VM guard (see guard.py):
    colab-link reap [--dry-run] [--loop SECONDS]      (run from cron)
    colab-link lease ENDPOINT|--all [--minutes N] | lease --list | lease --drop ENDPOINT
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from . import config
from .runner import run_job
from .session import ColabSession, _cli_env, drive_token

# Measured CU/hour (CLAUDE.md, 2026-10-14). Used only for the estimate shown by `profiles`.
CU_PER_HOUR = {None: 0.08, "T4": 1.07, "L4": 1.54, "A100": 5.30, "G4": 8.90}
STATE_DIR = Path.home() / ".cache" / "colab-link" / "state"


def _drive(cfg: config.Config):
    if not cfg.drive_project:
        sys.exit("no GCP quota project for the Drive API: set COLAB_LINK_DRIVE_PROJECT "
                 "(or [drive] project in profiles.toml); see README")
    os.environ["DRIVE_TOKEN"] = drive_token()
    os.environ["DRIVE_QUOTA_PROJECT"] = cfg.drive_project
    os.environ["DRIVE_ROOT"] = cfg.drive_root
    from . import drive_vm
    return drive_vm


def cmd_profiles(cfg, a):
    for p in cfg.profiles.values():
        print(f"{p.name:<18} gpu={p.gpu or 'CPU':<5} {CU_PER_HOUR.get(p.gpu, '?')} CU/h  "
              f"venv={p.venv or '-':<6} model={p.env.get('MODEL', '-')}")


def cmd_run(cfg, a):
    prof = cfg.profiles.get(a.profile) if a.profile else None
    if a.profile and not prof:
        sys.exit(f"unknown profile {a.profile!r}; see `colab-link profiles`")
    job = Path(a.job or (prof and prof.job) or "")
    if not job.is_file():
        job = config.ROOT / job
    if not job.is_file():
        sys.exit("no job script: pass --job or set `job` in the profile")
    gpu = a.gpu or (prof.gpu if prof else None)
    env = {**(prof.env if prof else {}), **dict(kv.split("=", 1) for kv in a.env)}
    out = a.out or config.ROOT / "results" / f"{a.profile or job.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    drive_project = None if a.no_cache else cfg.drive_project

    setup, post, python = [], [], "python"
    if prof and prof.venv:
        setup.append("python3 vm_env.py prepare --venv {} --packages {} --jit {}".format(
            shlex.quote(prof.venv), shlex.quote(" ".join(prof.packages)),
            shlex.quote(prof.jit_id({**env, "_gpu": gpu or "CPU"}))))
        post.append("python3 vm_env.py finalize")
        python = "/content/venv/bin/python"

    plan = {"profile": a.profile, "gpu": gpu or "CPU", "job": str(job), "input": str(a.input),
            "out": str(out), "env": env, "pip": a.pip, "drive_cache": bool(drive_project),
            "setup": setup, "post": post, "python": python}
    plan["transport"] = a.transport
    if a.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    if a.transport == "nats":
        import asyncio
        from .nats_runner import run_job_nats
        venv = ({"name": prof.venv, "packages": prof.packages,
                 "jit": prof.jit_id({**env, "_gpu": gpu or "CPU"})} if prof and prof.venv else None)
        return asyncio.run(run_job_nats(
            job, a.input, Path(out), gpu, env, venv, a.pip, cfg.drive_project,
            cache_root=cfg.drive_root, jobs_root=cfg.drive_jobs_root, max_hours=a.max_hours,
            keep_files=a.keep_files, cache=not a.no_cache))
    return run_job(job, a.input, Path(out), gpu, a.pip, env, a.poll, a.max_hours,
                   setup=setup, python=python, drive_project=drive_project, post=post,
                   drive_root=cfg.drive_root)


def cmd_serve(cfg, a):
    from .service import main as serve
    return serve(cfg)


def cmd_submit(cfg, a):
    import asyncio
    from .client import submit
    if a.profile and a.profile not in cfg.profiles:
        sys.exit(f"unknown profile {a.profile!r}; see `colab-link profiles`")
    if not a.profile and not a.job:
        sys.exit("pass a profile or --job")
    req = {"profile": a.profile, "gpu": a.gpu, "env": dict(kv.split("=", 1) for kv in a.env),
           "pip": a.pip, "cache": not a.no_cache, "idle_minutes": a.idle_minutes}
    if a.job:
        req["script"] = Path(a.job).read_text()
    return asyncio.run(submit(cfg, req, a.input, a.out, a.wait))


def cmd_jobs(cfg, a):
    import asyncio
    from .client import call
    res = asyncio.run(call("status", {"job_id": a.job_id} if a.job_id else {}))
    if a.job_id:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    for r in res:
        t = time.strftime("%m-%d %H:%M", time.localtime(r.get("submitted_at", 0)))
        print(f"{r['job_id']}  {t}  {r['status']:<9} {r['pool']:<26} {r.get('vm') or '-':<12} "
              f"exit={r.get('exitcode', '-')}")


def cmd_fetch(cfg, a):
    import asyncio
    from .client import fetch
    return asyncio.run(fetch(cfg, a.job_id, a.out))


def cmd_cancel(cfg, a):
    import asyncio
    from .client import call
    print(json.dumps(asyncio.run(call("cancel", {"job_id": a.job_id})), indent=2))


def cmd_workers(cfg, a):
    import asyncio
    from .client import call
    ws = asyncio.run(call("workers", {}))
    for w in ws:
        print(f"{w['vm']:<12} {w['gpu'] or 'CPU':<5} {w['state']:<9} {w['pool']:<26} "
              f"job={w['job_id'] or '-'} up={w['up_s']}s")
    print(f"{len(ws)} worker(s)")


IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
OCR_PROMPT = ("Transcribe all text in this image exactly as it appears, in reading order, "
              "keeping line breaks. Output only the transcription.")


def cmd_ocr(cfg, a):
    """Batch OCR: images -> tasks.jsonl for jobs/vlm_batch.py -> one .txt per image."""
    import asyncio
    import shutil
    import tempfile
    from .client import submit
    if a.profile not in cfg.profiles:
        sys.exit(f"unknown profile {a.profile!r}; see `colab-link profiles`")
    images: list[tuple[str, Path]] = []
    for p in a.inputs:
        if p.is_dir():
            images += [(str(f.relative_to(p)), f) for f in sorted(p.rglob("*")) if f.suffix.lower() in IMAGE_EXT]
        elif p.suffix.lower() in IMAGE_EXT:
            images.append((p.name, p))
        else:
            sys.exit(f"not an image or directory: {p}")
    if not images:
        sys.exit("no images found")
    prompt = Path(a.prompt_file).read_text() if a.prompt_file else a.prompt
    out = a.out or config.ROOT / "results" / f"ocr-{time.strftime('%Y%m%d-%H%M%S')}"
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "imgs").mkdir()
        with open(Path(tmp) / "tasks.jsonl", "w") as f:
            for i, (ident, src) in enumerate(images):
                dst = f"imgs/{i:05d}{src.suffix.lower()}"
                shutil.copyfile(src, Path(tmp) / dst)
                f.write(json.dumps({"id": ident, "prompt": prompt, "image": dst,
                                    "max_tokens": a.max_tokens}, ensure_ascii=False) + "\n")
        print(f"{len(images)} image(s) -> {a.profile}", file=sys.stderr)
        req = {"profile": a.profile, "idle_minutes": a.idle_minutes}
        code = asyncio.run(submit(cfg, req, Path(tmp), out, not a.no_wait))
    if a.no_wait or not (out / "results.jsonl").exists():
        return code
    text_dir = out / "text"
    for line in (out / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        dst = text_dir / (r["id"] + ".txt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(r["output"])
    print(f"text in {text_dir}", file=sys.stderr)
    return code


def cmd_llm(cfg, a):
    import asyncio
    from .client import _connect, _request, ensure_llm

    async def go():
        nc, topic = await _connect()
        try:
            if a.action == "ls":
                return await _request(nc, f"{topic}.llm_list", {})
            if not a.model:
                sys.exit(f"llm {a.action} needs a MODEL")
            if a.action == "stop":
                return [await _request(nc, f"{topic}.llm_stop", {"model": a.model})]
            if a.wait:
                return [await ensure_llm(nc, topic, a.model, idle_minutes=a.idle_minutes)]
            return [await _request(nc, f"{topic}.llm_start", {"model": a.model, "idle_minutes": a.idle_minutes})]
        finally:
            await nc.drain()

    for v in asyncio.run(go()):
        ws = ", ".join(f"{w['vm']} {w['state']} inflight={w['inflight']} requests={w['requests']} up={w['up_s']}s"
                       for w in v["workers"]) or ("starting" if v["wanted"] else "-")
        print(f"{v['model']:<18} {v['gpu'] or 'CPU':<4} {v['hf_model']:<28} {ws}")


def cmd_chat(cfg, a):
    import asyncio
    from .client import chat
    prompt = a.prompt if a.prompt is not None else sys.stdin.read()
    messages = ([{"role": "system", "content": a.system}] if a.system else []) + \
        [{"role": "user", "content": prompt}]
    params = {k: v for k, v in {"max_tokens": a.max_tokens, "temperature": a.temperature}.items() if v is not None}
    return asyncio.run(chat(a.model, messages, start=not a.no_start, **params))


def cmd_openai(cfg, a):
    from .openai_gateway import main as gateway
    return gateway(a.host, a.port, a.api_key or os.environ.get("COLAB_LINK_API_KEY"), start=not a.no_start)


def cmd_reap(cfg, a):
    from .guard import main as reap
    return reap(a.dry_run, a.loop)


def cmd_lease(cfg, a):
    from . import colab_api, guard
    if a.list or (not a.endpoint and not a.all and not a.drop):
        held = guard.leases()
        for ep, rec in held.items():
            print(f"{ep}  until {time.strftime('%m-%d %H:%M', time.localtime(rec['until']))}  {rec.get('reason', '')}")
        print(f"{len(held)} lease(s)")
        return
    if a.drop:
        guard.drop_lease(a.drop)
        return
    eps = [x["endpoint"] for x in colab_api.assignments(drive_token())] if a.all else [a.endpoint]
    for ep in eps:
        rec = guard.lease(ep, a.minutes * 60, a.reason)
        print(f"leased {ep} until {time.strftime('%H:%M', time.localtime(rec['until']))}")


def cmd_cache(cfg, a):
    d = _drive(cfg)
    if a.action == "ls":
        files = sorted(d.ls(), key=lambda f: f["name"])
        for f in files:
            print(f"{int(f.get('size', 0)) / 1e9:7.2f} GB  {f['modifiedTime'][:16]}  {f['name']}")
        print(f"{sum(int(f.get('size', 0)) for f in files) / 1e9:7.2f} GB total in Drive:/{cfg.drive_root}")
    else:
        for name in a.names:
            print(("removed " if d.rm(name) else "not found: ") + name)


def cmd_sessions(cfg, a):
    return subprocess.run(["colab", "sessions"], env=_cli_env()).returncode


def cmd_usage(cfg, a):
    return subprocess.run(["colab", "usage"], env=_cli_env()).returncode


def cmd_stop_all(cfg, a):
    """Stop every VM this tool started that still has live local state."""
    n = 0
    for f in sorted(STATE_DIR.glob("*.json")):
        try:
            live = json.loads(f.read_text() or "{}") != {}
        except ValueError:
            live = False
        s = ColabSession(f.stem, state_dir=STATE_DIR)
        if live:
            print(f"stopping {f.stem}")
            s.started = True
            s.stop()
            n += 1
        s.forget()
    print(f"stopped {n} session(s)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="colab-link", description=__doc__.split("\n")[0])
    ap.add_argument("--profiles", help="profiles.toml path (default: repo root)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("profiles", help="list profiles")

    r = sub.add_parser("run", help="run a job on a fresh VM, collect outputs, release the VM")
    r.add_argument("profile", nargs="?", help="profile name from profiles.toml")
    r.add_argument("--input", type=Path, required=True, help="directory shipped to the VM as ./in")
    r.add_argument("--out", type=Path, help="local output dir (default results/<profile>-<time>)")
    r.add_argument("--job", help="job script (overrides the profile's)")
    r.add_argument("--gpu", help="T4 | L4 | A100 | G4 (overrides the profile's)")
    r.add_argument("--env", action="append", default=[], help="KEY=VALUE for the job (repeatable)")
    r.add_argument("--pip", action="append", default=[], help="extra package to install (repeatable)")
    r.add_argument("--no-cache", action="store_true", help="skip Drive caches (build venv every time)")
    r.add_argument("--poll", type=float, default=20, help="poll interval seconds (each poll ~5 s)")
    r.add_argument("--max-hours", type=float, default=6)
    r.add_argument("--dry-run", action="store_true", help="print the plan, do not rent a VM")
    r.add_argument("--transport", choices=["nats", "exec"], default="nats",
                   help="nats: worker agent over NATS + files via Drive (default); "
                        "exec: legacy colab-exec polling + colab upload/download")
    r.add_argument("--keep-files", action="store_true",
                   help="nats: keep the job's input/output files on Drive")

    sub.add_parser("serve", help="run the service: NATS endpoints, job queue, warm workers")
    sm = sub.add_parser("submit", help="queue a job on `colab-link serve` (warm workers)")
    sm.add_argument("profile", nargs="?", help="profile name from profiles.toml")
    sm.add_argument("--input", type=Path, required=True, help="directory shipped to the VM as ./in")
    sm.add_argument("--out", type=Path, help="with --wait: output dir (default results/<profile>-<job>)")
    sm.add_argument("--job", help="job script (overrides the profile's)")
    sm.add_argument("--gpu", help="T4 | L4 | A100 | G4 (overrides the profile's)")
    sm.add_argument("--env", action="append", default=[], help="KEY=VALUE for the job (repeatable)")
    sm.add_argument("--pip", action="append", default=[], help="extra package to install (repeatable)")
    sm.add_argument("--no-cache", action="store_true", help="skip Drive caches")
    sm.add_argument("--idle-minutes", type=float, help="warm worker idle timeout (default: profile, 10)")
    sm.add_argument("--wait", action="store_true", help="follow the job and download its output")
    j = sub.add_parser("jobs", help="recent jobs on the service, or one job's record")
    j.add_argument("job_id", nargs="?")
    fe = sub.add_parser("fetch", help="download a finished job's output (and remove its Drive files)")
    fe.add_argument("job_id")
    fe.add_argument("--out", type=Path, help="output dir (default results/<profile>-<job>)")
    cn = sub.add_parser("cancel", help="cancel a queued or running job")
    cn.add_argument("job_id")
    sub.add_parser("workers", help="warm workers of the service")
    oc = sub.add_parser("ocr", help="batch OCR: images or directories of images -> text, on `serve`")
    oc.add_argument("inputs", nargs="+", type=Path)
    oc.add_argument("--profile", default="qwen35-9b-l4")
    oc.add_argument("--prompt", default=OCR_PROMPT)
    oc.add_argument("--prompt-file", help="read the prompt from a file")
    oc.add_argument("--max-tokens", type=int, default=4096)
    oc.add_argument("--out", type=Path, help="output dir (default results/ocr-<time>)")
    oc.add_argument("--idle-minutes", type=float)
    oc.add_argument("--no-wait", action="store_true", help="queue only; collect later with `fetch`")

    ll = sub.add_parser("llm", help="LLMs served on Colab: ls | start MODEL | stop MODEL")
    ll.add_argument("action", choices=["ls", "start", "stop"])
    ll.add_argument("model", nargs="?", help="a servable profile name")
    ll.add_argument("--idle-minutes", type=float, help="stop after this long without requests")
    ll.add_argument("--wait", action="store_true", help="start: wait until it serves")
    ch = sub.add_parser("chat", help="one chat completion from a Colab LLM, streamed")
    ch.add_argument("model")
    ch.add_argument("prompt", nargs="?", help="default: read stdin")
    ch.add_argument("--system")
    ch.add_argument("--max-tokens", type=int)
    ch.add_argument("--temperature", type=float)
    ch.add_argument("--no-start", action="store_true", help="fail instead of starting the model")
    oa = sub.add_parser("openai", help="local OpenAI-compatible HTTP endpoint for the Colab LLMs")
    oa.add_argument("--host", default="127.0.0.1")
    oa.add_argument("--port", type=int, default=8000)
    oa.add_argument("--api-key", help="require this bearer key (or env COLAB_LINK_API_KEY)")
    oa.add_argument("--no-start", action="store_true", help="answer 503 instead of starting a model")

    rp = sub.add_parser("reap", help="stop Colab sessions nobody supervises or leases (run from cron)")
    rp.add_argument("--dry-run", action="store_true")
    rp.add_argument("--loop", type=float, help="repeat every N seconds instead of one pass")
    ls_ = sub.add_parser("lease", help="keep a session safe from the reaper (manual or browser use)")
    ls_.add_argument("endpoint", nargs="?", help="session endpoint (see `colab-link sessions`)")
    ls_.add_argument("--all", action="store_true", help="lease every live session")
    ls_.add_argument("--minutes", type=float, default=120)
    ls_.add_argument("--reason", default="manual")
    ls_.add_argument("--list", action="store_true")
    ls_.add_argument("--drop", metavar="ENDPOINT")

    c = sub.add_parser("cache", help="Drive cache files")
    csub = c.add_subparsers(dest="action", required=True)
    csub.add_parser("ls")
    rm = csub.add_parser("rm")
    rm.add_argument("names", nargs="+")

    sub.add_parser("sessions", help="list live Colab sessions (server side)")
    sub.add_parser("stop-all", help="stop every VM started by colab-link")
    sub.add_parser("usage", help="compute-unit balance")

    a = ap.parse_args(argv)
    cfg = config.load(a.profiles)
    handler = {"profiles": cmd_profiles, "run": cmd_run, "cache": cmd_cache,
               "sessions": cmd_sessions, "stop-all": cmd_stop_all, "usage": cmd_usage,
               "serve": cmd_serve, "submit": cmd_submit, "jobs": cmd_jobs, "cancel": cmd_cancel, "fetch": cmd_fetch,
               "workers": cmd_workers, "ocr": cmd_ocr, "llm": cmd_llm, "chat": cmd_chat,
               "openai": cmd_openai, "reap": cmd_reap, "lease": cmd_lease}[a.cmd]
    from .client import ServiceError
    try:
        sys.exit(handler(cfg, a) or 0)
    except ServiceError as e:
        sys.exit(str(e))

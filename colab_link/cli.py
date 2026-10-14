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
               "workers": cmd_workers}[a.cmd]
    sys.exit(handler(cfg, a) or 0)

"""Keep idle Colab sessions from burning compute units.

    colab-link reap [--dry-run] [--loop SECONDS]     (cron: every 5 min)
    colab-link lease ENDPOINT|--all [--minutes N] [--reason TEXT] | lease --list | lease --drop ENDPOINT

Three independent lines of defence, so that no single crash, lost connection or
an assistant hitting its usage limit leaves a VM billing:

1. The worker on each VM releases its own VM when its supervisor (`serve` or the
   `run` client) has said nothing for 10 minutes, or its access token is about
   to expire (worker.py self-stop). Works even if this machine is down.
2. Supervisors stop VMs when a worker goes idle or quiet (service.py, nats_runner.py).
3. This reaper, run from cron, looks at EVERY session on the account (also ones
   opened in a browser or with a bare `colab new`) and releases those that have
   gone without supervision or a lease for GRACE_S:
   - kept: a colab-link worker heartbeat that its supervisor pinged recently,
     unless it has sat idle IDLE_MARGIN_S past its own idle limit, or the
     session is older than MAX_HOURS;
   - kept: a lease (`colab-link lease`) that has not expired; take one before
     using Colab by hand (browser notebook, manual tests);
   - otherwise stopped GRACE_S after it was last seen supervised or leased.
   It needs nothing from the service: the Colab API (through the CLI's login)
   and a few seconds of listening to worker heartbeats on NATS.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from . import colab_api
from .nats_runner import FLAREX, NATS_URL, log
from .session import drive_token

GUARD_DIR = Path(os.environ.get("COLAB_LINK_GUARD", Path.home() / ".cache" / "colab-link" / "guard"))
GRACE_S = 15 * 60          # unsupervised and unleased this long -> stopped (covers VM start-up)
SUPERVISED_S = 180         # supervisors ping every 30 s
IDLE_MARGIN_S = 10 * 60    # past the worker's own idle limit
MAX_HOURS = 12             # any session without a lease
LISTEN_S = 12              # workers heartbeat every 5 s


# -- leases ----------------------------------------------------------------------
def _leases_dir() -> Path:
    d = GUARD_DIR / "leases"
    d.mkdir(parents=True, exist_ok=True)
    return d


def lease(endpoint: str, seconds: float, reason: str) -> dict:
    rec = {"endpoint": endpoint, "until": time.time() + seconds, "reason": reason}
    (_leases_dir() / f"{endpoint}.json").write_text(json.dumps(rec))
    return rec


def drop_lease(endpoint: str) -> None:
    (_leases_dir() / f"{endpoint}.json").unlink(missing_ok=True)


def leases() -> dict[str, dict]:
    out = {}
    for f in _leases_dir().glob("*.json"):
        try:
            rec = json.loads(f.read_text())
        except ValueError:
            rec = {"until": 0}
        if rec.get("until", 0) < time.time():
            f.unlink(missing_ok=True)
        else:
            out[f.stem] = rec
    return out


# -- reaper ----------------------------------------------------------------------
async def heartbeats(seconds: float = LISTEN_S) -> dict[str, dict] | None:
    """Latest worker heartbeat per Colab endpoint, or None if NATS is unreachable."""
    import nats

    try:
        edge = (FLAREX / "id").read_text().strip()
        nc = await nats.connect(NATS_URL, user_credentials=str(FLAREX / "user.creds"),
                                name=f"Colab Link Reaper - {edge}", connect_timeout=10,
                                max_reconnect_attempts=1)
    except Exception as e:
        log(f"reaper: NATS unreachable ({e}); deciding without heartbeats")
        return None
    seen: dict[str, dict] = {}

    async def on_beat(msg):
        data = json.loads(msg.data)
        if data.get("endpoint"):
            seen[data["endpoint"]] = data

    await nc.subscribe(f"edges.{edge}.colab-link.workers.*.heartbeat", cb=on_beat)
    await asyncio.sleep(seconds)
    await nc.drain()
    return seen


def verdict(a: dict, hb: dict | None, lease_rec: dict | None, st: dict, t: float) -> tuple[str, str]:
    """('keep' | 'stop' | 'grace', why) for one session."""
    age_h = (t - st["first"]) / 3600
    if lease_rec:
        return "keep", f"lease until {time.strftime('%H:%M', time.localtime(lease_rec['until']))} ({lease_rec.get('reason', '')})"
    if hb and hb.get("supervised_s", 1e9) < SUPERVISED_S and t - hb.get("t", 0) < 60:
        limit, idle = hb.get("idle_limit_s"), hb.get("idle_s")
        if limit and idle is not None and idle > limit + IDLE_MARGIN_S:
            return "stop", f"worker {hb.get('vm_id')} idle {idle / 60:.0f} min (limit {limit / 60:.0f})"
        if age_h > MAX_HOURS:
            return "stop", f"running {age_h:.1f} h (max {MAX_HOURS} h without a lease)"
        what = "busy" if hb.get("busy") else (hb.get("llm") or {}).get("state") or "idle"
        return "keep", f"worker {hb.get('vm_id')} supervised ({what})"
    quiet = t - st["ok"]
    why = (f"worker {hb.get('vm_id')} unsupervised" if hb else "no colab-link worker") + f" for {quiet / 60:.0f} min"
    return ("stop" if quiet > GRACE_S else "grace"), why


def reap(dry_run: bool = False) -> int:
    """One pass over every session on the account; returns how many were stopped."""
    token = drive_token()
    sessions = colab_api.assignments(token)
    state_file = GUARD_DIR / "reaper.json"
    GUARD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        state = json.loads(state_file.read_text())
    except (OSError, ValueError):
        state = {}
    hbs = (asyncio.run(heartbeats()) or {}) if sessions else {}
    held = leases()
    t = time.time()
    stopped = 0
    for a in sessions:
        ep = a["endpoint"]
        st = state.setdefault(ep, {"first": t, "ok": t})
        action, why = verdict(a, hbs.get(ep), held.get(ep), st, t)
        if action == "keep":
            st["ok"] = t
        label = f"{ep} {a.get('accelerator', '?')}"
        if action == "stop":
            if dry_run:
                log(f"reaper: would stop {label}: {why}")
                continue
            try:
                colab_api.unassign(token, ep)
                stopped += 1
                log(f"reaper: stopped {label}: {why}")
            except Exception as e:
                log(f"reaper: could not stop {label}: {e}")
        else:
            log(f"reaper: {action} {label}: {why}")
    live = {a["endpoint"] for a in sessions}
    state = {ep: st for ep, st in state.items() if ep in live}
    if not dry_run:
        state_file.write_text(json.dumps(state, indent=1))
    if not sessions:
        log("reaper: no sessions")
    return stopped


def main(dry_run: bool = False, loop: float | None = None) -> int:
    while True:
        try:
            reap(dry_run)
        except Exception as e:  # keep looping; one failed pass must not end the guard
            log(f"reaper: pass failed: {e}")
            if not loop:
                return 1
        if not loop:
            return 0
        time.sleep(loop)

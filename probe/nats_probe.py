"""NATS reachability probe between this host and a Colab VM.

Follows the FlareX edge convention: ~/.flarex/colab-link/{id,user.creds},
subjects under edges.<edge_id>.colab-link.

    # local check only
    env -u PYTHONPATH uv run -q --no-project --with "nats-py[nkeys]" --with aiohttp probe/nats_probe.py local
    # full: rent a CPU VM, run the responder there, measure request/reply + VM push
    env -u PYTHONPATH uv run -q --no-project --with "nats-py[nkeys]" --with aiohttp probe/nats_probe.py vm

On the VM the same file runs as `responder` (started by the vm mode).
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

URL = os.environ.get("NATS_URL", "wss://nats.flarex.io")
FLAREX = Path(os.environ.get("FLAREX_DIR", Path.home() / ".flarex" / "colab-link"))


async def echo(msg) -> None:
    await msg.respond(msg.data)


def edge_id() -> str:
    return (FLAREX / "id").read_text().strip()


async def connect(creds: str, name: str):
    import nats
    return await nats.connect(URL, user_credentials=creds, name=name,
                              connect_timeout=15, max_reconnect_attempts=3)


async def local_check() -> dict:
    eid = edge_id()
    t = time.time()
    nc = await connect(str(FLAREX / "user.creds"), f"Colab Link - {eid}")
    connect_s = time.time() - t
    subj = f"edges.{eid}.colab-link.probe.echo"
    sub = await nc.subscribe(subj, cb=echo)
    rtts = []
    for i in range(10):
        t = time.time()
        await nc.request(subj, f"ping {i}".encode(), timeout=5)
        rtts.append((time.time() - t) * 1000)
    await sub.unsubscribe()
    await nc.drain()
    return {"connect_s": round(connect_s, 2), "loopback_rtt_ms_median": round(statistics.median(rtts), 1)}


async def responder(seconds: float) -> None:
    """VM side: echo requests, push a heartbeat every 2 s, exit after `seconds`."""
    eid = edge_id()
    t = time.time()
    nc = await connect(os.environ["NATS_CREDS"], f"Colab Link VM - {eid}")
    base = f"edges.{eid}.colab-link.probe.vm"
    await nc.subscribe(f"{base}.echo", cb=echo)
    await nc.publish(f"{base}.ready", json.dumps({"connect_s": round(time.time() - t, 2)}).encode())
    print(f"[probe] connected in {time.time() - t:.2f}s", flush=True)
    end, n = time.time() + seconds, 0
    while time.time() < end:
        n += 1
        await nc.publish(f"{base}.heartbeat", json.dumps({"n": n, "t": time.time()}).encode())
        await asyncio.sleep(2)
    await nc.drain()


async def vm_probe() -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from colab_link.session import ColabSession

    eid = edge_id()
    base = f"edges.{eid}.colab-link.probe.vm"
    nc = await connect(str(FLAREX / "user.creds"), f"Colab Link - {eid}")
    ready, beats = asyncio.Event(), []
    report: dict = {}

    async def on_ready(m):
        report["vm_connect_s"] = json.loads(m.data)["connect_s"]
        ready.set()

    async def on_beat(m):
        beats.append(time.time() - json.loads(m.data)["t"])

    await nc.subscribe(f"{base}.ready", cb=on_ready)
    await nc.subscribe(f"{base}.heartbeat", cb=on_beat)

    sess = ColabSession(f"nats-probe-{int(time.time()) % 100000}")
    try:
        await asyncio.to_thread(sess.start)
        await asyncio.to_thread(sess.wait_ready)
        await asyncio.to_thread(sess.exec, "import os; os.makedirs('/content/probe', exist_ok=True)")
        # Creds go up as a file, never inside exec code (the CLI logs executed code).
        await asyncio.to_thread(sess.upload, FLAREX / "user.creds", "/content/probe/.nats.creds")
        await asyncio.to_thread(sess.upload, FLAREX / "id", "/content/probe/id")
        await asyncio.to_thread(sess.upload, Path(__file__), "/content/probe/nats_probe.py")
        t = time.time()
        await asyncio.to_thread(sess.exec, (
            "import subprocess, os\n"
            "os.chmod('/content/probe/.nats.creds', 0o600)\n"
            "subprocess.run('uv pip install --system -q nats-py[nkeys] aiohttp', shell=True, check=True)\n"
            "subprocess.Popen(['python3', '/content/probe/nats_probe.py', 'responder', '120'],\n"
            "  env=dict(os.environ, NATS_CREDS='/content/probe/.nats.creds', FLAREX_DIR='/content/probe'),\n"
            "  stdout=open('/content/probe/log.txt', 'w'), stderr=subprocess.STDOUT, start_new_session=True)\n"
            "print('STARTED')"), 180)
        try:
            await asyncio.wait_for(ready.wait(), 90)
        except asyncio.TimeoutError:
            report["error"] = "VM never connected"
            report["vm_log"] = await asyncio.to_thread(
                sess.exec, "print(open('/content/probe/log.txt').read()[-1500:])")
            return report
        report["launch_to_ready_s"] = round(time.time() - t, 1)
        rtts = []
        for i in range(30):
            t = time.time()
            await nc.request(f"{base}.echo", f"ping {i}".encode(), timeout=5)
            rtts.append((time.time() - t) * 1000)
            await asyncio.sleep(0.2)
        await asyncio.sleep(10)  # collect heartbeats
        q = sorted(rtts)
        report.update({"rtt_ms_median": round(statistics.median(q), 1), "rtt_ms_p95": round(q[28], 1),
                       "rtt_ms_max": round(q[-1], 1), "heartbeats": len(beats),
                       "push_latency_ms_median": round(statistics.median(beats) * 1000, 1) if beats else None})
    finally:
        await asyncio.to_thread(sess.stop)
        sess.forget()
        await nc.drain()
    return report


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "responder":
        asyncio.run(responder(float(sys.argv[2])))
    else:
        print(json.dumps(asyncio.run(local_check() if mode == "local" else vm_probe()), indent=2))

"""Connection-stability probe for Colab sessions.

Two sessions run side by side for --minutes:
  * "pinged": an exec every --interval s, recording latency and failures.
  * "idle":   a detached heartbeat process writes a timestamp every 10 s; we
              never touch the kernel until the end, to see whether a VM whose
              kernel is idle (but whose background process is busy) is reclaimed.

    python3 -m probe.stability --gpu "" --minutes 60 --interval 30
Writes results/stability-<ts>.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from colab_link.session import ColabSession  # noqa: E402

HEARTBEAT = r"""
import os, subprocess
if not os.path.exists("/content/hb.started"):
    open("/content/hb.started", "w").close()
    subprocess.Popen(["bash", "-c", "while true; do date +%s > /content/hb; sleep 10; done"],
                     start_new_session=True)
print("HB_OK")
"""


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--interval", type=float, default=30)
    a = ap.parse_args()
    gpu = a.gpu or None
    ts = time.strftime("%Y%m%d-%H%M%S")
    pinged = ColabSession(f"stab-ping-{ts[-6:]}", gpu=gpu)
    idle = ColabSession(f"stab-idle-{ts[-6:]}", gpu=gpu)
    samples, failures = [], []
    report = {"gpu": gpu or "CPU", "minutes": a.minutes, "interval_s": a.interval}
    try:
        for s in (pinged, idle):
            s.start()
            s.wait_ready()
        idle.exec(HEARTBEAT)
        log("both sessions ready; idle session heartbeat started")

        end = time.time() + a.minutes * 60
        n = 0
        while time.time() < end:
            n += 1
            t = time.time()
            proc = pinged.cli("exec", "-s", pinged.name, "--timeout", "30",
                              stdin="print('PONG')", timeout=90, check=False)
            dt = time.time() - t
            if "PONG" in proc.stdout:
                samples.append(dt)
            else:
                err = (proc.stderr.strip().splitlines() or ["?"])[-1][:200]
                failures.append({"t": time.strftime("%H:%M:%S"), "latency_s": round(dt, 1), "err": err})
                log(f"ping #{n} FAILED after {dt:.1f}s: {err}")
            if n % 10 == 0:
                log(f"ping #{n}: ok={len(samples)} fail={len(failures)} "
                    f"median={statistics.median(samples):.2f}s" if samples else f"ping #{n}: no successes")
            time.sleep(max(0, a.interval - (time.time() - t)))

        # Now look at the idle session for the first time since launch.
        raw = idle.cli("exec", "-s", idle.name, "--timeout", "30",
                       stdin="import time; print('HB_AGE', time.time() - int(open('/content/hb').read()))",
                       timeout=90, check=False)
        line = next((l for l in raw.stdout.splitlines() if l.startswith("HB_AGE")), None)
        report["idle_session"] = ({"alive": True, "heartbeat_age_s": round(float(line.split()[1]), 1)}
                                  if line else {"alive": False, "err": raw.stderr[-500:]})
    finally:
        for s in (pinged, idle):
            s.stop()

    if samples:
        q = sorted(samples)
        report["ping"] = {"ok": len(samples), "fail": len(failures),
                          "median_s": round(statistics.median(q), 2),
                          "p95_s": round(q[int(len(q) * 0.95) - 1], 2), "max_s": round(q[-1], 2)}
    report["failures"] = failures
    out = Path(__file__).resolve().parent.parent / "results" / f"stability-{ts}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log(f"report: {json.dumps(report, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

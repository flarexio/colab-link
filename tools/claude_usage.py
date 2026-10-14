"""Print Claude plan usage (5-hour session + weekly) in one line, local time.

    python3 tools/claude_usage.py          # human line
    python3 tools/claude_usage.py --json   # {"max_pct":..., "resume_at":...}

Exit code 2 when any active limit is >= --threshold (default 90): the caller
should wrap up, stop Colab sessions and schedule a resume at `resume_at`.
The token is read from ~/.claude/.credentials.json and never printed.
"""
import argparse
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--threshold", type=float, default=90)
ap.add_argument("--json", action="store_true")
a = ap.parse_args()

tok = json.loads((Path.home() / ".claude/.credentials.json").read_text())["claudeAiOauth"]["accessToken"]
req = urllib.request.Request("https://api.anthropic.com/api/oauth/usage",
                             headers={"Authorization": f"Bearer {tok}",
                                      "anthropic-beta": "oauth-2025-04-20"})
d = json.loads(urllib.request.urlopen(req, timeout=20).read())


def local(ts):
    return datetime.fromisoformat(ts).astimezone().strftime("%m-%d %H:%M") if ts else "?"


rows = [(name, d[key]) for name, key in (("5h", "five_hour"), ("week", "seven_day")) if d.get(key)]
over = [(n, v) for n, v in rows if v["utilization"] >= a.threshold]
max_pct = max((v["utilization"] for _, v in rows), default=0)
# Resume after the latest reset among the limits that are over threshold.
resume_at = max((v["resets_at"] for _, v in over), default=None)

if a.json:
    print(json.dumps({"max_pct": max_pct, "over": bool(over),
                      "resume_at": resume_at and datetime.fromisoformat(resume_at).astimezone().isoformat(),
                      **{n: v["utilization"] for n, v in rows}}))
else:
    print(" | ".join(f"{n} {v['utilization']:.0f}% (resets {local(v['resets_at'])})" for n, v in rows)
          + ("  >>> OVER THRESHOLD, pause until " + local(resume_at) if over else ""))
sys.exit(2 if over else 0)

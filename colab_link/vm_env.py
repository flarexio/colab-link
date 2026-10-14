"""Prepare / persist the job environment on the Colab VM. Runs ON the VM (stdlib only).

    python3 vm_env.py prepare --venv vllm --packages "vllm==0.31.0" [--jit ID]
    python3 vm_env.py finalize

prepare:  restore the venv (/content/venv) from Drive, or build it with uv;
          restore the JIT/compile caches (vLLM, flashinfer, Triton) if present.
finalize: after a successful job, upload whatever prepare had to build.

Drive names are derived here because they depend on the VM image:
  venv-<name>-<pkghash>-py<ver>-torch<ver>.tar
  jit-<id>-<venvhash>.tar      (compiled kernels are only valid for one venv)
Without DRIVE_QUOTA_PROJECT in the environment, caching is skipped entirely.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

VENV = "/content/venv"
STATE = Path(os.environ.get("VM_ENV_STATE", "/content/run/env_state.json"))
# Relative to /root. worker.py points TORCHINDUCTOR_CACHE_DIR at .cache/torchinductor.
JIT_DIRS = [".cache/vllm", ".cache/flashinfer", ".triton", ".cache/torchinductor"]
JIT_FORMAT = 2  # bump when JIT_DIRS changes so old caches are not reused


def log(msg: str) -> None:
    print(f"[env] {msg}", flush=True)


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout.strip()


def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:8]


def drive():
    if not os.environ.get("DRIVE_QUOTA_PROJECT"):
        return None
    sys.path.insert(0, str(Path(__file__).parent))
    import drive_vm
    return drive_vm


def prepare(a) -> None:
    d = drive()
    py = f"{sys.version_info[0]}{sys.version_info[1]}"
    torch_v = sh("python3 -c 'import torch; print(torch.__version__)'").replace("+", "-")
    packages = sorted(a.packages.split())
    venv_key = f"venv-{a.venv}-{short_hash(' '.join(packages))}-py{py}-torch{torch_v}.tar"
    jit_key = f"jit-{a.jit}-{short_hash(venv_key)}-v{JIT_FORMAT}.tar" if a.jit else None
    st = {"venv_key": venv_key, "jit_key": jit_key, "venv_restored": False, "jit_restored": False}

    t = time.time()
    if d and d.find(venv_key):
        d.get(venv_key, "/content/venv.tar")
        sh(f"tar -xf /content/venv.tar -C /content && rm /content/venv.tar")
        st["venv_restored"] = True
        log(f"venv restored from Drive: {venv_key} ({time.time() - t:.0f}s)")
    else:
        sh(f"uv venv -q {VENV} --system-site-packages")
        sh(f"uv pip install -q --python {VENV}/bin/python " + " ".join(map(shlex.quote, packages)))
        log(f"venv built: {' '.join(packages)} ({time.time() - t:.0f}s)"
            + ("; will upload after the job" if d else ""))
    st["venv_s"] = round(time.time() - t, 1)

    if d and jit_key:
        t = time.time()
        if d.find(jit_key):
            d.get(jit_key, "/content/jit.tar")
            sh("tar -xf /content/jit.tar -C /root && rm /content/jit.tar")
            st["jit_restored"] = True
            log(f"JIT cache restored: {jit_key} ({time.time() - t:.0f}s)")
        else:
            log(f"no JIT cache yet ({jit_key}); will save after the job")
    STATE.write_text(json.dumps(st))


def finalize(a) -> None:
    d = drive()
    if not d or not STATE.exists():
        return
    st = json.loads(STATE.read_text())
    if not st["venv_restored"]:
        t = time.time()
        sh(f"tar -cf /content/venv.tar -C /content venv")
        d.put("/content/venv.tar", st["venv_key"])
        os.remove("/content/venv.tar")
        log(f"venv uploaded: {st['venv_key']} ({time.time() - t:.0f}s)")
    if st["jit_key"] and not st["jit_restored"]:
        dirs = [p for p in JIT_DIRS if os.path.isdir(f"/root/{p}")]
        if dirs:
            t = time.time()
            sh("tar -cf /content/jit.tar -C /root " + " ".join(dirs))
            d.put("/content/jit.tar", st["jit_key"])
            os.remove("/content/jit.tar")
            log(f"JIT cache saved: {st['jit_key']} ({', '.join(dirs)}, {time.time() - t:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--venv", required=True, help="venv name from profiles.toml")
    p.add_argument("--packages", required=True, help="space-separated pip specs")
    p.add_argument("--jit", default=None, help="JIT cache id (profile + settings hash)")
    sub.add_parser("finalize")
    a = ap.parse_args()
    {"prepare": prepare, "finalize": finalize}[a.cmd](a)

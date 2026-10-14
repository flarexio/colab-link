"""Build a vLLM venv on the VM, push it to Drive, and time every step. Runs ON the VM.

    python job.py IN_DIR OUT_DIR      (runner --drive PROJECT ships drive_vm.py + token)

The venv lives at /content/venv (fixed path: venv scripts embed absolute
paths) with --system-site-packages, so only what vLLM adds or upgrades on
top of the Colab image is stored. The Drive name embeds the image's Python
and torch versions, because the venv leans on the image's site-packages.
Env: PACKAGES (default "vllm").
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sh(cmd: str) -> str:
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout.strip()


def main():
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, os.getcwd())
    import drive_vm

    m = {}
    pkgs = os.environ.get("PACKAGES", "vllm")
    py = sh("python3 -c 'import sys;print(f\"{sys.version_info[0]}{sys.version_info[1]}\")'")
    torch_v = sh("python3 -c 'import torch;print(torch.__version__)'").replace("+", "-")

    t = time.time()
    sh("uv venv -q /content/venv --system-site-packages")
    sh(f"uv pip install -q --python /content/venv/bin/python {pkgs}")
    m["install_s"] = round(time.time() - t, 1)
    vllm_v = sh("/content/venv/bin/python -c 'import vllm;print(vllm.__version__)'")
    m["vllm"] = vllm_v
    m["venv_gb"] = round(int(sh("du -sb /content/venv | cut -f1")) / 1e9, 2)

    zstd = shutil.which("zstd") is not None
    m["zstd"] = zstd
    name = f"venv-vllm{vllm_v}-py{py}-torch{torch_v}.tar" + (".zst" if zstd else "")
    archive = f"/content/{name}"
    t = time.time()
    sh(f"tar -cf - -C /content venv | {'zstd -T0 -3 -q' if zstd else 'cat'} > {archive}")
    m["pack_s"] = round(time.time() - t, 1)
    m["archive_gb"] = round(os.path.getsize(archive) / 1e9, 2)

    t = time.time()
    drive_vm.put(archive, name)
    m["drive_put_s"] = round(time.time() - t, 1)
    m["drive_put_MBps"] = round(m["archive_gb"] * 1e3 / m["drive_put_s"], 0)

    t = time.time()
    drive_vm.get(name, "/tmp/check.tar")
    m["drive_get_s"] = round(time.time() - t, 1)
    m["drive_get_MBps"] = round(m["archive_gb"] * 1e3 / m["drive_get_s"], 0)

    os.makedirs("/tmp/restore", exist_ok=True)
    t = time.time()
    sh(f"{'zstd -dc -T0' if zstd else 'cat'} /tmp/check.tar | tar -xf - -C /tmp/restore")
    m["unpack_s"] = round(time.time() - t, 1)
    m["drive_name"] = name

    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    print("[job] done", json.dumps(m), flush=True)


if __name__ == "__main__":
    main()

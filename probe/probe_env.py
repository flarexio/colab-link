"""Report the Colab VM's hardware, preinstalled stack and download speed."""
import json
import shutil
import subprocess
import sys
import time
import urllib.request


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


info = {
    "python": sys.version.split()[0],
    "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
    "cuda": sh("nvcc --version | tail -1"),
    "cpu_count": sh("nproc"),
    "ram": sh("free -g | awk '/Mem:/{print $2\" GiB\"}'"),
    "disk_free_gb": round(shutil.disk_usage("/content").free / 1e9, 1),
}
for pkg in ["torch", "transformers", "vllm", "accelerate", "bitsandbytes", "huggingface_hub"]:
    try:
        mod = __import__(pkg)
        info[pkg] = getattr(mod, "__version__", "?")
    except Exception:
        info[pkg] = None

# ~100 MB slice of a public HF file to estimate model download bandwidth.
url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors"
req = urllib.request.Request(url, headers={"Range": "bytes=0-104857599"})
t = time.time()
n = len(urllib.request.urlopen(req, timeout=60).read())
info["hf_download_MBps"] = round(n / 1e6 / (time.time() - t), 1)

print(json.dumps(info, indent=2))

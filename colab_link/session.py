"""Thin wrapper around the `colab` CLI: one ColabSession == one rented VM.

Every CLI call gets its own `--config` state file so concurrent jobs never
fight over ~/.config/colab-cli/sessions.json.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


class ColabError(RuntimeError):
    pass


def _cli_env() -> dict:
    # Hermes injects a Python 3.14 PYTHONPATH that breaks the CLI's own venv.
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


class ColabSession:
    def __init__(self, name: str, gpu: str | None = None, state_dir: Path | None = None,
                 high_mem: bool = False):
        self.name = name
        self.gpu = gpu
        self.high_mem = high_mem
        state_dir = state_dir or Path.home() / ".cache" / "colab-link" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = state_dir / f"{name}.json"
        self.started = False

    # -- low level ---------------------------------------------------------
    def cli(self, *args: str, stdin: str | None = None, timeout: float = 300,
            check: bool = True, retries: int = 0) -> subprocess.CompletedProcess:
        """Run one CLI command. Only pass retries>0 for idempotent commands:
        the runtime proxy throws ReadTimeout / 5xx for a while after READY."""
        cmd = ["colab", "--config", str(self.state_file), *args]
        for attempt in range(retries + 1):
            try:
                proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True,
                                      timeout=timeout, env=_cli_env())
            except subprocess.TimeoutExpired as e:
                proc = subprocess.CompletedProcess(cmd, 124, str(e.stdout or ""), "local timeout")
            if proc.returncode == 0 or not check:
                return proc
            if attempt < retries:
                time.sleep(min(5 * 2 ** attempt, 30))
        raise ColabError(f"{' '.join(args[:1])} failed ({proc.returncode}):\n"
                         f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        args = ["new", "-s", self.name]
        if self.gpu:
            args += ["--gpu", self.gpu]
        if self.high_mem:
            args.append("--high-mem")
        out = self.cli(*args, timeout=600)
        if "READY" not in out.stdout + out.stderr:
            raise ColabError(f"session did not become READY:\n{out.stdout}\n{out.stderr}")
        self.started = True

    def wait_ready(self, max_s: float = 300) -> None:
        """READY from `colab new` does not mean the kernel proxy answers yet."""
        deadline = time.time() + max_s
        while True:
            proc = self.cli("exec", "-s", self.name, "--timeout", "30",
                            stdin="print('PING_OK')", timeout=90, check=False)
            if "PING_OK" in proc.stdout:
                return
            if time.time() > deadline:
                raise ColabError(f"kernel not reachable after {max_s}s:\n{proc.stderr[-1500:]}")
            time.sleep(10)

    def stop(self) -> None:
        if self.started:
            self.cli("stop", "-s", self.name, check=False, timeout=120)
            self.started = False

    def __enter__(self) -> "ColabSession":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- work --------------------------------------------------------------
    def exec(self, code: str, timeout: float = 60, retries: int = 3) -> str:
        """Run short Python in the kernel. `timeout` only bounds how long the
        CLI waits; the kernel keeps running past it, so long work must be
        launched detached (see runner.py) rather than through exec.
        Retried by default, so `code` must be idempotent."""
        out = self.cli("exec", "-s", self.name, "--timeout", str(timeout),
                       stdin=code, timeout=timeout + 60, retries=retries)
        return out.stdout

    def upload(self, local: Path, remote: str) -> None:
        self.cli("upload", "-s", self.name, str(local), remote, timeout=1800, retries=3)

    def download(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self.cli("download", "-s", self.name, remote, str(local), timeout=1800, retries=3)

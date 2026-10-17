"""Load profiles.toml (repo root, or $COLAB_LINK_PROFILES)."""
from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Profile:
    name: str
    gpu: str | None
    venv: str | None
    packages: list[str]
    job: str | None
    env: dict[str, str] = field(default_factory=dict)
    idle_minutes: float = 10   # serve: a warm worker leaves after this long without jobs
    max_workers: int = 1       # serve: VMs per pool (Colab allows 2 per GPU type)

    @property
    def servable(self) -> bool:
        """`colab-link llm` can serve it: a vLLM venv and a MODEL to load."""
        return any(p.startswith("vllm") for p in self.packages) and "MODEL" in self.env

    def jit_id(self, env: dict[str, str]) -> str:
        """Compiled kernels depend on model, quantization, limits and GPU."""
        key = json.dumps({"gpu": self.gpu, "env": dict(sorted(env.items()))})
        return f"{self.name}-{hashlib.sha1(key.encode()).hexdigest()[:6]}"


@dataclass
class Config:
    drive_project: str | None
    drive_root: str
    drive_jobs_root: str
    profiles: dict[str, Profile]


def load(path: str | None = None) -> Config:
    path = Path(path or os.environ.get("COLAB_LINK_PROFILES") or ROOT / "profiles.toml")
    raw = tomllib.loads(path.read_text())
    venvs = raw.get("venvs", {})
    profiles = {}
    for name, p in raw.get("profiles", {}).items():
        venv = p.get("venv")
        if venv and venv not in venvs:
            raise ValueError(f"profile {name}: unknown venv {venv!r}")
        profiles[name] = Profile(name=name, gpu=p.get("gpu"), venv=venv,
                                 packages=list(venvs.get(venv, {}).get("packages", [])),
                                 job=p.get("job"), env={k: str(v) for k, v in p.get("env", {}).items()},
                                 idle_minutes=float(p.get("idle_minutes", 10)),
                                 max_workers=int(p.get("max_workers", 1)))
    drive = raw.get("drive", {})
    return Config(drive_project=os.environ.get("COLAB_LINK_DRIVE_PROJECT") or drive.get("project") or None,
                  drive_root=drive.get("root", "colab-link-cache"),
                  drive_jobs_root=drive.get("jobs_root", "colab-link-jobs"), profiles=profiles)

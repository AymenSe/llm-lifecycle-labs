"""Environment and hardware report, reused by every training run."""

import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import psutil

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[3]

# Allowlists: the report never dumps os.environ, only these names/packages.
ENV_VARS = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
PACKAGES = ("torch", "numpy", "pyyaml", "nvidia-ml-py", "psutil", "mlflow", "lmlab")


def _git(repo_root: Path, *args: str) -> str:
    """Run `git -C repo_root <args>` and return stripped stdout. Raises on failure."""
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_state(repo_root: Path) -> dict:
    """{'sha': str, 'branch': str, 'dirty': bool, 'untracked': int}.

    Never raises outside a git repo (or without git): returns sha=None, and
    branch/dirty=None because the state is unknown, not clean.
    """
    try:
        sha = _git(repo_root, "rev-parse", "HEAD")
        branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
        # Not _git(): .strip() would eat the leading space of the first status line.
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (subprocess.CalledProcessError, OSError):
        return {"sha": None, "branch": None, "dirty": None, "untracked": 0}

    untracked = sum(1 for line in status if line.startswith("??"))
    dirty = any(line and not line.startswith("??") for line in status)
    return {"sha": sha, "branch": branch, "dirty": dirty, "untracked": untracked}


def torch_info() -> dict:
    """torch version, CUDA/cuDNN/NCCL versions and the matmul precision settings."""
    try:
        import torch
    except ImportError:
        return {"installed": False}

    cuda = torch.cuda.is_available()
    nccl = torch.cuda.nccl.version() if cuda else None
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": list(nccl) if nccl else None,
        "cuda_available": cuda,
        "visible_device_count": torch.cuda.device_count(),
        "bf16_supported": bool(cuda and torch.cuda.is_bf16_supported()),
        "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def gpu_inventory() -> list[dict]:
    raise NotImplementedError


def _topology() -> str | None:
    """Raw `nvidia-smi topo -m` text, or None if unavailable."""
    try:
        return subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None


def _package_versions() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _disk(path: str | None) -> dict:
    if not path:
        return {"path": None, "free_gib": None}
    try:
        free = shutil.disk_usage(path).free / 2**30
    except OSError:
        return {"path": path, "free_gib": None}
    return {"path": path, "free_gib": round(free, 1)}


def collect_env(include_gpu: bool = True) -> dict:
    """JSON-serializable report. No hostname, no secrets (allowlisted env vars only)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": _package_versions(),
        "torch": torch_info(),
        "gpus": gpu_inventory() if include_gpu else [],
        "topology": _topology() if include_gpu else None,
        "cpu": {
            "model": _cpu_model(),
            "logical_cores": os.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
        },
        "ram_gib": round(psutil.virtual_memory().total / 2**30, 1),
        "disk": _disk(os.environ.get("LMLAB_ROOT")),
        "git": git_state(REPO_ROOT),
        "env": {name: os.environ.get(name) for name in ENV_VARS},
        "warnings": [],
    }

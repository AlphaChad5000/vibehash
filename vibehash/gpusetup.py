"""Find this computer's GPU and install the PyTorch build that trains on it.

Detection works before PyTorch is installed: it asks the driver tools
(nvidia-smi, ROCm) and the operating system's device list what's present.
PyTorch publishes one build per GPU platform and CUDA / ROCm version, and the
set changes with every release, so the candidates for a card are checked
against PyTorch's package index at install time rather than pinned to one URL.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

INDEX = "https://download.pytorch.org/whl/"
Version = tuple[int, int]


@dataclass(frozen=True)
class Gpu:
    vendor: str  # nvidia, amd, intel or apple
    name: str
    cuda: Version | None = None  # newest CUDA the NVIDIA driver supports
    compute: Version | None = None  # NVIDIA compute capability
    rocm: Version | None = None  # installed ROCm / HIP SDK

    def describe(self) -> str:
        extra = f" (driver supports CUDA {self.cuda[0]}.{self.cuda[1]})" if self.cuda else ""
        return self.name + extra


@dataclass(frozen=True)
class Build:
    label: str
    tag: str | None  # PyTorch's local version tag, e.g. cu130; None for the plain PyPI build
    min_compute: Version = (0, 0)
    below_compute: Version = (99, 0)
    needs: Version = (0, 0)  # CUDA or ROCm version the driver must support

    @property
    def index(self) -> str | None:
        return INDEX + self.tag if self.tag else None


def _cuda(tag: str, cuda: Version, min_compute: Version, below_compute: Version = (99, 0)) -> Build:
    return Build(f"CUDA {cuda[0]}.{cuda[1]}", tag, min_compute, below_compute, cuda)


# Newest first. CUDA 13 dropped Maxwell, Pascal and Volta; PyTorch's 12.8+ builds dropped Maxwell and
# Pascal; RTX 50-series (Blackwell, compute 10.0 and up) needs 12.8 or later.
CUDA_BUILDS = [
    _cuda("cu132", (13, 2), (7, 5)), _cuda("cu130", (13, 0), (7, 5)),
    _cuda("cu129", (12, 9), (7, 0)), _cuda("cu128", (12, 8), (7, 0)),
    _cuda("cu126", (12, 6), (5, 0), (10, 0)), _cuda("cu124", (12, 4), (5, 0), (10, 0)),
    _cuda("cu121", (12, 1), (5, 0), (10, 0)), _cuda("cu118", (11, 8), (3, 7), (10, 0)),
]
ROCM_BUILDS = [Build(f"ROCm {v[0]}.{v[1]}", f"rocm{v[0]}.{v[1]}", needs=v)
               for v in [(7, 2), (7, 1), (7, 0), (6, 4), (6, 3), (6, 2)]]
XPU = Build("Intel XPU", "xpu")
MPS = Build("Apple MPS", None)


def _run(cmd: list[str], timeout: float = 20) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _version(text: str) -> Version | None:
    m = re.search(r"(\d+)\.(\d+)", text or "")
    return (int(m[1]), int(m[2])) if m else None


def parse_nvidia_smi(header: str, query: str) -> list[Gpu]:
    """`header` is plain `nvidia-smi` output; `query` is `--query-gpu=name,compute_cap` (or just name) CSV."""
    m = re.search(r"CUDA (?:UMD )?Version:\s*(\d+\.\d+)", header or "")
    cuda = _version(m[1]) if m else None
    gpus = []
    for line in (query or "").splitlines():
        name, _, cap = line.partition(",")
        if name.strip():
            gpus.append(Gpu("nvidia", name.strip(), cuda, _version(cap)))
    return gpus


def _os_gpu_names() -> list[str]:
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"])
        return [line.strip() for line in (out or "").splitlines() if line.strip()]
    if sys.platform.startswith("linux"):
        out = _run(["lspci"]) or ""
        return [line.split(": ", 1)[1] for line in out.splitlines()
                if re.search(r"VGA|3D controller|Display controller", line) and ": " in line]
    return []


def _rocm_version() -> Version | None:
    for f in (Path("/opt/rocm/.info/version"),):
        if f.exists():
            return _version(f.read_text())
    if os.environ.get("HIP_PATH"):  # Windows HIP SDK, e.g. C:\Program Files\AMD\ROCm\6.2\
        return _version(os.environ["HIP_PATH"])
    return _version(_run(["hipconfig", "--version"]) or "")


def classify(name: str) -> str | None:
    n = name.lower()
    if "nvidia" in n or "geforce" in n or "quadro" in n:
        return "nvidia"
    if "amd" in n or "radeon" in n or re.search(r"\bati\b", n):
        return "amd"
    if "intel" in n:
        return "intel"
    return None


def detect() -> list[Gpu]:
    """GPUs on this machine, whether or not PyTorch can use them yet."""
    found: list[Gpu] = []
    if shutil.which("nvidia-smi"):
        query = (_run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"])
                 or _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]))  # older drivers lack compute_cap
        found += parse_nvidia_smi(_run(["nvidia-smi"]) or "", query or "")
    if sys.platform == "darwin" and platform.machine() == "arm64":
        found.append(Gpu("apple", "Apple silicon GPU"))
    for name in _os_gpu_names():
        vendor = classify(name)
        if vendor == "nvidia" and any(g.vendor == "nvidia" for g in found):
            continue  # already listed by nvidia-smi, with driver details
        if vendor:
            found.append(Gpu(vendor, name, rocm=_rocm_version() if vendor == "amd" else None))
    return found


def builds_for(gpu: Gpu) -> tuple[list[Build], str | None]:
    """PyTorch builds that should run on `gpu`, best first, or why there are none."""
    if gpu.vendor == "nvidia":
        if gpu.cuda is None:
            return [], "its driver isn't installed (no nvidia-smi); get it from nvidia.com"
        fits = [b for b in CUDA_BUILDS if b.needs <= gpu.cuda
                and (gpu.compute is None or b.min_compute <= gpu.compute < b.below_compute)]
        return fits, None if fits else "its driver is too old for current PyTorch; update it from nvidia.com"
    if gpu.vendor == "apple":
        return [MPS], None
    if gpu.vendor == "intel":
        if "arc" not in gpu.name.lower():
            return [], "PyTorch only supports Intel Arc graphics (Arc cards and Core Ultra laptops)"
        return [XPU], None
    if gpu.vendor == "amd":
        if gpu.rocm is None:
            return [], "PyTorch needs AMD's ROCm installed first (Linux)"
        fits = [b for b in ROCM_BUILDS if b.needs <= gpu.rocm]
        return fits, None if fits else "its ROCm is too old for current PyTorch"
    return [], "unknown GPU"


def published(build: Build) -> bool:
    """Whether PyTorch currently publishes `build` for this Python version and operating system."""
    if build.index is None:
        return True
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "index", "versions", "torch", "--index-url", build.index],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return True  # can't check; let pip decide at install time
    if out.returncode == 0:
        return True
    return "No matching distribution" not in out.stdout + out.stderr


def choose(gpus: list[Gpu], available=published) -> tuple[Gpu | None, Build | None, str | None]:
    """The GPU to set up and its build, or (gpu, None, why not). Vendors in the order workers prefer them."""
    order = {"nvidia": 0, "amd": 1, "intel": 2, "apple": 3}
    reasons = []
    for gpu in sorted(gpus, key=lambda g: order.get(g.vendor, 9)):
        builds, why = builds_for(gpu)
        for b in builds:
            if available(b):
                return gpu, b, None
        reasons.append((gpu, why or f"PyTorch has no build for it on Python {sys.version_info[0]}.{sys.version_info[1]}"))
    return (reasons[0][0], None, reasons[0][1]) if reasons else (None, None, None)


def installed_torch() -> str | None:
    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        return None


def install(build: Build) -> int:
    """pip-install `build` into the Python running this, replacing a PyTorch built for something else."""
    pip = [sys.executable, "-m", "pip"]
    sys.stdout.flush()  # pip writes straight to the terminal; what we printed must come first
    current = installed_torch()
    if current and build.tag and not current.endswith("+" + build.tag):
        code = subprocess.call(pip + ["uninstall", "-y", "torch"])
        if code:
            return code
    return subprocess.call(pip + ["install", "--upgrade", "torch"] + (["--index-url", build.index] if build.index else []))


def devices() -> list[dict]:
    """`vibehash devices --json`, run in a fresh process: one that never imported torch, so pip can replace it
    (Windows locks loaded DLLs) and a just-installed build is the one tested."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent.parent),
                                                                     os.environ.get("PYTHONPATH")])))
    try:
        out = subprocess.run([sys.executable, "-m", "vibehash", "devices", "--json"], capture_output=True, text=True,
                             timeout=300, env=env)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return []

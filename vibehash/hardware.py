"""Measure the machine a worker is actually running on.

The benchmark times the job's own training loop rather than a peak-FLOPs
matmul, so the number reflects real throughput for this workload (Python
overhead included). The coordinator still caps a worker's capacity by the
speed it *observes*, so overstating this only gets a worker fewer units later.
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys
import time

from .model import train_segment
from .tasks import TrainingJob


def total_memory_mb() -> int:
    try:
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullTotalPhys // (1024 * 1024))
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024))
    except (AttributeError, ValueError, OSError):
        return 1024


def benchmark(job: TrainingJob, seconds: float = 1.0, backend=None) -> float:
    """Training FLOP/s this machine sustains on `job`, on `backend` (CPU or GPU)."""
    theta = job.spec.init_params(job.init_seed)
    train_segment(job.spec, job.data, theta, 0, 0, 1, job.lr, job.batch_size, backend)  # warm-up: GPU kernels compile here
    steps, start = 0, time.perf_counter()
    while True:
        theta = train_segment(job.spec, job.data, theta, 0, steps, job.checkpoint_every, job.lr, job.batch_size, backend)
        steps += job.checkpoint_every
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            return steps * job.spec.flops_per_step(job.batch_size) / elapsed


def describe() -> str:
    return f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs, {total_memory_mb():,} MB RAM"


# --- the hardware class a worker registers with -------------------------------------------------------------------
#
# A worker tells the node what kind of machine it is, in a form the node can count and a page can show ("12 of this
# card, 40 of that"): {"vendor", "arch", "memory_gb", "label"}. Every field is derived from what the driver reports and
# squeezed through a short whitelist, never typed by a person, so the census can't be filled with made-up cards. Nothing
# in it is checked against the work (a worker could lie about its card as it could about its speed), and nothing in it
# identifies a person: no serial numbers, no host names.
VENDORS = ("nvidia", "amd", "intel", "apple", "cpu")
MAX_LABEL, MAX_ARCH, MAX_MEMORY_GB = 40, 16, 4096
LABEL_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .+-/")
ARCH_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_.-")
_NOISE = ("NVIDIA", "GeForce", "AMD", "Radeon", "Intel", "Graphics", "Corporation")  # "(R)" and "(TM)" go first


def short_label(name: str) -> str:
    """A driver's device name as a card is usually called: "NVIDIA GeForce RTX 4060 Laptop GPU" -> "RTX 4060 Laptop"."""
    words = [w for w in str(name).replace("(R)", " ").replace("(TM)", " ").split() if w not in _NOISE and w != "GPU"]
    text = " ".join(words) or str(name).strip() or "GPU"
    text = "".join(ch for ch in text if ch in LABEL_CHARS).strip()
    return text[:MAX_LABEL].strip() or "GPU"


def hardware_class(backend) -> dict:
    """What kind of machine `backend` trains on: `{"vendor", "arch", "memory_gb", "label"}`, as registered."""
    name = getattr(backend, "name", "cpu")
    if name == "cpu":
        return _cpu_class()
    t, device = backend.torch, backend.device
    idx = int(device.split(":")[1]) if ":" in device else 0
    try:
        if name == "cuda":
            props = t.cuda.get_device_properties(idx)
            hip = getattr(t.version, "hip", None)
            arch = getattr(props, "gcnArchName", "") if hip else f"sm_{props.major}{props.minor}"
            return _canonical("amd" if hip else "nvidia", arch, props.total_memory / 2**30, short_label(props.name))
        if name == "xpu":
            props = t.xpu.get_device_properties(idx)
            return _canonical("intel", "xpu", props.total_memory / 2**30, short_label(props.name))
        if name == "mps":
            return _canonical("apple", "mps", total_memory_mb() / 1024, _apple_chip())
    except Exception:  # a driver that won't say: still a worker, just an unnamed one
        pass
    return _canonical({"cuda": "nvidia", "xpu": "intel", "mps": "apple"}.get(name, "cpu"), name, total_memory_mb() / 1024, "GPU")


def _cpu_class() -> dict:
    cpus = os.cpu_count() or 1
    size = "2 threads" if cpus <= 2 else f"{min(cpus, 128)} threads"  # a count, not a model: CPU names are long and many
    return _canonical("cpu", platform.machine().lower() or "cpu", total_memory_mb() / 1024, f"CPU {size}")


def _apple_chip() -> str:
    """"Apple M3" from the chip's brand string, or "Apple GPU" when macOS won't say."""
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2).stdout
        words = out.split()
        if words and words[0] == "Apple":
            return short_label(" ".join(words[:3]))
    except Exception:
        pass
    return "Apple GPU"


def _canonical(vendor: str, arch: str, memory_gb: float, label: str) -> dict:
    # ROCm names an arch with its feature flags, "gfx1100:sramecc+:xnack-": the same card, whatever its flags
    arch = "".join(ch for ch in str(arch).split(":")[0].lower() if ch in ARCH_CHARS)[:MAX_ARCH] or "unknown"
    return {"vendor": vendor if vendor in VENDORS else "cpu", "arch": arch,
            "memory_gb": int(min(max(round(memory_gb), 0), MAX_MEMORY_GB)), "label": short_label(label)}


def valid_class(d) -> bool:
    """Whether `d` is a hardware class as hardware_class makes one: the node refuses a registration whose isn't."""
    if not isinstance(d, dict) or set(d) != {"vendor", "arch", "memory_gb", "label"}:
        return False
    vendor, arch, mem, label = d["vendor"], d["arch"], d["memory_gb"], d["label"]
    return (vendor in VENDORS and isinstance(arch, str) and 0 < len(arch) <= MAX_ARCH and set(arch) <= ARCH_CHARS
            and isinstance(mem, int) and not isinstance(mem, bool) and 0 <= mem <= MAX_MEMORY_GB
            and isinstance(label, str) and 0 < len(label) <= MAX_LABEL and set(label) <= LABEL_CHARS
            and label == label.strip())


def class_name(d: dict) -> str:
    """One line for a hardware class, as the census and the worker's log show it: "RTX 4060, 8 GB (nvidia sm_89)"."""
    return f"{d['label']}, {d['memory_gb']} GB ({d['vendor']} {d['arch']})"


# --- the number setup a worker registers with ----------------------------------------------------------------------
#
# Beside its class, a worker says how its machine does arithmetic for Evo, the work whose checks compare token choices
# to a fraction of a logit (es2.Spec2.margin): {"backend", "dtype", "arch", "units", "torch", "toolkit"}. Two machines
# with the same setup get the same bits (an RTX 4070 SUPER checking another's answers: every gap 0), while a change in
# any one of them can move a logit: bf16 against float32 (0.375 measured, docs/GPUS.md), another card's number of SMs
# (cuBLAS promises equal results only on the same architecture with the same SM count), another CUDA or ROCm build, or
# another PyTorch. It goes in the register message as its own field ("numerics"), so a node from before it refuses just
# that field and the class still gets through (client.Client). Like the class, it's derived from what the drivers say
# and squeezed through whitelists, nothing identifies a person, and a worker could lie about it: a checker paired by it
# holds the liar to the setup it named, so lying only gets its own work rejected.
BACKENDS = ("cuda", "rocm", "xpu", "mps", "cpu", "npu")
DTYPES = ("bf16", "float32")
MAX_UNITS = 4096
VERSION_CHARS = set("0123456789.")
NUMERIC_KEYS = {"backend", "dtype", "arch", "units", "torch", "toolkit"}


def evo_dtype(device: str) -> str:
    """What Evo's model computes in on `device`: bf16 on CUDA and ROCm (which PyTorch calls "cuda"), float32 everywhere
    else (es_client.runtime2 and round_factors)."""
    return "bf16" if str(device).startswith("cuda") else "float32"


def numerics(backend) -> dict:
    """How `backend` does Evo's arithmetic, as registered: see above. A machine without PyTorch is the CPU's float32
    with no version."""
    name = getattr(backend, "name", "cpu")
    t = getattr(backend, "torch", None)
    device = getattr(backend, "device", "cpu")
    version = _torch_version(t)
    if name == "cpu" or t is None:
        return _canonical_numerics("cpu", "float32", platform.machine().lower() or "cpu", os.cpu_count() or 1, version, "")
    idx = int(device.split(":")[1]) if ":" in device else 0
    arch, units, toolkit, kind = name, 0, "", name
    try:
        if name == "cuda":
            props = t.cuda.get_device_properties(idx)
            hip = getattr(t.version, "hip", None)
            kind = "rocm" if hip else "cuda"
            arch = getattr(props, "gcnArchName", "") if hip else f"sm_{props.major}{props.minor}"
            units = int(getattr(props, "multi_processor_count", 0))
            toolkit = f"rocm{_major_minor(hip)}" if hip else f"cuda{_major_minor(getattr(t.version, 'cuda', ''))}"
        elif name == "xpu":
            props = t.xpu.get_device_properties(idx)
            units = int(getattr(props, "gpu_eu_count", 0) or 0)
        elif name == "mps":
            arch = "mps"
    except Exception:  # noqa: BLE001 - a driver that won't say: the setup without its details
        pass
    return _canonical_numerics(kind, evo_dtype(device), arch, units, version, toolkit)


def _torch_version(t) -> str:
    """PyTorch's version without its build tag: "2.14.1+cu126" -> "2.14.1" (the tag is the toolkit's, kept apart)."""
    return str(getattr(t, "__version__", "")).split("+")[0] if t is not None else ""


def _major_minor(v) -> str:
    """"12.6" from "12.6", "6.2" from "6.2.41133-dd7f95766"; "" when there's no version."""
    parts = [p for p in str(v or "").replace("-", ".").split(".") if p.isdigit()][:2]
    return ".".join(parts)


def _canonical_numerics(backend: str, dtype: str, arch: str, units, torch_version: str, toolkit: str) -> dict:
    arch = "".join(ch for ch in str(arch).split(":")[0].lower() if ch in ARCH_CHARS)[:MAX_ARCH] or "unknown"
    toolkit = "".join(ch for ch in str(toolkit).lower() if ch in ARCH_CHARS)[:MAX_ARCH]
    if toolkit in ("cuda", "rocm"):  # the build said no version
        toolkit = ""
    version = "".join(ch for ch in str(torch_version).split("+")[0] if ch in VERSION_CHARS)[:MAX_ARCH]
    try:
        units = int(min(max(int(units), 0), MAX_UNITS))
    except (TypeError, ValueError):
        units = 0
    return {"backend": backend if backend in BACKENDS else "cpu", "dtype": dtype if dtype in DTYPES else "float32",
            "arch": arch, "units": units, "torch": version, "toolkit": toolkit}


def valid_numerics(d) -> bool:
    """Whether `d` is a number setup as numerics() makes one: the node refuses a registration whose isn't."""
    if not isinstance(d, dict) or set(d) != NUMERIC_KEYS:
        return False
    arch, units, version, toolkit = d["arch"], d["units"], d["torch"], d["toolkit"]
    return (d["backend"] in BACKENDS and d["dtype"] in DTYPES
            and isinstance(arch, str) and 0 < len(arch) <= MAX_ARCH and set(arch) <= ARCH_CHARS
            and isinstance(units, int) and not isinstance(units, bool) and 0 <= units <= MAX_UNITS
            and isinstance(version, str) and len(version) <= MAX_ARCH and set(version) <= VERSION_CHARS
            and isinstance(toolkit, str) and len(toolkit) <= MAX_ARCH and set(toolkit) <= ARCH_CHARS
            and toolkit not in ("cuda", "rocm"))


def numerics_key(d) -> str | None:
    """One string for a number setup, the same for two machines that should get the same bits: "cuda bf16 sm_89 x24
    cuda12.6 torch2.14.1". None for no setup (or one that isn't canonical)."""
    if not valid_numerics(d):
        return None
    return (f"{d['backend']} {d['dtype']} {d['arch']} x{d['units']}"
            + (f" {d['toolkit']}" if d["toolkit"] else "") + (f" torch{d['torch']}" if d["torch"] else ""))

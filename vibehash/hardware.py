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
_NOISE = ("NVIDIA", "GeForce", "AMD", "Radeon", "Intel(R)", "Intel", "(R)", "(TM)", "Graphics", "Arc(TM)", "Corporation")


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
    arch = "".join(ch for ch in str(arch).lower() if ch in ARCH_CHARS)[:MAX_ARCH] or "unknown"
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

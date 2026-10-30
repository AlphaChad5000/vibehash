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

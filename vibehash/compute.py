"""Where training runs: NumPy on the CPU, or PyTorch on a GPU.

Both backends run exactly the same maths in the same order (model.py is the
reference), so a unit trained on an NVIDIA card can be checked on an Intel
laptop's CPU and vice versa. Floating-point results still differ slightly
between devices, which is what a job's `tolerance` absorbs.

Supported PyTorch devices: "cuda" (NVIDIA, and AMD via ROCm builds), "xpu"
(Intel Arc and Core Ultra graphics) and "mps" (Apple silicon). GPUs only run
float32 jobs; float64 jobs always run on the CPU because many consumer GPUs
(Intel Arc, Apple) have no float64 support.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .model import Dataset, ModelSpec, batch_indices, train_segment_numpy


class NumpyBackend:
    name = "cpu"
    device = "cpu"

    def train_segment(self, spec: ModelSpec, data: Dataset, theta: np.ndarray, unit_seed: int, start_step: int,
                      n_steps: int, lr: float, batch_size: int) -> np.ndarray:
        return train_segment_numpy(spec, data, theta, unit_seed, start_step, n_steps, lr, batch_size)

    def describe(self) -> str:
        return f"CPU (NumPy, {os.cpu_count()} threads)"


def pin_precision() -> None:
    """float32 matmuls in full float32 on every backend: TF32 off for cuBLAS and cuDNN (NVIDIA, AMD), oneDNN's reduced
    modes off on the CPU (TF32, or bf16 for float32 on CPUs that have it), and Intel GPUs' own TF32 switch off. On
    AMD, matmuls go to rocBLAS rather than hipBLASLt (below). TF32
    keeps only 10 mantissa bits, so a replay wouldn't match. Only writes the settings, never reads them: from PyTorch
    2.9 a library that sets the newer `fp32_precision` flags after the older `allow_tf32` ones were set makes *reading*
    `allow_tf32` raise, and turns TF32 back on, so the code paths that need full float32 call this again before their
    work, not only at start."""
    import torch

    torch.set_float32_matmul_precision("highest")  # cuBLAS's and the CPU's oneDNN matmuls
    torch.backends.cuda.matmul.allow_tf32 = False  # (also sets the newer torch.backends.cuda.matmul.fp32_precision)
    torch.backends.cudnn.allow_tf32 = False
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and getattr(xpu, "_is_compiled", lambda: False)():
        # An Intel GPU's oneDNN matmuls use TF32 by this flag alone, not set_float32_matmul_precision. (A build without
        # Intel GPU support only warns that it has none when it's set, so it's left alone there.)
        torch.backends.mkldnn.allow_tf32 = False
    if getattr(torch.version, "hip", None):
        # On ROCm "cublas" is rocBLAS. PyTorch's default there on RDNA3/4, MI200/300/350 and Strix Halo is hipBLASLt,
        # which deterministic mode doesn't cover, AMD's own table marks only partly safe in graphs, and has an open
        # report of graph replays sharing its split-K state ("hangs or wrong results"); Evo replays graphs of bf16
        # matmuls (docs/GPUS.md, risk 3). rocBLAS is slower on some shapes: not measured yet.
        try:
            torch.backends.cuda.preferred_blas_library("cublas")
        except (AttributeError, RuntimeError):  # an older PyTorch without the switch
            pass


def has_float64(device: str) -> bool:
    """Whether `device` computes in float64. Apple's MPS never does; Intel Arc A-series GPUs and some other Intel
    graphics don't (PyTorch reports it as has_fp64); NVIDIA and AMD GPUs and every CPU do."""
    kind = device.split(":")[0]
    if kind == "mps":
        return False
    if kind == "xpu":
        import torch
        try:
            return bool(getattr(torch.xpu.get_device_properties(int(device.split(":")[1]) if ":" in device else 0),
                                "has_fp64", False))
        except Exception:  # noqa: BLE001 - a driver that won't say: assume the worst
            return False
    return True


def gpu_kind(torch) -> str:
    """"ROCm" for PyTorch's AMD build (which calls its GPUs "cuda" too), else "CUDA"."""
    return "ROCm" if getattr(torch.version, "hip", None) else "CUDA"


class TorchBackend:
    """PyTorch on one device, configured for reproducibility (no TF32, deterministic kernels)."""

    def __init__(self, device: str):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; must precede CUDA init
        import torch

        self.torch = torch
        self.device = device
        self.name = device.split(":")[0]
        torch.use_deterministic_algorithms(True, warn_only=True)
        pin_precision()
        torch.backends.cudnn.benchmark = False
        self._cpu = NumpyBackend()
        self._data: dict[int, tuple] = {}

    def _tensors(self, data: Dataset):
        key = id(data)
        if key not in self._data:
            t = self.torch
            self._data[key] = (data, t.tensor(data.x_train, dtype=t.float32, device=self.device),
                               t.tensor(data.y_train, dtype=t.long, device=self.device))
        return self._data[key][1:]

    def train_segment(self, spec: ModelSpec, data: Dataset, theta: np.ndarray, unit_seed: int, start_step: int,
                      n_steps: int, lr: float, batch_size: int) -> np.ndarray:
        if spec.precision != "float32":
            return self._cpu.train_segment(spec, data, theta, unit_seed, start_step, n_steps, lr, batch_size)
        t = self.torch
        pin_precision()  # (again: something in this process may have turned TF32 on since)
        x_all, y_all = self._tensors(data)
        flat = t.tensor(np.asarray(theta, dtype=np.float32), device=self.device)
        shapes = list(zip(spec.widths, spec.widths[1:]))
        n = len(data.y_train)
        with t.no_grad():
            for step in range(start_step, start_step + n_steps):
                idx = t.from_numpy(batch_indices(unit_seed, step, n, batch_size)).to(self.device)
                x, y = x_all[idx], y_all[idx]
                layers, i = [], 0
                for a, b in shapes:
                    layers.append((flat[i:i + a * b].view(a, b), flat[i + a * b:i + a * b + b]))
                    i += a * b + b
                # Forward: same formula as model._forward.
                acts = [x]
                for w, b in layers[:-1]:
                    acts.append(t.tanh(acts[-1] @ w + b))
                w, b = layers[-1]
                logits = acts[-1] @ w + b
                logits = logits - logits.max(dim=1, keepdim=True).values
                probs = t.exp(logits)
                probs = probs / probs.sum(dim=1, keepdim=True)
                # Backward: same formula as model._gradient.
                d = probs.clone()
                d[t.arange(len(y), device=self.device), y] -= 1
                d = d / len(y)
                grads = []
                for li in range(len(layers) - 1, -1, -1):
                    w, _ = layers[li]
                    grads.append((acts[li].T @ d, d.sum(dim=0)))
                    if li:
                        d = (d @ w.T) * (1 - acts[li] ** 2)
                g = t.cat([part.reshape(-1) for gw, gb in reversed(grads) for part in (gw, gb)])
                flat = flat - t.tensor(lr, dtype=t.float32, device=self.device) * g
        return flat.cpu().numpy().astype(np.float64)

    def memory_mb(self) -> int | None:
        t, idx = self.torch, int(self.device.split(":")[1]) if ":" in self.device else 0
        try:
            if self.name == "cuda":
                return int(t.cuda.get_device_properties(idx).total_memory // 2**20)
            if self.name == "xpu":
                return int(t.xpu.get_device_properties(idx).total_memory // 2**20)
            if self.name == "mps":  # unified memory: what Metal lets one process use, about 2/3 to 3/4 of the RAM
                return int(t.mps.recommended_max_memory() // 2**20) or None
        except Exception:
            return None
        return None

    def describe(self) -> str:
        t = self.torch
        try:
            if self.name == "cuda":
                return f"{t.cuda.get_device_name(self.device)} ({gpu_kind(t)})"
            if self.name == "xpu":
                return f"{t.xpu.get_device_name(self.device)} (Intel XPU)"
        except Exception:
            pass
        return "Apple GPU (MPS)" if self.name == "mps" else self.device


@dataclass(frozen=True)
class Device:
    id: str
    label: str


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def list_devices() -> list[Device]:
    """Every device this machine can train on, best first."""
    found: list[Device] = []
    if torch_available():
        import torch
        if torch.cuda.is_available():
            found += [Device(f"cuda:{i}", f"{torch.cuda.get_device_name(i)} ({gpu_kind(torch)})")
                      for i in range(torch.cuda.device_count())]
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            found += [Device(f"xpu:{i}", f"{torch.xpu.get_device_name(i)} (Intel XPU)") for i in range(torch.xpu.device_count())]
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            found.append(Device("mps", "Apple GPU (MPS)"))
    found.append(Device("cpu", f"CPU (NumPy, {os.cpu_count()} threads)"))
    return found


SELF_TEST_SECONDS = 120  # a healthy GPU compiles its first kernels in seconds; an old driver can hang forever
HUNG = "the test hung"  # how a self_test problem begins when the device never answered


def self_test(backend, timeout: float = SELF_TEST_SECONDS) -> str | None:
    """Train a tiny model for a few steps and compare with NumPy. Returns None if the device works,
    else what went wrong. Catches broken or outdated GPU drivers before they cost a worker its rounds.

    The step runs on a daemon thread so a driver that never returns (seen on Intel Arc with a driver older
    than its PyTorch build) costs `timeout` seconds instead of the whole process. PyTorch releases the GIL
    while it waits on the device, so the caller gets control back; the stuck thread is abandoned."""
    import threading

    from .model import Dataset

    spec = ModelSpec(4, (16,), 3, "float32")
    data = Dataset.synthetic(spec, 1, n_train=64, n_val=8)
    theta = spec.init_params(2)
    result: list = []

    def run():
        try:
            result.append(backend.train_segment(spec, data, theta, 3, 0, 3, 0.1, 16))
        except Exception as e:  # driver/runtime failures surface as all sorts of exception types
            result.append(e)

    worker = threading.Thread(target=run, name=f"self-test {backend.device}", daemon=True)
    worker.start()
    worker.join(timeout)
    if not result:
        return f"{HUNG}: a tiny training step got no answer from the device in {timeout:.0f}s"
    got = result[0]
    if isinstance(got, Exception):
        return f"{type(got).__name__}: {got}"
    want = NumpyBackend().train_segment(spec, data, theta, 3, 0, 3, 0.1, 16)
    err = float(np.linalg.norm(got - want) / np.linalg.norm(want))
    return None if err < 1e-4 else f"results differ from the CPU by {err:.1e}"


def get_backend(device: str = "auto", log=None):
    """'auto' picks the first GPU that passes a self-test, else the CPU."""
    if device == "auto":
        for d in list_devices():
            if d.id == "cpu":
                return NumpyBackend()
            backend = TorchBackend(d.id)
            problem = self_test(backend)
            if problem is None:
                return backend
            if log:
                log.warning("%s isn't usable (%s); update its graphics driver. Trying the next device.", d.label, problem)
        return NumpyBackend()
    if device == "cpu":
        return NumpyBackend()
    if not torch_available():
        raise RuntimeError("GPU training needs PyTorch. See the install instructions for your GPU.")
    if device not in {d.id for d in list_devices()} and f"{device}:0" not in {d.id for d in list_devices()}:
        raise RuntimeError(f"device {device!r} not found; available: {', '.join(d.id for d in list_devices())}")
    backend = TorchBackend(device if ":" in device or device == "mps" else f"{device}:0")
    problem = self_test(backend)
    if problem:
        raise RuntimeError(f"{device} failed its self-test ({problem}). Updating the graphics driver usually fixes this.")
    return backend


_active = NumpyBackend()


def active():
    return _active


def use(backend) -> None:
    """Set the backend this process trains and verifies with."""
    global _active
    _active = backend

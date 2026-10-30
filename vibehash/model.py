"""The useful work: an MLP trained with deterministic SGD.

Everything a verifier needs to replay a segment is a pure function of
(start weights, dataset, unit seed, step range, hyperparameters), so honest
replays match up to floating-point noise, whether they ran on a CPU or a GPU.
The maths here is the NumPy reference; compute.py runs the identical maths on
PyTorch devices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crypto import h256

PRECISIONS = {"float64": np.float64, "float32": np.float32}


@dataclass(frozen=True)
class ModelSpec:
    n_in: int = 8
    n_hidden: int | tuple[int, ...] = 32  # one width, or a tuple of widths for a deeper net
    n_out: int = 4
    precision: str = "float64"  # GPUs train in float32; many consumer GPUs have no float64 at all

    @property
    def hidden(self) -> tuple[int, ...]:
        return (self.n_hidden,) if isinstance(self.n_hidden, int) else tuple(self.n_hidden)

    @property
    def widths(self) -> tuple[int, ...]:
        return (self.n_in, *self.hidden, self.n_out)

    @property
    def dtype(self):
        return PRECISIONS[self.precision]

    @property
    def n_params(self) -> int:
        w = self.widths
        return sum(a * b + b for a, b in zip(w, w[1:]))

    def flops_per_step(self, batch_size: int) -> int:
        # Forward ~2 FLOPs/param/sample, backward ~4.
        return 6 * self.n_params * batch_size

    def round(self, theta: np.ndarray) -> np.ndarray:
        """Snap to the job's precision. Weights are always carried as float64 arrays, but a float32
        job's values must be exactly float32-representable so hashes agree on every device."""
        return np.asarray(theta, dtype=self.dtype).astype(np.float64)

    def init_params(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        parts = []
        for a, b in zip(self.widths, self.widths[1:]):
            parts += [rng.normal(0, 1 / np.sqrt(a), (a, b)).ravel(), np.zeros(b)]
        return self.round(np.concatenate(parts))

    def unpack(self, theta: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """[(W, b), ...] views into the flat vector, layer by layer: W1, b1, W2, b2, ..."""
        layers, i = [], 0
        for a, b in zip(self.widths, self.widths[1:]):
            w = theta[i:i + a * b].reshape(a, b)
            i += a * b
            layers.append((w, theta[i:i + b]))
            i += b
        return layers


@dataclass(frozen=True)
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray

    @classmethod
    def synthetic(cls, spec: ModelSpec, seed: int, n_train: int = 4096, n_val: int = 1024) -> "Dataset":
        """Gaussian clusters with a nonlinear twist, so the MLP has something to learn."""
        rng = np.random.default_rng(seed)
        centers = rng.normal(0, 2.0, (spec.n_out, spec.n_in))

        def make(n):
            y = rng.integers(0, spec.n_out, n)
            x = centers[y] + rng.normal(0, 1.4, (n, spec.n_in))
            x[:, 0] += np.sin(x[:, 1] * 1.5)
            return x, y

        x_train, y_train = make(n_train)
        x_val, y_val = make(n_val)
        return cls(x_train, y_train, x_val, y_val)

    def commitment(self) -> bytes:
        return h256(self.x_train.tobytes(), self.y_train.tobytes(), self.x_val.tobytes(), self.y_val.tobytes())

    @classmethod
    def from_npz(cls, path, max_bytes: int | None = None) -> "Dataset":
        """Loads x_train, y_train, x_val, y_val. Never unpickles: the file may come from anyone. Nor does it unpack
        anything before reading the arrays' headers: they must be arrays of numbers that take at most `max_bytes` once
        loaded (default: 64 times the file's size, or 64 MB if that's more; compressed uint8 images take ~35 times),
        since a few hundred KB of compressed zeros would otherwise unpack into gigabytes."""
        import math
        import os
        import zipfile

        limit = max(2**26, 64 * os.path.getsize(path)) if max_bytes is None else max_bytes
        need = 0
        try:
            with zipfile.ZipFile(path) as z:
                members = set(z.namelist())
                for name, target in (("x_train", np.float64), ("y_train", np.int64), ("x_val", np.float64), ("y_val", np.int64)):
                    with z.open(name if name in members else f"{name}.npy") as m:  # the member np.load would read
                        version = np.lib.format.read_magic(m)
                        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                        shape, _, dtype = read(m)
                    if dtype.kind not in "biuf":
                        raise ValueError(f"{name} isn't an array of numbers")
                    n = math.prod(shape)
                    need += n * 8 + (0 if dtype == target else n * dtype.itemsize)  # loaded, then converted
        except (zipfile.BadZipFile, KeyError) as e:
            raise ValueError(f"not a dataset (.npz with x_train, y_train, x_val, y_val): {e}") from e
        if need > limit:
            raise ValueError(f"the dataset would take {need / 2**20:,.0f} MB of memory once loaded, over the "
                             f"{limit / 2**20:,.0f} MB allowed for a file its size")
        with np.load(path, allow_pickle=False) as f:
            return cls(
                np.ascontiguousarray(f["x_train"], dtype=np.float64), np.ascontiguousarray(f["y_train"], dtype=np.int64),
                np.ascontiguousarray(f["x_val"], dtype=np.float64), np.ascontiguousarray(f["y_val"], dtype=np.int64),
            )

    def check_shape(self, spec: ModelSpec) -> None:
        for x, y in ((self.x_train, self.y_train), (self.x_val, self.y_val)):
            if x.ndim != 2 or x.shape[1] != spec.n_in or y.shape != (x.shape[0],) or x.shape[0] == 0:
                raise ValueError("dataset shape doesn't match the model")
            if y.min() < 0 or y.max() >= spec.n_out or not np.all(np.isfinite(x)):
                raise ValueError("dataset has invalid labels or values")


def _forward(spec: ModelSpec, theta: np.ndarray, x: np.ndarray):
    """Returns (activations per layer input, softmax probabilities). tanh hidden layers."""
    layers = spec.unpack(theta)
    acts = [x]
    for w, b in layers[:-1]:
        acts.append(np.tanh(acts[-1] @ w + b))
    w, b = layers[-1]
    logits = acts[-1] @ w + b
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return acts, probs


def evaluate(spec: ModelSpec, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Returns (cross-entropy loss, accuracy). Always float64 NumPy: it's for reporting, not replay."""
    _, probs = _forward(spec, theta, x)
    loss = float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())
    acc = float((probs.argmax(axis=1) == y).mean())
    return loss, acc


def _gradient(spec: ModelSpec, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    layers = spec.unpack(theta)
    acts, probs = _forward(spec, theta, x)
    d = probs.copy()
    d[np.arange(len(y)), y] -= 1
    d /= len(y)
    grads = []
    for i in range(len(layers) - 1, -1, -1):
        w, _ = layers[i]
        grads.append((acts[i].T @ d, d.sum(axis=0)))
        if i:
            d = (d @ w.T) * (1 - acts[i] ** 2)
    return np.concatenate([g.ravel() for gw, gb in reversed(grads) for g in (gw, gb)])


def batch_indices(unit_seed: int, step: int, n: int, batch_size: int) -> np.ndarray:
    """Batch for a given step depends only on (seed, step), so any segment can be replayed alone."""
    return np.random.default_rng([unit_seed, step]).integers(0, n, batch_size)


def train_segment_numpy(spec: ModelSpec, data: Dataset, theta: np.ndarray, unit_seed: int, start_step: int,
                        n_steps: int, lr: float, batch_size: int) -> np.ndarray:
    dt = spec.dtype
    theta = np.asarray(theta, dtype=dt).copy()
    lr = dt(lr)
    n = len(data.y_train)
    for step in range(start_step, start_step + n_steps):
        idx = batch_indices(unit_seed, step, n, batch_size)
        theta -= lr * _gradient(spec, theta, data.x_train[idx].astype(dt), data.y_train[idx])
    return theta.astype(np.float64)


def train_segment(spec: ModelSpec, data: Dataset, theta: np.ndarray, unit_seed: int, start_step: int,
                  n_steps: int, lr: float, batch_size: int, backend=None) -> np.ndarray:
    """Run SGD steps [start_step, start_step + n_steps) on `backend` (default: this process's active one)."""
    from .compute import active

    return (backend or active()).train_segment(spec, data, theta, unit_seed, start_step, n_steps, lr, batch_size)


def params_hash(theta: np.ndarray) -> bytes:
    return h256(np.ascontiguousarray(theta, dtype=np.float64).tobytes())


def relative_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

"""The worker's side of a DeepSeek-V4.1 top-layers job's training stage (DEEPSEEK.md §8): a unit trains LoRA-FA on the
top layers from the prep's records, as model.train_segment runs a fine-tune's, and a scorer measures the held-back
loss. model.train_segment and tasks.TrainingJob.evaluate come here for a deepseek_job.DeepSeekSpec.

A unit's state is finetune.py's: the vector [B | m | v] over spec.modules(), Adam's moments zeroed at each round's
start. Step t of a unit (from 0) trains on batch_indices(unit_seed, t) of the job's training sequences, each its own
micro-batch, with bias correction for t + 1, through deepseek_top.train_step_streamed in float32, so a checker on
other hardware replays it to float32 rounding.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import deepseek as ds
from . import deepseek_split as split
from . import deepseek_top as top
from .deepseek_job import DeepSeekData, DeepSeekSpec  # noqa: F401 - DeepSeekData: callers' import
from .model import batch_indices


class Records:
    """The prep's records, a chunk at a time: fetched from `store` once, checked against the pinned root
    (deepseek_split.read_block's checks), then kept in `folder` (the worker's temporary files, deleted with the job)
    and read a record at a time, memory-mapped."""

    def __init__(self, store, manifest: dict, root: str, folder: Path, cut: top.Cut):
        self.store, self.manifest, self.root, self.folder, self.cut = store, manifest, root, Path(folder), cut
        self.where = {}  # sequence -> (chunk, position in it)
        for c, b in enumerate(manifest["blocks"]):
            for j in range(b["count"]):
                self.where[b["first"] + j] = (c, j)
        self.lock = threading.Lock()

    def _chunk(self, c: int) -> tuple[np.ndarray, list[int]]:
        path, meta_path = self.folder / f"block-{c:05d}.bin", self.folder / f"block-{c:05d}.json"
        with self.lock:
            if not path.exists():  # fetched once, every record checked, then kept
                bin_name, meta_name = split.block_names(c)
                data, meta = self.store.get(bin_name), self.store.get(meta_name)
                split.check_block(self.manifest, c, data, meta, self.root)
                self.folder.mkdir(parents=True, exist_ok=True)
                meta_path.write_bytes(meta)
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.replace(path)
        return np.memmap(path, np.uint8, "r"), json.loads(meta_path.read_text())["offsets"]

    def record(self, seq: int) -> dict:
        c, j = self.where[seq]
        data, offsets = self._chunk(c)
        return top.unpack(bytes(data[offsets[j]:offsets[j + 1]]))

    def examples(self, seqs, device="cpu") -> list[top.Example]:
        """One micro-batch a sequence, on `device`."""
        return [top.example_of([self.record(int(s))], self.cut).to(device) for s in seqs]


@dataclass
class _Runtime:
    model: ds.Model
    cut: top.Cut
    device: str
    busy: threading.Lock = field(default_factory=threading.Lock)


_runtimes: dict = {}
_lock = threading.Lock()


def runtime(spec: DeepSeekSpec, device: str, weights=None) -> _Runtime:
    """The trained layers and the head on `device`, built once a process (deepseek_top.build_trained: the routed
    experts memory-mapped off the disk). `weights`: their tensors, for the first call; by default the tiny config's
    random weights (Flash's come from the worker's model files)."""
    key = (spec.config_name, spec.cut, device)
    with _lock:
        rt = _runtimes.get(key)
        if rt is None:
            if weights is None:
                if spec.config_name != "tiny":
                    raise RuntimeError("Flash's top layers must be fetched first (the worker's model files)")
                weights = tiny_weights(spec)
            rt = _runtimes[key] = _Runtime(top.build_trained(spec.top, weights, device), spec.top, device)
        return rt


def tiny_weights(spec: DeepSeekSpec) -> dict:
    """The tiny config's trained layers and head: random_weights(tiny_config(), 0)'s, as every tiny job's."""
    state = ds.random_weights(spec.cfg, 0)
    return {k: v for k, v in state.items() if k.startswith(("norm.", "head."))
            or (k.startswith("layers.") and int(k.split(".")[1]) >= spec.cut)}


def unload() -> None:
    with _lock:
        _runtimes.clear()


def _adapters(spec: DeepSeekSpec, theta: np.ndarray, device: str) -> top.Adapters:
    state = torch.tensor(np.asarray(theta, dtype=np.float32), device=device)
    pb, pm, pv = spec.parts
    names, a, b, m, v, at = [], [], [], [], [], 0
    for (name, out, _), aa in zip(spec.modules(), spec.lora_a()):
        n = out * spec.rank
        names.append(name)
        a.append(torch.from_numpy(aa).to(device))
        b.append(state[pb][at:at + n].view(out, spec.rank).clone().requires_grad_())
        m.append(state[pm][at:at + n].view(out, spec.rank).clone())
        v.append(state[pv][at:at + n].view(out, spec.rank).clone())
        at += n
    return top.Adapters(names, a, b, spec.rank, spec.alpha, m, v)


def _vector(ad: top.Adapters) -> np.ndarray:
    parts = [torch.cat([t.detach().reshape(-1) for t in ts]) for ts in (ad.b, ad.m, ad.v)]
    return torch.cat(parts).cpu().numpy().astype(np.float64)


def train_segment(spec: DeepSeekSpec, data: DeepSeekData, theta: np.ndarray, unit_seed: int, start_step: int,
                  n_steps: int, lr: float, batch_size: int, backend=None) -> np.ndarray:
    """Adam steps [start_step, start_step + n_steps) of a unit from the state vector `theta` ([B | m | v])."""
    from .finetune import device_of
    if data.records is None:
        raise RuntimeError("the job's prep isn't done: no records to train on yet")
    device = device_of(backend)
    rt = runtime(spec, device)
    ad = _adapters(spec, theta, device)
    train = data.train
    with rt.busy:
        ad.attach(rt.model)
        for t in range(start_step, start_step + n_steps):
            seqs = [train[i] for i in batch_indices(unit_seed, t, len(train), batch_size)]
            batches = data.records.examples(seqs, device)
            if device == "cpu":
                top.train_step(rt.model, rt.cut, ad, batches, t + 1, lr)
            else:
                top.train_step_streamed(rt.model, rt.cut, ad, batches, t + 1, lr, device=device)
    return _vector(ad)


@torch.no_grad()
def evaluate(spec: DeepSeekSpec, data: DeepSeekData, theta: np.ndarray, backend=None) -> tuple[float, float]:
    """(mean cross-entropy, share of learned tokens the model ranks first) over the held-back sequences, with the
    adapters in `theta`: the scorers' number, which the job's loss curve shows."""
    from .finetune import device_of
    device = device_of(backend)
    rt = runtime(spec, device)
    ad = _adapters(spec, theta, device)
    total = right = count = 0.0
    with rt.busy:
        ad.attach(rt.model)
        for ex in data.records.examples(data.heldback, device):
            x, pre = ex.x, ex.pre
            for name in (str(i) for i in rt.cut.trained):
                x, pre = rt.model.layers[name](x, pre, ex.side, ds.Choices(candidates=ex.candidates))
            logits = rt.model.logits(x, pre)[ex.mask]
            targets = ex.targets[ex.mask].to(logits.device)
            total += float(torch.nn.functional.cross_entropy(logits.float(), targets, reduction="sum"))
            right += float((logits.argmax(-1) == targets).sum())
            count += int(ex.mask.sum())
    return total / max(count, 1), right / max(count, 1)

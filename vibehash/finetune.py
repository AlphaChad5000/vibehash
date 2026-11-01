"""Fine-tuning open-weight language models: LoRA adapters trained in verified units (FINETUNING.md).

A fine-tune is a TrainingJob whose `spec` is a LoraSpec rather than model.py's ModelSpec. The protocol treats both
the same way: a flat vector of weights that units train from the round's start, checkpoints whose hashes go into a
Merkle tree, segments that checkers replay, and accepted results averaged into the next round's start. What differs
is behind `train_segment` and `evaluate`, which model.py hands to this module for a LoraSpec.

**What's trained: B, with A frozen.** An adapted projection computes W·x + (alpha/rank)·B·A·x. The network averages
units' results, and averaging A and B separately isn't averaging their product: mean(B_i·A_i) ≠ mean(B_i)·mean(A_i).
So A stays fixed (drawn from the job's seed, as PEFT initialises it) and only B is trained. Then the change to every
weight, (alpha/rank)·B·A, is linear in B, and averaging the units' B is exactly averaging their changes to the model.
This is LoRA-FA (Zhang et al., 2023) and, for federated averaging, FFA-LoRA (Sun et al., 2024); training B matters
far more than training A (Zhu et al., 2024, "Asymmetry in Low-Rank Adapters"). It also halves what's trained and
checked. DESIGN.md §8 has the alternatives and why they lost.

**The vector: [B | m | v].** Training uses Adam, whose moments m and v must be in every checkpoint so any segment can
be replayed alone. Each unit starts its moments at zero (they're zero in the round's start vector: `fresh` zeroes
them after averaging) with Adam's usual bias correction, so the steps within a unit count from 1. Checkers compare
each part separately (`parts`), so a worker can't pass off doctored moments under the much larger norm of B. Adam
also bounds how far a weight can move in a step, whatever the gradient; `max_step` makes that a sanity check.

**Data** is the poster's work: chats rendered with the model's own chat template and tokenized with its own tokenizer
(`prepare`, which needs `tokenizers` and `jinja2`), training only on the assistant's replies. Workers get token ids,
never text, so they need neither package. A TokenDataset is one .npz: the token ids, which of them to learn, and the
model's manifest (hub.py) and config.json, both pinned by the job, so a worker needs nothing else to fetch and check
the base weights.

Workers fetch the base model once from Hugging Face (hub.TensorCache, checked tensor by tensor against the manifest)
and keep it on their GPU between units; only the adapters (and moments) travel as checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model import batch_indices

TARGETS = ("q", "k", "v", "o", "gate", "up", "down")  # llm.TARGETS, repeated so the node needn't import PyTorch
FAMILIES = ("llama", "mistral", "qwen2", "qwen3", "smollm3")  # llm.FAMILIES, likewise
MODULE_OF = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
             "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
DEFAULT_TARGETS = ("q", "k", "v", "o")  # FINETUNING.md: the attention projections, unless flags say otherwise
OPTIMIZERS = {"adam": (0.9, 0.999, 1e-8)}  # name -> (beta1, beta2, eps); the name in a spec fixes all three
DIMS = ("family", "vocab", "hidden", "layers", "heads", "kv_heads", "head_dim", "ffn", "tied")
EVAL_BATCH = 8  # validation sequences scored at once
MAX_VAL = 128  # sequences held back at most: the node scores them every round
SEQ_STEP = 64  # sequence lengths are rounded up to a multiple of this
# How far a replay may be from a claim, relative to the claim's own size, for each of B, m and v (verification.
# reproduces adds: within 1% of what the segment changed). Measured CPU against CUDA on Qwen3-0.6B: DESIGN.md §8.
TOLERANCE = 1e-4
_BASE = re.compile(r"^hf:([A-Za-z0-9][\w.-]*(?:/[A-Za-z0-9][\w.-]*)?)@([0-9a-f]{40})$", re.ASCII)
_SHA = re.compile(r"^[0-9a-f]{64}$")


class DataError(ValueError):
    """The poster's data can't be made into training examples; the message says where and why."""


# --- the spec ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LoraSpec:
    """A fine-tune's model: the pinned base, the adapters' shape, and how they're trained. ModelSpec's interface
    (precision, dtype, n_params, round, init_params, flops_per_step), so job code can read either."""
    base: str  # hf:org/name@<40-hex commit>
    manifest: str  # sha256 of the base's hub.Manifest, which the job's dataset carries
    dims: tuple[tuple[str, object], ...]  # llm.Arch.dims() as sorted pairs: sizes the adapters, prices the work
    rank: int = 16
    alpha: float = 32.0
    targets: tuple[str, ...] = DEFAULT_TARGETS
    seq_len: int = 512
    optimizer: str = "adam"
    seed: int = 11  # the job's init_seed, which draws the frozen A matrices
    precision: str = "float32"

    kind = "lora"

    def __post_init__(self):
        if isinstance(self.dims, dict):
            object.__setattr__(self, "dims", tuple(sorted(self.dims.items())))
        if not isinstance(self.targets, str):  # TARGETS order, so equal specs compare (and serialize) equal
            known = tuple(t for t in TARGETS if t in set(self.targets))  # unknown ones stay, for problems() to refuse
            object.__setattr__(self, "targets", known + tuple(sorted(set(self.targets) - set(TARGETS), key=str)))

    @property
    def model(self) -> dict:
        return dict(self.dims)

    @property
    def repo(self) -> str:
        return _BASE.match(self.base).group(1)

    @property
    def commit(self) -> str:
        return _BASE.match(self.base).group(2)

    def problems(self) -> list[str]:
        """Why the spec can't be trained, or []."""
        m, out = self.model, []
        if not isinstance(self.base, str) or not _BASE.match(self.base):
            out.append("base must be hf:org/name@<40-hex commit>")
        if not isinstance(self.manifest, str) or not _SHA.match(self.manifest):
            out.append("manifest must be a sha256")
        if set(m) != set(DIMS) or m.get("family") not in FAMILIES or not isinstance(m.get("tied"), bool) \
                or not all(type(m.get(k)) is int and 1 <= m[k] <= 1 << 20 for k in DIMS[1:-1]):
            out.append(f"model must give {', '.join(DIMS)} (family one of {', '.join(FAMILIES)})")
        elif m["heads"] % m["kv_heads"] or m["head_dim"] % 2:
            out.append("query heads must be a multiple of key/value heads, and head_dim even")
        if type(self.rank) is not int or not 1 <= self.rank <= 256:
            out.append("LoRA rank must be 1-256")
        if not 0 < self.alpha <= 4096:
            out.append("LoRA alpha must be in (0, 4096]")
        if not self.targets or not set(self.targets) <= set(TARGETS):
            out.append(f"LoRA targets must be some of {', '.join(TARGETS)}")
        if type(self.seq_len) is not int or not 16 <= self.seq_len <= 32768:
            out.append("seq_len must be 16-32768")
        if self.optimizer not in OPTIMIZERS:
            out.append(f"optimizer must be one of {', '.join(OPTIMIZERS)}")
        if self.precision != "float32":
            out.append("fine-tunes train in float32")
        return out

    # --- ModelSpec's interface ---

    @property
    def dtype(self):
        return np.float32

    @property
    def scaling(self) -> float:
        """What B·A is multiplied by: lora_alpha / rank, as in PEFT."""
        return self.alpha / self.rank

    def shape(self, target: str) -> tuple[int, int]:
        """A projection's weight shape, [out, in] (llm.Arch.shape)."""
        m = self.model
        d, f, q, kv = m["hidden"], m["ffn"], m["heads"] * m["head_dim"], m["kv_heads"] * m["head_dim"]
        return {"q": (q, d), "k": (kv, d), "v": (kv, d), "o": (d, q), "gate": (f, d), "up": (f, d), "down": (d, f)}[target]

    def modules(self) -> list[tuple[str, int, int]]:
        """Every adapted projection, (Hugging Face module name, out, in), layer by layer, targets in TARGETS order:
        the order of the B blocks in the vector."""
        return [(f"model.layers.{i}.{MODULE_OF[t]}", *self.shape(t)) for i in range(self.model["layers"])
                for t in self.targets]

    @property
    def n_adapter(self) -> int:
        """Trained weights: every B, [out, rank]."""
        return sum(out for _, out, _ in self.modules()) * self.rank

    @property
    def n_params(self) -> int:
        """What a checkpoint holds: B, then Adam's first and second moments for it."""
        return 3 * self.n_adapter

    @property
    def parts(self) -> tuple[slice, slice, slice]:
        """B, m and v in the vector. Checks compare each on its own (verification.reproduces)."""
        p = self.n_adapter
        return slice(0, p), slice(p, 2 * p), slice(2 * p, 3 * p)

    def round(self, theta: np.ndarray) -> np.ndarray:
        return np.asarray(theta, dtype=np.float32).astype(np.float64)

    def init_params(self, seed: int = 0) -> np.ndarray:
        """Fresh adapters: B zero, so the job starts as exactly the base model, and zero moments. (A isn't in the
        vector: it's drawn from `self.seed`, `lora_a`.)"""
        return np.zeros(self.n_params)

    def fresh(self, theta: np.ndarray) -> np.ndarray:
        """`theta` with its moments zeroed: what the next round starts from once results are averaged. Each unit runs
        its own Adam from scratch, so a round's start carries B alone."""
        out = np.array(theta, dtype=np.float64)
        out[self.n_adapter:] = 0
        return out

    def flops_per_step(self, batch_size: int) -> int:
        """Nominal FLOPs a step: every sequence counted at the full seq_len. Per token: the frozen matmuls forward
        (2 a weight), again when each layer is recomputed (2), and back to the inputs (2); attention's scores and
        mixing likewise (q·kᵀ and p·v over the whole square); the head and loss forward and back (4 a weight); and the
        adapters (6 a weight of A and B)."""
        m = self.model
        layer = sum(math.prod(self.shape(t)) for t in TARGETS)
        attention = 4 * self.seq_len * m["heads"] * m["head_dim"]
        adapters = sum(out + inp for _, out, inp in self.modules()) * self.rank
        per_token = m["layers"] * (6 * layer + 3 * attention) + 4 * m["vocab"] * m["hidden"] + 6 * adapters
        return per_token * self.seq_len * batch_size

    def max_step(self, lr: float, steps: int) -> float:
        """The furthest any one weight of B can move in `steps` Adam steps from fresh moments, whatever the gradients:
        lr times the sum over steps t of max |m̂_t / √v̂_t|. By Cauchy–Schwarz that maximum is
        (1-β1)/√(1-β2) · √(Σ_{k<t} (β1²/β2)^k) · √(1-β2^t) / (1-β1^t), which is 1 at t = 1 and grows slowly
        (16.6 in all for 16 steps; Kingma & Ba's 3.16 a step is its limit). eps only makes steps shorter."""
        b1, b2, _ = OPTIMIZERS[self.optimizer]
        total, geometric = 0.0, 0.0
        for t in range(1, steps + 1):
            geometric += (b1 * b1 / b2) ** (t - 1)
            total += (1 - b1) / math.sqrt(1 - b2) * math.sqrt(geometric) * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
        return lr * total

    def sane(self, start: np.ndarray, final: np.ndarray, lr: float, steps: int, ratio: float) -> bool:
        """A unit's result could have come from training: finite, second moments not negative, and no weight of B
        moved further than Adam can take it (times `ratio`, the job's max_update_ratio), plus a rounding step."""
        b, _, v = self.parts
        if not np.all(np.isfinite(final)) or np.any(final[v] < 0):
            return False
        moved = np.max(np.abs(final[b] - start[b])) if self.n_adapter else 0.0
        limit = ratio * self.max_step(lr, steps)
        return moved <= limit * (1 + 1e-6) + np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(final[b]))))

    def lora_a(self) -> list[np.ndarray]:
        """The frozen A matrices, [rank, in] float32 each in `modules` order: uniform in ±1/√in as PEFT draws them
        (kaiming_uniform with a = √5), from the job's seed. Built from PCG64's raw 64-bit stream, which numpy keeps
        stable across versions (its distributions it doesn't promise), 53 bits a value."""
        rng = np.random.Generator(np.random.PCG64([self.seed, 0x10A]))
        out = []
        for _, _, inp in self.modules():
            raw = rng.bit_generator.random_raw(self.rank * inp)
            u = (raw >> np.uint64(11)).astype(np.float64) * 2.0 ** -53  # [0, 1)
            bound = 1 / math.sqrt(inp)
            out.append(((2 * u - 1) * bound).astype(np.float32).reshape(self.rank, inp))
        return out

    def peft_config(self) -> dict:
        """adapter_config.json, in PEFT's format: what transformers, PEFT and vLLM read."""
        alpha = int(self.alpha) if float(self.alpha).is_integer() else self.alpha
        return {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": self.repo,
                "revision": self.commit, "r": self.rank, "lora_alpha": alpha,
                "target_modules": [MODULE_OF[t].split(".")[1] for t in self.targets], "lora_dropout": 0.0,
                "bias": "none", "use_rslora": False, "fan_in_fan_out": False, "inference_mode": True,
                "init_lora_weights": True, "modules_to_save": None}

    def to_json(self) -> dict:
        """The spec fields that describe it (tasks.TrainingJob.to_spec adds the rest)."""
        alpha = int(self.alpha) if float(self.alpha).is_integer() else self.alpha
        return {"base": self.base, "manifest": self.manifest, "model": self.model,
                "lora": {"rank": self.rank, "alpha": alpha, "targets": list(self.targets)},
                "seq_len": self.seq_len, "optimizer": self.optimizer, "precision": self.precision}

    @classmethod
    def from_json(cls, d: dict, seed: int) -> "LoraSpec":
        lora, model = d["lora"], d["model"]
        if not isinstance(lora, dict) or not isinstance(model, dict) or not isinstance(lora["targets"], list):
            raise TypeError("lora and model must be objects, and lora's targets a list")
        return cls(base=str(d["base"]), manifest=str(d["manifest"]), dims=dict(model), rank=lora["rank"],
                   alpha=float(lora["alpha"]), targets=tuple(str(t) for t in lora["targets"]), seq_len=d["seq_len"],
                   optimizer=str(d["optimizer"]), seed=seed, precision=str(d.get("precision", "float32")))


# --- token data -------------------------------------------------------------------------------------------------------

_ARRAYS = (("train_ids", np.int32), ("train_len", np.int32), ("train_mask", np.uint8),
           ("val_ids", np.int32), ("val_len", np.int32), ("val_mask", np.uint8),
           ("manifest", np.uint8), ("config", np.uint8))


@dataclass(frozen=True)
class TokenDataset:
    """Sequences of token ids [n, length] (zero-padded on the right), how long each is, and which tokens to learn: a
    1 in `mask` at position t means token t is predicted from the ones before it and counts in the loss. Plus the
    base model's manifest and config.json, as bytes, which the job pins (`check`)."""
    train_ids: np.ndarray
    train_len: np.ndarray
    train_mask: np.ndarray
    val_ids: np.ndarray
    val_len: np.ndarray
    val_mask: np.ndarray
    manifest: bytes
    config: bytes
    _parsed: dict = field(default_factory=dict, repr=False, compare=False)

    kind = "tokens"

    @property
    def n_train(self) -> int:
        return len(self.train_len)

    def batch(self, idx, val: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """(ids, mask) of these sequences, cut to the longest of them: nothing after it is a token."""
        ids, lens, mask = (self.val_ids, self.val_len, self.val_mask) if val else (self.train_ids, self.train_len, self.train_mask)
        idx = np.asarray(idx)
        n = int(lens[idx].max())
        return np.ascontiguousarray(ids[idx, :n], dtype=np.int64), np.ascontiguousarray(mask[idx, :n])

    def targets(self, val: bool = False) -> int:
        return int((self.val_mask if val else self.train_mask).sum())

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: getattr(self, name) for name, _ in _ARRAYS[:6]}
        arrays["manifest"] = np.frombuffer(self.manifest, dtype=np.uint8)
        arrays["config"] = np.frombuffer(self.config, dtype=np.uint8)
        with open(path, "wb") as f:  # a file object, so numpy doesn't add .npz to the name
            np.savez_compressed(f, **arrays)
        return path

    @classmethod
    def from_npz(cls, path, max_bytes: int | None = None) -> "TokenDataset":
        """Loads one without unpickling anything, and refuses one whose arrays would take more than `max_bytes` once
        loaded (default 64 times the file's size, or 64 MB if that's more), as model.Dataset.from_npz does."""
        import os

        limit = max(2**26, 64 * os.path.getsize(path)) if max_bytes is None else max_bytes
        need = 0
        try:
            with zipfile.ZipFile(path) as z:
                members = set(z.namelist())
                for name, target in _ARRAYS:
                    with z.open(name if name in members else f"{name}.npy") as m:
                        version = np.lib.format.read_magic(m)
                        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                        shape, _, dtype = read(m)
                    if dtype != np.dtype(target):
                        raise ValueError(f"{name} must be {np.dtype(target).name}, not {dtype}")
                    need += math.prod(shape) * dtype.itemsize
        except (zipfile.BadZipFile, KeyError) as e:
            raise ValueError(f"not a token dataset: {e}") from e
        if need > limit:
            raise ValueError(f"the dataset would take {need / 2**20:,.0f} MB of memory once loaded, over the "
                             f"{limit / 2**20:,.0f} MB allowed for a file its size")
        with np.load(path, allow_pickle=False) as f:
            d = cls(*(np.ascontiguousarray(f[name]) for name, _ in _ARRAYS[:6]),
                    manifest=f["manifest"].tobytes(), config=f["config"].tobytes())
        d.check_shape()
        return d

    def check_shape(self) -> None:
        """Consistent on its own: lengths within the rows, masks only on real tokens and never at position 0, every
        sequence with something to learn, ids not negative, and a manifest and config that parse."""
        from .hub import Manifest

        for ids, lens, mask, what in ((self.train_ids, self.train_len, self.train_mask, "training"),
                                      (self.val_ids, self.val_len, self.val_mask, "validation")):
            if ids.ndim != 2 or mask.shape != ids.shape or lens.shape != (ids.shape[0],) or ids.shape[0] == 0:
                raise ValueError(f"the {what} sequences' arrays don't match")
            if lens.min() < 2 or lens.max() > ids.shape[1] or ids.min() < 0 or mask.max() > 1:
                raise ValueError(f"the {what} sequences have impossible lengths or ids")
            beyond = np.arange(ids.shape[1])[None, :] >= lens[:, None]
            if mask[:, 0].any() or (mask.astype(bool) & beyond).any() or (ids[beyond] != 0).any():
                raise ValueError(f"the {what} sequences mark tokens to learn where there are none")
            if (mask.sum(axis=1) == 0).any():
                raise ValueError(f"a {what} sequence has nothing to learn")
        try:
            self._parsed["manifest"] = Manifest.from_bytes(self.manifest)
            self._parsed["config"] = json.loads(self.config)
        except (ValueError, UnicodeDecodeError) as e:
            raise ValueError(f"the dataset's model manifest or config.json is malformed: {e}") from None

    def check(self, spec: LoraSpec) -> None:
        """The dataset suits this job: its manifest is the one the job pins, for the job's base model, with this
        config.json, of the shape the spec says; sequences fit the job's length; every id is in the vocabulary."""
        if "manifest" not in self._parsed:
            self.check_shape()
        manifest = self._parsed["manifest"]
        if hashlib.sha256(self.manifest).hexdigest() != spec.manifest:
            raise ValueError("the dataset's model manifest isn't the one the job pins")
        if (manifest.repo, manifest.commit) != (spec.repo, spec.commit):
            raise ValueError("the dataset's manifest is for another model")
        pinned = manifest.files.get("config.json")
        if pinned is None or hashlib.sha256(self.config).hexdigest() != pinned["sha256"]:
            raise ValueError("the dataset's config.json isn't the one the manifest pins")
        dims = arch_dims(self._parsed["config"])
        if dims != spec.model:
            raise ValueError(f"the job's model shape doesn't match its config.json ({dims})")
        if max(self.train_ids.shape[1], self.val_ids.shape[1]) > spec.seq_len:
            raise ValueError(f"the dataset's sequences are longer than the job's {spec.seq_len} tokens")
        if max(int(self.train_ids.max()), int(self.val_ids.max())) >= spec.model["vocab"]:
            raise ValueError("the dataset has token ids outside the model's vocabulary")

    def parsed_manifest(self):
        if "manifest" not in self._parsed:
            self.check_shape()
        return self._parsed["manifest"]

    def parsed_config(self) -> dict:
        if "config" not in self._parsed:
            self.check_shape()
        return self._parsed["config"]


def arch_dims(config: dict) -> dict:
    """llm.Arch.from_hf_config(config).dims(), without importing PyTorch (the node checks datasets without it)."""
    if not isinstance(config, dict) or config.get("model_type") not in FAMILIES:
        raise ValueError(f"model_type {config.get('model_type') if isinstance(config, dict) else None!r} isn't supported")
    heads = int(config["num_attention_heads"])
    return {"family": config["model_type"], "vocab": int(config["vocab_size"]), "hidden": int(config["hidden_size"]),
            "layers": int(config["num_hidden_layers"]), "heads": heads,
            "kv_heads": int(config.get("num_key_value_heads") or heads),
            "head_dim": int(config.get("head_dim") or int(config["hidden_size"]) // heads),
            "ffn": int(config["intermediate_size"]), "tied": bool(config.get("tie_word_embeddings", False))}


# --- running it -------------------------------------------------------------------------------------------------------

@dataclass
class _Runtime:
    model: object  # llm.Model
    device: str
    lora: dict = field(default_factory=dict)  # spec -> [(module name, A tensor, B slice, B shape)]


_runtimes: dict[tuple[str, str], _Runtime] = {}
_lock = threading.Lock()


def available() -> bool:
    """Whether this process can run fine-tunes: PyTorch is installed."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def device_of(backend) -> str:
    """The PyTorch device to train on: the backend's GPU, or the CPU (where float64 jobs and NumPy workers run)."""
    from .compute import TorchBackend, active

    backend = backend or active()
    return backend.device if isinstance(backend, TorchBackend) else "cpu"


def runtime(spec: LoraSpec, data: TokenDataset, device: str, progress=None) -> _Runtime:
    """The base model on `device`, fetched (once, then from the cache) and loaded on first use, shared by every job
    and thread of this process that uses the same model on the same device. `progress` is hub.Progress, for the
    download."""
    key = (spec.manifest, device)
    with _lock:
        rt = _runtimes.get(key)
        if rt is None:
            from . import compute, hub, llm

            compute.TorchBackend(device)  # TF32 off, deterministic kernels: the settings replays depend on
            manifest, arch = data.parsed_manifest(), llm.Arch.from_hf_config(data.parsed_config())
            arch.check(spec.seq_len)
            names = arch.tensor_names()
            paths = hub.fetch_tensors(manifest, names, progress=progress,
                                      label=f"fetching {spec.repo} ({hub._gb(manifest.bytes_for(names))})")

            def read(name):
                e = manifest.tensors[name]
                return e.dtype, list(e.shape), paths[name].read_bytes()  # the cache checked them against the manifest
            rt = _runtimes[key] = _Runtime(llm.load(arch, read, device), device)
        if spec not in rt.lora:
            import torch

            slots, i = [], 0
            for (name, out, _), a in zip(spec.modules(), spec.lora_a()):
                slots.append((name, torch.from_numpy(a).to(device), slice(i, i + out * spec.rank), (out, spec.rank)))
                i += out * spec.rank
            rt.lora[spec] = slots
        return rt


def unload() -> None:
    """Drop every loaded model (tests, and workers switching jobs)."""
    with _lock:
        _runtimes.clear()


def train_segment(spec: LoraSpec, data: TokenDataset, theta: np.ndarray, unit_seed: int, start_step: int,
                  n_steps: int, lr: float, batch_size: int, backend=None) -> np.ndarray:
    """Adam steps [start_step, start_step + n_steps) of a unit, from the state vector `theta` ([B | m | v]). Step t of
    a unit (from 0) uses batch_indices(unit_seed, t) and bias correction for t + 1. Every elementwise operation is a
    separate, correctly rounded float32 operation in a fixed order, so given the same gradients every device takes
    the same step; only the gradients themselves differ in their last bits."""
    import torch

    rt = runtime(spec, data, device_of(backend))
    b1, b2, eps = (float(np.float32(x)) for x in OPTIMIZERS[spec.optimizer])
    state = torch.tensor(np.asarray(theta, dtype=np.float32), device=rt.device)
    pb, pm, pv = spec.parts
    b, m, v = state[pb], state[pm], state[pv]
    slots = rt.lora[spec]
    for step in range(start_step, start_step + n_steps):
        ids, mask = data.batch(batch_indices(unit_seed, step, data.n_train, batch_size))
        leaves = [b[s].view(shape).clone().requires_grad_() for _, _, s, shape in slots]
        adapters = {name: (a, leaf) for (name, a, _, _), leaf in zip(slots, leaves)}
        total, count = rt.model.loss(torch.from_numpy(ids), torch.from_numpy(mask), adapters, spec.scaling)
        (total / count).backward()
        g = torch.cat([leaf.grad.reshape(-1) for leaf in leaves])
        t = step + 1
        c1, c2 = float(np.float32(1 - b1 ** t)), float(np.float32(1 - b2 ** t))
        m.mul_(b1).add_(g * float(np.float32(1 - b1)))
        v.mul_(b2).add_(g * g * float(np.float32(1 - b2)))
        b.sub_((m / c1) / ((v / c2).sqrt() + eps) * float(np.float32(lr)))
    return state.cpu().numpy().astype(np.float64)


def evaluate(spec: LoraSpec, data: TokenDataset, theta: np.ndarray, backend=None) -> tuple[float, float]:
    """(mean cross-entropy, share of tokens the model ranks first) over the held-back replies, with the adapters in
    `theta`. For people following the job, not part of any check."""
    import torch

    rt = runtime(spec, data, device_of(backend))
    b = torch.tensor(np.asarray(theta[spec.parts[0]], dtype=np.float32), device=rt.device)
    adapters = {name: (a, b[s].view(shape)) for name, a, s, shape in rt.lora[spec]}
    total = right = count = 0
    n = len(data.val_len)
    for s in range(0, n, EVAL_BATCH):
        ids, mask = data.batch(np.arange(s, min(n, s + EVAL_BATCH)), val=True)
        lo, r, c = rt.model.score(torch.from_numpy(ids), torch.from_numpy(mask), adapters, spec.scaling)
        total, right, count = total + lo, right + r, count + c
    return total / count, right / count


def memory_mb(spec: LoraSpec, batch_size: int, weight_bytes: int) -> int:
    """GPU memory a worker needs, roughly: the weights as stored, the float32 copy of the largest matrix made while
    it multiplies, each layer's input kept for the backward, one layer's working set (attention scores dominate),
    the head's logits for a chunk of tokens, the adapters' state four times over, and the CUDA context."""
    m = spec.model
    tokens = batch_size * spec.seq_len
    largest = max(m["vocab"] * m["hidden"], *(math.prod(spec.shape(t)) for t in TARGETS)) * 4
    layer_inputs = m["layers"] * tokens * m["hidden"] * 4
    working = 4 * batch_size * m["heads"] * spec.seq_len ** 2 * 4 + 6 * tokens * max(m["ffn"], m["hidden"]) * 4
    head = 2 * 512 * m["vocab"] * 4
    total = weight_bytes + largest + layer_inputs + working + head + 4 * spec.n_params * 4 + 700 * 2**20
    return int(math.ceil(total * 1.2 / 2**20 / 256) * 256)


# --- the result -------------------------------------------------------------------------------------------------------

def write_safetensors(path: Path, tensors: dict[str, np.ndarray], metadata: dict | None = None) -> None:
    """The safetensors format: an 8-byte little-endian header length, the JSON header (padded with spaces to a
    multiple of 8), then each tensor's bytes back to back."""
    names = {np.dtype(np.float32): "F32", np.dtype(np.float16): "F16"}
    header, offset, blobs = {}, 0, []
    for name, a in tensors.items():
        a = np.ascontiguousarray(a)
        data = a.astype(a.dtype.newbyteorder("<")).tobytes()
        header[name] = {"dtype": names[a.dtype], "shape": list(a.shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(len(raw).to_bytes(8, "little"))
        f.write(raw)
        for data in blobs:
            f.write(data)


def export_peft(spec: LoraSpec, theta: np.ndarray, out: str | Path, about: dict | None = None) -> Path:
    """Write the adapters as a standard PEFT LoRA adapter: adapter_model.safetensors (every A and B, float32, under
    PEFT's names) and adapter_config.json, plus a README naming the base model, its commit and where it came from.
    Loads with PeftModel.from_pretrained, and in vLLM with --enable-lora."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    b = np.asarray(theta[spec.parts[0]], dtype=np.float32)
    tensors, i = {}, 0
    for (name, rows, _), a in zip(spec.modules(), spec.lora_a()):
        tensors[f"base_model.model.{name}.lora_A.weight"] = a
        tensors[f"base_model.model.{name}.lora_B.weight"] = b[i:i + rows * spec.rank].reshape(rows, spec.rank)
        i += rows * spec.rank
    write_safetensors(out / "adapter_model.safetensors", tensors, {"format": "pt"})
    (out / "adapter_config.json").write_text(json.dumps(spec.peft_config(), indent=2) + "\n", encoding="utf-8", newline="\n")
    about = dict(about or {})
    lines = [f"# LoRA adapter for {spec.repo}", "",
             f"Fine-tuned on the Vibe Hash network (job `{about.pop('job_id', '?')}`).", "",
             f"- Base model: [{spec.repo}](https://huggingface.co/{spec.repo}) at commit `{spec.commit}`",
             f"- LoRA rank {spec.rank}, alpha {spec.peft_config()['lora_alpha']}, on "
             f"{', '.join(spec.peft_config()['target_modules'])}; A frozen at PEFT's initialisation, B trained"]
    lines += [f"- {k.replace('_', ' ').capitalize()}: {v}" for k, v in about.items() if v is not None]
    lines += ["", "```python", "from transformers import AutoModelForCausalLM, AutoTokenizer", "from peft import PeftModel", "",
              f'base = AutoModelForCausalLM.from_pretrained("{spec.repo}", revision="{spec.commit}")',
              f'model = PeftModel.from_pretrained(base, "{out.name}")', "```", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return out


# --- the poster's side: from chats to tokens -----------------------------------------------------------------------

def read_examples(path: str | Path) -> list[dict]:
    """Training examples from a file: JSON lines (or a JSON array) of {"messages": [{"role", "content"}, ...]} chats
    or {"text": ...} documents, or plain text (.txt, .md), one document per file."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in (".txt", ".md"):
        if not raw.strip():
            raise DataError(f"{path.name} is empty")
        return [{"text": raw}]
    stripped = raw.lstrip()
    if stripped.startswith("["):
        try:
            records = json.loads(raw)
        except ValueError as e:
            raise DataError(f"{path.name} isn't valid JSON: {e}") from None
        where = [f"item {i + 1}" for i in range(len(records))]
    else:
        records, where = [], []
        for n, line in enumerate(raw.splitlines(), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except ValueError as e:
                    raise DataError(f"line {n} of {path.name} isn't JSON: {e}") from None
                where.append(f"line {n}")
    if not records:
        raise DataError(f"{path.name} has no examples")
    for rec, at in zip(records, where):
        if not isinstance(rec, dict) or not ("messages" in rec or "text" in rec):
            raise DataError(f"{at.capitalize()} of {path.name} has no `text` or `messages` field")
        if "messages" in rec:
            msgs = rec["messages"]
            if not isinstance(msgs, list) or not msgs or not all(
                    isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str) for m in msgs):
                raise DataError(f"{at.capitalize()} of {path.name}: `messages` must be a list of "
                                '{"role": ..., "content": "text"} objects')
            if not any(m["role"] == "assistant" for m in msgs):
                raise DataError(f"{at.capitalize()} of {path.name} has no assistant message to learn from")
        elif not isinstance(rec["text"], str) or not rec["text"].strip():
            raise DataError(f"{at.capitalize()} of {path.name}: `text` must be some text")
        rec["_at"] = at
    return records


def _token_text(v) -> str | None:
    if isinstance(v, dict):
        v = v.get("content")
    return v if isinstance(v, str) else None


class ChatTokenizer:
    """A model's own tokenizer (tokenizer.json, through the `tokenizers` package) and chat template (Jinja, rendered
    as transformers renders it), from the files its manifest pins."""

    def __init__(self, tokenizer_json: bytes, tokenizer_config: dict, template: str | None):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise DataError("fine-tuning needs the `tokenizers` and `jinja2` packages: "
                            "pip install \"vibehash[finetune]\"") from None
        self.tok = Tokenizer.from_str(tokenizer_json.decode("utf-8"))
        self.config = tokenizer_config
        if template is None:
            template = tokenizer_config.get("chat_template")
            if isinstance(template, list):  # named templates: the default one
                template = next((t.get("template") for t in template if t.get("name") == "default"), None)
        self.template = template
        self.specials = sorted({t.content for t in self.tok.get_added_tokens_decoder().values() if t.special},
                               key=len, reverse=True)

    def render(self, messages: list[dict]) -> str:
        if not self.template:
            raise DataError("this model has no chat template, so it can't learn from chats; use {\"text\": ...} lines")
        try:
            import jinja2
            from jinja2.ext import loopcontrols
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError:
            raise DataError("fine-tuning needs the `tokenizers` and `jinja2` packages: "
                            "pip install \"vibehash[finetune]\"") from None
        from datetime import datetime

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        tokens = {k: _token_text(self.config.get(k)) for k in ("bos_token", "eos_token", "pad_token", "unk_token")}
        try:
            return env.from_string(self.template).render(messages=messages, tools=None, documents=None,
                                                         add_generation_prompt=False, **tokens)
        except jinja2.exceptions.TemplateError as e:
            raise DataError(f"the model's chat template refused a chat: {e}") from None

    def chat(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        """(ids, mask) of a chat: the template's rendering tokenized, with the assistant's replies and the end-of-turn
        token after each marked to learn. Each reply is found in the rendering by its text, in order, so it works with
        any template, whatever it wraps around them (Qwen3's empty think block, Llama's headers)."""
        text = self.render([{"role": m["role"], "content": m["content"]} for m in messages])
        spans, pos = [], 0
        for i, m in enumerate(messages):
            if m["role"] != "assistant":
                continue
            content = m["content"]
            at = text.find(content, pos) if content else -1
            if at < 0 and content.strip():
                content = content.strip()
                at = text.find(content, pos)
            if at < 0:
                raise DataError(f"message {i + 1} (assistant) isn't in the chat template's rendering of the chat")
            end = at + len(content)
            for s in self.specials:  # the end-of-turn token right after the reply: learning it teaches when to stop
                if text.startswith(s, end):
                    end += len(s)
                    break
            spans.append((at, end))
            pos = end
        enc = self.tok.encode(text, add_special_tokens=False)
        mask = [int(any(a <= start < b for a, b in spans)) for start, _ in enc.offsets]
        return enc.ids, mask

    def text(self, text: str) -> tuple[list[int], list[int]]:
        """(ids, mask) of a document: every token after the first is learned."""
        ids = self.tok.encode(text, add_special_tokens=True).ids
        return ids, [0] + [1] * (len(ids) - 1)


@dataclass
class Prepared:
    data: TokenDataset
    examples: int  # in the file
    kept: int  # with something to learn left after cutting to seq_len
    cut: int  # longer than seq_len, so cut short
    tokens: int  # in the kept sequences, after cutting
    targets: int  # tokens learned from
    seq_len: int
    chats: int  # examples that were chats (the rest were documents)


def prepare(path: str | Path, tokenizer: ChatTokenizer, manifest: bytes, config: bytes, max_len: int = 1024,
            seed: bytes = b"") -> Prepared:
    """Tokenize a data file for a job: hold back a tenth (at most MAX_VAL sequences) to measure progress, cut what's
    longer than `max_len`, and size the sequences to the longest left (a multiple of SEQ_STEP)."""
    records = read_examples(path)
    seqs, cut, chats = [], 0, 0
    for rec in records:
        if "messages" in rec:
            ids, mask = tokenizer.chat(rec["messages"])
            chats += 1
        else:
            ids, mask = tokenizer.text(rec["text"])
        if len(ids) > max_len:
            ids, mask, cut = ids[:max_len], mask[:max_len], cut + 1
        mask[0] = 0
        if len(ids) >= 2 and any(mask):
            seqs.append((ids, mask))
    if len(seqs) < 2:
        raise DataError(f"{Path(path).name} has {len(seqs)} example{'s' * (len(seqs) != 1)} with something to learn; "
                        "it takes at least 2 (one is held back to measure progress)")
    order = np.random.default_rng(int.from_bytes(hashlib.sha256(seed or Path(path).read_bytes()).digest()[:8], "big")
                                  ).permutation(len(seqs))
    n_val = max(1, min(len(seqs) // 10, MAX_VAL))
    longest = max(len(ids) for ids, _ in seqs)
    seq_len = min(max_len, -(-longest // SEQ_STEP) * SEQ_STEP)

    def pack(which):
        ids = np.zeros((len(which), seq_len), dtype=np.int32)
        mask = np.zeros((len(which), seq_len), dtype=np.uint8)
        lens = np.zeros(len(which), dtype=np.int32)
        for row, k in enumerate(which):
            s, mk = seqs[k]
            ids[row, :len(s)], mask[row, :len(s)], lens[row] = s, mk, len(s)
        return ids, lens, mask

    val, train = order[:n_val], order[n_val:]
    data = TokenDataset(*pack(train), *pack(val), manifest=manifest, config=config)
    data.check_shape()
    return Prepared(data, len(records), len(seqs), cut, sum(len(s) for s, _ in seqs), sum(sum(m) for _, m in seqs),
                    seq_len, chats)


def settings_for(dims: dict) -> dict:
    """Training settings by model size: a learning rate for Adam on B (A frozen) and a batch of sequences.
    Measured on Qwen3-0.6B (DESIGN.md §8); larger models take smaller steps."""
    size = dims["layers"] * dims["hidden"] ** 2
    lr = 5e-4 if size < 5e7 else 3e-4 if size < 2e8 else 2e-4
    return {"lr": lr, "batch_size": 8}


# --- the poster's side: the plan `train DATA --base MODEL` shows before posting ---------------------------------------

UNIT_STEPS, SEGMENT_STEPS = 16, 4  # Adam steps a unit, and between checkpoints: 4 segments, 3 of them challenged


def manifest_path(repo: str, commit: str) -> Path:
    from . import settings

    return settings.home() / "cache" / "manifests" / repo.replace("/", "--") / f"{commit}.json"


def model_manifest(repo: str, commit: str, hub, ask=None, progress=None) -> bytes:
    """The model's manifest (hub.build_manifest), from the cache in ~/.vibehash/cache/manifests or built by streaming
    its weights once. `ask(bytes)` is asked first, and building stops if it says no (None: don't ask)."""
    from . import hub as hubmod

    path = manifest_path(repo, commit)
    if path.exists():
        try:
            m = hubmod.Manifest.from_bytes(path.read_bytes())
            if (m.repo, m.commit) == (repo, commit):
                return m.to_bytes()
        except hubmod.IntegrityError:
            pass
    listing = hub.list_files(repo, commit)
    if hubmod.INDEX_FILE not in listing and hubmod.SINGLE_FILE not in listing:
        raise hub.no_safetensors(repo, commit, listing)
    total = sum(f.size for f in listing.values() if f.path.endswith(".safetensors"))
    if ask is not None and not ask(total):
        raise DataError("stopped: a model's weights are checked once before a job can pin them")
    m = hubmod.build_manifest(repo, commit, hub=hub, listing=listing, progress=progress)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(m.to_bytes())
    tmp.replace(path)
    return m.to_bytes()


@dataclass
class Plan:
    """A fine-tune worked out but not posted: the pinned model, the tokenized data, the job, and what it'll take."""
    job: object  # tasks.TrainingJob
    prepared: Prepared
    weight_bytes: int
    licence: str | None = None
    rounds_per_epoch: int = 0
    workers: int | None = None  # who took part in the network's recent rounds, if the node said

    @property
    def rounds(self) -> int:
        """Rounds the budget buys."""
        return self.job.budget // self.job.fee_per_round if self.job.fee_per_round else 0

    def describe(self, data_name: str) -> str:
        job, spec, p = self.job, self.job.spec, self.prepared
        m, pc = spec.model, spec.peft_config()

        def vibe(syn):
            return f"{syn / 1e8:,.8g} VIBE"
        epochs = f" ({self.rounds / self.rounds_per_epoch:.1f} epochs)" if self.rounds_per_epoch and self.rounds else ""
        seen = (f" · {self.workers} worker{'s' * (self.workers != 1)} took part in the last rounds"
                if self.workers is not None else "")
        return "\n".join([
            f"Model     {spec.repo} @ {spec.commit[:12]} (pinned)",
            f"          {m['family']}, {m['layers']} layers of width {m['hidden']} · {self.weight_bytes / 1e9:.1f} GB "
            f"of weights · licence {self.licence or 'not stated'}",
            f"Data      {data_name}: {p.examples:,} example{'s' * (p.examples != 1)}"
            + (f" ({p.chats:,} chats, in the model's chat template)" if p.chats else "")
            + f", {p.tokens:,} tokens, {p.targets:,} of them to learn",
            f"          {len(p.data.train_len):,} to train, {len(p.data.val_len):,} held back"
            + (f" · {p.cut} cut at {p.seq_len} tokens" if p.cut else f" · sequences up to {p.seq_len} tokens"),
            f"Training  LoRA rank {spec.rank} (alpha {pc['lora_alpha']}) on {', '.join(pc['target_modules'])}: "
            f"{spec.n_adapter:,} trained weights, A frozen (adapter {spec.n_adapter * 8 / 1e6:.1f} MB)",
            f"          Adam at {job.lr:g}, {job.batch_size} sequences a step, {job.steps_per_unit} steps a unit, "
            f"{job.units_per_round} units a round: one epoch is about {self.rounds_per_epoch:,} rounds",
            f"Workers   a GPU with at least {job.memory_mb / 1024:.1f} GB, and PyTorch; each fetches the weights from "
            f"Hugging Face once{seen}",
            f"Cost      {vibe(job.fee_per_round)} a round · a budget of {vibe(job.budget)} buys {self.rounds:,} "
            f"rounds{epochs}, paid only for work that passes its checks",
        ])


def plan(data_path: str | Path, ref: str, *, hub=None, job_id: str = "fine-tune", budget: int = 0, fee: int = 0,
         rank: int = 16, targets=DEFAULT_TARGETS, lr: float | None = None, max_len: int = 1024,
         units_per_round: int = 8, ask=None, progress=None, licence: str | None = None) -> Plan:
    """Work out a fine-tune of `ref` (org/name[@revision]) on a data file: pin the commit, check its architecture is
    one this code trains, get the model's manifest (cached, or built by streaming the weights once, if `ask` agrees),
    tokenize with the model's own tokenizer and chat template, and size the job. Its data source's hash is a
    placeholder until the dataset is saved (the caller does that)."""
    from . import hub as hubmod
    from .tasks import TrainingJob

    hub = hub or hubmod.Hub()
    repo, commit = hub.resolve(ref)
    config_bytes = hub.get_file(repo, commit, "config.json")
    config = json.loads(config_bytes)
    try:
        dims = arch_dims(config)
    except (ValueError, KeyError, TypeError) as e:
        raise DataError(f"{repo} can't be fine-tuned here yet: {e}. Supported: {', '.join(FAMILIES)} models") from None
    mbytes = model_manifest(repo, commit, hub, ask=ask, progress=progress)
    manifest = hubmod.Manifest.from_bytes(mbytes)
    if hashlib.sha256(config_bytes).hexdigest() != manifest.files.get("config.json", {}).get("sha256"):
        raise DataError("config.json isn't the one its manifest pins; try again")
    files = {name: hubmod.fetch_file(manifest, name, hub=hub) for name in
             ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja") if name in manifest.files}
    if "tokenizer.json" not in files:
        raise DataError(f"{repo} has no tokenizer.json, which turning text into training data needs")
    template = files["chat_template.jinja"].decode("utf-8") if "chat_template.jinja" in files else None
    tokenizer = ChatTokenizer(files["tokenizer.json"], json.loads(files.get("tokenizer_config.json", b"{}")), template)
    limit = min(max_len, int(config.get("max_position_embeddings") or max_len))
    prepared = prepare(data_path, tokenizer, mbytes, config_bytes, max_len=limit)
    defaults = settings_for(dims)
    spec = LoraSpec(base=f"hf:{repo}@{commit}", manifest=hashlib.sha256(mbytes).hexdigest(), dims=dims, rank=rank,
                    alpha=2.0 * rank, targets=tuple(targets), seq_len=prepared.seq_len, seed=11)
    weight_bytes = manifest.bytes_for(n for n in manifest.tensors if n in _model_tensors(dims, config))
    batch = defaults["batch_size"]
    job = TrainingJob(job_id, "", budget=budget, fee_per_round=fee, spec=spec,
                      data_source={"kind": "tokens", "sha256": "0" * 64}, init_seed=11, lr=lr or defaults["lr"],
                      batch_size=batch, steps_per_unit=UNIT_STEPS, checkpoint_every=SEGMENT_STEPS,
                      units_per_round=units_per_round, memory_mb=memory_mb(spec, batch, weight_bytes),
                      tolerance=TOLERANCE, max_update_ratio=1.0)
    per_round = units_per_round * UNIT_STEPS * batch
    return Plan(job, prepared, weight_bytes, licence, max(1, math.ceil(len(prepared.data.train_len) / per_round)))


def _model_tensors(dims: dict, config: dict) -> set[str]:
    """The tensors the model reads (llm.Arch.tensor_names), without PyTorch."""
    names = {"model.embed_tokens.weight", "model.norm.weight"} | (set() if dims["tied"] else {"lm_head.weight"})
    for i in range(dims["layers"]):
        p = f"model.layers.{i}."
        names |= {p + MODULE_OF[t] + ".weight" for t in TARGETS}
        names |= {p + "input_layernorm.weight", p + "post_attention_layernorm.weight"}
        if dims["family"] == "qwen2" or config.get("attention_bias"):
            names |= {p + MODULE_OF[t] + ".bias" for t in ("q", "k", "v")}
        if dims["family"] == "qwen3":
            names |= {p + "self_attn.q_norm.weight", p + "self_attn.k_norm.weight"}
    return names

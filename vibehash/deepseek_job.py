"""A DeepSeek-V4.1-Flash top-layers job on the network (DEEPSEEK.md §8): its spec and its two stages, without PyTorch
(the node reads it; the work itself runs in deepseek_split and deepseek_top).

1. The prep: the frozen layers 0..cut-1 over the job's text, as deepseek_split's units (pieces of `pieces` layers over
   chunks of `chunk_tokens`, and the engram tables' row ranges), handing off through the bucket. Its last pieces
   write the records, whose Merkle root the job pins.
2. Training: LoRA-FA on the top layers from the records, in rounds like any fine-tune's (averaged B, checks, juries):
   DeepSeekSpec is a finetune.LoraSpec, so the protocol prices, checks and averages its rounds as a fine-tune's, and
   only what a unit computes differs (model.train_segment dispatches on `kind`).

The spec is a fine-tune's with a "deepseek" field: {"cut", "pieces", "chunk_tokens", "row_ranges"}. A node from before
DeepSeek jobs takes it for a fine-tune and refuses the unknown field, so it can't misread one.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .deepseek_config import COMMIT, LORA_MODULES, REPO, Config, Cut, tiny_config
from .deepseek_graph import ROWS_PER_PIECE, Graph, pieces, plan
from .finetune import OPTIMIZERS, LoraSpec

KIND = "deepseek"
FAMILY = "deepseek_v41"
CONFIGS = {"flash": Config.flash, "tiny": tiny_config}  # "tiny": tests and local testnets, random weights
BASES = {"flash": f"hf:{REPO}@{COMMIT}", "tiny": "hf:vibehash/deepseek-v41-tiny@" + "0" * 40}
# sha256 over the pinned commit's weight files, "name lfs-sha256\n" each, sorted: what the job's weights are
# (weights_digest computes it from the Hub's listing).
MANIFESTS = {"flash": "5d069a8a464c59210d35973f5c52b0adaf630eb6a013af0b91267382e73c6f29",
             "tiny": hashlib.sha256(b"vibehash deepseek-v41 tiny: random_weights(tiny_config(), 0)").hexdigest()}
DEEPSEEK_KEYS = {"cut", "pieces", "chunk_tokens", "row_ranges"}
CUTS = {"flash": 36, "tiny": 4}  # the default cut: Flash's top 4 trained; tiny's top 1 (a lower cut would train a kv
# source above another trained layer, which Cut.check refuses)


@lru_cache(4)
def config(name: str) -> Config:
    return CONFIGS[name]()


def weights_digest(listing: dict) -> str:
    """MANIFESTS["flash"] from a Hub listing ({name: info with .lfs_sha256}) of the pinned commit."""
    rows = sorted((n, i.lfs_sha256) for n, i in listing.items() if n.endswith(".safetensors"))
    return hashlib.sha256("".join(f"{n} {s}\n" for n, s in rows).encode()).hexdigest()


@dataclass(frozen=True)
class DeepSeekSpec(LoraSpec):
    """A DeepSeek top-layers fine-tune's model: LoraSpec's fields (base, manifest, dims = {"family", "config"}, rank,
    alpha, targets, seq_len, optimizer, seed, precision) and the cut and the prep's shape."""
    cut: int = 36
    pieces: int = 4  # frozen layers a prep unit runs (DEEPSEEK.md §8: G)
    chunk_tokens: int = 32768
    row_ranges: int = 15

    kind = KIND

    def __post_init__(self):
        if isinstance(self.dims, dict):
            object.__setattr__(self, "dims", tuple(sorted(self.dims.items())))
        if not isinstance(self.targets, str):  # LORA_MODULES' order: the order of the B blocks
            known = tuple(t for t in LORA_MODULES if t in set(self.targets))
            object.__setattr__(self, "targets", known + tuple(sorted(set(self.targets) - set(LORA_MODULES), key=str)))

    @property
    def config_name(self) -> str:
        return self.model.get("config", "")

    @property
    def cfg(self) -> Config:
        return config(self.config_name)

    @property
    def top(self) -> Cut:
        return Cut(self.cfg, self.cut)

    @property
    def align(self) -> int:
        """Row ranges' alignment: deepseek_prep's 64 MiB pieces of Flash's tables, single rows of the tiny ones."""
        return 1 if self.config_name == "tiny" else ROWS_PER_PIECE

    def prep_seconds(self) -> float:
        """The compute phase a prep round needs: a unit's transfers and work at the slowest link the network takes
        (deepseek_graph.deadline), for a chunk's stream in and out (28 KB a token compressed) over `pieces` layers.
        0 for the tiny config, whose units take a moment (testnets use their own phase lengths)."""
        if self.config_name == "tiny":
            return 0.0
        from .deepseek_graph import deadline
        stream = 28_300 * self.chunk_tokens
        return deadline(stream, stream, self.pieces * self.chunk_tokens)

    def graph(self, tokens: "TokenFile") -> Graph:
        """The prep's work graph over the job's text: what the node hands out and every worker plans alike."""
        return plan(self.top, tokens.chunks(self.chunk_tokens), pieces(self.cut, self.pieces), self.row_ranges,
                    self.align)

    def problems(self) -> list[str]:
        m, out = self.model, []
        if set(m) != {"family", "config"} or m.get("family") != FAMILY or m.get("config") not in CONFIGS:
            return [f"model must be {{'family': '{FAMILY}', 'config': one of {', '.join(CONFIGS)}}}"]
        if self.base != BASES[m["config"]]:
            out.append(f"base must be {BASES[m['config']]}")
        if self.manifest != MANIFESTS[m["config"]]:
            out.append("manifest isn't the pinned weights'")
        cfg = self.cfg
        if type(self.cut) is not int or not 0 < self.cut < cfg.n_layers:
            out.append(f"cut must leave layers on both sides of it (1-{cfg.n_layers - 1})")
        else:
            try:
                self.top.check()
            except NotImplementedError as e:
                out.append(str(e))
        if type(self.rank) is not int or not 1 <= self.rank <= 256:
            out.append("LoRA rank must be 1-256")
        if not 0 < self.alpha <= 4096:
            out.append("LoRA alpha must be in (0, 4096]")
        if not self.targets or not set(self.targets) <= set(LORA_MODULES):
            out.append(f"LoRA targets must be some of {', '.join(LORA_MODULES)}")
        if type(self.seq_len) is not int or not 16 <= self.seq_len <= 65536:
            out.append("seq_len must be 16-65536")
        if self.optimizer not in OPTIMIZERS:
            out.append(f"optimizer must be one of {', '.join(OPTIMIZERS)}")
        if self.precision != "float32":
            out.append("DeepSeek's top layers train in float32")
        if type(self.pieces) is not int or not 1 <= self.pieces <= max(1, self.cut):
            out.append("pieces must be 1 to the frozen layers")
        if type(self.chunk_tokens) is not int or not 1024 <= self.chunk_tokens <= 2**20:
            out.append("chunk_tokens must be 1024-1048576")
        if type(self.row_ranges) is not int or not 1 <= self.row_ranges <= 64:
            out.append("row_ranges must be 1-64")
        return out

    # --- LoraSpec's interface, for DeepSeek's projections ---

    def shape(self, target: str) -> tuple[int, int]:
        """A projection's [out, in]."""
        c = self.cfg
        return {"wq_a": (c.q_lora_rank, c.dim), "wq_b": (c.n_heads * c.head_dim, c.q_lora_rank),
                "wkv": (c.head_dim, c.dim), "wo_b": (c.dim, c.o_groups * c.o_lora_rank),
                "w1": (c.moe_inter_dim, c.dim), "w3": (c.moe_inter_dim, c.dim), "w2": (c.dim, c.moe_inter_dim)}[target]

    def modules(self) -> list[tuple[str, int, int]]:
        """Every adapted projection (its checkpoint name, out, in), trained layer by layer, targets in LORA_MODULES'
        order: the B blocks' order in the vector, and deepseek_top.Adapters' order."""
        return [(f"layers.{i}.{LORA_MODULES[t]}", *self.shape(t)) for i in self.top.trained for t in self.targets]

    def active_weights(self) -> int:
        """Weights a token goes through in one trained layer: attention, and the shared expert and the routed ones it
        picks."""
        c = self.cfg
        attn = sum(o * i for o, i in (self.shape(t) for t in ("wq_a", "wq_b", "wkv", "wo_b")))
        attn += c.n_heads * c.head_dim * c.o_lora_rank  # wo_a's groups
        return attn + (c.n_activated_experts + 1) * 3 * c.dim * c.moe_inter_dim

    def flops_per_step(self, batch_size: int) -> int:
        """Nominal FLOPs a step, every sequence at the full seq_len: the trained layers forward, recomputed and back
        to their inputs (6 a weight), the head and loss (4 a weight), and the adapters (6 a weight)."""
        c = self.cfg
        adapters = sum(out + inp for _, out, inp in self.modules()) * self.rank
        per_token = len(self.top.trained) * 6 * self.active_weights() + 4 * c.vocab_size * c.dim + 6 * adapters
        return per_token * self.seq_len * batch_size

    def peft_config(self) -> dict:
        """adapter_config.json in PEFT's format, as deepseek_top.Adapters.peft_config writes it."""
        alpha = int(self.alpha) if float(self.alpha).is_integer() else self.alpha
        return {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": REPO, "revision": COMMIT,
                "r": self.rank, "lora_alpha": alpha, "target_modules": sorted(self.targets),
                "layers_to_transform": list(self.top.trained), "layers_pattern": "layers", "lora_dropout": 0.0,
                "bias": "none", "use_rslora": False, "fan_in_fan_out": False, "inference_mode": True,
                "init_lora_weights": True, "modules_to_save": None}

    def to_json(self) -> dict:
        out = super().to_json()
        out["deepseek"] = {"cut": self.cut, "pieces": self.pieces, "chunk_tokens": self.chunk_tokens,
                           "row_ranges": self.row_ranges}
        return out

    @classmethod
    def from_json(cls, d: dict, seed: int) -> "DeepSeekSpec":
        base = LoraSpec.from_json(d, seed)
        x = d["deepseek"]
        if not isinstance(x, dict) or set(x) != DEEPSEEK_KEYS:
            raise TypeError(f"deepseek must be an object of {', '.join(sorted(DEEPSEEK_KEYS))}")
        return cls(base=base.base, manifest=base.manifest, dims=dict(base.dims), rank=base.rank, alpha=base.alpha,
                   targets=tuple(d["lora"]["targets"]), seq_len=base.seq_len, optimizer=base.optimizer, seed=seed,
                   precision=base.precision, cut=x["cut"], pieces=x["pieces"], chunk_tokens=x["chunk_tokens"],
                   row_ranges=x["row_ranges"])


def is_deepseek_spec(d) -> bool:
    """A parsed spec is a DeepSeek job's: a fine-tune's shape (lora, a model without n_in) with a "deepseek" field."""
    return isinstance(d, dict) and "deepseek" in d and "lora" in d and isinstance(d.get("model"), dict) \
        and "n_in" not in d["model"]


def spec(name: str = "flash", cut: int | None = None, rank: int = 16, alpha: float = 32.0, targets=None,
         seq_len: int = 512, seed: int = 11, pieces: int = 4, chunk_tokens: int = 32768,
         row_ranges: int = 15) -> DeepSeekSpec:
    """A DeepSeekSpec for a known config (the cut: CUTS)."""
    return DeepSeekSpec(base=BASES[name], manifest=MANIFESTS[name], dims={"family": FAMILY, "config": name},
                        rank=rank, alpha=alpha, targets=tuple(targets or ("wq_a", "wq_b", "wkv", "wo_b")),
                        seq_len=seq_len, seed=seed, cut=CUTS[name] if cut is None else cut, pieces=pieces,
                        chunk_tokens=chunk_tokens, row_ranges=row_ranges)


@dataclass
class TokenFile:
    """A DeepSeek job's text, as deepseek_prep reads it (an .npz of tokens, lengths and learn), without PyTorch: what
    the node checks and hosts as the job's dataset, like a fine-tune's, and what the prep's units run over. The
    records the prep makes from it live in the bucket, not here."""
    tokens: np.ndarray  # int32, every sequence end to end
    lengths: np.ndarray  # int32, one per sequence
    learn: np.ndarray  # uint8, 1 where a token is a target to learn

    @classmethod
    def from_npz(cls, path, max_bytes: int | None = None) -> "TokenFile":
        import os
        if max_bytes is not None and os.path.getsize(path) > max_bytes:
            raise ValueError(f"the token file is over {max_bytes} bytes")
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {"tokens", "lengths", "learn"}:
                raise ValueError("a DeepSeek job's data is an .npz of tokens, lengths and learn")
            out = cls(z["tokens"].astype(np.int64), z["lengths"].astype(np.int64), z["learn"].astype(np.uint8))
        if out.tokens.ndim != 1 or out.lengths.ndim != 1 or len(out.learn) != len(out.tokens)                 or int(out.lengths.sum()) != len(out.tokens) or (out.lengths < 2).any():
            raise ValueError("tokens, lengths and learn don't agree (or a sequence is shorter than 2)")
        return out

    def check(self, spec: DeepSeekSpec, data: dict) -> None:
        """That it's text the job can train on: token ids in the vocabulary, sequences no longer than seq_len, and the
        job's train and held-back spans inside it."""
        if len(self.tokens) and (self.tokens.min() < 0 or self.tokens.max() >= spec.cfg.vocab_size):
            raise ValueError(f"token ids must be below the vocabulary's {spec.cfg.vocab_size}")
        if self.lengths.max(initial=0) > spec.seq_len:
            raise ValueError(f"a sequence is {int(self.lengths.max())} tokens, over the job's seq_len {spec.seq_len}")
        n = len(self.lengths)
        if data["train"][1] > n or data["heldback"][1] > n:
            raise ValueError(f"the job's spans go past the file's {n} sequences")

    def chunks(self, chunk_tokens: int) -> list[list[int]]:
        """The prep's chunks, deepseek_prep.blocks_of's: whole sequences in order, at most chunk_tokens each (a longer
        sequence alone)."""
        out, cur, n = [], [], 0
        for i, length in enumerate(self.lengths.tolist()):
            if cur and n + length > chunk_tokens:
                out.append(cur)
                cur, n = [], 0
            cur.append(i)
            n += length
        return out + ([cur] if cur else [])


@dataclass
class DeepSeekData:
    """What a DeepSeek job's units read: its token file (the node hosts it, as a fine-tune's dataset) and, once its
    prep is done, the records in the bucket (`records`, set by the worker from the job's pinned root)."""
    tokens: TokenFile
    source: dict  # the spec's data: {"kind", "sha256", "train", "heldback"}
    records: object = None  # deepseek_unit.Records

    @property
    def train(self) -> range:
        return range(*self.source["train"])

    @property
    def heldback(self) -> range:
        return range(*self.source["heldback"])

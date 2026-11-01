"""A Llama-style transformer with LoRA adapters, trained with deterministic SGD: the reference for language-model jobs.

This is to language models what model.py is to the MLP. Every function is a pure function of its inputs, the
arithmetic runs at the job's precision, and a segment of training depends only on (weights, dataset, unit seed, step
range, hyperparameters), so any segment can be replayed alone. NumPy only.

The model. Tokens are embedded, then each of `n_layers` blocks does
    h = h + o_proj(attention(rope(q_proj(x)), rope(k_proj(x)), v_proj(x)))    where x = rmsnorm(h)
    h = h + down_proj(silu(gate_proj(x)) * up_proj(x))                       where x = rmsnorm(h)
with causal attention, rotary position embeddings (RoPE) and grouped-query attention, as in Llama 2 and 3 and
Mistral. Then a final RMSNorm and the LM head. The loss is next-token cross-entropy: the mean over every position of
every sequence in the batch.

Two flat vectors. A job's weights are two vectors, carried as float64 like model.py's, with values exactly
representable at the job's precision:
- `base`: the frozen pretrained weights, in `layout(spec)` order. Names and shapes are Hugging Face's Llama ones
  (`model.layers.3.self_attn.q_proj.weight`, [out, in]), so a checkpoint loads into it by name (`pack`).
- `trainable`: what training changes, checkpoints hold and the network averages. With LoRA (`lora_rank` > 0) it holds
  the adapters, in `adapter_layout(spec)` order: an A [rank, in] and a B [out, rank] for each targeted projection of
  each layer, under PEFT's names. A projection's weight is then W + (lora_alpha / lora_rank)·B·A. B starts at zero,
  so fresh adapters give exactly the base model. With rank 0, the trainable vector is the base itself, for small full
  fine-tunes, and `base` is unused.
Both are laid out layer by layer: the embedding, layer 0, layer 1, ..., the final norm and the head. So a block of
layers is one contiguous slice of each, which is what lets a team of workers split a model into stages (SHARDING.md).

Token data: train_segment and check_data take any object with
- `tokens_train`: an integer array [n_train, seq_len + 1];
- `tokens_val`: likewise, [n_val, seq_len + 1].
Each row is one sequence: the model reads row[:-1] and predicts row[1:]. Rows are fetched by index, so memory-mapped
arrays work and only a batch's rows are read. `TokenData` is the simplest such object; `toy_data` makes tasks for
tests.

Stages. `split` cuts the model into stages of whole layers. The first also embeds the tokens; the last also applies
the final norm, the head and the loss. `stage_forward` and `stage_backward` run one stage on its own slices of the two
vectors. Only the residual stream, [batch, seq, d_model], travels between stages: forward, and its gradient back. The
whole-model functions are the one-stage case of the same code, so a unit trained through any split ends at exactly the
bits `train_segment` gives (tests/test_transformer.py).

Matching it on other hardware. A GPU backend must run these operations in this order, at the job's precision, with
every constant (norm_eps, 1/sqrt(head_dim), lora_alpha/lora_rank, the learning rate) cast to it first:
- RMSNorm: (h · 1/sqrt(mean(h²) + eps)) · weight;
- a projection: x·Wᵀ, plus ((x·Aᵀ)·Bᵀ)·(alpha/rank) when it's adapted, as PEFT computes it;
- RoPE in Hugging Face's rotate-half layout, from cos and sin tables computed in float64 and then rounded (Hugging
  Face computes the angles in float32, which drifts at long positions);
- attention: (q·kᵀ)·scale, the causal mask as -inf, then a softmax as exp(s - max) / sum;
- SiLU through a sigmoid that never overflows (`_sigmoid`);
- the loss's gradient and the SGD step exactly as model.py.
It will still match only within a job's tolerance, not bit for bit (DESIGN §8).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .model import PRECISIONS, batch_indices

TARGETS = ("q", "k", "v", "o", "gate", "up", "down")  # the projections LoRA can adapt, in their order within a layer
MODULES = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
           "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
EMBED, NORM, HEAD = "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"

# Trained in full, a tied embedding is updated by both ends of the model: by the lookup in the first stage and as the
# head in the last. Splitting that across workers would need its gradient sent from one end to the other, summed in the
# whole model's order. Models big enough to need stages are trained with LoRA, which leaves the embedding frozen.
_TIED = "a tied embedding trained in full (lora_rank 0) can't be split into stages: both ends would update it"


@dataclass(frozen=True)
class TransformerSpec:
    vocab: int = 64
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int | None = None  # grouped-query attention: each key/value head serves n_heads / n_kv_heads query heads
    d_ff: int = 64
    seq_len: int = 16  # tokens a sequence feeds the model; data rows hold one more, the last position's target
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = False  # the head reuses the embedding matrix, as small Llamas do
    precision: str = "float64"  # as in ModelSpec: GPUs train in float32
    lora_rank: int = 0  # 0: train every weight
    lora_alpha: float = 16.0
    lora_targets: tuple[str, ...] = TARGETS
    head_dim: int | None = None  # None: d_model / n_heads. Some models set it apart

    def __post_init__(self):
        # Frozen, so defaults are filled in with object.__setattr__. Targets go into TARGETS order, so two specs that
        # train the same thing compare (and hash) equal, and the adapter layout never depends on how they were listed.
        if self.n_kv_heads is None:
            object.__setattr__(self, "n_kv_heads", self.n_heads)
        if self.head_dim is None:
            if self.n_heads < 1 or self.d_model % self.n_heads:
                raise ValueError("d_model must be a multiple of n_heads (or give head_dim)")
            object.__setattr__(self, "head_dim", self.d_model // self.n_heads)
        if isinstance(self.lora_targets, str) or not set(self.lora_targets) <= set(TARGETS):
            raise ValueError(f"lora_targets must be a subset of {TARGETS}")
        object.__setattr__(self, "lora_targets", tuple(t for t in TARGETS if t in set(self.lora_targets)))
        problems = [
            (self.vocab >= 2, "vocab must be at least 2"),
            (min(self.d_model, self.n_layers, self.d_ff, self.seq_len) >= 1, "sizes must be positive"),
            (1 <= self.n_kv_heads <= self.n_heads and self.n_heads % self.n_kv_heads == 0,
             "n_heads must be a multiple of n_kv_heads"),
            (self.head_dim >= 2 and self.head_dim % 2 == 0, "head_dim must be even: RoPE turns dimensions in pairs"),
            (self.rope_theta > 0 and self.norm_eps > 0, "rope_theta and norm_eps must be positive"),
            (self.precision in PRECISIONS, "precision must be 'float64' or 'float32'"),
            (self.lora_rank >= 0, "lora_rank can't be negative"),
            (not self.lora_rank or (self.lora_targets and self.lora_alpha > 0),
             "LoRA needs at least one target and a positive lora_alpha"),
        ]
        for ok, why in problems:
            if not ok:
                raise ValueError(why)

    @classmethod
    def from_hf_config(cls, config: dict, seq_len: int, **settings) -> "TransformerSpec":
        """The spec of a Hugging Face Llama or Mistral checkpoint, from its config.json (as a dict). `seq_len` and the
        training settings (precision, lora_rank, lora_alpha, lora_targets) are the job's, not the checkpoint's. Refuses
        what this reference doesn't compute rather than quietly train a different model under the checkpoint's name."""
        unsupported = [why for bad, why in [
            (config.get("model_type", "llama") not in ("llama", "mistral"), f"model_type {config.get('model_type')!r}"),
            (config.get("hidden_act", "silu") != "silu", f"activation {config.get('hidden_act')!r}"),
            (bool(config.get("rope_scaling")), "rope_scaling"),
            (bool(config.get("attention_bias") or config.get("mlp_bias")), "biases"),
            ((config.get("sliding_window") or seq_len) < seq_len, "a sliding window shorter than seq_len"),
        ] if bad]
        if unsupported:
            raise ValueError(f"not supported yet: {', '.join(unsupported)}")
        return cls(vocab=int(config["vocab_size"]), d_model=int(config["hidden_size"]),
                   n_layers=int(config["num_hidden_layers"]), n_heads=int(config["num_attention_heads"]),
                   n_kv_heads=int(config.get("num_key_value_heads") or config["num_attention_heads"]),
                   d_ff=int(config["intermediate_size"]), seq_len=seq_len,
                   rope_theta=float(config.get("rope_theta", 10000.0)), norm_eps=float(config.get("rms_norm_eps", 1e-6)),
                   tie_embeddings=bool(config.get("tie_word_embeddings", False)),
                   head_dim=int(config["head_dim"]) if config.get("head_dim") else None, **settings)

    @property
    def dtype(self):
        return PRECISIONS[self.precision]

    @property
    def groups(self) -> int:
        """Query heads per key/value head."""
        return self.n_heads // self.n_kv_heads

    @property
    def scaling(self) -> float:
        """What B·A is multiplied by: lora_alpha / lora_rank, as in PEFT."""
        return self.lora_alpha / self.lora_rank if self.lora_rank else 0.0

    def shape(self, target: str) -> tuple[int, int]:
        """A projection's weight shape as Hugging Face stores it, [out, in]."""
        d, f, q, kv = self.d_model, self.d_ff, self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
        return {"q": (q, d), "k": (kv, d), "v": (kv, d), "o": (d, q), "gate": (f, d), "up": (f, d), "down": (d, f)}[target]

    @property
    def n_base(self) -> int:
        return _size(_base_entries(self, _whole(self)))

    @property
    def n_trainable(self) -> int:
        return _size(_trainable_entries(self, _whole(self)))

    @property
    def n_params(self) -> int:
        """What a checkpoint holds: the trainable weights. ModelSpec's name for it, so job code can read either spec."""
        return self.n_trainable

    def flops_per_step(self, batch_size: int) -> int:
        """An estimate, per SGD step: about 2 FLOPs per projection weight per token forward, and 4 more backward (only
        2 with LoRA, where the frozen weights need no gradient of their own), plus attention's score and mixing
        products, whose backward is twice its forward, plus the adapters' small matmuls."""
        matmul = self.n_layers * sum(math.prod(self.shape(t)) for t in TARGETS) + self.vocab * self.d_model
        attention = self.n_layers * 4 * self.seq_len * self.n_heads * self.head_dim  # q·kᵀ and p·v, per token
        per_token = (6 * matmul if not self.lora_rank else 4 * matmul + 6 * self.n_trainable) + 3 * attention
        return per_token * self.seq_len * batch_size

    def round(self, theta: np.ndarray) -> np.ndarray:
        """Snap to the job's precision, carried as float64 (ModelSpec.round): so hashes agree on every device."""
        return np.asarray(theta, dtype=self.dtype).astype(np.float64)


@dataclass(frozen=True)
class TokenData:
    """The simplest token dataset: two integer arrays of sequences, [n, seq_len + 1] each. Anything with these two
    attributes will do (the module's docstring)."""
    tokens_train: np.ndarray
    tokens_val: np.ndarray


@dataclass(frozen=True)
class Stage:
    """Layers [lo, hi) of an n_layers model, held by one worker of a team. The same fields as shards.Shard."""
    index: int
    lo: int  # its first layer
    hi: int  # one past its last
    n_layers: int  # in the whole model

    @property
    def first(self) -> bool:
        return self.lo == 0

    @property
    def last(self) -> bool:
        return self.hi == self.n_layers


def _whole(spec: TransformerSpec) -> Stage:
    return Stage(0, 0, spec.n_layers, spec.n_layers)


# --- layouts: what each flat vector holds, by name ------------------------------------------------------------------

def _size(entries) -> int:
    return sum(math.prod(shape) for _, shape in entries)


def _layer(spec: TransformerSpec, i: int) -> list:
    """One block's weights, in the order Hugging Face's Llama registers them."""
    p = f"model.layers.{i}."
    return [(f"{p}{MODULES[t]}.weight", spec.shape(t)) for t in TARGETS] + [
        (p + "input_layernorm.weight", (spec.d_model,)), (p + "post_attention_layernorm.weight", (spec.d_model,))]


def _adapters(spec: TransformerSpec, i: int) -> list:
    p, r, out = f"model.layers.{i}.", spec.lora_rank, []
    for t in spec.lora_targets:
        n_out, n_in = spec.shape(t)
        out += [(f"{p}{MODULES[t]}.lora_A.weight", (r, n_in)), (f"{p}{MODULES[t]}.lora_B.weight", (n_out, r))]
    return out


@lru_cache(maxsize=256)
def _base_entries(spec: TransformerSpec, stage: Stage) -> tuple:
    """The base weights a stage holds, in the order its own base vector lays them out (stage_base)."""
    d, v = spec.d_model, spec.vocab
    out = [(EMBED, (v, d))] if stage.first else []
    for i in range(stage.lo, stage.hi):
        out += _layer(spec, i)
    if stage.last:
        out.append((NORM, (d,)))
        if not spec.tie_embeddings:
            out.append((HEAD, (v, d)))
        elif not stage.first:
            out.append((EMBED, (v, d)))  # a tied head is the embedding, so the last stage holds a copy of it
    return tuple(out)


@lru_cache(maxsize=256)
def _trainable_entries(spec: TransformerSpec, stage: Stage) -> tuple:
    if not spec.lora_rank:
        if spec.tie_embeddings and not (stage.first and stage.last):
            raise ValueError(_TIED)
        return _base_entries(spec, stage)
    return tuple(e for i in range(stage.lo, stage.hi) for e in _adapters(spec, i))


def layout(spec: TransformerSpec) -> list[tuple[str, tuple]]:
    """The base vector's (name, shape) entries: the embedding, each layer's q, k, v, o, gate, up and down projections
    and its two norms, the final norm, and the head unless it's tied to the embedding. Hugging Face's Llama names."""
    return list(_base_entries(spec, _whole(spec)))


def adapter_layout(spec: TransformerSpec) -> list[tuple[str, tuple]]:
    """The adapters' (name, shape) entries, layer by layer, each target's A then B, targets in TARGETS order. PEFT's
    names without its `base_model.model.` prefix. Empty with rank 0."""
    return list(_trainable_entries(spec, _whole(spec))) if spec.lora_rank else []


def peft_names(spec: TransformerSpec) -> tuple[dict[str, str], dict]:
    """How the adapters export to PEFT, which Hugging Face transformers and vLLM load: ({our name: PEFT's key}, in
    adapter_layout order, and the adapter_config.json fields). The layout is already PEFT's (A [rank, in], B [out,
    rank], W + (lora_alpha / rank)·B·A, not rsLoRA), so an export is the tensors under these keys in an
    adapter_model.safetensors, beside the config; the writer adds `base_model_name_or_path`."""
    if not spec.lora_rank:
        raise ValueError("with lora_rank 0 there are no adapters: export the trained model itself")
    keys = {name: "base_model.model." + name for name, _ in adapter_layout(spec)}
    alpha = int(spec.lora_alpha) if float(spec.lora_alpha).is_integer() else spec.lora_alpha
    config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": spec.lora_rank, "lora_alpha": alpha,
              "target_modules": [MODULES[t].split(".")[1] for t in spec.lora_targets], "lora_dropout": 0.0,
              "bias": "none", "use_rslora": False, "fan_in_fan_out": False}
    return keys, config


def trainable_layout(spec: TransformerSpec) -> list[tuple[str, tuple]]:
    """What the trainable vector holds: the adapters, or with rank 0 the whole base."""
    return list(_trainable_entries(spec, _whole(spec)))


def unpack(entries, flat: np.ndarray) -> dict[str, np.ndarray]:
    """{name: view} into a flat vector laid out as `entries`, (name, shape) pairs such as layout(spec)."""
    if flat.ndim != 1:
        raise ValueError("weights must be a flat vector")
    out, i = {}, 0
    for name, shape in entries:
        n = math.prod(shape)
        out[name] = flat[i:i + n].reshape(shape)
        i += n
    if i != len(flat):
        raise ValueError(f"expected {i:,} weights, got {len(flat):,}")
    return out


def pack(entries, tensors, dtype=np.float64) -> np.ndarray:
    """The flat vector of named tensors in `entries` order: how a loader turns a checkpoint's tensors into `base`, or
    into one stage's share of it with stage_layout. Shapes must match exactly."""
    parts = []
    for name, shape in entries:
        a = np.asarray(tensors[name], dtype=dtype)
        if a.shape != tuple(shape):
            raise ValueError(f"{name} has shape {a.shape}, expected {tuple(shape)}")
        parts.append(a.ravel())
    return np.concatenate(parts) if parts else np.zeros(0, dtype)


# --- starting weights -----------------------------------------------------------------------------------------------

def init_base(spec: TransformerSpec, seed: int) -> np.ndarray:
    """Random weights for tests and toy jobs; real jobs start from a pretrained checkpoint. Every matrix, the embedding
    included, is N(0, 1/columns), so activations and logits start near unit scale; norms are 1. Like
    ModelSpec.init_params, it depends on numpy's random stream."""
    rng = np.random.default_rng(seed)
    parts = [np.ones(shape) if len(shape) == 1 else rng.normal(0, 1 / math.sqrt(shape[1]), shape).ravel()
             for _, shape in layout(spec)]
    return spec.round(np.concatenate(parts))


def init_adapters(spec: TransformerSpec, seed: int) -> np.ndarray:
    """Fresh adapters, initialised as PEFT does: A uniform in ±1/sqrt(in), B zero. With B zero, the model starts as
    exactly the base."""
    if not spec.lora_rank:
        raise ValueError("with lora_rank 0 there are no adapters: the base itself is trained")
    rng = np.random.default_rng(seed)
    parts = [rng.uniform(-1 / math.sqrt(shape[1]), 1 / math.sqrt(shape[1]), shape).ravel()
             if name.endswith("lora_A.weight") else np.zeros(math.prod(shape)) for name, shape in adapter_layout(spec)]
    return spec.round(np.concatenate(parts))


def init_trainable(spec: TransformerSpec, base: np.ndarray, seed: int) -> np.ndarray:
    """A job's starting trainable vector: fresh adapters, or with rank 0 the base itself."""
    return init_adapters(spec, seed) if spec.lora_rank else spec.round(base)


def merge(spec: TransformerSpec, base: np.ndarray, adapters: np.ndarray) -> np.ndarray:
    """The base with the adapters folded in, W + (alpha/rank)·B·A for every adapted projection, computed in float64 and
    rounded to the precision: a plain model again, for `dataclasses.replace(spec, lora_rank=0)`."""
    out = np.array(base, dtype=np.float64)
    w = unpack(layout(spec), out)
    a = unpack(adapter_layout(spec), np.asarray(adapters, dtype=np.float64))
    for i in range(spec.n_layers):
        for t in spec.lora_targets:
            name = f"model.layers.{i}.{MODULES[t]}"
            w[name + ".weight"] += spec.scaling * (a[name + ".lora_B.weight"] @ a[name + ".lora_A.weight"])
    return spec.round(out)


# --- data -----------------------------------------------------------------------------------------------------------

def toy_data(spec: TransformerSpec, seed: int, n_train: int = 256, n_val: int = 64, task: str = "count") -> TokenData:
    """Sequences a small model learns in a few hundred steps, for tests and demos:
    - "count": each row counts up from a random token, wrapping at the end of the vocabulary;
    - "copy": random tokens, then the same tokens again, so the second half needs attention to look back."""
    rng = np.random.default_rng(seed)
    width = spec.seq_len + 1

    def make(n):
        if task == "count":
            return (rng.integers(0, spec.vocab, (n, 1)) + np.arange(width)) % spec.vocab
        if task == "copy":
            half = (width + 1) // 2
            first = rng.integers(0, spec.vocab, (n, half))
            return np.concatenate([first, first[:, :width - half]], axis=1)
        raise ValueError(f"unknown task {task!r}")

    return TokenData(make(n_train).astype(np.int32), make(n_val).astype(np.int32))


def check_data(spec: TransformerSpec, data) -> None:
    """Raises ValueError unless both token arrays are non-empty integer arrays of rows seq_len + 1 long, every token
    in the vocabulary. Reads the whole dataset; training checks each batch it reads anyway."""
    for name in ("tokens_train", "tokens_val"):
        a = getattr(data, name, None)
        if not isinstance(a, np.ndarray) or a.ndim != 2 or a.shape[1] != spec.seq_len + 1 or not len(a):
            raise ValueError(f"{name} must be an array of rows {spec.seq_len + 1} tokens long")
        if a.dtype.kind not in "iu" or a.min() < 0 or a.max() >= spec.vocab:
            raise ValueError(f"{name} has tokens outside the vocabulary")


def _tokens(spec: TransformerSpec, tokens) -> np.ndarray:
    """Checked token ids [batch, seq]. A negative id would otherwise index the embedding from its end, silently."""
    tokens = np.asarray(tokens)
    if tokens.ndim != 2 or tokens.dtype.kind not in "iu" or 0 in tokens.shape:
        raise ValueError("tokens must be a non-empty integer array [batch, seq]")
    if tokens.min() < 0 or tokens.max() >= spec.vocab:
        raise ValueError("tokens outside the vocabulary")
    return tokens


# --- the pieces of a block ------------------------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _rope(spec: TransformerSpec, t: int) -> tuple[np.ndarray, np.ndarray]:
    """cos and sin [t, head_dim] for positions 0..t-1 in Hugging Face's rotate-half layout: dimensions i and
    i + head_dim/2 form a pair, turned by position · rope_theta^(-2i/head_dim). Angles in float64, then rounded to the
    precision, so every device starts from the same tables."""
    d = spec.head_dim
    inv_freq = 1.0 / spec.rope_theta ** (np.arange(0, d, 2, dtype=np.float64) / d)
    angles = np.arange(t, dtype=np.float64)[:, None] * inv_freq
    angles = np.concatenate([angles, angles], axis=1)
    cos, sin = np.cos(angles).astype(spec.dtype), np.sin(angles).astype(spec.dtype)
    cos.flags.writeable = sin.flags.writeable = False  # cached and shared
    return cos, sin


@lru_cache(maxsize=64)
def _causal(t: int) -> np.ndarray:
    mask = np.tril(np.ones((t, t), dtype=bool))
    mask.flags.writeable = False
    return mask


def _rotate(x: np.ndarray) -> np.ndarray:
    """Hugging Face's rotate_half: (x1, x2) -> (-x2, x1), so x·cos + rotate(x)·sin turns each pair (x1[i], x2[i])."""
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _unrotate(y: np.ndarray) -> np.ndarray:
    """The transpose of _rotate, for the backward pass: (y1, y2) -> (y2, -y1)."""
    half = y.shape[-1] // 2
    return np.concatenate([y[..., half:], -y[..., :half]], axis=-1)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """1 / (1 + e^-x), written so the exponential never overflows."""
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e))


def _rmsnorm(h: np.ndarray, weight: np.ndarray, eps):
    """Returns (the output, h normalised, 1/rms): what the backward needs."""
    r = 1 / np.sqrt((h * h).mean(axis=-1, keepdims=True) + eps)
    n = h * r
    return n * weight, n, r


def _rmsnorm_back(dx: np.ndarray, weight: np.ndarray, n: np.ndarray, r: np.ndarray, grads: dict, name: str,
                  full: bool) -> np.ndarray:
    """The gradient at the norm's input. With n = h/rms(h), dh = (dn - n·mean(dn·n)) / rms(h)."""
    if full:
        grads[name] = (dx * n).reshape(-1, n.shape[-1]).sum(axis=0)
    dn = dx * weight
    return r * (dn - n * (dn * n).mean(axis=-1, keepdims=True))


def _linear(x: np.ndarray, w: dict, ad: dict, name: str, s):
    """x [tokens, in] through a projection: x·Wᵀ, plus ((x·Aᵀ)·Bᵀ)·s if it's adapted. Returns (y, x·Aᵀ or None)."""
    y = x @ w[name + ".weight"].T
    if name + ".lora_A.weight" not in ad:
        return y, None
    u = x @ ad[name + ".lora_A.weight"].T
    y += (u @ ad[name + ".lora_B.weight"].T) * s
    return y, u


def _linear_back(x: np.ndarray, dy: np.ndarray, u, w: dict, ad: dict, name: str, s, grads: dict, full: bool):
    """The gradient at the projection's input, recording its weight's gradient (full fine-tuning) or its adapters'.
    LoRA never forms the full weight gradient: the adapters' gradients go through the rank-r product x·Aᵀ."""
    dx = dy @ w[name + ".weight"]
    if full:
        grads[name + ".weight"] = dy.T @ x
    if u is not None:
        dz = dy * s
        grads[name + ".lora_B.weight"] = dz.T @ u
        du = dz @ ad[name + ".lora_B.weight"]
        grads[name + ".lora_A.weight"] = du.T @ x
        dx += du @ ad[name + ".lora_A.weight"]
    return dx


def _heads(a: np.ndarray, b: int, t: int, hk: int, n: int, d: int) -> np.ndarray:
    """[b·t, hk·n·d] -> [b, hk, n, t, d]: query head j is hk-group j // n, as Hugging Face's repeat_kv pairs them."""
    return np.ascontiguousarray(a.reshape(b, t, hk, n, d).transpose(0, 2, 3, 1, 4))


def _merge_heads(a: np.ndarray) -> np.ndarray:
    b, hk, n, t, d = a.shape
    return a.transpose(0, 3, 1, 2, 4).reshape(b * t, hk * n * d)


def _attention(spec: TransformerSpec, q, k, v, b: int, t: int):
    """Causal attention with RoPE. Queries are grouped [batch, kv head, group, seq, head_dim] and keys and values
    [batch, kv head, 1, seq, head_dim], so each key/value head serves its group by broadcasting, never copied."""
    hk, g, d, dt = spec.n_kv_heads, spec.groups, spec.head_dim, spec.dtype
    cos, sin = _rope(spec, t)
    q, k, v = _heads(q, b, t, hk, g, d), _heads(k, b, t, hk, 1, d), _heads(v, b, t, hk, 1, d)
    q = q * cos + _rotate(q) * sin
    k = k * cos + _rotate(k) * sin
    scores = (q @ k.swapaxes(-1, -2)) * dt(1 / math.sqrt(d))
    scores = np.where(_causal(t), scores, -np.inf)
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    return _merge_heads(p @ v), (q, k, v, p)


def _attention_back(spec: TransformerSpec, cache, do: np.ndarray, b: int, t: int):
    """Gradients at q, k and v (before RoPE), each [b·t, heads·head_dim]. A key/value head's gradient sums over its
    group. Masked scores have probability exactly 0, so they pass no gradient."""
    q, k, v, p = cache
    hk, g, d, dt = spec.n_kv_heads, spec.groups, spec.head_dim, spec.dtype
    cos, sin = _rope(spec, t)
    do = _heads(do, b, t, hk, g, d)
    dp = do @ v.swapaxes(-1, -2)
    dv = (p.swapaxes(-1, -2) @ do).sum(axis=2, keepdims=True)
    ds = p * (dp - (dp * p).sum(axis=-1, keepdims=True))
    ds *= dt(1 / math.sqrt(d))
    dq = ds @ k
    dk = (ds.swapaxes(-1, -2) @ q).sum(axis=2, keepdims=True)
    dq = dq * cos + _unrotate(dq * sin)
    dk = dk * cos + _unrotate(dk * sin)
    return _merge_heads(dq), _merge_heads(dk), _merge_heads(dv)


def _block(spec: TransformerSpec, w: dict, ad: dict, i: int, h: np.ndarray):
    """One decoder layer. Returns (its output, what its backward needs)."""
    p, (b, t, d), dt = f"model.layers.{i}.", h.shape, spec.dtype
    s, eps = dt(spec.scaling), dt(spec.norm_eps)
    c = {}
    x, c["n1"], c["r1"] = _rmsnorm(h, w[p + "input_layernorm.weight"], eps)
    x = c["x1"] = x.reshape(b * t, d)
    q, c["uq"] = _linear(x, w, ad, p + MODULES["q"], s)
    k, c["uk"] = _linear(x, w, ad, p + MODULES["k"], s)
    v, c["uv"] = _linear(x, w, ad, p + MODULES["v"], s)
    c["o"], c["attn"] = _attention(spec, q, k, v, b, t)
    a, c["uo"] = _linear(c["o"], w, ad, p + MODULES["o"], s)
    h = h + a.reshape(b, t, d)
    x, c["n2"], c["r2"] = _rmsnorm(h, w[p + "post_attention_layernorm.weight"], eps)
    x = c["x2"] = x.reshape(b * t, d)
    c["gate"], c["ug"] = _linear(x, w, ad, p + MODULES["gate"], s)
    c["up"], c["uu"] = _linear(x, w, ad, p + MODULES["up"], s)
    c["sg"] = _sigmoid(c["gate"])
    c["act"] = c["gate"] * c["sg"]  # SiLU
    c["m"] = c["act"] * c["up"]
    f, c["ud"] = _linear(c["m"], w, ad, p + MODULES["down"], s)
    return h + f.reshape(b, t, d), c


def _block_back(spec: TransformerSpec, w: dict, ad: dict, i: int, c: dict, dh: np.ndarray, grads: dict, full: bool):
    """The gradient at a layer's input, given the one at its output; records its weights' gradients in `grads`."""
    p, (b, t, d), dt = f"model.layers.{i}.", dh.shape, spec.dtype
    s = dt(spec.scaling)
    dm = _linear_back(c["m"], dh.reshape(b * t, d), c["ud"], w, ad, p + MODULES["down"], s, grads, full)
    sg = c["sg"]
    dgate = (dm * c["up"]) * (sg * (1 + c["gate"] * (1 - sg)))  # SiLU'(x) = σ(x)·(1 + x·(1 - σ(x)))
    dup = dm * c["act"]
    dx = _linear_back(c["x2"], dgate, c["ug"], w, ad, p + MODULES["gate"], s, grads, full)
    dx += _linear_back(c["x2"], dup, c["uu"], w, ad, p + MODULES["up"], s, grads, full)
    name = p + "post_attention_layernorm.weight"
    dh = dh + _rmsnorm_back(dx.reshape(b, t, d), w[name], c["n2"], c["r2"], grads, name, full)
    do = _linear_back(c["o"], dh.reshape(b * t, d), c["uo"], w, ad, p + MODULES["o"], s, grads, full)
    dq, dk, dv = _attention_back(spec, c["attn"], do, b, t)
    dx = _linear_back(c["x1"], dq, c["uq"], w, ad, p + MODULES["q"], s, grads, full)
    dx += _linear_back(c["x1"], dk, c["uk"], w, ad, p + MODULES["k"], s, grads, full)
    dx += _linear_back(c["x1"], dv, c["uv"], w, ad, p + MODULES["v"], s, grads, full)
    name = p + "input_layernorm.weight"
    return dh + _rmsnorm_back(dx.reshape(b, t, d), w[name], c["n1"], c["r1"], grads, name, full)


def _xent(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Each position's cross-entropy, from logits [positions, vocab], as log-sum-exp minus the target's logit."""
    z = logits - logits.max(axis=1, keepdims=True)
    return np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(z)), targets]


def _xent_back(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """The mean cross-entropy's gradient at the logits: softmax minus one-hot, over the number of positions, in the
    same operations as model._forward and model._gradient."""
    d = logits - logits.max(axis=1, keepdims=True)
    np.exp(d, out=d)
    d /= d.sum(axis=1, keepdims=True)
    d[np.arange(len(d)), targets] -= 1
    d /= len(d)
    return d


# --- stages ---------------------------------------------------------------------------------------------------------

def split(spec: TransformerSpec, n: int) -> list[Stage]:
    """`n` stages of whole layers holding about the same number of base weights each (the first also holds the
    embedding, the last the head), so each fits about the same memory. shards.split's rule."""
    sizes = [_size(_layer(spec, i)) for i in range(spec.n_layers)]
    if not 1 <= n <= len(sizes):
        raise ValueError(f"a {len(sizes)}-layer model splits into 1 to {len(sizes)} stages")
    if n > 1 and not spec.lora_rank and spec.tie_embeddings:
        raise ValueError(_TIED)
    sizes[0] += spec.vocab * spec.d_model
    sizes[-1] += spec.d_model + spec.vocab * spec.d_model  # the final norm, and the head or the tied embedding's copy
    cuts, run, total = [0], 0, sum(sizes)
    for i, size in enumerate(sizes[:-1]):
        run += size
        left = n - len(cuts)  # stages still to start
        if left and (run >= total * len(cuts) / n or len(sizes) - (i + 1) == left):
            cuts.append(i + 1)
    cuts.append(len(sizes))
    return [Stage(k, lo, hi, len(sizes)) for k, (lo, hi) in enumerate(zip(cuts, cuts[1:]))]


def stage_layout(spec: TransformerSpec, stage: Stage) -> list[tuple[str, tuple]]:
    """The base weights a stage holds, (name, shape) in its base vector's order: what a worker loads from a checkpoint
    by name (`pack`) without ever holding the rest. A tied model's last stage holds a copy of the embedding as its
    head, at the end."""
    return list(_base_entries(spec, stage))


def stage_trainable_layout(spec: TransformerSpec, stage: Stage) -> list[tuple[str, tuple]]:
    """The trainable weights a stage holds, in order: its layers' adapters, or with rank 0 its base weights."""
    return list(_trainable_entries(spec, stage))


def _start(spec: TransformerSpec, stage: Stage, entries) -> int:
    """Where a stage's weights start in a whole vector: after everything the layers before it hold."""
    return _size(entries(spec, Stage(0, 0, stage.lo, stage.n_layers))) if stage.lo else 0


def trainable_span(spec: TransformerSpec, stage: Stage) -> slice:
    """Where a stage's trainable weights sit in the whole trainable vector: one contiguous slice."""
    start = _start(spec, stage, _trainable_entries)
    return slice(start, start + _size(_trainable_entries(spec, stage)))


def stage_base(spec: TransformerSpec, stage: Stage, base: np.ndarray) -> np.ndarray:
    """A stage's share of the base, laid out as stage_layout: a contiguous slice of the whole (a view), except for a
    tied model's last stage, which also needs the embedding and gets a copy with it appended."""
    start, size = _start(spec, stage, _base_entries), _size(_base_entries(spec, stage))
    if stage.last and spec.tie_embeddings and not stage.first:
        embed = spec.vocab * spec.d_model
        return np.concatenate([base[start:start + size - embed], base[:embed]])
    return base[start:start + size]


def assemble(spec: TransformerSpec, stages: list[Stage], parts: list[np.ndarray]) -> np.ndarray:
    """The whole trainable vector from its stages' slices."""
    out = np.empty(spec.n_trainable)
    for stage, part in zip(stages, parts):
        out[trainable_span(spec, stage)] = part
    return out


def _weights(spec: TransformerSpec, stage: Stage, base, trainable) -> tuple[dict, dict]:
    """The stage's weights and adapters as {name: view} at the spec's dtype. Pass the vectors at that dtype already to
    skip a conversion (train_segment does)."""
    if stage.n_layers != spec.n_layers or not 0 <= stage.lo < stage.hi <= stage.n_layers:
        raise ValueError("the stage doesn't fit the model")
    if trainable is None:
        raise ValueError("no trainable weights (with lora_rank 0 they are the base itself)")
    dt = spec.dtype
    if not spec.lora_rank:
        return unpack(_base_entries(spec, stage), np.asarray(trainable, dtype=dt)), {}
    if base is None:
        raise ValueError("LoRA needs the base weights")
    return (unpack(_base_entries(spec, stage), np.asarray(base, dtype=dt)),
            unpack(_trainable_entries(spec, stage), np.asarray(trainable, dtype=dt)))


def stage_forward(spec: TransformerSpec, stage: Stage, base, trainable, inputs):
    """One stage's share of the forward pass. `base` and `trainable` are the stage's own slices (stage_base,
    trainable_span), `inputs` the token ids [batch, seq] for the first stage or the residual stream [batch, seq,
    d_model] the previous stage sent. Returns (cache, out): what stage_backward needs, and the residual stream to send
    on (C-contiguous, at the spec's dtype), or the logits [batch, seq, vocab] from the last stage."""
    w, ad = _weights(spec, stage, base, trainable)
    dt = spec.dtype
    cache = {"blocks": []}
    if stage.first:
        cache["tokens"] = _tokens(spec, inputs)
        h = w[EMBED][cache["tokens"]]
    else:
        h = np.ascontiguousarray(inputs, dtype=dt)  # whatever buffer it arrived in, computed on as a fresh layout
        if h.ndim != 3 or h.shape[2] != spec.d_model or 0 in h.shape:
            raise ValueError("the residual stream must be [batch, seq, d_model]")
    for i in range(stage.lo, stage.hi):
        h, c = _block(spec, w, ad, i, h)
        cache["blocks"].append(c)
    if not stage.last:
        return cache, np.ascontiguousarray(h)
    b, t, d = h.shape
    x, n, r = _rmsnorm(h, w[NORM], dt(spec.norm_eps))
    x = x.reshape(b * t, d)
    cache["final"] = (x, n, r)
    cache["logits"] = x @ w[EMBED if spec.tie_embeddings else HEAD].T
    return cache, cache["logits"].reshape(b, t, spec.vocab)


def stage_backward(spec: TransformerSpec, stage: Stage, base, trainable, cache: dict, g_out=None, targets=None):
    """One stage's share of the backward pass, with the same `base` and `trainable` slices its forward had. The last
    stage takes the `targets` [batch, seq] and starts from the loss; the others take `g_out`, the gradient at the
    residual stream they sent on. Returns (g_in, grad): the gradient at the residual stream it took in, to send back
    (None from the first stage), and the gradient of its trainable slice, at the spec's dtype."""
    w, ad = _weights(spec, stage, base, trainable)
    full, grads = not spec.lora_rank, {}
    if stage.last:
        x, n, r = cache["final"]
        b, t, d = n.shape
        targets = _tokens(spec, targets)
        if targets.shape != (b, t):
            raise ValueError("targets must be [batch, seq], like the inputs")
        dlogits = _xent_back(cache["logits"], targets.ravel())
        head = EMBED if spec.tie_embeddings else HEAD
        dx = dlogits @ w[head]
        if full:
            grads[head] = dlogits.T @ x
        dh = _rmsnorm_back(dx.reshape(b, t, d), w[NORM], n, r, grads, NORM, full)
    else:
        dh = np.ascontiguousarray(g_out, dtype=spec.dtype)
        if dh.shape != (*cache["blocks"][-1]["n1"].shape[:2], spec.d_model):
            raise ValueError("the gradient must have the shape of the residual stream sent on")
    for i in range(stage.hi - 1, stage.lo - 1, -1):
        dh = _block_back(spec, w, ad, i, cache["blocks"][i - stage.lo], dh, grads, full)
    if stage.first and full:
        embed = grads.get(EMBED)  # already the head's gradient, if it's tied and this stage is also the last
        if embed is None:
            embed = np.zeros((spec.vocab, spec.d_model), spec.dtype)
        np.add.at(embed, cache["tokens"], dh)  # each position adds to its token's row, in order
        grads[EMBED] = embed
    grad = np.concatenate([grads[name].ravel() for name, _ in _trainable_entries(spec, stage)])
    return (None if stage.first else np.ascontiguousarray(dh)), grad


# --- the whole model ------------------------------------------------------------------------------------------------

def forward(spec: TransformerSpec, base, trainable, tokens) -> np.ndarray:
    """Logits [batch, seq, vocab] at the spec's dtype for input tokens [batch, seq]. Positions start at 0."""
    return stage_forward(spec, _whole(spec), base, trainable, tokens)[1]


def loss(spec: TransformerSpec, base, trainable, tokens) -> float:
    """Mean next-token cross-entropy over rows [batch, seq + 1]: inputs row[:-1], targets row[1:]."""
    tokens = _tokens(spec, tokens)
    logits = forward(spec, base, trainable, tokens[:, :-1])
    return float(np.mean(_xent(logits.reshape(-1, spec.vocab), tokens[:, 1:].ravel()), dtype=np.float64))


def gradient(spec: TransformerSpec, base, trainable, tokens) -> np.ndarray:
    """The loss's gradient with respect to the trainable vector, at the spec's dtype, for rows [batch, seq + 1]."""
    tokens = _tokens(spec, tokens)
    stage = _whole(spec)
    cache, _ = stage_forward(spec, stage, base, trainable, tokens[:, :-1])
    return stage_backward(spec, stage, base, trainable, cache, targets=tokens[:, 1:])[1]


def _batch(data, unit_seed: int, step: int, batch_size: int) -> np.ndarray:
    rows = data.tokens_train
    return np.asarray(rows[batch_indices(unit_seed, step, len(rows), batch_size)])


def train_segment(spec: TransformerSpec, data, base, trainable, unit_seed: int, start_step: int, n_steps: int,
                  lr: float, batch_size: int) -> np.ndarray:
    """SGD steps [start_step, start_step + n_steps) on the trainable vector, as model.train_segment_numpy: each step's
    rows come from model.batch_indices, so any segment replays alone, and the arithmetic runs at the spec's precision.
    Plain SGD keeps no optimiser state, so a checkpoint is just the trainable vector. Returns it as float64, rounded to
    the precision. The base never changes."""
    dt = spec.dtype
    base = None if not spec.lora_rank else np.asarray(base, dtype=dt)
    theta = np.asarray(trainable, dtype=dt).copy()
    lr = dt(lr)
    for step in range(start_step, start_step + n_steps):
        theta -= lr * gradient(spec, base, theta, _batch(data, unit_seed, step, batch_size))
    return theta.astype(np.float64)


def train_stages(spec: TransformerSpec, stages: list[Stage], data, base, trainable, unit_seed: int, start_step: int,
                 n_steps: int, lr: float, batch_size: int) -> list[np.ndarray]:
    """train_segment run as a pipeline across `stages`, in one process: each step runs every stage's forward in order,
    then every backward in reverse, and each stage updates its own slice once its backward is done. Returns each
    stage's trainable slice (`assemble` joins them), which equals train_segment's bit for bit. A team does the same
    with each stage on its own worker and the edges sent between them."""
    dt = spec.dtype
    lr = dt(lr)
    bases = [None if not spec.lora_rank else np.asarray(stage_base(spec, s, base), dtype=dt) for s in stages]
    parts = [np.asarray(trainable[trainable_span(spec, s)], dtype=dt).copy() for s in stages]
    for step in range(start_step, start_step + n_steps):
        rows = _tokens(spec, _batch(data, unit_seed, step, batch_size))
        h, caches = rows[:, :-1], []
        for k, s in enumerate(stages):
            cache, h = stage_forward(spec, s, bases[k], parts[k], h)
            caches.append(cache)
        g = None
        for k in range(len(stages) - 1, -1, -1):
            s = stages[k]
            g, grad = stage_backward(spec, s, bases[k], parts[k], caches[k], g, rows[:, 1:] if s.last else None)
            parts[k] -= lr * grad
    return [p.astype(np.float64) for p in parts]


def evaluate(spec: TransformerSpec, base, trainable, tokens, batch_size: int = 64) -> tuple[float, float]:
    """(mean next-token cross-entropy, next-token accuracy) over rows [n, seq_len + 1], e.g. a dataset's tokens_val.
    For reporting, not replay: the forward runs at the spec's precision, `batch_size` rows at a time, and the totals
    are float64."""
    if not len(tokens):
        raise ValueError("nothing to evaluate")
    dt = spec.dtype
    base = None if not spec.lora_rank else np.asarray(base, dtype=dt)
    trainable = np.asarray(trainable, dtype=dt)
    total, right, count = 0.0, 0, 0
    for i in range(0, len(tokens), batch_size):
        rows = _tokens(spec, tokens[i:i + batch_size])
        logits = forward(spec, base, trainable, rows[:, :-1]).reshape(-1, spec.vocab)
        targets = rows[:, 1:].ravel()
        total += float(np.sum(_xent(logits, targets), dtype=np.float64))
        right += int((logits.argmax(axis=1) == targets).sum())
        count += len(targets)
    return total / count, right / count

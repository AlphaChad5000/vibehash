"""Pretrained decoder-only language models in plain PyTorch, for LoRA fine-tuning jobs (finetune.py).

The families that share Llama's shape: `llama` (Llama 2 and 3, with Llama 3's rotary scaling), `mistral`, `qwen2`
(biases on the query, key and value projections), `qwen3` (an RMSNorm on each query and key head) and `smollm3` (no
rotary positions on every fourth layer). Each layer does
    h = h + o_proj(attention(rope(q_proj(x)), rope(k_proj(x)), v_proj(x)))    where x = rmsnorm(h)
    h = h + down_proj(silu(gate_proj(x)) * up_proj(x))                       where x = rmsnorm(h)
with causal, grouped-query attention, then a final RMSNorm and the head (the embedding itself when tied). Names and
shapes are Hugging Face's, so a checkpoint's tensors load as they are, and Hugging Face transformers is the reference
it's checked against (tests/test_llm.py). transformer.py is the same model in NumPy for small Llamas.

How it computes, so that every device computes the same thing and only rounding differs:
- **float32 everywhere.** Frozen weights stay as stored (usually bf16) and are cast to float32 inside each matmul
  (`_Frozen`), which is exact, so every device multiplies by identical weights, and a 3B model's 6 GB of bf16 weights
  don't become 12 GB. Matmuls run with TF32 off and deterministic kernels (compute.TorchBackend's settings).
- **Attention written out** (scores, causal mask, softmax, mix) rather than a fused kernel whose algorithm differs by
  device, and each layer recomputed in the backward (activation checkpointing), so one layer's scores are the only
  big thing alive at once. Recomputing runs the same kernels on the same inputs, so it changes no bits.
- **Rotary tables in float64 on the CPU**, then rounded to float32 and copied over: every device rotates by the same
  numbers (Hugging Face computes them in float32 on the device, which differs in the last bits between devices).
- **The head and the loss fused** (`_HeadLoss`): logits are made a chunk of tokens at a time and only for the tokens
  that count, and their gradient is folded back at once, so a 150k-word vocabulary never needs the logits of every
  position in memory.

LoRA: an adapted projection computes x·Wᵀ + ((x·Aᵀ)·Bᵀ)·(alpha/rank), as PEFT does, with A [rank, in] and B [out, rank].
Which are trained, and how, is finetune.py's business: this module takes A and B as tensors and gives gradients for
whichever require them.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

FAMILIES = ("llama", "mistral", "qwen2", "qwen3", "smollm3")
TARGETS = ("q", "k", "v", "o", "gate", "up", "down")  # what LoRA can adapt, in their order within a layer (transformer.py's)
MODULES = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
           "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
EMBED, NORM, HEAD = "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"
STORED = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}  # weight dtypes it loads (safetensors names)
HEAD_CHUNK = 512  # tokens whose logits are made at once: 512 × a 152k vocabulary is 311 MB of float32


@dataclass(frozen=True)
class Arch:
    """What a checkpoint's config.json says about its shape and arithmetic, for the families above. `dims()` is the
    part a job pins in its spec: enough to size the adapters and price the work without fetching anything."""
    family: str
    vocab: int
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    tied: bool
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: tuple = ()  # Llama 3's: (factor, low_freq_factor, high_freq_factor, original_max_position_embeddings)
    qkv_bias: bool = False
    qk_norm: bool = False
    nope: tuple[int, ...] = ()  # layers without rotary positions (SmolLM3)
    window: int = 0  # a sliding window over the keys, 0 if none
    max_positions: int = 0  # the longest sequence it was trained for, 0 if the config doesn't say

    @classmethod
    def from_hf_config(cls, config: dict) -> "Arch":
        """The arch of a Hugging Face checkpoint, from its config.json. Refuses what this module doesn't compute
        rather than train a different model under the checkpoint's name."""
        if not isinstance(config, dict):
            raise ValueError("config.json isn't a JSON object")
        family = config.get("model_type")
        if family not in FAMILIES:
            raise ValueError(f"model_type {family!r} isn't supported yet (supported: {', '.join(FAMILIES)})")
        rope = dict(config.get("rope_parameters") or config.get("rope_scaling") or {})
        kind = rope.get("rope_type", rope.get("type", "default"))
        problems = [why for bad, why in [
            (config.get("hidden_act", "silu") != "silu", f"activation {config.get('hidden_act')!r}"),
            (kind not in ("default", "llama3"), f"rotary scaling {kind!r}"),
            (bool(config.get("mlp_bias")), "biases in the MLP"),
            (family not in ("qwen2",) and bool(config.get("attention_bias")), "attention biases"),
        ] if bad]
        if problems:
            raise ValueError(f"not supported yet: {', '.join(problems)}")
        layers, heads = int(config["num_hidden_layers"]), int(config["num_attention_heads"])
        scaling = ()
        if kind == "llama3":
            scaling = (float(rope["factor"]), float(rope["low_freq_factor"]), float(rope["high_freq_factor"]),
                       int(rope["original_max_position_embeddings"]))
        nope = ()
        if family == "smollm3":
            use = config.get("no_rope_layers") or [int((i + 1) % int(config.get("no_rope_layer_interval", 4)) != 0)
                                                  for i in range(layers)]
            nope = tuple(i for i, u in enumerate(use) if not u)
        window = 0
        if config.get("sliding_window") and (family == "mistral" or config.get("use_sliding_window")):
            window = int(config["sliding_window"])
        return cls(family=family, vocab=int(config["vocab_size"]), hidden=int(config["hidden_size"]), layers=layers,
                   heads=heads, kv_heads=int(config.get("num_key_value_heads") or heads),
                   head_dim=int(config.get("head_dim") or int(config["hidden_size"]) // heads),
                   ffn=int(config["intermediate_size"]), tied=bool(config.get("tie_word_embeddings", False)),
                   norm_eps=float(config.get("rms_norm_eps", 1e-6)),
                   rope_theta=float(rope.get("rope_theta", config.get("rope_theta", 10000.0))), rope_scaling=scaling,
                   qkv_bias=family == "qwen2" or bool(config.get("attention_bias")), qk_norm=family == "qwen3",
                   nope=nope, window=window, max_positions=int(config.get("max_position_embeddings") or 0))

    def dims(self) -> dict:
        """What a fine-tune's spec pins about the model, checked against config.json when the weights load."""
        return {k: v for k, v in asdict(self).items()
                if k in ("family", "vocab", "hidden", "layers", "heads", "kv_heads", "head_dim", "ffn", "tied")}

    def shape(self, target: str) -> tuple[int, int]:
        """A projection's weight shape as Hugging Face stores it, [out, in]."""
        d, f, q, kv = self.hidden, self.ffn, self.heads * self.head_dim, self.kv_heads * self.head_dim
        return {"q": (q, d), "k": (kv, d), "v": (kv, d), "o": (d, q), "gate": (f, d), "up": (f, d), "down": (d, f)}[target]

    @property
    def n_weights(self) -> int:
        """Weights in all, the embedding and head included."""
        return self.layers * (sum(math.prod(self.shape(t)) for t in TARGETS) + 2 * self.hidden) + \
            self.vocab * self.hidden * (1 if self.tied else 2) + self.hidden

    def tensor_names(self) -> list[str]:
        """Every tensor the model reads, in Hugging Face's names."""
        names = [EMBED]
        for i in range(self.layers):
            p = f"model.layers.{i}."
            names += [p + MODULES[t] + ".weight" for t in TARGETS]
            names += [p + "input_layernorm.weight", p + "post_attention_layernorm.weight"]
            if self.qkv_bias:
                names += [p + MODULES[t] + ".bias" for t in ("q", "k", "v")]
            if self.qk_norm:
                names += [p + "self_attn.q_norm.weight", p + "self_attn.k_norm.weight"]
        names.append(NORM)
        if not self.tied:
            names.append(HEAD)
        return names

    def check(self, seq_len: int) -> None:
        if self.window and self.window < seq_len:
            raise ValueError(f"its sliding window ({self.window} tokens) is shorter than the sequences ({seq_len}); "
                             "that isn't supported yet")
        if self.heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("query heads must be a multiple of key/value heads, and head_dim even")


# --- the frozen weights -------------------------------------------------------------------------------------------

class _Frozen(torch.autograd.Function):
    """x · Wᵀ for a frozen W kept as stored: cast to x's float32 in the forward and again in the backward, so autograd
    keeps nothing of it but the stored tensor."""

    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(w)
        return F.linear(x, w.to(x.dtype))

    @staticmethod
    def backward(ctx, dy):
        (w,) = ctx.saved_tensors
        return dy @ w.to(dy.dtype), None


def _rmsnorm(h: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Hugging Face's order: normalise, then scale."""
    return weight * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def rope_table(arch: Arch, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin, [length, head_dim] in Hugging Face's rotate-half layout, computed in float64 on the CPU (with Llama
    3's frequency scaling when the config has it). Rounded to float32 by the model, so the same on every device."""
    d = arch.head_dim
    inv = 1.0 / (arch.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    if arch.rope_scaling:
        factor, low, high, old = arch.rope_scaling
        wavelen = 2 * math.pi / inv
        scaled = torch.where(wavelen > old / low, inv / factor, inv)
        smooth = (old / wavelen - low) / (high - low)
        medium = (wavelen >= old / high) & (wavelen <= old / low)
        inv = torch.where(medium, (1 - smooth) * scaled / factor + smooth * scaled, scaled)
    angles = torch.outer(torch.arange(length, dtype=torch.float64), inv)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


# --- the model --------------------------------------------------------------------------------------------------------

class Model:
    """A pretrained model's frozen weights on one device, and the forward pass with optional LoRA adapters.

    `adapters` maps a projection's name (`model.layers.3.self_attn.q_proj`) to (A, B) tensors of the model's dtype on
    the same device; gradients flow to whichever of them require one. `scaling` is lora_alpha / rank. `dtype` is what
    it computes in: float32 for jobs, float64 for checking it against references."""

    def __init__(self, arch: Arch, weights: dict[str, torch.Tensor], device: str | torch.device = "cpu",
                 recompute: bool = True, dtype: torch.dtype = torch.float32):
        missing = [n for n in arch.tensor_names() if n not in weights]
        if missing:
            raise ValueError(f"the checkpoint has no {missing[0]}")
        self.arch, self.device, self.recompute, self.dtype = arch, torch.device(device), recompute, dtype
        self.w = {}
        for name in arch.tensor_names():
            t = weights[name]
            if t.dtype not in (*STORED.values(), torch.float64):
                raise ValueError(f"{name} is {t.dtype}, which isn't supported (bf16, f16 or f32)")
            # Norms and biases are small: kept in the compute dtype. Matrices stay as stored, cast inside each matmul.
            self.w[name] = t.to(self.device, dtype if t.ndim == 1 else t.dtype)
        self._rope: tuple[int, torch.Tensor, torch.Tensor] = (0, None, None)

    def _tables(self, length: int):
        if self._rope[0] < length:
            cos, sin = rope_table(self.arch, max(length, 256))
            self._rope = (cos.shape[0], cos.to(self.dtype).to(self.device), sin.to(self.dtype).to(self.device))
        return self._rope[1][:length], self._rope[2][:length]

    def _proj(self, x, name: str, adapters: dict, scaling: float):
        y = _Frozen.apply(x, self.w[name + ".weight"])
        bias = self.w.get(name + ".bias")
        if bias is not None:
            y = y + bias
        ad = adapters.get(name)
        if ad is not None:
            a, b = ad
            y = y + ((x @ a.T) @ b.T) * scaling
        return y

    def _layer(self, i: int, h, cos, sin, mask, adapters: dict, scaling: float):
        arch, w, p = self.arch, self.w, f"model.layers.{i}."
        n, t = h.shape[:2]
        hd, kvh, groups = arch.head_dim, arch.kv_heads, arch.heads // arch.kv_heads
        x = _rmsnorm(h, w[p + "input_layernorm.weight"], arch.norm_eps)
        q = self._proj(x, p + "self_attn.q_proj", adapters, scaling).view(n, t, arch.heads, hd)
        k = self._proj(x, p + "self_attn.k_proj", adapters, scaling).view(n, t, kvh, hd)
        v = self._proj(x, p + "self_attn.v_proj", adapters, scaling).view(n, t, kvh, hd)
        if arch.qk_norm:
            q = _rmsnorm(q, w[p + "self_attn.q_norm.weight"], arch.norm_eps)
            k = _rmsnorm(k, w[p + "self_attn.k_norm.weight"], arch.norm_eps)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [n, heads, t, hd]
        if i not in arch.nope:
            q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        # Grouped-query attention without copying keys and values: query heads [n, kvh, groups, t, hd] meet their
        # key/value head [n, kvh, 1, t, hd] by broadcasting. Each score is the same dot product either way.
        q = q.reshape(n, kvh, groups, t, hd)
        scores = (q @ k.unsqueeze(2).transpose(-1, -2)) * (1.0 / math.sqrt(hd)) + mask
        o = (torch.softmax(scores, dim=-1) @ v.unsqueeze(2)).reshape(n, arch.heads, t, hd)
        o = o.transpose(1, 2).reshape(n, t, arch.heads * hd)
        h = h + self._proj(o, p + "self_attn.o_proj", adapters, scaling)
        x = _rmsnorm(h, w[p + "post_attention_layernorm.weight"], arch.norm_eps)
        g = self._proj(x, p + "mlp.gate_proj", adapters, scaling)
        u = self._proj(x, p + "mlp.up_proj", adapters, scaling)
        return h + self._proj(F.silu(g) * u, p + "mlp.down_proj", adapters, scaling)

    def hidden(self, tokens: torch.Tensor, adapters: dict | None = None, scaling: float = 0.0) -> torch.Tensor:
        """The final hidden states (after the last norm), [n, t, hidden], for token ids [n, t]."""
        adapters = adapters or {}
        tokens = tokens.to(self.device)
        n, t = tokens.shape
        h = self.w[EMBED][tokens].to(self.dtype)
        cos, sin = self._tables(t)
        mask = torch.full((t, t), float("-inf"), device=self.device, dtype=self.dtype).triu(1)
        grad = torch.is_grad_enabled() and any(x.requires_grad for ab in adapters.values() for x in ab)
        for i in range(self.arch.layers):
            if grad and self.recompute:
                h = checkpoint(self._layer, i, h, cos, sin, mask, adapters, scaling, use_reentrant=False)
            else:
                h = self._layer(i, h, cos, sin, mask, adapters, scaling)
        return _rmsnorm(h, self.w[NORM], self.arch.norm_eps)

    @property
    def head(self) -> torch.Tensor:
        return self.w[EMBED] if self.arch.tied else self.w[HEAD]

    def logits(self, tokens: torch.Tensor, adapters: dict | None = None, scaling: float = 0.0) -> torch.Tensor:
        """Every position's logits, [n, t, vocab]. For checking against a reference; training uses `loss`."""
        return _Frozen.apply(self.hidden(tokens, adapters, scaling), self.head)

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor, adapters: dict | None = None, scaling: float = 0.0):
        """(sum of next-token cross-entropy over the target positions, how many there are). `targets` [n, t] is 1 where
        a token is one to learn: it's predicted from the tokens before it. Position 0 never is."""
        tokens, targets = tokens.to(self.device), targets.to(self.device)
        h = self.hidden(tokens, adapters, scaling)
        pick = targets[:, 1:].bool()
        return _HeadLoss.apply(h[:, :-1][pick], self.head, tokens[:, 1:][pick]), int(pick.sum())

    @torch.no_grad()
    def score(self, tokens: torch.Tensor, targets: torch.Tensor, adapters: dict | None = None,
              scaling: float = 0.0) -> tuple[float, int, int]:
        """(summed cross-entropy, tokens the model ranks first, target tokens), for a validation set."""
        tokens, targets = tokens.to(self.device), targets.to(self.device)
        h = self.hidden(tokens, adapters, scaling)
        pick = targets[:, 1:].bool()
        hs, want = h[:, :-1][pick], tokens[:, 1:][pick]
        w, total, right = self.head, 0.0, 0
        for s in range(0, len(want), HEAD_CHUNK):
            logits = _Frozen.apply(hs[s:s + HEAD_CHUNK], w)
            lse = torch.logsumexp(logits, dim=-1)
            total += float((lse - logits.gather(1, want[s:s + HEAD_CHUNK, None])[:, 0]).sum())
            right += int((logits.argmax(dim=-1) == want[s:s + HEAD_CHUNK]).sum())
        return total, right, len(want)


class _HeadLoss(torch.autograd.Function):
    """Σ over tokens of -log softmax(h·Wᵀ)[target], made HEAD_CHUNK tokens at a time. The gradient with respect to h,
    (softmax - onehot)·W, is worked out in the same pass and kept; the logits never are."""

    @staticmethod
    def forward(ctx, h, w, targets):
        total = h.new_zeros(())
        grad = torch.empty_like(h)
        w32 = w.to(h.dtype)
        for s in range(0, h.shape[0], HEAD_CHUNK):
            logits = F.linear(h[s:s + HEAD_CHUNK], w32)
            t = targets[s:s + HEAD_CHUNK]
            lse = torch.logsumexp(logits, dim=-1)
            total = total + (lse - logits.gather(1, t[:, None])[:, 0]).sum()
            p = torch.exp(logits - lse[:, None])
            p[torch.arange(len(t), device=p.device), t] -= 1
            grad[s:s + HEAD_CHUNK] = p @ w32
        ctx.save_for_backward(grad)
        return total

    @staticmethod
    def backward(ctx, g):
        (grad,) = ctx.saved_tensors
        return grad * g, None, None


def load(arch: Arch, read, device: str | torch.device = "cpu", recompute: bool = True,
         dtype: torch.dtype = torch.float32) -> Model:
    """A Model from `read(name) -> (safetensors dtype, shape, raw bytes)`, as hub.load_tensor gives them, checked
    against the arch's shapes."""
    expect = {EMBED: (arch.vocab, arch.hidden), NORM: (arch.hidden,), HEAD: (arch.vocab, arch.hidden)}
    for i in range(arch.layers):
        p = f"model.layers.{i}."
        expect |= {p + MODULES[t] + ".weight": arch.shape(t) for t in TARGETS}
        expect |= {p + "input_layernorm.weight": (arch.hidden,), p + "post_attention_layernorm.weight": (arch.hidden,),
                   p + "self_attn.q_norm.weight": (arch.head_dim,), p + "self_attn.k_norm.weight": (arch.head_dim,)}
        expect |= {p + MODULES[t] + ".bias": (arch.shape(t)[0],) for t in ("q", "k", "v")}
    weights = {}
    for name in arch.tensor_names():
        stored, shape, raw = read(name)
        if stored not in STORED:
            raise ValueError(f"{name} is stored as {stored}; only BF16, F16 and F32 weights are supported")
        if tuple(shape) != expect[name]:
            raise ValueError(f"{name} has shape {tuple(shape)}, but the config makes it {expect[name]}")
        weights[name] = torch.frombuffer(bytearray(raw), dtype=STORED[stored]).reshape(shape)
    return Model(arch, weights, device, recompute, dtype)

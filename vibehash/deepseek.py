"""DeepSeek-V4.1-Flash in plain PyTorch: differentiable, deterministic, and checked against DeepSeek's own code.

Step 1 of DEEPSEEK.md §5. DeepSeek's reference (inference/model.py at COMMIT) only runs inference: it works under
torch.inference_mode, writes into caches in place, and leans on tilelang kernels for its FP8 and FP4 matmuls, its
sparse attention and its Sinkhorn step. This is the same model as a function autograd can differentiate, written in
plain PyTorch ops with no kernels of its own, computing the same numbers. tests/test_deepseek.py holds the proofs.

The model. Tokens are embedded and the residual stream starts as hc_mult (4) copies of each token's dim-wide state.
Each Block then does:
- Hyper-connections. hc_mixes reads the whole stream and makes three sets of coefficients: `pre` (how to collapse the
  copies into a sublayer's input), `post` (how to spread its output back over them) and `comb` (a 4x4 mix of the
  copies, made doubly stochastic by Sinkhorn). A quirk the reference has and so this has: the `pre` a sublayer makes
  collapses the input of the *next* sublayer, so a block takes the previous block's `pre_mix` and hands on its own.
- Attention. Low-rank queries (wq_a, q_norm, wq_b); one key/value head shared by every query head (wkv, kv_norm: the
  keys are the values); RoPE on the last rope_head_dim dims, YaRN-stretched in compressing layers; attention sinks;
  and a grouped low-rank output (wo_a in o_groups blocks, then wo_b). Each query sees a sliding window of
  window_size raw tokens, and in a layer with compress ratio r > 0 also compressed tokens reaching further back: a
  Compressor pools each r tokens into one latent, and an Indexer, a small extra attention, picks the index_topk best
  of them for each query. Only kv_source_layers compress and only index_source_layers pick; the layers after a
  source reuse what it made (Side). From candidate_source_layer on, the indexer first keeps candidate_topk_blocks
  blocks of candidate_block_size and picks only inside them (it matters beyond 16K tokens).
- MoE. A router scores n_routed_experts with sqrt(softplus(x . w)); a per-expert bias steers which top-k are chosen
  but not how much each counts; the gates are the chosen scores normalised, times route_scale. Plus one shared expert
  every token goes through. SwiGLU experts, their up branch clamped to ±swiglu_limit and their gate branch below it.
- Engram, in engram_layer_ids only, before the block. Each token's 2..N-grams of (normalised) token ids are hashed
  into rows of a big table, which depend only on the text (engram_hashes); a gate per stream copy mixes their
  projection in.
After the last block the stream is collapsed once more, normed, and the head gives logits.

Weights stay as DeepSeek stores them: FP8 E4M3 matrices with one power-of-two (ue8m0) scale per 32x32 block; routed
experts in FP4 E2M1, two values to a byte, one scale per 32 inputs; everything else bf16 or float32. Quantised
matrices are dequantised one at a time, inside the matmul and again in its backward (_FrozenMatmul), so nothing
dequantised outlives one projection of one expert and a 7.4 GB layer fits a 12 GB GPU. The scales being powers of
two, dequantising is exact (an E4M3 or E2M1 number times a power of two is a bf16 number): every device computes with
exactly the same weights. Quantised matrices are frozen buffers; bf16 and float32 weights are Parameters.

Numerics. The default, Numerics(), is the reference's arithmetic: activations in bf16, float32 wherever it upcasts,
and activations rounded to FP8 (to FP4 in the indexer and the compressed keys) exactly where its kernels round them,
with a straight-through gradient. Numerics(torch.float64, quantize=False) is the same function without any rounding,
smooth enough for finite differences.

Discrete choices. The MoE's top-k experts, the indexer's picks and the candidate blocks are made by comparing scores,
so rounding can flip near-ties between GPUs (moe.py explains why a checker cares). A forward takes them as an input
(Choices), computes only the ones it isn't given, and returns all it used, so a worker can commit to them and a
checker replay with them. choice_gap measures how far a given choice is from a score's own top-k.

Side channels. A kv source's compressed keys reach the layers that reuse them, and so does their gradient on the way
back, so a pipeline split between a source and its users sends them across, both ways. Index keys, picks and
candidates are discrete and carry no gradient: the indexer is never trained by the language-model loss.

Scope: prefill only (training sees whole sequences: no KV cache, no decoding), text only (no vision tower, no image
routing bias), no MTP/DSpark draft layers.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass, field, fields
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from . import settings

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
COMMIT = "314113f8a7dadb388215396ba5fbee2c610a04d6"  # the weights and reference code this was checked against

BLOCK = 32  # one scale per 32x32 block of an FP8 matrix, per 32 inputs of an FP4 one, per 32 activations
FP8_MAX, FP4_MAX = 448.0, 6.0
# E2M1 values by 4-bit code: the low three bits the magnitude, the high one the sign (convert.py's FP4_TABLE).
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
SCORE_FUNCS = ("softmax", "sigmoid", "sqrtsoftplus")
# What LoRA can adapt, by short name: attention's four plain projections and the shared expert's three. Not wo_a: it's
# block-diagonal (o_groups groups), which PEFT's LoRA can't express. Not the routed experts: each token uses 6 of 384.
LORA_MODULES = {"wq_a": "attn.wq_a", "wq_b": "attn.wq_b", "wkv": "attn.wkv", "wo_b": "attn.wo_b",
                "w1": "ffn.shared_experts.w1", "w2": "ffn.shared_experts.w2", "w3": "ffn.shared_experts.w3"}
LORA_TARGETS = ("wq_a", "wq_b", "wkv", "wo_b")  # the default: attention, as finetune.py's q, k, v, o
HEAD_CHUNK = 512  # positions whose logits are made at once
IMAGE_ONLY = (".gate.bias_vl",)  # checkpoint tensors only image tokens use


def configure_torch() -> None:
    """Reproducible PyTorch, as compute.TorchBackend sets it up for MLP jobs, and no reduced-precision reductions in
    bf16 matmuls either. Call it before any CUDA work: cuBLAS reads CUBLAS_WORKSPACE_CONFIG when it starts."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False  # TF32 keeps 10 mantissa bits: replays wouldn't match
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


# --- the shape -----------------------------------------------------------------------------------------------------

# config.json's text_config names, where they differ from the reference's (inference/config.json's).
_HF_NAMES = {"hidden_size": "dim", "moe_intermediate_size": "moe_inter_dim", "num_hidden_layers": "n_layers",
             "num_attention_heads": "n_heads", "qk_rope_head_dim": "rope_head_dim", "rms_norm_eps": "norm_eps",
             "num_experts_per_tok": "n_activated_experts", "scoring_func": "score_func",
             "routed_scaling_factor": "route_scale", "sliding_window": "window_size",
             "kv_source_layer_ids": "kv_source_layers", "index_source_layer_ids": "index_source_layers",
             "candidate_source_layer_id": "candidate_source_layer", "engram_pad_token_id": "engram_pad_id"}


@dataclass(frozen=True)
class Config:
    """The model's shape. Field names and defaults are the reference's ModelArgs, so inference/config.json loads with
    from_dict and the defaults are its small 5-layer test model; flash() is the real one, and from_hf reads the
    Hub's config.json. Fields of ModelArgs this doesn't use (vision, MTP/DSpark, runtime limits) are left out."""
    vocab_size: int = 129280
    dim: int = 1024
    moe_inter_dim: int = 1024
    n_layers: int = 5
    n_heads: int = 16
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.0
    swiglu_limit: float = 0.0
    q_lora_rank: int = 256
    head_dim: int = 128
    rope_head_dim: int = 32
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 256
    window_size: int = 128
    compress_ratios: tuple[int, ...] = (0, 2, 2, 1, 1, 0)  # one per layer (MTP layers' too): 0 = window only
    kv_source_layers: tuple[int, ...] = (1, 3)
    index_source_layers: tuple[int, ...] = (1, 3)
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0  # > 0 turns on YaRN, in the layers that compress
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 16
    index_head_dim: int = 64
    index_topk: int = 64
    candidate_source_layer: int = -1  # < 0: no candidate blocks
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()  # table rows per engram layer: the sum of its bucket primes
    engram_max_ngram_size: int = 1
    engram_vocab_size: int = 0  # where each (n-gram size, head)'s search for its bucket prime starts
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_pad_id: int = 2  # the token that fills n-gram slots before the sequence starts
    engram_compressed_vocab_size: int = 0  # every hash multiplier derives from it
    dtype: str = "fp8"  # how the checkpoint stores matrices; only DeepSeek's own formats are read
    expert_dtype: str = "fp4"

    def __post_init__(self):
        def need(ok, what):
            if not ok:
                raise ValueError(f"unsupported DeepSeek-V4.1 shape: {what}")
        need(self.dtype == "fp8" and self.expert_dtype == "fp4", "weights stored other than as FP8 with FP4 experts")
        need(self.score_func in SCORE_FUNCS, f"score_func {self.score_func!r}")
        need(self.n_shared_experts == 1, "a shared expert count other than 1")
        need(len(self.compress_ratios) >= self.n_layers, "fewer compress_ratios than layers")
        for name in ("dim", "q_lora_rank", "head_dim", "moe_inter_dim", "index_head_dim", "o_lora_rank"):
            need(getattr(self, name) % BLOCK == 0, f"{name} not a multiple of {BLOCK} (the quantisation block)")
        need(self.n_heads * self.head_dim % (self.o_groups * BLOCK) == 0, "heads that don't split into o_groups")
        need(self.rope_head_dim % 2 == 0 and self.rope_head_dim <= self.head_dim, "rope_head_dim")
        kv = ix = None
        for i in range(self.n_layers):
            if i in self.kv_source_layers:
                kv = self.compress_ratios[i]
            if i in self.index_source_layers:
                ix = kv
            if self.compress_ratios[i]:
                need(kv == self.compress_ratios[i] and ix == kv,
                     f"layer {i} compresses but no kv and index source at or before it has its ratio")
        if self.engram_layer_ids:
            need(len(self.engram_num_embeddings) == len(self.engram_layer_ids), "engram_num_embeddings")
            need(self.engram_max_ngram_size >= 2 and self.engram_head_dim % BLOCK == 0, "engram n-grams or head_dim")

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """From the reference's own config (inference/config.json): its keys are these fields' names; others ignored."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in names})

    @classmethod
    def from_hf(cls, config: dict) -> "Config":
        """From the Hub's config.json (transformers' deepseek_v41, its shape in text_config)."""
        text = config.get("text_config", config)
        for key, want in (("hidden_act", "silu"), ("topk_method", "noaux_tc"), ("num_key_value_heads", 1)):
            if text.get(key, want) != want:
                raise ValueError(f"unsupported DeepSeek-V4.1 shape: {key} {text[key]!r}")
        d = {_HF_NAMES.get(k, k): v for k, v in text.items()}
        rope = text.get("rope_scaling") or {}
        if rope.get("rope_type") == "yarn":
            d.update(rope_factor=rope["factor"], beta_fast=rope["beta_fast"], beta_slow=rope["beta_slow"],
                     original_seq_len=rope["original_max_position_embeddings"])
        return cls.from_dict(d)

    @classmethod
    def flash(cls) -> "Config":
        """DeepSeek-V4.1-Flash, as inference/config.json at COMMIT has it."""
        return cls(dim=5120, moe_inter_dim=2304, n_layers=40, n_heads=64, n_routed_experts=384, n_activated_experts=6,
                   route_scale=1.5, swiglu_limit=10.0, q_lora_rank=1280, head_dim=512, rope_head_dim=64, o_lora_rank=1024,
                   compress_ratios=(0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0), kv_source_layers=(2, 8, 14, 20),
                   index_source_layers=(2, 8, 14, 20, 24, 28, 32, 36), compress_rope_theta=160000,
                   original_seq_len=65536, rope_theta=10000, rope_factor=16, index_n_heads=32, index_head_dim=128,
                   index_topk=512, candidate_source_layer=20, candidate_topk_blocks=2048, candidate_block_size=8,
                   engram_layer_ids=(1, 14), engram_num_embeddings=(384006168, 384016682), engram_max_ngram_size=4,
                   engram_vocab_size=16000000, engram_n_heads=8, engram_head_dim=256, engram_compressed_vocab_size=99092)

    def ratio(self, layer: int) -> int:
        return self.compress_ratios[layer]

    def source(self, layer: int, kind: str) -> int | None:
        """The layer whose compressed keys and index keys (kind "kv") or picks (kind "index") `layer` uses: the last
        kv or index source at or before it."""
        sources = self.kv_source_layers if kind == "kv" else self.index_source_layers
        return max((i for i in sources if i <= layer), default=None)

    def rope_args(self, layer: int) -> tuple:
        """(dim, original_seq_len, base, factor, beta_fast, beta_slow) of a layer's rotary table: compressing layers
        use compress_rope_theta and YaRN, window-only ones the plain rope_theta."""
        if self.ratio(layer):
            return self.rope_head_dim, self.original_seq_len, self.compress_rope_theta, self.rope_factor, \
                self.beta_fast, self.beta_slow
        return self.rope_head_dim, 0, self.rope_theta, self.rope_factor, self.beta_fast, self.beta_slow


@dataclass(frozen=True)
class Numerics:
    """How a forward rounds. `dtype` is the activations' (the reference's: bf16); `hi` is what it upcasts to.
    `quantize` rounds activations to FP8/FP4 where the reference's kernels do."""
    dtype: torch.dtype = torch.bfloat16
    quantize: bool = True

    @property
    def hi(self) -> torch.dtype:
        return torch.float64 if self.dtype == torch.float64 else torch.float32

    def param_dtype(self, stored: torch.dtype) -> torch.dtype:
        return torch.float64 if self.dtype == torch.float64 else stored

    def fp8(self, x: torch.Tensor) -> torch.Tensor:
        return _Rounded.apply(x, round_fp8) if self.quantize else x

    def fp4(self, x: torch.Tensor, block: int = BLOCK, e4m3_scale: bool = False) -> torch.Tensor:
        return _Rounded.apply(x, partial(round_fp4, block=block, e4m3_scale=e4m3_scale)) if self.quantize else x


@dataclass
class Choices:
    """What a forward decided by comparing scores. Given ones are used as they are; missing ones are computed and
    filled in. With keep_scores, the scores each was chosen from are kept too (for choice_gap)."""
    experts: dict[int, torch.Tensor] = field(default_factory=dict)  # layer -> [tokens, top_k] routed experts
    picks: dict[int, torch.Tensor] = field(default_factory=dict)  # index source -> [b, s, k] compressed tokens, -1 none
    candidates: torch.Tensor | None = None  # [b, s, compressed tokens] bool: candidate_source_layer's blocks
    keep_scores: bool = False
    scores: dict[tuple[str, int], torch.Tensor] = field(default_factory=dict)  # ("experts"|"picks", layer) -> scores


@dataclass
class Side:
    """What a layer hands to the layers after it, besides the residual stream."""
    compress_kv: torch.Tensor | None = None  # [b, s // r, head_dim], from the last kv source: carries a gradient
    index_k: torch.Tensor | None = None  # [b, s // r, index_head_dim], from the last kv source
    picks: torch.Tensor | None = None  # [b, s, k], from the last index source


def choice_gap(scores: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
    """Per row of scores [..., n] and chosen indices [..., k]: how far the best score left out beats the worst score
    chosen. At most 0 when the choice is a top-k of the scores; a checker accepts a committed choice if this is within
    its tie margin. -1 means "none", and is allowed only as padding: when fewer than k options have a finite score
    (the indexer early in a sequence, or inside few candidate blocks). A row that names an option twice, names one
    out of range, or leaves a slot empty while a finite-scored option is left out gets +inf: it's no top-k at all,
    and would otherwise let a worker skip an expert's work by naming another one twice."""
    scores, n, k = scores.double(), scores.size(-1), chosen.size(-1)
    chosen = chosen.long()
    bad = ((chosen < -1) | (chosen >= n)).any(-1)
    ranked = chosen.sort(-1).values
    bad |= ((ranked[..., 1:] == ranked[..., :-1]) & (ranked[..., 1:] >= 0)).any(-1)
    idx = torch.where((chosen >= 0) & (chosen < n), chosen, n)  # "none" goes to a spare column, dropped after
    mask = torch.zeros(*scores.shape[:-1], n + 1, dtype=torch.bool, device=scores.device).scatter(-1, idx, True)[..., :n]
    need = scores.isfinite().sum(-1).clamp(max=k)
    return _mask_gap(scores, mask, need).masked_fill(bad, math.inf)


def candidate_gap(score: torch.Tensor, visible: torch.Tensor, candidates: torch.Tensor, topk_blocks: int,
                  block: int) -> torch.Tensor:
    """choice_gap for committed candidate blocks ([b, s, n] bool, candidate_blocks' form), against the index scores
    they were chosen from ([b, s, n], -inf where unreachable) and each query's visible count ([s, 1]). +inf for a
    mask that isn't whole blocks, keeps more than topk_blocks, keeps an unreachable block or drops the pinned one."""
    width = score.size(-1)
    blocks = _block_scores(score.double(), visible, block)
    kept = F.pad(candidates, (0, -width % block), value=False).unflatten(-1, (-1, block)).any(-1)
    bad = (candidates != kept.repeat_interleave(block, dim=-1)[..., :width]).any(-1)
    bad |= (kept.sum(-1) > topk_blocks) | (kept & (blocks == -math.inf)).any(-1)
    need = (blocks > -math.inf).sum(-1).clamp(max=topk_blocks)
    return _mask_gap(blocks, kept, need).masked_fill(bad, math.inf)


def _mask_gap(scores, mask, need):
    worst = scores.masked_fill(~mask, math.inf).amin(-1)
    best = scores.masked_fill(mask, -math.inf).amax(-1)
    gap = torch.where(best == -math.inf, -math.inf, best - worst)
    return gap.masked_fill(mask.sum(-1) < need, math.inf)


# --- quantisation ----------------------------------------------------------------------------------------------------

def _inv(m: float, device) -> torch.Tensor:
    return torch.tensor(1 / m, dtype=torch.float32, device=device)  # the kernels multiply by 1/max as a float32


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(x)) for positive normal float32 x, from its bits, as the kernels' fast_round_scale computes it."""
    bits = x.contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return ((e + 127) << 23).view(torch.float32)


def _fp4_grid(v: torch.Tensor) -> torch.Tensor:
    """The nearest E2M1 value (0, 0.5, 1, 1.5, 2, 3, 4, 6, and negatives) to each of v in [-6, 6], ties to even: the
    grid steps 0.5, 1 and 2 apart below 2, below 4 and above, and torch.round rounds halves to even."""
    a = v.abs()
    r = torch.where(a < 2, torch.round(a * 2) / 2, torch.where(a < 4, torch.round(a), torch.round(a / 2) * 2))
    return r.copysign(v)


def round_fp8(x: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """x rounded as the reference's act_quant(x, 32, "ue8m0", inplace=True) rounds activations: each run of `block`
    values along the last axis shares the scale 2^ceil(log2(amax / 448)) (amax at least 1e-4), and each value becomes
    the nearest E4M3 number (ties to even) times it. In x's dtype, exactly (it's a bf16 number)."""
    b = x.float().unflatten(-1, (-1, block))
    s = _pow2_ceil(b.abs().amax(-1, keepdim=True).clamp_min(1e-4) * _inv(FP8_MAX, x.device))
    return ((b / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s).flatten(-2).to(x.dtype)


def round_fp4(x: torch.Tensor, block: int = BLOCK, e4m3_scale: bool = False) -> torch.Tensor:
    """x rounded as fp4_act_quant(x, block, inplace=True) rounds it: E2M1 values times one scale per `block`, either
    2^ceil(log2(amax / 6)) (the indexer's), or with e4m3_scale the E4M3 number nearest amax / 6 (the compressed keys',
    16 wide, at least 2^-9 so an all-zero group keeps a scale)."""
    b = x.float().unflatten(-1, (-1, block))
    amax = b.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        s = (amax.clamp_min(FP4_MAX * 2**-9) / FP4_MAX).clamp_max(FP8_MAX).to(torch.float8_e4m3fn).float()
    else:
        s = _pow2_ceil(amax.clamp_min(FP4_MAX * 2**-126) * _inv(FP4_MAX, x.device))
    return (_fp4_grid((b / s).clamp(-FP4_MAX, FP4_MAX)) * s).flatten(-2).to(x.dtype)


class _Rounded(torch.autograd.Function):
    """Forward: fn(x), the rounded value. Backward: the gradient passes straight through, as if unrounded."""

    @staticmethod
    def forward(ctx, x, fn):
        return fn(x)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def dequant_fp8(w: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """An FP8 E4M3 matrix [out, in] times its ue8m0 scale per 32x32 block ([out/32, in/32]), in `dtype`. Exact: an
    E4M3 number times a power of two is a bf16 number, and a float32 and float64 one."""
    out, inp = w.shape
    if out % BLOCK or inp % BLOCK or scale.shape != (out // BLOCK, inp // BLOCK):
        raise ValueError(f"an FP8 matrix {list(w.shape)} with scales {list(scale.shape)}: not 32x32 blocks")
    blocks = w.to(dtype).view(out // BLOCK, BLOCK, inp // BLOCK, BLOCK) * scale.to(dtype)[:, None, :, None]
    return blocks.view(out, inp)


@lru_cache(16)
def _fp4_pairs(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """[256, 2]: the two E2M1 values a byte holds, the low nibble's first."""
    return torch.tensor([[FP4_VALUES[b & 15], FP4_VALUES[b >> 4]] for b in range(256)], dtype=dtype, device=device)


def dequant_fp4(w: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """A packed FP4 matrix (int8 [out, in/2], two E2M1 codes a byte, the low nibble first as convert.py unpacks them)
    times its ue8m0 scale per 32 inputs ([out, in/32]), in `dtype`. Exact, for the same reason as dequant_fp8."""
    out, half = w.shape
    if scale.shape != (out, 2 * half // BLOCK):
        raise ValueError(f"an FP4 matrix {[out, 2 * half]} with scales {list(scale.shape)}: not 32-wide blocks")
    v = F.embedding(w.view(torch.uint8).int(), _fp4_pairs(dtype, w.device))  # [out, in/2, 2]: a gather, 40% faster than indexing
    return (v.view(out, -1, BLOCK) * scale.to(dtype).unsqueeze(-1)).view(out, 2 * half)


def dequant(w: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return dequant_fp4(w, scale, dtype) if w.dtype == torch.int8 else dequant_fp8(w, scale, dtype)


def quantize_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A matrix [out, in] as DeepSeek stores one: E4M3 values and a power-of-two scale per 32x32 block, the smallest
    that keeps the block's largest value within 448."""
    out, inp = w.shape
    b = w.float().view(out // BLOCK, BLOCK, inp // BLOCK, BLOCK)
    s = _pow2_ceil(b.abs().amax((1, 3), keepdim=True).clamp_min(2**-100) * _inv(FP8_MAX, w.device))
    q = (b / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(out, inp)
    return q, s.view(out // BLOCK, inp // BLOCK).to(torch.float8_e8m0fnu)


def quantize_fp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A matrix [out, in] as DeepSeek stores an expert's: E2M1 codes packed two a byte (int8 [out, in/2]) and a
    power-of-two scale per 32 inputs, the smallest that keeps each block's largest value within 6."""
    b = w.float().unflatten(-1, (-1, BLOCK))
    s = _pow2_ceil(b.abs().amax(-1, keepdim=True).clamp_min(2**-100) * _inv(FP4_MAX, w.device))
    v = _fp4_grid((b / s).clamp(-FP4_MAX, FP4_MAX)).flatten(-2)
    a = v.abs()
    code = torch.where(a < 2, a * 2, torch.where(a < 4, a + 2, a / 2 + 4)).to(torch.uint8) | ((v < 0).to(torch.uint8) << 3)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).view(torch.int8), s.squeeze(-1).to(torch.float8_e8m0fnu)


class _FrozenMatmul(torch.autograd.Function):
    """x · Wᵀ for a frozen quantised W: dequantised in the forward and again in the backward, so autograd keeps
    nothing of it but the stored bytes. W may live elsewhere (in pinned CPU memory, say); it's copied to x's device
    for each use. With groups > 1, W is block-diagonal: group g of x's last-but-one axis meets only W's g-th rows."""

    @staticmethod
    def forward(ctx, x, weight, scale, groups):
        ctx.save_for_backward(weight, scale)
        ctx.groups = groups
        return _matmul(x, _dequant_on(weight, scale, x), groups)

    @staticmethod
    def backward(ctx, dy):
        weight, scale = ctx.saved_tensors
        w = _dequant_on(weight, scale, dy)
        if ctx.groups == 1:  # dy · W, as a linear with Wᵀ laid out row by row: 35x faster than dy @ w on a CPU
            return F.linear(dy, w.t().contiguous()), None, None, None
        return torch.einsum("...gr,grd->...gd", dy, w.view(ctx.groups, -1, w.size(1))), None, None, None


def _dequant_on(weight, scale, like):
    return dequant(weight.to(like.device, non_blocking=True), scale.to(like.device, non_blocking=True), like.dtype)


def _matmul(x, w, groups):
    if groups == 1:
        return F.linear(x, w)
    return torch.einsum("...gd,grd->...gr", x, w.view(groups, -1, w.size(1)))


# --- rotary positions ------------------------------------------------------------------------------------------------

@lru_cache(32)
def _rope_table(seqlen: int, dim: int, original_seq_len: int, base: float, factor: float, beta_fast: int,
                beta_slow: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin, [seqlen, dim/2] float32, exactly as the reference's precompute_freqs_cis computes its complex
    table (YaRN when original_seq_len > 0: dimensions whose wavelength fits the training context keep their
    frequency, those far beyond it are divided by `factor`, with a linear ramp between). On the CPU, so every device
    rotates by the same table."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        smooth = 1 - ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    angles = torch.outer(torch.arange(seqlen), freqs)
    cis = torch.polar(torch.ones_like(angles), angles)
    return cis.real.contiguous(), cis.imag.contiguous()


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, num: Numerics, inverse: bool = False) -> torch.Tensor:
    """x [b, s, (heads,) d] with its last 2 * cos.size(-1) dims rotated by position, adjacent pairs (x0, x1) taken as
    x0 + i·x1 and multiplied by e^(i·angle) (conjugated with inverse), in num.hi, rounded back to x's dtype."""
    rd = 2 * cos.size(-1)
    if x.dim() == 4:
        cos, sin = cos[:, None], sin[:, None]
    cos, sin = cos.to(x.device, num.hi), sin.to(x.device, num.hi)
    if inverse:
        sin = -sin
    x0, x1 = x[..., -rd:].to(num.hi).unflatten(-1, (-1, 2)).unbind(-1)
    tail = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), -1).flatten(-2)
    return torch.cat((x[..., :-rd], tail.to(x.dtype)), -1)


# --- the pieces of a layer -------------------------------------------------------------------------------------------

def _param(num: Numerics, shape, stored: torch.dtype, device) -> nn.Parameter:
    return nn.Parameter(torch.empty(shape, dtype=num.param_dtype(stored), device=device))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, num: Numerics, device=None):
        super().__init__()
        self.eps, self.num = eps, num
        self.weight = _param(num, (dim,), torch.bfloat16, device)

    def forward(self, x):
        h = x.to(self.num.hi)
        h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight.to(self.num.hi) * h).to(x.dtype)


class Dense(nn.Module):
    """A matrix the checkpoint keeps in bf16 (the router, the compressor, the indexer's keys and head weights)."""

    def __init__(self, inp: int, out: int, num: Numerics, device=None):
        super().__init__()
        self.weight = _param(num, (out, inp), torch.bfloat16, device)


class QLinear(nn.Module):
    """A frozen quantised projection, held as stored: `weight` is FP8 E4M3 [out, in] (or with fp4, E2M1 packed in
    int8 [out, in/2]) and `scale` its ue8m0 scales. Its input is rounded to FP8 first, as the reference's FP8 and FP4
    GEMMs take theirs.

    `lora`, when set, is a LoRA adapter (A [rank, in], B [out, rank], scaling): the projection then adds
    ((x·Aᵀ)·Bᵀ)·scaling, from its input before the FP8 rounding (the rounding is part of the frozen layer), in
    num.hi, and rounds the sum to the activation dtype, which is what PEFT does with float32 adapters on a bf16
    layer. It isn't part of the state dict: adapters are a job's, the layer DeepSeek's."""

    def __init__(self, inp: int, out: int, num: Numerics, device=None, fp4: bool = False, groups: int = 1):
        super().__init__()
        self.num, self.groups, self.lora = num, groups, None
        if fp4:
            self.register_buffer("weight", torch.empty(out, inp // 2, dtype=torch.int8, device=device))
            self.register_buffer("scale", torch.empty(out, inp // BLOCK, dtype=torch.float8_e8m0fnu, device=device))
        else:
            self.register_buffer("weight", torch.empty(out, inp, dtype=torch.float8_e4m3fn, device=device))
            self.register_buffer("scale", torch.empty(out // BLOCK, inp // BLOCK, dtype=torch.float8_e8m0fnu,
                                                      device=device))

    def forward(self, x):
        return self.adapt(self.matmul(self.num.fp8(x)), x)

    def matmul(self, x):
        """x · Wᵀ for an x already rounded."""
        return _FrozenMatmul.apply(x, self.weight, self.scale, self.groups)

    def adapt(self, y, x):
        """y, the projection of x, plus the adapter's share when there is one."""
        if self.lora is None:
            return y
        a, b, scaling = self.lora
        hi = self.num.hi
        return (y.to(hi) + F.linear(F.linear(x.to(hi), a.to(hi)), b.to(hi)) * scaling).to(y.dtype)


class Compressor(nn.Module):
    """Pools each `ratio` consecutive tokens into one latent [head_dim], weighting them by a softmax over learned
    scores (ratio 1: a plain projection). Returns the latents before RoPE: the indexer's keys are made from those."""

    def __init__(self, cfg: Config, layer: int, num: Numerics, device=None):
        super().__init__()
        self.ratio, self.num = cfg.ratio(layer), num
        self.norm = RMSNorm(cfg.head_dim, cfg.norm_eps, num, device)
        self.wkv = Dense(cfg.dim, cfg.head_dim, num, device)
        if self.ratio > 1:
            self.wgate = Dense(cfg.dim, cfg.head_dim, num, device)

    def forward(self, x):  # [b, s, dim] -> [b, s // ratio, head_dim]
        r, lo, hi = self.ratio, x.dtype, self.num.hi
        if r == 1:  # nothing to pool: no gate, no upcast
            return self.norm(F.linear(x, self.wkv.weight.to(lo)))
        n = x.size(1) // r  # a trailing partial group isn't pooled (in decoding it would wait for its last tokens)
        h = x.to(hi)
        kv = F.linear(h, self.wkv.weight.to(hi))[:, :n * r].unflatten(1, (n, r))
        score = F.linear(h, self.wgate.weight.to(hi))[:, :n * r].unflatten(1, (n, r))
        return self.norm((kv * score.softmax(dim=2)).sum(dim=2).to(lo))


def candidate_blocks(score: torch.Tensor, visible: torch.Tensor, topk_blocks: int, block: int) -> torch.Tensor:
    """The reference's select_candidate_blocks: score [b, s, n] (unreachable positions -inf) to a mask [b, s, n] of
    the topk_blocks blocks of `block` positions with the best best-position, the block holding each query's newest
    position always kept (it's only partly filled, so it could be outscored)."""
    blocks = _block_scores(score, visible, block)
    top = blocks.topk(min(topk_blocks, blocks.size(-1)), dim=-1)
    keep = torch.zeros_like(blocks, dtype=torch.bool).scatter(-1, top.indices, top.values > -math.inf)
    return keep.repeat_interleave(block, dim=-1)[..., :score.size(-1)]


def _block_scores(score, visible, block):
    """Each block's best score, -inf padding the last one out, and the block with the newest visible position +inf."""
    blocks = F.pad(score, (0, -score.size(-1) % block), value=-math.inf).unflatten(-1, (-1, block)).amax(-1)
    return blocks.masked_fill(torch.arange(blocks.size(-1), device=score.device) == (visible - 1) // block, math.inf)


def check_picks(picks: torch.Tensor, b: int, s: int, n: int, k: int, ratio: int, candidates=None) -> None:
    """Refuse committed picks that aren't a top-k in form: [b, s, k] compressed positions a query can see, ascending
    with any -1 ("none") after them, as the indexer makes them, and -1 only where fewer than k positions are open to
    it (visible, and inside its candidate blocks when the layer uses them). Whether they're the *best* k is
    choice_gap's question, which needs the scores."""
    if tuple(picks.shape) != (b, s, k):
        raise ValueError(f"picks are {list(picks.shape)}, not [{b}, {s}, {k}]")
    visible = (torch.arange(1, s + 1, device=picks.device) // ratio).unsqueeze(-1)
    valid = picks >= 0
    open_ = visible.clamp(max=k).squeeze(-1).expand(b, s)
    if candidates is not None:
        seen = candidates[..., :n] & (torch.arange(n, device=picks.device) < visible)
        open_ = seen.sum(-1).clamp(max=k)
    ascending = ((picks[..., 1:] > picks[..., :-1]) | ~valid[..., 1:]).all(-1) & (valid[..., :-1] | ~valid[..., 1:]).all(-1)
    if bool((picks < -1).any() | (picks >= visible).any() | ~ascending.all() | (valid.sum(-1) < open_).any()):
        raise ValueError("picks must be distinct positions each query can see, ascending, with -1 (none) only after "
                         "them and only where fewer than index_topk positions are open")


def check_experts(chosen: torch.Tensor, tokens: int, cfg: "Config") -> None:
    """Refuse committed routing that isn't k distinct experts a token, as moe.py does."""
    k = cfg.n_activated_experts
    if tuple(chosen.shape) != (tokens, k):
        raise ValueError(f"routing is {list(chosen.shape)}, not [{tokens}, {k}]")
    ranked = chosen.sort(-1).values
    if bool((chosen < 0).any() | (chosen >= cfg.n_routed_experts).any() | (ranked[:, 1:] == ranked[:, :-1]).any()):
        raise ValueError(f"routing must name {k} distinct experts of {cfg.n_routed_experts} a token: it sends a token "
                         "to the same expert twice, or to one that doesn't exist")


class Indexer(nn.Module):
    """Picks, for each query, the index_topk compressed tokens it attends to: index_n_heads FP4 queries against one
    FP4 key per compressed token, scores rectified and summed with per-token head weights. Only a layer that
    compresses its own keys (a kv source) makes the index keys; the rest use that layer's. No gradient: picks are
    discrete, and the language-model loss never trains the indexer."""

    QUERY_CHUNK = 1024  # queries scored at once: a [chunk, heads, compressed tokens] bf16 tensor

    def __init__(self, cfg: Config, layer: int, num: Numerics, device=None):
        super().__init__()
        self.cfg, self.layer, self.num, self.ratio = cfg, layer, num, cfg.ratio(layer)
        self.owns_k = layer in cfg.kv_source_layers
        self.wq_b = QLinear(cfg.q_lora_rank, cfg.index_n_heads * cfg.index_head_dim, num, device)
        self.weights_proj = Dense(cfg.dim, cfg.index_n_heads, num, device)
        if self.owns_k:
            self.wk = Dense(cfg.head_dim, cfg.index_head_dim, num, device)
            self.k_norm = RMSNorm(cfg.index_head_dim, cfg.norm_eps, num, device)

    @torch.no_grad()
    def keys(self, latent, cos, sin):  # [b, n, head_dim] before RoPE -> [b, n, index_head_dim]
        k = self.k_norm(F.linear(latent, self.wk.weight.to(latent.dtype)))
        return self.num.fp4(_rope(k, cos, sin, self.num))

    @torch.no_grad()
    def scores(self, x, qr, index_k, cos, sin) -> torch.Tensor:
        """[b, s, n] in the activation dtype (as the reference scores them), -inf where a query can't see yet."""
        cfg, num = self.cfg, self.num
        b, s, _ = x.shape
        q = num.fp4(_rope(self.wq_b(qr).unflatten(-1, (cfg.index_n_heads, cfg.index_head_dim)), cos, sin, num))
        w = F.linear(x, self.weights_proj.weight.to(x.dtype)) * (cfg.index_head_dim ** -0.5 * cfg.index_n_heads ** -0.5)
        out = []
        for t0 in range(0, s, self.QUERY_CHUNK):
            t1 = min(s, t0 + self.QUERY_CHUNK)
            sc = torch.einsum("bshd,btd->bsht", q[:, t0:t1], index_k)
            out.append((sc.relu() * w[:, t0:t1].unsqueeze(-1)).sum(dim=2))
        visible = (torch.arange(1, s + 1, device=x.device) // self.ratio).unsqueeze(-1)  # a token, once whole
        return torch.cat(out, 1).masked_fill(torch.arange(index_k.size(1), device=x.device) >= visible, -math.inf)

    @torch.no_grad()
    def forward(self, x, qr, index_k, cos, sin, choices: Choices, picks: torch.Tensor | None) -> torch.Tensor:
        cfg, s = self.cfg, x.size(1)
        score = self.scores(x, qr, index_k, cos, sin)
        visible = (torch.arange(1, s + 1, device=x.device) // self.ratio).unsqueeze(-1)
        if self.layer == cfg.candidate_source_layer:
            if choices.candidates is None:
                choices.candidates = candidate_blocks(score, visible, cfg.candidate_topk_blocks,
                                                      cfg.candidate_block_size)
        elif 0 <= cfg.candidate_source_layer < self.layer:
            if choices.candidates is not None:
                score = score.masked_fill(~choices.candidates.to(score.device), -math.inf)
            elif score.size(-1) > cfg.candidate_topk_blocks * cfg.candidate_block_size:
                raise ValueError(f"layer {self.layer} picks inside layer {cfg.candidate_source_layer}'s candidate "
                                 "blocks: run that layer first, or pass its blocks in Choices.candidates")
            # else: no more blocks than the source keeps, so it keeps every one a query can reach: nothing to mask
        if choices.keep_scores:
            choices.scores[("picks", self.layer)] = score
        if picks is not None:
            return picks
        idx = score.topk(min(cfg.index_topk, score.size(-1)), dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idx < visible, idx, -1)


KERNEL_BLOCK = 64  # keys the reference's sparse_attn kernel takes at a time


def sparse_attention(q, kv, comp, picks, sink, scale: float, window: int, num: Numerics, chunk: int = 128):
    """Each query's softmax over its keys plus a sink, weighting the same rows as values (one shared kv head).
    Query t sees the window kv[t-window+1 .. t] and the compressed comp[picks[t]] (-1: none, picks ascending with
    the -1s last, as the indexer makes them); sink[h] adds exp(sink[h]) to head h's denominator only.

    Rounded as the reference's sparse_attn kernel rounds: scores and sums in float32, and the probabilities cast to
    the activation dtype before they weight the values, each relative to the running maximum of the kernel's online
    softmax. That kernel walks a query's index list (its window, oldest first, then its picks) 64 keys at a time, so
    a key's probability is rounded as exp(score - the max over the blocks up to its own), then rescaled. Rounding
    against the final max instead differs by under a bf16 step, but the FP8 rounding of the output that comes next
    turns that into whole FP8 steps for about 1% of values. Queries go `chunk` at a time, and with autograd each chunk
    is recomputed in the backward rather than kept."""
    s, out = q.size(1), []
    for t0 in range(0, s, chunk):
        t1 = min(s, t0 + chunk)
        args = (q[:, t0:t1], kv[:, max(0, t0 - window + 1):t1], comp, None if picks is None else picks[:, t0:t1], sink)
        fn = partial(_attend, t0=t0, window=window, n_window=min(s, window), scale=scale, num=num)
        if torch.is_grad_enabled() and any(a is not None and a.requires_grad for a in args):
            out.append(checkpoint(fn, *args, use_reentrant=False))
        else:
            out.append(fn(*args))
    return torch.cat(out, 1)


def _attend(q, keys, comp, picks, sink, *, t0, window, n_window, scale, num):
    hi, dev = num.hi, q.device
    b, c, h, _ = q.shape
    qh, kh = q.to(hi), keys.to(hi)
    tq = torch.arange(t0, t0 + c, device=dev)[:, None]
    tk = torch.arange(max(0, t0 - window + 1), t0 + c, device=dev)
    # Where each key sits in its query's index list, and so which of the kernel's blocks it's in.
    slot = [tk - (tq - window + 1).clamp_min(0)]  # [c, keys]: the window list starts at max(0, t - window + 1)
    s = torch.einsum("bchd,bjd->bchj", qh, kh) * scale
    parts = [s.masked_fill(((tk > tq) | (tk <= tq - window))[None, :, None, :], -math.inf)]
    if picks is not None and picks.size(-1):
        sel = comp[torch.arange(b, device=dev)[:, None, None], picks.clamp_min(0)].to(hi)  # [b, c, k, d]
        sc = torch.einsum("bchd,bckd->bchk", qh, sel) * scale
        parts.append(sc.masked_fill((picks < 0)[:, :, None, :], -math.inf))
        slot.append((n_window + torch.arange(picks.size(-1), device=dev)).expand(c, -1))
    s = torch.cat(parts, -1)
    block = (torch.cat(slot, -1).clamp_min(0) // KERNEL_BLOCK)[None, :, None].expand(b, c, h, -1)
    with torch.no_grad():
        sd = s.detach()
        n = int(block.max()) + 1
        best = torch.stack([sd.masked_fill(block != i, -math.inf).amax(-1) for i in range(n)], -1)
        running = best.cummax(-1).values.clamp_min(-1e30)  # the kernel's floor: a query seeing nothing gets 0
        m_key, m = running.gather(-1, block), running[..., -1:]
    p = torch.exp(s - m_key)  # each key against its block's running max, as the kernel computes it
    rescale = torch.exp(m_key - m)
    denom = (p * rescale).sum(-1) + torch.exp(sink.to(hi) - m.squeeze(-1))
    p = p.to(num.dtype).to(hi) * rescale
    o = torch.einsum("bchj,bjd->bchd", p[..., :kh.size(1)], kh)
    if len(parts) > 1:
        o = o + torch.einsum("bchk,bckd->bchd", p[..., kh.size(1):], sel)
    return (o / denom.unsqueeze(-1)).to(num.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: Config, layer: int, num: Numerics, device=None):
        super().__init__()
        self.cfg, self.layer, self.num, self.ratio = cfg, layer, num, cfg.ratio(layer)
        hd = cfg.n_heads * cfg.head_dim
        self.attn_sink = _param(num, (cfg.n_heads,), torch.float32, device)
        self.wq_a = QLinear(cfg.dim, cfg.q_lora_rank, num, device)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps, num, device)
        self.wq_b = QLinear(cfg.q_lora_rank, hd, num, device)
        self.wkv = QLinear(cfg.dim, cfg.head_dim, num, device)
        self.kv_norm = RMSNorm(cfg.head_dim, cfg.norm_eps, num, device)
        self.wo_a = QLinear(hd // cfg.o_groups, cfg.o_groups * cfg.o_lora_rank, num, device, groups=cfg.o_groups)
        self.wo_b = QLinear(cfg.o_groups * cfg.o_lora_rank, cfg.dim, num, device)
        self.compressor = Compressor(cfg, layer, num, device) if layer in cfg.kv_source_layers else None
        self.indexer = Indexer(cfg, layer, num, device) if layer in cfg.index_source_layers else None

    def forward(self, x, side: Side, choices: Choices):  # x [b, s, dim], normed
        cfg, num = self.cfg, self.num
        b, s, _ = x.shape
        cos, sin = _rope_table(s, *cfg.rope_args(self.layer))
        qr = self.q_norm(self.wq_a(x))
        q = _rope(self.wq_b(qr).unflatten(-1, (cfg.n_heads, cfg.head_dim)), cos, sin, num)
        kv = num.fp8(_rope(self.kv_norm(self.wkv(x)), cos, sin, num))  # rounded whole, RoPE tail included
        comp = picks = None
        if self.ratio:
            r = self.ratio
            ccos, csin = cos[:s - s % r:r], sin[:s - s % r:r]  # a compressed token sits at its group's first position
            if self.compressor is not None:
                latent = self.compressor(x)
                if self.indexer is not None:
                    side.index_k = self.indexer.keys(latent, ccos, csin)
                side.compress_kv = num.fp4(_rope(latent, ccos, csin, num), 16, e4m3_scale=True)
            comp = self._side(side, "compress_kv", (b, s // r)).to(x.dtype)  # FP4 values: exact in any dtype
            if self.indexer is not None:
                picks = choices.picks.get(self.layer)
                n = comp.size(1)
                uses = 0 <= cfg.candidate_source_layer < self.layer
                if picks is not None:
                    picks = picks.to(x.device, torch.long)
                    check_picks(picks, b, s, n, min(cfg.index_topk, n), r, choices.candidates if uses else None)
                if n == 0:
                    picks = x.new_empty(b, s, 0, dtype=torch.long)
                elif picks is None or choices.keep_scores or (self.layer == cfg.candidate_source_layer
                                                              and choices.candidates is None):
                    index_k = self._side(side, "index_k", (b, n)).to(x.dtype)
                    picks = self.indexer(x, qr, index_k, cos, sin, choices, picks)
                choices.picks[self.layer] = side.picks = picks
            picks = self._side(side, "picks", (b, s))
        o = sparse_attention(q, kv, comp, picks, self.attn_sink, cfg.head_dim ** -0.5, cfg.window_size, num)
        o = _rope(o, cos, sin, num, inverse=True)  # takes the query's rotation back out of the values' RoPE tails
        o = o.flatten(2).unflatten(-1, (cfg.o_groups, -1))  # o_groups groups of n_heads / o_groups heads each
        return self.wo_b(self.wo_a(o).flatten(2))

    def _side(self, side: Side, name: str, lead: tuple):
        """A side channel this layer reuses, refused if it's missing or not for this sequence: a shard that starts
        after a source layer must be handed what that layer made."""
        t = getattr(side, name)
        if t is None or tuple(t.shape[:2]) != lead:
            kind = "kv" if name in ("compress_kv", "index_k") else "index"
            src = self.cfg.source(self.layer, kind)
            raise ValueError(f"layer {self.layer} reuses layer {src}'s {name} ({list(lead)} first), and "
                             f"Side.{name} is {None if t is None else list(t.shape)}: pass it in from layer {src}")
        return t


class Expert(nn.Module):
    """One SwiGLU feed-forward: w2(silu(w1 x) · w3 x), the up branch (w3) clamped to ±limit and the gate branch (w1)
    clamped below +limit when limit > 0, as in training, where they keep FP8/FP4 activations in range."""

    def __init__(self, dim: int, inter: int, limit: float, num: Numerics, device=None, fp4: bool = False):
        super().__init__()
        self.limit, self.num = limit, num
        self.w1 = QLinear(dim, inter, num, device, fp4)
        self.w2 = QLinear(inter, dim, num, device, fp4)
        self.w3 = QLinear(dim, inter, num, device, fp4)

    def forward(self, x, gate=None):
        xq, hi = self.num.fp8(x), self.num.hi
        g, u = self.w1.adapt(self.w1.matmul(xq), x).to(hi), self.w3.adapt(self.w3.matmul(xq), x).to(hi)
        if self.limit > 0:
            u, g = u.clamp(-self.limit, self.limit), g.clamp(max=self.limit)
        h = F.silu(g) * u
        if gate is not None:
            h = gate * h
        return self.w2(h.to(x.dtype))


class Gate(nn.Module):
    def __init__(self, cfg: Config, num: Numerics, device=None):
        super().__init__()
        self.weight = _param(num, (cfg.n_routed_experts, cfg.dim), torch.bfloat16, device)
        # Steers selection only, and is set by load balancing, not by gradient: a buffer.
        self.register_buffer("bias", torch.empty(cfg.n_routed_experts, dtype=torch.float32, device=device))


class MoE(nn.Module):
    def __init__(self, cfg: Config, layer: int, num: Numerics, device=None):
        super().__init__()
        self.cfg, self.layer, self.num = cfg, layer, num
        self.gate = Gate(cfg, num, device)
        self.experts = nn.ModuleList(Expert(cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit, num, device, fp4=True)
                                     for _ in range(cfg.n_routed_experts))
        self.shared_experts = Expert(cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit, num, device)

    def scores(self, x):  # [tokens, dim] -> [tokens, experts], in num.hi
        cfg, hi = self.cfg, self.num.hi
        s = F.linear(x.to(hi), self.gate.weight.to(hi)) / cfg.gate_temp
        if cfg.score_func == "softmax":
            return s.softmax(dim=-1)
        return s.sigmoid() if cfg.score_func == "sigmoid" else F.softplus(s).sqrt()

    def forward(self, x, choices: Choices):
        cfg, hi, k = self.cfg, self.num.hi, self.cfg.n_activated_experts
        shape = x.shape
        x = x.reshape(-1, cfg.dim)
        scores = self.scores(x)
        chosen = choices.experts.get(self.layer)
        if chosen is not None:
            chosen = chosen.to(x.device, torch.long)
            check_experts(chosen, x.size(0), cfg)
        if chosen is None or choices.keep_scores:
            with torch.no_grad():
                biased = scores + self.gate.bias.to(hi)
                if choices.keep_scores:
                    choices.scores[("experts", self.layer)] = biased
                if chosen is None:
                    chosen = choices.experts[self.layer] = biased.topk(k, dim=-1).indices
        gates = scores.gather(1, chosen)
        if cfg.norm_topk_prob and k > 1:
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-20)  # 1e-20, not norm_eps: as in training
        gates = gates * cfg.route_scale
        y = torch.zeros_like(x, dtype=hi)
        flat = chosen.flatten()
        order = torch.argsort(flat, stable=True)  # by expert, then by token, as torch.where(indices == e) lists them
        start = 0
        for e, n in enumerate(torch.bincount(flat, minlength=cfg.n_routed_experts).tolist()):
            if n:
                tok, slot = order[start:start + n] // k, order[start:start + n] % k
                y.index_add_(0, tok, self.experts[e](x[tok], gates[tok, slot, None]).to(hi))
                start += n
        y = y + self.shared_experts(x).to(hi)
        return y.to(x.dtype).view(shape)


# --- engram ----------------------------------------------------------------------------------------------------------

def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin: these bases are exact for every n below 3.3e24."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d, r = d // 2, r + 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


@dataclass(frozen=True)
class EngramLayout:
    """Where n-grams land in the tables (the reference's EngramLayout and compute_hash_multipliers). Every (engram
    layer, n-gram size, head) has its own range of rows, a prime number of them, drawn in order and never reused."""
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [engram layer][n-gram size - 2][head]
    multipliers: tuple[tuple[int, ...], ...]  # [engram layer][look-back]: odd, small enough not to overflow int64

    @classmethod
    @lru_cache(8)
    def of(cls, cfg: Config) -> "EngramLayout":
        primes, seen = [], set()
        for _ in cfg.engram_layer_ids:
            per_size = []
            for _ in range(cfg.engram_max_ngram_size - 1):
                sizes, p = [], cfg.engram_vocab_size - 1
                for _ in range(cfg.engram_n_heads):
                    p += 1
                    while not _is_prime(p) or p in seen:
                        p += 1
                    seen.add(p)
                    sizes.append(p)
                per_size.append(tuple(sizes))
            primes.append(tuple(per_size))
        bound = max(1, (np.iinfo(np.int64).max // cfg.engram_compressed_vocab_size) // 2)
        mults = tuple(tuple(int(v) * 2 + 1 for v in np.random.default_rng(10007 * layer).integers(
            0, bound, size=(cfg.engram_max_ngram_size,), dtype=np.int64)) for layer in cfg.engram_layer_ids)
        return cls(tuple(primes), mults)

    def rows(self, i: int) -> int:
        """Rows engram layer i's table has (config.json's engram_num_embeddings)."""
        return sum(sum(per_size) for per_size in self.primes[i])


def engram_hashes(tokens: torch.Tensor, token_map: torch.Tensor, cfg: Config) -> torch.Tensor:
    """[b, s, engram layers, (N-1) * heads] table rows each token reads: the reference's NgramHashState, for whole
    sequences. token_map maps token ids to their normalised ids (engram.py's build_compressed_token_map); position t's
    n-gram is the normalised tokens t-n+1..t, padded with engram_pad_id's before the start, its hash an XOR of each
    token times its look-back's multiplier, then taken modulo its (size, head)'s prime and offset into the table.
    A pure function of the text."""
    layout = EngramLayout.of(cfg)
    n, dev = cfg.engram_max_ngram_size, tokens.device
    ids = token_map.to(dev)[tokens]
    pad = token_map[cfg.engram_pad_id].to(dev)
    s = ids.size(1)
    back = [ids] + [torch.cat((pad.expand(ids.size(0), min(k, s)), ids[:, :max(s - k, 0)]), 1) for k in range(1, n)]
    prods = torch.stack(back, -1).unsqueeze(2) * torch.tensor(layout.multipliers, device=dev)  # [b, s, layers, n]
    primes = torch.tensor(layout.primes, device=dev)  # [layers, n - 1, heads]
    offsets = torch.tensor([np.cumsum([0, *[p for size in layer for p in size][:-1]]).tolist()
                            for layer in layout.primes], device=dev)
    rolling, out = prods[..., 0], []
    for i in range(1, n):
        rolling = rolling ^ prods[..., i]
        out.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(out, -1) + offsets


class _Table(nn.Module):
    """An engram table as stored: FP8 rows with a ue8m0 scale per 32 values."""

    def __init__(self, rows: int, width: int, device=None):
        super().__init__()
        self.register_buffer("weight", torch.empty(rows, width, dtype=torch.float8_e4m3fn, device=device))
        self.register_buffer("scale", torch.empty(rows, width // BLOCK, dtype=torch.float8_e8m0fnu, device=device))


class Engram(nn.Module):
    """Mixes the table rows a token's n-grams hash to into the stream, through a gate per stream copy: a normalised
    dot product of that copy with a key made from the rows, through a signed square root and a sigmoid."""

    def __init__(self, cfg: Config, layer: int, num: Numerics, device=None, table: bool = True):
        super().__init__()
        self.cfg, self.num, self.index = cfg, num, cfg.engram_layer_ids.index(layer)
        cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
        # The real tables are 101 GB each: a worker holds the rows its job's text reads, and passes them in.
        self.embed = _Table(cfg.engram_num_embeddings[self.index], cfg.engram_head_dim, device) if table else None
        self.wkv = QLinear(cols * cfg.engram_head_dim, cfg.dim * (cfg.hc_mult + 1), num, device)
        self.q_weight = _param(num, (cfg.hc_mult, cfg.dim), torch.bfloat16, device)
        self.k_weight = _param(num, (cfg.hc_mult, cfg.dim), torch.bfloat16, device)

    def rows(self, hashes):  # [b, s, cols] -> [b, s, cols, head_dim], exactly dequantised
        v = F.embedding(hashes, self.embed.weight).float().unflatten(-1, (-1, BLOCK))
        return (v * F.embedding(hashes, self.embed.scale).float().unsqueeze(-1)).flatten(-2).to(self.num.dtype)

    def forward(self, x, hashes=None, rows=None):  # x [b, s, hc, dim]
        cfg, hi = self.cfg, self.num.hi
        rows = self.rows(hashes) if rows is None else rows.to(self.num.dtype)
        key, value = self.wkv(rows.flatten(-2)).split([cfg.hc_mult * cfg.dim, cfg.dim], dim=-1)
        key = key.to(hi).unflatten(-1, (cfg.hc_mult, cfg.dim))
        h = x.to(hi)
        # normalised per (token, stream copy) over dim, not over the copies together
        rstd = torch.rsqrt(h.square().mean(-1) + cfg.norm_eps) * torch.rsqrt(key.square().mean(-1) + cfg.norm_eps)
        dot = (h * (self.q_weight.to(hi) * self.k_weight.to(hi)) * key).sum(-1) * rstd * cfg.dim ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return (h + gate.unsqueeze(-1) * value.to(hi).unsqueeze(-2)).to(x.dtype)


# --- a layer and the model -------------------------------------------------------------------------------------------

def _sinkhorn_split(mixes, scale, base, hc: int, iters: int, eps: float):
    """The reference's hc_split_sinkhorn kernel: mixes [..., (2 + hc) * hc] into pre [..., hc], post [..., hc] and a
    doubly stochastic comb [..., hc, hc] (rows normalised by a softmax, then columns, rows, columns ... by Sinkhorn)."""
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * scale[2] + base[2 * hc:]).unflatten(-1, (hc, hc)).softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(x, pre, hi):
    """Collapse the stream's copies into one: [b, s, hc, dim] weighted by pre [b, s, hc] -> [b, s, dim]."""
    return torch.sum(pre.unsqueeze(-1) * x.to(hi), dim=2).to(x.dtype)


def hc_post(y, residual, post, comb, hi):
    """Spread a sublayer's output y [b, s, dim] back over the copies and mix the residual's in:
    out_k = post_k · y + Σ_j comb_jk · residual_j, [b, s, hc, dim]."""
    mixed = comb[..., 0, :, None] * residual[..., 0:1, :].to(hi)
    for j in range(1, residual.size(2)):
        mixed = mixed + comb[..., j, :, None] * residual[..., j:j + 1, :].to(hi)
    return (post.unsqueeze(-1) * y.unsqueeze(-2).to(hi) + mixed).to(y.dtype)


class Block(nn.Module):
    """One layer, its residual stream hc_mult copies wide (hyper-connections): see the module's docstring."""

    def __init__(self, cfg: Config, layer: int, num: Numerics = Numerics(), device=None, engram_table: bool = True):
        super().__init__()
        self.cfg, self.layer, self.num = cfg, layer, num
        self.attn = Attention(cfg, layer, num, device)
        self.ffn = MoE(cfg, layer, num, device)
        self.engram = Engram(cfg, layer, num, device, engram_table) if layer in cfg.engram_layer_ids else None
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps, num, device)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps, num, device)
        mix, width = (2 + cfg.hc_mult) * cfg.hc_mult, cfg.hc_mult * cfg.dim
        for part in ("attn", "ffn"):
            setattr(self, f"hc_{part}_fn", _param(num, (mix, width), torch.float32, device))
            setattr(self, f"hc_{part}_base", _param(num, (mix,), torch.float32, device))
            setattr(self, f"hc_{part}_scale", _param(num, (3,), torch.float32, device))

    def hc_mixes(self, x, part: str):
        cfg, hi = self.cfg, self.num.hi
        h = x.flatten(2).to(hi)  # normalised over the whole flattened stream: one statistic per token
        mixes = F.linear(h, getattr(self, f"hc_{part}_fn").to(hi)) * torch.rsqrt(h.square().mean(-1, keepdim=True)
                                                                                  + cfg.norm_eps)
        return _sinkhorn_split(mixes, getattr(self, f"hc_{part}_scale").to(hi), getattr(self, f"hc_{part}_base").to(hi),
                               cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps)

    def forward(self, x, pre_mix, side: Side, choices: Choices, hashes=None, rows=None):
        """x [b, s, hc, dim] and pre_mix [b, s, hc] (the previous layer's) -> this layer's (x, pre_mix). `hashes`
        ([b, s, cols]: this layer's slice of engram_hashes) or `rows` (their table rows) feed an engram layer. A
        stream in another dtype (from layers with other Numerics) is cast to this layer's first."""
        x, pre_mix = x.to(self.num.dtype), pre_mix.to(self.num.hi)
        if self.engram is not None:
            x = self.engram(x, hashes, rows)
        hi, residual = self.num.hi, x
        attn_pre, post, comb = self.hc_mixes(x, "attn")
        h = self.attn(self.attn_norm(hc_pre(x, pre_mix, hi)), side, choices)
        x = hc_post(h, residual, post, comb, hi)
        residual = x
        ffn_pre, post, comb = self.hc_mixes(x, "ffn")
        h = self.ffn(self.ffn_norm(hc_pre(x, attn_pre, hi)), choices)
        return hc_post(h, residual, post, comb, hi), ffn_pre


def _cross_entropy(h, t, w):
    logits = F.linear(h.to(w.dtype), w)
    return (torch.logsumexp(logits, -1) - logits.gather(1, t[:, None])[:, 0]).sum()


class Model(nn.Module):
    """The embedding, a run of Blocks, and the head: the whole model when `layers` is all of them, a pipeline shard
    otherwise (with the embedding only if it's the first, the head only if it's the last). A shard must be given what
    the layers before it hand on: the stream and pre_mix, the Side of the last kv and index sources before it (its
    layers refuse to run without what they reuse), the Choices of the candidate source if a layer picks inside its
    blocks, and the engram hashes. State dict names are the checkpoint's (layers.2.attn.wq_a.weight ...), so
    DeepSeek's safetensors load as they are (load_weights)."""

    def __init__(self, cfg: Config, layers=None, num: Numerics = Numerics(), device=None, embed: bool = True,
                 head: bool = True, engram_table: bool = True, layer_num: dict | None = None,
                 head_num: Numerics | None = None):
        """layer_num: {layer: Numerics} for layers computed otherwise than `num` (say, trained layers in float32 above
        frozen ones in the reference's arithmetic); head_num likewise for the final norm and the head."""
        super().__init__()
        self.cfg, self.num, self.head_num = cfg, num, head_num or num
        layers = range(cfg.n_layers) if layers is None else layers
        layer_num = layer_num or {}
        self.layers = nn.ModuleDict({str(i): Block(cfg, i, layer_num.get(i, num), device, engram_table) for i in layers})
        if embed:
            self.embed = Dense(cfg.dim, cfg.vocab_size, num, device)
        if head:
            self.norm = RMSNorm(cfg.dim, cfg.norm_eps, self.head_num, device)
            self.head = Dense(cfg.dim, cfg.vocab_size, self.head_num, device)

    def load_weights(self, state: dict, assign: bool = False):
        """Load checkpoint tensors (DeepSeek's names and formats), all of them: anything missing is an error. Tensors
        only images use are dropped. assign=True takes the tensors themselves, for a model built on the meta device
        (no second copy of 7.4 GB)."""
        state = {k: v for k, v in state.items() if not k.endswith(IMAGE_ONLY)}
        if any(p.is_meta for p in self.state_dict().values()):
            assign = True
        self.load_state_dict(state, strict=True, assign=assign)
        return self

    def start(self, tokens):
        """Token ids [b, s] -> the stream's first state [b, s, hc, dim] and pre_mix [b, s, hc] (all on copy 0)."""
        h = F.embedding(tokens, self.embed.weight).to(self.num.dtype)
        x = h.unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)
        pre = torch.zeros(*tokens.shape, self.cfg.hc_mult, dtype=self.num.hi, device=tokens.device)
        pre[..., 0] = 1
        return x, pre

    def logits(self, x, pre_mix):
        """The last layer's stream -> logits [b, s, vocab] in num.hi, every position."""
        hi = self.head_num.hi
        h = self.norm(hc_pre(x.to(self.head_num.dtype), pre_mix.to(hi), hi))
        return F.linear(h.to(hi), self.head.weight.to(hi))

    def loss(self, x, pre_mix, targets, mask, chunk: int = HEAD_CHUNK):
        """(Σ cross-entropy, count) of targets [b, s] at the positions mask [b, s] picks, from the last layer's
        stream. The head is made `chunk` positions at a time, and with autograd each chunk's logits are recomputed in
        the backward rather than kept: 4096 positions of a 129,280-word vocabulary would be 2 GB."""
        hi = self.head_num.hi
        h = self.norm(hc_pre(x.to(self.head_num.dtype), pre_mix.to(hi), hi))[mask]
        t = targets[mask].to(h.device, torch.long)
        w = self.head.weight.to(hi)
        total = h.new_zeros((), dtype=hi)
        for s0 in range(0, h.size(0), chunk):
            part = (h[s0:s0 + chunk], t[s0:s0 + chunk], w)
            total = total + (checkpoint(_cross_entropy, *part, use_reentrant=False) if torch.is_grad_enabled()
                             and h.requires_grad else _cross_entropy(*part))
        return total, int(t.numel())

    def lora_modules(self, targets=LORA_TARGETS) -> dict[str, "QLinear"]:
        """{module path: QLinear} of the projections LoRA may adapt in this model's layers, in layer and target order.
        Paths are the checkpoint's (layers.36.attn.wq_a), which is what an adapter file names them by."""
        out = {}
        for i, block in self.layers.items():
            for t in targets:
                out[f"layers.{i}.{LORA_MODULES[t]}"] = block.get_submodule(LORA_MODULES[t])
        return out


    def run(self, x, pre_mix, side: Side, choices: Choices, hashes=None):
        """This model's layers, in order. hashes: engram_hashes(...) for the whole model's engram layers."""
        for i, block in self.layers.items():
            hs = hashes[:, :, self.cfg.engram_layer_ids.index(int(i))] if block.engram is not None else None
            x, pre_mix = block(x, pre_mix, side, choices, hs)
        return x, pre_mix

    def forward(self, tokens, choices: Choices | None = None, token_map=None):
        """Token ids [b, s] -> (logits [b, s, vocab], the Choices made). token_map: see engram_hashes."""
        choices = Choices() if choices is None else choices
        hashes = engram_hashes(tokens, token_map, self.cfg) if self.cfg.engram_layer_ids else None
        x, pre = self.run(*self.start(tokens), Side(), choices, hashes)
        return self.logits(x, pre), choices


# --- random weights for tests, and DeepSeek's own -------------------------------------------------------------------

def random_weights(cfg: Config, seed: int = 0, layers=None, embed: bool = True, head: bool = True) -> dict:
    """A checkpoint in DeepSeek's stored formats and names, with random values: Gaussians quantised exactly as
    DeepSeek's are (FP8 per 32x32 block, FP4 experts per 32 inputs, engram rows per 32), scaled so that a forward
    stays in range and its choices aren't degenerate."""
    g = torch.Generator().manual_seed(seed)
    meta = Model(cfg, layers, device="meta", embed=embed, head=head)

    def normal(shape, std=1.0):
        return torch.randn(shape, generator=g) * std

    out = {}
    for name, t in meta.state_dict().items():
        shape, base = tuple(t.shape), name.rsplit(".", 1)[0]
        if name in out:  # a scale, made with its weight
            continue
        if name.endswith(".embed.weight"):  # an engram table
            v = normal(shape).unflatten(-1, (-1, BLOCK))
            s = _pow2_ceil(v.abs().amax(-1, keepdim=True) * _inv(FP8_MAX, "cpu"))
            out[name] = (v / s).to(torch.float8_e4m3fn).flatten(-2)
            out[base + ".scale"] = s.squeeze(-1).to(torch.float8_e8m0fnu)
        elif t.dtype == torch.int8:
            out[name], out[base + ".scale"] = quantize_fp4(normal((shape[0], 2 * shape[1]), (2 * shape[1]) ** -0.5))
        elif t.dtype == torch.float8_e4m3fn:
            out[name], out[base + ".scale"] = quantize_fp8(normal(shape, shape[1] ** -0.5))
        elif name.endswith(("norm.weight", "q_weight", "k_weight")):
            out[name] = (1 + normal(shape, 0.1)).to(t.dtype)
        elif name.endswith("attn_sink"):
            out[name] = normal(shape).to(t.dtype)
        elif "_fn" in name:  # hyper-connection projections: the mixes come out about N(0, 1)
            out[name] = normal(shape, shape[1] ** -0.5).to(t.dtype)
        elif name.endswith("_base"):
            out[name] = normal(shape, 0.5).to(t.dtype)
        elif name.endswith("_scale"):
            out[name] = (0.5 + torch.rand(shape, generator=g)).to(t.dtype)
        elif name.endswith("gate.bias"):
            out[name] = normal(shape, 0.1).to(t.dtype)
        elif name == "embed.weight":
            out[name] = normal(shape).to(t.dtype)
        else:  # bf16 matrices: the router, compressor, indexer, head
            out[name] = normal(shape, shape[1] ** -0.5).to(t.dtype)
    return out


SAFETENSORS_DTYPES = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "I8": torch.int8,
                      "BF16": torch.bfloat16, "F32": torch.float32}


def read_safetensors(path: str | Path, names=None, device="cpu") -> dict:
    """Tensors of a local safetensors file (all, or those in `names`), as stored. The header is checked as
    hub.parse_header checks any downloaded one."""
    from .hub import parse_header
    path = Path(path)
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = parse_header(f.read(n), path.stat().st_size, str(path))
        out = {}
        for name, (dtype, shape, begin, end) in header.tensors.items():
            if names is not None and name not in names:
                continue
            if dtype not in SAFETENSORS_DTYPES:
                raise ValueError(f"{path.name}: tensor {name} is {dtype}, which DeepSeek-V4.1 doesn't use")
            f.seek(header.data_start + begin)
            raw = torch.frombuffer(bytearray(f.read(end - begin)), dtype=torch.uint8)
            out[name] = raw.view(SAFETENSORS_DTYPES[dtype]).reshape(shape).to(device)
    return out


def cache_dir(commit: str = COMMIT) -> Path:
    """Where whole weight files go: ~/.vibehash/cache/deepseek/<commit>, never the working directory."""
    return settings.home() / "cache" / "deepseek" / commit


PIECE = 64 * 2**20  # a weight file downloads in pieces this big, several at once
FETCHERS = 6


def fetch_file(path: str, hub=None, repo: str = REPO, commit: str = COMMIT, log=None, workers: int = FETCHERS) -> Path:
    """One weight file of the pinned commit, downloaded whole into cache_dir() and checked against the sha256 the Hub
    publishes for it. It comes in 64 MiB ranged reads over `workers` connections at once: the Hub's CDN slows a long
    single stream (measured: a fresh connection 30-37 MiB/s, one that had been running two minutes 1.5 MiB/s), and
    each piece is a fresh request. A dropped connection resumes within its piece, and a restart skips the pieces
    already written (listed in a .done file beside the .part). Refuses to start without room for it on the disk."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from .hub import Hub, IntegrityError, _FileCheck
    hub = hub or Hub()
    info = hub.list_files(repo, commit)[path]
    folder = cache_dir(commit)
    folder.mkdir(parents=True, exist_ok=True)
    dest, part, listed = folder / path, folder / (path + ".part"), folder / (path + ".done")
    if dest.exists() and dest.stat().st_size == info.size:
        return dest
    n = -(-info.size // PIECE)
    done = set(json.loads(listed.read_text())) if listed.exists() and part.exists() else set()
    if shutil.disk_usage(folder).free < (n - len(done)) * PIECE + 2**30:
        raise OSError(f"{path} needs {(n - len(done)) * PIECE / 1e9:.1f} GB in {folder}, and the disk hasn't that free")
    if not part.exists() or part.stat().st_size != info.size:
        with open(part, "wb") as f:
            f.truncate(info.size)
        done = set()
    lock, local = threading.Lock(), threading.local()

    def get(i):
        if not hasattr(local, "remote"):  # one per connection: it keeps the CDN address the Hub redirected it to
            local.remote = hub.open(repo, commit, path, info.size, info.lfs_sha256)
        start = i * PIECE
        with open(part, "r+b") as f:
            f.seek(start)
            for piece in local.remote.iter_range(start, min(PIECE, info.size - start)):
                f.write(piece)
        with lock:
            done.add(i)
            listed.write_text(json.dumps(sorted(done)))
            if log and len(done) * 20 // n != (len(done) - 1) * 20 // n:
                log(f"{path}: {min(len(done) * PIECE, info.size) / 1e9:.1f} of {info.size / 1e9:.1f} GB")

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(get, [i for i in range(n) if i not in done]))
    check = _FileCheck(info)
    with open(part, "rb") as f:
        for block in iter(lambda: f.read(2**24), b""):
            check.update(block)
    try:
        check.finish()
    except IntegrityError:
        part.unlink()
        listed.unlink(missing_ok=True)
        raise
    os.replace(part, dest)
    listed.unlink(missing_ok=True)
    return dest


def layer_files(layer: int, hub=None, repo: str = REPO, commit: str = COMMIT) -> dict[str, list[str]]:
    """{weight file: the names of layer `layer`'s tensors in it}, from the Hub's index."""
    from .hub import Hub
    out: dict[str, list[str]] = {}
    for name, file in (hub or Hub()).weight_map(repo, commit).items():
        if name.startswith(f"layers.{layer}."):
            out.setdefault(file, []).append(name)
    return out


def load_layer(layer: int, device="cpu", hub=None, log=None) -> dict:
    """Layer `layer`'s tensors from DeepSeek's checkpoint, fetching the files that hold them (each is one whole layer,
    7.4 GB) unless cached. Layers 1 and 14 also have an engram table in a 101 GB file: not fetched here (per-job
    slices of it are DEEPSEEK.md's step 4)."""
    out = {}
    for file, names in layer_files(layer, hub).items():
        if any(".engram.embed." in n for n in names):
            raise ValueError(f"layer {layer}'s engram table is in {file} (101 GB); load_layer doesn't fetch tables")
        out.update(read_safetensors(fetch_file(file, hub, log=log), set(names), device))
    return out


def load_embedding(device="cpu", hub=None, log=None) -> torch.Tensor:
    """The token embedding, [vocab, dim] bf16 (1.3 GB, a file of its own)."""
    from .hub import Hub
    file = (hub or Hub()).weight_map(REPO, COMMIT)["embed.weight"]
    return read_safetensors(fetch_file(file, hub, log=log), {"embed.weight"}, device)["embed.weight"]


def describe_layer(cfg: Config, layer: int) -> str:
    """What a layer does, in a few words (for logs and benchmarks)."""
    parts = [f"layer {layer}"]
    r = cfg.ratio(layer)
    parts.append(f"compress ratio {r}" if r else "window only")
    if layer in cfg.kv_source_layers:
        parts.append("kv source")
    if layer in cfg.index_source_layers:
        parts.append("index source")
    if layer == cfg.candidate_source_layer:
        parts.append("candidate source")
    if layer in cfg.engram_layer_ids:
        parts.append("engram")
    return ", ".join(parts)

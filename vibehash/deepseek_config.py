"""DeepSeek-V4.1-Flash's shape, without PyTorch: the config, where the engram tables' rows fall, and what crosses a cut
between frozen and trained layers. A node reads a DeepSeek job's spec and plans its work graph from these alone (it
runs no model); deepseek.py and deepseek_top.py re-export them.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache

import numpy as np

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
COMMIT = "314113f8a7dadb388215396ba5fbee2c610a04d6"  # the weights and reference code this was checked against

BLOCK = 32  # one scale per 32x32 block of an FP8 matrix, per 32 inputs of an FP4 one, per 32 activations
SCORE_FUNCS = ("softmax", "sigmoid", "sqrtsoftplus")
# What LoRA can adapt, by short name: attention's four plain projections and the shared expert's three. Not wo_a: it's
# block-diagonal (o_groups groups), which PEFT's LoRA can't express. Not the routed experts: each token uses 6 of 384.
LORA_MODULES = {"wq_a": "attn.wq_a", "wq_b": "attn.wq_b", "wkv": "attn.wkv", "wo_b": "attn.wo_b",
                "w1": "ffn.shared_experts.w1", "w2": "ffn.shared_experts.w2", "w3": "ffn.shared_experts.w3"}
LORA_TARGETS = ("wq_a", "wq_b", "wkv", "wo_b")  # the default: attention, as finetune.py's q, k, v, o
IMAGE_ONLY = (".gate.bias_vl",)  # checkpoint tensors only image tokens use

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



@dataclass(frozen=True)
class Cut:
    """Layers [0, first) frozen and precomputed, [first, n_layers) trained."""
    cfg: Config
    first: int

    def __post_init__(self):
        if not 0 < self.first < self.cfg.n_layers:
            raise ValueError(f"a cut must leave layers on both sides: first trained layer {self.first} of "
                             f"{self.cfg.n_layers}")

    @property
    def trained(self) -> range:
        return range(self.first, self.cfg.n_layers)

    def crossing(self, stop: int | None = None) -> dict[str, int | list[int]]:
        """What the trained layers reuse from below the cut, besides the stream and pre_mix: {"compress_kv": its
        source layer, "index_k": ..., "picks": ..., "candidates": ..., "engram": [trained engram layers, which need
        table rows]}, each present only when it crosses. With `stop`, what layers [first, stop) alone reuse (a piece
        of the split prep, deepseek_split)."""
        cfg, out = self.cfg, {}
        for layer in range(self.first, self.cfg.n_layers if stop is None else stop):
            if cfg.ratio(layer):
                kv, ix = cfg.source(layer, "kv"), cfg.source(layer, "index")
                if kv < self.first:
                    out.setdefault("compress_kv", kv)
                    if layer in cfg.index_source_layers:  # it picks with the kv source's index keys
                        out.setdefault("index_k", kv)
                if ix < self.first:
                    out.setdefault("picks", ix)
                if layer in cfg.index_source_layers and 0 <= cfg.candidate_source_layer < self.first:
                    out.setdefault("candidates", cfg.candidate_source_layer)
            if layer in cfg.engram_layer_ids:
                out.setdefault("engram", []).append(layer)
        return out

    def bytes_per_token(self, fp8_stream: bool = False, seq_len: int = 4096) -> dict[str, float]:
        """What a precomputed token costs, by part, as pack writes it. Compressed and index keys are kept in bf16
        (1 KB a token at ratio 1), though they're exactly FP4 values: packed as such they'd take 0.5625 bytes a value
        (the "fp4" figures). Candidate blocks matter only for sequences beyond candidate_topk_blocks blocks, and are
        then a block number each (2 bytes)."""
        cfg, c = self.cfg, self.crossing()
        width = cfg.hc_mult * cfg.dim
        out = {"tokens": 4.0 + 1 / 8,  # an int32 id, and a bit saying whether its successor is learned
               "stream": width * (1 + 1 / BLOCK) if fp8_stream else 2.0 * width,
               "pre_mix": 4.0 * cfg.hc_mult}
        if "compress_kv" in c:
            out["compress_kv"] = 2.0 * cfg.head_dim / cfg.ratio(c["compress_kv"])
            out["compress_kv (fp4)"] = 0.5625 * cfg.head_dim / cfg.ratio(c["compress_kv"])
        if "index_k" in c:
            out["index_k"] = 2.0 * cfg.index_head_dim / cfg.ratio(c["index_k"])
            out["index_k (fp4)"] = (0.5 + 1 / BLOCK) * cfg.index_head_dim / cfg.ratio(c["index_k"])
        if "picks" in c:
            out["picks"] = (2.0 if seq_len <= 2**15 else 4.0) * cfg.index_topk
        if "candidates" in c:
            reach = cfg.candidate_topk_blocks * cfg.candidate_block_size * cfg.ratio(c["candidates"])
            out["candidates"] = 2.0 * cfg.candidate_topk_blocks if seq_len > reach else 0.0
        out["total"] = sum(v for k, v in out.items() if "(fp4)" not in k)
        return out

    def check(self) -> None:
        """Refuse a cut whose trained layers would need a compressed-keys gradient carried between them."""
        cfg = self.cfg
        for src in self.trained:
            if src in cfg.kv_source_layers and src != self.first:
                users = [layer for layer in self.trained if layer > src and cfg.ratio(layer)
                         and cfg.source(layer, "kv") == src]
                if users:
                    raise NotImplementedError(f"layer {src} is a trained kv source above trained layer {self.first},"
                                              f" and layers {users} reuse its compressed keys: their gradient would "
                                              "have to be carried between trained layers. Cut at or above it")


def tiny_config(**kw) -> Config:
    """Every feature, small enough for tests and testnets: a pooling (ratio 2) kv source and a layer reusing it, a
    ratio-1 kv source that's also the candidate source, an index source reading another layer's keys inside its
    candidate blocks, an engram layer, YaRN, the SwiGLU clamp and route scaling. (tests/test_deepseek.py's.)"""
    from dataclasses import replace
    cfg = dict(vocab_size=64, dim=64, moe_inter_dim=32, n_layers=5, n_heads=2, n_routed_experts=4,
               n_activated_experts=2, route_scale=1.5, swiglu_limit=1.0, q_lora_rank=32, head_dim=64, rope_head_dim=16,
               o_groups=2, o_lora_rank=32, window_size=4, compress_ratios=(0, 2, 2, 1, 1),
               kv_source_layers=(1, 3), index_source_layers=(1, 3, 4), original_seq_len=8, rope_factor=4,
               index_n_heads=2, index_head_dim=32, index_topk=3, candidate_source_layer=3, candidate_topk_blocks=2,
               candidate_block_size=2, engram_layer_ids=(1,), engram_max_ngram_size=3, engram_vocab_size=11,
               engram_n_heads=2, engram_head_dim=32, engram_compressed_vocab_size=40)
    cfg.update(kw)
    cfg["engram_num_embeddings"] = (1,) * len(cfg["engram_layer_ids"])  # a placeholder, until the primes are known
    cfg = Config(**cfg)
    if not cfg.engram_layer_ids:
        return cfg
    layout = EngramLayout.of(cfg)  # the tables' sizes are what their bucket primes add up to, as the real ones
    return replace(cfg, engram_num_embeddings=tuple(layout.rows(i) for i in range(len(cfg.engram_layer_ids))))

"""DeepSeek's own reference for DeepSeek-V4.1-Flash, runnable on any machine: what tests/test_deepseek.py checks
vibehash/deepseek.py against.

The reference is inference/model.py and inference/engram.py at deepseek.COMMIT (MIT licence), fetched from the Hub
into ~/.vibehash/cache/deepseek/<commit>/reference and checked against the Git blob ids pinned below. It isn't copied
into this repository. It imports four modules it can't have here, which this file provides instead:
- kernel: its tilelang kernels (inference/kernel.py at the same commit), which need a Hopper-class GPU (sparse_attn at
  the real head size wants ~200 KB of shared memory per block; an RTX 4070 SUPER has 99 KB) and don't install on
  Windows. The stand-ins below follow each kernel's source line by line, written separately from deepseek.py's
  versions so the two can catch each other's mistakes: the same block sizes, the same scale rounding (here through
  frexp, there through exponent bits), FP4 rounding by bucketing (there by torch.round), and sparse_attn's online
  softmax in blocks of 64 with its probabilities cast to bf16 per block (there one softmax over all keys). The GEMMs
  dequantise and multiply in float32, which is what the kernels compute up to the order of float32 additions inside a
  tensor-core MMA, an order no PyTorch code can reproduce.
- image_processor and vision: only the vision tower uses them, and the tests run text only.
- sympy, when it isn't installed: engram.py wants only its isprime.
engram.py's build_compressed_token_map needs a tokenizer; the tests pass a token map directly instead (run()).
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass, field, fields

import torch
import torch.nn.functional as F

from vibehash import deepseek

# Git blob ids at deepseek.COMMIT, from the Hub's tree listing (2026-11-01).
FILES = {"inference/model.py": "46cc8310bb64852f3c36c9cc5dd13d47acabf6ee",
         "inference/engram.py": "a36bc6c6b8cbf2d3dbac850863c23b55a7ecb5c7",
         "inference/kernel.py": "6fa7bd5aafbe1bc62887080963cc9fdd838d89f5",  # not run: what the stand-ins follow
         "inference/config.json": "7a915cc69e21abbc6d7fb939cef5d09c72aa0d24",
         "config.json": "09917a9139b22d5bf8be52132787f435147b1020"}


def folder():
    return deepseek.cache_dir() / "reference"


def fetch(hub=None):
    """The reference's files, fetched once and checked against FILES. Raises HubError when the Hub can't be reached."""
    from vibehash.hub import Hub
    out = folder()
    if all((out / p).exists() for p in FILES):
        return out
    hub = hub or Hub()
    listing = hub.list_files(deepseek.REPO, deepseek.COMMIT)
    for path, oid in FILES.items():
        info = listing[path]
        if info.oid != oid:
            raise ValueError(f"the Hub lists {path} at commit {deepseek.COMMIT[:12]} as blob {info.oid}, not {oid}")
        data = hub.get_file(deepseek.REPO, deepseek.COMMIT, path, info=info)  # checked against the blob id
        (out / path).parent.mkdir(parents=True, exist_ok=True)
        (out / path).write_bytes(data)
    return out


# --- stand-ins for inference/kernel.py -----------------------------------------------------------------------------

FP4_TABLE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)  # convert.py's


def _round_scale(amax, fmax):
    """fast_round_scale(amax, 1/fmax): 2^ceil(log2(amax * fl32(1/fmax))), here through frexp."""
    x = amax * torch.tensor(1 / fmax, dtype=torch.float32, device=amax.device)
    m, e = torch.frexp(x)  # x = m * 2^e, m in [0.5, 1)
    return torch.ldexp(torch.ones_like(x), torch.where(m == 0.5, e - 1, e))


_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)  # between E2M1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6
_UP_IS_EVEN = (0.75, 1.75, 3.5)  # midpoints whose upper neighbour has the even code: a tie there rounds up


def _cast_fp4(v):
    """T.Cast(FP4, v) for v in [-6, 6]: round to nearest, ties to the even code, by bucketing."""
    a = v.abs()
    i = torch.bucketize(a, torch.tensor(_MIDPOINTS, device=v.device))  # midpoints strictly below a
    for m in _UP_IS_EVEN:
        i = i + (a == m).long()
    return torch.tensor(FP4_TABLE[:8], device=v.device)[i].copysign(v)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    assert x.dtype == torch.bfloat16 and x.size(-1) % block_size == 0  # the kernel's X is bf16
    b = x.float().unflatten(-1, (-1, block_size))
    amax = b.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _round_scale(amax, 448.0) if scale_fmt is not None else amax * torch.tensor(1 / 448, dtype=torch.float32)
    y = (b / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((y.float() * s).flatten(-2).to(x.dtype))
        return x
    return y.flatten(-2), s.squeeze(-1).to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert inplace and x.dtype == torch.bfloat16  # model.py only calls it in place
    b = x.float().unflatten(-1, (-1, block_size))
    amax = b.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        s = (amax.clamp_min(6 * 2**-9) / 6.0).to(torch.float8_e4m3fn).float()
    else:
        s = _round_scale(amax.clamp_min(6 * 2**-126), 6.0)
    x.copy_((_cast_fp4((b / s).clamp(-6.0, 6.0)) * s).flatten(-2).to(x.dtype))
    return x


def _blocks_up(s, rows, cols, block):  # a [r/block, c/block] scale grid at full [rows, cols] size
    return s.float().repeat_interleave(block, 0)[:rows].repeat_interleave(block, 1)[:, :cols]


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    k = a.size(-1)
    deq_a = (a.float().unflatten(-1, (-1, block_size)) * a_s.float().unsqueeze(-1)).flatten(-2)
    deq_b = b.float() * _blocks_up(b_s, b.size(0), k, block_size)
    return (deq_a @ deq_b.T).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    raw = b.view(torch.uint8).long()
    table = torch.tensor(FP4_TABLE, dtype=torch.float32, device=b.device)
    vals = torch.stack([table[raw & 0x0F], table[(raw >> 4) & 0x0F]], dim=-1).flatten(-2)  # [N, K], low nibble first
    deq_b = vals * b_s.float().repeat_interleave(32, 1)
    deq_a = (a.float().unflatten(-1, (-1, act_block_size)) * a_s.float().unsqueeze(-1)).flatten(-2)
    return (deq_a @ deq_b.T).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, query_chunk=64, block=64):
    """The kernel's online softmax: over each query's index list 64 at a time, a running max, rescaled sums, the
    probabilities cast to bf16 before they weight the values, the sink added to the denominator at the end. (It pads
    to 16 heads for speed; zero heads with a zero sink change nothing in the real ones.)"""
    b, m, h, d = q.shape
    topk = topk_idxs.size(-1)
    out = torch.empty_like(q)
    rows = torch.arange(b, device=q.device)[:, None, None]
    for c0 in range(0, m, query_chunk):
        qc = q[:, c0:c0 + query_chunk].float()
        idx_all = topk_idxs[:, c0:c0 + query_chunk].long()
        c = qc.size(1)
        acc_o = torch.zeros(b, c, h, d, device=q.device)
        sum_exp = torch.zeros(b, c, h, device=q.device)
        scores_max = torch.full((b, c, h), -1e30, device=q.device)
        for t in range(0, topk, block):
            idx = idx_all[..., t:t + block]
            valid = idx != -1
            kv_blk = torch.where(valid[..., None], kv[rows, idx.clamp_min(0)], 0).float()  # [b, c, block, d]
            acc_s = torch.where(valid[:, :, None, :], 0.0, -math.inf) + torch.einsum("bchd,bcjd->bchj", qc, kv_blk)
            acc_s = acc_s * softmax_scale
            prev = scores_max
            scores_max = torch.maximum(prev, acc_s.amax(-1))
            scale = torch.exp(prev - scores_max)
            acc_s = torch.exp(acc_s - scores_max[..., None])
            sum_exp = sum_exp * scale + acc_s.sum(-1)
            acc_o = acc_o * scale[..., None] + torch.einsum("bchj,bcjd->bchd", acc_s.to(torch.bfloat16).float(), kv_blk)
        sum_exp = sum_exp + torch.exp(attn_sink.float() - scores_max)
        out[:, c0:c0 + c] = (acc_o / sum_exp[..., None]).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = mixes[..., 2 * hc:].unflatten(-1, (hc, hc)) * hc_scale[2] + hc_base[2 * hc:].view(hc, hc)
    comb = torch.exp(comb - comb.amax(-1, keepdim=True))
    comb = comb / comb.sum(-1, keepdim=True) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


# --- loading the reference ------------------------------------------------------------------------------------------

_loaded = None


def load():
    """inference/model.py as a module, with the stand-ins. Raises HubError when it isn't cached and can't be fetched."""
    global _loaded
    if _loaded is not None:
        return _loaded
    base = fetch() / "inference"
    kernel = types.ModuleType("kernel")
    for name in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm", "sparse_attn", "hc_split_sinkhorn"):
        setattr(kernel, name, globals()[name])
    image_processor = types.ModuleType("image_processor")
    image_processor.TEXT, image_processor.IMAGE, image_processor.IMAGE_START = -1, 0, 1
    image_processor.IMAGE_END, image_processor.IMAGE_NEW_LINE = 2, 3
    vision = types.ModuleType("vision")
    vision.ViT = vision.Aligner = None
    stand_ins = {"kernel": kernel, "image_processor": image_processor, "vision": vision}
    try:
        import sympy  # noqa: F401
    except ImportError:
        sympy = types.ModuleType("sympy")
        sympy.isprime = deepseek._is_prime
        stand_ins["sympy"] = sympy
    saved = {k: sys.modules.get(k) for k in [*stand_ins, "engram"]}
    sys.modules.update(stand_ins)
    try:
        engram = _module("engram", base / "engram.py")
        model = _module("deepseek_reference_model", base / "model.py")
    finally:
        for k, v in saved.items():  # model.py holds what it imported; nothing else should see these names
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    model.engram_module = engram
    _loaded = model
    return model


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@contextmanager
def default_dtype(dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@contextmanager
def unfilled():
    """torch.empty without the deterministic mode's fill (not implemented for float4_e2m1fn_x2, and every parameter
    is overwritten by load_state anyway)."""
    import torch.utils.deterministic as det
    prev = det.fill_uninitialized_memory
    det.fill_uninitialized_memory = False
    try:
        yield
    finally:
        det.fill_uninitialized_memory = prev


def model_args(ref, cfg: deepseek.Config, batch: int, seqlen: int):
    """The reference's ModelArgs for a deepseek.Config, its caches sized for one prefill."""
    names = {f.name for f in fields(ref.ModelArgs)}
    kw = {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.name in names}
    return ref.ModelArgs(**kw, max_batch_size=batch, max_seq_len=seqlen, vision_n_layers=0)


def reference_tensor(name, t, like):
    """A checkpoint tensor as the reference holds it, by convert.py's rules: wo_a dequantised to bf16, experts viewed
    as float4_e2m1fn_x2, and float tensors cast to the parameter's dtype (the head and the pooling compressors are
    float32 there)."""
    if t.dtype == torch.int8:
        return t.view(torch.float4_e2m1fn_x2)
    if t.dtype != like.dtype and like.dtype in (torch.float32, torch.bfloat16):
        return t.to(like.dtype)
    return t


def _dequant_wo_a(w, s):  # convert.py's wo_a conversion
    out = w.unflatten(0, (-1, 32)).unflatten(-1, (-1, 32)).float() * s[:, None, :, None].float()
    return out.flatten(2, 3).flatten(0, 1).bfloat16()


def load_state(module, state: dict, prefix: str = ""):
    """Point a reference module's parameters at checkpoint tensors, sharing them where the formats agree (the
    module may sit elsewhere: the parameters move to where the tensors are)."""
    params = dict(module.named_parameters())
    for name, p in params.items():
        key = prefix + name
        if key.endswith("attn.wo_a.weight"):
            t = _dequant_wo_a(state[key], state[key[:-len("weight")] + "scale"])
        else:
            t = reference_tensor(key, state[key], p)
        if t.shape != p.shape or t.dtype != p.dtype:
            raise ValueError(f"{key}: checkpoint {t.dtype} {list(t.shape)}, reference {p.dtype} {list(p.shape)}")
        p.data = t  # the caller put the state where the module should run
    used = {prefix + n for n in params} | {prefix + n[:-len("weight")] + "scale" for n in params
                                           if n.endswith("attn.wo_a.weight")}
    extra = {k for k in state if k.startswith(prefix) and k not in used and not k.endswith(deepseek.IMAGE_ONLY)}
    if extra:
        raise ValueError(f"checkpoint tensors the reference has no place for: {sorted(extra)[:5]}")


@dataclass
class Run:
    """What the reference computed, layer by layer."""
    logits: torch.Tensor  # [b, s, vocab] float32, every position (the head applied as ParallelHead applies it)
    last_logits: torch.Tensor  # [b, vocab]: what Transformer.forward itself returned
    block_in: dict = field(default_factory=dict)  # layer -> (x, pre_mix) entering the block (after the engram)
    engram_in: dict = field(default_factory=dict)  # layer -> x before the engram
    block_out: dict = field(default_factory=dict)  # layer -> (x, pre_mix)
    experts: dict = field(default_factory=dict)  # layer -> [tokens, k]
    picks: dict = field(default_factory=dict)  # index source -> [b, s, k] compressed positions, -1 none
    candidates: torch.Tensor | None = None
    compress_kv: dict = field(default_factory=dict)  # kv source -> [b, s // r, head_dim]
    index_k: dict = field(default_factory=dict)  # kv source -> [b, s // r, index_head_dim]
    hashes: dict = field(default_factory=dict)  # engram layer -> [b, s, cols] the rows it read


def run(cfg: deepseek.Config, state: dict, tokens: torch.Tensor, token_map=None, device="cpu") -> Run:
    """The reference Transformer on `tokens` (one prefill), with `state` (deepseek.random_weights' form) loaded."""
    ref = load()
    b, s = tokens.shape
    with default_dtype(torch.bfloat16), torch.device(device):
        ref.precompute_freqs_cis.cache_clear()  # its caches are keyed without the device
        ref.get_window_topk_idxs.cache_clear()
        args = model_args(ref, cfg, b, s)
        if cfg.engram_layer_ids:
            vocab = cfg.engram_compressed_vocab_size
            ref.engram_module.build_compressed_token_map = lambda tokenizer: (token_map.tolist(), vocab)
        with unfilled():
            model = ref.Transformer(args)
        load_state(model, {k: v.to(device) for k, v in state.items()})
        out = Run(None, None)
        hooks = []
        for i, layer in enumerate(model.layers):
            hooks.append(layer.register_forward_pre_hook(
                lambda mod, a, i=i: out.block_in.__setitem__(i, (a[0].clone(), a[2].clone()))))
            hooks.append(layer.register_forward_hook(
                lambda mod, a, o, i=i: out.block_out.__setitem__(i, (o[0].clone(), o[1].clone()))))
            hooks.append(layer.ffn.gate.register_forward_hook(
                lambda mod, a, o, i=i: out.experts.__setitem__(i, o[1].clone())))
            if layer.engram is not None:
                hooks.append(layer.engram.register_forward_pre_hook(
                    lambda mod, a, i=i: out.engram_in.__setitem__(i, a[0].clone())))
                hooks.append(layer.engram.register_forward_pre_hook(
                    lambda mod, a, i=i: out.hashes.__setitem__(i, a[1].clone())))
            if layer.attn.indexer is not None:
                def picked(mod, a, o, i=i):
                    offset = a[4]
                    out.picks[i] = torch.where(o == -1, -1, o.long() - offset)
                    if i == cfg.candidate_source_layer:
                        out.candidates = ref.shared_attn.candidates.clone()
                hooks.append(layer.attn.indexer.register_forward_hook(picked))
        normed = {}
        hooks.append(model.head.register_forward_pre_hook(lambda mod, a: normed.__setitem__("x", a[0].clone())))
        _, last, _ = model(tokens.to(device))
        for h in hooks:
            h.remove()
        out.logits, out.last_logits = F.linear(normed["x"].float(), model.head.weight), last.clone()
        for i in cfg.kv_source_layers:
            if i < cfg.n_layers:
                n = s // cfg.ratio(i)
                attn = model.layers[i].attn
                out.compress_kv[i] = attn.compress_kv_cache[:b, :n].clone()
                if attn.indexer is not None and attn.indexer.owns_k:
                    out.index_k[i] = attn.indexer.k_cache[:b, :n].clone()
    return out


def side_for(cfg: deepseek.Config, r: Run, layer: int) -> deepseek.Side:
    """The side channels the reference's earlier layers handed `layer`."""
    side = deepseek.Side()
    kv = [i for i in cfg.kv_source_layers if i < layer]
    ix = [i for i in cfg.index_source_layers if i < layer]
    if kv:
        side.compress_kv, side.index_k = r.compress_kv[kv[-1]], r.index_k.get(kv[-1])
    if ix:
        side.picks = r.picks[ix[-1]]
    return side


def block(cfg: deepseek.Config, layer: int, state: dict, device="cuda", max_seqlen=4096):
    """One reference Block for real-size weights: built on the CPU (uninitialised, so it costs address space, not
    time), then pointed at `state`'s tensors on `device`, sharing them with deepseek.py's Block."""
    ref = load()
    with default_dtype(torch.bfloat16):
        ref.precompute_freqs_cis.cache_clear()
        ref.get_window_topk_idxs.cache_clear()
        with unfilled():
            blk = ref.Block(layer, model_args(ref, cfg, 1, max_seqlen))
        load_state(blk, state, prefix=f"layers.{layer}.")
        blk.to(device)
    return blk


def block_run(blk, x, pre_mix):
    """A reference Block's forward on one prefill, as Transformer.forward calls it, and the choices it made:
    (x, pre_mix, experts [tokens, k], picks [b, s, k] or None)."""
    ref = load()
    got = {}
    hooks = [blk.ffn.gate.register_forward_hook(lambda mod, a, o: got.__setitem__("experts", o[1].clone()))]
    if blk.attn.indexer is not None:
        hooks.append(blk.attn.indexer.register_forward_hook(
            lambda mod, a, o: got.__setitem__("picks", torch.where(o == -1, -1, o.long() - a[4]))))
    ref.get_window_topk_idxs.cache_clear()
    try:
        with default_dtype(torch.bfloat16), torch.device(x.device), torch.inference_mode():
            y, pre = blk(x, 0, pre_mix, None)
    finally:
        for h in hooks:
            h.remove()
    return y.clone(), pre.clone(), got["experts"], got.get("picks")

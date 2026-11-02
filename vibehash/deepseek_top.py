"""Fine-tuning DeepSeek-V4.1-Flash's top layers only: the frozen layers' output, computed once, is the job's data.

DEEPSEEK.md §8. A job freezes layers [0, first) and trains LoRA adapters on the top layers [first, n_layers) only.
Then nothing below the cut ever changes, so what crosses it for the job's text can be computed once, by one machine
holding the whole model (the preparer), and shipped to miners as the job's dataset: the residual stream and pre_mix
leaving layer first - 1, and the side channels the top layers reuse from source layers below the cut (Cut.crossing).
A miner then needs only the top layers, the final norm and the head: about 30 GB for the top four, a game install.

What's here:
- Cut: what crosses a cut and how many bytes a token, and whether layer-major training handles it.
- precompute: the frozen layers on a batch of sequences, once, and their own choices (for spot-checks).
- pack / unpack: one sequence as a record of the job's dataset, bf16 or FP8 stream; record_root: their Merkle root.
- Adapters: LoRA on the top layers, A frozen and drawn from the job's seed as PEFT draws it, B trained (LoRA-FA, as
  finetune.py, so that averaging units' B averages their changes to the model), saved in PEFT's format.
- train_step: one Adam step of B on the top layers from precomputed micro-batches, layer-major (each layer runs on all
  of a step's micro-batches before the next, forward then backward, so only one layer's weights need be on the GPU),
  recomputing each layer in the backward from its stored input. whole_model_step is the same step through the whole
  model with its bottom frozen, the definition train_step must equal (tests/test_deepseek_top.py: bit for bit).

Gradients and the cut. Below the cut nothing is trained, so nothing needs a gradient there: the top layers' inputs
and side channels are constants. Within the top, picks, index keys and candidate blocks are discrete and carry no
gradient anyway. Compressed keys do carry one, but a trained kv source's compressed keys depend only on its own
input, so when it's the first trained layer they're constants for the adapters too. A cut that trains a kv source
above a trained layer would need that gradient carried between layers: Cut.check refuses it (cuts above layer 20 in
DeepSeek-V4.1-Flash never do).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import deepseek as ds
from .crypto import merkle_root
from .deepseek import Choices, Config, Model, Side

OPTIMIZER = (0.9, 0.999, 1e-8)  # Adam's beta1, beta2, eps: finetune.py's "adam"
# How the trained layers compute. The reference's bf16 with FP8 rounding (the preparer's, for the frozen layers) makes
# a training step on the CPU and on CUDA end 4-9% of B apart after four steps; float32 without activation rounding,
# 7e-5 (DEEPSEEK.md §8), which a checker can work with, as finetune.py's float32 units.
TRAIN = ds.Numerics(torch.float32, quantize=False)
RECORD_FORMAT = "vibehash-deepseek-top/1"


# --- the cut -----------------------------------------------------------------------------------------------------------

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
               "stream": width * (1 + 1 / ds.BLOCK) if fp8_stream else 2.0 * width,
               "pre_mix": 4.0 * cfg.hc_mult}
        if "compress_kv" in c:
            out["compress_kv"] = 2.0 * cfg.head_dim / cfg.ratio(c["compress_kv"])
            out["compress_kv (fp4)"] = 0.5625 * cfg.head_dim / cfg.ratio(c["compress_kv"])
        if "index_k" in c:
            out["index_k"] = 2.0 * cfg.index_head_dim / cfg.ratio(c["index_k"])
            out["index_k (fp4)"] = (0.5 + 1 / ds.BLOCK) * cfg.index_head_dim / cfg.ratio(c["index_k"])
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


# --- the preparer's side: precomputing -----------------------------------------------------------------------------------

@dataclass
class Example:
    """One micro-batch of whole sequences, as the trained layers see them: the token ids (for the loss), which
    positions' successors are learned, and what crosses the cut."""
    tokens: torch.Tensor  # [b, s]
    mask: torch.Tensor  # [b, s] bool: position t's next token, tokens[t + 1], is learned
    x: torch.Tensor  # [b, s, hc, dim], the stream leaving layer first - 1
    pre: torch.Tensor  # [b, s, hc], the pre_mix it hands on
    side: Side = field(default_factory=Side)
    candidates: torch.Tensor | None = None

    def to(self, device) -> "Example":
        move = (lambda t: None if t is None else t.to(device))
        side = Side(*(move(getattr(self.side, f.name)) for f in dataclasses.fields(Side)))
        return Example(move(self.tokens), move(self.mask), move(self.x), move(self.pre), side, move(self.candidates))

    @property
    def targets(self) -> torch.Tensor:
        return torch.cat((self.tokens[:, 1:], self.tokens[:, :1]), 1)  # the last column is never learned


@torch.no_grad()
def precompute(bottom: Model, cut: Cut, tokens, mask, token_map=None) -> tuple[Example, Choices]:
    """The frozen layers on token ids [b, s] (the model built with layers range(cut.first) and the embedding), once:
    the Example the trained layers start from, and the frozen layers' own choices, which the preparer commits to so
    that a checker can replay any frozen layer exactly."""
    choices, side = Choices(), Side()
    hashes = ds.engram_hashes(tokens, token_map, cut.cfg) if cut.cfg.engram_layer_ids else None
    x, pre = bottom.run(*bottom.start(tokens), side, choices, hashes)
    kept, cand = crossing_of(cut, side, choices, tokens.size(1))
    return Example(tokens, mask.bool(), x, pre, kept, cand), choices


def crossing_of(cut: Cut, side: Side, choices: Choices, seqlen: int) -> tuple[Side, torch.Tensor | None]:
    """Of the frozen layers' side channels and choices, what the trained layers need: Cut.crossing's, and the
    candidate blocks only when there are more blocks than the source keeps (otherwise it keeps every one)."""
    crossing, cfg = cut.crossing(), cut.cfg
    kept = Side(*(getattr(side, name) if name in crossing else None for name in ("compress_kv", "index_k", "picks")))
    cand = None
    if "candidates" in crossing:
        n = seqlen // cfg.ratio(crossing["candidates"])
        cand = choices.candidates if n > cfg.candidate_topk_blocks * cfg.candidate_block_size else None
    return kept, cand


def pack(ex: Example, i: int, fp8_stream: bool = False) -> bytes:
    """Sequence i of an Example as one record of the job's dataset: a JSON line naming the parts, then each part's
    bytes. With fp8_stream the stream is rounded to FP8 with a power-of-two scale per 32 values (as the model rounds
    its own activations), halving the record; the rounded values are then the job's data, exactly."""
    parts = {"tokens": ex.tokens[i].to(torch.int32), "mask": ex.mask[i].to(torch.uint8), "pre": ex.pre[i].float()}
    if fp8_stream:
        q, s = _fp8_rows(ex.x[i].float())
        parts["x_fp8"], parts["x_scale"] = q, s
    else:
        parts["x"] = ex.x[i].to(torch.bfloat16)
    for name in ("compress_kv", "index_k", "picks"):
        t = getattr(ex.side, name)
        if t is not None:
            parts[name] = t[i].to(torch.int16 if name == "picks" and t.max() < 2**15 else
                                  torch.int32 if name == "picks" else torch.bfloat16)
    if ex.candidates is not None:
        parts["candidates"] = ex.candidates[i].to(torch.uint8)
    head = {"format": RECORD_FORMAT, "parts": [[k, str(v.dtype).removeprefix("torch."), list(v.shape)]
                                               for k, v in parts.items()]}
    blob = b"".join(v.contiguous().cpu().view(torch.uint8).numpy().tobytes() for v in parts.values())
    return json.dumps(head, separators=(",", ":")).encode() + b"\n" + blob


def unpack(record: bytes) -> dict[str, torch.Tensor]:
    """A record's parts, as pack wrote them, the stream back in bf16 (FP8 records dequantised exactly)."""
    line, blob = record.split(b"\n", 1)
    head = json.loads(line)
    if head.get("format") != RECORD_FORMAT:
        raise ValueError(f"not a {RECORD_FORMAT} record")
    out, at = {}, 0
    for name, dtype, shape in head["parts"]:
        dt = getattr(torch, dtype)
        n = math.prod(shape) * torch.empty(0, dtype=dt).element_size()
        out[name] = torch.frombuffer(bytearray(blob[at:at + n]), dtype=torch.uint8).view(dt).reshape(shape)
        at += n
    if at != len(blob):
        raise ValueError("a record with bytes left over")
    if "x_fp8" in out:
        q, s = out.pop("x_fp8"), out.pop("x_scale")
        out["x"] = (q.float().unflatten(-1, (-1, ds.BLOCK)) * s.float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
    return out


def _fp8_rows(x):
    """x [..., n] as E4M3 values and a power-of-two scale per 32: ds.round_fp8's rounding, in its stored form."""
    b = x.unflatten(-1, (-1, ds.BLOCK))
    s = ds._pow2_ceil(b.abs().amax(-1, keepdim=True).clamp_min(1e-4) * ds._inv(ds.FP8_MAX, x.device))
    q = (b / s).clamp(-ds.FP8_MAX, ds.FP8_MAX).to(torch.float8_e4m3fn)
    return q.flatten(-2), s.squeeze(-1).to(torch.float8_e8m0fnu)


def record_root(records: list[bytes]) -> str:
    """The Merkle root a job pins for its dataset: one leaf per record, the sha256 of its bytes."""
    return merkle_root([hashlib.sha256(r).digest() for r in records]).hex()


def example_of(records: list[dict[str, torch.Tensor]], cut: Cut) -> Example:
    """Unpacked records of equal length as one micro-batch."""
    stack = (lambda name: torch.stack([r[name] for r in records]) if name in records[0] else None)
    side = Side(stack("compress_kv"), stack("index_k"), None if stack("picks") is None else stack("picks").long())
    cand = stack("candidates")
    return Example(stack("tokens").long(), stack("mask").bool(), stack("x"), stack("pre"), side,
                   None if cand is None else cand.bool())


# --- adapters -------------------------------------------------------------------------------------------------------------

@dataclass
class Adapters:
    """LoRA adapters on the trained layers: per adapted projection, A [rank, in] (frozen) and B [out, rank] (trained,
    from zero, so a fresh adapter changes nothing), with Adam's moments for B."""
    names: list[str]  # module paths, layer by layer (layers.36.attn.wq_a ...)
    a: list[torch.Tensor]
    b: list[torch.Tensor]
    rank: int
    alpha: float
    m: list[torch.Tensor] = field(default_factory=list)
    v: list[torch.Tensor] = field(default_factory=list)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    @classmethod
    def fresh(cls, model: Model, cut: Cut, rank: int = 16, alpha: float = 32.0, targets=ds.LORA_TARGETS,
              seed: int = 0, device="cpu") -> "Adapters":
        """New adapters on the cut's trained layers of `model`. A is uniform in ±1/√in, as PEFT draws it (kaiming
        uniform with a = √5), from PCG64's raw stream for `seed` (finetune.py's recipe: stable across numpy
        versions), in the order of `names`."""
        rng = np.random.Generator(np.random.PCG64([seed, 0x10A]))
        names, a, b = [], [], []
        for name, mod in model.lora_modules(targets).items():
            if int(name.split(".")[1]) not in cut.trained:
                continue
            out, inp = _shape(mod)
            raw = rng.bit_generator.random_raw(rank * inp)
            u = (raw >> np.uint64(11)).astype(np.float64) * 2.0 ** -53
            a.append(torch.tensor(((2 * u - 1) / math.sqrt(inp)).astype(np.float32).reshape(rank, inp), device=device))
            b.append(torch.zeros(out, rank, device=device, requires_grad=True))
            names.append(name)
        zeros = [torch.zeros_like(t, requires_grad=False) for t in b]
        return cls(names, a, b, rank, alpha, zeros, [z.clone() for z in zeros])

    def attach(self, model: Model) -> Model:
        """Point `model`'s projections at these adapters (those of its layers that have them)."""
        mods = model.lora_modules(tuple(ds.LORA_MODULES))
        for name, a, b in zip(self.names, self.a, self.b):
            if name in mods:
                mods[name].lora = (a, b, self.scaling)
        return model

    def peft_state(self) -> dict[str, torch.Tensor]:
        """The adapter as PEFT saves it (adapter_model.safetensors), under the checkpoint's module names."""
        out = {}
        for name, a, b in zip(self.names, self.a, self.b):
            out[f"base_model.model.{name}.lora_A.weight"] = a.detach().float().cpu()
            out[f"base_model.model.{name}.lora_B.weight"] = b.detach().float().cpu()
        return out

    def peft_config(self, base: str = f"{ds.REPO}") -> dict:
        """adapter_config.json: which modules, in which layers (layers_to_transform), as PEFT reads it."""
        alpha = int(self.alpha) if float(self.alpha).is_integer() else self.alpha
        targets = sorted({n.split(".")[-1] for n in self.names})
        layers = sorted({int(n.split(".")[1]) for n in self.names})
        return {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base,
                "revision": ds.COMMIT, "r": self.rank, "lora_alpha": alpha, "target_modules": targets,
                "layers_to_transform": layers, "layers_pattern": "layers", "lora_dropout": 0.0, "bias": "none",
                "use_rslora": False, "fan_in_fan_out": False, "inference_mode": True, "init_lora_weights": True,
                "modules_to_save": None}


def _shape(mod) -> tuple[int, int]:
    out = mod.scale.shape[0] if mod.weight.dtype == torch.int8 else mod.weight.shape[0]
    inp = 2 * mod.weight.shape[1] if mod.weight.dtype == torch.int8 else mod.weight.shape[1]
    return out, inp


def adam(adapters: Adapters, step: int, lr: float) -> None:
    """One Adam step on B from B.grad, step counted from 1, elementwise operations in finetune.py's order."""
    b1, b2, eps = (float(np.float32(x)) for x in OPTIMIZER)
    c1, c2 = float(np.float32(1 - b1 ** step)), float(np.float32(1 - b2 ** step))
    with torch.no_grad():
        for b, m, v in zip(adapters.b, adapters.m, adapters.v):
            g = b.grad
            m.mul_(b1).add_(g * float(np.float32(1 - b1)))
            v.mul_(b2).add_(g * g * float(np.float32(1 - b2)))
            b.sub_((m / c1) / ((v / c2).sqrt() + eps) * float(np.float32(lr)))
            b.grad = None


# --- the miner's side: training --------------------------------------------------------------------------------------------

def train_step(top: Model, cut: Cut, adapters: Adapters, batches: list[Example], step: int, lr: float,
               load=None, update: bool = True) -> float:
    """One Adam step of the adapters on the trained layers (the model built with layers cut.trained and the head),
    from precomputed micro-batches, layer-major: each layer runs on every micro-batch before the next layer, forward,
    then backward in reverse, recomputing each layer from the input it was given. `load(layer)`, if given, is called
    before a layer runs, to bring its weights onto the GPU (and may send the previous one's back). Returns the
    step's mean loss over the learned tokens. With update=False the gradients stay in B.grad, unapplied."""
    cut.check()
    order = [str(i) for i in cut.trained]
    count = sum(int(ex.mask.sum()) for ex in batches)
    states = [(ex.x, ex.pre, dataclasses.replace(ex.side), Choices(candidates=ex.candidates)) for ex in batches]
    inputs = {}
    with torch.no_grad():
        for name in order:
            if load is not None:
                load(int(name))
            block = top.layers[name]
            for j, (x, pre, side, choices) in enumerate(states):
                inputs[name, j] = (x, pre, dataclasses.replace(side))
                x, pre = block(x, pre, side, choices)
                states[j] = (x, pre, side, choices)
    if load is not None:
        load(None)  # the head's turn: no layer on the GPU
    total, grads = 0.0, []
    for j, (x, pre, _, _) in enumerate(states):
        x, pre = x.detach().requires_grad_(), pre.detach().requires_grad_()
        loss, _ = top.loss(x, pre, batches[j].targets, batches[j].mask)
        (loss / count).backward()
        total += float(loss.detach())
        grads.append((x.grad, pre.grad))
    for name in reversed(order):
        if load is not None:
            load(int(name))
        block, below = top.layers[name], int(name) != cut.first
        for j in range(len(batches)):
            x, pre, side = inputs[name, j]
            x, pre = x.detach().requires_grad_(below), pre.detach().requires_grad_(below)
            y, ypre = block(x, pre, dataclasses.replace(side), states[j][3])
            torch.autograd.backward([y, ypre], list(grads[j]))
            grads[j] = (x.grad, pre.grad)
    if update:
        adam(adapters, step, lr)
    return total / count


@torch.no_grad()
def forward_streamed(block, states: list, device, rows=None) -> list:
    """One layer over several sequences, expert-major: everything before the MoE and the router for each, then each
    routed expert brought to `device` once to run every sequence's tokens routed to it, then the rest. `states` is a
    list of (x, pre_mix, side, choices), updated in place to the layer's output. rows: engram table rows per sequence,
    for an engram layer. Returns each sequence's routed experts' sum (the streamed backward's starting point). Every
    sequence's arithmetic is the plain forward's, to the bit: only the order the sequences take turns in differs."""
    moe, parts = block.ffn, []
    for j, (x, pre, side, choices) in enumerate(states):
        h, x_mid, post, comb, ffn_pre = block.before_ffn(x, pre, side, choices, rows=None if rows is None else rows[j])
        flat = h.reshape(-1, h.size(-1))
        chosen, gates = moe.route(flat, choices)
        parts.append([flat, x_mid, post, comb, ffn_pre, moe.assignments(chosen), gates,
                      torch.zeros_like(flat, dtype=moe.num.hi), h.shape])
    for e in sorted({e for p in parts for e, _, _ in p[5]}):
        with _OnDevice(moe.experts[e], device) as expert:
            for p in parts:
                for e2, tok, slot in p[5]:
                    if e2 == e:
                        p[7].index_add_(0, tok, expert(p[0][tok], p[6][tok, slot, None]).to(moe.num.hi))
    for j, p in enumerate(parts):
        y = moe.combine(p[7], p[0]).view(p[8])
        states[j] = (block.after_ffn(y, p[1], p[2], p[3]), p[4], states[j][2], states[j][3])
    return [p[7] for p in parts]


class Meter:
    """Seconds, tokens a second and peak GPU memory of what runs inside it: what a unit reports with its work, so
    the first live job measures the real numbers (DEEPSEEK.md §8)."""

    def __init__(self, device="cpu"):
        self.cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        self.seconds, self.peak = 0.0, 0

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            torch.cuda.synchronize()
            self.peak = torch.cuda.max_memory_allocated()
        self.seconds = time.perf_counter() - self.t0

    def stats(self, tokens: int) -> dict:
        """tokens: what was processed (layer-tokens for a prep unit, tokens for a training step)."""
        return {"seconds": round(self.seconds, 3), "tokens": tokens,
                "tokens_per_s": round(tokens / max(self.seconds, 1e-9), 1), "peak_gpu_gb": round(self.peak / 2**30, 3)}


class _OnDevice:
    """A module's buffers (an expert's quantised matrices) copied to `device` for the length of a with-block, then
    pointed back at their pinned-RAM originals."""

    def __init__(self, module, device):
        self.module, self.device, self.saved = module, device, []

    def __enter__(self):
        for mod in self.module.modules():
            for name, b in list(mod._buffers.items()):
                if b is not None and b.device != torch.device(self.device):
                    self.saved.append((mod, name, b))
                    mod._buffers[name] = b.to(self.device, non_blocking=True)
        return self.module

    def __exit__(self, *exc):
        for mod, name, b in self.saved:
            mod._buffers[name] = b


def train_step_streamed(top: Model, cut: Cut, adapters: Adapters, batches: list[Example], step: int, lr: float,
                        device="cuda", keep="cpu", update: bool = True) -> float:
    """train_step's step for a GPU that can't hold a layer's 7.2 GB of routed experts: they stay in (pinned) RAM and
    each comes to the GPU once per layer for the forward and once for the backward, running every micro-batch's
    tokens routed to it, expert-major. Everything else of the model (0.2 GB a layer, the head) is on `device`; what
    each layer keeps for its backward (its input, the experts' summed output) waits in `keep`.

    The forward is train_step's exactly (each micro-batch's tokens go through each expert alone, in the same order).
    The backward is the same arithmetic split into stages: after the MoE; the routed experts, one at a time; the
    router and the shared expert; then everything before the MoE. Its gradients sum in another order than autograd's
    whole-layer backward, so they agree with train_step's to rounding, not to the bit (tests: 1e-6 of B in float32)."""
    cut.check()
    order = [str(i) for i in cut.trained]
    count = sum(int(ex.mask.sum()) for ex in batches)
    move = (lambda t, where: None if t is None else t.to(where, non_blocking=True))
    states = [(ex.x.to(device), ex.pre.to(device), dataclasses.replace(ex.side), Choices(candidates=ex.candidates))
              for ex in batches]
    saved = {}
    for name in order:
        for j, (x, pre, side, _) in enumerate(states):
            saved[name, j] = [move(x, keep), move(pre, keep), dataclasses.replace(side)]
        routed = forward_streamed(top.layers[name], states, device)
        for j, r in enumerate(routed):
            saved[name, j].append(move(r, keep))
    total, grads = 0.0, []
    for j, (x, pre, _, _) in enumerate(states):
        x, pre = x.detach().requires_grad_(), pre.detach().requires_grad_()
        loss, _ = top.loss(x, pre, batches[j].targets.to(device), batches[j].mask.to(device))
        (loss / count).backward()
        total += float(loss.detach())
        grads.append((x.grad, pre.grad))
    for name in reversed(order):
        block, below = top.layers[name], int(name) != cut.first
        moe, hi, parts = block.ffn, block.ffn.num.hi, []
        for j in range(len(batches)):
            x_in, pre_in, side, routed = saved[name, j]
            choices = states[j][3]
            with torch.no_grad():  # the MoE's input again, to differentiate the stages after it
                h, x_mid, post, comb, ffn_pre = block.before_ffn(x_in.to(device), pre_in.to(device),
                                                                 dataclasses.replace(side), choices)
            flat = h.reshape(-1, h.size(-1)).detach().requires_grad_()
            routed = routed.to(device).requires_grad_()
            x_mid, post, comb = (t.detach().requires_grad_() for t in (x_mid, post, comb))
            out = block.after_ffn(moe.combine(routed, flat).view(h.shape), x_mid, post, comb)
            torch.autograd.backward(out, grads[j][0])  # through the shared expert too
            with torch.no_grad():
                chosen, g = moe.route(flat.detach(), choices)
            parts.append([flat.detach(), flat.grad, routed.grad, move(x_mid.grad, keep), move(post.grad, keep),
                          move(comb.grad, keep), moe.assignments(chosen), x_in, pre_in, side, h.shape, g])
        gates = [p[11] for p in parts]  # the gates' values; their gradient gathers in dgates
        dgates = [torch.zeros_like(g) for g in gates]
        for e in sorted({e for p in parts for e, _, _ in p[6]}):
            with _OnDevice(moe.experts[e], device) as expert:
                for p, g, dg in zip(parts, gates, dgates):
                    for e2, tok, slot in p[6]:
                        if e2 == e:
                            xe = p[0][tok].requires_grad_()
                            ge = g[tok, slot, None].detach().requires_grad_()
                            torch.autograd.backward(expert(xe, ge).to(hi), p[2][tok])
                            p[1].index_add_(0, tok, xe.grad)
                            dg[tok, slot] += ge.grad[:, 0]
        for j, (p, dg) in enumerate(zip(parts, dgates)):
            flat = p[0].clone().requires_grad_()
            torch.autograd.backward(moe.route(flat, states[j][3])[1], dg)
            dh = (p[1] + flat.grad).view(p[10])
            x_in, pre_in, side = p[7], p[8], p[9]
            x_in = x_in.to(device).detach().requires_grad_(below)
            pre_in = pre_in.to(device).detach().requires_grad_(below)
            h, x_mid, post, comb, ffn_pre = block.before_ffn(x_in, pre_in, dataclasses.replace(side), states[j][3])
            torch.autograd.backward([h, x_mid, post, comb, ffn_pre],
                                    [dh, p[3].to(device), p[4].to(device), p[5].to(device), grads[j][1]])
            grads[j] = (x_in.grad, pre_in.grad)
    if update:
        adam(adapters, step, lr)
    return total / count


def whole_model_step(model: Model, cut: Cut, adapters: Adapters, tokens: list[torch.Tensor], masks: list[torch.Tensor],
                     step: int, lr: float, token_map=None) -> float:
    """The same step through the whole model, its layers below the cut frozen, one micro-batch at a time: what
    train_step from precomputed inputs has to equal."""
    count = sum(int(m.sum()) for m in masks)
    total = 0.0
    for t, m in zip(tokens, masks):
        hashes = ds.engram_hashes(t, token_map, cut.cfg) if cut.cfg.engram_layer_ids else None
        x, pre = model.run(*model.start(t), Side(), Choices(), hashes)
        ex = Example(t, m.bool(), x, pre)
        loss, _ = model.loss(x, pre, ex.targets, ex.mask)
        (loss / count).backward()
        total += float(loss.detach())
    adam(adapters, step, lr)
    return total / count


def freeze(model: Model) -> Model:
    """Every weight of the model frozen: in a top-layers job only the adapters train."""
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_trained(cut: Cut, weights: dict, device="cpu", num: ds.Numerics = TRAIN) -> Model:
    """The trained layers and the head as a miner holds them, frozen: everything on `device` (cast as `num` keeps
    it) but the routed experts' quantised matrices, which stay the very tensors in `weights`. With weights from
    read_safetensors(..., mmap=True), the experts (7.2 GB a layer) stay on the disk until train_step_streamed brings
    each one over, and the OS keeps what RAM allows: that's how a PC with 16 GB of RAM trains four layers
    (DEEPSEEK.md §8). With 48 GB they all stay in memory, and an NVMe SSD makes the difference small."""
    model = Model(cut.cfg, cut.trained, num, device="meta", embed=False)
    experts = {n for n, _ in model.named_buffers() if ".ffn.experts." in n}
    for name, mod in model.named_modules():
        if ".ffn.experts" not in name:
            mod.to_empty(device=device, recurse=False)
    rest = {k: v for k, v in weights.items() if k not in experts and not k.endswith(ds.IMAGE_ONLY)}
    got = model.load_state_dict(rest, strict=False)
    if got.unexpected_keys or set(got.missing_keys) - experts:
        raise ValueError(f"the trained layers' weights don't fit: missing {sorted(set(got.missing_keys) - experts)[:5]}, "
                         f"unexpected {got.unexpected_keys[:5]}")
    for name in experts:
        *parents, leaf = name.split(".")
        mod, t = model.get_submodule(".".join(parents)), weights[name]
        if t.shape != mod._buffers[leaf].shape or t.dtype != mod._buffers[leaf].dtype:
            raise ValueError(f"{name}: {t.dtype} {list(t.shape)}, not {mod._buffers[leaf].dtype} "
                             f"{list(mod._buffers[leaf].shape)}")
        mod._buffers[leaf] = t
    return freeze(model)

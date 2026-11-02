"""Time DeepSeek-V4.1-Flash's top-layers route on this machine's GPU: the preparer's forward and the miner's training.

    python scripts/deepseek_top_bench.py [--tokens 1024,2048,4096] [--step-tokens 8192] [--micro 1024]

Uses the cached real layer 2 (scripts/deepseek_bench.py --fetch-only) as a stand-in for every layer, played in the
roles the trained layers have when four are trained (DEEPSEEK.md §8): layer 36, an index source picking with layer
20's index keys, and layers 37-39, which reuse 36's picks and 20's compressed keys, all at compress ratio 1. The
weights are the same tensors renamed, so the shapes and the work are the real ones (only the values aren't).

- Precompute: one layer's forward, without autograd, per token, in each role, the layer on the GPU (the preparer
  runs each layer on a long batch of tokens before loading the next).
- Training: deepseek_top.train_step on the four top layers and the head, layer-major from pinned RAM: one layer's
  weights on the GPU at a time, every micro-batch through it, then the next; backward the same way in reverse.
  Reports seconds per step and per token, and peak GPU memory.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from vibehash import deepseek, deepseek_top as top  # noqa: E402
from vibehash.deepseek import Choices, Config, Model, Side  # noqa: E402

LAYER2 = "model-00005-of-00048.safetensors"


def tokens(n: int) -> torch.Tensor:
    import json
    ids = json.loads((ROOT / "tests" / "data" / "deepseek_readme_tokens.json").read_text())["ids"]
    return torch.tensor((ids * (n // len(ids) + 1))[:n])


def as_layer(state: dict, layer: int, cfg: Config) -> dict:
    """Layer 2's tensors under `layer`'s names, without the parts that layer doesn't have."""
    out = {}
    for k, v in state.items():
        name = k.replace("layers.2.", f"layers.{layer}.", 1)
        if ".compressor." in name and layer not in cfg.kv_source_layers:
            continue
        if ".indexer." in name and (layer not in cfg.index_source_layers or
                                    (layer not in cfg.kv_source_layers and name.split(".indexer.")[1].startswith(("wk.", "k_norm.")))):
            continue
        out[name] = v
    return out


def random_picks(cfg: Config, s: int, dev) -> torch.Tensor:
    """Picks in the form an index source makes them (ascending, -1 last), from random scores: what 37-39 reuse."""
    g = torch.Generator(device=dev).manual_seed(1)
    visible = torch.arange(1, s + 1, device=dev).unsqueeze(-1)
    score = torch.rand(1, s, s, generator=g, device=dev).masked_fill(torch.arange(s, device=dev) >= visible, -1)
    idx = score.topk(min(cfg.index_topk, s), dim=-1).indices.sort(dim=-1).values
    return torch.where(idx < visible, idx, -1)


def side_from_below(cfg: Config, s: int, dev) -> Side:
    """Layer 20's compressed keys and index keys for s tokens, in their real form (FP4-rounded), random values."""
    g = torch.Generator(device=dev).manual_seed(0)
    comp = deepseek.round_fp4(torch.randn(1, s, cfg.head_dim, generator=g, device=dev).bfloat16(), 16, True)
    idx = deepseek.round_fp4(torch.randn(1, s, cfg.index_head_dim, generator=g, device=dev).bfloat16())
    return Side(comp, idx)


class Slots:
    """The trained layers' weights in pinned RAM, one layer's on the GPU at a time (deepseek_top.train_step's load)."""

    def __init__(self, model: Model, dev):
        self.model, self.dev, self.on = model, dev, None
        self.cpu = {name: {k: b for k, b in blk.named_buffers()} for name, blk in model.layers.items()}

    def __call__(self, layer):
        if self.on is not None:
            for path, b in self.cpu[self.on].items():
                _set(self.model.layers[self.on], path, b)
        self.on = None if layer is None else str(layer)
        if self.on is not None:
            for path, b in self.cpu[self.on].items():
                _set(self.model.layers[self.on], path, b.to(self.dev, non_blocking=True))


def _set(module, path, tensor):
    *parents, leaf = path.split(".")
    for p in parents:
        module = getattr(module, p)
    module._buffers[leaf] = tensor


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokens", default="1024,2048,4096")
    ap.add_argument("--step-tokens", type=int, default=8192)
    ap.add_argument("--micro", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()
    deepseek.configure_torch()
    dev, cfg = "cuda", Config.flash()
    used = torch.cuda.mem_get_info()
    print(f"{torch.cuda.get_device_name()}: {(used[1] - used[0]) / 2**30:.1f} of {used[1] / 2**30:.1f} GiB in use "
          "by other processes at the start", flush=True)
    state = deepseek.read_safetensors(deepseek.cache_dir() / LAYER2)
    state = {k: v.pin_memory() for k, v in state.items() if not k.endswith(deepseek.IMAGE_ONLY)}
    embed = deepseek.load_embedding()

    print("\nprecompute: one layer's forward, no autograd, the layer on the GPU", flush=True)
    print(f"{'role':>34} {'tokens':>7} {'seconds':>8} {'ms/token':>9}")
    for layer, role in ((2, "kv + index source, ratio 2 (2)"), (36, "index source, ratio 1 (36)"),
                        (37, "reuses keys and picks (37-39)")):
        block = Model(cfg, [layer], embed=False, head=False, device="meta").load_weights(
            {k: v.to(dev) for k, v in as_layer(state, layer, cfg).items()}).layers[str(layer)]
        for n in (int(t) for t in args.tokens.split(",")):
            x = embed[tokens(n)].to(dev)[None, :, None].repeat(1, 1, cfg.hc_mult, 1)
            pre = torch.zeros(1, n, cfg.hc_mult, device=dev)
            pre[..., 0] = 1
            side = side_from_below(cfg, n, dev) if layer != 2 else Side()
            if layer == 37:
                side.picks = random_picks(cfg, n, dev)
            for rep in range(2):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    block(x, pre, Side(side.compress_kv, side.index_k, side.picks), Choices())
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
            print(f"{role:>34} {n:>7} {dt:>8.2f} {dt / n * 1e3:>9.3f}", flush=True)
        del block
        torch.cuda.empty_cache()

    print(f"\ntraining: layers 36-39 and the head, layer-major from pinned RAM, {args.step_tokens} tokens a step in "
          f"micro-batches of {args.micro}", flush=True)
    cut = top.Cut(cfg, 36)
    upper = {}
    for layer in cut.trained:
        upper.update(as_layer(state, layer, cfg))
    head = deepseek.read_safetensors(deepseek.cache_dir() / "model-00043-of-00048.safetensors",
                                     {"head.weight", "norm.weight"}) if (deepseek.cache_dir() /
                                                                         "model-00043-of-00048.safetensors").exists() else None
    if head is None:  # the head isn't cached: a random one of its shape (the work is the same)
        g = torch.Generator().manual_seed(0)
        head = {"head.weight": (torch.randn(cfg.vocab_size, cfg.dim, generator=g) * cfg.dim ** -0.5).bfloat16(),
                "norm.weight": torch.ones(cfg.dim, dtype=torch.bfloat16)}
    meta = Model(cfg, cut.trained, embed=False, device="meta")
    frozen = {name for name, _ in meta.named_buffers()}  # the quantised matrices: in pinned RAM until their turn
    model = top.freeze(meta.load_weights({**{k: v if k in frozen else v.to(dev) for k, v in upper.items()},
                                          **{k: v.to(dev) for k, v in head.items()}}))
    slots = Slots(model, dev)
    adapters = top.Adapters.fresh(model, cut, device=dev)
    adapters.attach(model)
    micro = []
    for j in range(args.step_tokens // args.micro):
        t = tokens(args.micro * (j + 1))[-args.micro:][None]
        x = embed[t[0]].to(dev)[None, :, None].repeat(1, 1, cfg.hc_mult, 1)
        pre = torch.zeros(1, args.micro, cfg.hc_mult, device=dev)
        pre[..., 0] = 1
        mask = torch.ones_like(t, dtype=torch.bool)
        mask[:, -1] = False
        micro.append(top.Example(t.to(dev), mask.to(dev), x, pre, side_from_below(cfg, args.micro, dev)))
    for step in range(1, args.steps + 1):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        loss = top.train_step(model, cut, adapters, micro, step, 1e-4, load=slots)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"  step {step}: loss {loss:.3f}, {dt:.1f} s, {dt / args.step_tokens * 1e3:.2f} ms a token "
              f"({dt / args.step_tokens / 4 * 1e3:.2f} a layer), peak {peak:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()

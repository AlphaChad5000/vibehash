"""What one DeepSeek-V4.1-Flash layer keeps for its backward, per token, by stage: the GPU memory a top-layers miner
needs beyond the weights (DEEPSEEK.md §8). Runs on the CPU in under 2 GB, with no download.

    python scripts/deepseek_top_memory.py [--tokens 128]

The real layer's sizes (5,120 wide, 64 heads of 512, the indexer, the 2,304-wide experts), with random weights and
8 routed experts instead of 384: what a layer keeps per token doesn't depend on how many experts there are, only on
the 6 each token uses. Counted with autograd's saved-tensor hooks (each storage once), in deepseek_top.TRAIN's float32,
for layer 36's role (an index source picking with layer 20's keys) and 37's (reusing them and 36's picks):
- whole layer: one micro-batch's graph of the whole layer, which train_step keeps while it differentiates that layer;
- before the MoE: the attention stage's graph, train_step_streamed's largest (its MoE stages keep a few rows each).
Also the step-long tensors each schedule keeps per token and layer, and the big transients that aren't saved.
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from vibehash import deepseek, deepseek_top as top  # noqa: E402
from vibehash.deepseek import Choices, Config, Model, Side  # noqa: E402


class Saved:
    """Bytes autograd saves for the backward inside a with-block, each storage counted once, leaving out the
    model's own tensors (weights, adapters), which are there anyway."""

    def __init__(self, *owned):
        self.owned = {t.untyped_storage().data_ptr() for t in owned}

    def __enter__(self):
        self.seen, self.bytes = set(self.owned), 0

        def pack(t):
            key = t.untyped_storage().data_ptr()
            if key not in self.seen:
                self.seen.add(key)
                self.bytes += t.untyped_storage().nbytes()
            return t
        self.hooks = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)
        self.hooks.__enter__()
        return self

    def __exit__(self, *exc):
        self.hooks.__exit__(*exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokens", type=int, default=128)
    args = ap.parse_args()
    n = args.tokens
    cfg = replace(Config.flash(), n_routed_experts=8)
    num = top.TRAIN
    g = torch.Generator().manual_seed(0)
    print(f"real sizes, 8 routed experts, {n} tokens, float32 (deepseek_top.TRAIN), counted on the CPU")
    for layer in (36, 37):
        state = deepseek.random_weights(cfg, 1, layers=[layer], embed=False, head=False)
        model = top.freeze(Model(cfg, [layer], num, embed=False, head=False).load_weights(state))
        ad = top.Adapters.fresh(model, top.Cut(cfg, layer), seed=1)
        ad.b = [torch.randn(b.shape, generator=g).mul_(1e-3).requires_grad_() for b in ad.b]  # as if trained
        ad.attach(model)
        block = model.layers[str(layer)]
        x = torch.randn(1, n, cfg.hc_mult, cfg.dim, generator=g)
        pre = torch.full((1, n, cfg.hc_mult), 1 / cfg.hc_mult)
        comp = deepseek.round_fp4(torch.randn(1, n, cfg.head_dim, generator=g), 16, True)
        idx = deepseek.round_fp4(torch.randn(1, n, cfg.index_head_dim, generator=g))
        side = Side(comp, idx)
        if layer == 37:  # 36's picks
            with torch.no_grad():
                c36 = Choices()
                s36 = Model(cfg, [36], num, embed=False, head=False).load_weights(
                    deepseek.random_weights(cfg, 2, layers=[36], embed=False, head=False))
                s36.layers["36"](x, pre, Side(comp, idx), c36)
                side.picks = c36.picks[36]
                del s36
        choices = Choices()
        with torch.no_grad():
            block(x, pre, replace(side), choices)
        owned = [*model.parameters(), *model.buffers(), *ad.a, *ad.b]
        with Saved(*owned) as whole:
            y, p = block(x.requires_grad_(), pre.requires_grad_(), replace(side), choices)
        del y, p
        with Saved(*owned) as before:
            out = block.before_ffn(x, pre, replace(side), choices)
        del out
        role = "36: index source" if layer == 36 else "37: reuses keys, picks"
        print(f"  layer {role:<24} whole layer {whole.bytes / n / 1024:7.1f} KB a token   "
              f"before the MoE {before.bytes / n / 1024:7.1f} KB a token")
    width = cfg.hc_mult * cfg.dim
    print(f"step-long, per token and trained layer: its input stream {width * 4 / 1024:.0f} KB (float32; the "
          f"first layer's is the precomputed bf16, {width * 2 / 1024:.0f} KB); streamed also keeps the experts' sum "
          f"{cfg.dim * 4 / 1024:.0f} KB")
    q = deepseek.Indexer.QUERY_CHUNK
    print(f"transients, not saved: the indexer scores {q} queries at once, {cfg.index_n_heads} heads x every "
          f"compressed token: {4 * q * cfg.index_n_heads / 2**20:.0f} MB per 1,000 tokens of context in float32, "
          f"plus a [tokens x tokens] float32 score matrix ({4 / 2**20 * 1e6:.1f} MB per million entries)")


if __name__ == "__main__":
    main()

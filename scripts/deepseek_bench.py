"""Time one real DeepSeek-V4.1-Flash layer on this machine's GPU, forward and backward: DEEPSEEK.md's measured numbers.

    python scripts/deepseek_bench.py [--layer 2] [--tokens 512,1024,2048,4096] [--experts-in-ram] [--fetch-only]

Fetches the layer (7.4 GB) and the token embedding (1.3 GB) into ~/.vibehash/cache/deepseek unless they're there
(also what tests/test_deepseek.py's opt-in RealLayerTests need), then for each token count runs the layer on real
token embeddings (this repository's README.md, tests/data/deepseek_readme_tokens.json, repeated past 4096) as
fine-tuning would: a forward with autograd, and a backward to the input stream and the layer's bf16 and float32
weights. Reports seconds and peak GPU memory, beyond the weights. --experts-in-ram keeps the 384 routed experts in
pinned CPU memory and copies each to the GPU when it's used, as a 12 GB card holding several layers would.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from vibehash import deepseek  # noqa: E402
from vibehash.deepseek import Choices, Config, Model, Side  # noqa: E402


def tokens(n: int) -> torch.Tensor:
    ids = json.loads((ROOT / "tests" / "data" / "deepseek_readme_tokens.json").read_text())["ids"]
    return torch.tensor((ids * (n // len(ids) + 1))[:n])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--tokens", default="512,1024,2048,4096")
    ap.add_argument("--experts-in-ram", action="store_true")
    ap.add_argument("--fetch-only", action="store_true")
    args = ap.parse_args()
    files = deepseek.layer_files(args.layer)
    for f in files:
        print("have", deepseek.fetch_file(f, log=print))
    embed = deepseek.load_embedding(log=print)
    if args.fetch_only:
        return
    deepseek.configure_torch()
    cfg, dev = Config.flash(), "cuda"
    print(torch.cuda.get_device_name(), "|", deepseek.describe_layer(cfg, args.layer),
          "| experts in", "pinned CPU memory" if args.experts_in_ram else "GPU memory")
    state = {}
    for f, names in files.items():
        for name, t in deepseek.read_safetensors(deepseek.cache_dir() / f, set(names)).items():
            routed = ".ffn.experts." in name
            state[name] = t.pin_memory() if routed and args.experts_in_ram else t.to(dev)
    block = Model(cfg, [args.layer], embed=False, head=False, device="meta").load_weights(state).layers[str(args.layer)]
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated()
    print(f"weights on the GPU: {weights / 2**30:.2f} GiB")
    print(f"{'tokens':>7} {'forward s':>10} {'backward s':>11} {'inference s':>12} {'peak GiB':>9} {'per token ms':>13}")
    for n in [int(t) for t in args.tokens.split(",")]:
        x = embed[tokens(n)].to(dev)[None, :, None].repeat(1, 1, cfg.hc_mult, 1).requires_grad_()
        pre = torch.zeros(1, n, cfg.hc_mult, device=dev)
        pre[..., 0] = 1
        g = torch.randn(1, n, cfg.hc_mult, cfg.dim, device=dev)
        times = []
        for _ in range(2):  # the first run warms up kernels and the allocator; the second is reported
            for p in block.parameters():
                p.grad = None
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            y, pre_out = block(x, pre, Side(), Choices())
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            ((y.float() * g).sum() + pre_out.sum()).backward()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            peak = torch.cuda.max_memory_allocated() - weights
            with torch.no_grad():
                t3 = time.perf_counter()
                block(x, pre, Side(), Choices())
                torch.cuda.synchronize()
                t4 = time.perf_counter()
            times.append((t1 - t0, t2 - t1, t4 - t3, peak))
            del y, pre_out
            x.grad = None
        f, b, inf, peak = times[-1]
        print(f"{n:>7} {f:>10.2f} {b:>11.2f} {inf:>12.2f} {peak / 2**30:>9.2f} {(f + b) / n * 1e3:>13.2f}")
        del x, g
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

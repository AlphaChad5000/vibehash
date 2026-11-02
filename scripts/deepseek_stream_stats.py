"""What the split prep's hand-offs carry, measured on DeepSeek-V4.1-Flash's real cached layers and real text, on the
CPU in under 2 GB of RAM (experts read off the disk one at a time, as each is needed):

    python scripts/deepseek_stream_stats.py --text chats.json [--route-seqs 32] [--copy-seqs 4] [--seq 2048]
    python scripts/deepseek_stream_stats.py --text chats.json --layer0 [--route-seqs 64]

--text is a JSON list of token-id lists (DeepSeek's tokenizer), packed end to end into sequences of --seq tokens.

- Routing: which of layer 2's 384 experts its tokens go to (top 6), over --route-seqs sequences. Layer 2's input is
  the embedding: layers 0 and 1 aren't cached, so it's a stand-in for theirs. With --layer0, layer 0's instead,
  whose real input is the embedding: only its 170 MB of tensors outside the experts are needed (routing doesn't
  run the experts), read by range from the Hub once into the cache.
- The stream through a chain of real layers: the embedding, layer 2 in layer 20's role (the kv source that layers
  36-39 read), then layers 36, 37, 38 and 39, over --copy-seqs sequences. After each layer: how alike the 4
  hyper-connection copies are (the spectrum of their 4x4 Gram matrix, per token), what lossless compression and
  some lossy codes do to them, and where each layer sends its tokens.
- FP8 at every cut: the same chain with the stream rounded to FP8 after every layer, as one-layer prep pieces handing
  off in FP8 would do, against the exact chain, down to the head's next-token distribution. And a control: the last
  bit of half the values flipped after every layer instead (about 0.4%), to tell what FP8 does from what the chain
  does to any change.
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from compression import zstd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vibehash import deepseek as ds  # noqa: E402
from vibehash.deepseek import Choices, Config, Model, Side  # noqa: E402

FILES = {2: "model-00005-of-00048.safetensors", 36: "model-00039-of-00048.safetensors",
         37: "model-00040-of-00048.safetensors", 38: "model-00041-of-00048.safetensors",
         39: "model-00042-of-00048.safetensors"}
EMBED, HEAD = "model-00002-of-00048.safetensors", "model-00043-of-00048.safetensors"
CHAIN = ((2, 20), (36, 36), (37, 37), (38, 38), (39, 39))  # (the file's layer, the role it plays)


class File:
    """A safetensors file read one tensor at a time."""

    def __init__(self, name: str):
        self.path = ds.cache_dir() / name
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header, self.start = json.loads(f.read(n)), 8 + n
        self.header.pop("__metadata__", None)

    def read(self, name: str) -> torch.Tensor:
        h = self.header[name]
        b, e = h["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.start + b)
            raw = bytearray(f.read(e - b))
        return torch.frombuffer(raw, dtype=torch.uint8).view(ds.SAFETENSORS_DTYPES[h["dtype"]]).reshape(h["shape"])

    def rows(self, name: str, which) -> torch.Tensor:
        """Rows of a bf16 matrix (a slice or a list of indices), memory-mapped: only those are read."""
        h = self.header[name]
        m = np.memmap(self.path, np.uint16, "r", self.start + h["data_offsets"][0], tuple(h["shape"]))
        return torch.from_numpy(np.array(m[which])).view(torch.bfloat16)


class Layer:
    """One layer's block with everything but its routed experts loaded; experts come in one at a time."""

    def __init__(self, cfg: Config, src: int, role: int, state: dict | None = None):
        """state: the layer's tensors outside the experts, if not from a cached file (then it can only route)."""
        self.file, self.src, self.role = None if state else File(FILES[src]), src, role
        self.block = Model(cfg, [role], embed=False, head=False, device="meta").layers[str(role)]
        prefix = f"layers.{src}."
        state = {k: (state[prefix + k] if state else self.file.read(prefix + k)) for k in self.block.state_dict()
                 if ".experts." not in k}
        missing = self.block.load_state_dict(state, strict=False, assign=True).missing_keys
        assert all(".experts." in k for k in missing), missing

    def expert(self, e: int):
        mod = self.block.ffn.experts[e]
        for w in ("w1", "w2", "w3"):
            q = getattr(mod, w)
            for b in ("weight", "scale"):
                q._buffers[b] = self.file.read(f"layers.{self.src}.ffn.experts.{e}.{w}.{b}")
        return mod

    def drop(self, e: int):
        for q in (self.block.ffn.experts[e].w1, self.block.ffn.experts[e].w2, self.block.ffn.experts[e].w3):
            for b in ("weight", "scale"):
                q._buffers[b] = torch.empty_like(q._buffers[b], device="meta")


@torch.no_grad()
def run(layer: Layer, groups: list[list], counts: list | None = None, route_only: bool = False) -> None:
    """The layer over each group's states [(x, pre, side, choices)] in turn, updated in place; each expert read once
    a group (one group's working state at a time keeps a 2 x 2048-token group under 2 GB). counts gets each
    sequence's expert histogram."""
    for states in groups:
        _run(layer, states, counts, route_only)


def _run(layer: Layer, states: list, counts: list | None, route_only: bool) -> None:
    block = layer.block
    moe, hi, n_exp = block.ffn, block.num.hi, len(block.ffn.experts)
    parts = []
    for j, (x, pre, side, choices) in enumerate(states):
        h, x_mid, post, comb, ffn_pre = block.before_ffn(x, pre, side, choices)
        flat = h.reshape(-1, h.size(-1))
        chosen, gates = moe.route(flat, choices)
        if counts is not None:
            counts.append(torch.bincount(chosen.flatten(), minlength=n_exp))
        if not route_only:
            parts.append(dict(states=states, j=j, flat=flat, x_mid=x_mid, post=post, comb=comb, pre=ffn_pre,
                              gates=gates, shape=h.shape, acc=torch.zeros(flat.shape, dtype=hi),
                              assign={e: (tok, slot) for e, tok, slot in moe.assignments(chosen)}))
    for e in sorted({e for p in parts for e in p["assign"]}):
        expert = layer.expert(e)
        for p in parts:
            if e in p["assign"]:
                tok, slot = p["assign"][e]
                p["acc"].index_add_(0, tok, expert(p["flat"][tok], p["gates"][tok, slot, None]).to(hi))
        layer.drop(e)
    for p in parts:
        y = moe.combine(p["acc"], p["flat"]).view(p["shape"])
        x, pre, side, choices = p["states"][p["j"]]
        p["states"][p["j"]] = (block.after_ffn(y, p["x_mid"], p["post"], p["comb"]), p["pre"], side, choices)


def start(embed: File, ids: list[int], cfg: Config):
    x = embed.rows("embed.weight", np.asarray(ids))[None, :, None].repeat(1, 1, cfg.hc_mult, 1)
    pre = torch.zeros(1, len(ids), cfg.hc_mult)
    pre[..., 0] = 1
    return x, pre


def histogram(counts: list[torch.Tensor], label: str, seq: int) -> dict:
    total = torch.stack(counts).sum(0).double()
    n_tok = len(counts) * seq
    share = total / total.sum()
    per_seq = [int((c > 0).sum()) for c in counts]
    srt = share.sort(descending=True).values
    out = dict(tokens=n_tok, experts_touched=int((total > 0).sum()), min_count=int(total.min()),
               max_count=int(total.max()), mean_count=float(total.mean()), cv=float(total.std() / total.mean()),
               top10pct_share=float(srt[:len(srt) // 10].sum()), max_over_mean=float(total.max() / total.mean()),
               min_over_mean=float(total.min() / total.mean()), experts_per_seq_mean=float(np.mean(per_seq)),
               experts_per_seq_min=int(min(per_seq)))
    seen, curve, n = torch.zeros_like(total, dtype=torch.bool), {}, 1
    for i, c in enumerate(counts):  # experts touched by the first n sequences, n doubling
        seen |= c > 0
        if i + 1 == n or i + 1 == len(counts):
            curve[(i + 1) * seq] = int(seen.sum())
            n *= 2
    out["touched_by_tokens"] = curve
    print(f"  {label}: {n_tok} tokens, {out['experts_touched']} of {len(total)} experts touched; counts "
          f"{out['min_count']}..{out['max_count']} (mean {out['mean_count']:.0f}, CV {out['cv']:.2f}); the busiest "
          f"10% take {out['top10pct_share']:.1%}; a {seq}-token sequence touches {out['experts_per_seq_mean']:.0f} "
          f"(min {out['experts_per_seq_min']}); touched by the first n tokens: {curve}", flush=True)
    return out


def zstd_ratio(data: bytes, level: int = 3) -> float:
    return len(zstd.compress(data, level)) / len(data)


def fp8_bytes(x: torch.Tensor) -> bytes:
    from vibehash.deepseek_top import _fp8_rows
    q, s = _fp8_rows(x.float())
    return q.view(torch.uint8).numpy().tobytes() + s.view(torch.uint8).numpy().tobytes()


def jitter(x: torch.Tensor, seed: int) -> torch.Tensor:
    """x with the last mantissa bit of half its values flipped at random: a perturbation of about 0.2%, far below
    FP8's, to tell a chain that amplifies any change from one that FP8 alone throws off."""
    g = torch.Generator().manual_seed(seed)
    flip = torch.randint(0, 2, x.shape, generator=g, dtype=torch.int16)
    return (x.contiguous().view(torch.int16) ^ flip).view(torch.bfloat16)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).square().mean().sqrt() / b.float().square().mean().sqrt())


def copies(x: torch.Tensor, label: str, sample: int = 1024) -> dict:
    """How alike the 4 copies are, and what codes would make of them, over `sample` tokens spread through x
    [n, hc, dim] bf16 (all of them in float32 would be gigabytes)."""
    x = x[::max(1, len(x) // sample)][:sample]
    xf = x.float()
    gram = xf @ xf.transpose(1, 2)  # [n, 4, 4]: its eigenvalues are the copies' squared singular values
    ev = torch.linalg.eigvalsh(gram).flip(-1).clamp_min(0)
    energy = ev.cumsum(-1) / ev.sum(-1, keepdim=True)
    rank_err = (1 - energy[:, :3]).clamp_min(0).sqrt()  # rank-r reconstruction's relative error, r = 1, 2, 3
    mean = xf.mean(1, keepdim=True)
    dev = float((xf - mean).square().mean().sqrt() / xf.square().mean().sqrt())
    cos = torch.nn.functional.normalize(xf, dim=-1)
    cos = (cos @ cos.transpose(1, 2))[:, *torch.triu_indices(4, 4, 1)].mean()
    # codes, on the first `sample` tokens
    s = x[:sample]
    raw = s.contiguous().view(torch.uint8).numpy()
    planes = np.concatenate([raw[..., 1::2].ravel(), raw[..., 0::2].ravel()]).tobytes()  # high bytes, then low
    fp8 = ds.round_fp8(s)
    m8 = ds.round_fp8(s.float().mean(1, keepdim=True).to(s.dtype))
    d4 = ds.round_fp4((s.float() - m8.float()).to(s.dtype))
    four = (m8.float() + d4.float())
    three = torch.cat([four[:, :3], 4 * m8.float() - four[:, :3].sum(1, keepdim=True)], 1)
    out = dict(rank1_err=float(rank_err[:, 0].mean()), rank2_err=float(rank_err[:, 1].mean()),
               rank3_err=float(rank_err[:, 2].mean()), rank2_err_p95=float(rank_err[:, 1].quantile(0.95)),
               deviation_rms=dev, cosine=float(cos), stream_rms=float(xf.square().mean().sqrt()),
               zstd_bf16=zstd_ratio(raw.tobytes()), zstd_bf16_planes=zstd_ratio(planes),
               zstd_bf16_planes_19=zstd_ratio(planes, 19), zstd_fp8=zstd_ratio(fp8_bytes(s)),
               fp8_err=rel(fp8, s), mean8_dev4x4_err=rel(four, s), mean8_dev4x3_err=rel(three, s))
    print(f"  {label}: copies' cosine {out['cosine']:.3f}, deviation from their mean {dev:.1%} of the stream; "
          f"rank-1/2/3 error {out['rank1_err']:.1%}/{out['rank2_err']:.1%}/{out['rank3_err']:.1%}; zstd: bf16 "
          f"{out['zstd_bf16']:.3f}, byte planes {out['zstd_bf16_planes']:.3f} (level 19: "
          f"{out['zstd_bf16_planes_19']:.3f}), fp8 {out['zstd_fp8']:.3f}; lossy: fp8 {out['fp8_err']:.2%}, fp8 mean "
          f"+ fp4 deviations {out['mean8_dev4x4_err']:.2%} (x3: {out['mean8_dev4x3_err']:.2%})", flush=True)
    return out


@torch.no_grad()
def head_compare(cfg: Config, head: File, exact, other: dict, ids_list, positions: int = 512) -> dict:
    """Next-token distributions from the exact chain's last stream and from others: KL, top-1 agreement, and the
    loss on the text, over the first `positions` positions of each sequence. Two passes over the head, a slice of
    the vocabulary at a time: the logits whole would be gigabytes."""
    norm = ds.RMSNorm(cfg.dim, cfg.norm_eps, ds.Numerics())
    norm.weight.data = head.read("norm.weight")
    hi = torch.float32

    def hidden(states):
        return torch.cat([norm(ds.hc_pre(x, pre.to(hi), hi))[0, :positions] for x, pre, _, _ in states]).to(hi)

    hs = {"exact": hidden(exact), **{k: hidden(v) for k, v in other.items()}}
    targets = torch.cat([torch.tensor(ids[1:positions + 1]) for ids in ids_list])
    V, chunk = cfg.vocab_size, 8192
    lse = {k: torch.full((len(targets),), -torch.inf) for k in hs}
    best = {k: torch.full((len(targets),), -torch.inf) for k in hs}
    arg = {k: torch.zeros(len(targets), dtype=torch.long) for k in hs}
    tgt = {k: torch.zeros(len(targets)) for k in hs}
    for v in range(0, V, chunk):  # pass 1: normalisers, argmax, the targets' logits
        w = head.rows("head.weight", slice(v, min(v + chunk, V))).to(hi)
        for k, h in hs.items():
            z = h @ w.T
            lse[k] = torch.logaddexp(lse[k], z.logsumexp(-1))
            m, i = z.max(-1)
            arg[k] = torch.where(m > best[k], i + v, arg[k])
            best[k] = torch.maximum(best[k], m)
            inside = (targets >= v) & (targets < v + z.size(1))
            tgt[k][inside] = z[inside, targets[inside] - v]
    kl = {k: torch.zeros(len(targets)) for k in hs}
    for v in range(0, V, chunk):  # pass 2: KL(exact || other)
        w = head.rows("head.weight", slice(v, min(v + chunk, V))).to(hi)
        le = hs["exact"] @ w.T - lse["exact"][:, None]
        for k, h in hs.items():
            kl[k] += (le.exp() * (le - (h @ w.T - lse[k][:, None]))).sum(-1)
    out = {}
    for k in hs:
        nll, top1 = float((lse[k] - tgt[k]).mean()), float((arg[k] == arg["exact"]).float().mean())
        out[k] = dict(nll=nll, kl_from_exact=float(kl[k].mean()), top1_agrees=top1)
        kl_k = out[k]["kl_from_exact"]
        print(f"  head, {k}: loss {nll:.3f}, KL from exact {kl_k:.2e}, top-1 agrees {top1:.1%}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--text", required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--route-seqs", type=int, default=32)
    ap.add_argument("--copy-seqs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--layer0", action="store_true", help="only layer 0's routing, on its real input")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = Config.flash()
    flat = [t for ids in json.loads(Path(args.text).read_text()) for t in ids]
    seqs = [flat[i:i + args.seq + 1] for i in range(0, len(flat) - args.seq - 1, args.seq)]  # +1: the next token
    print(f"{len(flat)} tokens of text, {len(seqs)} sequences of {args.seq}; {args.threads} CPU threads", flush=True)
    embed, results = File(EMBED), {}
    if args.layer0:
        path = ds.cache_dir() / "layer0-nonexpert.pt"
        if not path.exists():
            from vibehash.hub import Hub
            hub = Hub(backoff=2.0)
            name = hub.weight_map(ds.REPO, ds.COMMIT)["layers.0.attn.wq_a.weight"]
            header, remote = hub.header(ds.REPO, ds.COMMIT, name), hub.open(ds.REPO, ds.COMMIT, name)
            part = {}
            for k, (dtype, shape, b, e) in header.tensors.items():
                if k.startswith("layers.0.") and ".experts." not in k:
                    raw = torch.frombuffer(bytearray(remote.read(header.data_start + b, e - b)), dtype=torch.uint8)
                    part[k] = raw.view(ds.SAFETENSORS_DTYPES[dtype]).reshape(shape)
            torch.save(part, path)
        print(f"\nrouting: layer 0 over {args.route_seqs} sequences (its real input, the embedding)", flush=True)
        t0, layer, counts = time.perf_counter(), Layer(cfg, 0, 0, torch.load(path)), []
        for ids in seqs[:args.route_seqs]:
            x, pre = start(embed, ids[:-1], cfg)
            run(layer, [[(x, pre, Side(), Choices())]], counts, route_only=True)
        results["routing_layer0"] = histogram(counts, "layer 0", args.seq)
        results["routing_layer0"]["seconds"] = time.perf_counter() - t0
        results["routing_layer0"]["counts"] = torch.stack(counts).sum(0).tolist()
        if args.out:
            Path(args.out).write_text(json.dumps(results, indent=1))
        return

    print(f"\nrouting: layer 2 over {args.route_seqs} sequences (embedding in)", flush=True)
    t0 = time.perf_counter()
    l2, counts = Layer(cfg, 2, 2), []
    for ids in seqs[-args.route_seqs:]:  # the other end of the text from the chain's
        x, pre = start(embed, ids[:-1], cfg)
        run(l2, [[(x, pre, Side(), Choices())]], counts, route_only=True)
    results["routing_layer2"] = histogram(counts, "layer 2", args.seq)
    results["routing_layer2"]["seconds"] = time.perf_counter() - t0
    del l2

    print(f"\nthe chain, {args.copy_seqs} sequences of {args.seq}: embedding -> 2 (as 20) -> 36 -> 37 -> 38 -> 39",
          flush=True)
    ids_list = seqs[:args.copy_seqs]
    exact = [(*start(embed, ids[:-1], cfg), Side(), Choices()) for ids in ids_list]
    rounded = [(x.clone(), pre.clone(), Side(), Choices()) for x, pre, _, _ in exact]
    jittered = [(x.clone(), pre.clone(), Side(), Choices()) for x, pre, _, _ in exact]
    results["chain"] = []
    for src, role in CHAIN:
        t0 = time.perf_counter()
        layer, counts = Layer(cfg, src, role), []
        run(layer, [exact, rounded, jittered], counts)
        rounded[:] = [(ds.round_fp8(x), pre, side, ch) for x, pre, side, ch in rounded]  # the hand-off, in FP8
        jittered[:] = [(jitter(x, 100 * role + i), pre, side, ch) for i, (x, pre, side, ch) in enumerate(jittered)]
        label = f"after {role}" + (" (layer 2's weights)" if src != role else "")
        x = torch.cat([s[0][0] for s in exact])
        step = dict(layer=role, src=src, seconds=time.perf_counter() - t0, copies=copies(x, label),
                    routing=histogram(counts[:len(exact)], f"routing in {role}", args.seq),
                    fp8_every_cut_err=rel(torch.cat([s[0][0] for s in rounded]), x),
                    jitter_every_cut_err=rel(torch.cat([s[0][0] for s in jittered]), x))
        print(f"  the stream's distance from exact, rounded at every cut so far: FP8 {step['fp8_every_cut_err']:.2%}, "
              f"last-bit jitter {step['jitter_every_cut_err']:.2%} ({step['seconds']:.0f} s)", flush=True)
        results["chain"].append(step)
        del layer
    once = [(ds.round_fp8(x), pre, side, ch) for x, pre, side, ch in exact]
    results["head"] = head_compare(cfg, File(HEAD), exact, {"fp8 at the last cut only": once,
                                                             "fp8 at every cut": rounded,
                                                             "last-bit jitter at every cut": jittered}, ids_list)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

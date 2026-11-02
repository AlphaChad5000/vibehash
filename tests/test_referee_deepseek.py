"""The referee's DeepSeek ops (referee_deepseek.py) against deepseek.py: each op's torch expression computed on the
tiny config's random weights, in the reference's arithmetic (bf16, "ref") and in float32 ("f32"); every tile of its
output must pass the referee, and a number changed by a few bf16 steps ("ref") or 0.1% ("f32") must not."""
import unittest

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

from vibehash import dispute, merkle
from vibehash import referee as rf

if torch is not None:
    from tests.test_deepseek import tiny_config
    from vibehash import deepseek as ds
    from vibehash import referee_deepseek as rd
    from vibehash.program import Op


def mk(kind, out, ins, attrs, shape, ishapes):
    return Op(kind, out, tuple(ins), attrs, (), tuple(shape), tuple(tuple(s) for s in ishapes))


@unittest.skipIf(torch is None, "needs PyTorch")
class ForwardOpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = tiny_config()
        cls.state = ds.random_weights(cls.cfg, 0)
        cls.src = rd.StateWeights(cls.state)

    def judge(self, op, before, out, label):
        before = {k: np.asarray(v.detach().float().cpu().numpy() if torch.is_tensor(v) else v, dtype=np.float32)
                  for k, v in before.items()}
        out = (out.detach().float().cpu().numpy() if torch.is_tensor(out) else np.asarray(out, np.float32)).reshape(op.shape)
        x = dispute.Inputs({k: v.shape for k, v in before.items()}, dispute.Outside(None, None, self.src, keep=False),
                           full=before)
        flat = out.reshape(-1)
        caught = tampered = 0
        for leaf in range(merkle.n_leaves(*op.shape)):
            said = flat[leaf * merkle.LEAF:(leaf + 1) * merkle.LEAF].astype(np.float64)
            ok, why = rf.judge(op, x, leaf, said)
            self.assertTrue(ok, f"{label}: {why}")
            j = int(np.argmax(np.where(np.isfinite(said), np.abs(said), -1)))  # the largest: the least cancellation
            if np.isfinite(said[j]) and said[j] != 0:
                t = said.copy()
                loose = op.attrs.get("num", "ref") == "f32" or op.kind in ("ds_hc_mixes", "ds_hc_split") or op.attrs.get("hi")
                # (attention's output is a weighted sum of values that mostly cancel: its float32 bound is ~0.5%)
                t[j] *= 1.5 if op.kind in rf.EXACT else 1.03 if not loose else 1.01 if op.kind == "ds_attn" else 1.001
                tampered += 1
                caught += not rf.judge(op, x, leaf, t)[0]
        # (a sum with heavy cancellation can hide a 0.1% change in its bound: hc_mixes' 256-term dot products)
        self.assertGreaterEqual(caught, tampered - (1 if op.kind == "ds_hc_mixes" else 0), label)

    def test_every_forward_op(self):
        cfg, state = self.cfg, self.state
        REF, F32N = ds.Numerics(), ds.Numerics(torch.float32, quantize=False)
        T = 12
        for numname, num in (("ref", REF), ("f32", F32N)):

            dt = num.dtype
            m = ds.Model(cfg, layers=[1, 2], num=num, embed=False, head=False, engram_table=False)
            m.load_state_dict({k: v for k, v in state.items() if k.startswith(("layers.1.", "layers.2."))}, strict=False)
            blk = m.layers["1"]
            att = blk.attn
            g = torch.Generator().manual_seed(1)
            x = (torch.randn(T, cfg.dim, generator=g)).to(dt)
            # rounding (exact)
            if numname == "ref":
                self.judge(mk("ds_round_fp8", "y", ["x"], {}, (T, cfg.dim), [(T, cfg.dim)]), {"x": x}, ds.round_fp8(x), "round_fp8")
                self.judge(mk("ds_round_fp4", "y", ["x"], {}, (T, cfg.dim), [(T, cfg.dim)]), {"x": x}, ds.round_fp4(x), "round_fp4")
                self.judge(mk("ds_round_fp4", "y", ["x"], {"block": 16, "e4m3": True}, (T, cfg.dim), [(T, cfg.dim)]), {"x": x},
                          ds.round_fp4(x, 16, True), "round_fp4 e4m3/16")
            xq = num.fp8(x)
            # qlinear wq_a
            y = att.wq_a.matmul(xq)
            self.judge(mk("ds_qlinear", "y", ["x"], {"weight": "layers.1.attn.wq_a", "num": numname}, y.shape, [xq.shape]), {"x": xq}, y, "qlinear wq_a")
            # grouped wo_a
            o = torch.randn(T, cfg.n_heads * cfg.head_dim, generator=g).to(dt)
            oq = num.fp8(o)
            yo = att.wo_a.matmul(oq.unflatten(-1, (cfg.o_groups, -1))).flatten(-2)
            self.judge(mk("ds_qlinear", "y", ["x"], {"weight": "layers.1.attn.wo_a", "groups": cfg.o_groups, "num": numname}, yo.shape, [oq.shape]),
                      {"x": oq}, yo, "qlinear wo_a grouped")
            # experts' FP4 w1
            e1 = blk.ffn.experts[0].w1.matmul(xq)
            self.judge(mk("ds_qlinear", "y", ["x"], {"weight": "layers.1.ffn.experts.0.w1", "num": numname}, e1.shape, [xq.shape]), {"x": xq}, e1, "qlinear expert fp4")
            # dense router (hi) and weights_proj (act)
            sc = F.linear(x.to(num.hi), blk.ffn.gate.weight.to(num.hi))
            self.judge(mk("ds_dense", "y", ["x"], {"weight": "layers.1.ffn.gate.weight", "hi": True, "num": numname}, sc.shape, [x.shape]), {"x": x}, sc, "dense router hi")
            if att.indexer is not None:
                wp = F.linear(x, att.indexer.weights_proj.weight.to(x.dtype))
                self.judge(mk("ds_dense", "y", ["x"], {"weight": "layers.1.attn.indexer.weights_proj.weight", "num": numname}, wp.shape, [x.shape]), {"x": x}, wp, "dense weights_proj")
                ws = wp * (cfg.index_head_dim ** -0.5 * cfg.index_n_heads ** -0.5)
                self.judge(mk("ds_scale", "y", ["x"], {"c": cfg.index_head_dim ** -0.5 * cfg.index_n_heads ** -0.5, "num": numname}, ws.shape, [wp.shape]), {"x": wp}, ws, "scale")
            # rmsnorm q_norm
            qa = att.wq_a(x)
            qr = att.q_norm(qa)
            self.judge(mk("ds_rmsnorm", "y", ["x"], {"weight": "layers.1.attn.q_norm.weight", "eps": cfg.norm_eps, "num": numname}, qr.shape, [qa.shape]), {"x": qa}, qr, "rmsnorm q_norm")
            # rope on q heads
            q = att.wq_b(qr)
            cos, sin = ds._rope_table(T, *cfg.rope_args(1))
            qh = ds._rope(q.unflatten(-1, (cfg.n_heads, cfg.head_dim))[None], cos, sin, num)[0].flatten(-2)
            self.judge(mk("ds_rope", "y", ["x"], {"rope": cfg.rope_args(1), "hd": cfg.head_dim, "num": numname}, qh.shape, [q.shape]), {"x": q}, qh, "rope q (yarn)")
            qi = ds._rope(q.unflatten(-1, (cfg.n_heads, cfg.head_dim))[None], cos, sin, num, inverse=True)[0].flatten(-2)
            self.judge(mk("ds_rope", "y", ["x"], {"rope": cfg.rope_args(1), "hd": cfg.head_dim, "inverse": True, "num": numname}, qi.shape, [q.shape]), {"x": q}, qi, "rope inverse")
            # hyper-connections
            stream = torch.randn(1, T, cfg.hc_mult, cfg.dim, generator=g).to(dt)
            h = stream.flatten(2).to(num.hi)
            mixes = F.linear(h, blk.hc_attn_fn.to(num.hi)) * torch.rsqrt(h.square().mean(-1, keepdim=True) + cfg.norm_eps)
            self.judge(mk("ds_hc_mixes", "y", ["x"], {"fn": "layers.1.hc_attn_fn", "eps": cfg.norm_eps}, mixes[0].shape, [stream[0].flatten(1).shape]),
                      {"x": stream[0].flatten(1)}, mixes[0], "hc_mixes")
            pre, post, comb = ds._sinkhorn_split(mixes, blk.hc_attn_scale.to(num.hi), blk.hc_attn_base.to(num.hi), cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps)
            split = torch.cat([pre, post, comb.flatten(-2)], -1)[0]
            self.judge(mk("ds_hc_split", "y", ["m"], {"scale": "layers.1.hc_attn_scale", "base": "layers.1.hc_attn_base", "hc": cfg.hc_mult,
                                                      "iters": cfg.hc_sinkhorn_iters, "eps": cfg.hc_eps}, split.shape, [mixes[0].shape]),
                      {"m": mixes[0]}, split, "hc_split (sinkhorn)")
            hp = ds.hc_pre(stream, pre, num.hi)[0]
            self.judge(mk("ds_hc_pre", "y", ["x", "s"], {"col": 0, "num": numname}, hp.shape, [stream[0].flatten(1).shape, split.shape]),
                      {"x": stream[0].flatten(1), "s": split}, hp, "hc_pre")
            yy = torch.randn(1, T, cfg.dim, generator=g).to(dt)
            hpost = ds.hc_post(yy, stream, post, comb, num.hi)[0].flatten(1)
            self.judge(mk("ds_hc_post", "y", ["y", "r", "s"], {"post": cfg.hc_mult, "comb": 2 * cfg.hc_mult, "num": numname}, hpost.shape,
                         [yy[0].shape, stream[0].flatten(1).shape, split.shape]), {"y": yy[0], "r": stream[0].flatten(1), "s": split}, hpost, "hc_post")
            # compressor pooling (layer 1: ratio 2)
            comp = att.compressor
            if comp is not None and comp.ratio > 1:
                hh = x[None].to(num.hi)
                n = T // comp.ratio
                kv = F.linear(hh, comp.wkv.weight.to(num.hi))[:, :n * comp.ratio].unflatten(1, (n, comp.ratio))
                scr = F.linear(hh, comp.wgate.weight.to(num.hi))[:, :n * comp.ratio].unflatten(1, (n, comp.ratio))
                pooled = (kv * scr.softmax(dim=2)).sum(dim=2).to(x.dtype)[0]
                self.judge(mk("ds_compress", "y", ["x"], {"wkv": "layers.1.attn.compressor.wkv.weight", "wgate": "layers.1.attn.compressor.wgate.weight",
                                                          "ratio": comp.ratio, "num": numname}, pooled.shape, [x.shape]), {"x": x}, pooled, "compress pool")
            # indexer scores
            if att.indexer is not None and numname == "ref":
                ix = att.indexer
                qq = num.fp4(torch.randn(1, T, cfg.index_n_heads, cfg.index_head_dim, generator=g).to(dt))
                kk = num.fp4(torch.randn(1, T // 2, cfg.index_head_dim, generator=g).to(dt))
                ww = (torch.randn(1, T, cfg.index_n_heads, generator=g)).to(dt)
                scs = torch.einsum("bshd,btd->bsht", qq, kk)
                out = (scs.relu() * ww.unsqueeze(-1)).sum(dim=2)
                vis = (torch.arange(1, T + 1) // 2).unsqueeze(-1)
                out = out.masked_fill(torch.arange(T // 2) >= vis, -float("inf"))[0]
                self.judge(mk("ds_index_scores", "y", ["q", "k", "w"], {"heads": cfg.index_n_heads, "ratio": 2, "num": numname}, out.shape,
                             [qq[0].flatten(1).shape, kk[0].shape, ww[0].shape]), {"q": qq[0].flatten(1), "k": kk[0], "w": ww[0]}, out, "index_scores")
            # swiglu with clamps and a gate
            gg = (torch.randn(T, cfg.moe_inter_dim, generator=g) * 2).to(dt)
            uu = (torch.randn(T, cfg.moe_inter_dim, generator=g) * 2).to(dt)
            gate = torch.rand(T, 1, generator=g).to(num.hi)
            L = cfg.swiglu_limit
            u2, g2 = uu.to(num.hi).clamp(-L, L), gg.to(num.hi).clamp(max=L)
            hsw = (gate * (F.silu(g2) * u2)).to(dt)
            self.judge(mk("ds_swiglu", "y", ["g", "u", "gate"], {"limit": L, "num": numname}, hsw.shape, [gg.shape, uu.shape, gate.shape]),
                      {"g": gg, "u": uu, "gate": gate}, hsw, "swiglu")

    def test_attention_and_the_router(self):
        cfg, state = self.cfg, self.state
        g = torch.Generator().manual_seed(5)
        for numname, num in (("ref", ds.Numerics()), ("f32", ds.Numerics(torch.float32, quantize=False))):
            dt = num.dtype
            T, H, hd, n, k, W = 150, cfg.n_heads, cfg.head_dim, 40, 3, 70  # several of the kernel's 64-key blocks
            q = torch.randn(1, T, H, hd, generator=g).to(dt)
            kv = num.fp8(torch.randn(1, T, hd, generator=g).to(dt))
            comp = num.fp4(torch.randn(1, n, hd, generator=g).to(dt), 16, True)
            vis = (torch.arange(1, T + 1) // 2).clamp(max=n)
            picks = torch.full((1, T, k), -1, dtype=torch.long)
            for i in range(T):
                c = sorted(torch.randperm(int(vis[i]), generator=g)[:k].tolist()) if vis[i] else []
                picks[0, i, :len(c)] = torch.tensor(c, dtype=torch.long)
            sink = state["layers.1.attn.attn_sink"]
            for cmp_, pk in ((comp, picks), (None, None)):
                o = ds.sparse_attention(q, kv, cmp_, pk, sink, hd ** -0.5, W, num)[0].flatten(1)
                ins = ["q", "kv"] + (["c", "p"] if cmp_ is not None else [])
                before = {"q": q[0].flatten(1), "kv": kv[0]}
                ishapes = [before["q"].shape, before["kv"].shape]
                if cmp_ is not None:
                    before |= {"c": cmp_[0], "p": pk[0].float()}
                    ishapes += [cmp_[0].shape, pk[0].shape]
                attrs = {"heads": H, "window": W, "scale": hd ** -0.5, "sink": "layers.1.attn.attn_sink",
                         "n_window": min(T, W), "num": numname}
                self.judge(mk("ds_attn", "o", ins, attrs, o.shape, ishapes), before, o, f"attn {numname}")
            m = ds.Model(cfg, layers=[1], num=num, embed=False, head=False, engram_table=False)
            m.load_state_dict({kk: v for kk, v in state.items() if kk.startswith("layers.1.")}, strict=False)
            moe = m.layers["1"].ffn
            xx = torch.randn(20, cfg.dim, generator=g).to(dt)
            lg = F.linear(xx.to(num.hi), moe.gate.weight.to(num.hi))
            sc = moe.scores(xx)
            self.judge(mk("ds_moe_scores", "s", ["l"], {"temp": cfg.gate_temp, "func": cfg.score_func}, sc.shape, [lg.shape]),
                       {"l": lg}, sc, "moe_scores")
            biased = sc + moe.gate.bias.to(num.hi)
            self.judge(mk("ds_bias", "b", ["s"], {"bias": "layers.1.ffn.gate.bias"}, biased.shape, [sc.shape]), {"s": sc},
                       biased, "bias")
            chosen, gates = moe.route(xx, ds.Choices())
            self.judge(mk("ds_gates", "g", ["s", "c"], {"norm": cfg.norm_topk_prob, "route_scale": cfg.route_scale},
                          gates.shape, [sc.shape, chosen.shape]), {"s": sc, "c": chosen.float()}, gates, "gates")

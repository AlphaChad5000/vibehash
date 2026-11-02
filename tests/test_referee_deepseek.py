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

    def verdict(self, op, before, after, tile):
        """What the node decides from a defender's evidence for one tile (dispute.make_evidence and judge_evidence)."""
        before = {k: np.asarray(v, dtype=np.float32) for k, v in before.items()}
        after = {**before, **{k: np.asarray(v, dtype=np.float32) for k, v in after.items()}}
        mb, _ = dispute.state_manifest(before)
        ma, _ = dispute.state_manifest(after)
        ev, _, _ = dispute.make_evidence(op, tile, before, after, (mb, ma), dispute.Outside(None, None, self.src, keep=False))
        return dispute.judge_evidence(op, tile, merkle.state_root(mb), merkle.state_root(ma), ev,
                                      dispute.Outside(None, None, self.src, keep=False))

    def test_choices_are_judged_by_whole_rows(self):
        g = torch.Generator().manual_seed(7)
        # Experts: a top-k of the biased scores, distinct, any order.
        sc = torch.randn(6, 24, generator=g)
        chosen = sc.topk(4, dim=-1).indices.float().numpy()
        op = mk("ds_experts", "c", ["s"], {}, chosen.shape, [sc.shape])
        self.assertEqual(self.verdict(op, {"s": sc.numpy()}, {"c": chosen}, 0)[0], "defender")
        self.assertEqual(self.verdict(op, {"s": sc.numpy()}, {"c": chosen[:, ::-1].copy()}, 0)[0], "defender")
        worse = chosen.copy()
        worse[2, 1] = sc[2].argsort()[0]  # the worst expert in place of a chosen one
        self.assertEqual(self.verdict(op, {"s": sc.numpy()}, {"c": worse}, 0)[0], "challenger")
        twice = chosen.copy()
        twice[3, 1] = twice[3, 0]
        self.assertEqual(self.verdict(op, {"s": sc.numpy()}, {"c": twice}, 0)[0], "challenger")
        # Picks: 300 of up to 600 positions, so a row spans two tiles; query r sees (r + 1) // ratio of them.
        T, n, K, ratio = 610, 600, 300, 1
        visible = (torch.arange(1, T + 1) // ratio)[:, None]
        sp = torch.randn(T, n, generator=g).masked_fill(torch.arange(n)[None, :] >= visible, -float("inf"))
        idx = sp.topk(K, dim=-1).indices.sort(dim=-1).values
        picks = torch.where(idx < visible, idx, -1).float().numpy()
        sp = sp.numpy()
        op = mk("ds_picks", "p", ["s"], {"ratio": ratio}, picks.shape, [sp.shape])
        for r in (100, 450):  # a query with fewer than K positions open (-1 padding), and one with more
            for tile in ((r * K) // 256, (r * K + K - 1) // 256):
                self.assertEqual(self.verdict(op, {"s": sp}, {"p": picks}, tile)[0], "defender", (r, tile))
        bad = picks.copy()
        row = bad[450]
        out_ = [j for j in np.argsort(sp[450]) if j not in set(row.astype(int)) and np.isfinite(sp[450, j])][0]
        row[np.argmax(row)] = out_  # a low-scoring position in place of the last pick
        bad[450] = np.sort(row)
        for tile in ((450 * K) // 256, (450 * K + K - 1) // 256):  # whichever tile of the row is named
            self.assertEqual(self.verdict(op, {"s": sp}, {"p": bad}, tile)[0], "challenger")
        # Candidate blocks: deepseek.candidate_blocks' mask; dropping the pinned block is refused.
        T = n = 40
        vis = torch.arange(1, T + 1)[:, None]
        cs = torch.randn(1, T, n, generator=g).masked_fill(torch.arange(n) >= vis, -float("inf"))
        mask = ds.candidate_blocks(cs, vis, 2, 4)[0].float().numpy()
        op = mk("ds_candidates", "m", ["s"], {"block": 4, "topk_blocks": 2, "ratio": 1}, mask.shape, [(T, n)])
        for tile in range(merkle.n_leaves(T, n)):
            self.assertEqual(self.verdict(op, {"s": cs[0].numpy()}, {"m": mask}, tile)[0], "defender", tile)
        drop = mask.copy()
        drop[30, 28:32] = 0  # query 30's newest position, 30, is in block 7: always kept
        self.assertEqual(self.verdict(op, {"s": cs[0].numpy()}, {"m": drop}, (30 * n) // 256)[0], "challenger")


    def test_embedding_engram_and_the_moe_sum(self):
        from tests.test_deepseek import token_map_for
        from vibehash.deepseek_config import EngramLayout

        cfg, state, g = self.cfg, self.state, torch.Generator().manual_seed(9)
        T = 14
        tokens = torch.randint(0, cfg.vocab_size, (1, T), generator=g)
        tmap = token_map_for(cfg)
        src = rd.StateWeights({**state, "engram.token_map": tmap})
        old, self.src = self.src, src
        try:
            for numname, num in (("ref", ds.Numerics()), ("f32", ds.Numerics(torch.float32, quantize=False))):
                m = ds.Model(cfg, layers=[1], num=num, embed=True, head=False)
                m.load_state_dict({k: v for k, v in state.items() if k.startswith(("layers.1.", "embed."))}, strict=False)
                x0, _ = m.start(tokens)
                emb = x0[0, :, 0]
                tok = tokens[0, :, None].float()
                self.judge(mk("ds_embed", "e", ["t"], {"weight": "embed.weight"}, emb.shape, [tok.shape]), {"t": tok}, emb,
                           f"embed {numname}")
                eng = m.layers["1"].engram
                layout = EngramLayout.of(cfg)
                flat = [p for size in layout.primes[0] for p in size]
                attrs = {"n": cfg.engram_max_ngram_size, "multipliers": list(layout.multipliers[0]),
                         "primes": [list(p) for p in layout.primes[0]], "offsets": np.cumsum([0, *flat[:-1]]).tolist(),
                         "pad": int(tmap[cfg.engram_pad_id]), "token_map": "engram.token_map",
                         "table": "layers.1.engram.embed"}
                hashes = ds.engram_hashes(tokens, tmap, cfg)[:, :, 0]
                rows = eng.rows(hashes)[0].flatten(-2)
                self.judge(mk("ds_engram_rows", "rw", ["t"], attrs, rows.shape, [tok.shape]), {"t": tok}, rows,
                           f"engram rows {numname}")
                stream = x0.to(num.dtype) + 0.1 * torch.randn(x0.shape, generator=g).to(num.dtype)
                rows4 = eng.rows(hashes)  # [1, T, cols, head_dim], as Engram.forward takes them
                kvo = eng.wkv(rows4.flatten(-2).to(num.dtype))[0]
                out = eng(stream, rows=rows4)[0].flatten(1)
                self.judge(mk("ds_engram_gate", "o", ["s", "kv"], {"q_weight": "layers.1.engram.q_weight",
                                                               "k_weight": "layers.1.engram.k_weight",
                                                               "eps": cfg.norm_eps, "num": numname},
                              out.shape, [stream[0].flatten(1).shape, kvo.shape]),
                           {"s": stream[0].flatten(1), "kv": kvo}, out, f"engram gate {numname}")
                moe = m.layers["1"].ffn
                xx = torch.randn(T, cfg.dim, generator=g).to(num.dtype)
                chosen, gates = moe.route(xx, ds.Choices())
                k = cfg.n_activated_experts
                routed = torch.zeros(T, k, cfg.dim, dtype=num.dtype)
                for e, tk, slot in moe.assignments(chosen):
                    routed[tk, slot] = moe.experts[e](xx[tk], gates[tk, slot, None])
                shared = moe.shared_experts(xx)
                y = moe(xx, ds.Choices(experts={1: chosen}))
                self.judge(mk("ds_moe_sum", "y", ["r", "c", "sh"], {"num": numname}, y.shape,
                              [routed.flatten(1).shape, chosen.shape, shared.shape]),
                           {"r": routed.flatten(1), "c": chosen.float(), "sh": shared}, y, f"moe sum {numname}")
        finally:
            self.src = old


def np32(t):
    return t.detach().float().cpu().numpy()


@unittest.skipIf(torch is None, "needs PyTorch")
class RealLayerTests(unittest.TestCase):
    """The ops on DeepSeek-V4.1-Flash's real layer 2 (a kv and index source: the ratio-2 compressor, its own indexer
    keys, YaRN; FP8 and FP4 weights at full size), op by op through Block.forward in the reference's arithmetic on
    CUDA, 256 tokens of README.md: sampled tiles of every op are judged, honest and tampered by 3%. Opt-in: needs the
    cached layer (tests/test_deepseek.py's RealLayerTests) and a CUDA GPU."""

    @classmethod
    def setUpClass(cls):
        from tests.test_deepseek import REAL_FILES
        from vibehash.deepseek_config import Config

        paths = [ds.cache_dir() / f for f in REAL_FILES]
        if not all(p.exists() for p in paths) or not torch.cuda.is_available():
            raise unittest.SkipTest("opt-in: the real layer 2 (scripts/deepseek_bench.py --fetch-only) and a CUDA GPU")
        ds.configure_torch()
        cls.cfg = Config.flash()
        cls.state = ds.read_safetensors(paths[0], device="cuda")
        cls.embed = ds.read_safetensors(paths[1], {"embed.weight"})["embed.weight"]
        cls.src = rd.StateWeights(cls.state)
        model = ds.Model(cls.cfg, layers=[2], num=ds.Numerics(), device="meta", embed=False, head=False,
                         engram_table=False)
        model.load_weights({k: v for k, v in cls.state.items() if k.startswith("layers.2.")})
        cls.blk = model.layers["2"]

    @classmethod
    def tearDownClass(cls):
        cls.state = cls.blk = None
        torch.cuda.empty_cache()

    def judge(self, kind, ins, attrs, out, label, before):
        before = {k: np32(v).reshape(v.shape[0], -1) if v.dim() > 1 else np32(v)[None] for k, v in before.items()}
        out = np32(out).reshape(out.shape[0], -1)
        op = Op(kind, "y", tuple(ins), attrs, (), tuple(out.shape), tuple(before[n].shape for n in ins))
        x = dispute.Inputs({k: v.shape for k, v in before.items()}, dispute.Outside(None, None, self.src, keep=False), full=before)
        x.after = dispute.Inputs({"y": out.shape}, x.outside, full={"y": out})
        n = merkle.n_leaves(*op.shape)
        flat = out.reshape(-1)
        tiles = sorted(set(self.rng.choice(n, min(self.samples, n), replace=False).tolist()))
        fails = caught = tampered = 0
        used = 0.0
        for leaf in tiles:
            said = flat[leaf * merkle.LEAF:(leaf + 1) * merkle.LEAF].astype(np.float64)
            ok, why = rf.judge(op, x, leaf, said)
            if not ok:
                fails += 1
                print("   honest tile refused:", label, why)
            else:
                frac = float(why.split("(")[-1].split(" of")[0]) if "of that)" in why else 0.0
                used = max(used, frac)
            if kind in rf.CHOICES:
                continue
            j = int(np.argmax(np.where(np.isfinite(said), np.abs(said), -1)))
            t = said.copy()
            t[j] *= 1.5 if kind in rf.EXACT else 1.03
            tampered += 1
            caught += not rf.judge(op, x, leaf, t)[0]
        self.assertEqual(fails, 0, label)
        self.assertEqual(caught, tampered, label)
        self.assertLess(used, 1.0, label)  # an honest number never needs the referee's margin

    def test_every_forward_op_on_the_real_layer(self):
        from tests.test_deepseek import readme_tokens

        cfg, num, T, SAMPLES = self.cfg, ds.Numerics(), 256, 6
        embed, blk = self.embed, self.blk
        att, ffn = blk.attn, blk.ffn
        self.rng, self.samples = np.random.default_rng(0), SAMPLES
        with torch.no_grad():
            ids = torch.tensor(readme_tokens(T)).cuda()
            x0 = embed[ids.cpu()].cuda()
            x = x0[None, :, None].repeat(1, 1, cfg.hc_mult, 1).to(num.dtype)
            pre_mix = torch.zeros(1, T, cfg.hc_mult, device="cuda")
            pre_mix[..., 0] = 1
            hi = num.hi
            stream = x[0].flatten(1)
            h = x.flatten(2).to(hi)
            mixes = F.linear(h, blk.hc_attn_fn.to(hi)) * torch.rsqrt(h.square().mean(-1, keepdim=True) + cfg.norm_eps)
            self.judge("ds_hc_mixes", ["s"], {"fn": "layers.2.hc_attn_fn", "eps": cfg.norm_eps}, mixes[0], "hc_mixes", {"s": stream})
            apre, post, comb = ds._sinkhorn_split(mixes, blk.hc_attn_scale.to(hi), blk.hc_attn_base.to(hi), cfg.hc_mult,
                                                  cfg.hc_sinkhorn_iters, cfg.hc_eps)
            split = torch.cat([apre, post, comb.flatten(-2)], -1)[0]
            self.judge("ds_hc_split", ["m"], {"scale": "layers.2.hc_attn_scale", "base": "layers.2.hc_attn_base", "hc": 4,
                                         "iters": cfg.hc_sinkhorn_iters, "eps": cfg.hc_eps}, split, "hc_split", {"m": mixes[0]})
            hp = ds.hc_pre(x, pre_mix.to(hi), hi)[0]
            self.judge("ds_hc_pre", ["s", "p"], {"col": 0}, hp, "hc_pre", {"s": stream, "p": pre_mix[0]})
            xn = blk.attn_norm(hp)
            self.judge("ds_rmsnorm", ["h"], {"weight": "layers.2.attn_norm.weight", "eps": cfg.norm_eps}, xn, "attn_norm", {"h": hp})
            xq = num.fp8(xn)
            self.judge("ds_round_fp8", ["h"], {}, xq, "round_fp8", {"h": xn})
            qa = att.wq_a.matmul(xq)
            self.judge("ds_qlinear", ["x"], {"weight": "layers.2.attn.wq_a"}, qa, "qlinear wq_a (K=5120)", {"x": xq})
            qr = att.q_norm(qa)
            self.judge("ds_rmsnorm", ["h"], {"weight": "layers.2.attn.q_norm.weight", "eps": cfg.norm_eps}, qr, "q_norm", {"h": qa})
            qrq = num.fp8(qr)
            q = att.wq_b.matmul(qrq)
            self.judge("ds_qlinear", ["x"], {"weight": "layers.2.attn.wq_b"}, q, "qlinear wq_b (32768 out)", {"x": qrq})
            cos, sin = ds._rope_table(T, *cfg.rope_args(2))
            qh = ds._rope(q.unflatten(-1, (cfg.n_heads, cfg.head_dim))[None], cos, sin, num)[0].flatten(-2)
            self.judge("ds_rope", ["q"], {"rope": cfg.rope_args(2), "hd": cfg.head_dim}, qh, "rope q (YaRN)", {"q": q})
            kv0 = att.wkv.matmul(xq)
            kvn = att.kv_norm(kv0)
            kvr = ds._rope(kvn[None], cos, sin, num)[0]
            self.judge("ds_rope", ["k"], {"rope": cfg.rope_args(2), "hd": cfg.head_dim}, kvr, "rope kv", {"k": kvn})
            kv = num.fp8(kvr)
            r = cfg.ratio(2)
            latent = att.compressor(x0[None].to(num.dtype))[0]
            hh = x0[None].to(num.dtype).to(hi)
            n = T // r
            kvc = F.linear(hh, att.compressor.wkv.weight.to(hi))[:, :n * r].unflatten(1, (n, r))
            scc = F.linear(hh, att.compressor.wgate.weight.to(hi))[:, :n * r].unflatten(1, (n, r))
            pooled = (kvc * scc.softmax(dim=2)).sum(dim=2).to(num.dtype)[0]
            self.judge("ds_compress", ["x"], {"wkv": "layers.2.attn.compressor.wkv.weight", "wgate": "layers.2.attn.compressor.wgate.weight",
                                         "ratio": r}, pooled, "compress (pool)", {"x": x0.to(num.dtype)})
            ccos, csin = cos[:T - T % r:r], sin[:T - T % r:r]
            comp_r = ds._rope(latent[None], ccos, csin, num)[0]
            self.judge("ds_rope", ["l"], {"rope": cfg.rope_args(2), "hd": cfg.head_dim, "stride": r}, comp_r, "rope compressed (stride)", {"l": latent})
            comp = num.fp4(comp_r, 16, e4m3_scale=True)
            self.judge("ds_round_fp4", ["c"], {"block": 16, "e4m3": True}, comp, "round_fp4 e4m3/16", {"c": comp_r})
            ix = att.indexer
            ik = ix.keys(latent[None], ccos, csin)[0]
            qi = att.indexer.wq_b.matmul(qrq)
            qir = ds._rope(qi.unflatten(-1, (cfg.index_n_heads, cfg.index_head_dim))[None], cos, sin, num)[0].flatten(-2)
            qi4 = num.fp4(qir)
            wproj = F.linear(xn, ix.weights_proj.weight.to(xn.dtype))
            self.judge("ds_dense", ["x"], {"weight": "layers.2.attn.indexer.weights_proj.weight"}, wproj, "dense weights_proj", {"x": xn})
            c = cfg.index_head_dim ** -0.5 * cfg.index_n_heads ** -0.5
            ws = wproj * c
            score = ix.scores(xn[None], qr[None], ik[None], cos, sin)[0]
            self.judge("ds_index_scores", ["q", "k", "w"], {"heads": cfg.index_n_heads, "ratio": r}, score, "index_scores",
                  {"q": qi4, "k": ik, "w": ws})
            K = min(cfg.index_topk, n)
            vis = (torch.arange(1, T + 1, device="cuda") // r).unsqueeze(-1)
            idx = score.topk(K, dim=-1, sorted=False).indices.sort(dim=-1).values
            picks = torch.where(idx < vis, idx, -1)
            self.judge("ds_picks", ["s"], {"ratio": r}, picks.float(), "picks (choice)", {"s": score})
            o = ds.sparse_attention(qh.unflatten(-1, (cfg.n_heads, cfg.head_dim))[None], kv[None], comp[None], picks[None],
                                    att.attn_sink, cfg.head_dim ** -0.5, cfg.window_size, num)[0].flatten(1)
            self.judge("ds_attn", ["q", "kv", "c", "p"], {"heads": cfg.n_heads, "window": cfg.window_size, "scale": cfg.head_dim ** -0.5,
                                                    "sink": "layers.2.attn.attn_sink", "n_window": min(T, cfg.window_size)},
                  o, "attention (one query, one head)", {"q": qh, "kv": kv, "c": comp, "p": picks.float()})
            oi = ds._rope(o.unflatten(-1, (cfg.n_heads, cfg.head_dim))[None], cos, sin, num, inverse=True)[0].flatten(-2)
            oq = num.fp8(oi)
            woa = att.wo_a.matmul(oq.unflatten(-1, (cfg.o_groups, -1))).flatten(-2)
            self.judge("ds_qlinear", ["o"], {"weight": "layers.2.attn.wo_a", "groups": cfg.o_groups}, woa, "qlinear wo_a (grouped)", {"o": oq})
            y = att.wo_b.matmul(num.fp8(woa))
            hpost = ds.hc_post(y[None], x, post, comb, hi)[0].flatten(1)
            self.judge("ds_hc_post", ["y", "r", "s"], {"post": 4, "comb": 8}, hpost, "hc_post", {"y": y, "r": stream, "s": split})
            # the MoE
            xm = blk.ffn_norm(hp)
            lg = F.linear(xm.to(hi), ffn.gate.weight.to(hi))
            self.judge("ds_dense", ["x"], {"weight": "layers.2.ffn.gate.weight", "hi": True}, lg, "dense router (hi)", {"x": xm})
            sc = ffn.scores(xm)
            self.judge("ds_moe_scores", ["l"], {"temp": cfg.gate_temp, "func": cfg.score_func}, sc, "moe_scores", {"l": lg})
            biased = sc + ffn.gate.bias.to(hi)
            chosen, gates = ffn.route(xm, ds.Choices())
            self.judge("ds_experts", ["b"], {}, chosen.float(), "experts (choice)", {"b": biased})
            self.judge("ds_gates", ["s", "c"], {"norm": cfg.norm_topk_prob, "route_scale": cfg.route_scale}, gates, "gates",
                  {"s": sc, "c": chosen.float()})
            e, toks, slots = ffn.assignments(chosen)[0]
            ex = ffn.experts[e]
            xe = num.fp8(xm[toks])
            g1 = ex.w1.matmul(xe)
            self.judge("ds_qlinear", ["x"], {"weight": f"layers.2.ffn.experts.{e}.w1"}, g1, "qlinear expert w1 (FP4)", {"x": xe})
            u3 = ex.w3.matmul(xe)
            gate = gates[toks, slots, None]
            L = cfg.swiglu_limit
            hsw = (gate * (F.silu(g1.to(hi).clamp(max=L)) * u3.to(hi).clamp(-L, L))).to(num.dtype)
            self.judge("ds_swiglu", ["g", "u", "gate"], {"limit": L}, hsw, "swiglu (clamps, gate)", {"g": g1, "u": u3, "gate": gate})

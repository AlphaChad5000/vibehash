"""The transformer reference (vibehash/transformer.py): exact gradients, LoRA as W + (alpha/r)·B·A, causal attention
with RoPE and grouped-query heads, stages that train to exactly the whole model's bits, and deterministic training."""
import unittest
from dataclasses import replace

import numpy as np

from vibehash import transformer as tf
from vibehash.transformer import Stage, TransformerSpec


def tiny(**settings) -> TransformerSpec:
    """Small enough for finite differences: 4 heads of 4 dimensions, 2 layers, 6 positions."""
    return TransformerSpec(**{**dict(vocab=11, d_model=16, n_layers=2, n_heads=4, d_ff=24, seq_len=6), **settings})


def general(spec: TransformerSpec, seed: int = 0):
    """Base and trainable weights in general position: norms away from 1 and adapters' B away from 0, so every
    gradient term is exercised."""
    rng = np.random.default_rng(seed)
    base = tf.init_base(spec, seed)
    for name, view in tf.unpack(tf.layout(spec), base).items():
        if view.ndim == 1:
            view += rng.normal(0, 0.3, view.shape)
    base = spec.round(base)
    if not spec.lora_rank:
        return None, base
    return base, spec.round(rng.normal(0, 0.3, spec.n_trainable))


def rows(spec: TransformerSpec, n: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, spec.vocab, (n, spec.seq_len + 1)).astype(np.int32)


class LayoutTests(unittest.TestCase):
    def test_the_base_uses_hugging_faces_llama_names_and_shapes(self):
        spec = tiny(n_kv_heads=2, d_model=24, head_dim=8)  # q is 4·8 = 32 wide, k and v 2·8 = 16, the model 24
        entries = tf.layout(spec)
        self.assertEqual(entries[0], ("model.embed_tokens.weight", (11, 24)))
        self.assertEqual(entries[1:10], [
            ("model.layers.0.self_attn.q_proj.weight", (32, 24)), ("model.layers.0.self_attn.k_proj.weight", (16, 24)),
            ("model.layers.0.self_attn.v_proj.weight", (16, 24)), ("model.layers.0.self_attn.o_proj.weight", (24, 32)),
            ("model.layers.0.mlp.gate_proj.weight", (24, 24)), ("model.layers.0.mlp.up_proj.weight", (24, 24)),
            ("model.layers.0.mlp.down_proj.weight", (24, 24)), ("model.layers.0.input_layernorm.weight", (24,)),
            ("model.layers.0.post_attention_layernorm.weight", (24,))])
        self.assertTrue(entries[10][0].startswith("model.layers.1."))
        self.assertEqual(entries[-2:], [("model.norm.weight", (24,)), ("lm_head.weight", (11, 24))])
        self.assertNotIn("lm_head.weight", dict(tf.layout(replace(spec, tie_embeddings=True))))
        self.assertEqual(spec.n_base, sum(np.prod(s) for _, s in entries))
        self.assertEqual(spec.n_params, spec.n_base)  # rank 0: the base is what's trained and checkpointed

    def test_adapters_follow_peft(self):
        """A [rank, in] and B [out, rank] for each target, layer by layer, under PEFT's names: an export to Hugging
        Face PEFT or vLLM is a rename."""
        spec = tiny(n_kv_heads=2, lora_rank=3, lora_alpha=6.0, lora_targets=("down", "v", "q"))
        self.assertEqual(spec.lora_targets, ("q", "v", "down"))  # listed in any order, laid out in one
        self.assertEqual(tf.adapter_layout(spec)[:6], [
            ("model.layers.0.self_attn.q_proj.lora_A.weight", (3, 16)),
            ("model.layers.0.self_attn.q_proj.lora_B.weight", (16, 3)),
            ("model.layers.0.self_attn.v_proj.lora_A.weight", (3, 16)),
            ("model.layers.0.self_attn.v_proj.lora_B.weight", (8, 3)),
            ("model.layers.0.mlp.down_proj.lora_A.weight", (3, 24)),
            ("model.layers.0.mlp.down_proj.lora_B.weight", (16, 3))])
        self.assertEqual(len(tf.adapter_layout(spec)), 12)
        self.assertEqual(tf.trainable_layout(spec), tf.adapter_layout(spec))
        keys, config = tf.peft_names(spec)
        self.assertEqual(list(keys), [name for name, _ in tf.adapter_layout(spec)])
        self.assertEqual(keys["model.layers.1.mlp.down_proj.lora_B.weight"],
                         "base_model.model.model.layers.1.mlp.down_proj.lora_B.weight")
        self.assertEqual(config, {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 3, "lora_alpha": 6,
                                  "target_modules": ["q_proj", "v_proj", "down_proj"], "lora_dropout": 0.0,
                                  "bias": "none", "use_rslora": False, "fan_in_fan_out": False})
        with self.assertRaises(ValueError):
            tf.peft_names(tiny())

    def test_pack_and_unpack_are_inverse_and_check_shapes(self):
        spec = tiny(tie_embeddings=True)
        base = tf.init_base(spec, 3)
        named = tf.unpack(tf.layout(spec), base)
        np.testing.assert_array_equal(tf.pack(tf.layout(spec), named), base)
        with self.assertRaises(ValueError):
            tf.pack(tf.layout(spec), {**named, "model.norm.weight": np.ones(15)})
        with self.assertRaises(ValueError):
            tf.unpack(tf.layout(spec), base[:-1])

    def test_spec_fills_in_defaults_and_refuses_what_it_cant_compute(self):
        self.assertEqual(tiny().n_kv_heads, 4)
        self.assertEqual(tiny(), tiny(n_kv_heads=4, head_dim=4))
        self.assertEqual(hash(tiny(lora_targets=["v", "q"])), hash(tiny(lora_targets=("q", "v"))))
        for bad in (dict(n_kv_heads=3), dict(d_model=18), dict(head_dim=3), dict(precision="bfloat16"),
                    dict(lora_rank=-1), dict(lora_rank=2, lora_targets=()), dict(lora_targets=("q", "embed")),
                    dict(lora_targets="q")):
            with self.assertRaises(ValueError, msg=str(bad)):
                tiny(**bad)

    def test_from_hf_config(self):
        config = {"model_type": "llama", "vocab_size": 32000, "hidden_size": 2048, "num_hidden_layers": 22,
                  "num_attention_heads": 32, "num_key_value_heads": 4, "intermediate_size": 5632,
                  "rope_theta": 10000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False, "hidden_act": "silu"}
        spec = TransformerSpec.from_hf_config(config, seq_len=512, lora_rank=8, precision="float32")
        self.assertEqual((spec.d_model, spec.n_layers, spec.n_heads, spec.n_kv_heads, spec.head_dim, spec.d_ff),
                         (2048, 22, 32, 4, 64, 5632))
        self.assertEqual(spec.n_base, 1_100_048_384)  # TinyLlama-1.1B's parameter count
        for change in ({"rope_scaling": {"rope_type": "llama3", "factor": 8.0}}, {"attention_bias": True},
                       {"hidden_act": "gelu"}, {"model_type": "gemma"}, {"sliding_window": 256}):
            with self.assertRaises(ValueError, msg=str(change)):
                TransformerSpec.from_hf_config({**config, **change}, seq_len=512)


class GradientTests(unittest.TestCase):
    """Central differences in float64 along a random direction inside each named tensor, so every weight of every
    kind is covered: the embedding, each projection, the norms, the head, and the adapters' A and B."""

    def check(self, spec: TransformerSpec):
        base, trainable = general(spec)
        batch = rows(spec, 3)
        grad = tf.gradient(spec, base, trainable, batch)
        self.assertEqual(grad.dtype, np.float64)
        rng, eps = np.random.default_rng(2), 1e-6
        index = tf.unpack(tf.trainable_layout(spec), np.arange(spec.n_trainable))
        for name, where in index.items():
            direction = np.zeros(spec.n_trainable)
            direction[where.ravel()] = rng.normal(size=where.size)
            up = tf.loss(spec, base, trainable + eps * direction, batch)
            down = tf.loss(spec, base, trainable - eps * direction, batch)
            numeric, exact = (up - down) / (2 * eps), float(grad @ direction)
            self.assertGreater(abs(exact), 1e-6, name)  # the tensor matters, so the check isn't vacuous
            self.assertLess(abs(numeric - exact), 1e-7 * max(1.0, abs(exact)), f"{spec}: {name}")
        return index

    def test_full_fine_tuning(self):
        for settings in (dict(), dict(n_kv_heads=2, tie_embeddings=True), dict(n_kv_heads=1)):
            names = self.check(tiny(**settings))
            self.assertIn("model.embed_tokens.weight", names)
            self.assertIn("model.layers.1.post_attention_layernorm.weight", names)
            self.assertEqual("lm_head.weight" in names, not settings.get("tie_embeddings"))

    def test_lora(self):
        for settings in (dict(n_kv_heads=2), dict(tie_embeddings=True, lora_targets=("q", "v", "down")),
                         dict(n_kv_heads=1, tie_embeddings=True, lora_targets=("k", "o", "gate", "up"))):
            spec = tiny(lora_rank=2, lora_alpha=3.0, **settings)
            names = self.check(spec)
            self.assertEqual(len(names), 2 * len(spec.lora_targets) * spec.n_layers)

    def test_float32_gradients_are_float32_and_close_to_float64(self):
        spec = tiny(n_kv_heads=2, lora_rank=2)
        base, trainable = general(spec)
        batch = rows(spec, 4)
        g64 = tf.gradient(spec, base, trainable, batch)
        g32 = tf.gradient(replace(spec, precision="float32"), base, trainable, batch)
        self.assertEqual(g32.dtype, np.float32)
        self.assertEqual(tf.forward(replace(spec, precision="float32"), base, trainable, batch[:, :-1]).dtype, np.float32)
        self.assertLess(np.linalg.norm(g32 - g64) / np.linalg.norm(g64), 1e-5)


class LoraTests(unittest.TestCase):
    def test_fresh_adapters_give_exactly_the_base_model(self):
        for precision in ("float64", "float32"):
            plain = tiny(n_kv_heads=2, precision=precision)
            spec = replace(plain, lora_rank=4)
            _, base = general(plain)
            adapters = tf.init_adapters(spec, 5)
            self.assertTrue(np.any(adapters))  # A is random; only B is zero
            tokens = rows(spec, 3)[:, :-1]
            np.testing.assert_array_equal(tf.forward(spec, base, adapters, tokens), tf.forward(plain, None, base, tokens))
            np.testing.assert_array_equal(tf.merge(spec, base, adapters), base)

    def test_an_adapted_projection_is_w_plus_alpha_over_r_b_a(self):
        spec = tiny(n_kv_heads=2, lora_rank=3, lora_alpha=5.0, lora_targets=("k", "o", "up"))
        base, adapters = general(spec)
        merged = tf.merge(spec, base, adapters)
        w, a, m = (tf.unpack(tf.layout(spec), base), tf.unpack(tf.adapter_layout(spec), adapters),
                   tf.unpack(tf.layout(spec), merged))
        for name in w:
            module = name.removesuffix(".weight")
            if module + ".lora_A.weight" in a:
                np.testing.assert_allclose(m[name], w[name] + 5.0 / 3 * a[module + ".lora_B.weight"] @
                                           a[module + ".lora_A.weight"], rtol=1e-14, atol=1e-14)
            else:
                np.testing.assert_array_equal(m[name], w[name])
        # The adapted model computes as the merged one, though not in the same order (x·Wᵀ + x·Aᵀ·Bᵀ·s).
        tokens = rows(spec, 3)[:, :-1]
        np.testing.assert_allclose(tf.forward(spec, base, adapters, tokens),
                                   tf.forward(replace(spec, lora_rank=0), None, merged, tokens), rtol=1e-12, atol=1e-12)

    def test_lora_training_changes_only_the_targets(self):
        spec = tiny(lora_rank=2, lora_targets=("v", "down"))
        base = tf.init_base(spec, 1)
        data = tf.toy_data(spec, 2, task="count")
        trained = tf.train_segment(spec, data, base, tf.init_adapters(spec, 3), 4, 0, 5, 0.5, 8)
        before, after = tf.unpack(tf.layout(spec), base), tf.unpack(tf.layout(spec), tf.merge(spec, base, trained))
        changed = {name for name in before if not np.array_equal(before[name], after[name])}
        self.assertEqual(changed, {f"model.layers.{i}.{m}.weight" for i in range(2)
                                   for m in ("self_attn.v_proj", "mlp.down_proj")})


class AttentionTests(unittest.TestCase):
    def test_changing_a_later_token_never_changes_earlier_logits(self):
        for precision in ("float64", "float32"):
            spec = tiny(n_kv_heads=2, precision=precision)
            _, weights = general(spec)
            tokens = rows(spec, 2)[:, :-1]
            logits = tf.forward(spec, None, weights, tokens)
            for t in range(spec.seq_len):
                changed = tokens.copy()
                changed[:, t] = (changed[:, t] + 1) % spec.vocab
                after = tf.forward(spec, None, weights, changed)
                np.testing.assert_array_equal(after[:, :t], logits[:, :t])
                self.assertFalse(np.allclose(after[:, t:], logits[:, t:]))

    def rope(self, spec, x, position):
        cos, sin = tf._rope(spec, position + 1)
        return x * cos[position] + tf._rotate(x) * sin[position]

    def test_rope_turns_pairs_half_a_head_apart_as_hugging_face_does(self):
        spec = tiny(head_dim=8, d_model=32, rope_theta=500.0)
        x = np.random.default_rng(0).normal(size=8)
        for position in (0, 1, 7):
            angle = position * 500.0 ** (-np.arange(4) * 2 / 8)
            want = np.concatenate([x[:4] * np.cos(angle) - x[4:] * np.sin(angle),
                                   x[4:] * np.cos(angle) + x[:4] * np.sin(angle)])
            np.testing.assert_allclose(self.rope(spec, x, position), want, rtol=1e-13, atol=1e-13)

    def test_attention_scores_depend_only_on_relative_position(self):
        spec = tiny()
        rng = np.random.default_rng(1)
        q, k = rng.normal(size=spec.head_dim), rng.normal(size=spec.head_dim)
        score = lambda m, n: self.rope(spec, q, m) @ self.rope(spec, k, n)  # noqa: E731
        for m, n in ((3, 1), (5, 5), (2, 9)):
            for shift in (1, 4, 20):
                self.assertAlmostEqual(score(m, n), score(m + shift, n + shift), places=12)
        self.assertNotAlmostEqual(score(3, 1), score(3, 2), places=6)

    def test_grouped_query_heads_are_shared_as_hugging_faces_repeat_kv(self):
        """A key/value head serves the query heads of its group: copying each one once per query head, as Hugging Face
        does, gives an ordinary multi-head model with the same output."""
        gqa = tiny(n_kv_heads=2)
        mha = replace(gqa, n_kv_heads=4)
        _, weights = general(gqa)
        named = tf.unpack(tf.layout(gqa), weights)
        d = gqa.head_dim
        for name in [n for n in named if "k_proj" in n or "v_proj" in n]:
            named[name] = np.repeat(named[name].reshape(2, d, -1), 2, axis=0).reshape(4 * d, -1)
        tokens = rows(gqa, 3)[:, :-1]
        np.testing.assert_allclose(tf.forward(mha, None, tf.pack(tf.layout(mha), named), tokens),
                                   tf.forward(gqa, None, weights, tokens), rtol=1e-12, atol=1e-12)


SEED, STEPS, LR, BATCH = 9, 6, 0.3, 4


def deep(precision="float64", **settings) -> TransformerSpec:
    return TransformerSpec(**{**dict(vocab=23, d_model=32, n_layers=4, n_heads=4, n_kv_heads=2, d_ff=48, seq_len=8,
                                     precision=precision), **settings})


def start(spec: TransformerSpec):
    base = tf.init_base(spec, 3)
    return base, tf.init_trainable(spec, base, 4)


def through_the_wire(a: np.ndarray) -> np.ndarray:
    """An edge tensor as the neighbour receives it: its bytes, landed in a buffer at an odd offset (unaligned)."""
    raw = a.tobytes()
    buffer = np.zeros(len(raw) + 8, np.uint8)
    buffer[3:3 + len(raw)] = np.frombuffer(raw, np.uint8)
    return np.frombuffer(buffer[3:3 + len(raw)], a.dtype).reshape(a.shape)


class StageTests(unittest.TestCase):
    def test_stages_tile_both_vectors(self):
        for settings in (dict(), dict(lora_rank=2), dict(lora_rank=2, tie_embeddings=True)):
            spec = deep(**settings)
            base, trainable = start(spec)
            for n in range(1, spec.n_layers + 1):
                stages = tf.split(spec, n)
                self.assertEqual([s.lo for s in stages[1:]], [s.hi for s in stages[:-1]])
                self.assertTrue(stages[0].first and stages[-1].last and len(stages) == n)
                spans = [tf.trainable_span(spec, s) for s in stages]
                self.assertEqual((spans[0].start, spans[-1].stop), (0, spec.n_trainable))
                self.assertEqual([p.start for p in spans[1:]], [p.stop for p in spans[:-1]])
                np.testing.assert_array_equal(tf.assemble(spec, stages, [trainable[p] for p in spans]), trainable)
                named = tf.unpack(tf.layout(spec), base)
                for s in stages:  # a stage's base is what a loader packs from the checkpoint's tensors by name
                    np.testing.assert_array_equal(tf.stage_base(spec, s, base), tf.pack(tf.stage_layout(spec, s), named))
                    if n > 1:
                        self.assertLess(len(tf.stage_base(spec, s, base)), spec.n_base)
            with self.assertRaises(ValueError):
                tf.split(spec, spec.n_layers + 1)

    def test_a_tied_model_trained_in_full_stays_whole(self):
        spec = deep(tie_embeddings=True)
        self.assertEqual(len(tf.split(spec, 1)), 1)
        with self.assertRaises(ValueError):
            tf.split(spec, 2)
        with self.assertRaises(ValueError):
            tf.trainable_span(spec, Stage(0, 0, 2, 4))

    def test_stage_split_training_is_bit_identical_to_the_whole_model(self):
        """The same operations in the same order, cut at layer boundaries: so the bits match, and a team's unit
        averages and replays like one worker's."""
        for precision in ("float64", "float32"):
            for settings in (dict(lora_rank=2), dict(lora_rank=3, tie_embeddings=True, lora_targets=("q", "v", "up")),
                             dict()):
                spec = deep(precision, **settings)
                data = tf.toy_data(spec, 1, n_train=64, n_val=8)
                base, trainable = start(spec)
                whole = tf.train_segment(spec, data, base, trainable, SEED, 3, STEPS, LR, BATCH)
                self.assertFalse(np.array_equal(whole, trainable))
                for n in range(1, spec.n_layers + 1):
                    stages = tf.split(spec, n)
                    parts = tf.train_stages(spec, stages, data, base, trainable, SEED, 3, STEPS, LR, BATCH)
                    np.testing.assert_array_equal(tf.assemble(spec, stages, parts), whole,
                                                  err_msg=f"{precision}, {settings}, {n} stages")

    def test_edges_sent_as_bytes_replay_exactly_one_stage_at_a_time(self):
        """Workers on different computers: every edge tensor arrives as bytes in a buffer of its own, and the unit
        still ends at the whole model's bits. A checker holding one stage replays it from the edges it took in and
        gets exactly what it sent, which is what checking a cell (SHARDING.md) needs."""
        spec = deep("float32", lora_rank=2)
        data = tf.toy_data(spec, 1, n_train=64, n_val=8)
        base, trainable = start(spec)
        stages = tf.split(spec, 3)
        bases = [np.asarray(tf.stage_base(spec, s, base), np.float32) for s in stages]
        parts = [np.asarray(trainable[tf.trainable_span(spec, s)], np.float32) for s in stages]
        sent = []  # per step: (each stage's weights, h it took in, h it sent on, gradient it took in, gradient it sent)
        for step in range(STEPS):
            batch = tf._batch(data, SEED, step, BATCH)
            h, caches, record = batch[:, :-1], [], [dict(weights=p.copy()) for p in parts]
            for k, s in enumerate(stages):
                record[k]["h_in"] = h
                cache, h = tf.stage_forward(spec, s, bases[k], parts[k], h)
                caches.append(cache)
                record[k]["h_out"] = h
                h = through_the_wire(h)
            g = None
            for k in range(len(stages) - 1, -1, -1):
                record[k]["g_out"] = g
                g, grad = tf.stage_backward(spec, stages[k], bases[k], parts[k], caches[k], g,
                                            batch[:, 1:] if stages[k].last else None)
                record[k]["g_in"] = g
                parts[k] = parts[k] - np.float32(LR) * grad
                g = None if g is None else through_the_wire(g)
            sent.append(record)
        whole = tf.train_segment(spec, data, base, trainable, SEED, 0, STEPS, LR, BATCH)
        np.testing.assert_array_equal(tf.assemble(spec, stages, [p.astype(np.float64) for p in parts]), whole)
        middle, rec = stages[1], sent[4][1]
        cache, out = tf.stage_forward(spec, middle, bases[1], rec["weights"], through_the_wire(rec["h_in"]))
        np.testing.assert_array_equal(out, rec["h_out"])
        g_in, _ = tf.stage_backward(spec, middle, bases[1], rec["weights"], cache, through_the_wire(rec["g_out"]))
        np.testing.assert_array_equal(g_in, rec["g_in"])
        self.assertEqual(rec["h_out"].shape, (BATCH, spec.seq_len, spec.d_model))


class TrainingTests(unittest.TestCase):
    def test_training_learns_to_count_in_full_and_with_lora(self):
        for settings in (dict(), dict(lora_rank=4, lora_alpha=8.0)):
            spec = TransformerSpec(vocab=16, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, seq_len=8,
                                   precision="float32", **settings)
            data = tf.toy_data(spec, 1, task="count")
            tf.check_data(spec, data)
            base = tf.init_base(spec, 2)
            theta = tf.init_trainable(spec, base, 3)
            loss_before, _ = tf.evaluate(spec, base, theta, data.tokens_val)
            trained = tf.train_segment(spec, data, base, theta, 5, 0, 100, 0.3, 16)
            loss_after, accuracy = tf.evaluate(spec, base, trained, data.tokens_val)
            self.assertLess(loss_after, loss_before / 5, settings)
            self.assertGreater(accuracy, 0.9, settings)
            np.testing.assert_array_equal(spec.round(trained), trained)  # checkpoints are rounded to the precision

    def test_training_learns_to_copy_by_looking_back(self):
        """The second half of each row repeats the first. Its tokens are random, so no model that doesn't look back
        can beat log(vocab) on them; this one gets well under, which only attention can do."""
        spec = TransformerSpec(vocab=16, d_model=32, n_layers=2, n_heads=4, d_ff=64, seq_len=8, precision="float32")
        data = tf.toy_data(spec, 1, task="copy")
        np.testing.assert_array_equal(data.tokens_train[:, 5:], data.tokens_train[:, :4])
        base = tf.init_base(spec, 2)
        trained = tf.train_segment(spec, data, None, base, 5, 0, 200, 1.0, 16)

        def copied_half(weights):  # the loss on targets 5..8, the repeats
            logits = tf.forward(spec, None, weights, data.tokens_val[:, :-1])
            return tf._xent(logits[:, 4:].reshape(-1, spec.vocab), data.tokens_val[:, 5:].ravel()).mean()

        self.assertGreater(copied_half(base), np.log(spec.vocab) * 0.9)
        self.assertLess(copied_half(trained), np.log(spec.vocab) / 2)

    def test_same_seeds_same_bits_and_segments_replay_alone(self):
        for precision in ("float64", "float32"):
            spec = deep(precision, lora_rank=2)
            data = tf.toy_data(spec, 1, n_train=64, n_val=8)
            base, adapters = start(spec)
            run = lambda theta, first, n, seed=SEED: tf.train_segment(spec, data, base, theta, seed, first, n, LR, BATCH)  # noqa: E731
            whole = run(adapters, 0, 8)
            np.testing.assert_array_equal(run(adapters, 0, 8), whole)
            np.testing.assert_array_equal(run(run(adapters, 0, 3), 3, 5), whole)  # from a checkpoint, as a checker does
            self.assertFalse(np.array_equal(run(adapters, 0, 8, seed=SEED + 1), whole))
            np.testing.assert_array_equal(tf.init_base(spec, 3), base)
            np.testing.assert_array_equal(tf.init_adapters(spec, 4), adapters)

    def test_evaluate_matches_loss_and_bad_tokens_are_refused(self):
        spec = tiny(lora_rank=2)
        base, adapters = general(spec)
        batch = rows(spec, 10)
        loss, accuracy = tf.evaluate(spec, base, adapters, batch, batch_size=3)
        self.assertAlmostEqual(loss, tf.loss(spec, base, adapters, batch), places=12)
        self.assertTrue(0 <= accuracy <= 1)
        for bad in (batch - 1, batch + 1, batch.astype(np.float64)):
            with self.assertRaises(ValueError):
                tf.loss(spec, base, adapters, bad)
            with self.assertRaises(ValueError):
                tf.check_data(spec, tf.TokenData(bad, batch))
        with self.assertRaises(ValueError):
            tf.check_data(spec, tf.TokenData(batch[:, :-1], batch))

    def test_flops_count_lora_as_cheaper_than_full_training(self):
        spec = deep()
        lora = replace(spec, lora_rank=4)
        self.assertGreater(spec.flops_per_step(8), lora.flops_per_step(8))
        self.assertGreater(lora.flops_per_step(8), 4 * spec.seq_len * 8 * spec.n_layers * spec.d_model ** 2)
        self.assertEqual(spec.flops_per_step(16), 2 * spec.flops_per_step(8))


if __name__ == "__main__":
    unittest.main()

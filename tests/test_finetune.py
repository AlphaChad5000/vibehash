"""Fine-tuning: LoRA jobs on pretrained models (finetune.py, llm.py).

Everything here runs offline on tiny random checkpoints (tests/finetune_fixtures.py) with CPU PyTorch. Checks against
Hugging Face transformers and PEFT run when those are installed. Real models are opt-in:
    VIBEHASH_REAL_MODELS=1 python -m unittest tests.test_finetune
fetches Qwen3-0.6B (1.5 GB, cached in ~/.vibehash/cache) and compares it with transformers, measures CPU against CUDA
and loads an exported adapter with PEFT.
"""
import copy
import hashlib
from collections import defaultdict
import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from vibehash import finetune as ft
from vibehash.blobs import file_sha256
from vibehash.coordinator import CoordinatorConfig
from vibehash.crypto import KeyPair
from vibehash.ledger import Chain
from vibehash.model import Dataset, ModelSpec
from vibehash.tasks import MAX_SPEC_BYTES, InvalidJob, TrainingJob, load_dataset
from vibehash.verification import adam_impossible, rescale_update, reproduces

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tests.finetune_fixtures import CONFIGS, TinyModel
    from vibehash import llm

HAVE_HF = torch is not None and all(importlib.util.find_spec(m) for m in ("transformers", "peft"))
HAVE_TOKENIZERS = all(importlib.util.find_spec(m) for m in ("tokenizers", "jinja2"))
REAL = os.environ.get("VIBEHASH_REAL_MODELS") == "1"
needs_torch = unittest.skipIf(torch is None, "needs PyTorch")
VIBE = 10**8


def tmpdir(test) -> Path:
    d = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    test.addCleanup(d.cleanup)
    return Path(d.name)


def spec_of(**kw) -> ft.LoraSpec:
    """A spec shaped like Qwen3-0.6B, without needing its files."""
    dims = {"family": "qwen3", "vocab": 151936, "hidden": 1024, "layers": 28, "heads": 16, "kv_heads": 8,
            "head_dim": 128, "ffn": 3072, "tied": True}
    return ft.LoraSpec(base="hf:Qwen/Qwen3-0.6B@" + "c" * 40, manifest="a" * 64, dims=dims, **kw)


def job_of(spec=None, **kw) -> TrainingJob:
    args = dict(budget=10 * VIBE, fee_per_round=VIBE, spec=spec or spec_of(), data_source={"kind": "tokens", "sha256": "b" * 64},
                lr=1e-3, batch_size=8, steps_per_unit=16, checkpoint_every=4, tolerance=1e-3, max_update_ratio=1.0)
    return TrainingJob("ft", "", **{**args, **kw})


class SpecTests(unittest.TestCase):
    def test_round_trip(self):
        job = job_of(compute_s=60.0)
        again = TrainingJob.from_spec_json(job.spec_json(), "someone", 5)
        self.assertTrue(again.finetune)
        self.assertEqual(again.spec_json(), job.spec_json())
        self.assertEqual(again.spec, job.spec)
        self.assertLess(len(job.spec_json()), MAX_SPEC_BYTES)
        d = json.loads(job.spec_json())
        self.assertEqual(d["base"], job.spec.base)  # where trainui.parse_job reads it, for the forge's emblem
        self.assertEqual(d["lora"], {"rank": 16, "alpha": 32, "targets": ["q", "k", "v", "o"]})

    def test_an_mlp_spec_with_a_stray_base_is_still_an_mlp(self):
        """Nodes before fine-tunes ignored unknown fields, so an MLP spec with "base" in it was valid, and may be on a
        chain. Only `lora` and a model without n_in make a fine-tune: a shape those nodes refused."""
        spec = TrainingJob("mlp-with-base", "", 1, 1).to_spec()
        spec["base"] = "anything"
        job = TrainingJob.from_spec_json(json.dumps(spec, sort_keys=True), "req", 100)
        self.assertFalse(job.finetune)
        from vibehash.tasks import is_finetune_spec
        self.assertTrue(is_finetune_spec(json.loads(job_of().spec_json())))
        self.assertFalse(is_finetune_spec({**json.loads(job_of().spec_json()), "model": {"n_in": 4, "n_out": 2}}))

    def test_fine_tunes_need_block_version_3(self):
        """A chain can't hold a fine-tune until its blocks are version 3, which nodes before fine-tunes refuse: so a
        validator that hasn't upgraded stops at that block instead of forking off."""
        from vibehash import ledger
        req = KeyPair.generate(b"req")
        chain = Chain({req.address: 100 * VIBE})
        tx = ledger.Tx.signed(req, "post_job", ledger.SYSTEM, 10 * VIBE, 0, job_of().spec_json(), chain.chain_id)
        for version, ok in ((2, False), (3, True)):
            state = copy.deepcopy(chain.state)
            with self.subTest(version=version):
                if ok:
                    ledger._apply_user_tx(state, tx, 1, chain.chain_id, version)
                    self.assertIn("ft", state.jobs)
                else:
                    with self.assertRaisesRegex(ledger.InvalidBlock, "version 3"):
                        ledger._apply_user_tx(state, tx, 1, chain.chain_id, version)
        self.assertEqual(ledger.VERSION, ledger.FINETUNE_VERSION)

    def test_an_oversized_score_is_refused_not_a_crash(self):
        from vibehash.coordinator import ProtocolError, _score
        for bad in ([10 ** 400, 0.5], [float("inf"), 0.5], [1.0, 2.0], ["a", 0.5], [1.0], None):
            with self.subTest(bad=bad), self.assertRaises(ProtocolError):
                _score(bad)
        self.assertEqual(_score([1.5, 0.25]), (1.5, 0.25))

    def test_a_fine_tune_round_has_room_for_its_disputes(self):
        """Validators wait for a proposal as long as the round can take: a fine-tune's has a dispute phase, long enough
        for every move of a dispute made just in time. And no phase is longer than an hour, whatever a spec asks for."""
        from types import SimpleNamespace

        from vibehash.node import DISPUTE_MOVES, MAX_PHASE_S, Node, PhaseTimes
        node = SimpleNamespace(times=PhaseTimes())
        t = Node.times_for(node, job_of())
        self.assertEqual((t.move_s, t.dispute_s), (t.verify_s, DISPUTE_MOVES * t.verify_s))
        self.assertEqual(t.round_s, t.register_s + t.compute_s + t.reveal_s + t.verify_s + t.dispute_s)
        self.assertEqual(Node.times_for(node, TrainingJob("j", "", 1, 1)).dispute_s, 0)
        huge = TrainingJob("big", "", 1, 1, spec=ModelSpec(4096, (8192,) * 3, 1000, "float32"), batch_size=4096,
                           steps_per_unit=100_000, checkpoint_every=100_000)
        self.assertTrue(all(x <= MAX_PHASE_S for x in (Node.times_for(node, huge).compute_s,
                                                       Node.times_for(node, huge).verify_s)))

    def test_mlp_specs_are_unchanged(self):
        """Fine-tunes have a spec of their own; MLP jobs' specs keep exactly the fields they had."""
        mlp = TrainingJob("j", "", 1, 1, spec=ModelSpec(4, (8,), 3, "float32"))
        self.assertFalse(mlp.finetune)
        self.assertEqual(set(json.loads(mlp.spec_json())), {
            "id", "fee_per_round", "model", "precision", "data", "init_seed", "lr", "batch_size", "steps_per_unit",
            "checkpoint_every", "units_per_round", "memory_mb", "tolerance", "max_update_ratio"})

    def test_bad_specs_are_refused(self):
        good = json.loads(job_of().spec_json())
        edits = {
            "unknown field": lambda d: d.update(colour="red"),
            "base not pinned": lambda d: d.update(base="hf:Qwen/Qwen3-0.6B@main"),
            "manifest": lambda d: d.update(manifest="xyz"),
            "family": lambda d: d["model"].update(family="gemma3"),
            "a missing dimension": lambda d: d["model"].pop("ffn"),
            "rank": lambda d: d["lora"].update(rank=0),
            "targets": lambda d: d["lora"].update(targets=["q", "lm_head"]),
            "no targets": lambda d: d["lora"].update(targets=[]),
            "seq_len": lambda d: d.update(seq_len=8),
            "optimizer": lambda d: d.update(optimizer="sgd"),
            "precision": lambda d: d.update(precision="float64"),
            "data kind": lambda d: d.update(data={"kind": "npz", "sha256": "b" * 64}),
            "lr": lambda d: d.update(lr=2.0),
            "segments": lambda d: d.update(checkpoint_every=5),
        }
        for what, edit in edits.items():
            d = copy.deepcopy(good)
            edit(d)
            with self.subTest(what), self.assertRaises(InvalidJob):
                TrainingJob.from_spec_json(json.dumps(d), "", 0)
        with self.assertRaises(InvalidJob):
            job_of(shards=2)

    def test_sizes(self):
        s = spec_of()
        # rank 16 on q, k, v, o of Qwen3-0.6B: 28 layers × 16 × (2048 + 1024 + 1024 + 1024)
        self.assertEqual(s.n_adapter, 28 * 16 * 5120)
        self.assertEqual(s.n_params, 3 * s.n_adapter)
        self.assertEqual([p.stop - p.start for p in s.parts], [s.n_adapter] * 3)
        self.assertGreater(s.flops_per_step(8), s.flops_per_step(4))
        job = job_of()
        self.assertEqual(job.checkpoint_bytes, 4 * s.n_params)
        lengths = job.phase_lengths()
        self.assertGreater(lengths["compute"], job.unit_flops / 5e12)  # a gaming GPU's rate, not the laptop CPU's
        self.assertLess(lengths["compute"], 600)
        self.assertGreater(ft.memory_mb(s, 8, 1_500_000_000), 2000)

    def test_frozen_a_is_drawn_like_peft_and_pinned(self):
        s = spec_of(targets=("q", "o"))
        a = s.lora_a()
        self.assertEqual([x.shape for x in a[:2]], [(16, 1024), (16, 2048)])
        for x, (_, _, inp) in zip(a, s.modules()):
            self.assertLessEqual(np.abs(x).max(), 1 / np.sqrt(inp))
            self.assertGreater(np.abs(x).max(), 0.99 / np.sqrt(inp))
        self.assertEqual(a[0].dtype, np.float32)
        # Every worker must draw the same A; PCG64's raw stream is stable across numpy versions, and this pins it.
        self.assertEqual(hashlib.sha256(np.concatenate([x.ravel() for x in a]).tobytes()).hexdigest()[:16],
                         PINNED_A)
        self.assertFalse(np.array_equal(spec_of(seed=12).lora_a()[0], a[0]))

    def test_adam_bound_holds_for_any_gradients(self):
        """No sequence of gradients moves a weight further than max_step says, and honest-looking ones come close."""
        s = spec_of()
        b1, b2, eps = ft.OPTIMIZERS["adam"]
        rng = np.random.default_rng(0)
        for steps in (1, 4, 16, 64):
            bound, worst = s.max_step(1.0, steps), 0.0
            for trial in range(300):
                kind = trial % 3
                g = (rng.normal(0, 1, steps) * np.logspace(rng.uniform(-8, 0), rng.uniform(-8, 0), steps) if kind == 0
                     else np.r_[np.full(steps - 1, 1e-6), 1.0] if kind == 1 else np.ones(steps))
                m = v = x = 0.0
                for t in range(1, steps + 1):
                    m, v = b1 * m + (1 - b1) * g[t - 1], b2 * v + (1 - b2) * g[t - 1] ** 2
                    x -= (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
                worst = max(worst, abs(x))
            self.assertLessEqual(worst, bound)
            self.assertGreater(worst, 0.95 * steps)
        self.assertAlmostEqual(s.max_step(1.0, 1), 1.0)

    def test_fresh_zeroes_only_the_moments(self):
        s = spec_of(targets=("q",))
        theta = np.arange(s.n_params, dtype=np.float64)
        out = s.fresh(theta)
        np.testing.assert_array_equal(out[s.parts[0]], theta[s.parts[0]])
        self.assertFalse(out[s.n_adapter:].any())
        np.testing.assert_array_equal(s.init_params(), np.zeros(s.n_params))

    def test_peft_config(self):
        c = spec_of(rank=8, alpha=16.0, targets=("q", "v", "down")).peft_config()
        self.assertEqual((c["peft_type"], c["r"], c["lora_alpha"], c["target_modules"]),
                         ("LORA", 8, 16, ["q_proj", "v_proj", "down_proj"]))
        self.assertEqual((c["base_model_name_or_path"], c["revision"]), ("Qwen/Qwen3-0.6B", "c" * 40))


PINNED_A = "13d870abcba58031"  # sha256 of spec_of(targets=("q", "o")).lora_a(), first 16 hex digits


@needs_torch
class DataTests(unittest.TestCase):
    def setUp(self):
        self.dir = tmpdir(self)
        self.tiny = TinyModel(self.dir)
        self.data = self.tiny.dataset()

    def test_round_trip_and_sniffing(self):
        path = self.data.save(self.dir / "d.npz")
        back = load_dataset(path)
        self.assertIsInstance(back, ft.TokenDataset)
        for name in ("train_ids", "train_len", "train_mask", "val_ids", "val_len", "val_mask"):
            np.testing.assert_array_equal(getattr(back, name), getattr(self.data, name))
        self.assertEqual((back.manifest, back.config), (self.data.manifest, self.data.config))
        back.check(self.tiny.spec())
        mlp = self.dir / "t.npz"
        d = Dataset.synthetic(ModelSpec(), 1, 16, 4)
        np.savez(mlp, x_train=d.x_train, y_train=d.y_train, x_val=d.x_val, y_val=d.y_val)
        self.assertIsInstance(load_dataset(mlp), Dataset)

    def test_batches_are_cut_to_their_longest(self):
        ids, mask = self.data.batch([0, 1])
        self.assertEqual(ids.shape[1], max(self.data.train_len[:2]))
        self.assertEqual(ids.dtype, np.int64)

    def test_malformed_datasets_are_refused(self):
        d = self.data

        def broken(**changes):
            arrays = {k: getattr(d, k).copy() for k in ("train_ids", "train_len", "train_mask", "val_ids", "val_len", "val_mask")}
            for k, f in changes.items():
                f(arrays[k])
            return ft.TokenDataset(**arrays, manifest=d.manifest, config=d.config)
        cases = {
            "learning position 0": dict(train_mask=lambda m: m.__setitem__((0, 0), 1)),
            "learning past the end": dict(train_mask=lambda m: m.__setitem__((0, -1), 1), train_len=lambda n: n.__setitem__(0, 5)),
            "tokens past the end": dict(train_ids=lambda x: x.__setitem__((0, -1), 7), train_len=lambda n: n.__setitem__(0, 5)),
            "nothing to learn": dict(val_mask=lambda m: m.__setitem__(0, 0)),
            "negative ids": dict(train_ids=lambda x: x.__setitem__((0, 1), -3)),
            "too long": dict(train_len=lambda n: n.__setitem__(0, 999)),
        }
        for what, change in cases.items():
            with self.subTest(what), self.assertRaises(ValueError):
                broken(**change).check_shape()
        with self.assertRaises(ValueError):
            ft.TokenDataset(d.train_ids, d.train_len, d.train_mask, d.val_ids, d.val_len, d.val_mask,
                            manifest=b"{}", config=d.config).check_shape()
        path = self.dir / "wrong-dtype.npz"
        arrays = {k: getattr(d, k) for k in ("train_ids", "train_len", "train_mask", "val_ids", "val_len", "val_mask")}
        np.savez(path, **{**arrays, "train_ids": d.train_ids.astype(np.int64)},
                 manifest=np.frombuffer(d.manifest, np.uint8), config=np.frombuffer(d.config, np.uint8))
        with self.assertRaisesRegex(ValueError, "int32"):
            load_dataset(path)

    def test_a_config_that_isnt_a_model_is_refused_on_arrival(self):
        """A dataset whose config.json lacks a size passed the upload check before, then made every round of every job
        fail. Now it's refused when it arrives."""
        from vibehash.storage import Store
        d = self.data
        bad = {k: v for k, v in self.tiny.config.items() if k != "num_attention_heads"}
        broken = ft.TokenDataset(d.train_ids, d.train_len, d.train_mask, d.val_ids, d.val_len, d.val_mask,
                                 manifest=d.manifest, config=json.dumps(bad).encode())
        path = broken.save(self.dir / "bad.npz")
        with self.assertRaisesRegex(ValueError, "num_attention_heads"):
            load_dataset(path)
        store = Store(self.dir / "node")
        self.addCleanup(store.close)
        with self.assertRaises(ValueError):
            store.add_dataset(path)
        for key, value in (("hidden_size", 0), ("num_key_value_heads", 3), ("head_dim", 7), ("vocab_size", "96")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ft.arch_dims({**self.tiny.config, key: value})

    def test_dataset_must_suit_the_job(self):
        spec = self.tiny.spec()
        other = TinyModel(self.dir / "other", seed=5)
        with self.assertRaisesRegex(ValueError, "manifest isn't the one"):
            other.dataset().check(spec)
        with self.assertRaisesRegex(ValueError, "longer than"):
            self.data.check(self.tiny.spec(seq_len=16))
        bad = self.tiny.dataset()
        bad.train_ids[0, 1] = self.tiny.vocab
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            bad.check(spec)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.data.check(ft.LoraSpec(spec.base, spec.manifest, {**spec.model, "ffn": 99}, seq_len=32))
        job = job_of(spec)
        path = self.data.save(self.dir / "d.npz")
        with self.assertRaisesRegex(InvalidJob, "hash"):
            job.attach_data(load_dataset(path), "0" * 64)
        job.data_source = {"kind": "tokens", "sha256": file_sha256(path)}
        job.attach_data(load_dataset(path), file_sha256(path))
        with self.assertRaisesRegex(InvalidJob, "token"):
            job.attach_data(Dataset.synthetic(ModelSpec(), 1, 8, 2), file_sha256(path))


@unittest.skipUnless(HAVE_TOKENIZERS, "needs the tokenizers and jinja2 packages")
class ChatTests(unittest.TestCase):
    TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")

    def tokenizer(self, template=TEMPLATE) -> ft.ChatTokenizer:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.train_from_iterator(["hello there how are you", "I am fine thanks", "user assistant system"] * 20,
                                trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]", "<|im_start|>", "<|im_end|>"],
                                                    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        return ft.ChatTokenizer(tok.to_str().encode(), {"eos_token": "<|im_end|>"}, template)

    def test_only_the_replies_and_their_end_are_learned(self):
        t = self.tokenizer()
        chat = [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hello there"},
                {"role": "assistant", "content": "I am fine"}, {"role": "user", "content": "how are you"},
                {"role": "assistant", "content": "fine thanks"}]
        ids, mask = t.chat(chat)
        learned = t.tok.decode([i for i, m in zip(ids, mask) if m], skip_special_tokens=False)
        self.assertEqual(learned.replace(" ", ""), "Iamfine<|im_end|>finethanks<|im_end|>")
        self.assertEqual(t.tok.decode(ids, skip_special_tokens=False), t.render(chat))

    def test_a_template_that_changes_the_reply_is_caught(self):
        t = self.tokenizer(self.TEMPLATE.replace("{{ m.content }}", "{{ m.content | upper }}"))
        with self.assertRaisesRegex(ft.DataError, "message 2"):
            t.chat([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}])

    def test_reading_examples(self):
        d = tmpdir(self)
        good = d / "chats.jsonl"
        good.write_text("\n".join(json.dumps(r) for r in [
            {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hello there"}]},
            {"text": "how are you"}, {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "I am fine"}]}]))
        self.assertEqual(len(ft.read_examples(good)), 3)
        bad = d / "data.jsonl"
        for lines, message in ((['{"text": "a"}', '{"text": "b"}', '{"prompt": "c"}'], "Line 3 of data.jsonl has no `text` or `messages`"),
                               (['{"messages": [{"role": "user", "content": "hi"}]}'], "no assistant message"),
                               (["{not json"], "line 1 of data.jsonl isn't JSON")):
            bad.write_text("\n".join(lines))
            with self.subTest(message), self.assertRaisesRegex(ft.DataError, message):
                ft.read_examples(bad)

    @needs_torch
    def test_prepare_holds_some_back_and_sizes_sequences(self):
        d = tmpdir(self)
        tiny = TinyModel(d)
        path = d / "chats.jsonl"
        path.write_text("\n".join(json.dumps({"messages": [{"role": "user", "content": "hello there " * (i % 5 + 1)},
                                                         {"role": "assistant", "content": "I am fine " * (i % 7 + 1)}]})
                                  for i in range(40)))
        p = ft.prepare(path, self.tokenizer(), tiny.manifest_bytes, tiny.config_bytes, max_len=1024)
        self.assertEqual((p.examples, p.kept, p.cut, p.chats), (40, 40, 0, 40))
        self.assertEqual(len(p.data.val_len), 4)
        self.assertEqual(p.seq_len % ft.SEQ_STEP, 0)
        self.assertGreaterEqual(p.seq_len, int(p.data.train_len.max()))
        short = ft.prepare(path, self.tokenizer(), tiny.manifest_bytes, tiny.config_bytes, max_len=20)
        self.assertGreater(short.cut, 0)
        self.assertEqual(short.seq_len, 20)
        again = ft.prepare(path, self.tokenizer(), tiny.manifest_bytes, tiny.config_bytes, max_len=1024)
        np.testing.assert_array_equal(again.data.val_ids, p.data.val_ids)  # the split follows the file


@needs_torch
class ModelTests(unittest.TestCase):
    def test_matches_the_numpy_reference_in_float64(self):
        """llm.py and transformer.py are written apart; on a Llama they agree to float64 rounding, loss and LoRA
        gradient both."""
        from vibehash import transformer as tr

        config = {k: v for k, v in CONFIGS["llama"].items() if k != "rope_scaling"} | {"rope_theta": 10000.0}
        tiny = TinyModel(tmpdir(self), "llama", **{k: config[k] for k in ("rope_theta",)}, rope_scaling=None)
        arch = llm.Arch.from_hf_config(tiny.config)
        tensors = {n: _array(*t) for n, t in tiny.tensors.items()}
        rspec = tr.TransformerSpec.from_hf_config(tiny.config, seq_len=12, precision="float64", lora_rank=4,
                                                  lora_alpha=8.0, lora_targets=("q", "v", "down"))
        base = tr.pack(tr.layout(rspec), tensors)
        rng = np.random.default_rng(3)
        adapters = tr.init_adapters(rspec, 5)
        adapters += rng.normal(0, 0.1, adapters.shape) * np.array(["lora_B" in n for n, s in tr.adapter_layout(rspec)
                                                                  for _ in range(int(np.prod(s)))])
        tokens = rng.integers(0, arch.vocab, (3, 13))
        want_loss = tr.loss(rspec, base, adapters, tokens)
        want_grad = tr.unpack(tr.adapter_layout(rspec), tr.gradient(rspec, base, adapters, tokens))

        model = llm.Model(arch, {n: torch.from_numpy(a) for n, a in tensors.items()}, dtype=torch.float64)
        named = tr.unpack(tr.adapter_layout(rspec), adapters)
        ad = {}
        for name in {n.rsplit(".lora_", 1)[0] for n in named}:
            ad[name] = (torch.from_numpy(named[name + ".lora_A.weight"].copy()),
                        torch.from_numpy(named[name + ".lora_B.weight"].copy()).requires_grad_())
        mask = torch.ones(tokens.shape, dtype=torch.uint8)
        mask[:, 0] = 0
        total, count = model.loss(torch.from_numpy(tokens), mask, ad, rspec.scaling)
        (total / count).backward()
        total = total.detach()
        self.assertAlmostEqual((total / count).item(), want_loss, delta=1e-12)
        for name, (_, b) in ad.items():
            np.testing.assert_allclose(b.grad.numpy(), want_grad[name + ".lora_B.weight"], rtol=1e-9, atol=1e-13)

    def test_fused_head_and_loss_match_autograd(self):
        torch.manual_seed(0)
        h = torch.randn(1100, 24, dtype=torch.float64, requires_grad=True)  # more than one chunk
        w = torch.randn(50, 24, dtype=torch.float64)
        t = torch.randint(0, 50, (1100,))
        llm._HeadLoss.apply(h, w, t).backward()
        h2 = h.detach().clone().requires_grad_()
        torch.nn.functional.cross_entropy(h2 @ w.T, t, reduction="sum").backward()
        torch.testing.assert_close(h.grad, h2.grad)

    def test_recomputing_layers_changes_no_bits(self):
        tiny = TinyModel(tmpdir(self)).use(self)
        spec, data = tiny.spec(), tiny.dataset()
        rt = ft.runtime(spec, data, "cpu")
        ids, mask = (torch.from_numpy(x) for x in data.batch(np.arange(8)))
        grads = []
        for recompute in (True, False, True):
            rt.model.recompute = recompute
            b = [torch.full(shape, 0.01).requires_grad_() for _, _, _, shape in rt.lora[spec]]
            total, _ = rt.model.loss(ids, mask, {n: (a, x) for (n, a, _, _), x in zip(rt.lora[spec], b)}, spec.scaling)
            total.backward()
            grads.append(torch.cat([x.grad.ravel() for x in b]))
        self.assertTrue(torch.equal(grads[0], grads[1]) and torch.equal(grads[0], grads[2]))

    def test_unsupported_models_are_refused(self):
        for config, why in (({**CONFIGS["qwen3"], "model_type": "gemma3"}, "gemma3"),
                            ({**CONFIGS["llama"], "rope_scaling": {"rope_type": "yarn", "factor": 4.0}}, "yarn"),
                            ({**CONFIGS["llama"], "hidden_act": "gelu"}, "gelu")):
            with self.subTest(why), self.assertRaisesRegex(ValueError, why):
                llm.Arch.from_hf_config(config)
        arch = llm.Arch.from_hf_config({**CONFIGS["qwen3"], "model_type": "mistral", "sliding_window": 64})
        with self.assertRaisesRegex(ValueError, "sliding window"):
            arch.check(128)
        self.assertEqual(llm.Arch.from_hf_config(CONFIGS["smollm3"]).nope, (3,))


def _array(dtype, shape, raw) -> np.ndarray:
    """bf16 bytes to float64."""
    assert dtype == "BF16"
    return (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(shape)


@unittest.skipUnless(HAVE_HF, "needs transformers and peft")
class HuggingFaceTests(unittest.TestCase):
    """The same tiny checkpoints through Hugging Face transformers, and adapters through PEFT."""

    def load_hf(self, folder):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32, attn_implementation="eager").eval()

    def test_every_family_matches_transformers(self):
        for family in CONFIGS:
            with self.subTest(family):
                tiny = TinyModel(tmpdir(self), family)
                ours = llm.Model(llm.Arch.from_hf_config(tiny.config),
                                 {n: torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(s)
                                  for n, (_, s, raw) in tiny.tensors.items()})
                tokens = torch.randint(0, tiny.vocab, (2, 96), generator=torch.Generator().manual_seed(1))
                with torch.no_grad():
                    want = self.load_hf(tiny.folder)(tokens).logits
                    got = ours.logits(tokens)
                torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    def test_llama3_rotary_scaling_matches_transformers(self):
        from transformers import AutoConfig
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        config = AutoConfig.for_model(**CONFIGS["llama"])
        inv, _ = ROPE_INIT_FUNCTIONS["llama3"](config, "cpu")
        cos, _ = llm.rope_table(llm.Arch.from_hf_config(CONFIGS["llama"]), 2)
        theirs = torch.cos(inv.double())
        torch.testing.assert_close(cos[1, :len(inv)], theirs, rtol=1e-6, atol=1e-7)

    def test_an_exported_adapter_loads_in_peft_and_gives_our_outputs(self):
        from peft import PeftModel

        d = tmpdir(self)
        tiny = TinyModel(d).use(self)
        spec, data = tiny.spec(targets=("q", "v", "up")), tiny.dataset()
        theta = ft.train_segment(spec, data, spec.init_params(), 3, 0, 8, 1e-2, 8)
        out = ft.export_peft(spec, theta, d / "adapter", {"job_id": "test"})
        self.assertEqual(json.loads((out / "adapter_config.json").read_text())["target_modules"], ["q_proj", "v_proj", "up_proj"])
        self.assertIn(spec.commit, (out / "README.md").read_text())
        model = PeftModel.from_pretrained(self.load_hf(tiny.folder), out).eval()
        rt = ft.runtime(spec, data, "cpu")
        b = torch.from_numpy(theta[spec.parts[0]].astype(np.float32))
        ad = {n: (a, b[s].view(shape)) for n, a, s, shape in rt.lora[spec]}
        ids = torch.from_numpy(data.batch(np.arange(4))[0])
        with torch.no_grad():
            want, got = model(ids).logits, rt.model.logits(ids, ad, spec.scaling)
            base = rt.model.logits(ids)
        self.assertGreater(float((got - base).abs().max()), 1e-3)  # the adapter does something
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@needs_torch
class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tiny = TinyModel(tmpdir(self)).use(self)
        self.spec, self.data = self.tiny.spec(), self.tiny.dataset()

    def test_deterministic_and_resumable(self):
        s, d = self.spec, self.data
        a = ft.train_segment(s, d, s.init_params(), 7, 0, 8, 1e-2, 8)
        np.testing.assert_array_equal(a, ft.train_segment(s, d, s.init_params(), 7, 0, 8, 1e-2, 8))
        half = ft.train_segment(s, d, s.init_params(), 7, 0, 4, 1e-2, 8)
        np.testing.assert_array_equal(a, ft.train_segment(s, d, half, 7, 4, 4, 1e-2, 8))
        self.assertTrue(np.all(a[s.parts[2]] >= 0) and np.any(a[s.parts[1]]))
        self.assertFalse(np.array_equal(a, ft.train_segment(s, d, s.init_params(), 8, 0, 8, 1e-2, 8)))

    def test_averaging_b_averages_the_change_to_the_model(self):
        """With A frozen, the network's average of units' B gives exactly the average of their changes to every weight."""
        s, d = self.spec, self.data
        finals = [ft.train_segment(s, d, s.init_params(), u, 0, 8, 1e-2, 8) for u in range(3)]
        mean_b = np.mean([f[s.parts[0]] for f in finals], axis=0)

        def deltas(b):
            out, i = [], 0
            for (_, rows, _), a in zip(s.modules(), s.lora_a()):
                out.append(s.scaling * b[i:i + rows * s.rank].reshape(rows, s.rank) @ a.astype(np.float64))
                i += rows * s.rank
            return out
        for got, *each in zip(deltas(mean_b), *(deltas(f[s.parts[0]]) for f in finals)):
            np.testing.assert_allclose(got, np.mean(each, axis=0), rtol=1e-12, atol=1e-15)

    def test_training_learns(self):
        s, d = self.tiny.spec(), self.tiny.dataset(task="copy", n_train=64)
        theta = s.init_params()
        before, _ = ft.evaluate(s, d, theta)
        for r in range(6):
            theta = s.fresh(np.mean([ft.train_segment(s, d, theta, 10 * r + u, 0, 16, 1e-2, 8) for u in range(2)], axis=0))
        after, _ = ft.evaluate(s, d, theta)
        self.assertLess(after, before - 0.1)

    def test_checks_catch_doctored_results(self):
        s, d, lr = self.spec, self.data, 1e-2
        start = ft.train_segment(s, d, s.init_params(), 1, 0, 4, lr, 8)
        honest = ft.train_segment(s, d, start, 1, 4, 4, lr, 8)
        self.assertTrue(reproduces(honest, honest, start, 1e-3, s))
        self.assertTrue(s.sane(s.init_params(), honest, lr, 8, 1.0))
        self.assertFalse(s.sane(s.init_params(), s.init_params() + 3 * honest, lr, 8, 1.0), "a boosted update")
        negative = honest.copy()
        negative[s.parts[2].start] = -1e-9
        self.assertFalse(s.sane(s.init_params(), negative, lr, 8, 1.0))
        from vibehash.crypto import Draw
        canary = rescale_update(s, start, honest, Draw(b"t"))
        self.assertFalse(reproduces(canary, honest, start, 1e-3, s))

    def test_moments_are_checked_on_their_own(self):
        """In a real model B grows round after round while the moments stay gradient-sized, so in one norm a worker
        could point its momentum anywhere and pass; part by part, it can't."""
        from vibehash.verification import _reproduces
        s, rng = self.spec, np.random.default_rng(0)
        b, m, v = s.parts
        start = np.zeros(s.n_params)
        start[b] = rng.normal(0, 1e-2, s.n_adapter)
        honest = start.copy()
        honest[b] += rng.normal(0, 1e-3, s.n_adapter)
        honest[m] = rng.normal(0, 1e-6, s.n_adapter)
        honest[v] = honest[m] ** 2 * 10
        doctored = honest.copy()
        doctored[m] = -honest[m]  # momentum the other way
        self.assertTrue(_reproduces(doctored, honest, start, 1e-3, s))
        self.assertFalse(reproduces(doctored, honest, start, 1e-3, s))
        self.assertTrue(reproduces(honest, honest, start, 1e-3, s))


@needs_torch
class ProtocolTests(unittest.TestCase):
    """Fine-tune rounds through the same Coordinator a node runs, with cheaters, then replayed and rechecked."""

    def test_rounds_train_pay_catch_cheats_and_replay(self):
        from vibehash.protocol import Protocol
        from vibehash.replay import replay_block
        from vibehash.worker import HardwareProfile, Worker

        d = tmpdir(self)
        tiny = TinyModel(d).use(self)
        path = tiny.dataset().save(d / "data.npz")
        sha = file_sha256(path)
        behaviors = [("honest", "honest")] * 7 + [("lazy", "honest"), ("freezer", "honest"), ("poisoner", "honest"),
                                                  ("shortcut", "honest")]
        hw = HardwareProfile("gpu", 1e12, 16384, 1.0)
        workers = [Worker(f"w{i}", KeyPair.generate(f"w{i}".encode()), hw, *b, fp_noise=0) for i, b in enumerate(behaviors)]
        req = KeyPair.generate(b"req")
        chain = Chain({req.address: 1000 * VIBE}, stakes={w.address: 10 * VIBE for w in workers})
        proto = Protocol(chain, workers, CoordinatorConfig(), seed=3)
        job = job_of(tiny.spec(), data_source={"kind": "tokens", "sha256": sha}, units_per_round=12, lr=1e-2,
                     memory_mb=2048, budget=100 * VIBE, fee_per_round=10 * VIBE)
        job.attach_data(load_dataset(path), sha)
        proto.post_job(req, job)
        before = copy.deepcopy(proto.chain)
        reports = [proto.run_round() for _ in range(2)]
        by = {w.address: w for w in workers}
        for rep in reports:
            for res in rep.results:
                honest = by[res.worker].behavior == "honest"
                self.assertEqual(res.verdict.value == "accepted", honest, (by[res.worker].behavior, res.verdict))
            self.assertTrue(np.isfinite(rep.val_loss))
            self.assertFalse(rep.params[tiny.spec().n_adapter:].any(), "the next round starts Adam afresh")
        self.assertFalse([s for rep in reports for s in rep.slashes if s[2] in ("approved a canary", "false accusation")])
        for w in workers[7:]:
            self.assertLess(proto.chain.state.stake(w.address), 10 * VIBE, w.behavior)

        class Recheck:
            def array(self, bid, job):
                return proto.checkpoints.array(bid, job)

            def attach(self, j):
                j.attach_data(load_dataset(path), sha)
        chain, stats = copy.deepcopy(before), {}
        for rep in reports:
            self.assertEqual(replay_block(chain, rep.block, copy.deepcopy(rep.record), Recheck(), stats), [])
            chain.append(rep.block)
        self.assertGreater(stats["rechecked"], 10)
        self.assertEqual(stats["unchecked"], 0)


    def test_canaries_pass_adams_invariants_and_rescaled_ones_dont(self):
        """Why canaries changed: a rescaled [B | m | v] update breaks Adam's invariants, so a checker can spot it with
        numpy alone; another unit's real segment, which canaries now are, keeps them, as honest work does."""
        from vibehash.crypto import Draw
        tiny = TinyModel(tmpdir(self)).use(self)
        spec, data = tiny.spec(), tiny.dataset()
        runs = []
        for seed in (100, 200):
            ckpts = [spec.init_params()]
            for seg in range(3):
                ckpts.append(spec.round(ft.train_segment(spec, data, ckpts[-1], seed, 4 * seg, 4, 1e-2, 8)))
            runs.append(ckpts)
        honest = [(c[k], c[k + 1]) for c in runs for k in range(3)]
        self.assertFalse(any(adam_impossible(spec, a, b, 4) for a, b in honest))
        rescaled = [(a, rescale_update(spec, a, b, Draw(b"old", bytes([i])))) for i, (a, b) in enumerate(honest)]
        self.assertGreater(sum(adam_impossible(spec, a, b, 4) for a, b in rescaled), len(rescaled) // 2)
        # Today's canary for unit 1: unit 2's segment. It passes every invariant, and fails a replay under unit 1's seed.
        a, b = runs[1][1], runs[1][2]
        self.assertFalse(adam_impossible(spec, a, b, 4))
        self.assertFalse(reproduces(b, spec.round(ft.train_segment(spec, data, a, 100, 4, 4, 1e-2, 8)), a, 1e-4, spec))

    def test_canaries_of_fine_tunes_fail_replay(self):
        """A canary copies a real reveal's challenged checkpoints with one segment's update rescaled: an honest checker
        replaying it rejects it, as it accepts the real one."""
        from vibehash.crypto import Draw
        from vibehash.tasks import WorkUnit
        from vibehash.verification import challenge_segments, plant_canary, replay_openings
        from vibehash.worker import HardwareProfile, Worker

        tiny = TinyModel(tmpdir(self)).use(self)
        job = job_of(tiny.spec(), lr=1e-2)
        job._data = tiny.dataset()
        w = Worker("w", KeyPair.generate(b"w"), HardwareProfile("gpu", 1e12, 16384), fp_noise=0)
        start = job.spec.init_params()
        from vibehash.model import params_hash
        unit = WorkUnit.make(job, 1, 0, params_hash(start))
        w.compute(unit, job, start)
        rv = w.reveal(unit, job, b"beacon", 3)
        segs = challenge_segments(b"beacon", unit.unit_id, w.address, job.n_segments, 3)
        real = {i: rv.openings[i] for sg in segs for i in (sg, sg + 1)}
        self.assertEqual(replay_openings(job, unit, real, segs), (True, None))
        canary = plant_canary(job, real, segs, Draw(b"canary"))
        ok, bad = replay_openings(job, unit, canary, segs)
        self.assertFalse(ok)
        self.assertIn(bad, segs)


# Makes `import torch` fail in a process, for real: a finder ahead of every other on sys.meta_path.
BLOCK_TORCH = """
import sys
class _NoTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("PyTorch is blocked in this process")
sys.meta_path.insert(0, _NoTorch())
"""


class _Seeded:
    """os.urandom made repeatable (round secrets, attempt labels, salts), so the draws of a round are the same each run."""

    def __init__(self, seed: bytes = b"0"):
        self.seed, self.n = seed, 0

    def __call__(self, k: int) -> bytes:
        self.n += 1
        return hashlib.sha256(self.seed + self.n.to_bytes(8, "big")).digest()[:k] if k <= 32 else bytes(k)


@needs_torch
class NodeFreeRoundTests(unittest.TestCase):
    """Record version 4: a fine-tune's round where the node never runs the model. Rounds are driven by hand here, so
    each checker, scorer and party to a dispute does as a test needs, and then replayed from their records."""

    def network(self, behaviors, **cfg):
        from unittest import mock

        from vibehash.protocol import Protocol
        from vibehash.worker import HardwareProfile, Worker

        d = tmpdir(self)
        self.tiny = TinyModel(d).use(self)
        path = self.tiny.dataset().save(d / "data.npz")
        self.path, self.sha = path, file_sha256(path)
        hw = HardwareProfile("gpu", 1e12, 16384, 1.0)
        self.workers = [Worker(f"w{i}", KeyPair.generate(f"w{i}".encode()), hw, *b, fp_noise=0) for i, b in enumerate(behaviors)]
        self.by = {w.address: w for w in self.workers}
        req = KeyPair.generate(b"req")
        stakes = cfg.pop("stakes", None)
        chain = Chain({req.address: 1000 * VIBE}, stakes=stakes or {w.address: 10 * VIBE for w in self.workers},
                      earn_in=cfg.pop("earn_in", False))
        self.proto = Protocol(chain, self.workers, CoordinatorConfig(**cfg), seed=3)
        self.job = job_of(self.tiny.spec(), data_source={"kind": "tokens", "sha256": self.sha}, units_per_round=8,
                          lr=1e-2, memory_mb=2048, budget=100 * VIBE, fee_per_round=10 * VIBE)
        self.job.attach_data(load_dataset(path), self.sha)
        patch = mock.patch("os.urandom", _Seeded())
        patch.start()
        self.addCleanup(patch.stop)
        self.proto.post_job(req, self.job)
        self.before = copy.deepcopy(self.proto.chain)

    def round(self, answer=None, moves=None):
        """One round: everyone trains and reveals honestly as their behaviour says; verify-phase tasks are answered by
        `answer(task, worker) -> (ok, bad) | None` (None: silent; default: as the worker would); disputes are played
        by their parties (tracer.Party) unless `moves(game, who) -> move | None | "default"` says otherwise (None: it
        never moves). Returns the report."""
        p, c = self.proto, self.proto.coord
        c.open_round()
        r = c.r
        job = c.job(r.job_id)
        for addr, w in p.workers.items():
            if not c.eligible(addr):  # slashed below the minimum stake in an earlier round
                continue
            body = {"round": r.round_no, "flops_per_sec": w.hardware.flops_per_sec, "memory_mb": w.hardware.memory_mb}
            c.register(addr, w.hardware.flops_per_sec, w.hardware.memory_mb, envelope=p._signed(w, "register", body))
        c.start_compute()
        for addr, uids in list(r.assignments.items()):
            for uid in uids:
                c.commit(p.workers[addr].compute(r.units[uid], job, r.start))
        beacon = c.start_reveal()
        for uid, addr in sorted(r.commitments):
            rv = p.workers[addr].reveal(r.units[uid], job, beacon, c.config.challenge_segments)
            c.reveal(rv, wire=p._wire(rv, job))
        c.start_verify()
        self.tasks = dict(r.tasks)
        self._answer(answer)
        self.disputed = c.start_disputes()
        if self.disputed:
            self._play(moves)
        self.games = dict(r.games)
        return c.finalize(proposer_keys=p._proposer(beacon), timestamp=p.chain.tip.height + 1)

    def _answer(self, policy):
        p, c = self.proto, self.proto.coord
        r, job = c.r, c.job(c.r.job_id)
        for addr, w in p.workers.items():
            for t in c.tasks_for(addr):
                if t.kind == "score":
                    got, score = (True, None), list(w.score(job, r.start))
                else:
                    got = policy(t, w) if policy else "default"
                    if got == "default":
                        got = w.verify(job, t.unit, t.openings, t.segments)
                    score = None
                if got is None:
                    continue
                ok, bad = got
                body = {"round": r.round_no, "task_id": t.task_id, "ok": bool(ok), "bad_segment": bad}
                if score is not None:
                    body["score"] = score
                c.submit_verdict(addr, t.task_id, ok, bad, envelope=p._signed(w, "verdict", body), score=score)

    def _play(self, policy):
        from vibehash.program import Program
        from vibehash.protocol import _Blobs
        from vibehash.tracer import Party
        from vibehash.worker import tamper

        p, c = self.proto, self.proto.coord
        job = c.job(c.r.job_id)
        remote, rt = _Blobs(c), ft.runtime(job.spec, job.data, "cpu")
        parties = {a: Party(rt, job.spec, job.data, remote, tamper if w.behavior == "tamper" else None)
                   for a, w in p.workers.items()}
        quiet, moved = set(), True
        while moved:
            moved = False
            for g in c.game_views():
                if g["turn"] == "done" or g["id"] in quiet:
                    continue
                who = g["challenger"] if g["turn"] == "pick" else g["defender"]
                move = policy(g, who) if policy else "default"
                if move is None:
                    quiet.add(g["id"])
                    continue
                if move == "default":
                    party, prog = parties[who], Program.of(job, int(g["seed"]))
                    move = (party.pick(prog, g) if g["turn"] == "pick" else {"split": party.split(prog, g)}
                            if g["turn"] == "split" else {"evidence": remote.put(party.evidence(prog, g))})
                body = {"round": c.r.round_no, "game": g["id"], "n": g["n"], "move": move}
                c.submit_move(who, body, envelope=p._signed(p.workers[who], "move", body))
                moved = True

    def replays(self, reports):
        from vibehash.replay import replay_block
        p = self.proto

        class Recheck:
            def array(self, bid, job):
                return p.checkpoints.array(bid, job)

            def attach(self_, j):
                j.attach_data(load_dataset(self.path), self.sha)
        chain, stats = copy.deepcopy(self.before), {}
        for rep in reports:
            self.assertEqual(rep.record["version"], 4)
            self.assertEqual(replay_block(chain, rep.block, copy.deepcopy(rep.record), Recheck(), stats), [])
            chain.append(rep.block)
        if any(rep.record["reveals"] for rep in reports):  # (with nothing revealed, there's nothing to recheck against)
            self.assertEqual(stats["unchecked"], 0)

    def results(self, rep):
        return {(res.unit.unit_id, res.worker): res for res in rep.results}

    def test_the_node_runs_no_model(self):
        """Every check that needs the model is the workers': the coordinator's own replays and scoring are off limits,
        and the round still trains, pays, scores and seals."""
        from unittest import mock

        self.network([("honest", "honest")] * 6)
        boom = AssertionError("the node ran the model")
        with mock.patch("vibehash.coordinator.replay_segment", side_effect=boom), \
                mock.patch("vibehash.coordinator.replay_openings", side_effect=boom), \
                mock.patch("vibehash.coordinator.full_replay", side_effect=boom), \
                mock.patch("vibehash.tasks.TrainingJob.evaluate", side_effect=boom), \
                mock.patch("vibehash.finetune.runtime", wraps=ft.runtime) as runs:
            rep = self.round()
        self.assertTrue(all(res.verdict.value == "accepted" for res in rep.results))
        self.assertGreater(rep.block.summary.accepted_units, 0)
        self.assertFalse(rep.slashes)
        self.assertGreater(runs.call_count, 0)  # (the workers ran it)
        loss, acc = rep.record["score"]
        want, want_acc = ft.evaluate(self.job.spec, self.job.data, self.job.spec.init_params())
        self.assertAlmostEqual(loss, want, places=4)  # the scorers scored the round's start: here, the base model
        self.assertEqual((rep.val_loss, rep.val_acc), (loss, acc))
        checkers = defaultdict(set)
        for t in self.tasks.values():
            if t.kind == "check" and not t.canary:
                checkers[(t.unit.unit_id, t.submitter)].add(t.verifier)
                self.assertNotEqual(t.verifier, t.submitter)
        self.assertTrue(all(len(v) == 2 for v in checkers.values()), "two checkers a claim")
        self.assertEqual(sum(t.kind == "score" for t in self.tasks.values()), 3)
        self.replays([rep])

    def test_disputes_catch_a_tampered_op_and_false_accusations(self):
        """A worker that scaled one op's output deep in one step of each segment, checked by honest checkers, and a
        checker that rejects everything. Every rejection becomes a dispute the node referees: the tampering worker
        loses on that very op (and is slashed, its accuser the catcher); the false accuser, finding nothing the referee
        would refuse in the honest work it disputes, withdraws each of its own (unpaid for checking and marked missed,
        but not slashed: an honest checker mistaken the same way mustn't be); and every honest claim is accepted and
        paid. Then the round replays."""
        self.network([("honest", "honest")] * 5 + [("tamper", "honest"), ("honest", "false_accuser")],
                     replication_rate=0.0, canaries_per_round=0)
        cheat, liar = self.workers[-2].address, self.workers[-1].address
        rep = self.round()
        self.assertTrue(self.disputed)
        res, slashes = self.results(rep), rep.slashes
        cheats = [v for k, v in res.items() if k[1] == cheat]
        self.assertTrue(cheats and all(v.verdict.value == "failed_replay" for v in cheats), [v.detail for v in cheats])
        self.assertTrue(all("attn_dv → b.dv" in v.detail for v in cheats), [v.detail for v in cheats])
        for k, v in res.items():
            if k[1] != cheat:
                self.assertEqual(v.verdict.value, "accepted", v.detail)
        self.assertTrue(any(s[0] == cheat and s[1] not in (None, cheat) for s in slashes))
        self.assertEqual({s[0] for s in slashes}, {cheat})
        mine = [g for g in self.games.values() if g.challenger == liar]
        self.assertTrue(mine and all(g.outcome in ("withdrawn", "challenger") for g in mine), [g.outcome for g in mine])
        self.assertTrue(any(g.outcome == "withdrawn" for g in mine))
        self.assertNotIn(liar, rep.block.summary.verifier_flops)  # unpaid for checking
        self.assertIn(liar, rep.block.summary.missed)
        self.assertTrue(all(g.outcome == "challenger" for g in self.games.values() if g.defender == cheat))
        paid = rep.block.summary.worker_flops
        self.assertNotIn(cheat, paid)
        self.assertTrue(set(paid) >= {w.address for w in self.workers[:5]} & {k[1] for k in res})
        games = rep.record["games"]
        self.assertTrue(all(e["moves"] and len(e["moves"]) <= 14 for e in games))
        self.assertTrue(all(e["evidence"] for e in games if e["outcome"] in ("challenger", "defender")))
        self.replays([rep])
        # What a validator votes by: a plain replay. Its referee needs the job's data (which sequences a step trains
        # on), which a node passes as its store.
        from vibehash.replay import replay_block
        bare = replay_block(copy.deepcopy(self.before), rep.block, copy.deepcopy(rep.record))
        self.assertTrue(bare and "needs its dataset" in bare[0], bare)

        class Store:
            def attach_dataset(_, j):
                j.attach_data(load_dataset(self.path), self.sha)
        self.assertEqual(replay_block(copy.deepcopy(self.before), rep.block, copy.deepcopy(rep.record),
                                      datasets=Store()), [])

    def test_silent_parties_lose(self):
        """A dispute whose defender never moves is lost by the defender; one whose challenger stops picking, by the
        challenger. Both are slashed as if they had lost at the referee."""
        self.network([("honest", "honest")] * 5 + [("honest", "false_accuser")], replication_rate=0.0,
                     canaries_per_round=0)
        liar = self.workers[-1].address
        order = []

        def moves(g, who):
            if g["id"] not in order:
                order.append(g["id"])
            if g["id"] == order[0] and who == g["defender"]:
                return None  # never splits
            if len(order) > 1 and g["id"] == order[1] and who == g["challenger"] and g["n"] >= 3:
                return None  # picks once, then goes quiet
            return "default"
        rep = self.round(moves=moves)
        silent_defence, quitting = self.games[order[0]], self.games[order[1]]
        self.assertEqual((silent_defence.outcome, silent_defence.detail),
                         ("challenger", "the defender didn't split in time"))
        self.assertEqual((quitting.outcome, quitting.detail), ("defender", "the challenger didn't pick in time"))
        slashed = {(s[0], s[2]) for s in rep.slashes}
        self.assertIn((silent_defence.defender, "failed_replay"), slashed)
        self.assertIn((liar, "false accusation"), slashed)
        self.replays([rep])

    def test_replicas_that_disagree_dispute_each_other(self):
        """Every unit replicated, one segment challenged: a tampering worker's unit and its honest twin's disagree, so
        each challenges the other. The honest twin finds the tampered op and wins; the tampering one finds nothing
        wrong with its twin's unit and concedes, which costs a replica nothing. The honest twin is paid."""
        self.network([("honest", "honest")] * 8 + [("tamper", "honest")], replication_rate=1.0, challenge_segments=1,
                     canaries_per_round=0)
        cheat = self.workers[-1].address
        for _ in range(3):
            rep = self.round()
            twins = [g for g in self.games.values() if g.why == "replicas"]
            if twins:
                break
        self.assertTrue(twins, "a tampering worker's unit was replicated")
        by_defender = {g.defender: g for g in twins}
        self.assertEqual(by_defender[cheat].outcome, "challenger", by_defender[cheat].detail)
        self.assertIn("attn_dv", by_defender[cheat].detail)
        honest = [g for g in twins if g.defender != cheat]
        self.assertTrue(honest and all(g.outcome == "conceded" for g in honest))
        res = self.results(rep)
        self.assertTrue(all(v.verdict.is_fraud for k, v in res.items() if k[1] == cheat))
        self.assertTrue(all(v.verdict.value == "accepted" for k, v in res.items() if k[1] != cheat))
        self.assertFalse([s for s in rep.slashes if s[0] != cheat])

    def test_canaries_catch_a_checker_that_spots_rescaled_updates(self):
        """A lazy checker that knows Adam: it approves without replaying unless a segment breaks what Adam's moments
        must satisfy, which spotted most of the old, rescaled canaries. Today's are real training under another unit's
        seed, so it approves one and is slashed; and once it has, its approvals count for nothing (a claim only it
        approved ends unchecked)."""
        self.network([("honest", "honest")] * 7 + [("honest", "knowing")])
        knowing = self.workers[-1].address

        def answer(t, w):  # everyone as their behaviour says; the knowing checker's rejections are noted
            got = w.verify(self.job, t.unit, t.openings, t.segments)
            if w.address == knowing:
                flagged.append(not got[0])
            return got
        flagged, reports = [], []
        for _ in range(5):
            reports.append(self.round(answer))
            if any(s[0] == knowing and s[2] == "approved a canary" for s in reports[-1].slashes):
                break
        caught = reports[-1]
        self.assertTrue(any(s[0] == knowing and s[2] == "approved a canary" for s in caught.slashes),
                        "the knowing checker was given a canary and approved it")
        self.assertFalse(any(flagged), "honest work and canaries both pass Adam's invariants")
        alone = [k for k in {(t.unit.unit_id, t.submitter) for t in self.tasks.values() if t.kind == "check" and not t.canary}
                 if {t.verifier for t in self.tasks.values() if (t.unit.unit_id, t.submitter) == k and t.verdict
                     is not None and not t.canary} == {knowing}]
        for key in alone:
            self.assertEqual(self.results(caught)[key].verdict.value, "unchecked")
        self.replays(reports)

    def test_two_workers_are_enough_to_start(self):
        """With the node refereeing, one honest checker a claim makes its verdict right: two staked workers train,
        checking each other."""
        from vibehash.coordinator import FINETUNE_QUORUM
        self.assertEqual(FINETUNE_QUORUM, 2)
        self.network([("honest", "honest")] * 2, replication_rate=0.0, canaries_per_round=0)
        rep = self.round()
        self.assertGreater(rep.block.summary.accepted_units, 0)
        self.assertTrue(all(res.verdict.value == "accepted" for res in rep.results))
        self.replays([rep])

    def test_the_new_model_is_averaged_and_claimed_once(self):
        """Settling a round averages the accepted results once, live and on a recheck (a leftover copy of that code
        once ran it twice, and counted the "model" claim twice on every recheck)."""
        from unittest import mock

        from vibehash.coordinator import Coordinator
        self.network([("honest", "honest")] * 3, replication_rate=0.0, canaries_per_round=0)
        real, claims = Coordinator._claim, []

        def counted(coord, kind, key, compute):
            claims.append((kind, coord.replaying is not None))
            return real(coord, kind, key, compute)
        with mock.patch.object(Coordinator, "_claim", autospec=True, side_effect=counted):
            rep = self.round()
            self.assertEqual(claims.count(("model", False)), 1)
            self.replays([rep])
        self.assertEqual(claims.count(("model", True)), 1)
        self.assertGreater(rep.block.summary.accepted_units, 0)

    def test_scorers_are_paid_only_when_they_agree(self):
        """The median of three scorers goes on record; a scorer is paid only when three answered and it agrees with
        them. One that makes its score up (and here checks nothing either) gets nothing for it."""
        self.network([("honest", "honest")] * 6 + [("honest", "lazy")], replication_rate=0.0, canaries_per_round=0)
        liar = self.workers[-1].address

        def answer(t, w):
            return None if w.address == liar else "default"
        for _ in range(6):
            rep = self.round(answer)
            if any(t.kind == "score" and t.verifier == liar for t in self.tasks.values()):
                break
        scores = {t.verifier: t.score for t in self.tasks.values() if t.kind == "score"}
        self.assertEqual(scores[liar], (0.01, 0.99))
        self.assertEqual(rep.record["score"][0], sorted(v[0] for v in scores.values())[1])
        self.assertGreater(rep.record["score"][0], 1.0)
        self.assertNotIn(liar, rep.verifier_pay)
        self.assertTrue(all(rep.verifier_pay.get(v, 0) > 0 for v in scores if v != liar))

    def test_without_a_quorum_of_staked_workers_it_waits(self):
        """On a chain where newcomers earn their stake in, a fine-tune trains only once enough staked workers can check
        each other's work; until then nothing is assigned, nobody scores (or is paid for scoring) nothing, and the
        round says why."""
        from vibehash.coordinator import FINETUNE_QUORUM
        self.network([("honest", "honest")] * FINETUNE_QUORUM, earn_in=True,
                     stakes={KeyPair.generate(f"w{i}".encode()).address: 10 * VIBE for i in range(FINETUNE_QUORUM - 1)})
        rep = self.round()
        self.assertEqual(rep.unassigned, self.job.units_per_round)
        self.assertIn(f"needs {FINETUNE_QUORUM} staked workers", rep.waiting)
        self.assertIn("ft", self.proto.coord.waiting)
        self.assertFalse(self.tasks, "no scorers when nothing was trained")
        self.assertFalse(rep.verifier_pay)
        self.replays([rep])

    def test_a_job_that_wont_load_doesnt_stop_the_others(self):
        from unittest import mock
        self.network([("honest", "honest")] * 6)
        c = self.proto.coord
        c.chain.state.escrow["broken"] = 10 * VIBE
        c.chain.state.jobs["broken"] = self.job.spec_json()
        real = c.job

        def job(job_id):
            if job_id == "broken":
                raise KeyError("num_attention_heads")
            return real(job_id)
        with mock.patch.object(c, "job", side_effect=job):
            self.assertEqual(c.active_job_id(), "ft")
        self.assertIn("broken", c.waiting)

    def test_a_round_replays_with_pytorch_blocked(self):
        """What a node does with a fine-tune's round, done in processes where importing PyTorch fails: the node's
        modules load, and a round with disputes replays and rechecks from its record and checkpoints, its referee
        recomputing each dispute's last op with NumPy. Two such processes, as two validators would, reach
        byte-identical verdicts."""
        import pickle
        import subprocess
        import sys

        self.network([("honest", "honest")] * 5 + [("tamper", "honest"), ("honest", "false_accuser")],
                     replication_rate=0.0, canaries_per_round=0)
        rep = self.round()
        self.assertTrue(rep.record["games"])
        d = tmpdir(self)
        arrays = {}
        for x in rep.record["reveals"]:
            for bid in [x["final"], *(v[0] for v in x["openings"].values())]:
                arrays[bid] = self.proto.checkpoints.array(bid, self.job)
        with open(d / "round.pkl", "wb") as f:
            pickle.dump({"before": self.before, "block": rep.block, "record": rep.record, "arrays": arrays,
                         "path": str(self.path), "sha": self.sha}, f)
        script = BLOCK_TORCH + """
import copy, hashlib, pickle
from vibehash import node, coordinator, replay, storage, finetune, dispute, referee  # noqa: F401
from vibehash.replay import replay_block
from vibehash.tasks import load_dataset
from vibehash.coordinator import Coordinator, CoordinatorConfig
from vibehash.ledger import canonical
d = pickle.load(open(sys.argv[1], "rb"))
class Recheck:
    def array(self, bid, job): return d["arrays"][bid]
    def attach(self, j): j.attach_data(load_dataset(d["path"]), d["sha"])
stats = {}
problems = replay_block(copy.deepcopy(d["before"]), d["block"], copy.deepcopy(d["record"]), Recheck(), stats)
assert "torch" not in sys.modules
verdicts = []
for e in d["record"]["games"]:
    verdicts.append([e["id"], e["outcome"], e["detail"]])
print("PROBLEMS", problems, "RECHECKED", stats["rechecked"], "UNCHECKED", stats["unchecked"])
print("VERDICTS", hashlib.sha256(canonical(verdicts)).hexdigest(), len(verdicts))
"""
        root = str(Path(__file__).resolve().parents[1])
        outs = [subprocess.run([sys.executable, "-c", script, str(d / "round.pkl")], capture_output=True, text=True,
                               cwd=root, env={**os.environ, "PYTHONPATH": root}, timeout=600) for _ in range(2)]
        for out in outs:
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            self.assertIn("PROBLEMS [] RECHECKED", out.stdout)
            self.assertIn("UNCHECKED 0", out.stdout)
        self.assertEqual(outs[0].stdout, outs[1].stdout)
        self.assertTrue(any(e["outcome"] == "challenger" and "attn_dv" in e["detail"] for e in rep.record["games"]))

    def test_a_fine_tune_record_must_be_version_4(self):
        from vibehash.coordinator import Coordinator, ProtocolError
        self.network([("honest", "honest")] * 6)
        rep = self.round()
        record = copy.deepcopy(rep.record)
        record["version"] = 2
        c = Coordinator(copy.deepcopy(self.before), config=CoordinatorConfig(**record["rules"]), replay=record)
        with self.assertRaisesRegex(ProtocolError, "recorded as version 4"):
            c.open_round()


CHATML = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
WORDS = "hello there how are you I am fine thanks what is the weather like today it is sunny and warm good".split()


def word_tokenizer(vocab: int) -> bytes:
    """tokenizer.json of a word-level tokenizer small enough for a tiny model's vocabulary, with ChatML's tokens."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator([" ".join(WORDS + ["user", "assistant", "system"])] * 3,
                            trainers.WordLevelTrainer(vocab_size=vocab, special_tokens=["[UNK]", "<|im_start|>", "<|im_end|>"]))
    return tok.to_str().encode()


@unittest.skipUnless(torch is not None and HAVE_TOKENIZERS, "needs PyTorch, tokenizers and jinja2")
class LiveNetworkTests(unittest.TestCase):
    """The whole thing over HTTP, offline: a fake Hugging Face (tests/test_hub.py) serving a tiny model and its
    tokenizer, a node and three workers, `vibehash train chats.jsonl --base`, rounds, and `vibehash job download`."""

    def test_post_train_and_download_an_adapter(self):
        import contextlib
        import io
        import logging
        from unittest import mock

        from tests.test_hub import FakeHub
        from tests.test_network import run_until, start_network
        from vibehash import cli
        from vibehash import wallet as wallets

        d = tmpdir(self)
        env = mock.patch.dict(os.environ, {"VIBEHASH_HOME": str(d / "home"), "HF_HOME": str(d / "hf")})
        env.start()
        self.addCleanup(env.stop)
        for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "VIBEHASH_NODE", "VIBEHASH_WALLET"):
            os.environ.pop(k, None)
        fake = FakeHub()
        self.addCleanup(fake.close)
        os.environ["HF_ENDPOINT"] = fake.endpoint
        logging.disable(logging.INFO)
        self.addCleanup(logging.disable, logging.NOTSET)
        ft.unload()
        self.addCleanup(ft.unload)

        tiny = TinyModel(d / "src")  # its own cache, under another home: the workers fetch from the fake Hub
        files = {p.name: p.read_bytes() for p in tiny.folder.iterdir()}
        files["tokenizer.json"] = word_tokenizer(tiny.vocab)
        files["tokenizer_config.json"] = json.dumps({"chat_template": CHATML, "eos_token": "<|im_end|>"}).encode()
        fake.add_commit("tiny/qwen3", files)
        rng = np.random.default_rng(0)
        chats = d / "chats.jsonl"
        chats.write_text("\n".join(json.dumps({"messages": [
            {"role": "user", "content": " ".join(rng.choice(WORDS, 4))},
            {"role": "assistant", "content": "I am fine " + " ".join(rng.choice(WORDS, 3))}]}) for _ in range(60)))

        from vibehash.node import PhaseTimes
        # Seven workers in one process take turns on a GPU (finetune._Runtime.busy): longer phases than the MLP tests'.
        # A dispute's move gets as long as the verify phase, and a defender may still be checking when its dispute opens.
        node, api, req, workers, clients, stop, stop_node, genesis, tmp = start_network(
            self, 7, times=PhaseTimes(register_s=1.0, compute_s=10.0, reveal_s=3.0, verify_s=15.0, idle_s=0.3))
        clients[-1].worker.verifier_behavior = "false_accuser"  # its rejections become disputes it loses
        wallets.save(req, d / "requester.json", None)

        def run(*argv) -> str:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                try:
                    cli.main(list(argv))
                except SystemExit as e:
                    if e.code not in (None, 0):
                        self.fail(f"vibehash {' '.join(argv)} exited with {e.code!r}\n{out.getvalue()}")
            return out.getvalue()
        said = run("train", str(chats), "--base", "tiny/qwen3", "--node", node.url, "--wallet", str(d / "requester.json"),
                   "--yes", "--plain", "--budget", "100", "--fee", "10", "--units", "6", "--id", "chats-1")
        self.assertIn("Model     tiny/qwen3", said)
        self.assertIn("60 examples (60 chats", said)
        self.assertIn("Posted job chats-1", said)

        run_until(node, clients, stop, 5, seconds=600)  # the job's block and four rounds (the first while workers join)
        info = api.get("/job/chats-1")
        spec = TrainingJob.from_spec_json(json.dumps(info["spec"]), "", 0).spec
        self.assertEqual(spec.repo, "tiny/qwen3")
        self.assertGreaterEqual(len(info["rounds"]), 2)
        self.assertTrue(all(r["accepted_units"] > 0 and np.isfinite(r["val_loss"]) for r in info["rounds"]))
        self.assertLess(node.chain.state.escrow["chats-1"], 100 * VIBE)
        self.assertTrue(any(node.chain.state.balance(w.address) for w in workers), "workers were paid")
        slashes = [s for b in node.chain.blocks if b.summary for s in b.summary.slashes]
        games = [(e["outcome"], e["detail"], len(e["moves"])) for b in node.chain.blocks if b.summary
                 for e in api.get(f"/round/{b.height}/record").get("games", [])]
        self.assertTrue(any(o == "withdrawn" for o, _, _ in games), games)  # the false accuser withdraws its rejections
        self.assertFalse(slashes, ("nobody is slashed: the accuser withdrew", games))
        self.assertTrue(all(r["version"] == 4 for r in (api.get(f"/round/{b.height}/record") for b in node.chain.blocks
                                                        if b.summary)))

        out = d / "adapter"
        said = run("job", "download", "chats-1", "--node", node.url, "--out", str(out))
        self.assertIn("a LoRA adapter for tiny/qwen3", said)
        from vibehash import hub
        raw = (out / "adapter_model.safetensors").read_bytes()
        header = hub.parse_header(raw[8:8 + int.from_bytes(raw[:8], "little")], len(raw))
        self.assertEqual(len(header.tensors), 2 * len(spec.modules()))
        self.assertIn("base_model.model.model.layers.1.self_attn.o_proj.lora_B.weight", header.tensors)
        self.assertEqual(json.loads((out / "adapter_config.json").read_text())["revision"], spec.commit)


@unittest.skipUnless(REAL and torch is not None, "set VIBEHASH_REAL_MODELS=1 to fetch and check Qwen3-0.6B")
class RealModelTests(unittest.TestCase):
    """Qwen3-0.6B, fetched from Hugging Face through hub.py and checked against its manifest."""
    REPO, COMMIT = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"

    @classmethod
    def setUpClass(cls):
        from vibehash import hub
        cls.mbytes = ft.model_manifest(cls.REPO, cls.COMMIT, hub.Hub())
        cls.manifest = hub.Manifest.from_bytes(cls.mbytes)
        cls.config = hub.fetch_file(cls.manifest, "config.json")
        cls.spec = ft.LoraSpec(base=f"hf:{cls.REPO}@{cls.COMMIT}", manifest=hashlib.sha256(cls.mbytes).hexdigest(),
                               dims=ft.arch_dims(json.loads(cls.config)), seq_len=256)
        rng = np.random.default_rng(0)

        def make(n):
            lens = rng.integers(64, 257, n).astype(np.int32)
            ids = np.zeros((n, 256), np.int32)
            mask = np.zeros((n, 256), np.uint8)
            for i, k in enumerate(lens):
                ids[i, :k] = rng.integers(0, 150000, k)
                mask[i, k // 2:k] = 1
            return ids, lens, mask
        cls.data = ft.TokenDataset(*make(32), *make(8), manifest=cls.mbytes, config=cls.config)
        cls.data.check(cls.spec)

    def test_matches_transformers(self):
        from transformers import AutoModelForCausalLM
        from vibehash import hub

        rt = ft.runtime(self.spec, self.data, "cuda:0" if torch.cuda.is_available() else "cpu")
        paths = hub.fetch_tensors(self.manifest, [n for n in self.manifest.tensors])
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(folder, ignore_errors=True))
        (folder / "config.json").write_bytes(self.config)
        from tests.finetune_fixtures import write_checkpoint
        write_checkpoint(folder, json.loads(self.config), {n: (e.dtype, list(e.shape), paths[n].read_bytes())
                                                           for n, e in self.manifest.tensors.items()})
        ref = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32, attn_implementation="eager").to(rt.device)
        ids = torch.from_numpy(self.data.batch(np.arange(2))[0])
        with torch.no_grad():
            got, want = rt.model.logits(ids), ref(ids.to(rt.device)).logits
        err = float((got - want).pow(2).mean().sqrt() / want.pow(2).mean().sqrt())
        print(f"\nQwen3-0.6B vs transformers: {err:.2e} rms relative")
        self.assertLess(err, 1e-5)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "needs CUDA")
    def test_cpu_and_cuda_replays_agree_within_tolerance(self):
        """A segment trained on CUDA, replayed on the CPU, passes; the gap is far inside the job's tolerance and the
        1% rule, and a rescaled (canary) segment fails."""
        from vibehash import compute
        from vibehash.crypto import Draw

        gpu, cpu = compute.TorchBackend("cuda:0"), compute.TorchBackend("cpu")
        s, d, lr = self.spec, self.data, 5e-4
        start = s.round(ft.train_segment(s, d, s.init_params(), 1, 0, 4, lr, 4, gpu))
        t0 = time.time()
        claim = s.round(ft.train_segment(s, d, start, 1, 4, 4, lr, 4, gpu))
        again = s.round(ft.train_segment(s, d, start, 1, 4, 4, lr, 4, gpu))
        np.testing.assert_array_equal(claim, again)
        replay = s.round(ft.train_segment(s, d, start, 1, 4, 4, lr, 4, cpu))
        print(f"\nCPU replay of 4 steps: {time.time() - t0:.0f} s")
        for name, part in zip("Bmv", s.parts):
            gap = np.linalg.norm(claim[part] - replay[part])
            print(f"  {name}: replay differs by {gap / np.linalg.norm(replay[part]):.1e} of its size, "
                  f"{gap / np.linalg.norm(replay[part] - start[part]):.1e} of the segment's change")
        self.assertTrue(reproduces(claim, replay, start, ft.TOLERANCE, s))
        self.assertFalse(reproduces(rescale_update(s, start, claim, Draw(b"x")), replay, start, ft.TOLERANCE, s))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "needs CUDA")
    def test_a_whole_unit_on_cuda_replays_on_the_cpu(self):
        """What comparing replicas assumes: a whole 16-step unit trained on CUDA, replayed on the CPU, passes
        the whole-unit rule (finetune.WHOLE_TOLERANCE, and 1% of the unit's own change). Measured 2026-11-01: 5.0e-5
        of B's size and 7.1e-5 of its change, so a segment's 1e-4 of the size would leave a margin of only two."""
        from vibehash import compute

        gpu, cpu = compute.TorchBackend("cuda:0"), compute.TorchBackend("cpu")
        s, d, lr = self.spec, self.data, 5e-4
        start = s.fresh(s.round(ft.train_segment(s, d, s.init_params(), 2, 0, 8, lr, 4, gpu)))  # a round's start
        claim = s.round(ft.train_segment(s, d, start, 3, 0, 16, lr, 4, gpu))
        t0 = time.time()
        replay = s.round(ft.train_segment(s, d, start, 3, 0, 16, lr, 4, cpu))
        print(f"\nCPU replay of a 16-step unit: {time.time() - t0:.0f} s")
        for name, part in zip("Bmv", s.parts):
            gap = np.linalg.norm(claim[part] - replay[part])
            print(f"  {name}: replay differs by {gap / np.linalg.norm(replay[part]):.1e} of its size, "
                  f"{gap / np.linalg.norm(replay[part] - start[part]):.1e} of the unit's change")
        self.assertTrue(reproduces(claim, replay, start, ft.WHOLE_TOLERANCE, s))


if __name__ == "__main__":
    unittest.main()

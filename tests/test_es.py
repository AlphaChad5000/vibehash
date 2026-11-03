"""Evolution strategies on the chain (vibehash/es.py): deterministic noise, the update rule against a direct numpy
computation, rebuilding the state from history bit for bit, Countdown's problems and rewards, the perturbed model on a
tiny random Qwen2 (CPU), and a local network whose state a fresh rebuild from its blocks and records arrives at, with a
cheating worker caught. CPU only; the model tests need PyTorch and transformers and are skipped without them.
"""
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from vibehash import es
from vibehash.crypto import KeyPair
from vibehash.ledger import Block, Chain, InvalidBlock, Tx, _apply_user_tx
from vibehash.tasks import SYN_PER_VIBE, InvalidJob, TrainingJob

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

VIBE = SYN_PER_VIBE
ROOT = Path(__file__).resolve().parents[1]
# noise(0, 1000)'s bytes: if numpy's Philox stream or this module's arithmetic ever changed, this would, and so would
# every ES chain's state.
NOISE_0_1000 = "adab75b82d1e64a2b1097b45991227aca494f15d495f5107ce2895d2c7c5d1b3"
PHILOX_0 = [213000021201967259, 4455796210202625458, 2055444239878205049, 10411612076246414556]


def spec_for(n: int = 5000, **kw) -> es.EsSpec:
    """A spec for the update rule's tests: n perturbed numbers of no model in particular."""
    base = dict(base="hf:vibehash/es-tiny@" + "0" * 40, manifest="1" * 64,
                dims={"family": "qwen2", "vocab": 260, "hidden": 32, "layers": 2, "heads": 4, "kv_heads": 2,
                      "head_dim": 8, "ffn": 64, "tied": True},
                tensors=("model.norm.weight",), n=n, tensors_sha="2" * 64, sigma=0.01, alpha=0.001, pairs=4,
                pairs_per_unit=2, problems=4, max_tokens=8)
    return es.EsSpec(**(base | kw))


class NoiseTests(unittest.TestCase):
    def test_known_answer(self):
        """The raw stream and the normals made from it are pinned: numpy promises bit generators' streams stay put."""
        self.assertEqual(np.random.Philox(key=0).random_raw(4).tolist(), PHILOX_0)
        self.assertEqual(hashlib.sha256(es.noise(0, 1000).tobytes()).hexdigest(), NOISE_0_1000)

    def test_identical_in_another_process_and_repeatable(self):
        code = ("import hashlib; from vibehash import es; "
                "print(hashlib.sha256(es.noise(123456789, 300001, 77).tobytes()).hexdigest())")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        here = hashlib.sha256(es.noise(123456789, 300001, 77).tobytes()).hexdigest()
        self.assertEqual(out.stdout.strip(), here)
        self.assertEqual(hashlib.sha256(es.noise(123456789, 300001, 77).tobytes()).hexdigest(), here)

    def test_any_stretch_is_that_stretch_of_the_whole(self):
        whole = es.noise(42, 3 * es._CHUNK + 5)
        for start, n in ((0, 1), (1, 7), (3, 1000), (es._CHUNK - 3, 10), (2 * es._CHUNK + 1, es._CHUNK)):
            with self.subTest(start=start, n=n):
                np.testing.assert_array_equal(es.noise(42, n, start), whole[start:start + n])
        with mock.patch.object(es, "_CHUNK", 64):  # made in other-sized pieces: the same bits
            np.testing.assert_array_equal(es.noise(42, 1000), whole[:1000])

    def test_standard_normal_and_close_to_libm_box_muller(self):
        z = es.noise(7, 400_000)
        self.assertEqual(z.dtype, np.float32)
        self.assertLess(abs(float(z.mean())), 0.01)
        self.assertLess(abs(float(z.std()) - 1), 0.01)
        raw = np.random.Philox(key=7).random_raw(1000)
        u1 = ((raw[0::2] >> np.uint64(11)).astype(np.float64) + 1) * 2.0 ** -53
        u2 = (raw[1::2] >> np.uint64(11)).astype(np.float64) * 2.0 ** -53
        r = np.sqrt(-2 * np.log(u1))
        ref = np.empty(1000)
        ref[0::2], ref[1::2] = r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)
        np.testing.assert_allclose(z[:1000], ref, rtol=0, atol=2e-6)

    def test_own_ln_sin_cos_are_accurate(self):
        u = np.random.default_rng(0).random(100_000)
        s, c = es._sincos(u)
        np.testing.assert_allclose(s, np.sin(2 * np.pi * u), rtol=0, atol=2e-15)
        np.testing.assert_allclose(c, np.cos(2 * np.pi * u), rtol=0, atol=2e-15)
        x = np.concatenate([u + 1e-300, [2.0 ** -53, 1.0, 0.5, 1e-300]])
        np.testing.assert_allclose(es._ln(x), np.log(x), rtol=1e-15, atol=1e-15)
        self.assertEqual(es._ln(np.array([1.0]))[0], 0.0)

    def test_seeds_differ(self):
        self.assertNotEqual(es.pair_seed(1, 0), es.pair_seed(1, 1))
        self.assertNotEqual(es.pair_seed(1, 0), es.pair_seed(2, 0))
        self.assertFalse(np.array_equal(es.noise(es.pair_seed(1, 0), 10), es.noise(es.pair_seed(1, 1), 10)))


def direct(spec: es.EsSpec, state: np.ndarray, seed: int, update: list) -> np.ndarray:
    """The update written the obvious way with numpy (its own sums and std), for comparison with es.step."""
    rewards = np.array([[rp, rm] for _, rp, rm in update], dtype=np.float64).reshape(-1)
    z = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    eps = np.stack([s * es.noise(es.pair_seed(seed, j), spec.n).astype(np.float64)
                    for j, _, _ in update for s in (1, -1)])
    g = (z[:, None] * eps).sum(0) / (len(rewards) * spec.sigma)
    n, out = spec.n, np.asarray(state, np.float64).copy()
    t = int(state[-es.CTRL + 2]) + 1
    b1, b2 = spec.betas
    if spec.optimizer == "sgd":
        out[:n] += spec.alpha * g
    elif spec.optimizer == "momentum":
        out[n:2 * n] = b1 * out[n:2 * n] + g
        out[:n] += spec.alpha * out[n:2 * n]
    else:
        out[n:2 * n] = b1 * out[n:2 * n] + (1 - b1) * g
        out[2 * n:3 * n] = b2 * out[2 * n:3 * n] + (1 - b2) * g * g
        out[:n] += spec.alpha * (out[n:2 * n] / (1 - b1 ** t)) / (np.sqrt(out[2 * n:3 * n] / (1 - b2 ** t)) + spec.eps)
    return out


class UpdateTests(unittest.TestCase):
    update = [[0, 0.75, 0.25], [1, 0.5, 0.5], [3, 0.0, 1.0]]

    def test_each_optimizer_matches_a_direct_numpy_computation(self):
        for opt in es.OPTIMIZERS:
            with self.subTest(optimizer=opt):
                spec = spec_for(optimizer=opt)
                state = spec.init_params()
                rng = np.random.default_rng(1)
                state[:spec.n * (1 + spec.moments)] = rng.normal(0, 0.01, spec.n * (1 + spec.moments))
                if opt == "adam":
                    state[2 * spec.n:3 * spec.n] = np.abs(state[2 * spec.n:3 * spec.n]) * 1e-3
                    state[-es.CTRL + 2] = 3  # a few steps in, so the bias corrections aren't the first step's
                got = es.step(spec, state, 99, self.update)
                want = direct(spec, state, 99, self.update)
                self.assertEqual(got.dtype, np.float32)
                np.testing.assert_allclose(got[:-es.CTRL], want[:-es.CTRL], rtol=1e-5, atol=1e-9)
                self.assertFalse(np.array_equal(got[:spec.n], state[:spec.n]))

    def test_the_pinned_order_gives_the_same_bits_every_time_and_in_any_chunking(self):
        spec = spec_for(n=3000, optimizer="adam")
        a = es.step(spec, spec.init_params(), 5, self.update)
        b = es.step(spec, spec.init_params(), 5, self.update)
        with mock.patch.object(es, "_CHUNK", 256):
            c = es.step(spec, spec.init_params(), 5, self.update)
        es.cache_noise(16)  # whole streams, kept: the same bits
        self.addCleanup(es.cache_noise, 0)
        d = es.step(spec, spec.init_params(), 5, self.update)
        e = es.step(spec, spec.init_params(), 5, self.update)
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(a.tobytes(), c.tobytes())
        self.assertEqual(a.tobytes(), d.tobytes())
        self.assertEqual(a.tobytes(), e.tobytes())
        self.assertTrue(es._cache)

    def test_sgd_is_alpha_over_n_sigma_times_the_shaped_rewards_noise(self):
        """θ += alpha/(N·σ)·Σ r̂_i·ε_i, exactly as the docstring's order works it out."""
        spec = spec_for(n=100)
        z = es.shaped([0.75, 0.25, 0.5, 0.5, 0.0, 1.0])
        g = np.zeros(100)
        for k, (j, _, _) in enumerate(self.update):
            c = z[2 * k] - z[2 * k + 1]
            if c != 0.0:
                g += es.noise(es.pair_seed(9, j), 100).astype(np.float64) * c
        want = (np.zeros(100) + (g * (1.0 / (6 * spec.sigma))) * spec.alpha).astype(np.float32)
        self.assertEqual(es.step(spec, spec.init_params(), 9, self.update)[:100].tobytes(), want.tobytes())
        np.testing.assert_allclose(es.shaped([0.75, 0.25, 0.5, 0.5, 0.0, 1.0]),
                                   (np.array([0.75, 0.25, 0.5, 0.5, 0, 1]) - 0.5) / (np.std([0.75, 0.25, 0.5, 0.5, 0, 1]) + 1e-8))

    def test_nothing_accepted_or_no_differences_changes_nothing_learned(self):
        spec = spec_for()
        s0 = spec.init_params()
        self.assertEqual(es.step(spec, s0, 1, []).tobytes(), s0.tobytes())
        same = es.step(spec, s0, 1, [[0, 0.5, 0.5], [1, 0.5, 0.5]])
        np.testing.assert_array_equal(same[:spec.n], s0[:spec.n])
        self.assertEqual(same[-es.CTRL + 2], 1)  # a step was taken all the same
        with self.assertRaises(ValueError):
            es.step(spec, s0, 1, [[1, 0, 1], [0, 1, 0]])  # out of order

    def test_retarget_moves_the_level_every_k_steps(self):
        spec = spec_for(retarget=(2, 0.5, 0.1), level=3)
        s = spec.init_params()
        self.assertEqual(es.level_of(s), 3)
        s = es.step(spec, s, 1, [[0, 1.0, 0.8]])
        self.assertEqual(es.level_of(s), 3)  # one step: not yet
        s = es.step(spec, s, 2, [[0, 0.9, 1.0]])
        self.assertEqual(es.level_of(s), 4)  # mean 0.925 over two steps: harder
        for k in range(4):
            s = es.step(spec, s, 3 + k, [[0, 0.0, 0.1]])
        self.assertEqual(es.level_of(s), 2)  # two windows of 0.05: easier, twice
        s = es.step(spec, s, 9, [[0, 0.5, 0.5]])
        s = es.step(spec, s, 10, [[0, 0.4, 0.6]])
        self.assertEqual(es.level_of(s), 2)  # within the band: stays
        off = spec_for(level=3)
        self.assertEqual(es.level_of(es.step(off, off.init_params(), 1, [[0, 1.0, 1.0]])), 3)  # the default: never

    def test_rebuild_from_history_is_live_training_bit_for_bit(self):
        """A 'live' run (rewards from a stand-in for answering, worked out each round from the state itself) and
        rebuild() from the history it left reach the same state, and the same hash, in another process too."""
        spec = spec_for(n=20_000, optimizer="adam", retarget=(2, 0.5, 0.1), level=2)
        state, history = spec.init_params(), []
        for rnd in range(6):
            seed = es.derive(b"test round", rnd)
            delta = es.delta_of(spec, state)
            update = []
            for j in range(spec.pairs):
                e = es.noise(es.pair_seed(seed, j), spec.n)
                score = lambda d: float(1 / (1 + np.exp(-200 * (d[:50].astype(np.float64).sum() + 0.01 * j))))  # noqa: E731
                update.append([j, score(delta + spec.sigma * e), score(delta - spec.sigma * e)])
            state = es.step(spec, state, seed, update)
            history.append({"seed": f"{seed:016x}", "update": update})
        rebuilt = es.rebuild(spec, json.loads(json.dumps(history)))  # through JSON, as records hold it
        self.assertEqual(rebuilt.tobytes(), state.tobytes())
        self.assertEqual(es.state_hash(rebuilt), es.state_hash(state))
        self.assertNotEqual(es.state_hash(state), es.state_hash(spec.init_params()))
        code = ("import json, sys; from vibehash import es; d = json.load(sys.stdin); "
                "s = es.EsSpec.from_json(d['spec']); print(es.state_hash(es.rebuild(s, d['history'])))")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
                             input=json.dumps({"spec": spec.to_json(), "history": history}))
        self.assertEqual(out.stdout.strip(), es.state_hash(state))

    def test_spec_round_trips_and_is_refused_when_wrong(self):
        spec = spec_for(optimizer="adam", retarget=(5, 0.5, 0.1), reward="partial")
        self.assertEqual(es.EsSpec.from_json(json.loads(json.dumps(spec.to_json()))), spec)
        self.assertEqual(spec.problems_list(), [])
        for bad in (dict(sigma=0), dict(pairs=3), dict(level=99), dict(optimizer="lion"), dict(reward="vibes"),
                    dict(tensors=()), dict(n=0), dict(check=9), dict(base="Qwen/Qwen2.5"), dict(retarget=(1, 2.0, 0.1))):
            with self.subTest(bad=bad):
                self.assertTrue(spec_for(**bad).problems_list())
        self.assertEqual(spec.n_params, 3 * spec.n + es.CTRL)
        self.assertEqual(spec_for().n_params, spec_for().n + es.CTRL)


class CountdownTests(unittest.TestCase):
    def test_every_problem_is_solvable_and_drawn_from_its_seed(self):
        for level, lv in enumerate(es.LEVELS):
            with self.subTest(level=level):
                ps = es.problems(11, 60, level)
                for p in ps:
                    self.assertEqual(len(p.numbers), lv.count)
                    self.assertTrue(all(1 <= n <= lv.top for n in p.numbers))
                    self.assertTrue(lv.lo <= p.target <= lv.hi)
                    self.assertEqual(es.reward(p, f"<answer>{p.solution}</answer>"), 1.0, p)
                    self.assertTrue(set(ch for ch in p.solution if not ch.isdigit()) <= set(lv.ops + "() "))
                self.assertEqual(es.problems(11, 60, level), ps)
                self.assertNotEqual(es.problems(12, 60, level), ps)
        self.assertEqual(len(es.heldout()), 200)
        self.assertEqual(es.heldout(), es.heldout())

    def test_reward_edge_cases(self):
        p = es.Problem((3, 5, 7), 22, "3 * 5 + 7")
        cases = [  # text, binary, partial
            ("<answer>3*5+7</answer>", 1.0, 1.0),
            ("<answer> (3 * 5) + 7 = 22 </answer>", 1.0, 1.0),
            ("<answer>7 + 15</answer>", 0.0, 0.1),  # 15 isn't given
            ("<answer>3*5*7</answer>", 0.0, 0.2),  # 105: given numbers, far off
            ("<answer>3*(5+7)</answer>", 0.0, float(es.Fraction(1, 5) + es.Fraction(3, 5) * (1 - es.Fraction(14, 22)))),
            ("<answer>5*5-3</answer>", 0.0, 0.1),  # 5 twice
            ("<answer>3/0+22</answer>", 0.0, 0.0),
            ("<answer>-3+25</answer>", 0.0, 0.0),  # no unary minus
            ("3*5+7", 0.0, 0.0),  # no tags
            ("<answer>3*5+7</answer> no, <answer>3+5</answer>", 0.0, 0.2 + 0.6 * (1 - 14 / 22)),  # the last counts
            ("<answer>3*5+7 = 23</answer>", 1.0, 1.0),  # what follows "=" is ignored
            ("<answer>3x5+7</answer>", 0.0, 0.0),
            ("<answer>((3*5)+7</answer>", 0.0, 0.0),
            ("<answer>" + "(" * 40 + "3" + ")" * 40 + "</answer>", 0.0, 0.0),  # nested too deep
            ("<answer>" + "+".join(["3"] * 120) + "</answer>", 0.0, 0.0),  # too long
            ("<answer>7*3+5/5</answer>", 0.0, 0.1),
            ("<answer>(7 - 5) * 3 + 15</answer>", 0.0, 0.1),
        ]
        for text, binary, partial in cases:
            with self.subTest(text=text):
                self.assertEqual(es.reward(p, text), binary)
                self.assertAlmostEqual(es.reward(p, text, "partial"), partial, places=12)
        self.assertEqual(es.reward(es.Problem((6, 4), 1, ""), "<answer>6/4 - 1/2</answer>"), 0.0)  # 1 and 2 not given
        self.assertEqual(es.reward(es.Problem((6, 4, 2), 1, ""), "<answer>(6 - 4) / 2</answer>"), 1.0)
        self.assertEqual(es.parse("6 / 4")[0], es.Fraction(3, 2))  # exact, not 1.5 in floats

    def test_mean_reward_order(self):
        self.assertEqual(es.mean_reward([1.0, 0.0, 0.1, 0.2]), (((1.0 + 0.0) + 0.1) + 0.2) / 4)


class GateTests(unittest.TestCase):
    def test_only_a_chain_made_for_es_jobs_takes_one(self):
        req = KeyPair.generate(b"es requester")
        job = TrainingJob.from_spec_json(json.dumps(es_job_spec("gate-job", spec_for())), req.address, 10 * VIBE)
        self.assertTrue(job.es)
        self.assertEqual(TrainingJob.from_spec_json(job.spec_json(), req.address, 10 * VIBE).spec, job.spec)
        for gate in (False, True):
            chain = Chain({req.address: 100 * VIBE}, es=gate)
            tx = Tx.signed(req, "post_job", "system", 10 * VIBE, 0, job.spec_json(), chain.chain_id)
            with self.subTest(gate=gate):
                state = chain.state
                if gate:
                    import copy
                    s = copy.deepcopy(state)
                    _apply_user_tx(s, tx, 1, chain.chain_id, 3)
                    self.assertIn("gate-job", s.jobs)
                else:
                    with self.assertRaisesRegex(InvalidBlock, "prototype"):
                        _apply_user_tx(state, tx, 1, chain.chain_id, 3)
        self.assertNotEqual(Chain({req.address: 1}, es=True).genesis_hash, Chain({req.address: 1}).genesis_hash)
        self.assertEqual(Chain.from_genesis({"chain_id": "x", "allocations": {req.address: 1}}).genesis_hash,
                         Chain({req.address: 1}, chain_id="x").genesis_hash)  # genesis files without it: unchanged

    def test_a_node_before_es_jobs_refuses_the_spec(self):
        """Its spec has no model n_in and no lora: an MLP's parser (what nodes before ES jobs run) refuses it."""
        d = es_job_spec("old-node", spec_for())
        self.assertNotIn("n_in", d["model"])
        with mock.patch("vibehash.tasks.is_es_spec", lambda d: False), self.assertRaisesRegex(InvalidJob, "malformed"):
            TrainingJob.from_spec_json(json.dumps(d), "x", 0)
        with self.assertRaisesRegex(InvalidJob, "no field"):
            TrainingJob.from_spec_json(json.dumps(d | {"lr": 0.1}), "x", 0)


def es_job_spec(job_id: str, spec: es.EsSpec, memory_mb: int = 1024) -> dict:
    return {"id": job_id, "fee_per_round": 1 * VIBE, **spec.to_json(), "init_seed": 0, "units_per_round": spec.units,
            "memory_mb": memory_mb}


# --- a tiny random Qwen2 on the CPU ----------------------------------------------------------------------------------

TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
PATTERNS = es.preset("norms+biases") + ("model.layers.1.mlp.*_proj.weight",)


def tiny_model(folder: Path, seed: int = 0) -> Path:
    """A random two-layer Qwen2 with a byte-level tokenizer and a chat template, saved as Hugging Face saves models."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens(specials)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = TEMPLATE
    folder.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(folder)
    ids = {s: tok.convert_tokens_to_ids(s) for s in specials}
    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=len(vocab) + len(specials), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                         tie_word_embeddings=True, bos_token_id=ids["<|endoftext|>"], eos_token_id=ids["<|im_end|>"])
    model = Qwen2ForCausalLM(config)
    with torch.no_grad():  # norms away from 1 and biases away from 0, so nudging them matters
        for name, p in model.named_parameters():
            if name.endswith("bias") or "norm" in name:
                p.add_(torch.randn_like(p) * 0.3)
    model.save_pretrained(folder)
    (folder / "generation_config.json").write_text(json.dumps(
        {"eos_token_id": [ids["<|im_end|>"], ids["<|endoftext|>"]], "pad_token_id": ids["<|endoftext|>"]}))
    return folder


def tiny_spec(folder: Path, **kw) -> es.EsSpec:
    from transformers import AutoConfig, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(folder)
    n, sha = es.tensors_digest(es.select(((k, p.shape) for k, p in model.named_parameters()), PATTERNS))
    manifest = hashlib.sha256(b"es tiny " + str(folder).encode()).hexdigest()
    es.register(manifest, folder)
    c = AutoConfig.from_pretrained(folder)
    base = dict(base="hf:vibehash/es-tiny@" + "0" * 40, manifest=manifest,
                dims={"family": "qwen2", "vocab": c.vocab_size, "hidden": c.hidden_size, "layers": c.num_hidden_layers,
                      "heads": c.num_attention_heads, "kv_heads": c.num_key_value_heads, "head_dim": 8,
                      "ffn": c.intermediate_size, "tied": True},
                tensors=PATTERNS, n=n, tensors_sha=sha, sigma=0.3, alpha=0.05, pairs=4, pairs_per_unit=1, problems=3,
                max_tokens=6, level=2, check=2, margin=1e-3)
    return es.EsSpec(**(base | kw))


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.folder = tiny_model(Path(cls.tmp.name) / "tiny")
        cls.spec = tiny_spec(cls.folder)
        cls.rt = es.runtime(cls.spec, "cpu")

    @classmethod
    def tearDownClass(cls):
        es.unload()
        cls.tmp.cleanup()

    def test_the_tensors_picked_are_the_specs(self):
        self.assertEqual([t[0] for t in self.rt.targets], sorted(t[0] for t in self.rt.targets))
        self.assertEqual(sum(math.prod(t[1]) for t in self.rt.targets), self.spec.n)
        with self.assertRaises(ValueError):
            es.runtime(es.EsSpec(**{**self.spec.__dict__, "tensors_sha": "3" * 64}), "cpu")
        with self.assertRaisesRegex(ValueError, "embedding"):
            es.select([("model.embed_tokens.weight", (4, 4))], ("model.*",))

    def test_a_group_runs_each_member_as_if_alone(self):
        import torch

        rt, spec = self.rt, self.spec
        delta = np.random.default_rng(0).normal(0, 0.05, spec.n).astype(np.float32)
        ps = es.problems(5, 3, 2)
        rows = es.prompt_ids(rt, ps)
        members = [(es.pair_seed(1, 0), 1), (es.pair_seed(1, 0), -1), (es.pair_seed(1, 1), 1), (0, 0)]
        with es.Group(rt, spec, delta, members):
            together = es.generate(rt, rows * len(members), 6)
        alone = []
        for m in members:
            with es.Group(rt, spec, delta, [m]):
                alone += es.generate(rt, rows, 6)
        self.assertEqual(together, alone)
        self.assertGreater(len({tuple(a) for a in together}), 3, "nudges this big should change some answers")
        # The unperturbed member is the model with delta added, the grouped forward agreeing with the plain one.
        ids, mask = es._left_pad(rows, rt.pad, "cpu")
        with es.Group(rt, spec, delta, [(0, 0)]):
            grouped = rt.model(input_ids=ids, attention_mask=mask).logits
        saved = {name: base.clone() for name, _, _, _, base in rt.targets}
        params, at = dict(rt.model.named_parameters()), 0
        with torch.no_grad():
            for name, shape, *_ in rt.targets:
                size = math.prod(shape)
                params[name].copy_((saved[name].double() + torch.from_numpy(delta[at:at + size]).double().view(shape)).float())
                at += size
            plain = rt.model(input_ids=ids, attention_mask=mask).logits
            for name in saved:
                params[name].copy_(saved[name])
        torch.testing.assert_close(grouped, plain, rtol=1e-5, atol=1e-5)

    def test_an_honest_claim_passes_and_forged_answers_dont(self):
        rt, spec = self.rt, self.spec
        delta = np.zeros(spec.n, np.float32)
        seed = 77
        c = es.claim(rt, spec, delta, seed, 2, 1)
        raw = es.claim_bytes("u1", c["answers"])
        self.assertEqual(es.rescore(spec, seed, 2, raw, "u1"), c["rewards"])
        self.assertIsNone(es.rescore(spec, seed, 2, raw, "u2"))  # another unit's
        self.assertEqual(es.check(rt, spec, delta, seed, 2, 1, raw, [0, 1], "u1")[:2], (True, None))
        self.assertFalse(es.check(rt, spec, delta, seed, 2, 0, raw, [0], "u1")[0])  # as another unit's members
        forged = json.loads(raw)
        a = forged["answers"][1][0]
        a["ids"] = rt.tokenizer.encode("<answer>1+1</answer>", add_special_tokens=False)[:spec.max_tokens]
        a["text"] = es.text_of(rt, a["ids"])
        ok, bad, why = es.check(rt, spec, delta, seed, 2, 1, es.claim_bytes("u1", forged["answers"]), [0, 1], "u1")
        self.assertEqual((ok, bad), (False, 1), why)
        lied = json.loads(raw)
        lied["answers"][0][0]["text"] = "<answer>2</answer>"
        self.assertFalse(es.check(rt, spec, delta, seed, 2, 1, es.claim_bytes("u1", lied["answers"]), [0], "u1")[0])

    def test_evaluate_and_heldout(self):
        self.assertEqual(es.evaluate(self.rt, self.spec, np.zeros(self.spec.n, np.float32), es.heldout(4, 2)), 0.0)


# --- a local network ------------------------------------------------------------------------------------------------

def fake_reward(p, text, kind="binary"):
    """A stand-in score a random tiny model's babble can earn: deterministic, from the text alone, in [0, 1]."""
    return (sum(text.encode()) % 5) / 4


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")
class NetworkTests(unittest.TestCase):
    def test_a_fresh_rebuild_from_the_chain_reaches_the_nodes_state_and_cheats_are_caught(self):
        from vibehash.client import NodeClient, WorkerClient
        from vibehash.coordinator import CoordinatorConfig
        from vibehash.node import Node, PhaseTimes
        from vibehash.replay import replay_block
        from vibehash.storage import Store
        from tests.test_network import run_until

        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        spec = tiny_spec(tiny_model(root / "tiny", seed=1), optimizer="adam", alpha=0.02, retarget=(2, 0.5, 0.1))
        self.addCleanup(es.unload)
        node_keys, req = KeyPair.generate(b"es node"), KeyPair.generate(b"es requester")
        workers = [KeyPair.generate(f"es worker {i}".encode()) for i in range(4)]
        genesis = {"chain_id": "vibehash-es-test", "timestamp": 0, "allocations": {req.address: 500 * VIBE},
                   "stakes": {k.address: 10 * VIBE for k in [node_keys, *workers]}, "es": True}
        patch = mock.patch.object(es, "reward", fake_reward)
        patch.start()
        self.addCleanup(patch.stop)
        node = Node(root / "node", genesis, node_keys, port=0,
                    times=PhaseTimes(register_s=1.0, compute_s=20.0, reveal_s=3.0, verify_s=15.0, idle_s=0.3),
                    config=CoordinatorConfig(newcomer_units=4)).start()
        stopped = []
        self.addCleanup(lambda: stopped or node.stop())
        api = NodeClient(node.url)
        clients = [WorkerClient(node.url, k, memory_mb=8192, flops_override=1e12, poll_s=0.1, device="cpu",
                                cache_dir=root / f"cache{i}") for i, k in enumerate(workers)]
        clients[2].es.cheat = "liar"  # claims rewards its answers didn't earn: the node's own re-score catches it
        clients[3].es.cheat = "forger"  # the right answers, which its model didn't give: checkers' teacher forcing does
        job = TrainingJob.from_spec_json(json.dumps(es_job_spec("es-job", spec)), req.address, 100 * VIBE)
        api.send_tx(req, "post_job", "system", 100 * VIBE, job.spec_json())
        stop = threading.Event()
        self.addCleanup(stop.set)
        run_until(node, clients, stop, 5, seconds=300)  # the job's block and four rounds
        self.assertGreaterEqual(node.chain.tip.height, 5, "the network didn't make its rounds in time")

        blocks = [Block.from_dict(d) for d in api.get(f"/blocks?from=0&to={node.chain.tip.height}")]
        records = {b.height: api.get(f"/round/{b.height}/record") for b in blocks if b.summary is not None}
        es_rounds = [b for b in blocks if b.summary is not None and b.summary.job_id == "es-job"]
        self.assertGreaterEqual(len(es_rounds), 3)
        # A fresh node's view: the state from the genesis, the blocks and their records alone.
        state, rounds = es.rebuild_chain(blocks, records.__getitem__, "es-job")
        self.assertEqual(rounds, len(es_rounds))
        self.assertEqual(es.state_hash(state), es_rounds[-1].summary.params_hash)
        np.testing.assert_array_equal(state, node.coord.params["es-job"])
        self.assertNotEqual(es.state_hash(state), es.state_hash(spec.init_params()), "the model never moved")
        updates = [records[b.height]["es"]["update"] for b in es_rounds]
        self.assertTrue(any(updates))
        last_es = es_rounds[-1].height
        forged = {**records, last_es: records[last_es] | {"es": records[last_es]["es"] | {"update": []}}}
        with self.assertRaisesRegex(ValueError, "commits to"):  # a record that isn't the block's
            es.rebuild_chain(blocks, forged.__getitem__, "es-job")
        # An auditor replays every round from its record, working each ES update out again from the state before it.
        audit, state = Chain.from_genesis(api.get("/genesis")), spec.init_params()
        last = None
        for b in blocks[1:]:
            if b.summary is not None:
                rec = records[b.height]
                start = state if b.summary.job_id == "es-job" else None
                given = lambda j, h, s=start: s if s is not None and es.state_hash(s) == h.hex() else None  # noqa: E731
                self.assertEqual(replay_block(audit, b, rec, es_states=given), [], f"block {b.height}")
                if start is not None:
                    last = (audit, b, rec, given)
                    state = es.step(spec, state, int(rec["es"]["seed"], 16), rec["es"]["update"])
            audit.append(b)
        self.assertEqual(es.state_hash(state), es_rounds[-1].summary.params_hash)
        # It works the update out rather than reading the record's hash: a replayer whose arithmetic differed (here, one
        # nudged number) disagrees with the block.
        _, b, rec, given = last
        audit = Chain.from_genesis(api.get("/genesis"))
        for x in blocks[1:b.height]:
            audit.append(x)
        step = es.step
        with mock.patch.object(es, "step", lambda *a: step(*a) + np.float32(1e-3) * (np.arange(a[0].n_params) == 0)):
            problems = replay_block(audit, b, rec, es_states=given)
        self.assertTrue(any("params_hash" in p for p in problems), problems)

        st = node.chain.state
        liar, forger = workers[2].address, workers[3].address
        self.assertLess(st.stake(liar), 10 * VIBE, "the liar wasn't slashed")
        slashed = [s for b in es_rounds for s in b.summary.slashes if s[0] == liar]
        self.assertTrue(slashed)
        self.assertEqual(st.stake(forger), 10 * VIBE, "a rejection isn't refereed: nobody is slashed for it")
        self.assertTrue(all(forger not in b.summary.accepted for b in es_rounds), "forged answers were accepted")
        honest = [w.address for w in workers[:2]]
        self.assertTrue(any(a in b.summary.accepted for b in es_rounds for a in honest))
        for k in workers[:2]:
            self.assertGreater(st.balance(k.address), 0, "an honest worker wasn't paid")
        self.assertGreater(sum(c.es.rebuilt for c in clients), 0, "no worker rebuilt its state from the chain")
        # The node keeps the state on disk, and a restart reads it back.
        params = node.coord.params["es-job"].copy()
        node.stop()
        stopped.append(True)
        store = Store(root / "node")
        self.addCleanup(store.close)
        store.open_chain(genesis)
        np.testing.assert_array_equal(store.load_params("es-job"), params)


# --- Evo mode: the switch (es_local.py, cli.py) ----------------------------------------------------------------------

class EvoModeTests(unittest.TestCase):
    """`testnet --evo` and `train --evo`, as people type them, against real nodes on this computer."""

    def setUp(self):
        import contextlib
        import io
        import logging
        import os
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        patch = mock.patch.dict(os.environ, {"VIBEHASH_HOME": str(self.root / "home")})
        patch.start()
        self.addCleanup(patch.stop)
        for var in ("VIBEHASH_NODE", "VIBEHASH_WALLET"):
            os.environ.pop(var, None)
        logging.disable(logging.INFO)  # cli.main turns on INFO logging for the whole process
        self.addCleanup(logging.disable, logging.NOTSET)
        self.io, self.redirect = io, contextlib.redirect_stdout

    def run_cli(self, *argv) -> tuple[object, str]:
        from vibehash import cli
        out = self.io.StringIO()
        with self.redirect(out):
            try:
                cli.main(list(argv))
                code = 0
            except SystemExit as e:
                code = e.code or 0
        return code, out.getvalue()

    def node(self, es_gate: bool):
        from vibehash import wallet as wallets
        from vibehash.client import NodeClient
        from vibehash.node import Node, PhaseTimes
        keys, req = KeyPair.generate(b"evo node"), KeyPair.generate(b"evo requester")
        genesis = {"chain_id": "vibehash-evo-test", "timestamp": 0, "allocations": {req.address: 100 * VIBE},
                   "stakes": {keys.address: 10 * VIBE}, **({"es": True} if es_gate else {})}
        node = Node(self.root / f"node-{es_gate}", genesis, keys, port=0,
                    times=PhaseTimes(register_s=0.5, compute_s=0.5, reveal_s=0.5, verify_s=0.5, idle_s=0.2)).start()
        self.addCleanup(node.stop)
        wallets.save(req, self.root / "requester.json", None)
        return node, NodeClient(node.url), req

    def test_train_evo_on_a_chain_without_evo_jobs_spends_nothing(self):
        """Today's public network's genesis has no "es": the command stops at its first request, before a model is
        looked for or the wallet opened, with one line."""
        from vibehash import cli, es_local
        node, api, req = self.node(es_gate=False)
        with mock.patch.object(es_local, "find", side_effect=AssertionError("looked for a model")), \
                mock.patch.object(cli, "load_wallet", side_effect=AssertionError("opened the wallet")):
            code, out = self.run_cli("train", "--evo", "--node", node.url, "--wallet", str(self.root / "requester.json"),
                                     "--yes", "--plain")
        self.assertEqual(code, es_local.LOCAL_ONLY)
        self.assertEqual(len(code.splitlines()), 1)
        self.assertIn("Evo mode is for local test networks for now", code)
        self.assertEqual(node.coord.mempool, [])
        self.assertEqual(api.account(req.address)["balance"], 100 * VIBE)
        self.assertFalse((self.root / "home" / "jobs").exists())

    def test_testnet_evo_on_a_testnet_made_without_it(self):
        d = self.root / "plain-testnet"
        d.mkdir()
        (d / "genesis.json").write_text(json.dumps({"chain_id": "vibehash-local-1", "allocations": {}}))
        code, _ = self.run_cli("testnet", "--evo", "--dir", str(d), "--plain")
        self.assertIn("made without Evo jobs", str(code))

    def test_a_folder_and_the_cache(self):
        """A model's folder is pinned by where it is; a Hugging Face model by its commit, looked for in the cache first
        (downloaded only when it isn't there), and a worker finds a local network's base in its cache by itself."""
        from vibehash import es_local
        folder = self.root / "My Model"
        folder.mkdir()
        (folder / "config.json").write_text("{}")
        found = es_local.find(str(folder))
        self.assertRegex(found.base, r"^hf:local/My-Model@[0-9a-f]{40}$")
        self.assertEqual(found.folder, folder.resolve())
        self.assertEqual(es_local.find(str(folder)), found)
        commit = "7ae557604adf67be50417f59c2c2f167def9a775"
        snap = self.root / "hub" / "snapshots" / commit
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
        (snap / "model.safetensors").write_bytes(b"")
        calls = []

        def snapshot(repo, revision=None, allow_patterns=None, local_files_only=False):
            calls.append((repo, revision, local_files_only))
            if local_files_only and not (snap / "model.safetensors").exists():
                raise FileNotFoundError("not cached")
            return str(snap)
        import huggingface_hub
        with mock.patch.object(huggingface_hub, "snapshot_download", snapshot):
            got = es_local.find("hf:Qwen/Qwen2.5-0.5B-Instruct")
            self.assertEqual(got.base, f"hf:Qwen/Qwen2.5-0.5B-Instruct@{commit}")
            self.assertEqual(calls, [("Qwen/Qwen2.5-0.5B-Instruct", None, True)])  # the cache, and no more
            spec = es.EsSpec(**{**spec_for().__dict__, "base": got.base, "manifest": es_local.manifest(got.base)})
            self.assertTrue(es_local.find_base(spec))
            self.assertEqual(es.LOCAL[spec.manifest], str(snap))
            del es.LOCAL[spec.manifest]
            (snap / "model.safetensors").unlink()  # not in the cache any more: a worker never downloads it
            calls.clear()
            self.assertFalse(es_local.find_base(spec))
            self.assertEqual(calls, [("Qwen/Qwen2.5-0.5B-Instruct", commit, True)])
            self.assertFalse(es_local.find_base(es.EsSpec(**{**spec.__dict__, "manifest": "9" * 64})))  # not a local one
            calls.clear()
            es_local.find("Qwen/Qwen2.5-0.5B-Instruct")  # a testnet's: downloaded once
            self.assertEqual(calls, [("Qwen/Qwen2.5-0.5B-Instruct", None, True), ("Qwen/Qwen2.5-0.5B-Instruct", None, False)])
        for bad in ("not a model", "https://example.com/x/y", "a/b/c d"):
            with self.assertRaises(ValueError):
                es_local.parse_ref(bad)
        self.assertEqual(es_local.parse_ref("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct@main"),
                         ("Qwen/Qwen2.5-0.5B-Instruct", "main"))
        self.assertTrue(es_local.accepts({"es": True}))
        for g in ({}, {"es": "true"}, {"es": 1}, None, []):
            self.assertFalse(es_local.accepts(g))

    @unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")
    def test_train_evo_posts_the_measured_runs_job(self):
        from vibehash import es_local, settings
        node, api, req = self.node(es_gate=True)
        tiny = tiny_model(self.root / "tiny", seed=1)
        code, out = self.run_cli("train", "--evo", "--base", str(tiny), "--node", node.url, "--wallet",
                                 str(self.root / "requester.json"), "--budget", "20", "--yes", "--plain")
        self.assertEqual(code, 0, out)
        self.assertIn("Posted Evo job evo-tiny-", out)
        self.assertIn("8 twin pairs a generation (16 nudged copies, sigma 0.001), 32 Countdown puzzles each", out)
        self.assertIn("up to 20 generations at 1 VIBE each", out)
        self.assertIn("workers   only this computer's", out)  # a folder's model: no cache elsewhere has it
        txs = [*node.coord.mempool, *(t for b in node.chain.blocks for t in b.txs)]  # waiting, or sealed already
        tx = next(t for t in txs if t.kind == "post_job")
        job = TrainingJob.from_spec_json(tx.memo, tx.sender, tx.amount)
        spec = job.spec
        self.assertTrue(job.es)
        self.assertEqual((tx.sender, tx.amount, job.fee_per_round), (req.address, 20 * VIBE, es_local.FEE))
        self.assertEqual(spec.tensors, es.preset("mlp", 1))  # the last layer's MLP
        self.assertEqual((spec.pairs, spec.pairs_per_unit, spec.problems, spec.max_tokens, spec.level, spec.retarget,
                          spec.optimizer, spec.sigma), (8, 2, 32, 64, 2, (2, 0.5, 0.1), "sgd", 1e-3))
        self.assertAlmostEqual(spec.alpha, 5e-7, delta=1e-18)  # the paper's step, sigma**2 / 2
        self.assertEqual(spec.manifest, es_local.manifest(spec.base))
        self.assertEqual(settings.load()["last_job"], job.job_id)

    @unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")
    def test_testnet_evo_end_to_end_and_carrying_on(self):
        """A whole Evo network on the CPU: its genesis takes ES jobs and stakes its workers, it posts the job, its
        workers rebuild each round's state from the chain, and the chain alone rebuilds the job's state. Run again
        without the switch, it carries on as an Evo network with the model it was made with."""
        from vibehash.storage import Store
        tiny = tiny_model(self.root / "tiny", seed=1)
        d = self.root / "evo-net"
        patch = mock.patch.object(es, "reward", fake_reward)  # scores a random tiny model's babble can earn
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(es.unload)
        self.addCleanup(es.cache_noise, 0)
        fast = ["--device", "cpu", "--port", "0", "--plain", "--register-s", "1", "--compute-s", "2", "--reveal-s", "1",
                "--verify-s", "2"]
        code, out = self.run_cli("testnet", "--evo", "--evo-model", str(tiny), "--dir", str(d), "--rounds", "3", *fast)
        self.assertEqual(code, 0, out)
        genesis = json.loads((d / "genesis.json").read_text())
        self.assertIs(genesis["es"], True)
        self.assertEqual(len(genesis["stakes"]), 8)
        self.assertIn("evo      Evo mode: training by evolution, the chain is the model: local/tiny", out)
        self.assertRegex(out, r"block \d+: Evo job evo-1, level \d, mean reward [\d.]+% over 8 pairs, \d+ B of seeds")
        self.assertIn("train --evo --node", out)
        code, again = self.run_cli("testnet", "--dir", str(d), "--rounds", "1", *fast)  # no --evo: carries on as one
        self.assertEqual(code, 0, again)
        self.assertIn("Finding " + str(tiny), again)
        store = Store(d / "node-data")
        self.addCleanup(store.close)
        chain = store.open_chain(genesis)
        state, rounds = es.rebuild_chain(chain.blocks, store.load_record, "evo-1")
        mine = [b for b in chain.blocks if b.summary is not None and b.summary.job_id == "evo-1"]
        self.assertEqual(rounds, len(mine))
        self.assertGreaterEqual(rounds, 4)
        self.assertEqual(es.state_hash(state), mine[-1].summary.params_hash)
        self.assertIn("evo-1", chain.state.jobs)
        lines = [json.loads(x) for x in (self.root / "home" / "logs" / "rounds.jsonl").read_text().splitlines()]
        self.assertTrue(any(x.get("kind") == "es" and (x.get("accepted") or 0) > 0 for x in lines),
                        lines[-3:])  # its round log counts an Evo round's units


if __name__ == "__main__":
    unittest.main()

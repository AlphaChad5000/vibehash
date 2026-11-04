"""`vibehash selftest`, and the plain-language advice and unattended running it goes with. No GPU and no internet:
devices are faked where a GPU matters, the node is an in-process one (test_network's helpers), and PyPI and Hugging
Face are stood in for."""
import contextlib
import errno
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from vibehash import advice, cli, compute, filelock, selftest, settings, unattended
from vibehash import wallet as wallets
from vibehash.client import NodeError
from vibehash.crypto import KeyPair, checksummed

from tests.test_network import start_network


class FakeGpu(compute.NumpyBackend):
    """A "GPU" that trains with NumPy, optionally off by `drift` (relative) every segment, as a broken driver or a
    non-deterministic setup would be."""
    name, device = "cuda", "cuda:0"

    def __init__(self, device="cuda:0", drift=0.0):
        self.device, self.drift = device, drift
        self.rng = np.random.default_rng(0)

    def train_segment(self, *args):
        out = super().train_segment(*args)
        return out * (1 + self.rng.normal(0, self.drift, out.shape)) if self.drift else out

    def memory_mb(self):
        return 12282

    def describe(self):
        return "Fake GPU (CUDA)"


def fake_devices(drift=0.0, self_test_passes=True):
    """Patches for a computer with one GPU (faked) and its CPU."""
    devices = [compute.Device("cuda:0", "Fake GPU (CUDA)"), compute.Device("cpu", "CPU (NumPy, 4 threads)")]
    return [mock.patch.object(compute, "list_devices", return_value=devices), no_driver(),
            mock.patch.object(compute, "TorchBackend", lambda device: FakeGpu(device, drift)),
            mock.patch.object(compute, "self_test", return_value=None if self_test_passes else "results differ")]


def no_driver():
    from vibehash import gpusetup
    return mock.patch.object(gpusetup, "detect", return_value=[])  # (nvidia-smi and the OS's device list: slow)


def cpu_only():
    return [mock.patch.object(compute, "list_devices", return_value=[compute.Device("cpu", "CPU (NumPy, 4 threads)")]),
            no_driver()]


def no_internet():
    """PyPI and Hugging Face, stood in for: the newest release is this one, and the Hub answers."""
    from vibehash import hub
    return [mock.patch.object(selftest, "newest_release", return_value=selftest.version()),
            mock.patch.object(hub.Hub, "model_info", return_value={})]


class Home(unittest.TestCase):
    def setUp(self):
        home = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(home.cleanup)
        patch = mock.patch.dict(os.environ, {"VIBEHASH_HOME": home.name})
        patch.start()
        self.addCleanup(patch.stop)
        for var in ("VIBEHASH_NODE", "VIBEHASH_WALLET", "VIBEHASH_PASSWORD", "RUNPOD_POD_ID"):
            os.environ.pop(var, None)
        self.home = Path(home.name)

    def run_test(self, *patches, **kw) -> tuple[int, dict, str]:
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            for p in [*no_internet(), *patches]:
                stack.enter_context(p)
            code, report = selftest.run(out=out, **kw)
        return code, report, out.getvalue()


def check(report, name):
    return next(c for c in report["checks"] if c["name"] == name)


class SelfTestTests(Home):
    def test_a_cpu_computer_against_a_node(self):
        node, api, req, *_ = start_network(self, 0)
        keys = KeyPair.generate(b"newcomer")
        wallets.save(keys, self.home / "wallet.json", None)
        settings.save(node=node.url, wallet=str(self.home / "wallet.json"))
        code, report, out = self.run_test(*cpu_only(), mock.patch.object(compute, "torch_available", return_value=False))
        self.assertEqual(code, 0, out)
        self.assertTrue(report["ok"])
        train = check(report, "train")
        self.assertEqual((train["status"], train["data"]["device"]), ("ok", "cpu"))
        self.assertTrue(train["data"]["repeats"])
        self.assertLess(train["data"]["distance"], 1e-4)  # replayed on the CPU, as a verifier would
        self.assertEqual(check(report, "node")["data"]["chain_id"], "vibehash-localtest")
        self.assertEqual(check(report, "clock")["status"], "ok")
        self.assertIn(checksummed(keys.address), check(report, "wallet")["text"])  # the text as people copy it
        self.assertEqual(check(report, "wallet")["data"]["address"], keys.address)  # --json's data as the ledger has it
        self.assertIn("balance 0 VIBE", check(report, "balance")["text"])
        self.assertEqual(check(report, "lora")["status"], "skip")  # no PyTorch here
        self.assertTrue(report["can_take"]["csv"])
        self.assertFalse(report["can_take"]["qwen3-0.6b"])
        self.assertEqual(report["next"], "vibehash worker")
        self.assertIn("on the CPU", out)
        self.assertIn("This computer can mine", out)

    def test_marks_fall_back_to_ascii(self):
        rep = selftest.Report(out=io.StringIO())  # no encoding: ASCII
        rep.add("x", "ok", "fine")
        rep.add("y", "fail", "broken")
        self.assertEqual(rep.out.getvalue().split(), ["ok", "fine", "XX", "broken"])
        utf8 = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        self.assertEqual(selftest.Report(out=utf8).marks["ok"], "✓")

    def test_a_clock_too_far_off_fails_with_what_to_do(self):
        node, *_ = start_network(self, 0)
        with mock.patch.object(selftest, "skew_from", return_value=-300.0):
            code, report, out = self.run_test(*cpu_only(), node=node.url)
        self.assertEqual(code, 1)
        clock = check(report, "clock")
        self.assertEqual(clock["status"], "fail")
        self.assertIn("300 s behind the node's", clock["text"])
        self.assertIn("120 s", clock["text"])
        self.assertIn("Sync now", clock["text"])
        self.assertIn("fix the clock", report["next"])

    def test_the_clock_is_read_from_the_date_header(self):
        sent = 1793660000.0
        date = "Mon, 02 Nov 2026 22:53:20 GMT"  # 1793660000, whole seconds (so the node's time is +0.5 s on average)
        self.assertAlmostEqual(selftest.skew_from(date, sent, sent + 0.2), -0.4, places=6)
        self.assertAlmostEqual(selftest.skew_from(date, sent + 200, sent + 200), 199.5, places=6)
        self.assertIsNone(selftest.skew_from(None, sent, sent))
        self.assertIsNone(selftest.skew_from("not a date", sent, sent))

    def test_a_node_out_of_reach(self):
        code, report, out = self.run_test(*cpu_only(), node="http://127.0.0.1:1")
        self.assertEqual(code, 1)
        node = check(report, "node")
        self.assertEqual(node["status"], "fail")
        self.assertIn("can't reach the node at http://127.0.0.1:1", node["text"])
        self.assertIn("Nothing is listening at 127.0.0.1 port 1", node["text"])
        self.assertIn("connection to the node", report["next"])

    def test_offline_needs_no_node(self):
        code, report, out = self.run_test(*cpu_only(), offline=True)
        self.assertEqual(code, 0, out)
        self.assertEqual(check(report, "node")["status"], "skip")
        self.assertEqual(check(report, "release")["status"], "skip")
        self.assertEqual(report["next"], "vibehash setup")  # no wallet yet

    def test_a_gpu_that_drifts_is_caught_and_the_cpu_takes_over(self):
        code, report, out = self.run_test(*fake_devices(drift=1e-3), offline=True)
        self.assertEqual(code, 0, out)  # it can still mine, on the CPU
        bad = check(report, "train cuda:0")
        self.assertEqual(bad["status"], "fail")
        self.assertIn("verifiers would reject it", bad["text"])
        self.assertIn("Update the graphics driver", bad["text"])
        self.assertEqual(check(report, "train")["data"]["device"], "cpu")
        self.assertIn("on the CPU", out)

    def test_a_gpu_that_fails_its_self_test(self):
        code, report, out = self.run_test(*fake_devices(self_test_passes=False), offline=True)
        self.assertEqual(code, 0)
        self.assertIn("NOT USABLE", check(report, "device cuda:0")["text"])
        self.assertEqual(check(report, "train")["data"]["device"], "cpu")

    def test_a_good_gpu_and_what_it_can_take(self):
        with mock.patch.object(selftest, "lora_step", return_value={"ok": True, "seconds": 0.1, "worst": 0.0,
                                                                    "tolerance": 1e-4, "moved": 1e-3, "steps": 4}):
            code, report, out = self.run_test(*fake_devices(), mock.patch.object(compute, "torch_available",
                                                                                  return_value=True),
                                              mock.patch("vibehash.modelstore.free_space", return_value=500 * 10**9),
                                              offline=True)
        self.assertEqual(code, 0, out)
        self.assertEqual(check(report, "train")["data"]["device"], "cuda:0")
        self.assertEqual(check(report, "lora")["status"], "ok")
        take = report["can_take"]
        self.assertTrue(take["csv"] and take["qwen3-0.6b"] and take["qwen3-4b"])  # 12 GB: Qwen3-4B only just
        self.assertIn("only just", out)

    def test_waiting_jobs_are_explained(self):
        node, *_ = start_network(self, 0)
        why = ("job pilot waits: a fine-tune's work is checked by workers, so it needs 2 staked workers with 7.8 GB of "
               "GPU memory each (one to train a unit, one to check it), and 0 registered. Workers earn their stake on "
               "other jobs first")
        with node.coord.lock:
            node.coord.waiting["pilot"] = why
        wallets.save(KeyPair.generate(b"newcomer"), self.home / "wallet.json", None)
        settings.save(node=node.url, wallet=str(self.home / "wallet.json"))
        code, report, out = self.run_test(*cpu_only())
        line = check(report, "waiting pilot")
        self.assertEqual(line["status"], "warn")
        # This test network has no earn-in, so the stake can only come from the operator.
        self.assertIn("doesn't let newcomers earn it", line["text"])
        self.assertIn("vibehash stake --amount 5", line["text"])

    def test_json_from_the_command_line(self):
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            for p in [*no_internet(), *cpu_only(), contextlib.redirect_stdout(out)]:
                stack.enter_context(p)
            with self.assertRaises(SystemExit) as e:
                cli.main(["selftest", "--offline", "--json"])
        self.assertEqual(e.exception.code, 0)
        report = json.loads(out.getvalue())
        self.assertEqual(set(report), {"ok", "version", "node", "checks", "can_take", "next"})
        self.assertTrue(all({"section", "name", "status", "text", "data"} <= set(c) for c in report["checks"]))

    def test_the_fruit_demo_job_is_the_public_ones(self):
        job = selftest.fruit_job()
        # The packaged fruit.csv makes the public job's arrays (its file's bytes depend on the system that saves it)
        self.assertEqual(selftest.arrays_sha(job.data), selftest.FRUIT_ARRAYS)
        self.assertEqual(json.loads(job.spec_json())["model"], {"n_in": 5, "n_out": 4, "n_hidden": [128, 128]})

    def test_fine_tune_memory_matches_the_measured_table(self):
        # FINETUNING.md: Qwen3-0.6B measured 3.5 GiB at batch 8, Qwen3-4B 11.1 GiB at batch 4 (512-token sequences).
        self.assertTrue(3.4 * 1024 <= selftest.finetune_need("qwen3-0.6b") <= 4.1 * 1024)
        self.assertTrue(11.1 * 1024 <= selftest.finetune_need("qwen3-4b") <= 11.6 * 1024)

    @unittest.skipUnless(compute.torch_available(), "needs PyTorch")
    def test_the_lora_step_replays(self):
        from vibehash import finetune
        before = dict(finetune._runtimes)
        got = selftest.lora_step("cpu")
        self.assertTrue(got["ok"], got)
        self.assertEqual(got["worst"], 0.0)  # the same device repeats it to the bit
        self.assertGreater(got["moved"], 0)
        self.assertEqual(finetune._runtimes, before)  # the tiny model is gone again

    def test_a_real_model_is_downloaded_only_with_a_yes(self):
        from vibehash import finetune, hub
        listing = {"model.safetensors": hub.FileInfo("model.safetensors", 1_500_000_000, "0" * 40, "1" * 64)}
        config = json.dumps({"model_type": "qwen3", "vocab_size": 151936, "hidden_size": 1024, "num_hidden_layers": 28,
                             "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
                             "intermediate_size": 3072, "tie_word_embeddings": True}).encode()
        asked = []
        with mock.patch.object(hub.Hub, "resolve", return_value=("Qwen/Qwen3-0.6B", "c" * 40)), \
                mock.patch.object(hub.Hub, "get_file", return_value=config), \
                mock.patch.object(hub.Hub, "list_files", return_value=listing), \
                mock.patch.object(finetune, "model_manifest", side_effect=AssertionError("downloaded without a yes")), \
                mock.patch.dict("sys.modules", {"torch": mock.MagicMock()}):
            self.assertIsNone(selftest.model_unit("qwen3-0.6b", "cuda:0", ask=None)["ok"])
            got = selftest.model_unit("qwen3-0.6b", "cuda:0", ask=lambda q: asked.append(q) or False)
        self.assertIsNone(got["ok"])
        self.assertIn("1.5 GB", asked[0])


class AdviceTests(unittest.TestCase):
    def test_why_a_connection_failed(self):
        cases = {"[Errno 11001] getaddrinfo failed": "dns", "[Errno -2] Name or service not known": "dns",
                 "[WinError 10061] No connection could be made because the target machine actively refused it": "refused",
                 "[Errno 111] Connection refused": "refused", "timed out": "timeout",
                 "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed": "tls", "something odd": "other"}
        for reason, kind in cases.items():
            self.assertEqual(advice.reach_kind(reason), kind, reason)
        text = advice.explain(NodeError("can't reach node at https://node.example: timed out"))
        self.assertIn("Dorm and campus Wi-Fi", text)
        self.assertIn("https://node.example", text)

    def test_messages_the_node_refuses(self):
        self.assertIn("Sync now", advice.explain(NodeError("POST /commit: message timestamp too far from node clock", 400)))
        self.assertIn("different network", advice.explain(NodeError("node switched networks (genesis changed); refusing")))
        self.assertIn("ask its operator", advice.explain(NodeError("POST /register: stake at least 500000000 syn to take "
                                                                   "work", 400)))
        self.assertIsNone(advice.explain(NodeError("POST /reveal: no such round open", 400)))

    def test_out_of_memory_and_a_full_disk(self):
        class OutOfMemoryError(RuntimeError):
            pass
        self.assertIn("ran out of memory", advice.explain(OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")))
        self.assertIn("ran out of memory", advice.explain(RuntimeError("CUDA error: out of memory")))
        full = OSError(errno.ENOSPC, "No space left on device", "/data/cache/x")
        self.assertIn("/data/cache/x is full", advice.explain(full))
        self.assertIsNone(advice.explain(ValueError("something else")))
        self.assertIn("needs about 11.2 GB", advice.out_of_memory("cuda:0", 11.2 * 1024, 12 * 1024, 3 * 1024))

    def test_a_waiting_job(self):
        why = "job ft waits: a fine-tune's work is checked by workers, so it needs 2 staked workers with 4.0 GB ..."
        unstaked = advice.waiting("ft", why, 10**8, 5 * 10**8, 12288, 4096, ["fruit-demo"], " (about 2 rounds)")
        self.assertIn("1 VIBE of the 5 VIBE stake", unstaked)
        self.assertIn("earns the rest on fruit-demo (about 2 rounds)", unstaked)
        self.assertIn("this one has 4.0 GB", advice.waiting("ft", why, 10**9, 5 * 10**8, 4096, 8000))
        self.assertIn("this one counts", advice.waiting("ft", why, 10**9, 5 * 10**8, 12288, 4096))


class UnattendedTests(Home):
    def test_nul_and_pipes_are_no_terminal(self):
        with open(os.devnull) as nul:  # Windows says NUL isatty(); there's no console behind it, though
            self.assertFalse(unattended.interactive(nul))
        self.assertFalse(unattended.interactive(io.StringIO()))

    def test_recovery_words_go_to_a_private_file(self):
        words = " ".join(["abandon"] * 23 + ["art"])
        path = unattended.save_words(words, "vibe1abcdef0123456789", self.home / "wallet.json")
        text = path.read_text()
        self.assertIn("24. art", text)
        self.assertIn("delete this file", text)
        if os.name != "nt":
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):  # never overwritten
            unattended.save_words(words, "vibe1abcdef0123456789", self.home / "wallet.json")
        self.assertIn("NOT printed", unattended.words_note(path))

    def test_one_worker_per_wallet(self):
        first = unattended.WorkerLock(self.home / "cache" / "vibe1me")
        self.assertTrue(first.held)
        second = unattended.WorkerLock(self.home / "cache" / "vibe1me")
        self.assertFalse(second.held)
        why = second.why_not("vibe1me0000000000000", "vibehash")
        self.assertIn(f"process {os.getpid()}", why)
        self.assertIn("--device cuda:1 --wallet gpu1.json", why)
        self.assertTrue(unattended.WorkerLock(self.home / "cache" / "vibe1other").held)  # another wallet: fine
        first.release()
        self.assertTrue(unattended.WorkerLock(self.home / "cache" / "vibe1me").held)

    def test_a_lock_waits_for_its_holder(self):
        import threading
        import time
        path, order = self.home / "x.lock", []
        held = filelock.try_hold(path)

        def later():
            with filelock.hold(path, poll_s=0.01):
                order.append("second")
        t = threading.Thread(target=later)
        t.start()
        time.sleep(0.2)
        order.append("first")
        held.release()
        t.join(5)
        self.assertEqual(order, ["first", "second"])

    def test_a_pod_keeps_its_settings_on_the_volume(self):
        os.environ.pop("VIBEHASH_HOME")
        fake_home = self.home / "root"
        with mock.patch.dict(os.environ, {"RUNPOD_POD_ID": "abc123"}), \
                mock.patch.object(settings.Path, "home", return_value=fake_home), \
                mock.patch.object(settings.os.path, "isdir", lambda p: p == "/workspace" or os.path.exists(p)):
            self.assertEqual(settings.home(), Path("/workspace/.vibehash"))
            (fake_home / ".vibehash").mkdir(parents=True)
            (fake_home / ".vibehash" / "wallet.json").write_text("{}")  # an earlier setup there: keep using it
            self.assertEqual(settings.home(), fake_home / ".vibehash")
        with mock.patch.object(settings.Path, "home", return_value=self.home / "elsewhere"):
            self.assertEqual(settings.home(), self.home / "elsewhere" / ".vibehash")  # not a pod


if __name__ == "__main__":
    unittest.main()

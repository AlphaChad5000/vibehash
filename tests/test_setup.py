"""Setup and training helpers: GPU detection and PyTorch build choice, saved settings, CSV tables and model
files. No GPU or network needed.

`vibehash setup`, `testnet` and `train` against a live node are in test_network.py.
"""
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from vibehash import gpusetup, settings, tabular
from vibehash.blobs import BlobError, BlobStore
from vibehash.cli import normalise_url
from vibehash.gpusetup import Gpu, builds_for, choose, parse_nvidia_smi
from vibehash.model import ModelSpec

# nvidia-smi's header line: 2026 drivers say "CUDA UMD Version", older ones "CUDA Version".
NEW_HEADER = "| NVIDIA-SMI 616.92                 KMD Version: 616.92        CUDA UMD Version: 13.4     |"
OLD_HEADER = "| NVIDIA-SMI 536.23                 Driver Version: 536.23       CUDA Version: 12.2     |"


def nvidia(name, cuda, compute):
    return Gpu("nvidia", name, cuda, compute)


def tags(gpu):
    return [b.tag for b in builds_for(gpu)[0]]


class GpuDetectionTests(unittest.TestCase):
    def test_parses_old_and_new_nvidia_smi(self):
        self.assertEqual(parse_nvidia_smi(NEW_HEADER, "NVIDIA GeForce RTX 4070 SUPER, 8.9\n"),
                         [nvidia("NVIDIA GeForce RTX 4070 SUPER", (13, 4), (8, 9))])
        self.assertEqual(parse_nvidia_smi(OLD_HEADER, "NVIDIA GeForce GTX 1060 6GB\n"),  # too old for compute_cap
                         [nvidia("NVIDIA GeForce GTX 1060 6GB", (12, 2), None)])

    def test_classify(self):
        for name, vendor in [("NVIDIA GeForce RTX 3060", "nvidia"), ("AMD Radeon RX 7900 XTX", "amd"),
                             ("Intel(R) Arc(TM) A770 Graphics", "intel"), ("Microsoft Basic Display Adapter", None)]:
            self.assertEqual(gpusetup.classify(name), vendor)

    def test_nvidia_builds_follow_driver_and_architecture(self):
        self.assertEqual(tags(nvidia("RTX 4070 SUPER", (13, 4), (8, 9)))[:3], ["cu134", "cu132", "cu130"])
        self.assertEqual(tags(nvidia("RTX 4070 SUPER", (13, 2), (8, 9)))[0], "cu132")
        self.assertEqual(tags(nvidia("Tesla V100", (12, 9), (7, 0)))[0], "cu126")  # Volta left 12.8/12.9 in 2.11
        self.assertEqual(tags(nvidia("Tesla V100", (13, 0), (7, 0)))[0], "cu126")
        self.assertEqual(tags(nvidia("Tesla T4", (12, 9), (7, 5)))[0], "cu129")  # Turing still has them
        self.assertEqual(tags(nvidia("RTX 4070 SUPER", (12, 8), (8, 9)))[0], "cu128")  # CUDA 13 needs a newer driver
        self.assertEqual(tags(nvidia("GTX 1060", (13, 0), (6, 1)))[0], "cu126")  # Pascal left the 12.8+ builds
        self.assertEqual(tags(nvidia("RTX 5090", (12, 8), (12, 0))), ["cu128"])  # Blackwell needs 12.8+
        self.assertEqual(tags(nvidia("RTX 5090", (12, 6), (12, 0))), [])
        self.assertIn("nvidia.com", builds_for(nvidia("RTX 5090", (12, 6), (12, 0)))[1])
        self.assertIn("driver", builds_for(Gpu("nvidia", "RTX 3060"))[1])  # Windows lists it, but no nvidia-smi

    def test_other_vendors(self):
        self.assertEqual(tags(Gpu("intel", "Intel(R) Arc(TM) Graphics")), ["xpu"])
        self.assertEqual(tags(Gpu("intel", "Intel(R) UHD Graphics 620")), [])
        self.assertEqual(tags(Gpu("intel", "Intel(R) Data Center GPU Max 1550")), ["xpu"])  # PyTorch's, on Linux
        self.assertEqual(tags(Gpu("intel", "Intel Corporation Ponte Vecchio XT (1 Tile)")), ["xpu"])  # as lspci may say
        self.assertEqual(tags(Gpu("intel", "Intel(R) Data Center GPU Flex 170")), [])  # IPEX's, never stock PyTorch's
        self.assertEqual(tags(Gpu("apple", "Apple silicon GPU")), [None])  # the plain PyPI build
        apple = Gpu("apple", "Apple silicon GPU")
        self.assertEqual(builds_for(apple, "darwin", (15, 1))[0][0].pin, "")
        # PyTorch 2.15 uses the GPU only from macOS 15: macOS 14 gets 2.14, and older says why (docs/GPUS.md, risk 12)
        [b], _ = builds_for(apple, "darwin", (14, 6))
        self.assertEqual((b.tag, b.pin), (None, "<2.15"))
        self.assertEqual(builds_for(apple, "darwin", (13, 5)), ([], "PyTorch uses an Apple GPU from macOS 14 on; "
                                                                 "update macOS to mine on it"))
        from unittest import mock

        from vibehash import gpusetup
        with mock.patch.object(gpusetup.subprocess, "call", return_value=0) as call,                 mock.patch.object(gpusetup, "installed_torch", return_value="2.15.0"):
            gpusetup.install(b)
        self.assertEqual(call.call_args[0][0][-1], "torch<2.15")  # pip's own limit, from the PyPI index
        amd = lambda rocm: [b.tag for b in builds_for(Gpu("amd", "Radeon RX 7900 XTX", rocm=rocm), "linux")[0]]  # noqa: E731
        self.assertEqual(amd((6, 4)), ["rocm6.4", "rocm6.3", "rocm6.2"])
        self.assertEqual(amd((7, 14))[:2], ["rocm7.14", "rocm7.2"])  # 7.14 is after 7.2 (ROCm's new numbering)
        self.assertEqual(amd((10, 1))[:3], ["rocm10.0", "rocm7.14", "rocm7.2"])
        self.assertIn("ROCm", builds_for(Gpu("amd", "Radeon RX 7900 XTX"), "linux")[1])
        # Windows: PyTorch has no ROCm build there, HIP SDK or not, and says so rather than "no build for Python"
        for rocm in (None, (6, 2)):
            builds, why = builds_for(Gpu("amd", "Radeon RX 7900 XTX", rocm=rocm), "win32")
            self.assertEqual(builds, [])
            self.assertIn("Linux", why)

    def test_choose_prefers_nvidia_and_skips_unpublished_builds(self):
        gpus = [Gpu("intel", "Intel(R) Arc(TM) Graphics"), nvidia("RTX 4070 SUPER", (13, 4), (8, 9))]
        gpu, build, why = choose(gpus, available=lambda b: b.tag not in ("cu134", "cu132"))
        self.assertEqual((gpu.vendor, build.tag, why), ("nvidia", "cu130", None))
        gpu, build, why = choose([nvidia("RTX 4070 SUPER", (13, 4), (8, 9))], available=lambda b: False)
        self.assertIsNone(build)
        self.assertIn("no build", why)
        self.assertEqual(choose([]), (None, None, None))


class SettingsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patch = mock.patch.dict(os.environ, {"VIBEHASH_HOME": tmp.name})
        patch.start()
        self.addCleanup(patch.stop)
        for var in ("VIBEHASH_NODE", "VIBEHASH_WALLET"):
            os.environ.pop(var, None)
        self.home = Path(tmp.name)

    def test_flag_beats_environment_beats_saved_beats_the_public_node(self):
        self.assertEqual(settings.node_url(), settings.PUBLIC_NODE)
        self.assertIsNone(settings.chosen_node())
        settings.save(node="http://NODE-ADDRESS:8545")  # the old guide's placeholder, saved by mistake, isn't a choice
        self.assertEqual(settings.node_url(), settings.PUBLIC_NODE)
        settings.save(node="http://saved:8545")
        self.assertEqual(settings.node_url(), "http://saved:8545")
        os.environ["VIBEHASH_NODE"] = "http://env:8545"
        self.assertEqual(settings.node_url(), "http://env:8545")
        self.assertEqual(settings.node_url("http://flag:8545"), "http://flag:8545")

    def test_typing_the_public_nodes_name_gets_its_https_address(self):
        from vibehash.cli import normalise_url
        self.assertEqual(normalise_url("node.vibehash.org"), settings.PUBLIC_NODE)
        self.assertEqual(normalise_url("192.168.1.20"), "http://192.168.1.20:8545")

    def test_default_wallet_only_once_it_exists(self):
        self.assertIsNone(settings.wallet_path())
        settings.default_wallet().write_text("{}")
        self.assertEqual(settings.wallet_path(), self.home / "wallet.json")
        settings.save(wallet="elsewhere.json")
        self.assertEqual(settings.wallet_path(), Path("elsewhere.json"))

    def test_save_merges(self):
        settings.save(node="a")
        settings.save(wallet="b")
        self.assertEqual(settings.load(), {"node": "a", "wallet": "b"})

    def test_every_command_takes_an_address_as_setup_does(self):
        """`worker --node 192.168.1.20` (and $VIBEHASH_NODE without http://, as the installer passes it on) used to
        fail with "unknown url type"; only setup added the scheme."""
        import argparse

        from vibehash.cli import node_of
        self.assertEqual(node_of(argparse.Namespace(node="192.168.1.20")), "http://192.168.1.20:8545")
        os.environ["VIBEHASH_NODE"] = "node.vibehash.org"
        self.assertEqual(node_of(argparse.Namespace(node=None)), settings.PUBLIC_NODE)

    def test_normalise_url(self):
        self.assertEqual(normalise_url("192.168.1.20"), "http://192.168.1.20:8545")
        self.assertEqual(normalise_url("node.example:9000/"), "http://node.example:9000")
        self.assertEqual(normalise_url("https://node.example"), "https://node.example")


class HeadlessSetupTests(unittest.TestCase):
    """`setup` with nobody at the keyboard (nohup, tmux left alone, a Docker CMD, a pod): stdin is NUL or /dev/null,
    output goes to a log. Nothing may wait for an answer, and the recovery words mustn't land in the log. (Windows
    calls NUL a terminal: before, getpass waited on the console for ever.)"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        patch = mock.patch.dict(os.environ, {"VIBEHASH_HOME": tmp.name})
        patch.start()
        self.addCleanup(patch.stop)
        for var in ("VIBEHASH_NODE", "VIBEHASH_WALLET", "VIBEHASH_PASSWORD", "RUNPOD_POD_ID"):
            os.environ.pop(var, None)
        self.home = Path(tmp.name)

    def setup(self, *argv) -> tuple[object, str]:
        import contextlib
        import io

        from vibehash import cli
        out = io.StringIO()
        with open(os.devnull) as nul, mock.patch("sys.stdin", nul), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                cli.main(["setup", "--node", "http://127.0.0.1:1", "--gpu", "skip", *argv])
                code = 0
            except SystemExit as e:
                code = e.code
        return code, out.getvalue()

    def test_no_password_says_which_variable_gives_it(self):
        code, out = self.setup()
        self.assertIn("VIBEHASH_PASSWORD", str(code))
        self.assertIn("no terminal", str(code))
        self.assertFalse((self.home / "wallet.json").exists())
        self.assertIn("Nothing is listening at 127.0.0.1 port 1", out)  # the node: what's wrong, in plain words

    def test_unattended_setup_keeps_the_words_out_of_the_log(self):
        from vibehash import mnemonic
        with mock.patch.dict(os.environ, {"VIBEHASH_PASSWORD": "unattended"}):
            code, out = self.setup("--deepseek", "no")
        self.assertEqual(code, 0, out)
        address = json.loads((self.home / "wallet.json").read_text())["address"]
        self.assertNotRegex(out, r"\d+\. [a-z]+ ")  # no words in the output
        self.assertIn("NOT printed", out)
        words = re.findall(r"^\s*\d+\. ([a-z]+)$", (self.home / f"recovery-words-{address[:12]}.txt").read_text(), re.M)
        self.assertEqual(mnemonic.keypair(" ".join(words)).address, address)
        self.assertEqual(settings.load()["deepseek"], "no")
        self.assertIn("Almost there", out)  # and it didn't wait to ask "Start mining now?"

    def test_the_worker_names_the_variable_too(self):
        from vibehash import cli
        from vibehash import wallet as wallets
        from vibehash.crypto import KeyPair
        wallets.save(KeyPair.generate(b"locked"), self.home / "wallet.json", "secret")
        with open(os.devnull) as nul, mock.patch("sys.stdin", nul), self.assertRaises(SystemExit) as e:
            cli.main(["worker", "--node", "http://127.0.0.1:1"])
        self.assertIn("VIBEHASH_PASSWORD", str(e.exception.code))


class TabularTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def csv(self, name, text):
        p = self.dir / name
        p.write_text(text)
        return p

    def test_reads_numbers_and_named_labels(self):
        rows = "\n".join(f"{i},{i * 2.5},{'big' if i > 9 else 'small'}" for i in range(20))
        t = tabular.read_csv(self.csv("a.csv", "x,y,size\n" + rows))
        self.assertEqual((t.features, t.label, t.x.shape), (["x", "y"], "size", (20, 2)))
        data, classes, mean, std = tabular.to_dataset(t)
        self.assertEqual(classes, ["big", "small"])
        self.assertEqual(len(data.y_train) + len(data.y_val), 20)
        self.assertAlmostEqual(float(data.x_train.mean()), 0.0, places=9)  # standardised on the training rows

    def test_numeric_labels_sort_as_numbers(self):
        self.assertEqual(tabular.class_names(["10", "9", "2"]), ["2", "9", "10"])

    def test_explains_bad_files(self):
        cases = {"x,label\n1,a\noops,b\n" + "3,a\n" * 10: "isn't a number", "x,label\n" + "1,a\n" * 12: "only one value",
                 "x,label\n1,a\n2,b\n": "too few", "x,label\n1,a,9\n": "has 3 values"}
        for text, message in cases.items():
            with self.subTest(message=message), self.assertRaises(tabular.TableError) as e:
                tabular.to_dataset(tabular.read_csv(self.csv("bad.csv", text)))
            self.assertIn(message, str(e.exception))
        with self.assertRaises(tabular.TableError) as e:
            tabular.read_csv(self.csv("b.csv", "x,label\n1,a\n"), label="price")
        self.assertIn("no column named 'price'", str(e.exception))

    def test_model_round_trip_and_predictions(self):
        spec = ModelSpec(2, (8,), 3, "float32")
        weights = spec.init_params(1)
        path = tabular.save_model(self.dir / "m.npz", spec, weights, ["a", "b"], ["x", "y", "z"], [1.0, 2.0], [3.0, 4.0], {"label": "kind"})
        m = tabular.load_model(path)
        self.assertEqual((m.features, m.classes, m.info["label"]), (["a", "b"], ["x", "y", "z"], "kind"))
        probs = m.predict([[1.0, 2.0], [4.0, 6.0]])
        self.assertEqual(probs.shape, (2, 3))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        # Columns are matched by name, so their order and extra columns in the file don't matter.
        t = tabular.read_csv(self.csv("new.csv", "b,note,a\n2,hi,1\n"), label="kind", features=m.features)
        np.testing.assert_array_equal(t.x, [[1.0, 2.0]])
        self.assertIsNone(t.y)
        out = tabular.write_predictions(self.dir / "new.csv", self.dir / "out.csv", ["y"], np.array([0.75]))
        self.assertEqual(out.read_text().splitlines(), ["b,note,a,predicted,confidence", "2,hi,1,y,0.750"])

    def test_refuses_files_that_arent_models(self):
        np.savez(self.dir / "x.npz", weights=np.zeros(3))
        with self.assertRaises(tabular.TableError):
            tabular.load_model(self.dir / "x.npz")


class ConcurrentUploadTests(unittest.TestCase):
    """Two workers holding a replicated unit upload identical checkpoints at the same moment."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.store = BlobStore(Path(tmp.name))
        self.data = bytes(range(256)) * 64
        import hashlib
        self.bid = hashlib.sha256(self.data).hexdigest()

    def test_interleaved_uploads_of_one_blob_both_succeed(self):
        half = len(self.data) // 2
        self.store.write_chunk(self.bid, 0, self.data[:half], len(self.data), "vibe1aaa")
        self.store.write_chunk(self.bid, 0, self.data[:half], len(self.data), "vibe1bbb")
        self.assertEqual(self.store.received(self.bid, "vibe1aaa"), half)
        self.assertEqual(self.store.write_chunk(self.bid, half, self.data[half:], len(self.data), "vibe1aaa"), len(self.data))
        self.assertEqual(self.store.write_chunk(self.bid, half, self.data[half:], len(self.data), "vibe1bbb"), len(self.data))
        self.assertEqual(self.store.path(self.bid).read_bytes(), self.data)

    def test_simultaneous_finishes(self):
        import threading
        owners = [f"vibe1{i:03d}" for i in range(8)]
        errors = []

        def upload(owner):
            try:
                self.store.write_chunk(self.bid, 0, self.data, len(self.data), owner)
            except Exception as e:  # noqa: BLE001 - any failure is the bug
                errors.append(e)
        threads = [threading.Thread(target=upload, args=(o,)) for o in owners]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.store.path(self.bid).read_bytes(), self.data)

    def test_junk_from_one_uploader_cant_spoil_another(self):
        n = len(self.data)
        self.store.write_chunk(self.bid, 0, self.data[:10], n, "vibe1honest")
        with self.assertRaises(BlobError):
            self.store.write_chunk(self.bid, 0, b"\0" * n, n, "vibe1griefer")
        self.store.write_chunk(self.bid, 10, self.data[10:], n, "vibe1honest")
        self.assertTrue(self.store.has(self.bid))

    def test_rejects_odd_uploader_names(self):
        with self.assertRaises(BlobError):
            self.store.write_chunk(self.bid, 0, self.data, len(self.data), "../escape")


if __name__ == "__main__":
    unittest.main()

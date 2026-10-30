"""Setup and training helpers: GPU detection and PyTorch build choice, saved settings, CSV tables and model
files. No GPU or network needed.

`vibehash setup`, `testnet` and `train` against a live node are in test_network.py.
"""
import os
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
        self.assertEqual(tags(nvidia("RTX 4070 SUPER", (13, 4), (8, 9)))[:3], ["cu132", "cu130", "cu129"])
        self.assertEqual(tags(nvidia("RTX 4070 SUPER", (12, 8), (8, 9)))[0], "cu128")  # CUDA 13 needs a newer driver
        self.assertEqual(tags(nvidia("GTX 1060", (13, 0), (6, 1)))[0], "cu126")  # Pascal left the 12.8+ builds
        self.assertEqual(tags(nvidia("RTX 5090", (12, 8), (12, 0))), ["cu128"])  # Blackwell needs 12.8+
        self.assertEqual(tags(nvidia("RTX 5090", (12, 6), (12, 0))), [])
        self.assertIn("nvidia.com", builds_for(nvidia("RTX 5090", (12, 6), (12, 0)))[1])
        self.assertIn("driver", builds_for(Gpu("nvidia", "RTX 3060"))[1])  # Windows lists it, but no nvidia-smi

    def test_other_vendors(self):
        self.assertEqual(tags(Gpu("intel", "Intel(R) Arc(TM) Graphics")), ["xpu"])
        self.assertEqual(tags(Gpu("intel", "Intel(R) UHD Graphics 620")), [])
        self.assertEqual(tags(Gpu("apple", "Apple silicon GPU")), [None])  # the plain PyPI build
        self.assertEqual(tags(Gpu("amd", "Radeon RX 7900 XTX", rocm=(6, 4))), ["rocm6.4", "rocm6.3", "rocm6.2"])
        self.assertIn("ROCm", builds_for(Gpu("amd", "Radeon RX 7900 XTX"))[1])

    def test_choose_prefers_nvidia_and_skips_unpublished_builds(self):
        gpus = [Gpu("intel", "Intel(R) Arc(TM) Graphics"), nvidia("RTX 4070 SUPER", (13, 4), (8, 9))]
        gpu, build, why = choose(gpus, available=lambda b: b.tag != "cu132")
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

    def test_normalise_url(self):
        self.assertEqual(normalise_url("192.168.1.20"), "http://192.168.1.20:8545")
        self.assertEqual(normalise_url("node.example:9000/"), "http://node.example:9000")
        self.assertEqual(normalise_url("https://node.example"), "https://node.example")


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

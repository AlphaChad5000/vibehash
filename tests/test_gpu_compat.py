"""Guards for GPU kinds this project can't test on (docs/GPUS.md): float32 matmuls that stay full float32 whatever
else in the process touched PyTorch's TF32 settings, float64 work kept off devices that have none (Apple's MPS,
Intel Arc A-series), Apple's usable memory, AMD's ROCm build named as such, and an Evo engine refusing a model on a
GPU other than the current one. Each runs on any machine with PyTorch; the CUDA-only parts skip without a GPU."""
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from vibehash import compute
from vibehash.model import Dataset, ModelSpec

HAVE_TORCH = compute.torch_available()
if HAVE_TORCH:
    import torch


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch")
class PinPrecisionTests(unittest.TestCase):
    def setUp(self):
        if not hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            self.skipTest("PyTorch before 2.9 has only the older allow_tf32 settings")
        self.addCleanup(compute.pin_precision)

    def assert_full_float32(self):
        b = torch.backends
        self.assertEqual(b.cuda.matmul.fp32_precision, "ieee")
        self.assertEqual(b.mkldnn.matmul.fp32_precision, "ieee")  # oneDNN: Intel GPUs and CPUs
        self.assertFalse(b.cuda.matmul.allow_tf32)  # and reading the older flag doesn't raise
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")

    def test_tf32_turned_on_through_the_newer_api_is_turned_off_again(self):
        compute.pin_precision()
        # A library using PyTorch 2.9's flags: TF32 (and bf16 in oneDNN) for every float32 matmul from here on, and
        # reading allow_tf32 now raises (the older and newer settings disagree)
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.mkldnn.matmul.fp32_precision = "bf16"
        compute.pin_precision()
        self.assert_full_float32()

    def test_set_float32_matmul_precision_high_is_undone(self):
        torch.set_float32_matmul_precision("high")  # TF32 on cuBLAS and oneDNN alike
        compute.pin_precision()
        self.assert_full_float32()

    def test_intel_gpus_tf32_switch_is_turned_off_on_builds_that_have_one(self):
        mkldnn = SimpleNamespace(allow_tf32=True)
        with mock.patch.object(torch.backends, "mkldnn", mkldnn), \
                mock.patch.object(torch.xpu, "_is_compiled", return_value=True):
            compute.pin_precision()
        self.assertIs(mkldnn.allow_tf32, False)
        mkldnn = SimpleNamespace(allow_tf32=None)
        with mock.patch.object(torch.backends, "mkldnn", mkldnn), \
                mock.patch.object(torch.xpu, "_is_compiled", return_value=False):
            compute.pin_precision()
        self.assertIsNone(mkldnn.allow_tf32)  # untouched: setting it there only warns there's no Intel GPU support

    def test_a_segment_pins_it_before_training(self):
        backend = compute.TorchBackend("cpu")
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        spec = ModelSpec(4, (8,), 3, "float32")
        data = Dataset.synthetic(spec, 1, n_train=32, n_val=4)
        backend.train_segment(spec, data, spec.init_params(1), 3, 0, 1, 0.1, 8)
        self.assert_full_float32()


class Float64Tests(unittest.TestCase):
    def test_which_devices_have_float64(self):
        self.assertTrue(compute.has_float64("cpu"))
        self.assertTrue(compute.has_float64("cuda:0"))  # NVIDIA and AMD (ROCm's "cuda")
        self.assertFalse(compute.has_float64("mps"))

    @unittest.skipUnless(HAVE_TORCH, "needs PyTorch")
    def test_intel_says_per_device(self):
        for props, want in ((SimpleNamespace(has_fp64=False), False),  # Arc A770
                            (SimpleNamespace(has_fp64=True), True),  # Data Center GPU Max
                            (SimpleNamespace(), False)):  # a PyTorch that doesn't say
            with mock.patch.object(torch.xpu, "get_device_properties", return_value=props) as got:
                self.assertEqual(compute.has_float64("xpu:1"), want)
                got.assert_called_with(1)
        with mock.patch.object(torch.xpu, "get_device_properties", side_effect=RuntimeError("no driver")):
            self.assertFalse(compute.has_float64("xpu"))


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch")
class DeviceFactsTests(unittest.TestCase):
    def test_apple_memory_is_what_metal_lets_a_process_use(self):
        backend = compute.TorchBackend("cpu")
        backend.name, backend.device = "mps", "mps"
        backend.torch = SimpleNamespace(mps=SimpleNamespace(recommended_max_memory=lambda: 21 * 2**30))
        self.assertEqual(backend.memory_mb(), 21 * 1024)
        backend.torch = SimpleNamespace(mps=SimpleNamespace())  # an older PyTorch without it: the RAM, as before
        self.assertIsNone(backend.memory_mb())

    def test_amd_cards_are_named_rocm(self):
        self.assertEqual(compute.gpu_kind(SimpleNamespace(version=SimpleNamespace(hip="6.4.43482"))), "ROCm")
        self.assertEqual(compute.gpu_kind(SimpleNamespace(version=SimpleNamespace(hip=None, cuda="12.6"))), "CUDA")
        self.assertEqual(compute.gpu_kind(torch), "ROCm" if torch.version.hip else "CUDA")


@unittest.skipUnless(HAVE_TORCH and torch.cuda.is_available(), "needs a CUDA (or ROCm) GPU")
class SameGpuTests(unittest.TestCase):
    def test_an_engine_refuses_a_model_on_another_gpu(self):
        from vibehash import es_fast
        es_fast._same_gpu(torch.device("cpu"))
        es_fast._same_gpu(torch.device("cuda"))
        es_fast._same_gpu(torch.device("cuda", torch.cuda.current_device()))
        other = torch.cuda.current_device() + 1  # (needn't exist: refused before anything touches it)
        with self.assertRaisesRegex(RuntimeError, f"CUDA_VISIBLE_DEVICES={other}"):
            es_fast._same_gpu(torch.device("cuda", other))


if __name__ == "__main__":
    unittest.main()

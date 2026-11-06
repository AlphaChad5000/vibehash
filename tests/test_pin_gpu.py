import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from vibehash import cli


class FakeCuda:
    def __init__(self, started):
        self.started = started

    def is_initialized(self):
        return self.started


class PinGpuTests(unittest.TestCase):
    """`worker --device cuda:N` sees GPU N alone (cli.pin_gpu): Evo's engines put their CUDA graphs on the current GPU,
    which is GPU 0 unless the process sees only the one it was given."""

    def pin(self, device, env=None, cuda_started=None):
        a = SimpleNamespace(device=device)
        environ = {} if env is None else {"CUDA_VISIBLE_DEVICES": env}
        modules = dict(sys.modules)
        if cuda_started is None:
            modules.pop("torch", None)
        else:
            modules["torch"] = SimpleNamespace(cuda=FakeCuda(cuda_started))
        with mock.patch.dict(os.environ, environ, clear=True), mock.patch.dict(sys.modules, modules, clear=True):
            cli.pin_gpu(a)
            return a.device, os.environ.get("CUDA_VISIBLE_DEVICES")

    def test_a_numbered_gpu_is_the_only_one_seen(self):
        self.assertEqual(self.pin("cuda:2"), ("cuda", "2"))
        self.assertEqual(self.pin("cuda:0"), ("cuda", "0"))

    def test_a_list_already_set_is_indexed_into(self):
        self.assertEqual(self.pin("cuda:1", env="4,5"), ("cuda", "5"))
        with self.assertRaises(SystemExit):
            self.pin("cuda:2", env="4,5")

    def test_other_devices_are_left_alone(self):
        for device in ("cuda", "auto", "cpu", "xpu", "mps", None):
            self.assertEqual(self.pin(device, env="3"), (device, "3"))

    def test_too_late_once_pytorch_has_started_cuda(self):
        self.assertEqual(self.pin("cuda:1", cuda_started=True), ("cuda:1", None))
        self.assertEqual(self.pin("cuda:1", cuda_started=False), ("cuda", "1"))


if __name__ == "__main__":
    unittest.main()

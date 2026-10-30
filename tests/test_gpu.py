"""float32 / deep-model jobs, and CPU <-> GPU cross-verification.

The cross-device tests run on whatever PyTorch devices this machine has
(CUDA, Intel XPU, Apple MPS, or PyTorch's CPU) and skip if PyTorch isn't
installed.
"""
import json
import unittest

import numpy as np

from vibehash.compute import NumpyBackend, TorchBackend, list_devices, self_test, torch_available
from vibehash.coordinator import CoordinatorConfig
from vibehash.crypto import KeyPair
from vibehash.ledger import Chain
from vibehash.model import ModelSpec, params_hash, relative_distance, train_segment
from vibehash.protocol import Protocol
from vibehash.tasks import SYN_PER_VIBE, InvalidJob, TrainingJob, Verdict, WorkUnit
from vibehash.verification import UPDATE_TOLERANCE, replay_openings
from vibehash.worker import HardwareProfile, Worker

VIBE = SYN_PER_VIBE
RELIABLE = HardwareProfile("desktop", 1e9, 8192, reliability=1.0)  # simulated profiles are sized for the tiny default model


def gpu_job(**kw) -> TrainingJob:
    defaults = dict(spec=ModelSpec(8, (128, 128), 4, "float32"), lr=0.05, batch_size=128, tolerance=1e-4, units_per_round=8)
    return TrainingJob("gpu-job", "req", budget=kw.pop("budget", 0), fee_per_round=10 * VIBE, **(defaults | kw))


class Float32JobTests(unittest.TestCase):
    def test_spec_round_trip_and_back_compat(self):
        job = gpu_job()
        again = TrainingJob.from_spec_json(job.spec_json(), "req", 0)
        self.assertEqual(again.spec, job.spec)
        old = json.loads(TrainingJob("old", "req", 0, 0).spec_json())
        del old["precision"]
        self.assertEqual(TrainingJob.from_spec_json(json.dumps(old), "req", 0).spec.precision, "float64")

    def test_rejects_bad_precision_and_oversized_models(self):
        with self.assertRaises(InvalidJob):
            gpu_job(spec=ModelSpec(8, 64, 4, "float16"))
        with self.assertRaises(InvalidJob):
            gpu_job(spec=ModelSpec(4096, (8192,) * 6, 4, "float32"))  # ~370M params, 1.4 GB: over the 1 GiB checkpoint cap

    def test_deep_float64_matches_original_single_layer_layout(self):
        # One hidden layer must keep the original parameter layout, so existing chains stay valid.
        spec = ModelSpec(8, 32, 4)
        self.assertEqual(spec.n_params, 8 * 32 + 32 + 32 * 4 + 4)
        self.assertEqual(ModelSpec(8, (32,), 4).n_params, spec.n_params)

    def test_float32_weights_are_exactly_float32(self):
        job = gpu_job()
        theta = train_segment(job.spec, job.data, job.spec.init_params(1), 7, 0, 5, job.lr, job.batch_size)
        np.testing.assert_array_equal(theta, theta.astype(np.float32).astype(np.float64))

    def test_float32_network_round(self):
        workers = [Worker(f"w{i}", KeyPair.generate(f"w{i}".encode()), RELIABLE, b)
                   for i, b in enumerate(["honest", "honest", "honest", "lazy"])]
        req = KeyPair.generate(b"req")
        chain = Chain({req.address: 1000 * VIBE}, stakes={w.address: 10 * VIBE for w in workers})
        proto = Protocol(chain, workers, CoordinatorConfig(), seed=4)
        job = gpu_job(budget=100 * VIBE)
        proto.post_job(req, job)
        first = proto.run_round()
        last = proto.run_round()
        self.assertIn(Verdict.FAILED_REPLAY, {r.verdict for r in first.results if r.worker == workers[3].address})
        for w in workers[:3]:
            self.assertIn(w.address, first.worker_flops)
        self.assertLess(last.val_loss, first.val_loss + 1e-9)
        self.assertTrue(proto.chain.validate())


@unittest.skipUnless(torch_available(), "PyTorch not installed")
class CrossDeviceTests(unittest.TestCase):
    """The same unit trained on each available PyTorch device must verify on the CPU, and vice versa."""

    @classmethod
    def setUpClass(cls):
        cls.job = gpu_job(spec=ModelSpec(8, (256, 256), 4, "float32"), batch_size=256)
        cls.start = cls.job.spec.init_params(cls.job.init_seed)
        cls.unit = WorkUnit.make(cls.job, 1, 0, params_hash(cls.start))
        # Every GPU that passes its self-test. With none usable, PyTorch's own CPU device still
        # exercises the PyTorch code path against the NumPy reference.
        gpus = [d.id for d in list_devices() if d.id != "cpu"]
        cls.broken = {g: p for g in gpus if (p := self_test(TorchBackend(g)))}
        cls.devices = [g for g in gpus if g not in cls.broken] or ["cpu"]
        print(f"\n  cross-device tests on: {cls.devices}" +(f"; skipped (failed self-test): {list(cls.broken)}" if cls.broken else ""))
        cls.cpu = NumpyBackend()

    def backend(self, device):
        return TorchBackend(device)

    def test_segment_divergence_is_within_tolerance(self):
        job, c = self.job, self.job.checkpoint_every
        ref = train_segment(job.spec, job.data, self.start, self.unit.seed, 0, c, job.lr, job.batch_size, self.cpu)
        for device in self.devices:
            with self.subTest(device=device):
                got = train_segment(job.spec, job.data, self.start, self.unit.seed, 0, c, job.lr, job.batch_size, self.backend(device))
                d = relative_distance(got, ref)
                moved = relative_distance(ref, self.start)
                print(f"\n  {device}: replay divergence {d:.2e} vs one segment's update {moved:.2e} "
                      f"({d / moved:.1e} of the update)")
                self.assertLess(d, job.tolerance)
                self.assertLess(d, UPDATE_TOLERANCE * moved / 10)  # 10x inside the update check too

    def test_gpu_work_verifies_on_cpu_and_back(self):
        # lr 1e-6: updates of a few float32 ulps, where rounding is largest next to the update itself.
        tiny = gpu_job(spec=self.job.spec, batch_size=self.job.batch_size, lr=1e-6)
        for job, unit in ((self.job, self.unit), (tiny, WorkUnit.make(tiny, 1, 0, params_hash(self.start)))):
            q = job.n_segments
            for device in self.devices:
                gpu = self.backend(device)
                for trainer, checker in ((gpu, self.cpu), (self.cpu, gpu)):
                    with self.subTest(lr=job.lr, device=device, trainer=trainer.name, checker=checker.name):
                        w = Worker("w", KeyPair.generate(b"w"), RELIABLE, fp_noise=0, backend=trainer)
                        w.compute(unit, job, self.start)
                        rv = w.reveal(unit, job, b"beacon", q)
                        ok, bad = replay_openings(job, unit, rv.openings, list(range(q)), backend=checker)
                        self.assertTrue(ok, f"segment {bad} failed")

    def test_gpu_cheater_still_caught_by_cpu(self):
        gpu = self.backend(self.devices[0])
        w = Worker("w", KeyPair.generate(b"s"), RELIABLE, "shortcut", fp_noise=0, backend=gpu)
        w.compute(self.unit, self.job, self.start)
        q = self.job.n_segments
        rv = w.reveal(self.unit, self.job, b"beacon", q)
        ok, bad = replay_openings(self.job, self.unit, rv.openings, list(range(q)), backend=self.cpu)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()

"""The terminal mining dashboard: every frame fits the terminal, whatever its size, state or character set."""
import logging
import os
import re
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from vibehash import minerui
from vibehash.client import WorkerStats
from vibehash.model import ModelSpec
from vibehash.tasks import SYN_PER_VIBE, TrainingJob

ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
JOB = TrainingJob("demo-1", "req", 0, 10, spec=ModelSpec(8, (512, 512), 4, "float32"), tolerance=1e-4)


def rig(busy=True, phase="compute", rnd=42):
    stats = WorkerStats(round=rnd, phase=phase, phase_ends=time.time() + 5, height=41, job=JOB, flops_per_sec=2.9e11,
                        activity="training unit 3 of 7" if busy else "waiting", busy=busy, assigned=7, trained=3,
                        committed=5, rounds=21, units=37, checks=112, training_flops=4.3e12)
    return SimpleNamespace(stats=stats, backend=SimpleNamespace(describe=lambda: "NVIDIA GeForce RTX 4070 SUPER (CUDA)"),
                           keys=SimpleNamespace(address="vibe17b0d8bce54f6eab50df24fd44e5a408a9525"))


class DashboardTests(unittest.TestCase):
    def frames(self, dash, size):
        with mock.patch("shutil.get_terminal_size", return_value=os.terminal_size(size)):
            return [dash.frame() for _ in range(3)]

    def dashboard(self, n=1, glyphs=minerui.GLYPHS, **kw):
        dash = minerui.MinerDashboard([(f"worker-{i}", rig(**kw)) for i in range(n)], "http://127.0.0.1:8545")
        dash.g, dash.chain_id = glyphs, "vibehash-local-123558"
        for r in dash.rigs:
            r.start, r.total, r.balance, r.stake, r.reputation = 0, 95 * SYN_PER_VIBE, 85 * SYN_PER_VIBE, 10 * SYN_PER_VIBE, 0.9
        dash.note("round 41: verified 7 tasks")
        dash.note("POST /commit: not in compute phase", logging.WARNING)
        return dash

    def test_every_line_fits_the_terminal(self):
        for size in ((40, 20), (66, 34), (120, 50)):
            for n in (1, 3):
                for glyphs in (minerui.GLYPHS, minerui.ASCII):
                    for state in ({}, {"busy": False, "phase": "idle", "rnd": None}, {"phase": "verify"}):
                        with self.subTest(size=size, workers=n, ascii=glyphs is minerui.ASCII, **state):
                            dash = self.dashboard(n, glyphs, **state)
                            dash.popup = (time.time(), 4 * SYN_PER_VIBE)  # the widest moment: a payout label
                            for frame in self.frames(dash, size):
                                lines = frame.split("\n")
                                self.assertLessEqual(len(lines), size[1])
                                for line in lines:
                                    self.assertLessEqual(len(ESCAPE.sub("", line)), size[0], line)

    def test_shows_the_work_and_the_money(self):
        text = ESCAPE.sub("", self.frames(self.dashboard(), (120, 50))[-1])
        for want in ("ROUND 42", "COMPUTE", "training unit 3 of 7", "RTX 4070 SUPER", "290.00 GFLOP/s", "37 units",
                     "112 checks", "85.00 VIBE", "+95.00 VIBE this session", "POST /commit"):
            self.assertIn(want, text)
        self.assertEqual(text.count(minerui.GLYPHS["ore"] * 4), 1)  # the cart is full: 5 committed, 4 slots

    def test_ascii_fallback_is_pure_ascii(self):
        dash = self.dashboard(glyphs=minerui.ASCII)
        dash.popup = (time.time(), SYN_PER_VIBE)
        for frame in self.frames(dash, (100, 40)):
            ESCAPE.sub("", frame).encode("ascii")

    def test_off_when_not_a_terminal(self):
        with mock.patch("sys.stdout.isatty", return_value=False):
            self.assertFalse(minerui.supported())
        self.assertFalse(minerui.supported(plain=True))

    def test_summary(self):
        self.assertIn("Mined 111 units", self.dashboard(3).summary())


if __name__ == "__main__":
    unittest.main()

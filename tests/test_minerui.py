"""The terminal mining dashboard: every frame fits the terminal, whatever its size, state or character set, and its
job selector follows the keys."""
import io
import logging
import os
import re
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from vibehash import minerui, termui
from vibehash.client import NodeClient, WorkerStats
from vibehash.model import ModelSpec
from vibehash.tasks import SYN_PER_VIBE, TrainingJob

ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
JOB = TrainingJob("demo-1", "req", 0, 10, spec=ModelSpec(8, (512, 512), 4, "float32"), tolerance=1e-4)
TONIGHT = ["fruit-demo", "pokemon-pun-starter-2d950a", "think-shorter-pilot"]  # the public node's jobs, 2026-11-02
MANY = [f"job-{i:02d}-" + "x" * (40 + i) for i in range(12)]  # more than the number keys reach, with long ids


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
        pickers = [([], None, False), (TONIGHT, None, True), (TONIGHT, "think-shorter-pilot", True),
                   (MANY, MANY[-1], True), (["gone-job"], "spent-job", True)]
        for size in ((40, 20), (66, 34), (120, 50)):
            for n in (1, 3):
                for glyphs in (minerui.GLYPHS, minerui.ASCII):
                    for state in ({}, {"busy": False, "phase": "idle", "rnd": None}, {"phase": "verify"}):
                        for jobs, chosen, keyed in pickers:
                            with self.subTest(size=size, workers=n, ascii=glyphs is minerui.ASCII, jobs=len(jobs),
                                              chosen=chosen, **state):
                                dash = self.dashboard(n, glyphs, **state)
                                dash.popup = (time.time(), 4 * SYN_PER_VIBE)  # the widest moment: a payout label
                                dash.jobs, dash.keyed = jobs, keyed
                                dash.choice.set(chosen)
                                for frame in self.frames(dash, size):
                                    lines = frame.split("\n")
                                    self.assertLessEqual(len(lines), size[1])
                                    for line in lines:
                                        self.assertLessEqual(len(ESCAPE.sub("", line)), size[0], line)
                                    if glyphs is minerui.ASCII:
                                        ESCAPE.sub("", frame).encode("ascii")
                                    picker = next(x for x in ESCAPE.sub("", frame).split("\n") if "job:" in x)
                                    i = ["auto", *jobs, chosen].index(chosen or "auto")
                                    self.assertIn(f"[{i} " if i < 10 else f"[{chosen[:3]}", picker)  # the choice shows

    def test_the_selector_fits_any_width_and_always_shows_the_choice(self):
        for glyphs in (minerui.GLYPHS, minerui.ASCII):
            for options in (["auto"], ["auto", *TONIGHT], ["auto", *MANY]):
                for sel in range(len(options)):
                    for width in range(30, 160, 7):
                        with self.subTest(ascii=glyphs is minerui.ASCII, n=len(options), sel=sel, width=width):
                            segs = minerui.selector(glyphs, options, sel, width)
                            text = "".join(t for t, _ in segs)
                            self.assertLessEqual(len(text), width, text)
                            self.assertTrue(text.startswith("job:"))
                            label = f"{sel} " if sel < 10 else ""
                            chosen = re.search(r"\[([^\]]*)\]", text)[1]  # in its brackets, shortened only if it must be
                            self.assertTrue(chosen.startswith(label + options[sel][:3]), text)
                            shown = sum(t in (" [", "  ") for t, _ in segs)  # each option starts with one of these
                            more = re.search(r"\+(\d+) more", text)
                            self.assertEqual(shown + (int(more[1]) if more else 0), len(options), text)
        wide = "".join(t for t, _ in minerui.selector(minerui.GLYPHS, ["auto", *TONIGHT], 0, 120))
        self.assertEqual(wide, "job: [0 auto]  1 fruit-demo   2 pokemon-pun-starter-2d950a   3 think-shorter-pilot "
                               "  tab or 0-9 to pick")
        narrow = "".join(t for t, _ in minerui.selector(minerui.GLYPHS, ["auto", *TONIGHT], 3, 38))
        self.assertEqual(narrow, "job: [3 think-shorter-pilot] +3 more")
        no_keys = "".join(t for t, _ in minerui.selector(minerui.ASCII, ["auto", "fruit-demo"], 1, 120, keys=False))
        self.assertEqual(no_keys, "job:  0 auto  [1 fruit-demo]")

    def test_keys_move_the_selector(self):
        """Tab, j/k, the arrows (as a Windows console and a POSIX terminal send them), a, and the number keys."""
        dash = self.dashboard(3)  # three rigs, as `testnet` shows them: one selector for all of them
        dash.jobs = TONIGHT
        dash.feed.clear()

        def press(raw, windows):
            keys = termui.windows_keys(raw) if windows else termui.ansi_keys(raw)[0]
            for k in keys:
                dash.press(k)
            return dash.choice.job_id
        for windows in (True, False):
            arrow = {True: {"right": "\xe0M", "left": "\xe0K", "right0": "\x00M", "shift-tab": "\x00\x0f"},
                     False: {"right": "\x1b[C", "left": "\x1b[D", "right0": "\x1bOC", "shift-tab": "\x1b[Z"}}[windows]
            with self.subTest(windows=windows):
                self.assertIsNone(press("a", windows))
                self.assertEqual(press("\t", windows), "fruit-demo")
                self.assertEqual(press(arrow["right"], windows), "pokemon-pun-starter-2d950a")
                self.assertEqual(press(arrow["right0"], windows), "think-shorter-pilot")
                self.assertIsNone(press("\t", windows))  # round again to auto
                self.assertEqual(press(arrow["left"], windows), "think-shorter-pilot")
                self.assertEqual(press(arrow["shift-tab"], windows), "pokemon-pun-starter-2d950a")
                self.assertEqual(press("k", windows), "fruit-demo")
                self.assertEqual(press("j", windows), "pokemon-pun-starter-2d950a")
                self.assertEqual(press("1", windows), "fruit-demo")
                self.assertEqual(press("3", windows), "think-shorter-pilot")
                self.assertEqual(press("7", windows), "think-shorter-pilot")  # no job 7: nothing changes
                self.assertEqual(press("x\r", windows), "think-shorter-pilot")  # other keys: nothing either
                self.assertIsNone(press("0", windows))
                self.assertEqual(press("2", windows), "pokemon-pun-starter-2d950a")
                self.assertIsNone(press("a", windows))
                self.assertEqual({r.client.choice.job_id for r in dash.rigs}, {None})
                self.assertEqual(press("\t\t", windows), "pokemon-pun-starter-2d950a")  # two keys in one read
                self.assertTrue(all(r.client.choice is dash.choice for r in dash.rigs))
                press("a", windows)
        said = [t for _, _, t in dash.feed]
        self.assertEqual(said[:3], ["mining fruit-demo only", "mining pokemon-pun-starter-2d950a only",
                                    "mining think-shorter-pilot only"])
        self.assertIn("mining every job (auto)", said)
        self.assertEqual(len(said), len([s for s in said if s.startswith("mining ")]))  # once a change, not once a rig

    def test_keys_on_the_wire(self):
        """Both platforms' raw codes: special keys come in two characters on Windows, as escape sequences on POSIX
        (split across reads, too); Ctrl+C read as a character still stops the dashboard."""
        self.assertEqual(termui.windows_keys("\t\xe0M\x00K\xe0Hj\x00;a"), ["tab", "right", "left", "up", "j", "a"])
        self.assertEqual(termui.ansi_keys("\t\x1b[C\x1bOD\x1b[1;5Ck"), (["tab", "right", "left", "k"], ""))
        self.assertEqual(termui.ansi_keys("2\x1b["), (["2"], "\x1b["))  # the rest arrives in the next read
        self.assertEqual(termui.ansi_keys("\x1b[" + "D"), (["left"], ""))
        self.assertEqual(termui.ansi_keys("\x1bx"), (["escape", "x"], ""))
        for decode in (termui.windows_keys, termui.ansi_keys):
            with self.assertRaises(KeyboardInterrupt):
                decode("\t\x03")

    def test_no_keys_without_a_terminal(self):
        """Piped stdin (or --plain, which shows no dashboard at all): no key handling, and the keys aren't offered."""
        with termui.keypresses(io.StringIO("\t")) as read:
            self.assertIsNone(read)
        dash = self.dashboard()
        dash.jobs = TONIGHT
        text = ESCAPE.sub("", self.frames(dash, (120, 50))[-1])
        self.assertIn("job: [0 auto]  1 fruit-demo", text)
        self.assertNotIn("tab", text)

    def test_the_job_list_comes_from_the_node_safely(self):
        """/network's jobs, without the finished ones, sorted so the number keys stay put as the training job moves to
        the top of the node's list each round, and without ids no real job has: they'd reach the terminal."""
        jobs = [{"id": "think-shorter-pilot", "status": "training"}, {"id": "fruit-demo", "status": "open"},
                {"id": "old-job", "status": "finished"}, {"id": "evil\x1b[2J\x1b]0;pwned\x07", "status": "open"},
                {"id": "Caps", "status": "open"}, {"id": "two-lines\n", "status": "open"}, {"id": 7, "status": "open"},
                "junk", {"status": "open"}]
        api = NodeClient("http://127.0.0.1:9")
        with mock.patch.object(api, "get", return_value={"jobs": jobs}):
            self.assertEqual(api.open_jobs(), ["fruit-demo", "think-shorter-pilot"])
        for reply in ([], {"jobs": "nope"}, {}):
            with mock.patch.object(api, "get", return_value=reply):
                self.assertEqual(api.open_jobs(), [])
        dash = self.dashboard()
        stop = mock.Mock(is_set=mock.Mock(side_effect=[False, True]))
        dash.api = SimpleNamespace(open_jobs=lambda: ["fruit-demo", "think-shorter-pilot"])
        dash._poll_jobs(stop)
        self.assertEqual(dash.jobs, ["fruit-demo", "think-shorter-pilot"])
        stop.wait.assert_called_once_with(minerui.JOBS_EVERY)

    def test_a_send_from_the_wallet_is_not_a_slash(self):
        """Sending a friend VIBE drops the balance too: it showed as "lost 97.42 VIBE: slashed" on test night. A
        wallet's own transactions raise its nonce, and a slash doesn't."""
        dash, v = self.dashboard(), SYN_PER_VIBE
        r = dash.rigs[0]
        r.start = r.total = None
        accounts = iter([{"balance": 100 * v, "stake": 10 * v, "reputation": 1.0, "nonce": 3},
                         {"balance": 3 * v, "stake": 10 * v, "reputation": 1.0, "nonce": 4},  # sent 100, paid 3
                         {"balance": 1 * v, "stake": 10 * v, "reputation": 1.0, "nonce": 4}])  # slashed 2
        dash.api = SimpleNamespace(status=lambda: {"chain_id": "c"}, account=lambda addr: next(accounts))
        polls = []
        stop = SimpleNamespace(is_set=lambda: len(polls) >= 3, wait=polls.append)
        dash.feed.clear()
        dash._poll(stop)
        sent, slashed = [(lvl, t) for _, lvl, t in dash.feed]
        self.assertEqual(sent[0], logging.INFO)
        self.assertIn("sent 97", sent[1])
        self.assertIn("not a slash", sent[1])
        self.assertEqual(slashed[0], logging.ERROR)
        self.assertIn("lost 2", slashed[1])
        self.assertEqual((r.sent, r.earned), (97 * v, -2 * v))  # the send isn't counted against the session

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

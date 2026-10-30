"""The terminal training dashboard: every frame fits, whatever the job's state, the node's health or the character
set; the job is followed politely; and the classic log lines tell the same story without a terminal."""
import contextlib
import io
import os
import re
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock

from vibehash import cli, trainui
from vibehash.client import NodeError
from vibehash.tasks import SYN_PER_VIBE as VIBE

ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
SGR = re.compile(r"\x1b\[[0-9;]*m")
JOB = "fruit-3fa2"


def history(n, fee=10, units=12):
    """n rounds of a job learning: accuracy climbing from a blind guess (4 classes), loss falling."""
    return [{"height": 40 + 2 * i, "accepted_units": units - (i % 4 == 3), "val_loss": round(1.25 * 0.78 ** i + 0.08, 4),
             "val_acc": round(0.31 + 0.66 * (1 - 0.72 ** (i + 1)), 4)} for i in range(n)]


def job_reply(n=0, budget=100, fee=10, units=12, model="full", rounds=None, escrow=None):
    rounds = history(n, fee, units) if rounds is None else rounds
    spent = sum(fee * VIBE * r["accepted_units"] // units for r in rounds)
    reply = {"spec": {"id": JOB, "fee_per_round": fee * VIBE, "units_per_round": units, "precision": "float32",
                      "model": {"n_in": 5, "n_hidden": [128, 128], "n_out": 4}, "data": {"kind": "npz", "sha256": "ab" * 32}},
             "owner": "vibe1requester", "escrow": max(0, budget * VIBE - spent) if escrow is None else escrow, "rounds": rounds}
    if model == "full":
        reply["model"] = {"height": rounds[-1]["height"] if rounds else 0, "bytes": 4 * 17540, "sha256": "cd" * 32}
    elif model == "null":
        reply["model"] = None
    elif model == "partial":
        reply["model"] = {"height": 0, "bytes": 4 * 17540}  # no sha256
    return reply


def status(rnd=60, phase="compute", job_id=JOB, height=59, ends_in=12.0):
    return {"chain_id": "vibehash-local-123558", "height": height, "phase": phase, "phase_ends": time.time() + ends_in,
            "round": rnd, "job_id": job_id, "min_fee": 10_000}


class FakeNode:
    """A node's read API, scripted: `job` None answers 404, `down` makes every request fail with it."""

    def __init__(self, job=None, st=None):
        self.url, self.job, self.status, self.down, self.calls = "http://127.0.0.1:8545", job, st or status(), None, []

    def get(self, path):
        self.calls.append(path)
        if self.down is not None:
            raise self.down
        if path == "/status":
            return self.status
        if path == f"/job/{JOB}":
            if self.job is None:
                raise NodeError(f"GET {path}: not found: '{JOB}'", 404)
            return self.job
        raise NodeError(f"GET {path}: not found", 404)


UNREACHABLE = NodeError("can't reach node at http://127.0.0.1:8545: [Errno 111] Connection refused")
BUSY = NodeError("GET /status: rate limited", 429)


def watching(node, budget=100 * VIBE, polls=1, **kw):
    watch = trainui.JobWatch(JOB, node.url, budget=budget, posted=True, api=node, **kw)
    for _ in range(polls):
        watch._job_at = 0  # every poll reads the job too
        watch.poll()
    return watch


def states():
    """(name, JobWatch) for every state the dashboard draws."""
    out = []
    node = FakeNode()
    out.append(("pending", watching(node)))
    node = FakeNode(job_reply(0), status(rnd=41, job_id="demo-3"))
    out.append(("no rounds yet", watching(node)))
    node = FakeNode(job_reply(3))
    w = watching(node)
    node.job = job_reply(4)
    w._job_at = 0
    w.poll()  # a round lands: the blade glows, the chart's newest column flashes
    out.append(("training", w))
    node = FakeNode(job_reply(6), status(rnd=61, phase="verify", job_id="demo-3"))
    out.append(("waiting its turn", watching(node)))
    node = FakeNode(job_reply(5))
    w = watching(node)
    node.down = UNREACHABLE
    w.poll()
    out.append(("node unreachable", w))
    node = FakeNode(job_reply(5))
    w = watching(node)
    node.down = BUSY
    w.poll()
    out.append(("node busy", w))
    node = FakeNode(job_reply(9))
    w = watching(node)
    node.job = job_reply(10, escrow=0)
    w._job_at = 0
    w.poll()
    out.append(("finished", w))
    node = FakeNode(job_reply(rounds=history(240), budget=5000, fee=10))
    out.append(("many rounds", watching(node, budget=None)))
    node = FakeNode(job_reply(rounds=history(240), escrow=0))
    out.append(("finished, many rounds", watching(node, budget=None)))
    return out


def frames(dash, size, n=4):
    with mock.patch("shutil.get_terminal_size", return_value=os.terminal_size(size)):
        return [dash.frame() for _ in range(n)]


def text_of(watch, size=(120, 50), glyphs=trainui.GLYPHS):
    dash = trainui.TrainDashboard(watch)
    dash.g = glyphs
    return ESCAPE.sub("", frames(dash, size)[-1])


def _ascii(watch):
    dash = trainui.TrainDashboard(watch)
    dash.g = trainui.ASCII
    return dash


class Scripted(FakeNode):
    """A node whose replies follow a timeline on a fake clock: [(seconds in, job, status, down), ...]."""

    def __init__(self, clock, timeline):
        super().__init__()
        self.clock, self.timeline, self.start = clock, timeline, clock[0]

    def get(self, path):
        _, self.job, self.status, self.down = [e for e in self.timeline if e[0] <= self.clock[0] - self.start][-1]
        return super().get(path)


class FakeClock:
    """time.time and time.sleep on a clock that only moves when slept on."""

    def __init__(self):
        self.now = [1_793_400_000.0]

    def time(self):
        return self.now[0]

    def sleep(self, s):
        self.now[0] += s

    def patch(self):
        """The clock, and no jitter in the backing off, so a story always plays out the same."""
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.multiple(trainui.time, time=self.time, sleep=self.sleep))
        stack.enter_context(mock.patch.object(trainui.random, "uniform", lambda a, b: a))
        return stack


class DashboardTests(unittest.TestCase):
    def test_every_line_fits_the_terminal(self):
        for name, watch in states():
            for size in ((40, 20), (66, 34), (120, 50), (52, 14)):
                for glyphs in (trainui.GLYPHS, trainui.ASCII):
                    with self.subTest(state=name, size=size, ascii=glyphs is trainui.ASCII):
                        dash = trainui.TrainDashboard(watch)
                        dash.g = glyphs
                        for frame in frames(dash, size):
                            lines = frame.split("\n")
                            self.assertLessEqual(len(lines), size[1])
                            for line in lines:
                                self.assertLessEqual(len(ESCAPE.sub("", line)), size[0], line)

    def test_shows_the_job_the_network_and_the_money(self):
        (_, training), = [s for s in states() if s[0] == "training"]
        text = text_of(training)
        for want in ("model forge  ·  fruit-3fa2", "5 → 128 → 128 → 4", "ROUND 60", "COMPUTE", "your job's turn",
                     "round 60 is training fruit-3fa2", "accuracy 79.3% +6.9 pts", "best 79.3%", "loss 0.6732",
                     "after round 46", "90% ┤", "40% ┤", "1.3300", "units", "11/12", "round 40", "39.17 spent",
                     "60.83 VIBE left", "4 trained · about 7 more at 10.00 VIBE each",
                     "round 46: accuracy 79.3% (+6.9 pts, best yet)", "Ctrl+C to stop watching"):
            self.assertIn(want, text)
        self.assertIn("► 79.3%", text)  # the blade's point, and how accurate the model is
        (_, waiting), = [s for s in states() if s[0] == "waiting its turn"]
        self.assertIn("round 61 trains demo-3; fruit-3fa2 waits its turn", text_of(waiting))
        self.assertIn("another job's turn", text_of(waiting))

    def test_the_blade_grows_with_accuracy(self):
        lengths = []
        for n in (1, 3, 8):
            dash = trainui.TrainDashboard(watching(FakeNode(job_reply(n))))
            text = ESCAPE.sub("", frames(dash, (120, 50), 30)[-1])  # long enough to finish growing
            blade = next(line for line in text.split("\n") if "►" in line)
            lengths.append(blade.count("█"))
        self.assertLess(lengths[0], lengths[1])
        self.assertLess(lengths[1], lengths[2])

    def test_ascii_fallback_is_pure_ascii(self):
        for name, watch in states():
            with self.subTest(state=name):
                for frame in frames(_ascii(watch), (100, 40)):
                    text = ESCAPE.sub("", frame)
                    text.encode("ascii")
                    self.assertNotIn("?", text)  # no Unicode left behind to be replaced

    def test_no_color(self):
        """NO_COLOR is read when the dashboards load, so look in a fresh interpreter, as a user's terminal would."""
        code = ("import os\nfrom tests.test_trainui import states, frames\nfrom vibehash import trainui\n"
                "for name, w in states():\n    for f in frames(trainui.TrainDashboard(w), (100, 40), 2):\n        print(f)\n")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = {k: v for k, v in os.environ.items() if k != "NO_COLOR"}
        colour = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, check=True)
        plain = subprocess.run([sys.executable, "-c", code], cwd=root, env={**env, "NO_COLOR": "1"}, capture_output=True,
                               text=True, check=True)
        self.assertRegex(colour.stdout, SGR)
        self.assertNotRegex(plain.stdout, SGR)
        self.assertIn("fruit-3fa2", plain.stdout)

    def test_off_when_not_a_terminal(self):
        with mock.patch("sys.stdout.isatty", return_value=False):
            self.assertFalse(trainui.supported())
        self.assertFalse(trainui.supported(plain=True))

    def test_terminal_put_back_when_it_ends_and_on_ctrl_c(self):
        node = FakeNode(job_reply(9))
        watch = watching(node)
        node.job, watch._job_at = job_reply(10, escrow=0), 0  # finishes while watched, then the dashboard closes itself
        out = io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch.object(trainui, "LINGER_S", 0.3), \
                mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((80, 30))):
            trainui.TrainDashboard(watch).run()
        self.assertTrue(out.getvalue().startswith("\x1b[?1049h"))
        self.assertTrue(out.getvalue().endswith("\x1b[?25h\x1b[?1049l"))
        self.assertIsNotNone(watch.finished_at)

        dash, stop, out = trainui.TrainDashboard(watching(FakeNode(job_reply(3)))), threading.Event(), io.StringIO()
        drawn, real = [], dash.frame

        def frame():  # Ctrl+C on the third frame
            drawn.append(1)
            if len(drawn) == 3:
                raise KeyboardInterrupt
            return real()

        dash.frame = frame
        with mock.patch("sys.stdout", out), self.assertRaises(KeyboardInterrupt):
            dash.run(stop)
        self.assertTrue(out.getvalue().endswith("\x1b[?25h\x1b[?1049l"))
        self.assertTrue(stop.is_set())  # and the polling thread is told to stop


class FollowTests(unittest.TestCase):
    """A job followed through its whole life, on a node that's sometimes away or busy, on a fake clock."""

    def story(self):
        later = status(rnd=43, job_id=JOB, height=42)
        other = status(rnd=41, job_id="demo-3", height=40)
        return [(0, None, other, None),  # posted, not in a block yet
                (5, job_reply(0), other, None),
                (12, job_reply(0), later, None),  # its turn
                (20, job_reply(1), other, None),
                (30, job_reply(2), other, None),
                (40, job_reply(2), other, UNREACHABLE),
                (55, job_reply(3), other, None),
                (80, job_reply(3), other, BUSY),
                (95, job_reply(9), status(rnd=60, height=59), None),
                (110, job_reply(10, escrow=0), status(rnd=62, job_id="demo-3", height=61), None)]

    def test_plain_mode_tells_the_story_and_stops_when_finished(self):
        clock, lines = FakeClock(), []
        node = Scripted(clock.now, self.story())
        watch = trainui.JobWatch(JOB, node.url, budget=100 * VIBE, posted=True, api=node)
        with clock.patch():
            trainui.follow_plain(watch, lines.append)
        said = [re.sub(r"^\d\d:\d\d:\d\d  ", "", line) for line in lines]
        expected = ["following job fruit-3fa2 on http://127.0.0.1:8545; Ctrl+C to stop",
                    "waiting for fruit-3fa2 to reach a block",
                    "fruit-3fa2 is on chain: 100.00 VIBE budget, about 10 rounds at 10.00 VIBE each",
                    "round 43 open: training fruit-3fa2 now",
                    "round 40: accuracy 49.5%, loss 1.3300, 12 units accepted",
                    "round 42: accuracy 62.8% (+13.3 pts, best yet), loss 1.0550, 12 units accepted",
                    "can't reach the node: [Errno 111] Connection refused; retrying",
                    "the node is answering again",
                    "round 44: accuracy 72.4% (+9.6 pts, best yet), loss 0.8405, 12 units accepted",
                    "the node is busy (rate limited): asking less often",
                    "the node is answering again"]
        self.assertEqual(said[:len(expected)], expected)
        self.assertIn("round 60 open: training fruit-3fa2 now", said)
        self.assertEqual(sum(s.startswith("round 5") for s in said), 5)  # rounds 50 to 58
        self.assertIn("budget low: 11.67 VIBE left, about 2 more rounds", said)
        self.assertEqual(said[-1], "finished: 10 rounds trained, accuracy 94.5% (best 94.5%); 100.00 VIBE spent")
        self.assertLess(clock.now[0] - 1_793_400_000, 110 + trainui.LINGER_S + 3)  # it stopped once it had lingered
        self.assertIn("fruit-3fa2 finished: 10 rounds trained, accuracy 94.5% on held-back rows (started at 49.5%, "
                      "best 94.5%).", watch.summary())

    def test_polls_politely_and_backs_off(self):
        clock = FakeClock()
        node = Scripted(clock.now, self.story())
        watch = trainui.JobWatch(JOB, node.url, budget=100 * VIBE, posted=True, api=node)
        waits = []
        with clock.patch():
            while clock.now[0] - 1_793_400_000 < 110:
                start = len(node.calls)
                waits.append((clock.now[0] - 1_793_400_000, watch.poll(), len(node.calls) - start))
                clock.sleep(waits[-1][1])
        elapsed = clock.now[0] - 1_793_400_000
        self.assertLessEqual(len(node.calls) / elapsed, 1.0, "about one request a second at most")
        down = [w for t, w, _ in waits if 40 <= t < 55]
        self.assertLessEqual(len(down), 6, "unreachable: it should try less and less often")
        self.assertTrue(all(b > a for a, b in zip(down, down[1:])), down)
        busy = [w for t, w, _ in waits if 80 <= t < 95]
        self.assertGreaterEqual(busy[0], trainui.BUSY_S, "busy: it should wait a while before asking again")
        self.assertTrue(all(w <= trainui.BACKOFF_MAX_S * 1.25 for _, w, _ in waits))
        job_reads = [t for t, _, n in waits if n == 2 and 20 <= t < 30]
        self.assertLessEqual(len(job_reads), 2, "the job is read when something changed, not every time")

    def test_states_as_the_dashboard_draws_them(self):
        clock = FakeClock()
        node = Scripted(clock.now, self.story())
        watch = trainui.JobWatch(JOB, node.url, budget=100 * VIBE, posted=True, api=node)
        dash = trainui.TrainDashboard(watch)
        seen = {}
        with clock.patch():
            for at, want in ((1, "waiting for fruit-3fa2 to reach a block"), (8, "no rounds yet"),
                             (15, "round 43 is training fruit-3fa2"), (45, "NODE UNREACHABLE"), (85, "NODE BUSY"),
                             (99, "budget low"), (112, "finished · closing in")):
                while clock.now[0] - 1_793_400_000 < at:
                    clock.sleep(watch.poll())
                seen[want] = ESCAPE.sub("", frames(dash, (120, 50), 1)[0])
                self.assertIn(want, seen[want])
        self.assertIn("the first round's score lands here", seen["no rounds yet"])
        self.assertIn("can't reach the node: [Errno 111] Connection refused", seen["NODE UNREACHABLE"])
        self.assertIn("finished: 10 rounds trained, the budget is spent", seen["finished · closing in"])
        self.assertIn("100.00 spent · 0.00 VIBE left", seen["finished · closing in"])
        with clock.patch():
            clock.sleep(trainui.LINGER_S)
            self.assertEqual(watch.lingering(clock.time()), 0)  # time to close

    def test_a_job_already_finished_stays_on_screen(self):
        watch = watching(FakeNode(job_reply(10, escrow=0)))
        self.assertTrue(watch.was_finished)
        self.assertIsNone(watch.lingering(time.time() + 3600))
        self.assertIn("finished · Ctrl+C to close", text_of(watch))

    def test_cancelled(self):
        node = FakeNode(job_reply(0))
        watch = watching(node)
        node.job = job_reply(0, escrow=0)
        watch._job_at = 0
        watch.poll()
        self.assertIn("it was cancelled", watch.events[-1][2])
        self.assertIn("cancelled: its budget went back to its owner", text_of(watch))
        self.assertEqual(watch.summary(), "fruit-3fa2 was cancelled before it trained.")


class DefensiveTests(unittest.TestCase):
    def test_odd_job_replies(self):
        """Any field of /job may be missing or odd: the model field (null, or without sha256) isn't needed at all."""
        rounds = history(3)
        odd = [job_reply(3, model="null"), job_reply(3, model="partial"), job_reply(0, model="null"), job_reply(3, model="none"),
               {}, None, [], "nonsense", {"spec": None, "rounds": None, "escrow": None},
               {"spec": {"fee_per_round": "ten", "model": {"n_hidden": "x"}}, "escrow": "lots", "rounds": [None, 7, {}]},
               {"spec": {"fee_per_round": 10 * VIBE, "units_per_round": 0}, "escrow": -5,
                "rounds": [{"height": 3, "val_acc": "NaN", "val_loss": float("inf"), "accepted_units": -2},
                           {"height": "4", "val_acc": 7.5, "val_loss": None, "accepted_units": True}]},
               {**job_reply(3), "rounds": rounds[::-1] + rounds[:1]}]  # out of order, and repeated
        for reply in odd:
            with self.subTest(reply=str(reply)[:80]):
                node = FakeNode(reply)
                watch = watching(node, budget=None, polls=2)
                for size in ((40, 20), (120, 50)):
                    for frame in frames(trainui.TrainDashboard(watch), size, 2):
                        for line in frame.split("\n"):
                            self.assertLessEqual(len(ESCAPE.sub("", line)), size[0])
                watch.summary()
        job = trainui.parse_job(odd[-2])
        self.assertEqual([r.acc for r in job.rounds], [None, 1.0])  # NaN dropped, 750% clamped
        self.assertEqual(job.escrow, 0)
        self.assertEqual([r.height for r in trainui.parse_job(odd[-1]).rounds], [40, 42, 44])

    def test_odd_status_replies(self):
        for reply in ({}, None, {"phase": "dancing", "round": "x", "phase_ends": "soon", "job_id": 5, "chain_id": 7},
                      {"round": 9, "phase": "reveal", "phase_ends": None, "job_id": JOB}):
            with self.subTest(reply=reply):
                watch = watching(FakeNode(job_reply(2), reply))
                text_of(watch, (66, 34))
                self.assertIn(watch.net.phase, trainui.PHASES)

    def test_unreadable_replies_are_waited_out(self):
        class Garbled(FakeNode):
            def get(self, path):
                raise ValueError("Expecting value: line 1 column 1 (char 0)")  # a proxy's HTML page, say

        watch = trainui.JobWatch(JOB, "http://x", api=Garbled())
        self.assertGreaterEqual(watch.poll(), trainui.POLL_S)
        self.assertIn("unreadable reply", watch.error)
        self.assertIn("NODE UNREACHABLE", text_of(watch))


class CommandTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        patch = mock.patch.dict(os.environ, {"VIBEHASH_HOME": home.name})
        patch.start()
        self.addCleanup(patch.stop)

    def run_cli(self, node, *argv):
        out = io.StringIO()
        with mock.patch.object(trainui, "Reader", lambda url, *a, **kw: node), mock.patch("sys.stdout", out):
            cli.main(list(argv))
        return out.getvalue()

    def test_job_watch_follows_with_log_lines_when_piped(self):
        out = self.run_cli(FakeNode(job_reply(10, escrow=0)), "job", "watch", JOB, "--node", "http://127.0.0.1:8545")
        self.assertIn(f"following job {JOB} on http://127.0.0.1:8545", out)
        self.assertIn("finished: 10 rounds trained, accuracy 94.5% (best 94.5%); 98.33 VIBE spent", out)
        self.assertIn(f"{JOB} finished: 10 rounds trained", out)
        self.assertRegex(out, rf"Download the model: +\S.* job download {JOB}")

    def test_job_watch_of_a_job_the_node_doesnt_know(self):
        with self.assertRaises(SystemExit) as e:
            self.run_cli(FakeNode(None), "job", "watch", "no-such-job", "--node", "http://127.0.0.1:8545")
        self.assertIn("has no job called no-such-job", str(e.exception.code))


if __name__ == "__main__":
    unittest.main()

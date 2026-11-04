"""The terminal mining dashboard: every frame fits the terminal, whatever its size, state or character set, and its
job selector follows the keys."""
import io
import json
import logging
import os
import re
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from vibehash import minerui, termui
from vibehash.client import NodeClient, NodeError, WorkerStats
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

    def test_a_phase_waiting_for_work_shows_how_much_is_in(self):
        """A phase ends as soon as its work is in (0.5.7), so while some is out its end is only the latest it can be:
        the bar says "3/4 in, up to …" and fills with what's in, and every frame still fits."""
        for size in ((40, 20), (66, 34), (120, 50)):
            for glyphs in (minerui.GLYPHS, minerui.ASCII):
                dash = self.dashboard(glyphs=glyphs)
                dash.rigs[0].client.stats.phase_ends = time.time() + 598
                dash.rigs[0].client.stats.progress = (3, 4)
                for frame in self.frames(dash, size):
                    for line in frame.split("\n"):
                        self.assertLessEqual(len(ESCAPE.sub("", line)), size[0], line)
        text = ESCAPE.sub("", self.frames(self.dashboard(), (120, 50))[-1])
        self.assertIn("s left", text, "no progress from the node (before 0.5.9): the countdown as before")
        dash = self.dashboard()
        dash.rigs[0].client.stats.phase_ends, dash.rigs[0].client.stats.progress = time.time() + 598, (3, 4)
        text = ESCAPE.sub("", self.frames(dash, (120, 50))[-1])
        self.assertIn("3/4 in, up to 59", text)
        self.assertNotIn("s left", text)
        dash.rigs[0].client.stats.progress = (4, 4)  # all in: phase_ends is the real end now
        self.assertIn("s left", ESCAPE.sub("", self.frames(dash, (120, 50))[-1]))

    def test_progress_from_the_node_is_only_two_small_whole_numbers(self):
        from vibehash.termui import parse_progress
        self.assertEqual(parse_progress([3, 4]), (3, 4))
        for bad in (None, [4, 3], [1], [1, 2, 3], ["1", "2"], [1.0, 2], [True, 2], [-1, 2], [0, 10**7], "3/4", {}):
            self.assertIsNone(parse_progress(bad), bad)

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


# --- an Evo job (es.py): a chain of blocks in place of the drill (esui.py) ------------------------------------------

def evo_job(job_id="evo-1"):
    """The measured run's job (es_local.SETTINGS) on Qwen2.5-0.5B-Instruct's shapes, as testnet --evo posts it."""
    from vibehash import es, es_local
    spec = es.EsSpec(base="hf:Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775", manifest="1" * 64,
                     dims={"family": "qwen2", "vocab": 151936, "hidden": 896, "layers": 24, "heads": 14, "kv_heads": 2,
                           "head_dim": 64, "ffn": 4864, "tied": True}, tensors=es.preset("mlp", 23), n=13_074_432,
                     tensors_sha="2" * 64, sigma=1e-3, alpha=5e-7,
                     **{k: v for k, v in es_local.SETTINGS.items() if k not in ("preset", "sigma")})
    return es_local.job(spec, job_id, "req", 500 * SYN_PER_VIBE)


def lineage(n=13, job_id="evo-1"):
    """n generations of an Evo job: the reward climbing, the level moving up after the fourth."""
    from vibehash import esui
    lin = esui.Lineage(job_id)
    for i in range(n):
        m = 0.18 + 0.022 * i
        update = [[j, round(min(1.0, m + 0.03 * ((j + i) % 3 - 1)), 5), round(max(0.0, m - 0.03 * ((j * i) % 3 - 1)), 5)]
                  for j in range(8)] if i != 5 else []  # generation 6: nothing accepted
        lin.gens.append(esui.parse_generation(i + 1, 28 + i, {"es": {"rule": 1, "level": 2 + (i >= 4), "seed": "ab" * 8,
                                                                      "update": update}}))
    lin.height, lin.caught_up = 28 + n, True
    return lin


def evo_rig(phase="compute", rnd=42, height=41, progress=(1, 4), busy=True):
    import numpy as np
    from vibehash import es
    job = evo_job()
    state = np.zeros(es.CTRL, np.float32)
    state[-es.CTRL + 1] = 3
    stats = WorkerStats(round=rnd, phase=phase, phase_ends=time.time() + 120, progress=progress, height=height, job=job,
                        flops_per_sec=2e12, activity="answering unit 2 of 2 (4 nudged models x 32 problems)", busy=busy,
                        assigned=2, trained=1, committed=1, rounds=14, units=26, checks=27, training_flops=4.3e14)
    return SimpleNamespace(stats=stats, backend=SimpleNamespace(describe=lambda: "NVIDIA GeForce RTX 4070 SUPER (CUDA)"),
                           keys=SimpleNamespace(address="vibe17b0d8bce54f6eab50df24fd44e5a408a9525"),
                           es=SimpleNamespace(rebuilt=13, fetched=1, states={job.job_id: (height, state)}),
                           jobs={job.job_id: job, "fruit-demo": JOB})


class EvoDashboardTests(unittest.TestCase):
    def dashboard(self, n=1, ascii=False, gens=13, **kw):
        from vibehash import esui
        dash = minerui.MinerDashboard([(f"worker-{i}", evo_rig(**kw)) for i in range(n)], "http://127.0.0.1:8545")
        dash.g, dash.eg = (minerui.ASCII, esui.ASCII) if ascii else (minerui.GLYPHS, esui.GLYPHS)
        dash.chain_id = "vibehash-local-3fa2c1"
        for r in dash.rigs:
            r.start, r.total, r.balance, r.stake, r.reputation = 0, 22 * SYN_PER_VIBE, 12 * SYN_PER_VIBE, 10 * SYN_PER_VIBE, 1.0
        if gens is not None:
            dash.lineages["evo-1"] = lineage(gens)
        dash.jobs = ["evo-1", "fruit-demo"]
        return dash

    def frames(self, dash, size):
        with mock.patch("shutil.get_terminal_size", return_value=os.terminal_size(size)):
            return [dash.frame() for _ in range(3)]

    def test_every_line_fits_the_terminal(self):
        states = [{}, {"phase": "register", "progress": None}, {"phase": "reveal", "progress": (3, 4)},
                  {"phase": "verify", "progress": (7, 8)}, {"phase": "idle", "height": 42},  # sealed: a new block opens
                  {"phase": "idle", "rnd": None, "busy": False}, {"gens": 0}, {"gens": 200}, {"gens": None}]
        for size in ((40, 20), (66, 34), (120, 50)):
            for n in (1, 3):
                for ascii in (False, True):
                    for state in states:
                        for hatching in (None, 0.5, 2.5):
                            with self.subTest(size=size, workers=n, ascii=ascii, hatching=hatching, **state):
                                dash = self.dashboard(n, ascii, **state)
                                lin = dash.lineages.get("evo-1")
                                if hatching is not None and lin is not None and lin.gens:
                                    lin.landed = (time.time() - hatching, lin.gens[-1])
                                for frame in self.frames(dash, size):
                                    lines = frame.split("\n")
                                    self.assertLessEqual(len(lines), size[1])
                                    for line in lines:
                                        self.assertLessEqual(len(ESCAPE.sub("", line)), size[0], line)
                                    if ascii:
                                        ESCAPE.sub("", frame).encode("ascii")

    def test_shows_the_generation_and_what_the_chain_holds(self):
        from vibehash import esui
        dash = self.dashboard()
        text = ESCAPE.sub("", self.frames(dash, (120, 50))[-1])
        gens = dash.lineages["evo-1"].gens
        for want in ("Evo miner", "generation 14", "population 16 (8 twin pairs, 8 yours)", "puzzles 32 a round",
                     "level     3: 3 numbers 1-20 with + − ×", "moves every 2 rounds toward 50% solved",
                     f"reward    {gens[-1].mean:.1%}", "the mean of generation 13", "for generation 13", "for all 13",
                     esui.size(gens[-1].bytes), esui.size(sum(x.bytes for x in gens)),
                     "which rebuild 13,074,432 numbers", "rebuilt from the chain 13x", "fetched 1x",
                     "RTX 4070 SUPER", "incubating 1/4", "8 twin pairs, + and -", "g13 L3", "1 evo-1 (Evo)",
                     "2 fruit-demo "):
            self.assertIn(want, text)
        self.assertNotIn("fruit-demo (Evo)", text)  # only Evo jobs are marked
        self.assertNotIn("GFLOP/s", text)  # the drill's lines aren't
        self.assertTrue(all(100 < x.bytes < 400 for x in gens if x.update))  # a few hundred bytes a generation
        early = "\n".join(ESCAPE.sub("", r) for r in esui.scene(esui.GLYPHS, 118, gens[:6], time.time(), 1, number=7))
        self.assertIn("g5 ▲3", early)  # where the level went up
        self.assertIn("g6 L3", early)
        self.assertEqual(early.count("╔══════╗"), 6)

    def test_a_rebuild_from_the_chain_shows_how_far_along_it_is(self):
        """A process's first round of an Evo v2 job replays the chain on the CPU (half an hour by a few hundred rounds):
        the status line and the model line say so, with the rounds done and the time left, at every size."""
        with mock.patch.object(minerui, "_rebuilding", return_value=(212, 480, "evo2-c2-sciq", 600.0)):
            text = ESCAPE.sub("", self.frames(self.dashboard(3), (120, 50))[-1])
            self.assertIn("rebuilding the model from the chain: 212 of 480 rounds (44%), now evo2-c2-sciq's · about "
                          "10 min left", text)
            self.assertIn("model     rebuilding from the chain on the CPU 212 of 480 rounds (44%) · about 10 min left", text)
            self.assertNotIn("workers busy", text)  # (every worker waits on the one rebuild)
            for size in ((40, 20), (66, 34)):
                for ascii in (False, True):
                    with self.subTest(size=size, ascii=ascii):
                        for line in self.frames(self.dashboard(3, ascii), size)[-1].split("\n"):
                            self.assertLessEqual(len(ESCAPE.sub("", line)), size[0], line)
                        self.assertIn("rebuilding", ESCAPE.sub("", self.frames(self.dashboard(3, ascii), size)[-1]))
        text = ESCAPE.sub("", self.frames(self.dashboard(3), (120, 50))[-1])
        self.assertNotIn("rebuilding", text)  # none going on: the usual lines
        self.assertIn("rebuilt from the chain 39x", text)

    def test_the_block_hatches_then_locks_into_the_chain(self):
        from vibehash import esui
        lin, now = lineage(5), time.time()
        g = esui.GLYPHS
        newest = lin.gens[-1]

        def scene(t):
            rows = [ESCAPE.sub("", r) for r in esui.scene(g, 100, lin.gens, now, 7, number=6, phase="register",
                                                          pairs=8, landed=(now - t, newest))]
            return rows, "\n".join(rows)
        rows, text = scene(0.5)  # hatching: the fitter twin of each accepted pair glows
        self.assertIn("generation 5", rows[0])
        self.assertIn(g["hatched"], rows[2])
        self.assertEqual(rows[2].count(g["hatched"]), sum(rp != rm for _, rp, rm in newest.update))
        self.assertIn("hatched", text)
        self.assertEqual(text.count("╔══════╗"), 4)  # the other four, sealed
        rows, text = scene(2.5)  # locking: the open block's sides turn the sealed blocks' double lines
        self.assertTrue(rows[0].rstrip().endswith("╗"))
        self.assertIn("sealed", text)
        rows, text = scene(esui.HATCH_S + 1)  # in the chain, and a new empty block open for generation 6
        self.assertEqual(text.count("╔══════╗"), 5)
        self.assertIn("generation 6", rows[0])
        self.assertIn("g5 ▲3", text)
        self.assertNotIn(g["hatched"], text)

    def test_other_jobs_keep_the_drill(self):
        """Only an Evo job's frame changes: a plain job's has no chain and none of its lines, whatever the hint."""
        dash = minerui.MinerDashboard([("worker-0", rig())], "http://127.0.0.1:8545", evo=True)
        dash.g, dash.chain_id = minerui.GLYPHS, "vibehash-local-123558"
        text = ESCAPE.sub("", self.frames(dash, (120, 50))[-1])
        for gone in ("generation", "Evo", "╔══════╗", "population"):
            self.assertNotIn(gone, text)
        self.assertIn("useful-work miner", text)
        self.assertIn("290.00 GFLOP/s", text)
        rig_ = rig()
        rig_.stats.job = None  # an Evo testnet before its job reaches a block: the chain, waiting for it
        dash = minerui.MinerDashboard([("worker-0", rig_)], "http://127.0.0.1:8545", evo=True)
        dash.g = minerui.GLYPHS
        text = ESCAPE.sub("", self.frames(dash, (120, 50))[-1])
        self.assertIn("waiting for the Evo job", text)
        self.assertIn("training by evolution, the chain is the model", text)

    def test_the_lineage_follows_the_chain_defensively(self):
        """Blocks and records as a node serves them, odd ones too: only the job's rounds count, a record's malformed
        scores are dropped, and a generation sealed while followed lands (with news), not those caught up on."""
        from vibehash import esui
        good = {"es": {"rule": 1, "level": 2, "seed": "ab" * 8, "update": [[0, 0.5, 0.25], [1, 0.75, 0.75]]}}
        odd = {"es": {"level": 99, "update": [[0, 2.0, 0.1], [1, float("nan"), 0], "junk", [2, 0.5], [3, 1, 0]]}}
        blocks = {1: {"height": 1, "summary": None}, 2: {"height": 2, "summary": {"job_id": "evo-1"}},
                  3: {"height": 3, "summary": {"job_id": "fruit-demo"}}, 4: {"height": 4, "summary": {"job_id": "evo-1"}},
                  5: {"height": 5, "summary": {"job_id": "evo-1"}}}
        records = {2: good, 4: odd, 5: good | {"es": good["es"] | {"level": 3}}}
        calls = []

        class Api:
            def get(self, path):
                calls.append(path)
                m = re.fullmatch(r"/blocks\?from=(\d+)&to=(\d+)", path)
                if m:
                    return [blocks[h] for h in range(int(m[1]), int(m[2]) + 1)] + ["junk", {"height": "7"}]
                h = int(re.fullmatch(r"/round/(\d+)/record", path)[1])
                if h == 4 and calls.count(path) == 1:
                    raise NodeError("GET: busy", 503)  # tried again at the next poll
                return records[h]
        lin = esui.Lineage("evo-1")
        with self.assertRaises(NodeError):
            lin.poll(Api(), 4)
        self.assertEqual([x.height for x in lin.gens], [2])
        self.assertEqual(lin.poll(Api(), 4)[0].height, 4)
        self.assertTrue(lin.caught_up)
        self.assertIsNone(lin.landed)  # caught up on: nothing lands
        first, second = lin.gens
        self.assertEqual((first.number, first.level, first.pairs, first.mean), (1, 2, 2, 0.5625))
        self.assertEqual(first.bytes, len(json.dumps(good["es"], sort_keys=True, separators=(",", ":"))))
        self.assertEqual((second.number, second.level, second.update), (2, None, ((3, 1, 0),)))
        new = lin.poll(Api(), 5)
        self.assertEqual([x.number for x in new], [3])
        self.assertIs(lin.landed[1], new[0])
        news = lin.news(new[0], esui.ASCII)
        self.assertIn("generation 3 hatched in block 5: reward 56.2% (+6.2 pts), 2 pairs", news[0][0])
        self.assertEqual(lin.poll(Api(), 5), [])  # nothing new: no request
        self.assertEqual(sum(c.startswith("/blocks") for c in calls), 3)

    def test_glyphs_have_an_ascii_twin(self):
        from vibehash import esui
        self.assertEqual(set(esui.GLYPHS), set(esui.ASCII))
        for k, v in esui.ASCII.items():
            ("".join(v.values()) if isinstance(v, dict) else v).encode("ascii")
            if isinstance(v, str) and k not in termui.GLYPHS:  # the scene's own: one cell for one cell
                self.assertEqual(len(v), len(esui.GLYPHS[k]), k)


if __name__ == "__main__":
    unittest.main()

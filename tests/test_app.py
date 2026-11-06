"""`vibehash app`: the worker started once and kept running, driven by orders in a file (app.py)."""
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from vibehash import app, settings

# A stand-in worker: says hello, then waits to be stopped (Ctrl-Break on Windows, SIGTERM elsewhere), or exits as told.
CHILD = ("import signal, sys, time\n"
         "signal.signal(getattr(signal, 'SIGBREAK', signal.SIGTERM), lambda *_: sys.exit(0))\n"
         "print('child up', flush=True)\n"
         "if len(sys.argv) > 1: sys.exit(int(sys.argv[1]))\n"
         "time.sleep(60)\n")


def child_command(exit_code=None):
    return [sys.executable, "-c", CHILD] + ([str(exit_code)] if exit_code is not None else [])


def wait_for(cond, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


class AppHome(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict(os.environ, {"VIBEHASH_HOME": self.tmp.name})
        self.env.start()
        self.addCleanup(self.env.stop)


class SupervisorTests(AppHome):
    def test_starts_the_worker_pauses_resumes_and_stops_it(self):
        s = app.Supervisor(child_command(), 7545, beat_s=0.05, grace_s=5)
        t = threading.Thread(target=s.run, daemon=True)
        t.start()
        self.assertTrue(wait_for(lambda: s.worker_alive()))
        self.assertTrue(app.running())
        state = app.read_state()
        self.assertEqual((state["paused"], state["worker"]["running"], state["watch"]), (False, True, "http://127.0.0.1:7545/"))
        app.order("pause")
        self.assertTrue(wait_for(lambda: s.paused and not s.worker_alive()))
        self.assertTrue(wait_for(lambda: app.read_state()["paused"]))  # written at the end of the turn
        app.order("resume")
        self.assertTrue(wait_for(lambda: not s.paused and s.worker_alive()))
        app.order("stop")
        t.join(timeout=10)
        self.assertFalse(t.is_alive())
        self.assertFalse(s.worker_alive())
        self.assertFalse(app.running())
        self.assertEqual(app.read_state()["beat"], 0)
        self.assertFalse(app.control_path().exists())  # orders are taken, not left lying

    def test_a_worker_that_stops_on_its_own_is_started_again_after_a_wait(self):
        s = app.Supervisor(child_command(exit_code=3), 7545, beat_s=0.05, backoff=(0.2, 0.2, 0.2), grace_s=5)
        app.app_dir().mkdir(parents=True)
        now = 1000.0
        s.tick(now)  # starts it
        self.assertTrue(wait_for(lambda: not s.worker_alive()))  # it exits with 3
        s.tick(now + 0.1)  # noticed: not started again yet
        self.assertEqual(s.restarts, 1)
        self.assertIsNone(s.child)
        self.assertIn("exit code 3", s.events[-1])
        s.tick(now + 0.2)  # too soon
        self.assertIsNone(s.child)
        s.tick(now + 0.4)  # the wait is over
        self.assertIsNotNone(s.child)
        s.stop_worker("test over")

    def test_running_needs_a_fresh_heartbeat(self):
        self.assertFalse(app.running())
        app.app_dir().mkdir(parents=True)
        app.state_path().write_text(json.dumps({"beat": 100.0, "started": 90.0}))
        self.assertTrue(app.running(now=105.0))
        self.assertFalse(app.running(now=100.0 + app.STALE_S + 1))
        app.state_path().write_text("not json")
        self.assertFalse(app.running())


class StatusTests(AppHome):
    def test_status_lines(self):
        self.assertIn("isn't running", app.status_text())
        app.app_dir().mkdir(parents=True)
        now = time.time()
        state = {"beat": now, "started": now - 60, "paused": False, "watch": "http://127.0.0.1:7545/",
                 "worker": {"running": True, "pid": 123, "restarts": 2}, "events": ["12:00:00  worker started"]}
        app.state_path().write_text(json.dumps(state))
        from vibehash.tasks import SYN_PER_VIBE
        feed = {"syn_per_vibe": SYN_PER_VIBE, "workers": [{"address": "vibe1abc", "round": 9, "activity": "training unit 1 of 2",
                                                           "session": {"rounds": 3, "units": 7},
                                                           "account": {"balance": int(2.5 * SYN_PER_VIBE), "stake": 10 * SYN_PER_VIBE}}]}
        text = app.status_text(now, fetch=lambda url: feed)
        self.assertIn("running since", text)
        self.assertIn("worker: running (pid 123, started 2 times again)", text)
        self.assertIn("vibe1abc: round 9, training unit 1 of 2; 3 rounds and 7 units this session; balance 2.5 VIBE, stake 10 VIBE", text)
        self.assertIn("worker started", text)

        def down(url):
            raise OSError("no")
        self.assertIn("feed isn't answering", app.status_text(now, fetch=down))
        state["paused"], state["worker"]["running"] = True, False
        app.state_path().write_text(json.dumps(state))
        text = app.status_text(now, fetch=lambda url: feed)
        self.assertIn(", paused", text)
        self.assertIn("worker: paused", text)

    def test_stop_and_wait_when_nothing_runs(self):
        self.assertTrue(app.stop_and_wait(timeout=1))


class PrepareTests(AppHome):
    def test_makes_a_wallet_unattended_and_keeps_it(self):
        with mock.patch.dict(os.environ, {"VIBEHASH_PASSWORD": "pw"}), mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            url, path = app.prepare(SimpleNamespace(node="192.168.1.20", gpu="skip"))
        self.assertEqual(url, "http://192.168.1.20:8545")
        self.assertEqual(path, settings.default_wallet())
        self.assertTrue(path.exists())
        self.assertEqual(settings.load()["node"], url)
        self.assertIn("recovery words", out.getvalue().lower())
        words = list(Path(self.tmp.name).glob("recovery-words-*.txt"))
        self.assertEqual(len(words), 1)  # no terminal: the words went to a file
        address = json.loads(path.read_text())["address"]
        with mock.patch.dict(os.environ, {"VIBEHASH_PASSWORD": "pw"}), mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            url2, path2 = app.prepare(SimpleNamespace(gpu="skip"))
        self.assertEqual((url2, path2), (url, path))
        self.assertIn(address.lower(), out.getvalue().lower())  # kept, not made again (shown checksummed)

    def test_without_a_password_and_no_terminal_it_says_what_to_set(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VIBEHASH_PASSWORD", None)
            with mock.patch("vibehash.unattended.interactive", return_value=False), self.assertRaises(SystemExit) as ctx:
                app.prepare(SimpleNamespace(gpu="skip"))
        self.assertIn("VIBEHASH_PASSWORD", str(ctx.exception))

    def test_the_gpu_is_checked_once_it_is_settled(self):
        with mock.patch.dict(os.environ, {"VIBEHASH_PASSWORD": "pw"}), mock.patch("vibehash.cli._setup_gpu") as gpu, \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            gpu.return_value = False  # the install failed (no network, say): tried again next start
            app.prepare(SimpleNamespace(gpu="auto"))
            app.prepare(SimpleNamespace(gpu="auto"))
            self.assertEqual(gpu.call_count, 2)
            gpu.return_value = True  # ready, or nothing to install
            app.prepare(SimpleNamespace(gpu="auto"))
            app.prepare(SimpleNamespace(gpu="auto"))
            self.assertEqual(gpu.call_count, 3)
            app.prepare(SimpleNamespace(gpu="auto", recheck_gpu=True))
            self.assertEqual(gpu.call_count, 4)

    def test_the_password_reaches_the_worker(self):
        """The worker is a child process that unlocks the wallet itself from $VIBEHASH_PASSWORD, so a password typed
        at the app's terminal is put there, after a check that it opens the wallet."""
        from vibehash import wallet as wallets
        env = {k: v for k, v in os.environ.items() if k != "VIBEHASH_PASSWORD"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("vibehash.cli.password", return_value="typed"), \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            url, path = app.prepare(SimpleNamespace(gpu="skip"))  # a new wallet, made with the typed password
            self.assertEqual(os.environ["VIBEHASH_PASSWORD"], "typed")
            self.assertTrue(wallets.is_encrypted(path))
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("vibehash.cli.password", return_value="typed"), \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            app.prepare(SimpleNamespace(gpu="skip"))  # the same wallet next time: asked, checked, handed on
            self.assertEqual(os.environ["VIBEHASH_PASSWORD"], "typed")
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("vibehash.cli.password", return_value="wrong"), \
                mock.patch("sys.stdout", new_callable=io.StringIO), self.assertRaises(SystemExit) as ctx:
            app.prepare(SimpleNamespace(gpu="skip"))
        self.assertIn("VIBEHASH_PASSWORD", str(ctx.exception))  # a line that says what to do, not a traceback

    def test_a_wallet_without_a_password_keeps_its_words_in_a_file_even_at_a_terminal(self):
        env = {k: v for k, v in os.environ.items() if k != "VIBEHASH_PASSWORD"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("vibehash.unattended.interactive", return_value=True), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            url, path = app.prepare(SimpleNamespace(gpu="skip", no_password=True))
        from vibehash import wallet as wallets
        self.assertFalse(wallets.is_encrypted(path))
        self.assertEqual(len(list(Path(self.tmp.name).glob("recovery-words-*.txt"))), 1)  # the app's log scrolls on
        self.assertIn("recovery words", out.getvalue().lower())
        self.assertNotIn("VIBEHASH_PASSWORD", os.environ)


class StartupTests(AppHome):
    def test_each_system_gets_its_file(self):
        calls = []
        run = lambda cmd, **kw: calls.append(cmd)  # noqa: E731
        with mock.patch.object(Path, "home", return_value=Path(self.tmp.name)), \
                mock.patch.dict(os.environ, {"APPDATA": str(Path(self.tmp.name) / "AppData")}):
            win = app.startup_on("Windows", run)
            self.assertEqual(win.name, "VibeHash.cmd")
            self.assertIn("-m vibehash app", win.read_text())
            self.assertIn(self.tmp.name, win.read_text())  # VIBEHASH_HOME, so the service finds the same wallet
            mac = app.startup_on("Darwin", run)
            self.assertEqual(mac.name, "org.vibehash.app.plist")
            self.assertIn("<string>app</string>", mac.read_text())
            self.assertEqual(calls[-1][:2], ["launchctl", "load"])
            lin = app.startup_on("Linux", run)
            self.assertEqual(lin.name, "vibehash.service")
            self.assertIn("ExecStart=", lin.read_text())
            self.assertEqual(calls[-1][:3], ["systemctl", "--user", "enable"])
            for system, path in (("Windows", win), ("Darwin", mac), ("Linux", lin)):
                app.startup_off(system, run)
                self.assertFalse(path.exists())
            self.assertEqual(calls[-1][:3], ["systemctl", "--user", "disable"])


class CommandTests(AppHome):
    def test_plain_app_is_start(self):
        from vibehash import cli
        a = cli.parser().parse_args(["app"])
        self.assertIsNone(a.acmd)
        self.assertIs(a.fn, cli.cmd_app)
        a = cli.parser().parse_args(["app", "--device", "cuda", "--gpu", "skip"])  # the start options, without "start"
        self.assertEqual((a.acmd, a.device, a.gpu), (None, "cuda", "skip"))
        self.assertEqual(cli.parser().parse_args(["app", "start", "--watch", "7000"]).watch, 7000)
        self.assertEqual(cli.parser().parse_args(["app", "startup", "off"]).mode, "off")

    def test_orders_need_a_running_app(self):
        with self.assertRaises(SystemExit) as ctx:
            app.main(SimpleNamespace(acmd="pause"))
        self.assertIn("isn't running", str(ctx.exception))
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            app.main(SimpleNamespace(acmd="stop"))
        self.assertIn("stopped", out.getvalue())

    def test_a_second_app_is_refused(self):
        app.app_dir().mkdir(parents=True)
        app.state_path().write_text(json.dumps({"beat": time.time(), "started": time.time()}))
        with self.assertRaises(SystemExit) as ctx:
            app.main(SimpleNamespace(acmd=None))
        self.assertIn("already running", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

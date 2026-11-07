"""`vibehash app`: the worker started once and kept running, driven by orders in a file (app.py)."""
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from vibehash import app, settings  # noqa: F401

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


class FakeWinreg:
    """Enough of winreg for the start-at-login and uninstall entries, in memory: {key path: {name: value}}."""
    HKEY_CURRENT_USER, KEY_SET_VALUE, REG_SZ, REG_DWORD = "HKCU", 2, 1, 4

    class Key:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def __init__(self):
        self.keys = {app.RUN_KEY: {}}  # the Run key is always there on Windows

    def CreateKey(self, root, path):  # noqa: N802
        self.keys.setdefault(path, {})
        return self.Key(path)

    def OpenKey(self, root, path, reserved=0, access=0):  # noqa: N802
        if path not in self.keys:
            raise OSError("no such key")
        return self.Key(path)

    def SetValueEx(self, key, name, reserved, kind, value):  # noqa: N802
        self.keys[key.path][name] = value

    def QueryValueEx(self, key, name):  # noqa: N802
        if name not in self.keys[key.path]:
            raise OSError("no such value")
        return self.keys[key.path][name], self.REG_SZ

    def DeleteValue(self, key, name):  # noqa: N802
        if self.keys[key.path].pop(name, None) is None:
            raise OSError("no such value")

    def DeleteKey(self, root, path):  # noqa: N802
        if self.keys.pop(path, None) is None:
            raise OSError("no such key")


class AppHome(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict(os.environ, {"VIBEHASH_HOME": self.tmp.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("VIBEHASH_LAUNCHER", None)
        self.winreg = FakeWinreg()  # never the machine's own registry
        self.reg = mock.patch.object(app, "_winreg", return_value=None)
        self.reg.start()
        self.addCleanup(self.reg.stop)


class SupervisorTests(AppHome):
    def start(self, s):
        """s.run() on a thread that always ends with the test: a failed assertion before the test's own stop order
        used to leave it running, and since app_dir() follows $VIBEHASH_HOME it moved into the next test's folder,
        taking that test's orders and writing its state.json."""
        done = threading.Event()
        t = threading.Thread(target=s.run, kwargs={"stop": done}, daemon=True)
        t.start()

        def end():
            done.set()
            if s._beating is not None:
                s._beating.set()
            t.join(timeout=15)  # grace_s 5 and stop_worker's 5 s wait, with room
            self.assertFalse(t.is_alive(), "the supervisor didn't stop: it would carry on into the next test")
        self.addCleanup(end)
        return t

    def test_starts_the_worker_pauses_resumes_and_stops_it(self):
        s = app.Supervisor(child_command(), 7545, beat_s=0.05, grace_s=5)
        t = self.start(s)
        self.assertTrue(wait_for(lambda: s.worker_alive()))
        # the state is written at the end of the turn that started the worker: wait for that turn's
        self.assertTrue(wait_for(lambda: ((app.read_state() or {}).get("worker") or {}).get("running")))
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

    def test_the_wait_grows_only_for_stops_soon_after_a_start(self):
        """The count of restarts never went back down: three stops days apart, and every restart after waited 60 s."""
        s = app.Supervisor(child_command(exit_code=3), 7545, beat_s=0.05, backoff=(5, 15, 60), grace_s=5)
        app.app_dir().mkdir(parents=True)
        waits = []
        for ran in (1, 1, 1, 1, app.STEADY_S + 1, 1):
            s.child = SimpleNamespace(returncode=3, poll=lambda: 3)
            s.worker_since = 5000.0 - ran
            with mock.patch.object(s, "worker_alive", return_value=False), mock.patch("sys.stdout", new_callable=io.StringIO):
                s.tick(5000.0)
            waits.append(s.next_start - 5000.0)
        self.assertEqual(waits, [5, 15, 60, 60, 5, 15])
        self.assertEqual(s.restarts, 6, "status still counts every restart")

    def test_the_window_and_the_avatar_ride_along_and_leave_with_the_app(self):
        s = app.Supervisor(child_command(), 7545, beat_s=0.05, grace_s=5, companions=[child_command(), child_command()])
        t = self.start(s)
        self.assertTrue(wait_for(lambda: s.worker_alive() and len(s.friends) == 2 and all(f.poll() is None for f in s.friends)))
        app.order("stop")
        t.join(timeout=10)
        self.assertFalse(t.is_alive())
        self.assertTrue(wait_for(lambda: all(f.poll() is not None for f in s.friends)))  # told to leave when the app left

    def test_the_avatar_is_turned_on_and_off_by_orders_and_the_setting(self):
        s = app.Supervisor(child_command(), 7545, beat_s=0.05, grace_s=5, avatar=child_command())
        t = self.start(s)
        self.assertTrue(wait_for(lambda: s.worker_alive() and s.avatar_alive()))
        self.assertTrue(wait_for(lambda: (app.read_state() or {}).get("avatar") is True))
        app.order("avatar-off")
        self.assertTrue(wait_for(lambda: not s.avatar_alive()))
        self.assertTrue(wait_for(lambda: (app.read_state() or {}).get("avatar") is False))
        app.order("avatar-on")
        self.assertTrue(wait_for(lambda: s.avatar_alive()))
        app.order("stop")
        t.join(timeout=10)
        self.assertTrue(wait_for(lambda: s.avatar.poll() is not None))  # leaves with the app
        settings.save(avatar=False)  # the setting: not started at all
        s = app.Supervisor(child_command(), 7545, beat_s=0.05, grace_s=5, avatar=child_command())
        t = self.start(s)
        self.assertTrue(wait_for(lambda: s.worker_alive()))
        self.assertFalse(s.avatar_alive())
        app.order("stop")
        t.join(timeout=10)

    def test_setup_is_reported_in_the_state_with_a_heartbeat_before_the_worker(self):
        s = app.Supervisor([], 7545, beat_s=0.05, grace_s=5)
        s.beat_while_preparing()
        self.addCleanup(s._beating.set)  # the heartbeat thread ends with the test, even one that fails before run()
        self.assertTrue(wait_for(lambda: app.running()))  # alive through setup, so a second app is refused
        s.report("gpu", 0.3, wallet={"address": "0xabc", "path": "w.json", "words_file": None})
        st = app.read_state()
        self.assertEqual(st["setup"], {"step": "gpu", "fraction": 0.3})
        self.assertEqual(st["wallet"]["address"], "0xabc")
        self.assertIn("setup: gpu", " ".join(s.events))
        s.command = child_command()
        t = self.start(s)
        self.assertTrue(wait_for(lambda: s.worker_alive() and (app.read_state() or {}).get("setup", {}).get("step") == "mining"))
        self.assertTrue(s._beating.is_set())  # run() took the heartbeat over
        app.order("stop")
        t.join(timeout=10)

    def test_a_companion_that_cannot_start_is_said_and_the_app_goes_on(self):
        s = app.Supervisor(child_command(), 7545, beat_s=0.05, grace_s=5, companions=[["no-such-program-xyz"]])
        t = self.start(s)
        self.assertTrue(wait_for(lambda: s.worker_alive()))
        self.assertTrue(any("couldn't start no-such-program-xyz" in e for e in s.events))
        app.order("stop")
        t.join(timeout=10)
        self.assertFalse(t.is_alive())

    def test_running_needs_a_fresh_heartbeat(self):
        self.assertFalse(app.running())
        app.app_dir().mkdir(parents=True)
        app.state_path().write_text(json.dumps({"beat": 100.0, "started": 90.0}))
        self.assertTrue(app.running(now=105.0))
        self.assertFalse(app.running(now=100.0 + app.STALE_S + 1))
        app.state_path().write_text("not json")
        self.assertFalse(app.running())


class HoldTests(AppHome):
    """The automatic pauses (Settings: while gaming, on battery, only when away): presence.hold decides, the Supervisor
    stops the worker while there's a reason and starts it again once the reason has been gone for the cooldown."""

    def test_the_decision(self):
        from vibehash import presence
        none = {"battery": None, "fullscreen": None, "idle_s": None}
        self.assertIsNone(presence.hold({}, {"battery": True, "fullscreen": True, "idle_s": 0}))  # nothing chosen
        self.assertEqual(presence.hold({"pause_on_battery": True}, {"battery": True}), "on battery")
        self.assertIsNone(presence.hold({"pause_on_battery": True}, {"battery": False}))
        self.assertIn("full-screen", presence.hold({"pause_when_fullscreen": True}, {"fullscreen": True}))
        self.assertIsNone(presence.hold({"pause_when_fullscreen": True}, {"fullscreen": False}))
        away = {"idle_only": True, "idle_minutes": 10}
        self.assertIn("after 10 min", presence.hold(away, {"idle_s": 599}))
        self.assertIsNone(presence.hold(away, {"idle_s": 600}))
        self.assertIn("after 5 min", presence.hold({"idle_only": True, "idle_minutes": "lots"}, {"idle_s": 1}))
        # a system that can't tell never holds mining off, whatever is chosen
        self.assertIsNone(presence.hold({"pause_on_battery": True, "pause_when_fullscreen": True, "idle_only": True}, none))

    def test_the_worker_is_held_off_and_comes_back_after_the_cooldown(self):
        facts = {"fullscreen": True}
        s = app.Supervisor(child_command(), 7545, beat_s=0.05, grace_s=5, probe=lambda: dict(facts),
                           hold_check_s=0.0, cooldown_s=60.0)
        app.app_dir().mkdir(parents=True)
        self.addCleanup(s.stop_worker, "test over")
        s.tick(1000.0)  # nothing chosen: it mines
        self.assertTrue(wait_for(s.worker_alive))
        settings.save(pause_when_fullscreen=True)
        s.tick(1001.0)  # a game full screen: held off, the worker stopped
        self.assertIn("full-screen", s.held)
        self.assertFalse(s.worker_alive())
        self.assertEqual(app.read_state()["held"], s.held)
        self.assertEqual(s.restarts, 0)  # held off isn't "stopped on its own"
        facts["fullscreen"] = False
        s.tick(1002.0)  # the game closed: still held for the cooldown, so a flicker doesn't reload the model
        self.assertIsNotNone(s.held)
        self.assertIsNone(s.child)
        s.tick(1061.0)
        self.assertIsNotNone(s.held)
        s.tick(1062.0)  # a minute without the reason: mining again
        self.assertIsNone(s.held)
        self.assertTrue(wait_for(s.worker_alive))
        self.assertIsNone(app.read_state()["held"])

    def test_a_probe_that_fails_never_holds_mining_off(self):
        settings.save(pause_on_battery=True)

        def broken():
            raise OSError("no such call here")
        s = app.Supervisor(child_command(), 7545, beat_s=0.05, grace_s=5, probe=broken, hold_check_s=0.0)
        app.app_dir().mkdir(parents=True)
        self.addCleanup(s.stop_worker, "test over")
        s.tick(1000.0)
        self.assertIsNone(s.held)
        self.assertTrue(wait_for(s.worker_alive))

    def test_your_own_pause_wins_over_a_reason_ending(self):
        facts = {"battery": True}
        settings.save(pause_on_battery=True)
        s = app.Supervisor(child_command(), 7545, beat_s=0.05, grace_s=5, probe=lambda: dict(facts),
                           hold_check_s=0.0, cooldown_s=0.0)
        app.app_dir().mkdir(parents=True)
        self.addCleanup(s.stop_worker, "test over")
        s.tick(1000.0)
        self.assertEqual(s.held, "on battery")
        app.order("pause")
        s.tick(1001.0)
        facts["battery"] = False
        s.tick(1002.0)  # plugged in, but paused by hand: stays off
        self.assertIsNone(s.held)
        self.assertTrue(s.paused)
        self.assertIsNone(s.child)


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
            self.assertIn("-m vibehash app --no-window", win.read_text())  # logging in opens no window
            self.assertIn(self.tmp.name, win.read_text())  # VIBEHASH_HOME, so the service finds the same wallet
            self.assertNotIn("--avatar", win.read_text())
            self.assertIn("-m vibehash app --no-window --avatar", app.startup_on("Windows", run, avatar=True).read_text())
            with mock.patch.dict(os.environ, {"VIBEHASH_LAUNCHER": r"C:\x\VibeHash.exe"}):  # the download: the launcher itself
                text = app.startup_on("Windows", run, avatar=True).read_text()  # (no winreg, as on another system)
                self.assertIn('"C:\\x\\VibeHash.exe" -- --no-window', text)  # what follows -- goes to the app
                self.assertNotIn("-m vibehash", text)
                with mock.patch.object(app, "_winreg", return_value=self.winreg):  # on Windows itself: a Run value, no .cmd
                    where = app.startup_on("Windows", run, avatar=True)
                    self.assertEqual(where.name, "VibeHash")
                    self.assertEqual(self.winreg.keys[app.RUN_KEY]["VibeHash"], '"C:\\x\\VibeHash.exe" -- --no-window')
                    self.assertFalse(win.exists())  # the .cmd an earlier version wrote: gone
                    self.assertTrue(app.startup_is_on("Windows"))
                    app.startup_off("Windows", run)
                    self.assertNotIn("VibeHash", self.winreg.keys[app.RUN_KEY])
                    self.assertFalse(app.startup_is_on("Windows"))
                win = app.startup_on("Windows", run)  # back to a file, for the rest of the test
            lin = app.startup_on("Linux", run, avatar=True)
            self.assertIn("-m vibehash app --no-window --avatar", lin.read_text())
            mac = app.startup_on("Darwin", run)
            self.assertEqual(mac.name, "org.vibehash.app.plist")
            self.assertIn("<string>app</string><string>--no-window</string>", mac.read_text())
            self.assertEqual(calls[-1][:2], ["launchctl", "load"])
            lin = app.startup_on("Linux", run)
            self.assertEqual(lin.name, "vibehash.service")
            self.assertIn("ExecStart=", lin.read_text())
            self.assertEqual(calls[-1][:3], ["systemctl", "--user", "enable"])
            for system, path in (("Windows", win), ("Darwin", mac), ("Linux", lin)):
                app.startup_off(system, run)
                self.assertFalse(path.exists())
            self.assertEqual(calls[-1][:3], ["systemctl", "--user", "disable"])

    def test_a_windows_path_the_codepage_cant_write_or_with_a_percent_still_starts(self):
        """The .cmd was written in the OEM codepage with "replace": a user folder named Дмитрий came out as ??????? and
        the app never started at login. Such a file switches cmd to UTF-8 first; a % is %% (cmd expands %NAME%)."""
        home = Path(self.tmp.name) / "Дмитрий 100%" / ".vibehash"
        with mock.patch.dict(os.environ, {"APPDATA": str(Path(self.tmp.name) / "AppData"), "VIBEHASH_HOME": str(home)}), \
                mock.patch.object(Path, "home", return_value=Path(self.tmp.name)):
            raw = app.startup_on("Windows", lambda *a, **kw: None).read_bytes()
        if b"chcp 65001" in raw:  # (a codepage without Cyrillic: every English or western Windows)
            self.assertTrue(raw.startswith(b"@echo off\r\nchcp 65001 >nul\r\n"))
            self.assertIn(f'set "VIBEHASH_HOME={str(home).replace("%", "%%")}"'.encode("utf-8"), raw)
        else:  # (one that has it, or another system, where it's written in UTF-8)
            self.assertIn(f'set "VIBEHASH_HOME={str(home).replace("%", "%%")}"'
                          .encode("oem" if os.name == "nt" else "utf-8"), raw)
        self.assertNotIn(b"?", raw)
        if os.name == "nt":  # what cmd itself reads from it
            probe, seen = Path(self.tmp.name) / "probe.cmd", Path(self.tmp.name) / "seen.txt"
            probe.write_bytes(raw.split(b"start ")[0] + b'"' + sys.executable.encode() + b'" -c "import os,sys; '
                              b'open(sys.argv[1],\'w\',encoding=\'utf-8\').write(os.environ[\'VIBEHASH_HOME\'])" "'
                              + str(seen).encode() + b'"\r\n')
            subprocess.run(["cmd", "/c", str(probe)], check=True, capture_output=True)
            self.assertEqual(seen.read_text(encoding="utf-8"), str(home))


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

    def test_options_before_start_are_kept(self):
        """`start`'s defaults overwrote what `app` had read: `app --no-window start` ran with the window, and the
        launcher's `app --no-password --avatar -- start ...` lost both."""
        from vibehash import cli
        a = cli.parser().parse_args(["app", "--no-window", "--no-password", "--avatar", "--watch", "7600", "start"])
        self.assertEqual((a.acmd, a.no_window, a.no_password, a.avatar, a.watch), ("start", True, True, True, 7600))
        a = cli.parser().parse_args(["app", "--device", "cpu", "start", "--gpu", "skip"])
        self.assertEqual((a.device, a.gpu), ("cpu", "skip"))
        a = cli.parser().parse_args(["app", "start"])
        self.assertEqual((a.no_window, a.avatar, a.watch, a.device, a.gpu), (False, False, None, "auto", "auto"))
        self.assertTrue(cli.parser().parse_args(["app", "--avatar", "startup", "on"]).avatar)
        self.assertFalse(cli.parser().parse_args(["app", "startup", "on"]).avatar)

    def test_stop_while_setting_up_ends_setup_at_its_next_step(self):
        """Nothing read the order until mining began: `app stop` said it hadn't stopped, and the app installed on, then
        quit as it started mining."""
        sup = app.Supervisor([sys.executable, "-c", "pass"], 7699, companions=[])
        app.app_dir().mkdir(parents=True, exist_ok=True)
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            sup.report("gpu", 0.3)
            app.order("stop")
            with self.assertRaisesRegex(SystemExit, "told to stop while it set up"):
                sup.report("ready", 0.95)
        self.assertFalse(app.control_path().exists())
        with mock.patch.object(app, "stop_and_wait", return_value=False), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:  # a long step, the GPU build: say so
            app.main(SimpleNamespace(acmd="stop"))
        self.assertIn("still setting up (now at: gpu)", out.getvalue())

    def test_orders_need_a_running_app(self):
        with self.assertRaises(SystemExit) as ctx:
            app.main(SimpleNamespace(acmd="pause"))
        self.assertIn("isn't running", str(ctx.exception))
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            app.main(SimpleNamespace(acmd="stop"))
        self.assertIn("stopped", out.getvalue())

    def test_a_second_app_opens_the_running_ones_window_or_is_refused(self):
        app.app_dir().mkdir(parents=True)
        app.state_path().write_text(json.dumps({"beat": time.time(), "started": time.time()}))
        with self.assertRaises(SystemExit) as ctx:
            app.main(SimpleNamespace(acmd=None, no_window=True))
        self.assertIn("already running", str(ctx.exception))
        with mock.patch.object(app.subprocess, "call", return_value=0) as call, mock.patch("sys.stdout", new_callable=io.StringIO), \
                self.assertRaises(SystemExit) as ctx:  # the Start menu, a second double-click: the way back in is the window
            app.main(SimpleNamespace(acmd=None, watch=7000))
        self.assertEqual(ctx.exception.code, 0)
        self.assertEqual(call.call_args.args[0][-2:], ["--watch", "7000"])


if __name__ == "__main__":
    unittest.main()

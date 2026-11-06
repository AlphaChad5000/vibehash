"""`vibehash app`: mining as something you start once and forget. It gets the computer ready without asking (the node,
a wallet, GPU support), starts the worker with its live feed (`worker --watch`, watch.py), starts it again if it
stops, and takes orders from the other `app` commands:

    vibehash app                 # get ready, join the network, mine, with the window open; Ctrl+C stops it
    vibehash app pause           # stop mining for now (the worker finishes nothing: it leaves its round)
    vibehash app resume          # mine again
    vibehash app stop            # stop the app itself
    vibehash app status          # is it running, is it paused, what is the worker doing
    (the window's buttons and Settings give the same orders, plus avatar-on and avatar-off)
    vibehash app startup on|off  # start it when you log in (Windows, macOS and Linux), without the window

It is what the desktop app's launcher runs, and what a tray icon's menu drives, so everything here works with no
terminal: the wallet's password comes from $VIBEHASH_PASSWORD (a terminal is asked once if there is one), a new
wallet's recovery words go to a file only you can read, and the GPU build of PyTorch is installed without a question.
The window (window.py: the dashboard in a web view of its own) and the avatar (avatar.py, `--avatar`) are companion
processes the app starts once and closes when it leaves; `--no-window` keeps the window shut, which is how a
start-at-login entry runs, so logging in doesn't open a window (the avatar, if chosen, is the way back in).

How the commands reach it: the app keeps `state.json` in ~/.vibehash/app (where it is, since when, the worker's
feed, whether it is paused), with a heartbeat it writes every second, and reads a one-word `control` file (pause,
resume, stop) the other commands write. No process ids are signalled: on Windows `os.kill` with any signal but
Ctrl-C and Ctrl-Break kills outright, so the heartbeat says whether it is alive. The worker is a child process
(`python -m vibehash worker --plain --watch`), stopped the way Ctrl+C would stop it (Ctrl-Break on Windows, SIGTERM
elsewhere: cli._mine turns both into KeyboardInterrupt), so its round log is written out before it goes.
"""
from __future__ import annotations

import json
import os
import platform
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import presence, settings, window
from .settings import WATCH_PORT as DEFAULT_PORT

BEAT_S = 1.0  # how often the app writes its heartbeat
STALE_S = 10.0  # a heartbeat older than this: the app isn't running
STOP_GRACE_S = 25.0  # how long a worker gets to leave its round before it is killed
BACKOFF_S = (5, 15, 60)  # between restarts of a worker that keeps stopping
HOLD_CHECK_S = 5.0  # how often the automatic pauses look (presence.py: battery, a full-screen app, idle time)
HOLD_COOLDOWN_S = 60.0  # how long a reason to hold off must stay gone before the worker starts again: a game
#                          flicking out of full screen, or a mouse nudged, mustn't reload the model each time
LABEL = "org.vibehash.app"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"  # start at login under the launcher: a Run value, no console flash
ORDERS = ("pause", "resume", "stop", "avatar-on", "avatar-off")


def app_dir() -> Path:
    return settings.home() / "app"


def state_path() -> Path:
    return app_dir() / "state.json"


def control_path() -> Path:
    return app_dir() / "control"


def read_state() -> dict | None:
    try:
        return json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def running(state: dict | None = None, now: float | None = None) -> bool:
    """Whether an app is alive: its heartbeat is fresh."""
    state = read_state() if state is None else state
    now = time.time() if now is None else now
    return bool(state) and isinstance(state.get("beat"), (int, float)) and now - state["beat"] < STALE_S


def order(word: str) -> None:
    """Leave an order (pause, resume, stop) for the running app."""
    app_dir().mkdir(parents=True, exist_ok=True)
    tmp = control_path().with_suffix(".tmp")  # whole or not there: the app never reads half an order
    tmp.write_text(word + "\n", encoding="utf-8")
    os.replace(tmp, control_path())


def take_order() -> str | None:
    """The order left, if any, and remove it."""
    try:
        word = control_path().read_text(encoding="utf-8").strip()
        control_path().unlink()
    except OSError:
        return None
    return word or None


# --- getting ready ------------------------------------------------------------------------------------------------

def prepare(a, report=None) -> tuple[str, Path]:
    """Node, wallet and GPU support, without a question that can't be answered: -> (node url, wallet path).
    The same steps as `setup`, kept: a second run changes nothing. `report(step, fraction, **facts)`, if given,
    hears each step as it starts (the window shows it): node, wallet (with the wallet's address, file and words
    file), gpu, ready."""
    from . import cli, unattended
    from . import wallet as wallets

    report = report or (lambda step, fraction, **facts: None)
    report("node", 0.05)
    url = settings.chosen_node(getattr(a, "node", None)) or settings.PUBLIC_NODE
    url = cli.normalise_url(url)
    settings.save(node=url)
    print(f"node: {url}")

    report("wallet", 0.15)
    path = settings.wallet_path(getattr(a, "wallet", None)) or settings.default_wallet()
    address, pw = cli.ensure_wallet(path, "wallet: ", no_password=bool(getattr(a, "no_password", False)))
    words = unattended.words_file(address)
    report("gpu", 0.3, wallet={"address": address, "path": str(path), "words_file": str(words) if words.exists() else None})
    if pw is None and wallets.is_encrypted(path):
        # The worker is a child process that unlocks the wallet itself: it reads $VIBEHASH_PASSWORD, so the password
        # asked for here (once, at a terminal) is handed to it that way, after a check that it opens the wallet.
        pw = cli.password(f"Password for {path}: ")
        if pw is None:
            raise unattended.need(f"the password of {path}", "set VIBEHASH_PASSWORD to it")
        try:
            wallets.load(path, pw)  # a wrong one stops here, not in the worker over and over
        except wallets.WalletError as e:
            sys.exit(f"error: {e} ({path}); set VIBEHASH_PASSWORD to the wallet's password, or move the wallet aside "
                     "and run again to make a new one")
    if pw is not None:
        os.environ["VIBEHASH_PASSWORD"] = pw
    settings.save(wallet=str(path))

    if getattr(a, "gpu", "auto") == "skip":
        print("graphics card: skipped")
    elif settings.load().get("app_gpu_ready") and not getattr(a, "recheck_gpu", False):
        print("graphics card: checked before (app --recheck-gpu checks again)")
    else:
        print("graphics card:")
        if cli._setup_gpu(assume_yes=True):  # settled: ready, or nothing to install; a failed install is tried again
            settings.save(app_gpu_ready=True)
    report("ready", 0.95)
    return url, path


# --- running the worker --------------------------------------------------------------------------------------------

def worker_command(url: str, wallet: Path, port: int, device: str = "auto", job: str | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "vibehash", "worker", "--plain", "--watch", str(port), "--node", url,
           "--wallet", str(wallet), "--device", device]
    if job:
        cmd += ["--job", job]
    return cmd


class Supervisor:
    """Keeps one worker running (`command`), stops it on pause and stop orders, starts it again on resume or when it
    stopped on its own, and writes the app's state with a heartbeat."""

    def __init__(self, command: list[str], port: int, beat_s: float = BEAT_S, backoff=BACKOFF_S, grace_s: float = STOP_GRACE_S,
                 companions: list[list[str]] | None = None, avatar: list[str] | None = None, probe=None,
                 hold_check_s: float = HOLD_CHECK_S, cooldown_s: float = HOLD_COOLDOWN_S):
        self.command, self.port, self.beat_s, self.backoff, self.grace_s = list(command), port, beat_s, backoff, grace_s
        self.companions = [list(c) for c in companions or []]  # the window: started once, not kept alive
        self.friends: list[subprocess.Popen] = []
        self.avatar_cmd = list(avatar) if avatar else None  # the avatar: the same, but orders turn it on and off
        self.avatar: subprocess.Popen | None = None
        self.setup = {"step": "starting", "fraction": 0.0}  # what prepare() is doing, for the window
        self.wallet: dict | None = None
        self._started = False
        self._beating: threading.Event | None = None
        self._state_lock = threading.Lock()  # write_state, from the heartbeat thread and the main one
        self.child: subprocess.Popen | None = None
        self.paused = False
        self.restarts = 0
        self.started = time.time()
        self.worker_since: float | None = None
        self.next_start = 0.0  # when the worker may be started again (after it stopped on its own)
        self.events: list[str] = []
        # The automatic pauses (Settings: on battery, while a full-screen app runs, only when away): `held` is the
        # reason mining is held off now, apart from `paused`, which is the person's own Pause.
        self.probe = probe or presence.facts
        self.hold_check_s, self.cooldown_s = hold_check_s, cooldown_s
        self.held: str | None = None
        self._hold_checked = float("-inf")
        self._clear_since: float | None = None

    # -- the child --

    @staticmethod
    def spawn(command: list[str]) -> subprocess.Popen:
        """A child in a group (Windows) or session (elsewhere) of its own, so a Ctrl+C at the app's terminal doesn't
        also reach it directly: the app stops the worker, once, the way stop_worker does, and the companions when
        it leaves."""
        if os.name == "nt":
            return subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        return subprocess.Popen(command, start_new_session=True)

    def start_worker(self) -> None:
        self.child = self.spawn(self.command)
        self.worker_since = time.time()
        self.say(f"worker started (pid {self.child.pid}): {shlex.join(self.command)}")

    def stop_worker(self, why: str) -> None:
        child, self.child = self.child, None
        if child is None or child.poll() is not None:
            return
        try:
            if os.name == "nt":
                child.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                child.terminate()
        except OSError:  # no console to send Ctrl-Break through, say: then the hard way
            child.kill()
        deadline = time.time() + self.grace_s
        while child.poll() is None and time.time() < deadline:  # the heartbeat goes on while the worker leaves its round
            self.write_state(time.time())
            time.sleep(0.25)
        if child.poll() is None:
            child.kill()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self.worker_since = None
        self.say(f"worker stopped: {why}")

    def worker_alive(self) -> bool:
        return self.child is not None and self.child.poll() is None

    # -- the companions --

    def start_companions(self) -> None:
        """The window and the avatar (if wanted), once; before setup, so the window can show it."""
        if self._started:
            return
        self._started = True
        for companion in self.companions:
            try:  # processes of their own; no screen, and each says so and leaves, which is fine
                self.friends.append(self.spawn(companion))
            except OSError as e:
                self.say(f"couldn't start {shlex.join(companion)}: {e}")
        if self.avatar_cmd and settings.load().get("avatar", True):
            self.avatar_on()

    def avatar_alive(self) -> bool:
        return self.avatar is not None and self.avatar.poll() is None

    def avatar_on(self) -> None:
        if self.avatar_cmd and not self.avatar_alive():
            try:
                self.avatar = self.spawn(self.avatar_cmd)
            except OSError as e:
                self.say(f"couldn't start the avatar: {e}")

    def avatar_off(self) -> None:
        if self.avatar_alive():
            self.avatar.terminate()
        self.avatar = None

    # -- setup, as the window sees it --

    def report(self, step: str, fraction: float, **facts) -> None:
        """prepare()'s progress: into the state file, for the window."""
        self.setup = {"step": step, "fraction": fraction}
        app_dir().mkdir(parents=True, exist_ok=True)
        if facts.get("wallet"):
            self.wallet = facts["wallet"]
        self.say(f"setup: {step}")
        self.write_state(time.time())

    def beat_while_preparing(self) -> None:
        """A heartbeat through setup (minutes, when PyTorch is installed), so `app status` and a second `app` know
        this one is alive; run() takes over."""
        self._beating = threading.Event()
        app_dir().mkdir(parents=True, exist_ok=True)

        def beat(stop=self._beating):
            while not stop.wait(self.beat_s):
                self.write_state(time.time())

        threading.Thread(target=beat, daemon=True, name="app-beat").start()

    # -- the loop --

    def tick(self, now: float | None = None) -> bool:
        """One turn: take an order, keep the worker as it should be, write the heartbeat. False once told to stop."""
        now = time.time() if now is None else now
        word = take_order()
        if word == "stop":
            self.stop_worker("the app was told to stop")
            self.write_state(now, alive=False)
            return False
        if word == "avatar-on":
            self.avatar_on()
        elif word == "avatar-off":
            self.avatar_off()
        if word == "pause" and not self.paused:
            self.paused = True
            self.stop_worker("paused")
        elif word == "resume" and self.paused:
            self.paused = False
            self.next_start = 0.0
            self.say("resumed")
        self.update_hold(now)
        if self.held and self.worker_alive():
            self.stop_worker(f"held off: {self.held}")
        if not self.paused and not self.held and not self.worker_alive():
            if self.child is not None:  # it stopped on its own: say so, and wait a little longer each time
                code = self.child.returncode
                self.child = None
                self.worker_since = None
                wait = self.backoff[min(self.restarts, len(self.backoff) - 1)]
                self.restarts += 1
                self.next_start = now + wait
                self.say(f"worker stopped on its own (exit code {code}); starting it again in {wait} s")
            elif now >= self.next_start:
                self.start_worker()
        self.write_state(now)
        return True

    def update_hold(self, now: float) -> None:
        """Look (every hold_check_s) whether a setting says to hold mining off now, and let go once the reason has
        been gone for cooldown_s. A probe or settings file that fails never holds mining off."""
        if now - self._hold_checked < self.hold_check_s:
            return
        self._hold_checked = now
        try:
            why = presence.hold(settings.load(), self.probe() or {})
        except Exception:  # noqa: BLE001  (an OS call that fails: mine as if the setting were off)
            why = None
        if why:
            if why != self.held:
                self.say(f"holding mining off: {why}")
            self.held, self._clear_since = why, None
        elif self.held:
            self._clear_since = now if self._clear_since is None else self._clear_since
            if now - self._clear_since >= self.cooldown_s:
                self.say(f"carrying on ({self.held}: no longer)")
                self.held, self._clear_since = None, None

    def run(self, stop=None) -> None:
        from .cli import _stop_like_ctrl_c
        _stop_like_ctrl_c()  # `kill`, a service manager or a closed console window: the worker is stopped first
        if self._beating is not None:
            self._beating.set()
        app_dir().mkdir(parents=True, exist_ok=True)
        if not self._started:
            take_order()  # a stale order from an earlier app
        self.start_companions()
        self.setup = {"step": "mining", "fraction": 1.0}
        try:
            while (stop is None or not stop.is_set()) and self.tick():
                time.sleep(self.beat_s)
        except KeyboardInterrupt:
            self.stop_worker("Ctrl+C")
            self.write_state(time.time(), alive=False)
        finally:
            if self.worker_alive():
                self.stop_worker("the app is leaving")
                self.write_state(time.time(), alive=False)
            for friend in self.friends + ([self.avatar] if self.avatar_alive() else []):
                if friend.poll() is None:
                    friend.terminate()

    # -- what it says --

    def say(self, text: str) -> None:
        self.events = (self.events + [f"{time.strftime('%H:%M:%S')}  {text}"])[-20:]
        print(f"app: {text}", flush=True)

    def write_state(self, now: float, alive: bool = True) -> None:
        # One writer at a time: the heartbeat thread (beat_while_preparing) and report() both write while setup runs,
        # and through the one state.tmp a second writer could truncate it under the first one's os.replace, leaving
        # state.json empty or half (read_state() -> None: the app looked not running, the window saw no setup). The
        # state is read inside the lock too, so each file is one consistent picture.
        with self._state_lock:
            child = self.child
            state = {"beat": now if alive else 0, "started": self.started, "pid": os.getpid(), "paused": self.paused,
                     "watch": f"http://127.0.0.1:{self.port}/", "worker": {"running": self.worker_alive(),
                     "since": self.worker_since, "pid": child.pid if child else None, "restarts": self.restarts},
                     "setup": self.setup, "wallet": self.wallet, "avatar": self.avatar_alive(), "events": self.events,
                     "held": self.held}
            try:
                tmp = state_path().with_suffix(".tmp")
                tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
                os.replace(tmp, state_path())
            except OSError:
                pass


# --- the status line -----------------------------------------------------------------------------------------------

def status_text(now: float | None = None, fetch=None) -> str:
    """What `app status` prints: from state.json, and the worker's feed when it answers (`fetch(url) -> dict`)."""
    now = time.time() if now is None else now
    state = read_state()
    if not running(state, now):
        return "the app isn't running (start it with: vibehash app)"
    since = time.strftime("%Y-%m-%d %H:%M", time.localtime(state["started"]))
    lines = [f"the app is running since {since}" + (", paused" if state.get("paused") else "")]
    w = state.get("worker", {})
    if w.get("running"):
        lines.append(f"worker: running (pid {w.get('pid')}, started {w.get('restarts', 0)} times again), feed at {state.get('watch')}")
        try:
            from .cli import fmt
            feed = (fetch or _fetch)(state["watch"])
            for rig in feed.get("workers", []):
                acct = rig.get("account") or {}
                lines.append(f"  {rig['address']}: round {rig.get('round')}, {rig.get('activity')}; "
                             f"{rig['session']['rounds']} rounds and {rig['session']['units']} units this session; "
                             f"balance {fmt(acct.get('balance', 0))}, stake {fmt(acct.get('stake', 0))}")
        except Exception:  # noqa: BLE001  (a feed that doesn't answer is a line, not an error)
            lines.append("  (the feed isn't answering yet)")
    else:
        lines.append("worker: " + ("paused" if state.get("paused") else "starting"))
    lines += state.get("events", [])[-5:]
    return "\n".join(lines)


def _fetch(url: str) -> dict:
    import urllib.request
    with urllib.request.urlopen(url, timeout=3) as resp:
        return json.loads(resp.read())


def stop_and_wait(timeout: float = STOP_GRACE_S + 10) -> bool:
    """`app stop`: leave the order and wait for the heartbeat to go. -> whether it did."""
    if not running():
        return True
    order("stop")
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = read_state()
        if not state or state.get("beat", 0) == 0:
            return True
        time.sleep(0.5)
    return not running()


# --- starting when you log in --------------------------------------------------------------------------------------

def _winreg():
    try:
        import winreg
        return winreg
    except ImportError:
        return None


def startup_is_on(system: str | None = None) -> bool:
    """Whether a start-at-login entry exists (a file, or the Run value under the launcher)."""
    system = system or platform.system()
    if startup_file(system).exists():
        return True
    winreg = _winreg()
    if system == "Windows" and winreg is not None:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
                winreg.QueryValueEx(k, "VibeHash")
            return True
        except OSError:
            pass
    return False


def startup_file(system: str | None = None) -> Path:
    system = system or platform.system()
    if system == "Windows":
        appdata = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        return appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "VibeHash.cmd"
    if system == "Darwin":
        return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    return Path.home() / ".config" / "systemd" / "user" / "vibehash.service"


def startup_on(system: str | None = None, run=None, avatar: bool = False) -> Path:
    """Start `vibehash app` at login: a Startup-folder script (Windows), a LaunchAgent (macOS) or a systemd user
    service (Linux), without the window (`--no-window`: logging in shouldn't open one; the avatar or `vibehash window`
    opens it). `avatar`: with the avatar, as the app was started. -> the file written."""
    system = system or platform.system()
    run = run or subprocess.run
    path = startup_file(system)
    path.parent.mkdir(parents=True, exist_ok=True)
    home = settings.home()
    # Under the downloadable launcher (it says where it is), start the launcher, so its daily update check still
    # happens (it passes --avatar itself, and hands what follows -- to the app); otherwise this Python with the app command.
    launcher = os.environ.get("VIBEHASH_LAUNCHER")
    exe, args = ((launcher, ["--", "--no-window"]) if launcher
                 else (sys.executable, ["-m", "vibehash", "app", "--no-window"] + (["--avatar"] if avatar else [])))
    winreg = _winreg()
    if system == "Windows" and launcher and winreg is not None:
        # The launcher is a windowed program: a Run value starts it with no console flashing by, as a Startup-folder
        # .cmd would (which an earlier version wrote: removed here).
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            winreg.SetValueEx(k, "VibeHash", 0, winreg.REG_SZ, f'"{exe}" {" ".join(args)}')
        path.unlink(missing_ok=True)
        return Path("HKCU") / RUN_KEY / "VibeHash"
    if system == "Windows":
        # cmd.exe reads a batch file in the OEM codepage, and a text-mode write would turn \r\n into \r\r\n
        text = f'@echo off\r\nset "VIBEHASH_HOME={home}"\r\nstart "Vibe Hash" /min "{exe}" {" ".join(args)}\r\n'.replace(' \r\n', '\r\n')
        path.write_bytes(text.encode("oem" if os.name == "nt" else "utf-8", "replace"))
    elif system == "Darwin":
        from xml.sax.saxutils import escape  # a path with & or < in it is still a plist
        log = escape(str(home / "app" / "app.log"))
        argv = "".join(f"<string>{escape(a)}</string>" for a in [exe, *args])
        path.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key><array>{argv}</array>
  <key>EnvironmentVariables</key><dict><key>VIBEHASH_HOME</key><string>{escape(str(home))}</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><false/>
  <key>StandardOutPath</key><string>{log}</string>
  <key>StandardErrorPath</key><string>{log}</string>
</dict></plist>
""", encoding="utf-8")
        run(["launchctl", "load", "-w", str(path)], check=False)
    else:
        path.write_text(f"""[Unit]
Description=Vibe Hash: mine by training AI models
After=network-online.target

[Service]
ExecStart={" ".join(shlex.quote(a) for a in [exe, *args])}
Environment="VIBEHASH_HOME={home}"
Restart=no

[Install]
WantedBy=default.target
""", encoding="utf-8")
        run(["systemctl", "--user", "daemon-reload"], check=False)
        run(["systemctl", "--user", "enable", "--now", "vibehash.service"], check=False)
    return path


def startup_off(system: str | None = None, run=None) -> Path:
    system = system or platform.system()
    run = run or subprocess.run
    path = startup_file(system)
    winreg = _winreg()
    if system == "Windows" and winreg is not None:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
                winreg.DeleteValue(k, "VibeHash")
        except OSError:
            pass
    if system == "Darwin" and path.exists():
        run(["launchctl", "unload", "-w", str(path)], check=False)
    elif system == "Linux" and path.exists():
        run(["systemctl", "--user", "disable", "--now", "vibehash.service"], check=False)
    path.unlink(missing_ok=True)
    return path


# --- the command ---------------------------------------------------------------------------------------------------

def main(a) -> None:
    """`vibehash app ...`, as cli.py dispatches it."""
    sub = getattr(a, "acmd", None) or "start"
    if sub == "status":
        print(status_text())
    elif sub in ("pause", "resume"):
        if not running():
            sys.exit("the app isn't running (start it with: vibehash app)")
        order(sub)
        print(f"{sub}d" if sub == "pause" else "resuming")
    elif sub == "stop":
        print("stopped" if stop_and_wait() else "the app didn't stop in time")
    elif sub == "startup":
        if a.mode == "on":
            print(f"will start when you log in: {startup_on(avatar=bool(getattr(a, 'avatar', False)))}")
        else:
            print(f"won't start when you log in (removed {startup_off()})")
    else:
        if running():
            since = time.strftime('%H:%M', time.localtime(read_state()['started']))
            if getattr(a, "no_window", False):
                sys.exit(f"an app is already running (since {since}); `vibehash app status` says what it is doing, "
                         "`vibehash app stop` stops it")
            # The Start menu, a second double-click: the way back into the running one is its window
            port = a.watch if getattr(a, "watch", None) else DEFAULT_PORT
            print(f"an app is already running (since {since}); opening its window")
            sys.exit(subprocess.call(window.command(port)))
        port = a.watch if getattr(a, "watch", None) else DEFAULT_PORT
        companions = [] if getattr(a, "no_window", False) else [window.command(port)]
        avatar = [sys.executable, "-m", "vibehash", "avatar", "--watch", str(port)] if getattr(a, "avatar", False) else None
        sup = Supervisor([], port, companions=companions, avatar=avatar)
        app_dir().mkdir(parents=True, exist_ok=True)
        take_order()  # a stale order from an earlier app
        sup.start_companions()  # the window first, so setup (minutes, the first time) is watched, not waited for blind
        sup.beat_while_preparing()
        try:
            url, wallet = prepare(a, report=sup.report)
        except BaseException as e:
            sup._beating.set()
            why = str(e) if isinstance(e, (SystemExit, Exception)) and str(e) else type(e).__name__
            sup.setup = {"step": "failed", "fraction": 0.0, "why": why[:400]}
            sup.write_state(time.time(), alive=False)
            for friend in sup.friends + ([sup.avatar] if sup.avatar_alive() else []):
                if friend.poll() is None:
                    friend.terminate()
            raise
        sup.command = worker_command(url, wallet, port, getattr(a, "device", "auto"), getattr(a, "job", None))
        page = f"http://127.0.0.1:{port}/dashboard"
        where = f"the window shows it (also at {page})" if not getattr(a, "no_window", False) else f"the dashboard is at {page}"
        print(f"mining; {where} and `vibehash app status` sums it up. Ctrl+C stops it.")
        sup.run()
        print("app stopped")

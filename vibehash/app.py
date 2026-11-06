"""`vibehash app`: mining as something you start once and forget. It gets the computer ready without asking (the node,
a wallet, GPU support), starts the worker with its live feed (`worker --watch`, watch.py), starts it again if it
stops, and takes orders from the other `app` commands:

    vibehash app                 # get ready, join the network, mine; Ctrl+C stops it
    vibehash app pause           # stop mining for now (the worker finishes nothing: it leaves its round)
    vibehash app resume          # mine again
    vibehash app stop            # stop the app itself
    vibehash app status          # is it running, is it paused, what is the worker doing
    vibehash app startup on|off  # start it when you log in (Windows, macOS and Linux)

It is what the desktop app's launcher runs, and what a tray icon's menu drives, so everything here works with no
terminal: the wallet's password comes from $VIBEHASH_PASSWORD (a terminal is asked once if there is one), a new
wallet's recovery words go to a file only you can read, and the GPU build of PyTorch is installed without a question.

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
import time
from pathlib import Path

from . import settings
from .watch import DEFAULT_PORT

BEAT_S = 1.0  # how often the app writes its heartbeat
STALE_S = 10.0  # a heartbeat older than this: the app isn't running
STOP_GRACE_S = 25.0  # how long a worker gets to leave its round before it is killed
BACKOFF_S = (5, 15, 60)  # between restarts of a worker that keeps stopping
LABEL = "org.vibehash.app"


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

def prepare(a) -> tuple[str, Path]:
    """Node, wallet and GPU support, without a question that can't be answered: -> (node url, wallet path).
    The same steps as `setup`, kept: a second run changes nothing."""
    from . import cli, unattended
    from . import wallet as wallets

    url = settings.chosen_node(getattr(a, "node", None)) or settings.PUBLIC_NODE
    url = cli.normalise_url(url)
    settings.save(node=url)
    print(f"node: {url}")

    path = settings.wallet_path(getattr(a, "wallet", None)) or settings.default_wallet()
    address, pw = cli.ensure_wallet(path, "wallet: ", no_password=bool(getattr(a, "no_password", False)))
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

    def __init__(self, command: list[str], port: int, beat_s: float = BEAT_S, backoff=BACKOFF_S, grace_s: float = STOP_GRACE_S):
        self.command, self.port, self.beat_s, self.backoff, self.grace_s = list(command), port, beat_s, backoff, grace_s
        self.child: subprocess.Popen | None = None
        self.paused = False
        self.restarts = 0
        self.started = time.time()
        self.worker_since: float | None = None
        self.next_start = 0.0  # when the worker may be started again (after it stopped on its own)
        self.events: list[str] = []

    # -- the child --

    def start_worker(self) -> None:
        # In a group (Windows) or session (elsewhere) of its own, so a Ctrl+C at the app's terminal doesn't also reach
        # the worker directly: the app stops it, once, the way stop_worker does.
        if os.name == "nt":
            self.child = subprocess.Popen(self.command, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            self.child = subprocess.Popen(self.command, start_new_session=True)
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

    # -- the loop --

    def tick(self, now: float | None = None) -> bool:
        """One turn: take an order, keep the worker as it should be, write the heartbeat. False once told to stop."""
        now = time.time() if now is None else now
        word = take_order()
        if word == "stop":
            self.stop_worker("the app was told to stop")
            self.write_state(now, alive=False)
            return False
        if word == "pause" and not self.paused:
            self.paused = True
            self.stop_worker("paused")
        elif word == "resume" and self.paused:
            self.paused = False
            self.next_start = 0.0
            self.say("resumed")
        if not self.paused and not self.worker_alive():
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

    def run(self, stop=None) -> None:
        from .cli import _stop_like_ctrl_c
        _stop_like_ctrl_c()  # `kill`, a service manager or a closed console window: the worker is stopped first
        app_dir().mkdir(parents=True, exist_ok=True)
        take_order()  # a stale order from an earlier app
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

    # -- what it says --

    def say(self, text: str) -> None:
        self.events = (self.events + [f"{time.strftime('%H:%M:%S')}  {text}"])[-20:]
        print(f"app: {text}", flush=True)

    def write_state(self, now: float, alive: bool = True) -> None:
        state = {"beat": now if alive else 0, "started": self.started, "pid": os.getpid(), "paused": self.paused,
                 "watch": f"http://127.0.0.1:{self.port}/", "worker": {"running": self.worker_alive(),
                 "since": self.worker_since, "pid": self.child.pid if self.child else None, "restarts": self.restarts},
                 "events": self.events}
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

def startup_file(system: str | None = None) -> Path:
    system = system or platform.system()
    if system == "Windows":
        appdata = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        return appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "VibeHash.cmd"
    if system == "Darwin":
        return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    return Path.home() / ".config" / "systemd" / "user" / "vibehash.service"


def startup_on(system: str | None = None, run=subprocess.run) -> Path:
    """Start `vibehash app` at login: a Startup-folder script (Windows), a LaunchAgent (macOS) or a systemd user
    service (Linux). -> the file written."""
    system = system or platform.system()
    path = startup_file(system)
    path.parent.mkdir(parents=True, exist_ok=True)
    home = settings.home()
    # Under the downloadable launcher (it says where it is), start the launcher, so its daily update check still
    # happens; otherwise this Python with the app command.
    launcher = os.environ.get("VIBEHASH_LAUNCHER")
    exe, args = (launcher, []) if launcher else (sys.executable, ["-m", "vibehash", "app"])
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


def startup_off(system: str | None = None, run=subprocess.run) -> Path:
    system = system or platform.system()
    path = startup_file(system)
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
            print(f"will start when you log in: {startup_on()}")
        else:
            print(f"won't start when you log in (removed {startup_off()})")
    else:
        if running():
            sys.exit(f"an app is already running (since {time.strftime('%H:%M', time.localtime(read_state()['started']))}); "
                     "`vibehash app status` says what it is doing, `vibehash app stop` stops it")
        url, wallet = prepare(a)
        port = a.watch if getattr(a, "watch", None) else DEFAULT_PORT
        cmd = worker_command(url, wallet, port, getattr(a, "device", "auto"), getattr(a, "job", None))
        print(f"mining; the worker's feed is at http://127.0.0.1:{port}/ and `vibehash app status` sums it up. Ctrl+C stops it.")
        Supervisor(cmd, port).run()
        print("app stopped")

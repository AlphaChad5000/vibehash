"""The app's window: the dashboard (data/dashboard.html) in a window of its own, not a browser tab.

    vibehash window               # the window, for a running `vibehash app` or `worker --watch` (WATCH_PORT)
    vibehash app                  # opens it as it starts (`--no-window` not to); a click on the avatar opens it too

The window is the system's own web view (Edge WebView2 on Windows, WebKit on macOS, WebKitGTK or Qt on Linux) through
pywebview, a small package the `app` extra brings (`pip install vibehash[app]`; the downloadable launcher has it
inside and hosts the window itself, `VibeHash.exe --window PORT`, so it carries the app's name and icon). No browser
engine is shipped: Windows 10 and 11 and macOS have theirs; Linux needs the distribution's python3-gi and WebKit2GTK
(or PySide6). Where none of that is there, the dashboard opens in the browser instead, as it always could, and this
says so.

The page is the one watch.py serves at /dashboard, loaded here from the package itself (pywebview serves it from a
port of its own), told where the worker's feed is (window.FEED), so it is up before the worker is: through the first
run's setup it shows the progress, the new wallet and its recovery words, and a tour. What the page may ask of the
app goes through `Api` (pywebview's js_api, reachable from this window only, never from a browser): the app's state
(~/.vibehash/app/state.json), orders (pause, resume, stop, the avatar on or off, as `vibehash app ...` would give
them), the wallet's address and words file, and settings (the avatar, start at login, whether the welcome was seen).

Closing the window doesn't stop mining: the Stop button, the avatar's menu or `vibehash app stop` does. It is its own
process (`command`), so a window that is closed, or a web view that fails, never touches the worker. One window per
feed: the process holds a lock file in ~/.vibehash/app while it runs, and a second one (the app's own window is open
and the avatar is clicked, say) sees it and leaves quietly.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from . import app as app_module
from . import settings
from .settings import WATCH_PORT as DEFAULT_PORT

TITLE = "Vibe Hash"
SIZE = (1100, 760)
MIN_SIZE = (760, 520)
DASHBOARD = Path(__file__).resolve().parent / "data" / "dashboard.html"
ORDERS = ("pause", "resume", "stop", "avatar-on", "avatar-off")


def feed_url(port: int = DEFAULT_PORT) -> str:
    return f"http://127.0.0.1:{port}"


def dashboard_url(port: int = DEFAULT_PORT) -> str:
    return feed_url(port) + "/dashboard"


def command(port: int = DEFAULT_PORT, env: dict | None = None) -> list[str]:
    """How the app and the avatar start the window: a process of its own; the launcher itself when there is one
    (VIBEHASH_LAUNCHER, set by it: the window then has the launcher's name and icon)."""
    env = os.environ if env is None else env
    launcher = env.get("VIBEHASH_LAUNCHER")
    if launcher and Path(launcher).exists():
        return [launcher, "--window", str(port)]
    return [sys.executable, "-m", "vibehash", "window", "--watch", str(port)]


def page(port: int = DEFAULT_PORT) -> str:
    """The dashboard page, told where the feed is (the page's fetches are relative when the feed serves it)."""
    text = DASHBOARD.read_text(encoding="utf-8")
    return text.replace("<head>", f"<head>\n<script>window.FEED = {json.dumps(feed_url(port))};</script>", 1)


class Api:
    """What the page in the window may ask of the app (pywebview js_api): every public method here, from JavaScript
    as `pywebview.api.name(...)`, each answering a plain dict."""

    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port

    # -- the app --

    def state(self) -> dict:
        """~/.vibehash/app/state.json with `running` added (a fresh heartbeat), or {"running": False}."""
        s = app_module.read_state() or {}
        return {"running": app_module.running(s), **{k: v for k, v in s.items() if k != "pid"}}

    def order(self, word: str) -> dict:
        """pause, resume, stop, avatar-on, avatar-off: as `vibehash app ...` would leave it for the running app."""
        if word not in ORDERS:
            return {"ok": False, "why": f"no such order: {word}"}
        if not app_module.running():
            return {"ok": False, "why": "the app isn't running"}
        app_module.order(word)
        return {"ok": True, "order": word}

    # -- the account --

    def wallet(self) -> dict:
        """The wallet the app runs with: its address and file (from the app's state, which setup writes), and the
        recovery words while their file is still there (setup saves a new wallet's words to a file only this user can
        read; the page shows them once and says to keep them somewhere safe)."""
        info = dict((app_module.read_state() or {}).get("wallet") or {})
        words_file = info.get("words_file")
        words: list[str] = []
        if words_file:
            try:
                for line in Path(words_file).read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split(". ", 1)
                    if len(parts) == 2 and parts[0].strip().isdigit():
                        words.append(parts[1].strip())
            except OSError:
                info["words_file"] = None
        info["words"] = words
        return info

    # -- settings --

    def settings(self) -> dict:
        cfg = settings.load()
        return {"avatar": bool(cfg.get("avatar", True)), "welcomed": bool(cfg.get("welcomed", False)),
                "startup": app_module.startup_is_on(), "home": str(settings.home()), "version": _version()}

    def set_avatar(self, on: bool) -> dict:
        """The floating avatar: kept in settings for the next start, and told to the running app now."""
        settings.save(avatar=bool(on))
        if app_module.running():
            app_module.order("avatar-on" if on else "avatar-off")
        return {"ok": True, "avatar": bool(on)}

    def set_startup(self, on: bool) -> dict:
        """Start at login, as `vibehash app startup on|off`."""
        try:
            where = app_module.startup_on(avatar=bool(settings.load().get("avatar", True))) if on else app_module.startup_off()
        except OSError as e:
            return {"ok": False, "why": str(e)}
        return {"ok": True, "startup": bool(on), "where": str(where)}

    def welcomed(self) -> dict:
        settings.save(welcomed=True)
        return {"ok": True}

    def welcome_again(self) -> dict:
        settings.save(welcomed=False)
        return {"ok": True}

    def open_home(self) -> dict:
        """~/.vibehash in the file manager (the wallet, the words file, the logs)."""
        home = settings.home()
        try:
            if sys.platform == "win32":
                os.startfile(str(home))  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(home)])
            else:
                subprocess.Popen(["xdg-open", str(home)])
        except OSError as e:
            return {"ok": False, "why": str(e)}
        return {"ok": True, "home": str(home)}


def _version() -> str:
    try:
        from . import _version as v
        return v()
    except Exception:  # noqa: BLE001
        return "unknown"


class Lock:
    """One window per feed: an exclusive lock on a file in the app's folder, held for as long as the process runs
    (the system lets it go when the process ends, however it ends). `held` says whether this process got it."""

    def __init__(self, port: int):
        self.path = app_module.app_dir() / f"window-{port}.lock"
        self.file = None
        self.held = False

    def acquire(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = open(self.path, "a+")  # noqa: SIM115  (held open on purpose)
            if os.name == "nt":
                import msvcrt
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.held = True
        except OSError:  # another window holds it; or no folder to lock in, and then there is no one to stop
            self.held = self.file is None
            if self.file is not None and not self.held:
                self.file.close()
                self.file = None
        return self.held

    def release(self) -> None:
        if self.file is not None:
            try:
                if os.name == "nt" and self.held:
                    import msvcrt
                    self.file.seek(0)
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
                self.file.close()
            except OSError:
                pass
            self.file = None


def show(port: int = DEFAULT_PORT, fallback=webbrowser.open, say=lambda text: print(text, file=sys.stderr)) -> str:
    """The window, until it is closed. -> "window"; "browser" when the web view isn't there and the browser got
    the feed's page instead; "open" when another window for this feed is already up; "none" with no screen at all."""
    url = dashboard_url(port)
    if sys.platform not in ("win32", "darwin") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        say("the window needs a screen; none here (the dashboard is at " + url + ")")  # not a console browser over the app
        return "none"
    lock = Lock(port)
    if not lock.acquire():
        say("the window is already open")
        return "open"
    try:
        try:
            import webview
        except ImportError:
            say("the window needs pywebview (pip install vibehash[app]); opening the dashboard in the browser instead")
            fallback(url)
            return "browser"
        try:
            webview.create_window(TITLE, html=page(port), js_api=Api(port), width=SIZE[0], height=SIZE[1], min_size=MIN_SIZE,
                                  text_select=True)
            webview.start(http_server=True)  # served from a port of pywebview's own, so the page has an origin the feed allows
        except KeyboardInterrupt:
            pass
        except Exception as e:  # noqa: BLE001  (no WebKitGTK or Qt on this Linux, say: pywebview raises its own kinds)
            say(f"the window can't open here ({e}); opening the dashboard in the browser instead")
            fallback(url)
            return "browser"
        return "window"
    finally:
        lock.release()


def main(a) -> None:
    show(getattr(a, "watch", None) or DEFAULT_PORT)


if __name__ == "__main__":
    show(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT)

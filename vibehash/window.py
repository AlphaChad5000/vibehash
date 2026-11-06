"""The app's window: the dashboard (data/dashboard.html, which watch.py serves at /dashboard) in a window of its own,
not a browser tab.

    vibehash window               # the window, for a running `vibehash app` or `worker --watch` (DEFAULT_PORT)
    vibehash app                  # opens it as it starts (`--no-window` not to); a click on the avatar opens it too

The window is the system's own web view (Edge WebView2 on Windows, WebKit on macOS, WebKitGTK or Qt on Linux) through
pywebview, a small package the `app` extra brings (`pip install vibehash[app]`; the downloadable launcher installs
it). No browser engine is shipped: Windows 10 and 11 and macOS have theirs; Linux needs the distribution's
python3-gi and WebKit2GTK (or PySide6). Where none of that is there, the dashboard opens in the browser instead, as
it always could, and this says so.

It shows a "starting" page until the worker's feed answers (the worker takes a few seconds to load), then the
dashboard. Closing the window doesn't stop mining: the avatar's menu or `vibehash app stop` does. It is its own
process (`command`), so a window that is closed, or a web view that fails, never touches the worker. One window
per feed: the process holds a lock file in ~/.vibehash/app while it runs, and a second one (the app's own window
is open and the avatar is clicked, say) sees it and leaves quietly.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
import webbrowser

from . import app as app_module
from .watch import DEFAULT_PORT

TITLE = "Vibe Hash"
SIZE = (1100, 720)
MIN_SIZE = (720, 480)
WAIT_S = 1.0  # between looks for the feed while the worker starts
STARTING = """<!doctype html><html><head><meta charset="utf-8"><title>Vibe Hash</title><style>
html,body{height:100%;margin:0;background:#0e1016;color:#e7e9ee;font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
body{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;text-align:center;padding:0 24px}
svg{width:72px;height:96px}.dim{color:#8a8f9c;font-size:14px;max-width:34em}
@keyframes swing{0%,100%{transform:rotate(-6deg)}50%{transform:rotate(6deg)}}svg{animation:swing 1.6s ease-in-out infinite;transform-origin:50% 90%}
</style></head><body>
<svg viewBox="0 0 3 4" aria-hidden="true"><rect x="0" y="0" width="1" height="3" fill="#f1b23c"/><rect x="2" y="0" width="1" height="3" fill="#f1b23c"/><rect x="1" y="3" width="1" height="1" fill="#f1b23c"/></svg>
<div>Vibe Hash is starting&hellip;</div>
<div class="dim">The worker is loading. This becomes the dashboard as soon as its feed answers; closing this window
doesn't stop mining (the avatar's menu or <code>vibehash app stop</code> does).</div>
</body></html>"""


def dashboard_url(port: int = DEFAULT_PORT) -> str:
    return f"http://127.0.0.1:{port}/dashboard"


def command(port: int = DEFAULT_PORT) -> list[str]:
    """How the app and the avatar start the window: a process of its own."""
    return [sys.executable, "-m", "vibehash", "window", "--watch", str(port)]


_LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 127.0.0.1 never goes through a proxy


def feed_ready(url: str, timeout: float = 3.0) -> bool:
    """Does the feed behind the dashboard URL answer yet? Asked of /events, which answers from memory (the feed's
    root asks the node for its chain id first, which can take longer than this)."""
    try:
        with _LOCAL.open(url.rsplit("/", 1)[0] + "/events?after=0", timeout=timeout) as r:
            r.read()
            return r.status == 200
    except Exception:  # noqa: BLE001  (refused, timed out, half an answer from a server just starting: not yet)
        return False


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


def wait_then_load(window, url: str, closed: threading.Event, ready=None, sleep=None) -> bool:
    """Runs beside the web view: once the feed answers, show the dashboard. False if the window closed first."""
    ready, sleep = ready or feed_ready, sleep or time.sleep
    while not closed.is_set():
        if ready(url):
            window.load_url(url)
            return True
        sleep(WAIT_S)
    return False


def show(port: int = DEFAULT_PORT, fallback=webbrowser.open, say=lambda text: print(text, file=sys.stderr)) -> str:
    """The window, until it is closed. -> "window"; "browser" when the web view isn't there and the browser got
    the page instead; "open" when another window for this feed is already up; "none" with no screen at all."""
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
        closed = threading.Event()
        try:
            win = webview.create_window(TITLE, html=STARTING, width=SIZE[0], height=SIZE[1], min_size=MIN_SIZE, text_select=True)
            win.events.closed += closed.set
            webview.start(wait_then_load, (win, url, closed))
        except KeyboardInterrupt:
            closed.set()
        except Exception as e:  # noqa: BLE001  (no WebKitGTK or Qt on this Linux, say: pywebview raises its own kinds)
            closed.set()
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

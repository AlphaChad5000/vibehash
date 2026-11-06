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
process (`command`), so a window that is closed, or a web view that fails, never touches the worker.
"""
from __future__ import annotations

import sys
import threading
import time
import urllib.request
import webbrowser

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


def feed_ready(url: str, timeout: float = 3.0) -> bool:
    """Does the feed behind the dashboard URL answer yet? Asked of /events, which answers from memory (the feed's
    root asks the node for its chain id first, which can take longer than this)."""
    try:
        with urllib.request.urlopen(url.rsplit("/", 1)[0] + "/events?after=0", timeout=timeout) as r:
            r.read()
            return r.status == 200
    except (OSError, ValueError):
        return False


def wait_then_load(window, url: str, closed: threading.Event, ready=None, sleep=time.sleep) -> bool:
    """Runs beside the web view: once the feed answers, show the dashboard. False if the window closed first."""
    ready = ready or feed_ready
    while not closed.is_set():
        if ready(url):
            window.load_url(url)
            return True
        sleep(WAIT_S)
    return False


def open(port: int = DEFAULT_PORT, fallback=webbrowser.open, say=lambda text: print(text, file=sys.stderr)) -> str:  # noqa: A001  (the module's one verb)
    """The window, until it is closed. -> "window", or "browser" when the web view isn't there and the browser got
    the page instead."""
    url = dashboard_url(port)
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
    except Exception as e:  # noqa: BLE001  (no WebKitGTK or Qt on this Linux, say: pywebview raises its own kinds)
        closed.set()
        say(f"the window can't open here ({e}); opening the dashboard in the browser instead")
        fallback(url)
        return "browser"
    return "window"


def main(a) -> None:
    open(getattr(a, "watch", None) or DEFAULT_PORT)


if __name__ == "__main__":
    open(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT)

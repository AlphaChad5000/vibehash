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
import re
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

from . import app as app_module
from . import presence, settings
from .settings import WATCH_PORT as DEFAULT_PORT

TITLE = "Vibe Hash"
SIZE = (1100, 760)
MIN_SIZE = (760, 520)
DASHBOARD = Path(__file__).resolve().parent / "data" / "dashboard.html"
ORDERS = ("pause", "resume", "stop", "avatar-on", "avatar-off")
# The only places the page can send someone, by name (Api.open_link): never a URL the page makes up, so nothing in the
# page, or put into it, can open an arbitrary address. They open in the system's browser, never inside this window.
LINKS = {"site": "https://vibehash.org", "whitepaper": "https://vibehash.org/whitepaper.pdf",
         "source": "https://github.com/VibeHash/vibehash", "releases": "https://github.com/VibeHash/vibehash/releases"}
OPEN = webbrowser.open  # how a link leaves the app (tests replace it)
PYPI = "https://pypi.org/pypi/vibehash/json"
UPDATE_EVERY_S = 6 * 3600.0
# Switches the Settings tab may set in config.json (and idle_minutes, one of presence.IDLE_CHOICES).
OPTIONS = presence.OPTIONS + ("reduced_motion",)
CAP_GB = (5, 2000)  # the model store's cap, as Settings' slider allows it
# Keys whose values a diagnostic report never carries, at any depth (redact).
SECRET = re.compile(r"word|mnemonic|seed|private|secret|password|passphrase|token|mfa", re.I)


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


def page_file(port: int = DEFAULT_PORT) -> Path:
    """page(port) written to the app's folder, in a folder of its own: pywebview serves the file's folder, and that
    folder holds nothing else."""
    folder = app_module.app_dir() / "window"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"dashboard-{port}.html"
    path.write_text(page(port), encoding="utf-8")
    return path


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
                "startup": app_module.startup_is_on(), "home": str(settings.home()), "version": _version(),
                **{k: bool(cfg.get(k, False)) for k in OPTIONS}, "idle_minutes": presence.idle_minutes(cfg),
                "idle_choices": list(presence.IDLE_CHOICES), "supported": presence.supported(),
                "words_file": _words_file_present()}

    def set_option(self, name: str, value) -> dict:
        """One of Settings' switches (OPTIONS: the automatic pauses, reduced motion), or idle_minutes. The running app
        reads them every few seconds (app.Supervisor.update_hold), so they apply without a restart."""
        if name == "idle_minutes":
            if isinstance(value, bool) or value not in presence.IDLE_CHOICES:
                return {"ok": False, "why": f"idle_minutes is one of {list(presence.IDLE_CHOICES)}"}
            settings.save(idle_minutes=int(value))
            return {"ok": True, name: int(value)}
        if name not in OPTIONS or not isinstance(value, bool):
            return {"ok": False, "why": f"no such setting: {name}"}
        settings.save(**{name: value})
        return {"ok": True, name: value}

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

    def delete_words(self) -> dict:
        """Delete the file setup saved a new wallet's recovery words in, once they are written down elsewhere: until
        then anyone who can read this user's files can take the wallet's coins with them. The wallet stays."""
        info = (app_module.read_state() or {}).get("wallet") or {}
        path = info.get("words_file")
        if not path:
            return {"ok": False, "why": "there is no recovery-words file"}
        try:
            Path(path).unlink()
        except FileNotFoundError:
            return {"ok": True, "deleted": False}
        except OSError as e:
            return {"ok": False, "why": str(e)}
        return {"ok": True, "deleted": True}

    # -- links out --

    def links(self) -> dict:
        return dict(LINKS)

    def open_link(self, name: str) -> dict:
        """One of LINKS, by name, in the system's browser (the page asks first: "Leaving Vibe Hash")."""
        url = LINKS.get(name) if isinstance(name, str) else None
        if url is None:
            return {"ok": False, "why": "not a link the app opens"}
        OPEN(url)
        return {"ok": True, "url": url}

    # -- storage --

    def cache(self) -> dict:
        """The model store (big downloads: modelstore.py): where, how much it holds, its cap."""
        from . import modelstore
        try:
            store = modelstore.Store.from_settings()
            return {"ok": True, "folder": str(store.folder), "used": store.usage(), "cap": store.cap,
                    "files": len(store.files), "cap_range_gb": list(CAP_GB)}
        except OSError as e:
            return {"ok": False, "why": str(e)}

    def set_cache_cap(self, gb) -> dict:
        from . import modelstore
        if isinstance(gb, bool) or not isinstance(gb, (int, float)) or not CAP_GB[0] <= gb <= CAP_GB[1]:
            return {"ok": False, "why": f"the cap is {CAP_GB[0]} to {CAP_GB[1]} GB"}
        store = modelstore.Store.from_settings(cap=int(gb * modelstore.GB))
        store.remember()
        return {"ok": True, "cap": store.cap}

    def clear_cache(self) -> dict:
        """Empty the model store, only while nothing is mining: a worker mid-round may be reading those files."""
        from . import modelstore
        s = app_module.read_state() or {}
        if app_module.running(s) and not s.get("paused") and (s.get("worker") or {}).get("running"):
            return {"ok": False, "why": "pause mining first: the worker may be using these files"}
        try:
            return {"ok": True, "freed": modelstore.Store.from_settings().clear()}
        except OSError as e:
            return {"ok": False, "why": str(e)}

    # -- the network --

    def node(self) -> dict:
        """The node the app mines with: how this computer trusts it, and how long it takes to answer."""
        from .client import NodeClient, NodeError
        url = settings.node_url()
        ca, fingerprint = settings.tls_trust(url)
        trust = ("not encrypted (plain http)" if url.startswith("http://")
                 else f"pinned certificate, SHA-256 {fingerprint[:16]}..." if fingerprint
                 else f"certificate checked against {ca}" if ca else "certificate checked by this system's authorities")
        out = {"url": url, "trust": trust, "encrypted": url.startswith("https://"), "official": url == settings.PUBLIC_NODE}
        t0 = time.perf_counter()
        try:
            st = NodeClient(url, timeout=5).get("/status")
            out.update(ok=True, ms=round(1000 * (time.perf_counter() - t0)), chain_id=st.get("chain_id"),
                       height=st.get("height"), version=st.get("version"))
        except (NodeError, OSError, ValueError) as e:
            out.update(ok=False, why=str(e)[:200])
        return out

    # -- updates --

    _update: dict | None = None  # the last answer, shared by every Api in the process: {"at", "answer"}

    def update(self) -> dict:
        """Whether PyPI has a newer Vibe Hash than this one (asked at most every UPDATE_EVERY_S). The launcher
        installs it the next time the app starts, so the page's banner says to restart."""
        now = time.time()
        cached = Api._update
        if cached and now - cached["at"] < UPDATE_EVERY_S:
            return cached["answer"]
        mine = _version()
        try:
            latest = PYPI_LATEST()
        except (OSError, ValueError, KeyError):
            return {"current": mine, "latest": None, "newer": False}
        answer = {"current": mine, "latest": latest, "newer": newer(latest, mine)}
        Api._update = {"at": now, "answer": answer}
        return answer

    # -- the record --

    def history(self, limit: int = 20) -> dict:
        """This wallet's last sealed rounds from the worker's round log (records.RoundLog's rounds.jsonl), newest
        first."""
        from . import records
        address = ((app_module.read_state() or {}).get("wallet") or {}).get("address")
        limit = max(1, min(limit if isinstance(limit, int) and not isinstance(limit, bool) else 20, 200))
        try:
            lines = (records.logs_dir() / "rounds.jsonl").read_text(encoding="utf-8").splitlines()[-2000:]
        except OSError:
            return {"rows": []}
        rows = []
        for line in reversed(lines):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not isinstance(r, dict) or not r.get("sealed") or (address and r.get("worker") not in (None, address)):
                continue
            rows.append({k: r.get(k) for k in ("time", "round", "job", "kind", "accepted", "checks", "earned_vibe",
                                                "lost_vibe", "reputation", "outcome")})
            if len(rows) >= limit:
                break
        return {"rows": rows}

    # -- diagnostics --

    def diagnostics(self) -> dict:
        """`vibehash selftest --json` (Python, packages, the GPU, a real unit, the node, the clock, disk), with
        anything secret taken out (redact). A few minutes: it trains and replays a real unit."""
        cmd = [str(python()), "-m", "vibehash", "selftest", "--json"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            report = json.loads(out.stdout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "why": "the check took over 15 minutes"}
        except (OSError, ValueError) as e:
            return {"ok": False, "why": f"the check couldn't run: {e}"}
        return redact(report)

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


def python(executable: str | None = None, package: Path | None = None, frozen: bool | None = None) -> Path:
    """The Python that runs `-m vibehash` for this window. Usually this process's own; but under the downloadable
    launcher (`VibeHash.exe --window`) the window is hosted by the frozen .exe, whose sys.executable doesn't know
    `-m`, so it is the environment's python the package was imported from (env/Scripts/python.exe, env/bin/python)."""
    executable = executable or sys.executable
    frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    if not frozen:
        return Path(executable)
    package = package or Path(__file__).resolve().parent
    # Only the environment the package sits in: env/Lib/site-packages (Windows) or env/lib/pythonX.Y/site-packages,
    # never any python further up the tree (/bin/python is the system's, not the app's).
    sp = next((p for p in [package, *package.parents] if p.name == "site-packages"), None)
    if sp is not None:
        for env in (sp.parent.parent, sp.parent.parent.parent):
            for candidate in (env / "Scripts" / "python.exe", env / "bin" / "python"):
                if candidate.is_file():
                    return candidate
    return Path(executable)


def _words_file_present() -> bool:
    """Whether setup's recovery-words file is still on disk (Settings offers to delete it), without reading it."""
    path = ((app_module.read_state() or {}).get("wallet") or {}).get("words_file")
    return bool(path) and Path(path).is_file()


def _pypi_latest() -> str:
    with urllib.request.urlopen(urllib.request.Request(PYPI, headers={"User-Agent": "vibehash-app"}), timeout=10) as r:
        return str(json.loads(r.read())["info"]["version"])


PYPI_LATEST = _pypi_latest  # the newest release's version (tests replace it)


def _parts(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v.split("+")[0])[:4])


def newer(latest: str | None, mine: str) -> bool:
    """Whether release `latest` is newer than `mine` (dotted numbers compared as numbers). A version this install
    can't read is never older: no banner on every start for it."""
    return bool(latest) and bool(_parts(mine)) and _parts(latest) > _parts(mine)


_ADDRESS = re.compile(r"\b(vibe1[0-9a-zA-Z]{4})[0-9a-zA-Z]{8,}([0-9a-zA-Z]{4})\b")


def redact(value, home: str | None = None):
    """A report fit to paste into a support chat: keys that could hold a secret dropped at any depth (SECRET), this
    user's home folder written as ~, and any wallet address cut to its ends."""
    home = str(Path.home()) if home is None else home
    if isinstance(value, dict):
        return {k: redact(v, home) for k, v in value.items() if not SECRET.search(str(k))}
    if isinstance(value, list):
        return [redact(v, home) for v in value]
    if isinstance(value, str):
        if len(home) > 3:
            value = value.replace(home, "~").replace(home.replace("\\", "/"), "~")
        return _ADDRESS.sub(r"\1...\2", value)
    return value


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
            # The page as a file pywebview serves from a port of its own (http_server), so it has an http://127.0.0.1
            # origin, which the feed's CORS allows. Handed over as html= instead, WebView2 loads it with
            # NavigateToString: origin "null", which the feed rightly refuses, and the window showed no mining at all.
            webview.create_window(TITLE, url=str(page_file(port)), js_api=Api(port), width=SIZE[0], height=SIZE[1],
                                  min_size=MIN_SIZE, text_select=True)
            webview.start(http_server=True)
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

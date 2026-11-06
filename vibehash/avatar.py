"""The floating avatar: a small always-on-top pickaxe on the desktop that shows, at a glance, what the worker is doing.

    vibehash avatar               # beside a running `vibehash app` or `worker --watch` (the feed on DEFAULT_PORT)
    vibehash app --avatar         # the app starts and stops it itself; what the downloadable app does

It reads the worker's feed (watch.py, once a second, on a thread of its own so the window never waits on it) and
moves with it: resting while the worker waits, swinging while it trains, glowing while it checks other workers'
results, throwing gold sparks with the amount when a round pays, and going grey when there is no feed to read. Drag
it anywhere (the place is remembered, and kept on a screen that is still there); a click opens the window (the
feed's /dashboard page, in the browser); a right-click gives the menu: open the window, pause or resume mining,
start at login or not, hide, and stop mining.

Plain tkinter, which the Python the launcher installs has, so there is nothing to add: no browser engine, no Rust.
The drawing is the brand mark (docs/site: a pickaxe of three blocks, in the dashboard's gold) on a transparent
window where the system allows one (Windows, macOS), else on a small dark panel. Everything that decides what to
show is pure (`Mood`, `mood_of`), so it is tested without a screen; only `run` needs one.
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
import urllib.request
import webbrowser
from dataclasses import dataclass

from . import app as app_module
from . import settings
from .watch import DEFAULT_PORT

GOLD, GOLD_DIM, GREY, SPARK = "#f1b23c", "#8a6a2a", "#5a5f6a", "#ffe08a"
SIZE = 96  # the window's square, in pixels
POLL_S = 1.0
PAYDAY_S = 3.0  # how long the sparks of a payment last
KEYED = "#010101"  # the colour Windows keys out of the window: what the canvas is painted where nothing is drawn
# The brand mark on a 3 by 4 grid: two uprights and the head's block (docs/site/index.html's icon).
MARK = ((0, 0, 1, 3), (2, 0, 1, 3), (1, 3, 1, 1))


@dataclass(frozen=True)
class Mood:
    name: str  # offline, resting, training, checking, uploading, paid
    label: str  # one line, for the tooltip and the menu's first entry
    colour: str
    speed: float  # bobs a second; 0 is still


@dataclass
class Seen:
    """What mood_of carries from one poll to the next."""
    earned: float | None = None  # this session's earnings, as last reported
    paid_at: float | None = None  # when a rise in them last showed
    gain: float = 0.0  # by how much


def mood_of(snapshot: dict | None, seen: Seen, now: float) -> Mood:
    """What to show for the feed's snapshot (None: no feed), updating `seen`. A rise in this session's earnings
    shows as paid, with the amount, for PAYDAY_S seconds; otherwise the worker's phase."""
    if not snapshot or not snapshot.get("workers"):
        return Mood("offline", "no worker to watch (start it with: vibehash app)", GREY, 0.0)
    w = snapshot["workers"][0]
    earned = w.get("session", {}).get("earned_vibe")
    if earned is not None and seen.earned is not None and earned > seen.earned + 1e-9:
        seen.paid_at, seen.gain = now, earned - seen.earned
    if earned is not None:
        seen.earned = earned
    if seen.paid_at is not None and now - seen.paid_at < PAYDAY_S:
        return Mood("paid", f"paid +{seen.gain:.4f} VIBE", SPARK, 3.0)
    activity = str(w.get("activity", ""))
    phase = w.get("phase")
    if w.get("busy") and ("check" in activity or "scoring" in activity or phase == "verify"):
        return Mood("checking", activity, GOLD, 1.5)
    if w.get("busy") and "upload" in activity:
        return Mood("uploading", activity, GOLD, 1.0)
    if w.get("busy"):
        return Mood("training", activity, GOLD, 2.5)
    return Mood("resting", activity or "waiting for work", GOLD_DIM, 0.4)


def fetch(url: str, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:  # noqa: BLE001  (no feed: shown as offline, not raised)
        return None


def saved_place() -> tuple[int, int] | None:
    place = settings.load().get("avatar_place")
    if isinstance(place, list) and len(place) == 2 and all(isinstance(v, int) for v in place):
        return place[0], place[1]
    return None


def save_place(x: int, y: int) -> None:
    settings.save(avatar_place=[int(x), int(y)])


def on_screen(place: tuple[int, int] | None, width: int, height: int) -> tuple[int, int]:
    """Where the window goes: the saved place, kept on the screen there is (a monitor may have gone), else the
    bottom-right corner."""
    if place is None:
        return width - SIZE - 24, height - SIZE - 96
    return min(max(place[0], 0), max(width - SIZE, 0)), min(max(place[1], 0), max(height - SIZE, 0))


class Avatar:
    """The window. `feed` is the worker's feed URL; `tk` and `root` are passed in by run(), or faked by a test."""

    def __init__(self, feed: str, tk, root, now=time.time):
        self.feed, self.tk, self.root, self.now = feed.rstrip("/"), tk, root, now
        self.mood = Mood("offline", "connecting…", GREY, 0.0)
        self.seen = Seen()
        self.latest: dict | None = None  # the feed's last answer, from the polling thread
        self._lock = threading.Lock()
        self.phase = 0.0
        self.hidden = False
        self.trouble: str | None = None  # the first frame that failed, said once
        self._drag: tuple[int, int] | None = None
        self.transparent = self._make_window()
        bg = ("systemTransparent" if sys.platform == "darwin" else KEYED) if self.transparent else "#171a21"
        self.canvas = tk.Canvas(root, width=SIZE, height=SIZE, highlightthickness=0, bd=0, bg=bg)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Button-3>", self._menu)
        self.canvas.bind("<Button-2>", self._menu)  # macOS trackpads
        self.draw()

    # -- the window --

    def _make_window(self) -> bool:
        root = self.root
        root.overrideredirect(True)
        try:
            root.attributes("-topmost", True)
        except Exception:  # noqa: BLE001
            pass
        x, y = on_screen(saved_place(), root.winfo_screenwidth(), root.winfo_screenheight())
        root.geometry(f"{SIZE}x{SIZE}+{x}+{y}")
        try:
            if sys.platform == "win32":
                root.attributes("-transparentcolor", KEYED)
                return True
            if sys.platform == "darwin":
                root.attributes("-transparent", True)
                root.config(bg="systemTransparent")
                return True
        except Exception:  # noqa: BLE001  (a tk without it: the panel look)
            pass
        return False

    def _press(self, e):
        self._drag = (e.x, e.y)

    def _move(self, e):
        if self._drag:
            self.root.geometry(f"+{e.x_root - self._drag[0]}+{e.y_root - self._drag[1]}")

    def _release(self, e):
        if self._drag and (abs(e.x - self._drag[0]) > 3 or abs(e.y - self._drag[1]) > 3):
            save_place(self.root.winfo_x(), self.root.winfo_y())
        elif self._drag:
            self.open_window()
        self._drag = None

    # -- what it does --

    def open_window(self) -> None:
        webbrowser.open(self.feed + "/dashboard")

    def menu_items(self) -> list[tuple[str, object]]:
        """The right-click menu: (label, what it does), with None for a line."""
        running = app_module.running()
        state = app_module.read_state() or {}
        items: list = [(self.mood.label[:60], None), None, ("Open the window", self.open_window)]
        if running:
            items.append(("Resume mining", lambda: app_module.order("resume")) if state.get("paused")
                         else ("Pause mining", lambda: app_module.order("pause")))
        items += [None, ("Start when I log in", lambda: app_module.startup_on(avatar=True)),
                  ("Don't start when I log in", lambda: app_module.startup_off()), None,
                  ("Hide the avatar", self.hide)]
        if running:
            items.append(("Stop mining", lambda: (app_module.order("stop"), self.hide())))
        return items

    def _menu(self, e):
        menu = self.tk.Menu(self.root, tearoff=0)
        for item in self.menu_items():
            if item is None:
                menu.add_separator()
            else:
                label, fn = item
                if fn is None:
                    menu.add_command(label=label, state="disabled")
                else:
                    menu.add_command(label=label, command=fn)
        try:
            menu.tk_popup(e.x_root, e.y_root)
        finally:
            menu.grab_release()

    def hide(self) -> None:
        self.hidden = True
        self.root.destroy()

    # -- the feed --

    def poll_once(self) -> None:
        snapshot = fetch(self.feed + "/")
        with self._lock:
            self.latest = snapshot

    def _poll_forever(self) -> None:
        while not self.hidden:
            self.poll_once()
            time.sleep(POLL_S)

    # -- drawing --

    def draw(self) -> None:
        c = self.canvas
        c.delete("all")
        unit = SIZE / 6  # the mark is 3 by 4 units, centred, with a bob and a swing by mood
        t = self.phase
        bob = math.sin(t * 2 * math.pi) * (4 if self.mood.speed else 0)
        swing = math.sin(t * 2 * math.pi) * (0.18 if self.mood.name == "training" else 0.05 if self.mood.speed else 0)
        ox, oy = SIZE / 2 - 1.5 * unit, SIZE / 2 - 2 * unit + bob
        for (x, y, w, h) in MARK:
            x0, y0, x1, y1 = ox + x * unit, oy + y * unit, ox + (x + w) * unit, oy + (y + h) * unit
            cx, cy = SIZE / 2, SIZE / 2 + bob  # swing about the centre
            pts = []
            for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                dx, dy = px - cx, py - cy
                pts += [cx + dx * math.cos(swing) - dy * math.sin(swing), cy + dx * math.sin(swing) + dy * math.cos(swing)]
            c.create_polygon(pts, fill=self.mood.colour, outline="")
        if self.mood.name == "checking":  # a scan line
            yy = SIZE * (0.2 + 0.6 * ((t * 0.5) % 1))
            c.create_line(8, yy, SIZE - 8, yy, fill=SPARK, width=2)
        if self.mood.name == "paid":  # sparks
            for i in range(8):
                a = i / 8 * 2 * math.pi + t
                r = 28 + 10 * math.sin(t * 3 + i)
                sx, sy = SIZE / 2 + r * math.cos(a), SIZE / 2 + r * math.sin(a)
                c.create_oval(sx - 3, sy - 3, sx + 3, sy + 3, fill=SPARK, outline="")
            c.create_text(SIZE / 2, SIZE - 10, text=self.mood.label, fill=SPARK, font=("TkDefaultFont", 9, "bold"))
        if self.mood.name == "offline":
            c.create_text(SIZE / 2, SIZE - 10, text="offline", fill=GREY, font=("TkDefaultFont", 8))

    def tick(self) -> None:
        """One frame: the mood from the feed's latest answer, the drawing."""
        with self._lock:
            latest = self.latest
        self.mood = mood_of(latest, self.seen, self.now())
        self.phase += 0.05 * self.mood.speed
        self.draw()

    def run(self) -> None:
        threading.Thread(target=self._poll_forever, daemon=True, name="avatar-poll").start()

        def frame():
            if self.hidden:
                return
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001  (a bad frame is skipped; the first is said, so it can be fixed)
                if self.trouble is None:
                    self.trouble = repr(e)
                    print(f"avatar: a frame failed ({self.trouble}); carrying on", file=sys.stderr)
            self.root.after(50, frame)
        frame()
        self.root.mainloop()


def run(port: int = DEFAULT_PORT) -> int:
    """`vibehash avatar`: the window until it is hidden or closed. -> exit code (1: no display)."""
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:  # noqa: BLE001  (no display, no tkinter: say so and leave quietly)
        print(f"the avatar needs a screen and tkinter: {e}", file=sys.stderr)
        return 1
    root.title("Vibe Hash")
    try:
        Avatar(f"http://127.0.0.1:{port}", tk, root).run()
    except KeyboardInterrupt:
        pass
    return 0


def main(a) -> None:
    sys.exit(run(getattr(a, "watch", None) or DEFAULT_PORT))


if __name__ == "__main__":
    sys.exit(run(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT))

"""What the person at this computer is doing, for the app's automatic pauses (app.Supervisor): on battery, a game or
another app full screen, how long since the last key press or mouse move.

Each probe answers None when this system can't tell (they are the operating system's own calls, through ctypes, so
nothing is installed for them). Windows answers all three. Linux answers the battery from /sys. Elsewhere they answer
None, and the settings that need them say so in the window (`supported`).

`hold(cfg, facts)` is the decision, kept apart from the probes so it is the same everywhere and tested anywhere: the
reason to hold mining off now, in words for the window, or None.
"""
from __future__ import annotations

import sys
from pathlib import Path

IDLE_MINUTES = 5  # the default for "only when I'm away": no key or mouse for this long
IDLE_CHOICES = (1, 2, 5, 10, 15, 30, 60)
OPTIONS = ("pause_on_battery", "pause_when_fullscreen", "idle_only")


def on_battery() -> bool | None:
    """True when the computer runs on its battery (unplugged), False when plugged in or with no battery."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class Status(ctypes.Structure):  # unsigned bytes: wintypes.BYTE is signed, and 128 ("no battery") read -128
                _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                            ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                            ("BatteryLifeTime", wintypes.DWORD), ("BatteryFullLifeTime", wintypes.DWORD)]

            s = Status()
            if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
                return None
            if s.BatteryFlag == 128:  # no system battery: a desktop
                return False
            return {0: True, 1: False}.get(s.ACLineStatus)  # 255: unknown
        except (OSError, AttributeError):
            return None
    if sys.platform.startswith("linux"):
        root = Path("/sys/class/power_supply")
        try:
            mains = [p for p in root.iterdir() if (p / "type").read_text().strip() == "Mains"]
            if not mains:
                return False if not any((p / "type").read_text().strip() == "Battery" for p in root.iterdir()) else None
            return not any((p / "online").read_text().strip() == "1" for p in mains)
        except OSError:
            return None
    return None


# SHQueryUserNotificationState's answers that mean "don't disturb": an app full screen (busy), a Direct3D game full
# screen, or presentation mode.
_FULLSCREEN = {2, 3, 4}


def fullscreen() -> bool | None:
    """True while a game or another app is full screen (or Windows is in presentation mode)."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        state = ctypes.c_int(0)
        if ctypes.windll.shell32.SHQueryUserNotificationState(ctypes.byref(state)) != 0:
            return None
        return state.value in _FULLSCREEN
    except (OSError, AttributeError):
        return None


def idle_seconds() -> float | None:
    """Seconds since the last key press or mouse move in this session."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class LastInput(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        li = LastInput()
        li.cbSize = ctypes.sizeof(LastInput)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(li)):
            return None
        now = ctypes.windll.kernel32.GetTickCount() & 0xFFFFFFFF
        return ((now - li.dwTime) & 0xFFFFFFFF) / 1000.0  # both wrap after 49.7 days
    except (OSError, AttributeError):
        return None


def facts() -> dict:
    return {"battery": on_battery(), "fullscreen": fullscreen(), "idle_s": idle_seconds()}


def supported() -> dict:
    """Which settings this computer can honour: the window greys out the rest and says why."""
    f = facts()
    return {"pause_on_battery": f["battery"] is not None, "pause_when_fullscreen": f["fullscreen"] is not None,
            "idle_only": f["idle_s"] is not None}


def hold(cfg: dict, f: dict) -> str | None:
    """The reason to hold mining off now, or None: from the settings (config.json) and the probes' facts. A probe
    that can't tell never holds mining off."""
    if cfg.get("pause_on_battery") and f.get("battery") is True:
        return "on battery"
    if cfg.get("pause_when_fullscreen") and f.get("fullscreen") is True:
        return "a full-screen app or game is running"
    if cfg.get("idle_only") and f.get("idle_s") is not None:
        minutes = idle_minutes(cfg)
        if f["idle_s"] < minutes * 60:
            return f"you're using the computer (mining starts after {minutes} min away)"
    return None


def idle_minutes(cfg: dict) -> int:
    v = cfg.get("idle_minutes", IDLE_MINUTES)
    return v if isinstance(v, int) and not isinstance(v, bool) and v in IDLE_CHOICES else IDLE_MINUTES

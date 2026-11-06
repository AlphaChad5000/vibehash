"""The Vibe Hash launcher: the one file people download from vibehash.org (VibeHash.exe on Windows), which installs
itself like any app the first time and then just runs `vibehash app`.

    VibeHash.exe                # first run: installs; every run: mines, with the window and the avatar
    VibeHash.exe --update       # update Vibe Hash now, rather than when a day has passed
    VibeHash.exe --uninstall    # remove what the launcher installed (never the wallet, which is in ~/.vibehash)
    VibeHash.exe --window PORT  # the app's window alone (what `vibehash app` and the avatar start under the launcher)
    VibeHash.exe -- --device cpu --node 192.168.1.20   # anything after -- goes to `vibehash app`

On Windows it is a windowed program (no console): a small window shows what the first run is doing, then the app's
own window takes over (the download's one job is to feel like installing any other app). What it does on that first
run, in its own folder (%LOCALAPPDATA%\\VibeHash on Windows, ~/Library/Application Support/VibeHash on macOS,
~/.local/share/vibehash on Linux) and nowhere else (uv's Python and cache are kept there too):
- copies itself there, puts "Vibe Hash" in the Start menu and on the desktop, and in Settings > Apps (an entry whose
  Uninstall runs `--uninstall`), then hands over to the installed copy, so the download can be deleted (Windows);
- uv (astral.sh), one pinned release, its download checked against the SHA-256 the release publishes;
- a Python of its own, which uv downloads, so the computer's Python (if any) is never touched;
- Vibe Hash from PyPI, updated (it alone, not every library under it) when a day has passed since the last check;
- the GPU build of PyTorch, chosen by `vibehash app` itself (gpusetup.py), a one-time download of up to 3 GB.

It is plain Python with nothing but the standard library (tkinter for the progress window, where there is one), so
it also runs as a script, and PyInstaller turns it into the .exe (.github/workflows/launcher.yml), with pywebview
inside so `--window` can host the app's window (vibehash/window.py, imported from the environment it installed)
under the launcher's own name and icon. It never reads or writes the wallet: `vibehash app` does that, and the
launcher only passes `--no-password`, so a first wallet is made without a question (the file is then as safe as the
user's account; `vibehash wallet create` makes a password-protected one for anyone who wants it), and `--avatar`,
the floating pickaxe on the desktop (no screen, and it leaves quietly). The app runs with no console of its own; what
it prints goes to app.log in the launcher's folder, and the launcher's own lines to launcher.log. `app startup on`
under the launcher starts the launcher at login (it says where it is in $VIBEHASH_LAUNCHER), so the daily check
still happens.
"""
from __future__ import annotations

import hashlib
import io
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

VERSION = "1.1"  # the launcher's own (Settings > Apps shows it); Vibe Hash's version is the package's
UV_VERSION = "0.13.7"
UV_RELEASES = "https://github.com/astral-sh/uv/releases/download"
UV_ASSETS = {  # (platform.system(), platform.machine().lower()) -> the release asset
    ("Windows", "amd64"): "uv-x86_64-pc-windows-msvc.zip",
    ("Windows", "x86_64"): "uv-x86_64-pc-windows-msvc.zip",
    ("Windows", "arm64"): "uv-aarch64-pc-windows-msvc.zip",
    ("Darwin", "arm64"): "uv-aarch64-apple-darwin.tar.gz",
    ("Darwin", "x86_64"): "uv-x86_64-apple-darwin.tar.gz",
    ("Linux", "x86_64"): "uv-x86_64-unknown-linux-gnu.tar.gz",
    ("Linux", "amd64"): "uv-x86_64-unknown-linux-gnu.tar.gz",
    ("Linux", "aarch64"): "uv-aarch64-unknown-linux-gnu.tar.gz",
    ("Linux", "arm64"): "uv-aarch64-unknown-linux-gnu.tar.gz",
}
PYTHON = "3.12"
PACKAGE = "vibehash[finetune,app]"  # app: pywebview, for the window (window.py)
UPDATE_EVERY_S = 86400
BANNER = "Vibe Hash  -  mining that trains real AI models  -  vibehash.org"  # (ASCII: any console shows it)
EXE = "VibeHash.exe"
UNINSTALL_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\VibeHash"  # Settings > Apps (HKCU)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # a child with a console, hidden, so Ctrl-Break still reaches a worker


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


# --- where it talks ------------------------------------------------------------------------------------------------

class Screen:
    """Where the launcher's lines go: the console when there is one; else (the windowed .exe) a small window while
    it installs; and always launcher.log in its folder. `say` may be called from any thread."""

    def __init__(self, folder: Path | None = None, gui: bool | None = None):
        self.folder = folder
        self.gui = (sys.stdout is None) if gui is None else gui  # a windowed .exe has no stdout
        self.root = None
        self.label = None
        self.lines: list[str] = []

    def say(self, text: str) -> None:
        self.lines.append(text)
        if sys.stdout is not None:
            print(f"  {text}", flush=True)
        self._log(text)
        if self.root is not None:
            try:
                self.root.after(0, lambda: self.label.config(text=text))
            except Exception:  # noqa: BLE001  (the window is gone: the log has it)
                pass

    def _log(self, text: str) -> None:
        if self.folder is None:
            return
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            with open(self.folder / "launcher.log", "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}\n")
        except OSError:
            pass

    def show(self, title: str = "Setting up Vibe Hash") -> bool:
        """The progress window (gui only). -> whether there is one."""
        if not self.gui or self.root is not None:
            return self.root is not None
        try:
            import tkinter as tk
            root = tk.Tk()
        except Exception:  # noqa: BLE001  (no tkinter, no display: the log is what there is)
            return False
        root.title("Vibe Hash")
        icon = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "VibeHash.ico"  # in the .exe (--add-data), or beside the script
        if icon.exists():
            try:
                root.iconbitmap(default=str(icon))
            except Exception:  # noqa: BLE001  (not Windows, say: tk's own icon then)
                pass
        root.configure(bg="#0f1115")
        root.resizable(False, False)
        w, h = 440, 230
        root.geometry(f"{w}x{h}+{(root.winfo_screenwidth() - w) // 2}+{(root.winfo_screenheight() - h) // 3}")
        c = tk.Canvas(root, width=48, height=64, bg="#0f1115", highlightthickness=0)
        c.place(x=28, y=36)
        for x, y, bw, bh in ((0, 0, 1, 3), (2, 0, 1, 3), (1, 3, 1, 1)):  # the mark: two uprights and the head
            c.create_rectangle(x * 16, y * 16, (x + bw) * 16, (y + bh) * 16, fill="#f1b23c", width=0)
        tk.Label(root, text=title, font=("Segoe UI", 15, "bold"), fg="#e6e8ee", bg="#0f1115").place(x=96, y=34)
        self.label = tk.Label(root, text="starting...", font=("Segoe UI", 10), fg="#9aa3b2", bg="#0f1115",
                              wraplength=320, justify="left", anchor="w")
        self.label.place(x=96, y=70, width=320)
        tk.Label(root, text="This happens once. PyTorch for your graphics card can be a few GB, so the first run "
                 "takes a few minutes; after that, seconds. The window opens when it is ready.",
                 font=("Segoe UI", 9), fg="#5f6572", bg="#0f1115", wraplength=384, justify="left").place(x=28, y=150)
        root.protocol("WM_DELETE_WINDOW", lambda: None)  # closing it mid-install would leave half an install
        self.root = root
        return True

    def run_while(self, work, poll_s: float = 0.1):
        """Do `work()` on a thread while the window shows what it says; -> what work returned (or raises)."""
        if self.root is None:
            return work()
        box: dict = {}

        def go():
            try:
                box["result"] = work()
            except BaseException as e:  # noqa: BLE001  (handed back to the caller's thread)
                box["error"] = e
            finally:
                self.root.after(0, self.root.quit)

        threading.Thread(target=go, daemon=True).start()
        self.root.mainloop()
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def close(self) -> None:
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:  # noqa: BLE001
                pass
            self.root = None

    def tell(self, text: str, error: bool = False) -> None:
        """A last word: a message box in gui mode, else a line."""
        self.say(text)
        if self.gui:
            try:
                from tkinter import messagebox
                (messagebox.showerror if error else messagebox.showinfo)("Vibe Hash", text)
            except Exception:  # noqa: BLE001
                pass

    def ask(self, text: str) -> bool:
        """Yes or no: a box in gui mode, the terminal otherwise (no terminal: yes, as a script run unattended)."""
        self.say(text)
        if self.gui:
            try:
                from tkinter import messagebox
                return bool(messagebox.askyesno("Vibe Hash", text))
            except Exception:  # noqa: BLE001
                return True
        if sys.stdin and sys.stdin.isatty():
            try:
                return input("  [y/N] ").strip().lower().startswith("y")
            except (EOFError, KeyboardInterrupt):
                return False
        return True


screen = Screen()


def say(text: str) -> None:
    screen.say(text)


# --- where things go ----------------------------------------------------------------------------------------------

def app_dir(system: str | None = None, env: dict | None = None) -> Path:
    system = system or platform.system()
    env = os.environ if env is None else env
    if env.get("VIBEHASH_LAUNCHER_DIR"):
        return Path(env["VIBEHASH_LAUNCHER_DIR"])
    if system == "Windows":
        return Path(env.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "VibeHash"
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "VibeHash"
    return Path(env.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "vibehash"


def uv_asset(system: str | None = None, machine: str | None = None) -> str:
    key = (system or platform.system(), (machine or platform.machine()).lower())
    try:
        return UV_ASSETS[key]
    except KeyError:
        raise SystemExit(f"no uv build for {key[0]} on {key[1]}; install Vibe Hash with pip instead: "
                         "https://vibehash.org") from None


def uv_path(folder: Path, system: str | None = None) -> Path:
    return folder / ("uv.exe" if (system or platform.system()) == "Windows" else "uv")


def env_python(env: Path, system: str | None = None) -> Path:
    if (system or platform.system()) == "Windows":
        return env / "Scripts" / "python.exe"
    return env / "bin" / "python"


def site_packages(env: Path, system: str | None = None) -> Path:
    """Where the environment keeps what uv installed (the window imports vibehash from there)."""
    if (system or platform.system()) == "Windows":
        return env / "Lib" / "site-packages"
    return env / "lib" / f"python{PYTHON}" / "site-packages"


def child_env(folder: Path) -> dict:
    """The environment uv, Python and the app run with: uv keeps its Python and its cache in the launcher's folder, so
    --uninstall removes all of it; the frozen launcher's own library path (PyInstaller sets LD_LIBRARY_PATH to its
    unpacked bundle) is taken back out, so a child never loads the launcher's libraries instead of its own; and the
    app knows where the launcher is, so `app startup on` starts it rather than the environment's python, and the
    window is hosted by it (`--window`)."""
    env = dict(os.environ)
    env["UV_PYTHON_INSTALL_DIR"] = str(folder / "python")
    env["UV_CACHE_DIR"] = str(folder / "cache")
    env["UV_PYTHON_PREFERENCE"] = "only-managed"
    for name in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        orig = env.pop(name + "_ORIG", None)
        if frozen():
            if orig:
                env[name] = orig
            else:
                env.pop(name, None)
    if frozen():  # PyInstaller points the frozen program's tkinter at its unpacked bundle: a child's tkinter must not look there
        for name in ("TCL_LIBRARY", "TK_LIBRARY"):
            env.pop(name, None)
    if frozen():
        env["VIBEHASH_LAUNCHER"] = sys.executable
    return env


# --- installing itself (Windows) ----------------------------------------------------------------------------------

def shortcut_paths(env: dict | None = None) -> list[Path]:
    """Where "Vibe Hash" usually goes: the Start menu and the desktop (the shell is asked for the real folders when
    the shortcuts are made, a desktop under OneDrive say; these are the guesses uninstall also clears)."""
    env = os.environ if env is None else env
    appdata = Path(env.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    profile = Path(env.get("USERPROFILE") or Path.home())
    return [appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Vibe Hash.lnk", profile / "Desktop" / "Vibe Hash.lnk"]


def make_shortcuts(exe: Path, run=subprocess.run) -> list[Path]:
    """Windows shortcuts to the .exe in the Start menu and on the desktop, where the shell says those are, through
    its own COM object (PowerShell, no window). -> the shortcuts made (none when it didn't go)."""
    q = lambda p: "'" + str(p).replace("'", "''") + "'"  # noqa: E731
    script = ("$s = New-Object -ComObject WScript.Shell; "
              "$prog = [Environment]::GetFolderPath('Programs'); if (-not $prog) { $prog = Join-Path $env:APPDATA 'Microsoft\\Windows\\Start Menu\\Programs' }; "
              "$desk = [Environment]::GetFolderPath('Desktop'); if (-not $desk) { $desk = Join-Path $env:USERPROFILE 'Desktop' }; "
              "foreach ($dir in @($prog, $desk)) { "
              "if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }; $p = Join-Path $dir 'Vibe Hash.lnk'; $l = $s.CreateShortcut($p); "
              f"$l.TargetPath = {q(exe)}; $l.WorkingDirectory = {q(exe.parent)}; $l.IconLocation = {q(exe)} + ',0'; "
              "$l.Description = 'Vibe Hash: mining that trains real AI models'; $l.Save(); Write-Output $p }")
    try:
        done = run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                   creationflags=NO_WINDOW, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    return [Path(line.strip()) for line in (done.stdout or "").splitlines() if line.strip().lower().endswith(".lnk")]


def _winreg():
    try:
        import winreg
        return winreg
    except ImportError:
        return None


def register(exe: Path, folder: Path, winreg=None) -> bool:
    """Settings > Apps: an entry for Vibe Hash whose Uninstall runs the launcher's --uninstall (HKCU: no admin)."""
    winreg = winreg or _winreg()
    if winreg is None:
        return False
    try:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as k:
            for name, value in (("DisplayName", "Vibe Hash"), ("DisplayIcon", str(exe)), ("DisplayVersion", VERSION),
                                ("Publisher", "Vibe Hash"), ("InstallLocation", str(folder)),
                                ("UninstallString", f'"{exe}" --uninstall'), ("URLInfoAbout", "https://vibehash.org")):
                winreg.SetValueEx(k, name, 0, winreg.REG_SZ, value)
            for name in ("NoModify", "NoRepair"):
                winreg.SetValueEx(k, name, 0, winreg.REG_DWORD, 1)
        return True
    except OSError:
        return False


def unregister(winreg=None) -> None:
    winreg = winreg or _winreg()
    if winreg is None:
        return
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY)
    except OSError:
        pass


def install_self(folder: Path, system: str | None = None, exe: Path | None = None, run=subprocess.run,
                 winreg=None) -> Path | None:
    """The downloaded .exe, run from wherever it was saved, copies itself into the launcher's folder, makes the
    shortcuts and the Settings > Apps entry, and -> the installed copy for the caller to hand over to. Run from the
    folder already (the installed copy, a shortcut, start-at-login): None, after making sure the shortcuts and the
    entry are there."""
    system = system or platform.system()
    if system != "Windows" or not (exe or frozen()):
        return None
    me = Path(exe or sys.executable).resolve()
    target = folder / EXE
    installed = me.parent == folder.resolve()
    if not installed:
        folder.mkdir(parents=True, exist_ok=True)
        fresh = folder / (EXE + ".new")
        try:  # whole or not at all: copied beside, then swapped in
            shutil.copy2(me, fresh)
            os.replace(fresh, target)
        except OSError as e:
            fresh.unlink(missing_ok=True)
            if not target.exists():
                raise SystemExit(f"couldn't put the launcher in {folder}: {e}") from None
            say(f"the installed Vibe Hash is in use, so this download can't replace it now; starting the installed one "
                f"(stop Vibe Hash and run the download again to put the new launcher in)")
    made = folder / "shortcuts.txt"  # the shortcuts, made once (a deleted one stays deleted) and listed for uninstall
    if not made.exists():
        links = make_shortcuts(target, run=run)
        if links:
            made.write_text("".join(f"{link}\n" for link in links), encoding="utf-8")
    register(target, folder, winreg=winreg)
    return None if installed else target


# --- downloading -------------------------------------------------------------------------------------------------

def fetch(url: str, timeout: float = 60) -> bytes:
    """A whole file, with a line of progress for a big one."""
    req = urllib.request.Request(url, headers={"User-Agent": "vibehash-launcher"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        if total < 2**20:
            total = 0  # a checksum file: not worth a progress line
        buf, got, last = io.BytesIO(), 0, 0.0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            buf.write(chunk)
            got += len(chunk)
            if total and time.time() - last > 0.5:
                if sys.stdout is not None:
                    print(f"\r  {got / 2**20:6.1f} / {total / 2**20:.1f} MB", end="", flush=True)
                elif screen.root is not None:
                    screen.root.after(0, lambda g=got: screen.label.config(text=f"downloading... {g / 2**20:.0f} of {total / 2**20:.0f} MB"))
                last = time.time()
        if total and sys.stdout is not None:
            print("\r" + " " * 40 + "\r", end="")
        return buf.getvalue()


def sha256_ok(data: bytes, published: str) -> bool:
    """`published` is the text of a release's NAME.sha256 file: "<hex>  NAME" (or just the hex)."""
    want = published.strip().split()[0].lower() if published.strip() else ""
    return len(want) == 64 and hashlib.sha256(data).hexdigest() == want


def ensure_uv(folder: Path, system: str | None = None, machine: str | None = None, fetch=fetch) -> Path:
    """uv at its pinned version in `folder`, downloaded and checked once."""
    system = system or platform.system()
    exe, stamp = uv_path(folder, system), folder / "uv.version"
    if exe.exists() and stamp.exists() and stamp.read_text().strip() == UV_VERSION:
        return exe
    asset = uv_asset(system, machine)
    url = f"{UV_RELEASES}/{UV_VERSION}/{asset}"
    say(f"getting uv {UV_VERSION} (the installer's installer, about 15 MB)...")
    data, published = fetch(url), fetch(url + ".sha256").decode("utf-8", "replace")
    if not sha256_ok(data, published):
        raise SystemExit(f"the download of {asset} doesn't match the checksum its release publishes; not installing it. "
                         "Check your connection (a proxy?) and try again.")
    folder.mkdir(parents=True, exist_ok=True)
    name = exe.name
    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            member = next(m for m in z.namelist() if m.endswith(name))
            exe.write_bytes(z.read(member))
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
            member = next(m for m in t.getmembers() if m.name.endswith("/" + name) or m.name == name)
            exe.write_bytes(t.extractfile(member).read())
        exe.chmod(0o755)
    stamp.write_text(UV_VERSION + "\n")
    return exe


# --- the environment -----------------------------------------------------------------------------------------------

def run(cmd: list[str], **kw) -> int:
    """A child, waited for; with no console of our own (the windowed .exe), none for it either, and what it says
    goes to the log."""
    if sys.stdout is not None:
        sys.stdout.flush()
        return subprocess.call(cmd, **kw)
    done = subprocess.run(cmd, creationflags=NO_WINDOW, capture_output=True, text=True, errors="replace", **kw)
    for line in (done.stdout + done.stderr).splitlines()[-30:]:
        screen._log("  | " + line)
    return done.returncode


def ensure_env(uv: Path, env: Path, system: str | None = None, run=run, child_env: dict | None = None) -> Path:
    """A Python of its own in `env` (uv fetches PYTHON; it never bases the environment on a Python the computer
    happens to have, which could be uninstalled later), with pip, so `vibehash install-gpu` can swap PyTorch builds
    inside it."""
    python = env_python(env, system)
    if python.exists():
        return python
    say(f"making a Python {PYTHON} of its own for Vibe Hash (a one-time download of about 25 MB)...")
    if run([str(uv), "venv", str(env), "--python", PYTHON, "--python-preference", "only-managed", "--seed", "--quiet"],
           env=child_env):
        raise SystemExit("couldn't make the Python environment; uv's message (launcher.log) says why")
    return python


def ensure_vibehash(uv: Path, python: Path, env: Path, force: bool = False, now: float | None = None, run=run,
                    child_env: dict | None = None) -> bool:
    """Vibe Hash installed, and brought up to date once a day (or now, with --update): the package itself, not every
    library under it, so a PyTorch build `install-gpu` put in stays matched. -> whether it ran pip."""
    now = time.time() if now is None else now
    stamp = env / ".vibehash-checked"
    try:
        last = float(stamp.read_text().strip())
    except (OSError, ValueError):
        last = 0.0
    if last and not force and now - last < UPDATE_EVERY_S:  # (no stamp: never installed, whatever the clock says)
        return False
    say("installing Vibe Hash from PyPI..." if not last else "checking for a newer Vibe Hash...")
    if run([str(uv), "pip", "install", "--python", str(python), "--upgrade-package", "vibehash", "--quiet", PACKAGE],
           env=child_env):
        if last:  # it is installed already: a failed check (offline?) is no reason not to mine
            say("couldn't check for updates (offline?); carrying on with what is installed")
            return False
        raise SystemExit("couldn't install Vibe Hash; pip's message (launcher.log) says why. Check you're online and try again.")
    stamp.write_text(f"{now:.0f}\n")
    return True


def run_app(python: Path, args: list[str], child_env: dict | None = None, log: Path | None = None) -> int:
    """`vibehash app`, waited for. Ctrl+C reaches the app and the launcher alike; the launcher ignores it and waits,
    so the app gets to stop its worker properly (round log written, dashboard summed up) before both end. Run from
    the environment's own folder: `python -m vibehash` would otherwise import a `vibehash` folder that happened to be
    in whatever folder the launcher was started from (a checkout, say), not the installed package. With no console
    (the windowed .exe) the app gets a hidden one (its worker is stopped with Ctrl-Break, which needs a console) and
    writes to `log` (app.log) instead of a screen."""
    cmd = [str(python), "-m", "vibehash", "app", "--no-password", "--avatar", *args]  # (no screen: the avatar leaves quietly)
    kw: dict = {}
    if sys.stdout is None:
        kw["creationflags"] = NO_WINDOW
        if log is not None:
            try:
                mode = "w" if log.exists() and log.stat().st_size > 5 * 2**20 else "a"  # a second launch mustn't wipe a running app's log
                kw["stdout"] = kw["stderr"] = open(log, mode, encoding="utf-8", errors="replace")  # noqa: SIM115
            except OSError:
                kw["stdout"] = kw["stderr"] = subprocess.DEVNULL
        else:
            kw["stdout"] = kw["stderr"] = subprocess.DEVNULL
    else:
        sys.stdout.flush()
    child = subprocess.Popen(cmd, cwd=str(python.parent.parent), env=child_env, **kw)
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        return child.wait()
    finally:
        signal.signal(signal.SIGINT, previous)
        out = kw.get("stdout")
        if hasattr(out, "close"):
            out.close()


# --- the window ----------------------------------------------------------------------------------------------------

def window(folder: Path, port: int, system: str | None = None, run=subprocess.call) -> int:
    """`--window PORT`: the app's window (vibehash/window.py) hosted by the launcher itself, so it carries the
    launcher's name and icon rather than python.exe's. pywebview is inside the .exe; vibehash is imported from the
    environment the launcher made (the same Python version, pure Python). Without pywebview here (a script run), or
    with no environment yet, the environment's own python shows it."""
    env = folder / "env"
    try:
        import webview  # noqa: F401  (bundled in the .exe)
        sys.path.insert(0, str(site_packages(env, system)))
        from vibehash import window as app_window
    except ImportError:
        python = env_python(env, system)
        if not python.exists():
            say("the window needs Vibe Hash installed first; run the launcher")
            return 1
        return run([str(python), "-m", "vibehash", "window", "--watch", str(port)],
                   creationflags=NO_WINDOW if sys.stdout is None else 0, env=child_env(folder))
    return 0 if app_window.show(port) != "none" else 1


# --- leaving -------------------------------------------------------------------------------------------------------

def uninstall(folder: Path, system: str | None = None, run=run, winreg=None) -> bool:
    """Remove what the launcher installed: a running app is stopped (`app stop`, waited for) and its start-at-login
    entry taken out (it would point at a python that is about to go), then the shortcuts and the Settings > Apps
    entry, then the folder. The wallet, settings and logs live in ~/.vibehash and stay. A running .exe inside the
    folder can't delete itself on Windows: `remove_later` takes the folder once this program has left.
    -> whether everything went (or is about to)."""
    system = system or platform.system()
    python = env_python(folder / "env", system)
    if python.exists():
        for words in (("app", "stop"), ("app", "startup", "off")):
            run([str(python), "-m", "vibehash", *words], cwd=str(folder / "env"), env=child_env(folder))
    if system == "Windows":
        listed = folder / "shortcuts.txt"
        links = [Path(p) for p in listed.read_text(encoding="utf-8").splitlines() if p.strip()] if listed.exists() else []
        for link in links + shortcut_paths():
            link.unlink(missing_ok=True)
        unregister(winreg=winreg)
    inside = frozen() and folder.exists() and Path(sys.executable).resolve().parent == folder.resolve()
    if folder.exists() and not inside:
        shutil.rmtree(folder, ignore_errors=True)
    if folder.exists() and not inside:
        say(f"couldn't remove all of {folder} (is Vibe Hash still running? stop it, then try again)")
        return False
    screen.folder = None  # the log goes with the folder; nothing more is written there
    say(f"removed {folder}. Your wallet and settings are still in ~/.vibehash; delete that folder yourself if you "
        "want them gone too (back up its recovery words first).")
    return True


def remove_later(folder: Path) -> None:
    """The folder that holds this very program, taken by cmd once we have left: tried every second for a minute,
    since the .exe is locked until it has really exited."""
    loop = (f'for /l %i in (1,1,60) do (rmdir /s /q "{folder}" 2>nul & if not exist "{folder}" exit /b 0 & '
            'ping -n 2 127.0.0.1 >nul)')
    subprocess.Popen(["cmd", "/c", loop], creationflags=NO_WINDOW | 0x8, close_fds=True)  # 0x8: DETACHED_PROCESS


# --- the command ----------------------------------------------------------------------------------------------------

def split_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """(the launcher's own flags, what goes to `vibehash app`): everything after the first -- is the app's."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    folder = app_dir()
    screen.folder = folder
    mine, passed = split_args(argv)
    if mine[:1] == ["--window"]:
        try:
            port = int(mine[1])
        except (IndexError, ValueError):
            say("--window needs the feed's port, as in: --window 7545")
            return 2
        return window(folder, port)
    say(BANNER)
    unknown = [a for a in mine if a not in ("--update", "--uninstall")]
    if unknown:
        screen.tell(f"unknown option {unknown[0]}; the launcher takes --update, --uninstall and --window PORT, and anything "
                    "after -- goes to `vibehash app`", error=True)
        return 2
    if "--uninstall" in mine:
        if not screen.ask("Remove Vibe Hash from this computer? Your wallet and settings in ~/.vibehash stay."):
            return 0
        inside = frozen() and Path(sys.executable).resolve().parent == folder.resolve()
        ok = uninstall(folder)
        screen.tell(screen.lines[-1], error=not ok)  # (a box: the folder can only go once this is dismissed and we are gone)
        if ok and inside and folder.exists():
            remove_later(folder)
        return 0 if ok else 1
    try:
        installed = install_self(folder)
        if installed is not None:  # the download: hand over to the installed copy, which the shortcuts point at
            say(f"installed to {installed}; starting it")
            subprocess.Popen([str(installed), *argv], close_fds=True)
            return 0
        env_ = child_env(folder)
        env = folder / "env"
        if not env_python(env).exists() or "--update" in mine:
            screen.show()

        def install():
            uv = ensure_uv(folder)
            python = ensure_env(uv, env, child_env=env_)
            ensure_vibehash(uv, python, env, force="--update" in mine, child_env=env_)
            return python

        python = screen.run_while(install)
        screen.close()
    except SystemExit as e:
        screen.close()
        screen.tell(str(e), error=True)
        return 1
    except KeyboardInterrupt:
        screen.close()
        say("stopped before it was installed; run it again any time")
        return 130
    except Exception as e:  # noqa: BLE001  (a launcher shows what went wrong, it doesn't crash)
        screen.close()
        screen.tell(f"something went wrong while installing: {e!r} (launcher.log in {folder} has more)", error=True)
        return 1
    try:
        return run_app(python, passed, child_env=env_, log=folder / "app.log")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    code = main()
    if code and code != 130 and os.name == "nt" and sys.stdin and sys.stdin.isatty():
        try:
            input("\n  Press Enter to close this window.")
        except (EOFError, KeyboardInterrupt):
            pass
    sys.exit(code)

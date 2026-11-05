"""The Vibe Hash launcher: the one file people download from vibehash.org (VibeHash.exe on Windows), which sets
everything up the first time and then just runs `vibehash app`.

    VibeHash.exe                # first run: installs; every run: mines (Ctrl+C stops it)
    VibeHash.exe --update       # update Vibe Hash now, rather than when a day has passed
    VibeHash.exe --uninstall    # remove what the launcher installed (never the wallet, which is in ~/.vibehash)
    VibeHash.exe -- --device cpu --node 192.168.1.20   # anything after -- goes to `vibehash app`

What it installs, in its own folder (%LOCALAPPDATA%\\VibeHash on Windows, ~/Library/Application Support/VibeHash on
macOS, ~/.local/share/vibehash on Linux) and nowhere else (uv's Python and cache are kept there too):
- uv (astral.sh), one pinned release, its download checked against the SHA-256 the release publishes;
- a Python of its own, which uv downloads, so the computer's Python (if any) is never touched;
- Vibe Hash from PyPI, updated (it alone, not every library under it) when a day has passed since the last check;
- the GPU build of PyTorch, chosen by `vibehash app` itself (gpusetup.py), a one-time download of up to 3 GB.

It is plain Python with nothing but the standard library, so it also runs as a script, and PyInstaller turns it into
the .exe (.github/workflows/launcher.yml). It never reads or writes the wallet: `vibehash app` does that, and the
launcher only passes `--no-password` so a first wallet is made without a question (the file is then as safe as the
user's account; `vibehash wallet create` makes a password-protected one for anyone who wants it). While the app runs,
Ctrl+C goes to the app, which stops its worker properly; the launcher waits for it. `app startup on` under the
launcher starts the launcher at login (it says where it is in $VIBEHASH_LAUNCHER), so the daily check still happens.
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
import time
import urllib.request
import zipfile
from pathlib import Path

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
PACKAGE = "vibehash[finetune]"
UPDATE_EVERY_S = 86400
BANNER = "Vibe Hash  -  mining that trains real AI models  -  vibehash.org"  # (ASCII: any console shows it)


def say(text: str) -> None:
    print(f"  {text}", flush=True)


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


def child_env(folder: Path) -> dict:
    """The environment uv, Python and the app run with: uv keeps its Python and its cache in the launcher's folder, so
    --uninstall removes all of it; the frozen launcher's own library path (PyInstaller sets LD_LIBRARY_PATH to its
    unpacked bundle) is taken back out, so a child never loads the launcher's libraries instead of its own; and the
    app knows where the launcher is, so `app startup on` starts it rather than the environment's python."""
    env = dict(os.environ)
    env["UV_PYTHON_INSTALL_DIR"] = str(folder / "python")
    env["UV_CACHE_DIR"] = str(folder / "cache")
    env["UV_PYTHON_PREFERENCE"] = "only-managed"
    for name in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        orig = env.pop(name + "_ORIG", None)
        if getattr(sys, "frozen", False):
            if orig:
                env[name] = orig
            else:
                env.pop(name, None)
    if getattr(sys, "frozen", False):
        env["VIBEHASH_LAUNCHER"] = sys.executable
    return env


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
                print(f"\r  {got / 2**20:6.1f} / {total / 2**20:.1f} MB", end="", flush=True)
                last = time.time()
        if total:
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
    sys.stdout.flush()
    return subprocess.call(cmd, **kw)


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
        raise SystemExit("couldn't make the Python environment; uv's message above says why")
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
        raise SystemExit("couldn't install Vibe Hash; pip's message above says why. Check you're online and try again.")
    stamp.write_text(f"{now:.0f}\n")
    return True


def run_app(python: Path, args: list[str], child_env: dict | None = None) -> int:
    """`vibehash app`, waited for. Ctrl+C reaches the app and the launcher alike; the launcher ignores it and waits,
    so the app gets to stop its worker properly (round log written, dashboard summed up) before both end. Run from
    the environment's own folder: `python -m vibehash` would otherwise import a `vibehash` folder that happened to be
    in whatever folder the launcher was started from (a checkout, say), not the installed package."""
    cmd = [str(python), "-m", "vibehash", "app", "--no-password", *args]
    sys.stdout.flush()
    child = subprocess.Popen(cmd, cwd=str(python.parent.parent), env=child_env)
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        return child.wait()
    finally:
        signal.signal(signal.SIGINT, previous)


def uninstall(folder: Path, system: str | None = None, run=run) -> bool:
    """Remove what the launcher installed, and the start-at-login entry if `app startup on` made one, since it
    would point at a python that is about to go. The wallet, settings and logs live in ~/.vibehash and stay.
    -> whether everything went."""
    python = env_python(folder / "env", system)
    if python.exists():
        run([str(python), "-m", "vibehash", "app", "startup", "off"], cwd=str(folder / "env"), env=child_env(folder))
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    if platform.system() == "Windows" and os.environ.get("APPDATA"):  # app.ps1's Start menu entry
        link = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Vibe Hash.lnk"
        link.unlink(missing_ok=True)
    if folder.exists():
        say(f"couldn't remove all of {folder} (is Vibe Hash still running? stop it, then try again)")
        return False
    say(f"removed {folder}. Your wallet and settings are still in ~/.vibehash; delete that folder yourself if you "
        "want them gone too (back up its recovery words first).")
    return True


# --- the command ----------------------------------------------------------------------------------------------------

def split_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """(the launcher's own flags, what goes to `vibehash app`): everything after the first -- is the app's."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    print(f"\n  {BANNER}\n", flush=True)
    folder = app_dir()
    mine, passed = split_args(argv)
    unknown = [a for a in mine if a not in ("--update", "--uninstall")]
    if unknown:
        say(f"unknown option {unknown[0]}; the launcher takes --update and --uninstall, and anything after -- goes to "
            "`vibehash app`")
        return 2
    if "--uninstall" in mine:
        return 0 if uninstall(folder) else 1
    try:
        env_ = child_env(folder)
        uv = ensure_uv(folder)
        env = folder / "env"
        python = ensure_env(uv, env, child_env=env_)
        ensure_vibehash(uv, python, env, force="--update" in mine, child_env=env_)
    except SystemExit as e:
        say(str(e))
        return 1
    except KeyboardInterrupt:
        say("stopped before it was installed; run it again any time")
        return 130
    except Exception as e:  # noqa: BLE001  (a launcher shows what went wrong, it doesn't crash)
        say(f"something went wrong while installing: {e!r}")
        return 1
    try:
        return run_app(python, passed, child_env=env_)
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

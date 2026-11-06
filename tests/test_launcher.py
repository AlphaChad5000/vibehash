"""The downloadable launcher (launcher/vibehash_launcher.py): where it installs, how it checks uv's download, what
it runs, and that it never touches the wallet. Nothing here goes online."""
import hashlib
import importlib.util
import io
import os
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent


def load():
    spec = importlib.util.spec_from_file_location("vibehash_launcher", ROOT / "launcher" / "vibehash_launcher.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = load()
from tests.test_app import FakeWinreg  # noqa: E402  (the in-memory registry)


class WhereTests(unittest.TestCase):
    def test_a_folder_per_system(self):
        local = r"C:\Users\me\AppData\Local"  # (joined the same way on every system: CI runs this on Linux)
        self.assertEqual(L.app_dir("Windows", {"LOCALAPPDATA": local}), Path(local) / "VibeHash")
        self.assertEqual(L.app_dir("Darwin", {}), Path.home() / "Library" / "Application Support" / "VibeHash")
        self.assertEqual(L.app_dir("Linux", {"XDG_DATA_HOME": "/data"}), Path("/data/vibehash"))
        self.assertEqual(L.app_dir("Linux", {}), Path.home() / ".local" / "share" / "vibehash")
        self.assertEqual(L.app_dir("Linux", {"VIBEHASH_LAUNCHER_DIR": "/tmp/x"}), Path("/tmp/x"))

    def test_the_uv_build_for_the_machine(self):
        self.assertEqual(L.uv_asset("Windows", "AMD64"), "uv-x86_64-pc-windows-msvc.zip")
        self.assertEqual(L.uv_asset("Darwin", "arm64"), "uv-aarch64-apple-darwin.tar.gz")
        self.assertEqual(L.uv_asset("Linux", "x86_64"), "uv-x86_64-unknown-linux-gnu.tar.gz")
        with self.assertRaises(SystemExit):
            L.uv_asset("Linux", "mips")
        self.assertEqual(L.uv_path(Path("f"), "Windows").name, "uv.exe")
        self.assertEqual(L.env_python(Path("e"), "Windows"), Path("e") / "Scripts" / "python.exe")
        self.assertEqual(L.env_python(Path("e"), "Linux"), Path("e") / "bin" / "python")


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def test_checksum_lines(self):
        data = b"uv"
        hexd = hashlib.sha256(data).hexdigest()
        self.assertTrue(L.sha256_ok(data, f"{hexd}  uv-x86_64-pc-windows-msvc.zip\n"))
        self.assertTrue(L.sha256_ok(data, hexd.upper()))
        self.assertFalse(L.sha256_ok(data, "0" * 64))
        self.assertFalse(L.sha256_ok(data, ""))

    def fake_fetch(self, payload: bytes, good=True):
        def fetch(url):
            if url.endswith(".sha256"):
                return (hashlib.sha256(payload).hexdigest() if good else "0" * 64).encode() + b"  asset\n"
            return payload
        return fetch

    def test_uv_from_a_zip_once(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("uv-x86_64-pc-windows-msvc/uv.exe", b"MZ fake uv")
        fetch = mock.Mock(side_effect=self.fake_fetch(buf.getvalue()))
        exe = L.ensure_uv(self.folder, "Windows", "AMD64", fetch=fetch)
        self.assertEqual(exe.read_bytes(), b"MZ fake uv")
        self.assertEqual((self.folder / "uv.version").read_text().strip(), L.UV_VERSION)
        self.assertEqual(fetch.call_count, 2)
        self.assertIn(f"/{L.UV_VERSION}/uv-x86_64-pc-windows-msvc.zip", fetch.call_args_list[0].args[0])
        L.ensure_uv(self.folder, "Windows", "AMD64", fetch=fetch)  # there already: nothing fetched
        self.assertEqual(fetch.call_count, 2)

    def test_uv_from_a_tarball(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            info = tarfile.TarInfo("uv-x86_64-unknown-linux-gnu/uv")
            info.size = 7
            t.addfile(info, io.BytesIO(b"ELF uv!"))
        exe = L.ensure_uv(self.folder, "Linux", "x86_64", fetch=self.fake_fetch(buf.getvalue()))
        self.assertEqual(exe.read_bytes(), b"ELF uv!")
        if os.name != "nt":
            self.assertTrue(os.access(exe, os.X_OK))

    def test_a_download_that_fails_its_checksum_is_not_installed(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("uv.exe", b"tampered")
        with self.assertRaises(SystemExit) as ctx:
            L.ensure_uv(self.folder, "Windows", "AMD64", fetch=self.fake_fetch(buf.getvalue(), good=False))
        self.assertIn("checksum", str(ctx.exception))
        self.assertFalse((self.folder / "uv.exe").exists())


class EnvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = Path(self.tmp.name) / "env"
        self.uv = Path(self.tmp.name) / "uv"

    def test_the_environment_is_made_once_with_pip(self):
        calls = []
        python = L.ensure_env(self.uv, self.env, "Linux", run=lambda cmd, **kw: calls.append(cmd) or 0)
        self.assertEqual(python, self.env / "bin" / "python")
        self.assertEqual(calls[0][:3], [str(self.uv), "venv", str(self.env)])
        self.assertIn("--seed", calls[0])  # pip inside, for `vibehash install-gpu`
        self.assertIn(L.PYTHON, calls[0])
        python.parent.mkdir(parents=True)
        python.write_text("")
        L.ensure_env(self.uv, self.env, "Linux", run=lambda cmd, **kw: calls.append(cmd) or 0)
        self.assertEqual(len(calls), 1)
        with self.assertRaises(SystemExit):
            L.ensure_env(self.uv, Path(self.tmp.name) / "other", "Linux", run=lambda cmd, **kw: 1)

    def test_vibehash_is_installed_then_checked_once_a_day(self):
        self.env.mkdir()
        python = self.env / "bin" / "python"
        calls = []
        ok = lambda cmd, **kw: calls.append(cmd) or 0  # noqa: E731
        self.assertTrue(L.ensure_vibehash(self.uv, python, self.env, now=1000.0, run=ok))
        self.assertEqual(calls[0][:4], [str(self.uv), "pip", "install", "--python"])
        self.assertIn(L.PACKAGE, calls[0])
        self.assertFalse(L.ensure_vibehash(self.uv, python, self.env, now=1000.0 + 3600, run=ok))  # an hour later: no
        self.assertTrue(L.ensure_vibehash(self.uv, python, self.env, now=1000.0 + L.UPDATE_EVERY_S + 1, run=ok))
        self.assertTrue(L.ensure_vibehash(self.uv, python, self.env, now=1000.0 + L.UPDATE_EVERY_S + 2, force=True, run=ok))
        self.assertEqual(len(calls), 3)
        # offline later: a failed check is no reason not to mine
        self.assertFalse(L.ensure_vibehash(self.uv, python, self.env, now=2e6, run=lambda cmd, **kw: 1))
        # but a first install that fails stops with a reason
        with self.assertRaises(SystemExit):
            L.ensure_vibehash(self.uv, python, Path(self.tmp.name) / "fresh", run=lambda cmd, **kw: 1)

    def test_the_app_is_run_without_a_password_question_and_waited_for(self):
        child = mock.Mock()
        child.wait.return_value = 7
        with mock.patch.object(L.subprocess, "Popen", return_value=child) as popen, \
                mock.patch.object(L.signal, "signal") as sig:
            code = L.run_app(Path("e") / "bin" / "py", ["--device", "cpu"], child_env={"X": "1"})
        self.assertEqual(code, 7)
        self.assertEqual(popen.call_args.args[0], [str(Path("e") / "bin" / "py"), "-m", "vibehash", "app", "--no-password", "--avatar", "--device", "cpu"])
        self.assertEqual(popen.call_args.kwargs["cwd"], "e")  # the environment's folder, not wherever it was started
        self.assertEqual(popen.call_args.kwargs["env"], {"X": "1"})
        self.assertNotIn("stdout", popen.call_args.kwargs)  # a console: the app talks there
        self.assertEqual(sig.call_args_list[0].args, (L.signal.SIGINT, L.signal.SIG_IGN))  # Ctrl+C is the app's to handle

    def test_with_no_console_the_app_gets_a_hidden_one_and_a_log(self):
        child = mock.Mock()
        child.wait.return_value = 0
        log = Path(tempfile.mkdtemp()) / "app.log"
        with mock.patch.object(L.subprocess, "Popen", return_value=child) as popen, mock.patch.object(L.signal, "signal"), \
                mock.patch.object(L.sys, "stdout", None):  # the windowed .exe
            L.run_app(Path("e") / "bin" / "py", [], log=log)
        kw = popen.call_args.kwargs
        self.assertEqual(kw["creationflags"], L.NO_WINDOW)
        self.assertIs(kw["stdout"], kw["stderr"])
        self.assertTrue(kw["stdout"].closed)  # closed after the wait
        self.assertTrue(log.exists())

    def test_the_children_keep_everything_in_the_launchers_folder(self):
        folder = Path("f")
        with mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": "/tmp/_MEI123", "LD_LIBRARY_PATH_ORIG": "/usr/lib/x"}):
            env = L.child_env(folder)
            self.assertEqual(env["UV_PYTHON_INSTALL_DIR"], str(folder / "python"))
            self.assertEqual(env["UV_CACHE_DIR"], str(folder / "cache"))
            self.assertEqual(env["UV_PYTHON_PREFERENCE"], "only-managed")
            self.assertNotIn("VIBEHASH_LAUNCHER", env)  # a script, not the frozen .exe
            self.assertEqual(env["LD_LIBRARY_PATH"], "/tmp/_MEI123")  # as a script: left alone
            with mock.patch.object(L.sys, "frozen", True, create=True), \
                    mock.patch.dict(os.environ, {"TCL_LIBRARY": "/tmp/_MEI123/_tcl_data", "TK_LIBRARY": "/tmp/_MEI123/_tk_data"}):
                env = L.child_env(folder)
                self.assertEqual(env["LD_LIBRARY_PATH"], "/usr/lib/x")  # the .exe's unpacked bundle taken back out
                self.assertEqual(env["VIBEHASH_LAUNCHER"], L.sys.executable)
                self.assertNotIn("TCL_LIBRARY", env)  # the avatar's tkinter would look in the .exe's bundle otherwise
                self.assertNotIn("TK_LIBRARY", env)
        with mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": "/tmp/_MEI123"}, clear=False), \
                mock.patch.object(L.sys, "frozen", True, create=True):
            os.environ.pop("LD_LIBRARY_PATH_ORIG", None)
            self.assertNotIn("LD_LIBRARY_PATH", L.child_env(folder))

    def test_only_vibehash_itself_is_upgraded(self):
        env = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(env, ignore_errors=True))
        calls = []
        L.ensure_vibehash(Path("uv"), env / "bin" / "python", env, run=lambda cmd, **kw: calls.append(cmd) or 0)
        self.assertIn("--upgrade-package", calls[0])
        self.assertEqual(calls[0][calls[0].index("--upgrade-package") + 1], "vibehash")
        self.assertNotIn("--upgrade", calls[0])


class MainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Never this machine's Start menu, desktop or registry: a home of its own, and winreg stood in for
        self.env = mock.patch.dict(os.environ, {"APPDATA": self.tmp.name + "/AppData", "USERPROFILE": self.tmp.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.winreg = FakeWinreg()
        self.reg = mock.patch.object(L, "_winreg", return_value=self.winreg)
        self.reg.start()
        self.addCleanup(self.reg.stop)
        self.ask = mock.patch.object(L.screen, "ask", return_value=True)
        self.ask.start()
        self.addCleanup(self.ask.stop)

    def test_uninstall_removes_the_launchers_folder_shortcuts_and_entry_only(self):
        with mock.patch.dict(os.environ, {"VIBEHASH_LAUNCHER_DIR": self.tmp.name + "/vh"}), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            tmp = self.tmp.name
            folder = Path(tmp) / "vh"
            folder.mkdir()
            (folder / "uv.exe").write_text("")
            wallet = Path(tmp) / ".vibehash" / "wallet.json"  # not the launcher's: must stay
            wallet.parent.mkdir()
            wallet.write_text("{}")
            links = L.shortcut_paths() + [Path(tmp) / "OneDrive" / "Desktop" / "Vibe Hash.lnk"]
            for link in links:
                link.parent.mkdir(parents=True, exist_ok=True)
                link.write_text("")
            (folder / "shortcuts.txt").write_text(f"{links[-1]}\n")  # the one the shell put somewhere else
            L.register(folder / L.EXE, folder, winreg=self.winreg)
            python = L.env_python(folder / "env", "Windows")
            python.parent.mkdir(parents=True)
            python.write_text("")
            ran = []
            with mock.patch.object(L.platform, "system", return_value="Windows"), \
                    mock.patch.object(L.subprocess, "call", side_effect=lambda cmd, **kw: ran.append(cmd) or 0):
                self.assertEqual(L.main(["--uninstall"]), 0)
            self.assertEqual([c[-2:] for c in ran], [["app", "stop"], ["startup", "off"]])  # a running app is stopped first
            self.assertFalse(folder.exists())
            self.assertTrue(wallet.exists())
            self.assertFalse(any(link.exists() for link in links))
            self.assertNotIn(L.UNINSTALL_KEY, self.winreg.keys)
            self.assertIn("wallet", out.getvalue())

    def test_uninstall_asks_first_in_the_window(self):
        with mock.patch.dict(os.environ, {"VIBEHASH_LAUNCHER_DIR": self.tmp.name + "/vh"}), \
                mock.patch.object(L.screen, "ask", return_value=False), mock.patch("sys.stdout", new_callable=io.StringIO):
            folder = Path(self.tmp.name) / "vh"
            folder.mkdir()
            self.assertEqual(L.main(["--uninstall"]), 0)
            self.assertTrue(folder.exists())  # no

    def test_the_download_installs_itself_then_hands_over(self):
        tmp = Path(self.tmp.name)
        download, folder = tmp / "Downloads" / "VibeHash.exe", tmp / "vh"
        download.parent.mkdir()
        download.write_bytes(b"MZ launcher")
        ran = []
        made = [tmp / "Programs" / "Vibe Hash.lnk", tmp / "OneDrive" / "Desktop" / "Vibe Hash.lnk"]  # where the shell says
        run = lambda cmd, **kw: ran.append(cmd) or mock.Mock(returncode=0, stdout="".join(f"{p}\n" for p in made))  # noqa: E731
        installed = L.install_self(folder, "Windows", exe=download, run=run, winreg=self.winreg)
        self.assertEqual(installed, folder / "VibeHash.exe")
        self.assertEqual(installed.read_bytes(), b"MZ launcher")
        self.assertFalse((folder / "VibeHash.exe.new").exists())  # copied beside, then swapped in
        self.assertEqual(ran[0][:2], ["powershell", "-NoProfile"])  # the shortcuts, through the shell's COM object
        self.assertIn("GetFolderPath('Desktop')", ran[0][-1])  # the real desktop (OneDrive's, say), not a guess
        self.assertIn(str(installed), ran[0][-1])
        self.assertEqual((folder / "shortcuts.txt").read_text().splitlines(), [str(p) for p in made])  # remembered for uninstall
        entry = self.winreg.keys[L.UNINSTALL_KEY]
        self.assertEqual(entry["DisplayName"], "Vibe Hash")
        self.assertEqual(entry["UninstallString"], f'"{installed}" --uninstall')
        self.assertEqual(entry["DisplayVersion"], L.VERSION)
        # the installed copy itself: nothing to copy, no hand-over, the shortcuts not made again (a deleted one stays
        # deleted), the entry refreshed
        self.assertIsNone(L.install_self(folder, "Windows", exe=installed, run=run, winreg=self.winreg))
        self.assertEqual(len(ran), 1)
        self.assertIsNone(L.install_self(folder, "Linux", exe=download, run=run, winreg=self.winreg))  # Windows only
        with mock.patch.dict(os.environ, {"VIBEHASH_LAUNCHER_DIR": str(folder)}), mock.patch.object(L, "install_self", return_value=installed), \
                mock.patch.object(L.subprocess, "Popen") as popen, mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(L.main(["--", "--device", "cpu"]), 0)
        self.assertEqual(popen.call_args.args[0], [str(installed), "--", "--device", "cpu"])  # the same arguments, to the installed copy

    def test_the_window_is_hosted_here_with_vibehash_from_the_environment(self):
        folder = Path(self.tmp.name) / "vh"
        sp = L.site_packages(folder / "env", "Linux")
        self.assertEqual(sp, folder / "env" / "lib" / f"python{L.PYTHON}" / "site-packages")
        self.assertEqual(L.site_packages(folder / "env", "Windows"), folder / "env" / "Lib" / "site-packages")
        shown = []
        fake_window = mock.Mock(show=lambda port: shown.append(port) or "window")
        with mock.patch.dict(sys.modules, {"webview": mock.Mock(), "vibehash": mock.Mock(window=fake_window), "vibehash.window": fake_window}), \
                mock.patch.object(L.sys, "path", list(sys.path)):
            self.assertEqual(L.window(folder, 7545, "Linux"), 0)
            self.assertEqual(shown, [7545])
            self.assertEqual(L.sys.path[0], str(sp))  # vibehash came from the environment
        ran = []
        with mock.patch.dict(sys.modules, {"webview": None}):  # a script run, no pywebview here: the environment's python shows it
            self.assertEqual(L.window(folder, 7545, "Linux", run=lambda cmd, **kw: ran.append(cmd) or 1), 1)  # no environment yet
            self.assertEqual(ran, [])
            python = L.env_python(folder / "env", "Linux")
            python.parent.mkdir(parents=True)
            python.write_text("")
            self.assertEqual(L.window(folder, 7545, "Linux", run=lambda cmd, **kw: ran.append(cmd) or 0), 0)
            self.assertEqual(ran[0], [str(python), "-m", "vibehash", "window", "--watch", "7545"])
        with mock.patch.object(L, "window", return_value=0) as w, mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(L.main(["--window", "7000"]), 0)
            self.assertEqual(w.call_args.args[1], 7000)
            self.assertEqual(L.main(["--window"]), 2)
            self.assertIn("--window needs", out.getvalue())

    def test_the_screen_talks_to_the_console_and_the_log(self):
        folder = Path(self.tmp.name) / "vh"
        s = L.Screen(folder, gui=False)
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            s.say("hello")
            self.assertFalse(s.show())  # no window without a gui
            self.assertEqual(s.run_while(lambda: 42), 42)  # straight through
        self.assertIn("hello", out.getvalue())
        self.assertIn("hello", (folder / "launcher.log").read_text())

    def test_main_installs_then_runs_the_app_and_passes_arguments_through(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"VIBEHASH_LAUNCHER_DIR": tmp}), \
                mock.patch.object(L, "install_self", return_value=None), \
                mock.patch.object(L, "ensure_uv", return_value=Path("uv")), \
                mock.patch.object(L, "ensure_env", return_value=Path("py")), \
                mock.patch.object(L, "ensure_vibehash") as vh, mock.patch.object(L, "run_app", return_value=0) as ra, \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(L.main(["--update", "--", "--device", "cpu"]), 0)
            self.assertTrue(vh.call_args.kwargs["force"])
            self.assertEqual(ra.call_args.args[1], ["--device", "cpu"])
            self.assertEqual(ra.call_args.kwargs["log"], Path(tmp) / "app.log")
            self.assertEqual(L.main([]), 0)
            self.assertFalse(vh.call_args.kwargs["force"])

    def test_an_install_problem_is_a_line_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"VIBEHASH_LAUNCHER_DIR": tmp}), \
                mock.patch.object(L, "install_self", return_value=None), \
                mock.patch.object(L, "ensure_uv", side_effect=SystemExit("no uv for you")), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(L.main([]), 1)
            self.assertIn("no uv for you", out.getvalue())


class AppScriptTests(unittest.TestCase):
    """app.ps1, the one-line route to the same launcher (irm https://vibehash.org/app.ps1 | iex)."""

    def test_it_pins_the_launchers_uv_so_the_launcher_reuses_its_download(self):
        text = (Path(__file__).resolve().parent.parent / "app.ps1").read_text(encoding="utf-8")
        self.assertIn(f"$uvVersion = '{L.UV_VERSION}'", text)

    def test_it_is_ascii(self):  # irm may not decode it as UTF-8
        (Path(__file__).resolve().parent.parent / "app.ps1").read_bytes().decode("ascii")


if __name__ == "__main__":
    unittest.main()

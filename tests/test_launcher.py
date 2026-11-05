"""The downloadable launcher (launcher/vibehash_launcher.py): where it installs, how it checks uv's download, what
it runs, and that it never touches the wallet. Nothing here goes online."""
import hashlib
import importlib.util
import io
import os
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


class WhereTests(unittest.TestCase):
    def test_a_folder_per_system(self):
        self.assertEqual(L.app_dir("Windows", {"LOCALAPPDATA": r"C:\Users\me\AppData\Local"}), Path(r"C:\Users\me\AppData\Local\VibeHash"))
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
        self.assertEqual(popen.call_args.args[0], [str(Path("e") / "bin" / "py"), "-m", "vibehash", "app", "--no-password", "--device", "cpu"])
        self.assertEqual(popen.call_args.kwargs["cwd"], "e")  # the environment's folder, not wherever it was started
        self.assertEqual(popen.call_args.kwargs["env"], {"X": "1"})
        self.assertEqual(sig.call_args_list[0].args, (L.signal.SIGINT, L.signal.SIG_IGN))  # Ctrl+C is the app's to handle

    def test_the_children_keep_everything_in_the_launchers_folder(self):
        folder = Path("f")
        with mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": "/tmp/_MEI123", "LD_LIBRARY_PATH_ORIG": "/usr/lib/x"}):
            env = L.child_env(folder)
            self.assertEqual(env["UV_PYTHON_INSTALL_DIR"], str(folder / "python"))
            self.assertEqual(env["UV_CACHE_DIR"], str(folder / "cache"))
            self.assertEqual(env["UV_PYTHON_PREFERENCE"], "only-managed")
            self.assertNotIn("VIBEHASH_LAUNCHER", env)  # a script, not the frozen .exe
            self.assertEqual(env["LD_LIBRARY_PATH"], "/tmp/_MEI123")  # as a script: left alone
            with mock.patch.object(L.sys, "frozen", True, create=True):
                env = L.child_env(folder)
                self.assertEqual(env["LD_LIBRARY_PATH"], "/usr/lib/x")  # the .exe's unpacked bundle taken back out
                self.assertEqual(env["VIBEHASH_LAUNCHER"], L.sys.executable)
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
    def test_uninstall_removes_the_launchers_folder_only(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"VIBEHASH_LAUNCHER_DIR": tmp + "/vh"}), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            folder = Path(tmp) / "vh"
            folder.mkdir()
            (folder / "uv.exe").write_text("")
            wallet = Path(tmp) / ".vibehash" / "wallet.json"  # not the launcher's: must stay
            wallet.parent.mkdir()
            wallet.write_text("{}")
            self.assertEqual(L.main(["--uninstall"]), 0)
            self.assertFalse(folder.exists())
            self.assertTrue(wallet.exists())
            self.assertIn("wallet", out.getvalue())

    def test_main_installs_then_runs_the_app_and_passes_arguments_through(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"VIBEHASH_LAUNCHER_DIR": tmp}), \
                mock.patch.object(L, "ensure_uv", return_value=Path("uv")), \
                mock.patch.object(L, "ensure_env", return_value=Path("py")), \
                mock.patch.object(L, "ensure_vibehash") as vh, mock.patch.object(L, "run_app", return_value=0) as ra, \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(L.main(["--update", "--", "--device", "cpu"]), 0)
            self.assertTrue(vh.call_args.kwargs["force"])
            self.assertEqual(ra.call_args.args[1], ["--device", "cpu"])
            self.assertEqual(L.main([]), 0)
            self.assertFalse(vh.call_args.kwargs["force"])

    def test_an_install_problem_is_a_line_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"VIBEHASH_LAUNCHER_DIR": tmp}), \
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

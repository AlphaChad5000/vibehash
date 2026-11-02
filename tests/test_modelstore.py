"""vibehash/modelstore.py: where a miner's PC keeps model files, and for how long (DEEPSEEK.md §8).

- CapTests: the store never holds more than its cap. Other jobs' least recently used files go first, and a job that
  can't fit, under the cap or on the disk, is refused with nothing deleted.
- LifetimeTests: a job's files go when it ends or after a week unmined, unless kept; a file two jobs share stays
  while either needs it; temporaries go when released.
- PartialTests: an interrupted download resumes where it stopped, even in a new process; clear and sweep leave no
  partial download behind.
- AskTests: the question before a download (yes, no, unattended, another place), the drives, and what
  `vibehash cache` would list.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vibehash import modelstore as ms
from vibehash import settings


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def fetcher(sizes, log, stop_at=None):
    """A download as deepseek.fetch_file does one: through dest.part, resuming one already there."""
    def fetch(name, dest):
        part = dest.with_name(dest.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        log.append((name, have))
        with open(part, "ab") as f:
            for i in range(have, sizes[name]):
                if stop_at is not None and i == stop_at:
                    raise ConnectionError("the link dropped")
                f.write(b"x")
        os.replace(part, dest)
    return fetch


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        env = mock.patch.dict(os.environ, {"VIBEHASH_HOME": str(self.tmp / "home")})
        env.start()
        self.addCleanup(env.stop)
        self.clock = Clock()
        self.log = []

    def store(self, **kw):
        return ms.Store(self.tmp / "models", **{"cap": 1000, "clock": self.clock, **kw})

    def get(self, store, job, files, **kw):
        plan = store.plan(job, files)
        self.assertTrue(plan.fits, plan.why_not)
        return store.acquire(plan, fetcher(files, self.log, **kw))


class CapTests(Base):
    def test_least_recently_used_go_first(self):
        st = self.store()
        self.get(st, "job-a", {"a/1": 400, "a/2": 400})
        self.assertEqual(st.usage(), 800)
        self.clock.t += 10
        plan = st.plan("job-b", {"b/1": 500})
        self.assertEqual(plan.evict, ["a/1"])  # just enough room, the oldest first
        st.acquire(plan, fetcher({"b/1": 500}, self.log))
        self.assertFalse((st.folder / "a/1").exists())
        self.assertEqual(st.usage(), 900)
        self.assertLessEqual(st.usage(), st.cap)

    def test_a_job_that_cant_fit_is_refused(self):
        st = self.store()
        self.get(st, "job-a", {"a/1": 600})
        plan = st.plan("job-big", {"big/1": 1200})
        self.assertFalse(plan.fits)
        self.assertIn("the space you allow", plan.why_not)
        said = []
        self.assertFalse(st.ask(plan, yes=True, say=said.append))  # not even unattended
        self.assertIn("other jobs carry on", said[0])
        self.assertEqual(st.usage(), 600)  # nothing deleted for it
        with self.assertRaises(RuntimeError):
            st.acquire(plan, fetcher({"big/1": 1200}, self.log))

    def test_nor_if_the_disk_is_too_full(self):
        st = self.store(cap=10 * ms.GB)
        with mock.patch.object(ms, "free_space", return_value=ms.DISK_MARGIN + 100):
            self.assertFalse(st.plan("job-a", {"a/1": 500}).fits)
            self.assertTrue(st.plan("job-a", {"a/1": 50}).fits)


class LifetimeTests(Base):
    def test_files_go_when_the_job_ends_unless_another_needs_them(self):
        st = self.store()
        self.get(st, "job-a", {"shared": 100, "a/1": 100})
        self.get(st, "job-b", {"shared": 100, "b/1": 100})
        self.assertEqual(self.log[-1][0], "b/1")  # the shared file wasn't fetched twice
        self.assertEqual(sorted(st.job_ended("job-a")), ["a/1"])
        self.assertTrue((st.folder / "shared").exists())
        self.assertEqual(sorted(st.job_ended("job-b")), ["b/1", "shared"])
        self.assertEqual(st.usage(), 0)

    def test_a_week_unmined(self):
        st = self.store()
        self.get(st, "job-a", {"a/1": 100})
        self.clock.t += 6 * 86400
        st.touch("job-a")  # mined again: the week starts over
        self.clock.t += 6 * 86400
        self.assertEqual(st.sweep(), [])
        self.clock.t += 2 * 86400
        self.assertEqual(st.sweep(), ["a/1"])
        self.assertFalse((st.folder / "a/1").exists())

    def test_kept_for_next_time(self):
        st = self.store(keep=True)
        self.get(st, "job-a", {"a/1": 100})
        self.assertEqual(st.job_ended("job-a"), [])
        self.clock.t += 30 * 86400
        self.assertEqual(st.sweep(), [])
        self.assertTrue((st.folder / "a/1").exists())
        self.get(st, "job-b", {"b/1": 950})  # but it still gives way to the cap
        self.assertFalse((st.folder / "a/1").exists())

    def test_temporaries(self):
        st = self.store(keep=True)
        path = st.temporary("job-a", "c00000/stream-04")
        path.write_bytes(b"x" * 300)
        self.assertEqual(st.usage(), 300)
        st.release("c00000/stream-04")  # used and accepted
        self.assertFalse(path.exists())
        st.temporary("job-a", "shard-0").write_bytes(b"x" * 10)
        self.assertEqual(st.job_ended("job-a"), ["shard-0"])  # kept or not, a job's temporaries go with it


class PartialTests(Base):
    def test_an_interrupted_download_resumes(self):
        st = self.store()
        plan = st.plan("job-a", {"a/1": 300})
        with self.assertRaises(ConnectionError):
            st.acquire(plan, fetcher({"a/1": 300}, self.log, stop_at=150))
        self.assertEqual(st.usage(), 300)  # counted at its full size meanwhile
        again = self.store()  # a new process
        self.get(again, "job-a", {"a/1": 300})
        self.assertEqual(self.log, [("a/1", 0), ("a/1", 150)])  # it carried on from byte 150
        self.assertEqual((again.folder / "a/1").stat().st_size, 300)
        self.assertEqual(again.entries()[0]["bytes"], 300)

    def test_nothing_partial_left_behind(self):
        st = self.store()
        with self.assertRaises(ConnectionError):
            st.acquire(st.plan("job-a", {"a/1": 300}), fetcher({"a/1": 300}, self.log, stop_at=10))
        self.assertIn("a download under way", st.entries()[0]["why"])
        self.clock.t += 8 * 86400
        self.assertEqual(st.sweep(), ["a/1"])  # a partial nobody resumed for a week
        self.assertEqual([p for p in st.folder.rglob("*") if p.name != ms.INDEX], [])

    def test_only_its_own_files(self):
        """The fourth review's finding 1: given a drive's root (a folder standing in for one), the store keeps to
        <root>/vibehash-models, and sweep and clear leave what its index doesn't name, and folders it didn't make."""
        root = self.tmp / "root"
        (root / "Downloads").mkdir(parents=True)
        (root / "Downloads" / "big-video.mp4.part").write_bytes(b"x")
        st = ms.Store(root, cap=1000, clock=self.clock)
        self.assertEqual(st.folder, root / "vibehash-models")
        with self.assertRaises(ConnectionError):
            st.acquire(st.plan("job-a", {"a/1": 300}), fetcher({"a/1": 300}, self.log, stop_at=10))
        (st.folder / "theirs.part").write_bytes(b"x")
        st.files["../victim"] = {"size": 1, "jobs": ["job-a"], "kind": "model", "what": "", "used": 0}
        (st.folder.parent / "victim").write_bytes(b"keep")
        self.clock.t += 8 * 86400
        st.sweep()
        st.clear()
        self.assertEqual(sorted(p.relative_to(root).as_posix() for p in root.rglob("*")),
                         ["Downloads", "Downloads/big-video.mp4.part", "vibehash-models",
                          "vibehash-models/store.json", "vibehash-models/theirs.part", "victim"])

    def test_clear_one_job(self):
        st = self.store()
        self.get(st, "job-a", {"shared": 100, "a/1": 100})
        self.get(st, "job-b", {"shared": 100})
        self.assertEqual(st.clear("job-a"), 100)
        self.assertEqual(sorted(e["name"] for e in st.entries()), ["shared"])


class AskTests(Base):
    def plan(self, st):
        return st.plan("think-shorter-pilot", {"deepseek/model-00039.safetensors": 600}, "DeepSeek-V4.1-Flash's top layers")

    def test_yes_no_and_unattended(self):
        st = self.store()
        said = []
        self.assertFalse(st.ask(self.plan(st), ask=lambda q: "n", say=said.append))
        self.assertIn("think-shorter-pilot needs DeepSeek-V4.1-Flash's top layers", said[0])
        self.assertIn(str(st.folder), said[0])
        self.assertIn("deleted when the job ends", said[-1])
        self.assertTrue(st.ask(self.plan(st), ask=lambda q: "", say=said.append))
        self.assertTrue(st.ask(self.plan(st), yes=True, ask=lambda q: self.fail("asked"), say=said.append))

    def test_another_place_is_remembered(self):
        st = self.store()
        self.get(st, "job-old", {"old/1": 100})
        elsewhere = self.tmp / "big-drive" / "vibehash-models"
        answers = iter(["c", str(elsewhere), "y"])
        said = []
        self.assertTrue(st.ask(self.plan(st), ask=lambda q: next(answers), say=said.append))
        self.assertEqual(st.folder, elsewhere)
        self.assertFalse((self.tmp / "models" / "old/1").exists())  # the old place isn't left cluttered
        self.assertEqual(settings.load()["models"]["folder"], str(elsewhere))
        self.assertEqual(ms.Store.from_settings().folder, elsewhere)

    def test_settings(self):
        settings.save(models={"folder": str(self.tmp / "m"), "cap_gb": 40, "keep": True})
        st = ms.Store.from_settings()
        self.assertEqual((st.folder, st.cap, st.keep), (self.tmp / "m" / "vibehash-models", 40 * ms.GB, True))

    def test_drives(self):
        found = ms.drives()
        self.assertTrue(found)
        self.assertEqual([d[1] for d in found], sorted((d[1] for d in found), reverse=True))  # the roomiest first
        self.assertEqual(ms.default_folder(), ms.folder_on(found[0][0]))
        self.assertIn(ms.folder_on(found[0][0]).name, ("models", "vibehash-models"))


if __name__ == "__main__":
    unittest.main()

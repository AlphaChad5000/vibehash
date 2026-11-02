"""Where a miner's PC keeps big model files, and for how long. The owner's rule: DeepSeek's ~30 GB must never
become permanent clutter on someone's C: drive. So (DEEPSEEK.md §8, "Model files on a miner's PC"):

1. Ask before any big download (`ask`): what it is, how big, and where it goes, with the option to put it elsewhere.
   `drives` lists the drives with their free space; the default is the one with the most room (`default_folder`),
   and the choice is remembered (config.json's "models"). `yes` says yes unattended, and the folder can be set
   outright.
2. A cap the user sets ("use up to 40 GB"). The store never holds more. To make room it evicts the least recently
   used model files of other jobs first, and a job whose files won't fit even then is refused (`Plan.fits`),
   never overflowed. It also leaves the disk 2 GB free.
3. Not permanent by default. A job's model files go when the job ends (`job_ended`), or after `idle_days` (7)
   without mining it (`sweep`), whichever comes first. `keep` keeps them for next time instead, which saves
   re-downloading on a slow link; they still give way to the cap. A file two jobs use stays while either needs it.
   Temporary files (hand-offs, data shards, checkpoints) go as soon as they've been used and accepted (`release`).
4. Visible and removable. `entries` is what `vibehash cache` lists: what, where, how big and why. `clear` removes
   everything, or one job's files; `usage` is the disk use the worker's dashboard shows.
5. No partial download left behind. A download goes to its .part file and resumes from there (deepseek.fetch_file
   does). It's in the index while under way, so `clear`, `job_ended` and `sweep` remove a partial nobody resumes.

The worker's prompt, `vibehash cache`, the dashboard and setup call this once PR #5's worker merges; the fine-tune
path can use it the same way. `Store.from_settings` reads the policy from config.json.
"""
from __future__ import annotations

import json
import os
import shutil
import string
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import settings

GB = 10**9
DEFAULT_CAP_GB = 40.0
IDLE_DAYS = 7.0
DISK_MARGIN = 2 * GB  # never fill a drive to its last bytes
INDEX = "store.json"
PARTIALS = (".part", ".done")  # a download under way (deepseek.fetch_file's files)


def drives() -> list[tuple[Path, int, int]]:
    """[(drive or mount point, free bytes, total bytes)] where model files could go, the roomiest first."""
    seen, out = set(), []
    if sys.platform == "win32":
        roots = [Path(f"{letter}:\\") for letter in string.ascii_uppercase]
    else:
        roots = [Path.home(), Path("/")] + [p for base in ("/mnt", "/media", "/Volumes") if Path(base).is_dir()
                                            for p in Path(base).iterdir()]
    for root in roots:
        try:
            if not root.exists():
                continue
            dev = root.stat().st_dev
            use = shutil.disk_usage(root)
        except OSError:  # an empty card reader, a disconnected share
            continue
        if dev in seen:
            continue
        seen.add(dev)
        out.append((root, use.free, use.total))
    return sorted(out, key=lambda d: -d[1])


def folder_on(root: Path) -> Path:
    """Where model files go on a drive: Vibe Hash's own folder on the drive it's on, else <drive>/vibehash-models."""
    home = settings.home()
    try:
        same = Path(root).stat().st_dev == (home if home.exists() else Path.home()).stat().st_dev
    except OSError:
        same = False
    return home / "models" if same else Path(root) / "vibehash-models"


def default_folder() -> Path:
    """On the drive with the most room."""
    found = drives()
    return folder_on(found[0][0]) if found else settings.home() / "models"


def free_space(folder: Path) -> int:
    """Free bytes on the drive `folder` is (or would be) on."""
    p = Path(folder)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def gb(n: float) -> str:
    return f"{n / GB:.1f} GB"


@dataclass
class Plan:
    """What getting a job's files would take: `need` downloaded, `evict` deleted to make room under the cap."""
    job: str
    what: str
    need: dict[str, int]
    have: list[str]
    evict: list[str]
    fits: bool
    why_not: str = ""

    @property
    def download(self) -> int:
        return sum(self.need.values())


@dataclass
class Store:
    folder: Path
    cap: int = int(DEFAULT_CAP_GB * GB)
    keep: bool = False
    idle_days: float = IDLE_DAYS
    clock: object = time.time
    files: dict = field(default_factory=dict)  # name -> {"size", "jobs", "used", "kind", "what", "partial"}

    def __post_init__(self):
        self.folder = Path(self.folder)
        try:
            self.files = json.loads((self.folder / INDEX).read_text())["files"]
        except (OSError, ValueError, KeyError):
            self.files = {}

    @classmethod
    def from_settings(cls, **overrides) -> "Store":
        """The store as config.json's "models" sets it: {"folder", "cap_gb", "keep"} (setup asks for the folder
        once; until then, the default)."""
        cfg = settings.load().get("models", {})
        args = {"folder": Path(cfg["folder"]) if cfg.get("folder") else default_folder(),
                "cap": int(float(cfg.get("cap_gb", DEFAULT_CAP_GB)) * GB), "keep": bool(cfg.get("keep", False))}
        return cls(**{**args, **overrides})

    def remember(self) -> None:
        """Save this store's folder, cap and keep to config.json, for next time."""
        settings.save(models={"folder": str(self.folder), "cap_gb": self.cap / GB, "keep": self.keep})

    # --- the index ---

    def _save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.folder / (INDEX + ".tmp")
        tmp.write_text(json.dumps({"files": self.files}, indent=1))
        os.replace(tmp, self.folder / INDEX)

    def path(self, name: str) -> Path:
        return self.folder / name

    def _on_disk(self, name: str) -> int:
        """Bytes a file takes now: the file, or its partial download."""
        total = 0
        for p in [self.path(name)] + [self.path(name + suffix) for suffix in PARTIALS]:
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    def _taken(self, name: str) -> int:
        """What a file counts for against the cap: its bytes, or a partial download's final size."""
        e = self.files.get(name, {})
        return max(self._on_disk(name), e.get("size", 0) if e.get("partial") else 0)

    def usage(self) -> int:
        """Bytes the store counts against the cap (the dashboard's disk use)."""
        return sum(self._taken(n) for n in self.files)

    def _delete(self, name: str) -> None:
        for p in [self.path(name)] + [self.path(name + suffix) for suffix in PARTIALS]:
            p.unlink(missing_ok=True)
        self.files.pop(name, None)
        self._tidy(self.path(name).parent)

    def _tidy(self, parent: Path) -> None:
        """Remove folders left empty, up to the store's own."""
        while parent != self.folder and self.folder in parent.parents and parent.is_dir() \
                and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent

    # --- getting a job's files ---

    def plan(self, job: str, files: dict[str, int], what: str = "model files") -> Plan:
        """What fetching `files` ({name: bytes}) for `job` would take, under the cap and the disk's free space."""
        have = [n for n in files if n in self.files and not self.files[n].get("partial")
                and self.path(n).exists()]
        need = {n: s for n, s in files.items() if n not in have}
        wanted = sum(need.values())
        resumed = sum(self._on_disk(n) for n in need)  # bytes of partial downloads already here
        room = self.cap - sum(self._taken(n) for n in self.files if n not in need)
        evict = []
        for _, n in sorted((e.get("used", 0), n) for n, e in self.files.items()
                           if n not in files and e["kind"] == "model"):  # least recently used first
            if wanted <= room:
                break
            room += self._taken(n)
            evict.append(n)
        why = ""
        if wanted > room:
            why = (f"this job's files are {gb(sum(files.values()))} and the space you allow is {gb(self.cap)}: "
                   "raise it (vibehash setup) to mine this job; other jobs carry on")
        else:
            free = free_space(self.folder) + sum(self._on_disk(n) for n in evict)
            if wanted - resumed > free - DISK_MARGIN:
                why = (f"{self.folder} has {gb(free)} free and this job needs {gb(wanted - resumed)} more: choose "
                       "another drive (vibehash setup); other jobs carry on")
        return Plan(job, what, need, have, evict, not why, why)

    def ask(self, plan: Plan, yes: bool = False, ask=input, say=print) -> bool:
        """Before a big download: say what, how big and where; yes, no, or another place (the drives listed with
        their free space, the roomiest first). `yes`: unattended, say yes. -> whether to go ahead (with the folder
        possibly changed, and remembered)."""
        if not plan.fits:
            say(f"Can't take {plan.job}: {plan.why_not}.")
            return False
        if not plan.need:
            return True
        while True:
            say(f"{plan.job} needs {plan.what}: {gb(plan.download)} to download into {self.folder} "
                f"({gb(free_space(self.folder))} free there; you allow up to {gb(self.cap)}).")
            if plan.evict:
                say(f"To make room, {len(plan.evict)} older model file(s) would be deleted: "
                    f"{gb(sum(self.files[n]['size'] for n in plan.evict))}.")
            say("They're deleted when the job ends, or after a week without mining it"
                + (" (you've chosen to keep them for next time instead)." if self.keep else "."))
            if yes:
                return True
            answer = ask("Download? [Y]es, [n]o, [c]hoose another place: ").strip().lower()
            if answer in ("", "y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            if answer in ("c", "choose"):
                self.choose(ask, say)
                plan = self.plan(plan.job, {**plan.need, **{n: self.files[n]["size"] for n in plan.have}}, plan.what)
                if not plan.fits:
                    say(f"Can't take {plan.job} there: {plan.why_not}.")
                    return False

    def choose(self, ask=input, say=print) -> Path:
        """Pick where model files go (setup asks this once): the drives with their free space, the roomiest first
        and the default, or any folder typed in. Remembered. What the old place holds is deleted, not left behind
        (a job downloads its files into the new place when it needs them)."""
        found = drives()
        for k, (root, free, total) in enumerate(found, 1):
            say(f"  {k}. {root}  {gb(free)} free of {gb(total)}")
        answer = ask(f"Where should model files go? [1-{len(found)}] or a folder (Enter: {default_folder()}): ").strip()
        if not answer:
            chosen = default_folder()
        elif answer.isdigit() and 1 <= int(answer) <= len(found):
            chosen = folder_on(found[int(answer) - 1][0])
        else:
            chosen = Path(answer).expanduser()
        if chosen != self.folder:
            if self.files:
                say(f"The {gb(self.usage())} in {self.folder} is deleted; jobs download their files into {chosen}.")
                self.clear()
            self.folder = chosen
            self.__post_init__()
        self.remember()
        return chosen

    def acquire(self, plan: Plan, fetch) -> dict[str, Path]:
        """Make room (plan.evict), then fetch what's missing: fetch(name, dest) writes dest, through its .part file
        (deepseek.fetch_file with folder=dest.parent, say), and resumes one left from before. -> {name: path} of all
        the job's files, marked used now."""
        if not plan.fits:
            raise RuntimeError(plan.why_not)
        for n in plan.evict:
            self._delete(n)
        now = self.clock()
        for n, size in plan.need.items():
            entry = self.files.setdefault(n, {"size": size, "jobs": [], "kind": "model", "what": plan.what})
            entry.update(partial=True, used=now)
            if plan.job not in entry["jobs"]:
                entry["jobs"].append(plan.job)
            self._save()  # a crash now leaves an index that knows of the partial: clear and sweep can find it
            dest = self.path(n)
            dest.parent.mkdir(parents=True, exist_ok=True)
            fetch(n, dest)
            entry["partial"] = False
            entry["size"] = dest.stat().st_size
            self._save()
        for n in plan.have:
            if plan.job not in self.files[n]["jobs"]:
                self.files[n]["jobs"].append(plan.job)
        self.touch(plan.job)
        return {n: self.path(n) for n in list(plan.need) + plan.have}

    def touch(self, job: str) -> None:
        """The job was mined just now: its files aren't idle."""
        now = self.clock()
        for e in self.files.values():
            if job in e["jobs"]:
                e["used"] = now
        self._save()

    # --- temporary files: hand-offs, data shards, checkpoints ---

    def temporary(self, job: str, name: str, what: str = "a hand-off") -> Path:
        """A path for a temporary file of `job`, counted against the cap like the rest; release() it once it's been
        used and accepted."""
        self.files[name] = {"size": 0, "jobs": [job], "kind": "temporary", "what": what, "used": self.clock()}
        self._save()
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def release(self, name: str) -> None:
        self._delete(name)
        self._save()

    # --- going away ---

    def job_ended(self, job: str) -> list[str]:
        """The job is over: its temporaries go, and its model files too unless `keep` or another job still uses
        them. -> the files deleted."""
        gone = []
        for n, e in list(self.files.items()):
            if job not in e["jobs"]:
                continue
            e["jobs"].remove(job)
            if e["kind"] == "temporary" or (not e["jobs"] and (not self.keep or e.get("partial"))):
                self._delete(n)
                gone.append(n)
        self._save()
        return gone

    def sweep(self) -> list[str]:
        """Delete model files no job has used for `idle_days` (unless `keep`), and stray partial downloads: run it
        when the worker starts and now and then. -> the files deleted."""
        cutoff = self.clock() - self.idle_days * 86400
        gone = [n for n, e in self.files.items()
                if e.get("used", 0) < cutoff and (not self.keep or e.get("partial") or e["kind"] == "temporary")]
        for n in gone:
            self._delete(n)
        for p in self._partials():  # partial downloads the index doesn't know (from before it, or another tool)
            rel = p.relative_to(self.folder).as_posix()
            if rel.rsplit(".", 1)[0] not in self.files:
                p.unlink(missing_ok=True)
                self._tidy(p.parent)
                gone.append(rel)
        self._save()
        return gone

    def _partials(self) -> list[Path]:
        if not self.folder.exists():
            return []
        return [p for p in self.folder.rglob("*") if p.is_file() and p.name.endswith(PARTIALS)]

    def clear(self, job: str | None = None) -> int:
        """Remove every file in the store (job=None), partial downloads included, or one job's (files another job
        still uses stay). -> bytes freed."""
        before = sum(self._on_disk(n) for n in self.files)
        if job is None:
            for n in list(self.files):
                self._delete(n)
            for p in self._partials():
                p.unlink(missing_ok=True)
                self._tidy(p.parent)
        else:
            for n, e in list(self.files.items()):
                if job in e["jobs"]:
                    e["jobs"].remove(job)
                    if not e["jobs"]:
                        self._delete(n)
        self._save()
        return before - sum(self._on_disk(n) for n in self.files)

    def entries(self) -> list[dict]:
        """What's stored, for `vibehash cache`: each file's name, path, bytes, kind, what it is, which jobs use it, when
        last used, and why it's still here."""
        out = []
        for n, e in sorted(self.files.items()):
            if e.get("partial"):
                why = "a download under way or interrupted: it resumes, or goes with `vibehash cache clear`"
            elif e["kind"] == "temporary":
                why = "in use; deleted once used and accepted"
            elif self.keep:
                why = "kept for next time (your setting); deleted only to make room under your cap"
            else:
                left = max(0.0, e["used"] + self.idle_days * 86400 - self.clock()) / 86400
                why = f"for {', '.join(e['jobs']) or 'no job'}: deleted when it ends, or in {left:.1f} days unmined"
            out.append({"name": n, "path": str(self.path(n)), "bytes": self._on_disk(n), "kind": e["kind"],
                        "what": e.get("what", ""), "jobs": list(e["jobs"]), "used": e.get("used"), "why": why})
        return out

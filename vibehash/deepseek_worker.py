"""A worker's side of a DeepSeek-V4.1 top-layers job (DEEPSEEK.md §8): whether this PC can take its work, the model
files that needs (through modelstore: asked first, within the space the user allows, deleted with the job), running
a prep unit it was given, and checking another miner's by replaying it.
"""
from __future__ import annotations

from pathlib import Path

import torch

from . import deepseek as ds
from . import deepseek_prep as prep
from . import deepseek_split as split
from . import hardware
from . import modelstore
from .deepseek_job import DeepSeekSpec, TokenFile

# A replay on other hardware: each output's distance (deepseek_split.distance) from the committed one may be this
# much. The reference's arithmetic differs by about 1% a layer between devices (§4), and a piece runs `pieces` layers;
# the same hardware agrees to the bit. A first guess, which the first live job measures (its units' stats).
PREP_TOLERANCE = 0.05
NEEDS = {"prep": {"gpu_gb": 6, "ram_gb": 16}, "train": {"gpu_gb": 8, "ram_gb": 16}}


def machine() -> dict:
    """What this PC has: its NVIDIA GPU (name and GB, or None) and its RAM in GB."""
    gpu = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {"name": p.name, "gb": p.total_memory / 2**30}
    return {"gpu": gpu, "ram_gb": hardware.total_memory_mb() / 1024}


def can_take(stage: str, mach: dict | None = None, plan: modelstore.Plan | None = None) -> tuple[bool, str]:
    """Whether this PC can take a DeepSeek job's `stage` ("prep" or "train") work, and if not, what to tell its owner:
    the reason, and that it keeps earning on other jobs."""
    mach = mach or machine()
    need = NEEDS[stage]
    why = ""
    if mach["gpu"] is None:
        why = f"it needs an NVIDIA graphics card with {need['gpu_gb']} GB or more (RTX 20-series or newer)"
    elif mach["gpu"]["gb"] < need["gpu_gb"] - 0.5:  # (an "8 GB" card reports a little under 8)
        why = f"its {mach['gpu']['name']} has {mach['gpu']['gb']:.0f} GB, and this work needs {need['gpu_gb']} GB"
    elif mach["ram_gb"] < need["ram_gb"] - 1:
        why = f"it has {mach['ram_gb']:.0f} GB of memory, and this work needs {need['ram_gb']} GB"
    elif plan is not None and not plan.fits:
        why = plan.why_not
    if why:
        return False, f"This PC can't take DeepSeek work: {why}. It keeps earning on other jobs."
    return True, ""


def store_name(file: str) -> str:
    """Where one of DeepSeek's files sits in the model store."""
    return f"deepseek/{ds.COMMIT}/{file}"


class Weights:
    """A DeepSeek job's model files on this PC. Flash's are fetched from the pinned commit into the model store (asked
    first, within its cap, deleted with the job) and read memory-mapped; the tiny config's are random, and nothing is
    fetched."""

    def __init__(self, spec: DeepSeekSpec, job_id: str, store: modelstore.Store | None = None, hub=None,
                 yes: bool = False, ask=input, say=print):
        self.spec, self.job_id, self.store, self.yes, self.ask, self.say = spec, job_id, store, yes, ask, say
        self._hub, self._tiny = hub, None

    @property
    def tiny(self) -> bool:
        return self.spec.config_name == "tiny"

    @property
    def hub(self):
        if self._hub is None:
            from .hub import Hub
            self._hub = Hub(backoff=2.0)
        return self._hub

    def files(self, layers, embed: bool = False, head: bool = False, tokenizer: bool = False) -> dict[str, int]:
        """The store names and sizes of the files holding these layers (their engram tables aside: rows units stream
        those), and the embedding, the head and the tokenizer if asked."""
        listing = self.hub.list_files(ds.REPO, ds.COMMIT)
        wm = self.hub.weight_map(ds.REPO, ds.COMMIT)
        want = {f for n, f in wm.items() if n.startswith(tuple(f"layers.{i}." for i in layers)) and ".engram.embed." not in n}
        if embed:
            want.add(wm["embed.weight"])
        if head:
            want.add(wm["head.weight"])
        if tokenizer:
            want.add("tokenizer.json")
        return {store_name(f): listing[f].size for f in sorted(want)}

    def get(self, files: dict[str, int], what: str) -> bool:
        """Have these files here, asking first. -> False if the owner said no or they can't fit (and why is said)."""
        if self.tiny or not files:
            return True
        plan = self.store.plan(self.job_id, files, what)
        ok, why = can_take("prep", plan=plan) if not plan.fits else (True, "")
        if not ok:
            self.say(why)
            return False
        if not self.store.ask(plan, yes=self.yes, ask=self.ask, say=self.say):
            return False

        def fetch(name, dest):
            file = name.split("/", 2)[2]
            if file == "tokenizer.json":
                info = self.hub.list_files(ds.REPO, ds.COMMIT)[file]
                dest.write_bytes(self.hub.get_file(ds.REPO, ds.COMMIT, file, info=info))
            else:
                ds.fetch_file(file, self.hub, folder=dest.parent, log=self.say)
        self.store.acquire(plan, fetch)
        return True

    def source(self):
        """Where a prep unit reads its weights (deepseek_prep's source interface)."""
        if self.tiny:
            if self._tiny is None:
                self._tiny = prep.LocalSource(ds.random_weights(self.spec.cfg, 0))
            return self._tiny
        return _StoreSource(self.store.folder / f"deepseek/{ds.COMMIT}", self.hub)

    def token_map(self, work: Path) -> torch.Tensor:
        """The compressed token map engram hashing uses: rebuilt from the tokenizer for Flash; fixed for tiny."""
        if self.tiny:
            return tiny_token_map(self.spec.cfg)
        path = self.store.folder / store_name("tokenizer.json")
        return torch.from_numpy(prep.compressed_token_map(path))

    def trained(self) -> dict:
        """The trained layers' and the head's tensors, the routed experts memory-mapped (deepseek_top.build_trained)."""
        if self.tiny:
            from .deepseek_unit import tiny_weights
            return tiny_weights(self.spec)
        out = {}
        for name in self.files(self.spec.top.trained, head=True):
            out.update(ds.read_safetensors(self.store.folder / name, mmap=True))
        return {k: v for k, v in out.items() if k.startswith(("norm.", "head.")) or
                (k.startswith("layers.") and int(k.split(".")[1]) >= self.spec.cut)}


class _StoreSource(prep.HubSource):
    """deepseek_prep.HubSource with its weight files in the model store's folder rather than the cache."""

    def __init__(self, folder: Path, hub):
        self.hub, self.folder = hub, Path(folder)
        self.wm = hub.weight_map(ds.REPO, ds.COMMIT)

    def tokenizer(self, work: Path) -> Path:
        return self.folder / "tokenizer.json"

    def fetch(self, layers, progress=None) -> None:
        pass  # Weights.get fetched them, asking first

    def embed(self) -> torch.Tensor:
        return ds.read_safetensors(self.folder / self.wm["embed.weight"], {"embed.weight"})["embed.weight"]

    def layer(self, layer: int, work: Path) -> dict:
        out = {}
        for f in self._files(layer):
            names = {n for n, g in self.wm.items() if g == f and n.startswith(f"layers.{layer}.")}
            out.update(ds.read_safetensors(self.folder / f, names, mmap=True))
        if layer in ds.Config.flash().engram_layer_ids:
            out.update(self.engram_small(layer, work))
        return out


def sequences(tokens: TokenFile) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The token file as deepseek_prep.load_dataset gives it: (ids, learn mask) per sequence."""
    out, at = [], 0
    for n in tokens.lengths.tolist():
        ids = torch.from_numpy(tokens.tokens[at:at + n].copy())
        learn = torch.from_numpy(tokens.learn[at:at + n].astype(bool))
        out.append((ids, torch.cat((learn[1:], torch.zeros(1, dtype=torch.bool)))))
        at += n
    return out


def tiny_token_map(cfg) -> torch.Tensor:
    """The tiny config's engram token map: fixed, as its weights are (tests and local testnets)."""
    return torch.randint(0, cfg.engram_compressed_vocab_size, (cfg.vocab_size,),
                         generator=torch.Generator().manual_seed(1))


def run_unit(spec: DeepSeekSpec, tokens: TokenFile, unit: split.Unit, store, weights: Weights, work: Path,
             want: dict | None = None, device: str = "cpu", tamper=None) -> dict:
    """Run a prep unit it was given: its inputs from `store` (checked against `want`, their writers' digests), its
    outputs written there. -> its commitment."""
    seqs, chunks = sequences(tokens), tokens.chunks(spec.chunk_tokens)
    tmap = weights.token_map(work)
    if unit.kind == "rows":
        return split.run_rows(weights.source(), store, seqs, chunks, spec.top, unit.a, unit.b, tmap, work,
                              spec.row_ranges, spec.align)
    return split.run_piece(weights.source(), store, seqs, chunks, spec.top, unit, tmap, work, device,
                           ranges=spec.row_ranges, align=spec.align, want=want, tamper=tamper)


def check_rows(spec: DeepSeekSpec, tokens: TokenFile, unit: split.Unit, commit: dict, weights: Weights,
               work: Path) -> bool:
    """Check another miner's rows unit: the same rows streamed again, compared to the bit (they're the table's own
    bytes, the same on any hardware)."""
    scratch = split.LocalStore(Path(work) / f"rows-{unit.a}-{unit.b}")
    mine = run_unit(spec, tokens, unit, scratch, weights, Path(work))
    return mine["outputs"] == commit.get("outputs") and mine["range"] == commit.get("range")


def check_unit(spec: DeepSeekSpec, tokens: TokenFile, unit: split.Unit, commit: dict, store, weights: Weights,
               work: Path, want: dict | None = None, device: str = "cpu",
               tolerance: float = PREP_TOLERANCE) -> tuple[bool, str]:
    """Check another miner's layers unit by replaying it. -> (whether it holds, what was worst)."""
    seqs, chunks = sequences(tokens), tokens.chunks(spec.chunk_tokens)
    got = split.replay_close(weights.source(), store, seqs, chunks, spec.top, unit, commit, tolerance,
                             weights.token_map(work), work, device, ranges=spec.row_ranges, align=spec.align,
                             want=want)
    return got["agrees"], f"worst {got['worst']:.3g} at {got['where']}" if got["where"] else "the same, to the bit"


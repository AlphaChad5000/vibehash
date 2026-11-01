"""The one-time prep of a top-layers DeepSeek-V4.1-Flash job (DEEPSEEK.md §8): the frozen layers over the job's text,
once, published as the job's dataset with its Merkle root.

    python -m vibehash.deepseek_prep --data tokens.npz --repo USER/NAME [--cut 36] [--work DIR] [--fp8]

for a Linux or Windows machine with a CUDA GPU. It needs the runner's own Hugging Face token (`hf auth login`, or
$HF_TOKEN) and `pip install torch huggingface_hub tokenizers`. It runs unattended and can be stopped at any point:
run the same command again and it carries on from where it was, after a reboot or a dropped connection too.
Progress, with an ETA, goes to the terminal and to WORK/prep.log. The output isn't trusted: it's checked like any
work (DEEPSEEK.md §8, "Checking").

The input is the job's text as token ids: an .npz with `tokens` (int32, every sequence end to end), `lengths`
(int32, one per sequence) and `learn` (uint8, 1 where a token is a target to learn).

Phases, each resumable (WORK/state.json says where it is):
1. hashes: the engram n-gram hashes of every token, and so the table rows each engram layer below the cut reads.
   Needs the tokenizer (fetched from the pinned commit) for the compressed token map.
2. rows: each engram table (101 GB) streamed once from the pinned commit in parallel pieces, keeping only those rows
   (WORK/engram-L.npz, a few GB). Nothing else of it is stored.
3. layers: the frozen layers' weight files fetched whole into the cache (~/.vibehash/cache/deepseek, or
   $VIBEHASH_HOME), each checked against the Hub's sha256: 267 GB for a cut at 36.
4. blocks: blocks of sequences through every frozen layer in turn. Each layer comes off the disk once per block; on
   a GPU with less than --whole-layer-gb free, its routed experts stay in RAM and come over one at a time
   (deepseek_top.forward_streamed). After each layer, each sequence's state is hashed. Each block's records and
   hashes are written, uploaded, and then deleted locally.
5. manifest: the Merkle root over every record, which the job pins, with each record's per-layer hashes. Uploaded
   last; the repo's commit is printed.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import deepseek as ds
from . import deepseek_top as top
from .crypto import merkle_root
from .deepseek import Choices, Config, Model, Side

FORMAT = "vibehash-deepseek-prep/1"
log = logging.getLogger("vibehash.deepseek_prep")


class Stop(Exception):
    """Raised by the tests' stop_after, as an interruption would."""


# --- the input -------------------------------------------------------------------------------------------------------

def load_dataset(path: str | Path) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """[(token ids [s], learn mask [s]: position t's next token is learned)], one per sequence."""
    with np.load(path) as z:
        tokens, lengths, learn = z["tokens"].astype(np.int64), z["lengths"].astype(np.int64), z["learn"].astype(bool)
    if lengths.sum() != len(tokens) or len(learn) != len(tokens) or (lengths < 2).any():
        raise ValueError(f"{path}: tokens, lengths and learn don't agree (or a sequence is shorter than 2)")
    out, at = [], 0
    for n in lengths:
        ids, tgt = torch.from_numpy(tokens[at:at + n]), torch.from_numpy(learn[at:at + n])
        out.append((ids, torch.cat((tgt[1:], torch.zeros(1, dtype=torch.bool)))))
        at += n
    return out


def save_dataset(path: str | Path, seqs: list[tuple[list[int], list[int]]]) -> None:
    """The input format, from [(token ids, learn flags)]: for tests and for posters' own tools."""
    np.savez(path, tokens=np.concatenate([np.asarray(t, np.int32) for t, _ in seqs]),
             lengths=np.asarray([len(t) for t, _ in seqs], np.int32),
             learn=np.concatenate([np.asarray(m, np.uint8) for _, m in seqs]))


# --- engram: the token map, the rows a dataset reads, and streaming them out of a table ------------------------------

def compressed_token_map(tokenizer_json: str | Path) -> np.ndarray:
    """engram.py's build_compressed_token_map: every token id to an id shared by tokens that normalise alike
    (NFKC, accents stripped, lower case, whitespace runs as one space). Needs the `tokenizers` package. For
    DeepSeek-V4.1-Flash's tokenizer it makes 99,092 ids, as config.json's engram_compressed_vocab_size says."""
    from tokenizers import Regex, Tokenizer, normalizers
    tok = Tokenizer.from_file(str(tokenizer_json))
    sentinel = ""  # keeps a token that is one space from being stripped to nothing
    norm = normalizers.Sequence([normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(),
                                 normalizers.Lowercase(), normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                                 normalizers.Replace(Regex(r"^ $"), sentinel), normalizers.Strip(),
                                 normalizers.Replace(sentinel, " ")])
    keys: dict[str, int] = {}
    out = np.zeros(tok.get_vocab_size(with_added_tokens=True), np.int64)
    for i in range(len(out)):
        text = tok.decode([i], skip_special_tokens=False)
        key = tok.id_to_token(i) if "�" in text else (norm.normalize_str(text) or text)
        out[i] = keys.setdefault(key, len(keys))
    return out


def needed_rows(seqs, token_map: torch.Tensor, cfg: Config, layers) -> dict[int, np.ndarray]:
    """{engram layer: the sorted distinct table rows the dataset's tokens read}."""
    found = {layer: set() for layer in layers}
    for ids, _ in seqs:
        h = ds.engram_hashes(ids[None], token_map, cfg)[0]  # [s, engram layers, cols]
        for layer in layers:
            found[layer].update(np.unique(h[:, cfg.engram_layer_ids.index(layer)].numpy()).tolist())
    return {layer: np.asarray(sorted(v), np.int64) for layer, v in found.items()}


def extract_rows(read, start: int, row_bytes: int, wanted: np.ndarray, folder: Path, piece_rows: int = 2**18,
                 workers: int = 6, progress=None, stop_after: int | None = None) -> np.ndarray:
    """The bytes of rows `wanted` (sorted) of a table that starts at byte `start` of a remote file, `read(offset,
    size)` giving its bytes: the table streamed in pieces of piece_rows rows, several at once, only the pieces
    holding a wanted row, each piece's rows kept in `folder` as it finishes, so a restart skips the pieces done.
    -> uint8 [len(wanted), row_bytes]."""
    from concurrent.futures import ThreadPoolExecutor
    folder.mkdir(parents=True, exist_ok=True)
    pieces = np.unique(wanted // piece_rows)
    todo = [int(p) for p in pieces if not (folder / f"piece-{p}.npy").exists()]
    done = [len(pieces) - len(todo)]

    def get(p):
        rows = wanted[(wanted >= p * piece_rows) & (wanted < (p + 1) * piece_rows)]
        lo, hi = int(rows[0]), int(rows[-1]) + 1  # just the span holding them
        blob = np.frombuffer(read(start + lo * row_bytes, (hi - lo) * row_bytes), np.uint8).reshape(hi - lo, row_bytes)
        tmp = folder / f"piece-{p}.tmp.npy"
        np.save(tmp, blob[rows - lo])
        os.replace(tmp, folder / f"piece-{p}.npy")
        done[0] += 1
        if progress:
            progress(done[0], len(pieces))
        if stop_after is not None and done[0] - (len(pieces) - len(todo)) >= stop_after:
            raise Stop()

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(get, p) for p in todo]
        try:
            for f in futures:
                f.result()
        except BaseException:  # a failed piece (or a stop): start no more, and let a rerun carry on
            for f in futures:
                f.cancel()
            raise
    return np.concatenate([np.load(folder / f"piece-{int(p)}.npy") for p in pieces])


@dataclasses.dataclass
class Slice:
    """The rows of an engram table a dataset reads: sorted row ids, their FP8 values and ue8m0 scales as stored."""
    ids: np.ndarray
    weight: np.ndarray  # uint8 [n, head_dim]
    scale: np.ndarray  # uint8 [n, head_dim / 32]

    def save(self, folder: Path) -> None:
        """Three .npy files in `folder`, read back memory-mapped: a slice for 10M tokens can be tens of GB."""
        tmp = folder.with_name(folder.name + ".tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        for name in ("ids", "weight", "scale"):
            np.save(tmp / f"{name}.npy", getattr(self, name))
        if folder.exists():
            shutil.rmtree(folder)
        os.replace(tmp, folder)

    @classmethod
    def load(cls, folder: Path) -> "Slice":
        return cls(*(np.load(folder / f"{name}.npy", mmap_mode="r") for name in ("ids", "weight", "scale")))

    def rows(self, hashes: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Table rows for hash ids [..., cols] -> [..., cols, head_dim], dequantised exactly as Engram.rows does."""
        at = np.searchsorted(self.ids, hashes.numpy())
        if not np.array_equal(self.ids[np.minimum(at, len(self.ids) - 1)], hashes.numpy()):
            raise KeyError("an engram row the slice doesn't have")
        w = torch.from_numpy(self.weight[at]).view(torch.float8_e4m3fn).float().unflatten(-1, (-1, ds.BLOCK))
        s = torch.from_numpy(self.scale[at]).view(torch.float8_e8m0fnu).float().unsqueeze(-1)
        return (w * s).flatten(-2).to(dtype)


# --- where the weights come from -------------------------------------------------------------------------------------

class HubSource:
    """DeepSeek's checkpoint at deepseek.COMMIT: whole weight files into the cache, engram tables streamed."""

    def __init__(self, hub=None):
        from .hub import Hub
        self.hub = hub or Hub(backoff=2.0)
        self.wm = self.hub.weight_map(ds.REPO, ds.COMMIT)

    def tokenizer(self, work: Path) -> Path:
        path = work / "tokenizer.json"
        if not path.exists():
            info = self.hub.list_files(ds.REPO, ds.COMMIT)["tokenizer.json"]
            path.write_bytes(self.hub.get_file(ds.REPO, ds.COMMIT, "tokenizer.json", info=info))
        return path

    def fetch(self, layers, progress=None) -> None:
        files = {self.wm["embed.weight"]} | {f for i in layers for f in self._files(i)}
        for f in sorted(files):
            ds.fetch_file(f, self.hub, log=progress)

    def _files(self, layer: int) -> set[str]:
        return {f for n, f in self.wm.items() if n.startswith(f"layers.{layer}.") and ".engram." not in n}

    def embed(self) -> torch.Tensor:
        return ds.read_safetensors(ds.cache_dir() / self.wm["embed.weight"], {"embed.weight"})["embed.weight"]

    def layer(self, layer: int, work: Path) -> dict:
        """Layer `layer`'s tensors, without its engram table (whose rows are in the slice)."""
        out = {}
        for f in self._files(layer):
            names = {n for n, g in self.wm.items() if g == f and n.startswith(f"layers.{layer}.")}
            out.update(ds.read_safetensors(ds.cache_dir() / f, names))
        small = work / f"engram-{layer}-small.pt"
        if small.exists():
            out.update(torch.load(small))
        return out

    def slice(self, layer: int, wanted: np.ndarray, work: Path, progress=None, stop_after=None) -> Slice:
        """Stream layer `layer`'s engram table from the Hub, keeping rows `wanted`; its small tensors (the
        projection and gate weights) are read whole, by range."""
        prefix = f"layers.{layer}.engram."
        file = self.wm[prefix + "embed.weight"]
        import threading
        header = self.hub.header(ds.REPO, ds.COMMIT, file)
        remote, local = self.hub.open(ds.REPO, ds.COMMIT, file), threading.local()

        def read(offset, size):  # one connection per thread: a RemoteFile keeps its CDN address
            if not hasattr(local, "remote"):
                local.remote = self.hub.open(ds.REPO, ds.COMMIT, file)
            return local.remote.read(offset, size)
        parts = {}
        for name in ("embed.weight", "embed.scale"):
            dtype, shape, begin, _ = header.tensors[prefix + name]
            assert dtype in ("F8_E4M3", "F8_E8M0")  # a byte a value: a row is shape[1] bytes
            parts[name] = extract_rows(read, header.data_start + begin, shape[1], wanted,
                                       work / f"engram-{layer}-{name}", progress=progress, stop_after=stop_after)
        small = {}
        for name, (dtype, shape, begin, end) in header.tensors.items():
            if name.startswith(prefix) and ".embed." not in name:
                raw = torch.frombuffer(bytearray(remote.read(header.data_start + begin, end - begin)), dtype=torch.uint8)
                small[name] = raw.view(ds.SAFETENSORS_DTYPES[dtype]).reshape(shape)
        torch.save(small, work / f"engram-{layer}-small.pt")
        return Slice(wanted, parts["embed.weight"], parts["embed.scale"])


class LocalSource:
    """Weights from a state dict in DeepSeek's names and formats (tests, or a checkpoint already on disk)."""

    def __init__(self, state: dict):
        self.state = state

    def fetch(self, layers, progress=None) -> None:
        pass

    def embed(self) -> torch.Tensor:
        return self.state["embed.weight"]

    def layer(self, layer: int, work: Path) -> dict:
        return {k: v for k, v in self.state.items() if k.startswith(f"layers.{layer}.") and ".engram.embed." not in k}

    def slice(self, layer: int, wanted: np.ndarray, work: Path, progress=None, stop_after=None) -> Slice:
        w = self.state[f"layers.{layer}.engram.embed.weight"].view(torch.uint8).numpy()
        s = self.state[f"layers.{layer}.engram.embed.scale"].view(torch.uint8).numpy()
        weight = extract_rows(lambda o, n: w.reshape(-1)[o:o + n].tobytes(), 0, w.shape[1], wanted,
                              work / f"engram-{layer}-embed.weight", piece_rows=64, progress=progress,
                              stop_after=stop_after)
        scale = extract_rows(lambda o, n: s.reshape(-1)[o:o + n].tobytes(), 0, s.shape[1], wanted,
                             work / f"engram-{layer}-embed.scale", piece_rows=64)
        return Slice(wanted, weight, scale)


# --- where the output goes --------------------------------------------------------------------------------------------

class HubUploader:
    """A Hugging Face dataset repo under the runner's own account and token (huggingface_hub)."""

    def __init__(self, repo: str, private: bool = True, token: str | None = None):
        from huggingface_hub import HfApi
        self.api, self.repo = HfApi(token=token), repo
        self.api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)

    def put(self, local: Path, name: str) -> str:
        for attempt in range(8):  # a dorm network drops connections: wait longer each time
            try:
                info = self.api.upload_file(path_or_fileobj=str(local), path_in_repo=name, repo_id=self.repo,
                                            repo_type="dataset", commit_message=f"prep: {name}")
                return info.oid
            except Exception as e:  # huggingface_hub raises several kinds for network faults
                wait = min(600, 15 * 2 ** attempt)
                log.warning("upload of %s failed (%s); retrying in %d s", name, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"couldn't upload {name}")

    def has(self, name: str) -> bool:
        return self.api.file_exists(self.repo, name, repo_type="dataset")


class LocalUploader:
    """A folder standing in for the repo (tests, dry runs)."""

    def __init__(self, folder: Path):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)

    def put(self, local: Path, name: str) -> str:
        shutil.copyfile(local, self.folder / name)
        return hashlib.sha1(name.encode()).hexdigest()

    def has(self, name: str) -> bool:
        return (self.folder / name).exists()


# --- the prep ---------------------------------------------------------------------------------------------------------

def layer_hash(x, pre, side: Side, choices: Choices, layer: int, cfg: Config) -> bytes:
    """One sequence's state after a frozen layer, as the preparer commits to it: the stream, pre_mix, the side
    channels this layer made, and its choices (routing, picks, candidate blocks). A dispute starts from these."""
    h = hashlib.sha256(b"%d" % layer)
    parts = [x, pre, choices.experts.get(layer)]
    if layer in cfg.kv_source_layers:
        parts += [side.compress_kv, side.index_k]
    if layer in cfg.index_source_layers:
        parts.append(choices.picks.get(layer))
    if layer == cfg.candidate_source_layer:
        parts.append(choices.candidates)
    for t in parts:
        if t is None:
            h.update(b"-")
        else:
            t = t.detach().contiguous().cpu()
            h.update(repr((str(t.dtype), tuple(t.shape))).encode())
            h.update(t.view(torch.uint8).numpy().tobytes())
    return h.digest()


class Progress:
    """Lines of progress with an ETA, to the log."""

    def __init__(self, what: str, total: float, unit: str):
        self.what, self.total, self.unit, self.t0, self.last = what, total, unit, time.time(), 0.0

    def __call__(self, done: float, total: float | None = None, force: bool = False):
        total = total if total is not None else self.total
        now = time.time()
        if not force and now - self.last < 10 and done < total:
            return
        self.last = now
        rate = done / max(now - self.t0, 1e-9)
        eta = (total - done) / rate if rate > 0 else float("inf")
        log.info("%s: %s of %s %s, %s left", self.what, f"{done:,.0f}", f"{total:,.0f}", self.unit,
                 _duration(eta) if eta != float("inf") else "?")


def _duration(s: float) -> str:
    return f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min" if s >= 60 else f"{s:.0f} s"


def blocks_of(seqs, block_tokens: int) -> list[list[int]]:
    out, cur, n = [], [], 0
    for i, (ids, _) in enumerate(seqs):
        if cur and n + len(ids) > block_tokens:
            out.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += len(ids)
    return out + ([cur] if cur else [])


def run(source, uploader, seqs, cut: top.Cut, work: Path, block_tokens: int = 131072, fp8: bool = False,
        device="cpu", whole_layer_gb: float = 14.0, token_map=None, stop_after_blocks: int | None = None,
        stop_after_pieces: int | None = None) -> dict:
    """The prep, from wherever WORK says it stopped. Returns the manifest (also uploaded as manifest.json)."""
    cfg = cut.cfg
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"format": FORMAT, "blocks": {}}

    def save():
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        os.replace(tmp, state_path)

    frozen = list(range(cut.first))
    engram = [layer for layer in frozen if layer in cfg.engram_layer_ids]
    slices = {}
    if engram:  # phases 1 and 2
        if token_map is None:
            token_map = torch.from_numpy(compressed_token_map(source.tokenizer(work)))
        for layer in engram:
            path = work / f"engram-{layer}"
            if not path.exists():
                wanted = needed_rows(seqs, token_map, cfg, [layer])[layer]
                log.info("engram layer %d: the dataset reads %s of its table's %s rows", layer, f"{len(wanted):,}",
                         f"{cfg.engram_num_embeddings[cfg.engram_layer_ids.index(layer)]:,}")
                source.slice(layer, wanted, work, Progress(f"engram table {layer}", 0, "pieces"),
                            stop_after=stop_after_pieces).save(path)
            slices[layer] = Slice.load(path)
    source.fetch(frozen, progress=lambda m: log.info("%s", m))  # phase 3
    embed = source.embed()
    blocks = blocks_of(seqs, block_tokens)
    total = sum(len(seqs[i][0]) for b in blocks for i in b)
    done_tokens = sum(len(seqs[i][0]) for k, b in enumerate(blocks) if str(k) in state["blocks"] for i in b)
    progress = Progress("prep", total, "tokens")
    progress.t0 -= 1e-6
    t_start, fresh = time.time(), 0
    for k, members in enumerate(blocks):  # phase 4
        if str(k) in state["blocks"]:
            continue
        info = _block(source, uploader, seqs, members, k, cut, work, embed, slices, token_map, fp8, device,
                      whole_layer_gb)
        state["blocks"][str(k)] = info
        save()
        fresh += sum(len(seqs[i][0]) for i in members)
        done_tokens += sum(len(seqs[i][0]) for i in members)
        rate = fresh / max(time.time() - t_start, 1e-9)
        log.info("prep: block %d of %d done; %s of %s tokens, %s left", k + 1, len(blocks), f"{done_tokens:,}",
                 f"{total:,}", _duration((total - done_tokens) / rate))
        if stop_after_blocks is not None and fresh and len(state["blocks"]) >= stop_after_blocks:
            raise Stop()
    leaves = [bytes.fromhex(leaf) for k in range(len(blocks)) for leaf in state["blocks"][str(k)]["leaves"]]
    manifest = {"format": FORMAT, "repo": ds.REPO, "commit": ds.COMMIT, "cut": cut.first, "fp8_stream": fp8,
                "sequences": len(seqs), "tokens": total, "blocks": [state["blocks"][str(k)] for k in range(len(blocks))],
                "root": merkle_root(leaves).hex(),
                "leaf": "sha256(sha256(record) + the sequence's per-layer hashes, in block-NNNNN.json)"}
    path = work / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    manifest["upload"] = uploader.put(path, "manifest.json")  # phase 5
    log.info("prep: done. Merkle root %s; manifest uploaded (commit %s)", manifest["root"], manifest["upload"])
    return manifest


def _block(source, uploader, seqs, members, k, cut, work, embed, slices, token_map, fp8, device, whole_layer_gb):
    cfg = cut.cfg
    num = ds.Numerics()
    states = []
    for i in members:
        ids = seqs[i][0][None]
        x = F.embedding(ids, embed).to(num.dtype)
        pre = torch.zeros(1, ids.size(1), cfg.hc_mult, dtype=num.hi)
        pre[..., 0] = 1
        states.append((x.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1), pre, Side(), Choices()))  # parked in RAM
    hashes = [[] for _ in members]
    for layer in range(cut.first):
        weights = source.layer(layer, work)
        block = _load_block(cfg, layer, weights, device, whole_layer_gb)
        rows = None
        if layer in slices:
            col = cfg.engram_layer_ids.index(layer)
            rows = [slices[layer].rows(ds.engram_hashes(seqs[i][0][None], token_map, cfg)[:, :, col], num.dtype)
                    for i in members]
        run_layer(block, states, device, rows=rows)
        for j, (x, pre, side, choices) in enumerate(states):
            hashes[j].append(layer_hash(x, pre, side, choices, layer, cfg))
        del block, weights
    records, leaves = [], []
    for j, i in enumerate(members):
        x, pre, side, choices = states[j]
        kept, cand = top.crossing_of(cut, side, choices, seqs[i][0].numel())
        ex = top.Example(seqs[i][0][None], seqs[i][1][None], x.cpu(), pre.cpu(),
                         Side(*(None if t is None else t.cpu() for t in (kept.compress_kv, kept.index_k, kept.picks))),
                         None if cand is None else cand.cpu())
        rec = top.pack(ex, 0, fp8_stream=fp8)
        records.append(rec)
        leaves.append(hashlib.sha256(hashlib.sha256(rec).digest() + b"".join(hashes[j])).hexdigest())
    name = f"block-{k:05d}.bin"
    path = work / name
    offsets = np.cumsum([0] + [len(r) for r in records]).tolist()
    with open(path, "wb") as f:
        for r in records:
            f.write(r)
    meta = work / f"block-{k:05d}.json"
    meta.write_text(json.dumps({"sequences": list(members), "offsets": offsets,
                                "layer_hashes": [[h.hex() for h in hs] for hs in hashes]}))
    commit = uploader.put(path, name)
    uploader.put(meta, meta.name)
    path.unlink()
    meta.unlink()
    return {"file": name, "first": members[0], "count": len(members), "leaves": leaves, "commit": commit}


@torch.no_grad()
def run_layer(block, states: list, device, rows=None) -> None:
    """One frozen layer over a block's sequences, their states parked in RAM (updated in place): each sequence comes
    to `device` for the stages before and after the MoE, and each routed expert gets every token of the block routed
    to it in one transfer (and is itself brought over once, if it isn't there), then works through them sequence by
    sequence. So the arithmetic is the plain forward's, sequence by sequence, to the bit, while a block of 128K tokens
    needs only one sequence's working memory on the GPU."""
    moe, hi = block.ffn, block.ffn.num.hi
    to = (lambda t, where: None if t is None else t.to(where))
    parts = []
    for j, (x, pre, side, choices) in enumerate(states):
        side = Side(*(to(t, device) for t in (side.compress_kv, side.index_k, side.picks)))
        h, x_mid, post, comb, ffn_pre = block.before_ffn(x.to(device), pre.to(device), side, choices,
                                                         rows=None if rows is None else rows[j].to(device))
        flat = h.reshape(-1, h.size(-1))
        chosen, gates = moe.route(flat, choices)
        states[j] = (x, pre, Side(*(to(t, "cpu") for t in (side.compress_kv, side.index_k, side.picks))), choices)
        parts.append([flat.cpu(), x_mid.cpu(), post.cpu(), comb.cpu(), ffn_pre.cpu(),
                      [(e, tok.cpu(), slot.cpu()) for e, tok, slot in moe.assignments(chosen)], gates.cpu(), h.shape,
                      torch.zeros(flat.shape, dtype=hi)])
    for e in sorted({e for p in parts for e, _, _ in p[5]}):
        mine = [(j, tok, slot) for j, p in enumerate(parts) for e2, tok, slot in p[5] if e2 == e]
        xs = torch.cat([parts[j][0][tok] for j, tok, _ in mine]).to(device)  # one transfer for the whole block
        gs = torch.cat([parts[j][6][tok, slot] for j, tok, slot in mine]).to(device)
        with top._OnDevice(moe.experts[e], device) as expert:
            at = 0
            for j, tok, _ in mine:  # each sequence's tokens alone, as the plain forward runs them
                n = len(tok)
                parts[j][8].index_add_(0, tok, expert(xs[at:at + n], gs[at:at + n, None]).to(hi).cpu())
                at += n
    for j, p in enumerate(parts):
        y = moe.combine(p[8].to(device), p[0].to(device)).view(p[7])
        out = block.after_ffn(y, p[1].to(device), p[2].to(device), p[3].to(device))
        states[j] = (out.cpu(), p[4], states[j][2], states[j][3])


def _load_block(cfg: Config, layer: int, weights: dict, device, whole_layer_gb: float):
    """The layer's Block on `device`, its routed experts there too if the GPU has room for the whole layer, else
    left in (pinned) RAM for forward_streamed to bring over one at a time."""
    room = torch.cuda.mem_get_info()[0] / 2**30 if str(device).startswith("cuda") else float("inf")
    whole = room >= whole_layer_gb
    meta = Model(cfg, [layer], embed=False, head=False, device="meta", engram_table=False)
    experts = {n for n, _ in meta.named_buffers() if ".ffn.experts." in n}
    placed = {}
    for name, t in weights.items():
        if name in experts and not whole:
            placed[name] = t.pin_memory() if str(device).startswith("cuda") else t
        else:
            placed[name] = t.to(device)
    return top.freeze(meta.load_weights(placed)).layers[str(layer)]


# --- the command -------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vibehash.deepseek_prep", description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="the job's text as token ids (.npz: tokens, lengths, learn)")
    ap.add_argument("--repo", required=True, help="your Hugging Face dataset repo for the output, USER/NAME")
    ap.add_argument("--cut", type=int, default=36, help="the first trained layer (default 36: the top 4 trained)")
    ap.add_argument("--work", default=None, help="working folder (default ~/.vibehash/prep/<repo>)")
    ap.add_argument("--fp8", action="store_true", help="store the stream in FP8: half the size, rounded")
    ap.add_argument("--block-tokens", type=int, default=131072,
                    help="tokens a block, held in RAM (about 15 GB for the default): fewer for less RAM, but every "
                         "block reads all the frozen layers off the disk once")
    ap.add_argument("--whole-layer-gb", type=float, default=14.0,
                    help="free GPU memory at which a whole layer goes on the GPU (else experts one at a time)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--public", action="store_true", help="make the dataset repo public (default: private)")
    args = ap.parse_args(argv)
    work = Path(args.work) if args.work else ds.settings.home() / "prep" / args.repo.replace("/", "__")
    work.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(work / "prep.log")])
    ds.configure_torch()
    cut = top.Cut(Config.flash(), args.cut)
    cut.check()
    seqs = load_dataset(args.data)
    log.info("prep of %s for a cut at layer %d: %d sequences, %s tokens; working in %s", args.data, args.cut,
             len(seqs), f"{sum(len(s) for s, _ in seqs):,}", work)
    run(HubSource(), HubUploader(args.repo, private=not args.public), seqs, cut, work, args.block_tokens,
        args.fp8, args.device, args.whole_layer_gb)
    return 0


if __name__ == "__main__":
    sys.exit(main())

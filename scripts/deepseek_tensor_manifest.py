"""Per-tensor sha256 of DeepSeek-V4.1-Flash's trained layers and head, from the pinned commit's files on disk: the
manifest the node's referee checks those weights against (DEEPSEEK.md §8, "Who the referee trusts").

    python scripts/deepseek_tensor_manifest.py [FOLDER] [--out vibehash/data/deepseek_flash_top.json]

FOLDER (default ~/.vibehash/cache/deepseek/<commit>) holds the safetensors files the trained layers and the head are
in (model.safetensors.index.json says which). Each file is read once: its own sha256, which must be the one Hugging Face
publishes for it at the pinned commit (its Git LFS id, checked against the Hub's listing), and every tensor's. The
output is a vibehash-hf-manifest/1 manifest (hub.Manifest) of those files; anyone with the files can make it again and
get the same bytes, so its sha256 is what deepseek_job pins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vibehash import hub  # noqa: E402
from vibehash.deepseek_config import COMMIT, REPO  # noqa: E402

CHUNK = 64 * 2**20


def tensors_of(path: Path) -> tuple[dict, int]:
    """A safetensors file's header (name -> dtype, shape, [begin, end) in the data) and where its data starts."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def hash_file(path: Path) -> tuple[str, dict]:
    """(the file's sha256, {tensor: its sha256}), reading the file once, a chunk at a time."""
    header, start = tensors_of(path)
    spans = sorted((start + t["data_offsets"][0], start + t["data_offsets"][1], name) for name, t in header.items())
    whole, per = hashlib.sha256(), {}
    open_ = [(a, b, name, hashlib.sha256()) for a, b, name in spans]
    at = 0
    with open(path, "rb") as f:
        while True:
            data = f.read(CHUNK)
            if not data:
                break
            whole.update(data)
            lo, hi = at, at + len(data)
            for a, b, name, h in open_:
                if a < hi and b > lo:
                    h.update(data[max(a, lo) - lo:min(b, hi) - lo])
            at = hi
    for a, b, name, h in open_:
        per[name] = h.hexdigest()
    return whole.hexdigest(), per


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", nargs="?", default=str(Path.home() / ".vibehash" / "cache" / "deepseek" / COMMIT))
    ap.add_argument("--out", default="vibehash/data/deepseek_flash_top.json")
    ap.add_argument("--layers", default="36,37,38,39")
    a = ap.parse_args()
    folder = Path(a.folder)
    layers = {int(x) for x in a.layers.split(",")}
    index = hub.Hub().weight_map(REPO, COMMIT)

    def wanted(name: str) -> bool:
        if name.startswith(("head.", "norm.")):
            return True
        return name.startswith("layers.") and int(name.split(".")[1]) in layers
    files = sorted({f for n, f in index.items() if wanted(n)})
    listing = hub.Hub().list_files(REPO, COMMIT)
    out_files, out_tensors = {}, {}
    for name in files:
        path = folder / name
        print(f"{name}: hashing {path.stat().st_size / 2**30:.1f} GB", flush=True)
        sha, per = hash_file(path)
        lfs = listing[name].lfs_sha256
        if sha != lfs:
            sys.exit(f"{name}'s sha256 {sha} isn't the one the Hub publishes at {COMMIT}: {lfs}")
        out_files[name] = {"size": path.stat().st_size, "sha256": sha}
        header, start = tensors_of(path)
        for t, info in header.items():
            b0, b1 = info["data_offsets"]
            out_tensors[t] = {"file": name, "dtype": info["dtype"], "shape": list(info["shape"]),
                              "offset": start + b0, "length": b1 - b0, "sha256": per[t]}
    m = hub.Manifest.from_dict({"format": hub.MANIFEST_FORMAT, "repo": REPO, "commit": COMMIT, "files": out_files,
                                "tensors": out_tensors})
    data = m.to_bytes()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes(data)
    print(f"{len(out_tensors)} tensors in {len(out_files)} files: {a.out}, sha256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()

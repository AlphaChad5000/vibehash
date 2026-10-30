"""vibehash.hub against a local fake of the Hugging Face Hub and of its CDN (huggingface.co isn't reachable from here).

The fake's routes, JSON shapes and headers follow huggingface_hub 2.4's source, cited at each one, so what passes
here is what that client expects from the real Hub. It runs two servers on two ports, like huggingface.co and the
CDN it redirects large files to, so the tests can see which bytes and which headers reach which host.
"""
import gzip
import hashlib
import json
import os
import random
import stat
import struct
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from vibehash import hub
from vibehash.hub import (AccessError, DiskFullError, Hub, HubError, IntegrityError, Manifest, TensorCache,
                          build_manifest, load_tensor, parse_header, verify_tensor)


def safetensors_bytes(tensors: dict, metadata: dict | None = None) -> bytes:
    """A safetensors file laid out as the safetensors library writes one: the tensors back to back after a JSON
    header padded with spaces to a multiple of 8 bytes."""
    header, data = {}, b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    if metadata:
        header["__metadata__"] = metadata
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    return struct.pack("<Q", len(h)) + h + data


def rand(n: int, seed) -> bytes:
    return random.Random(seed).randbytes(n)


def git_oid(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def is_lfs(path: str) -> bool:
    return path.endswith((".safetensors", ".bin"))


CONFIG = {"architectures": ["TinyForCausalLM"], "model_type": "tiny", "torch_dtype": "bfloat16", "num_hidden_layers": 2,
          "n_routed_experts": 8, "num_experts_per_tok": 2, "quantization_config": {"quant_method": "fp8", "fmt": "e4m3"}}


def shard1() -> dict:
    return {"model.embed_tokens.weight": ("BF16", [64, 32], rand(4096, 1)),
            "model.layers.0.self_attn.q_proj.weight": ("F8_E4M3", [32, 32], rand(1024, 2)),
            "model.layers.0.self_attn.q_proj.weight_scale_inv": ("F32", [1, 1], rand(4, 3)),
            "model.layers.0.empty": ("F32", [0], b""),
            "model.layers.0.mlp.down_proj.weight": ("BF16", [32, 64], rand(4096, 4))}


def shard2() -> dict:
    return {"model.layers.1.mlp.down_proj.weight": ("BF16", [128, 256], rand(65536, 5)),
            "model.layers.1.mlp.experts.packed": ("F4", [8, 16], rand(64, 6)),
            "model.norm.weight": ("BF16", [32], rand(64, 7))}


def sharded_files() -> dict:
    files = {"model-00001-of-00002.safetensors": safetensors_bytes(shard1(), {"format": "pt"}),
             "model-00002-of-00002.safetensors": safetensors_bytes(shard2())}
    weight_map = {n: f for f, t in (("model-00001-of-00002.safetensors", shard1()),
                                    ("model-00002-of-00002.safetensors", shard2())) for n in t}
    files[hub.INDEX_FILE] = json.dumps({"metadata": {"total_size": 79428}, "weight_map": weight_map}, indent=2).encode()
    files["config.json"] = json.dumps(CONFIG).encode()
    files["tokenizer.json"] = b'{"version": "1.0", "model": {"type": "BPE"}}'
    files["README.md"] = b"---\nlicense: mit\n---\n# tiny\n"
    return files


def single_files() -> dict:
    return {hub.SINGLE_FILE: safetensors_bytes({**shard1(), **shard2()}), "config.json": json.dumps(CONFIG).encode()}


class FakeHub:
    """huggingface.co and its CDN, in miniature. Repos are {commit: {path: bytes}} with branches and tags; files
    stored with LFS (weights) are served by the CDN from `self.lfs`, keyed by sha256, which tests may tamper with."""

    def __init__(self):
        self.repos: dict[str, dict] = {}
        self.lfs: dict[str, bytes] = {}
        self.lock = threading.Lock()
        self.hub_auth: list = []  # the Authorization header of every request each server saw (None if absent)
        self.cdn_auth: list = []
        self.cdn_bytes = 0  # body bytes the CDN has sent
        self.cdn_ranges: list = []  # (first, last) byte of every ranged request the CDN answered
        self.resolves = 0  # resolve requests for LFS files (each one a redirect)
        self.drop_after: int | None = None  # the CDN cuts its next response off after this many bytes
        self.fail_next = 0  # the Hub answers this many API requests with 503 first
        self.rate_limited = 0  # ...or with 429 and the Hub's RateLimit header
        self.gzip_text = False  # gzip API replies and small files even though the client asked for identity
        self.issued: list[str] = []  # signatures of the CDN links handed out
        self.revoked: set[str] = set()  # ...and the ones that have expired
        self.page_size = 1000
        self.hub = self._serve(self._hub_handler)
        self.cdn = self._serve(self._cdn_handler)
        self.endpoint = f"http://127.0.0.1:{self.hub.server_port}"
        self.cdn_url = f"http://127.0.0.1:{self.cdn.server_port}"

    def _serve(self, handle) -> ThreadingHTTPServer:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                handle(self)

            def log_message(self, *args):
                pass

            def reply(self, status: int, body: bytes = b"", headers: dict | None = None):
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, str(v))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def error(self, status: int, code: str, message: str, headers: dict | None = None):
                # utils/_http.hf_raise_for_status reads the reason from X-Error-Code / X-Error-Message
                self.reply(status, json.dumps({"error": message}).encode(),
                           {"X-Error-Code": code, "X-Error-Message": message, **(headers or {})})

        del fake
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, args=(0.02,), daemon=True).start()  # quick to shut down
        return server

    def close(self):
        for s in (self.hub, self.cdn):
            s.shutdown()
            s.server_close()

    # --- repos ------------------------------------------------------------------------

    def add_commit(self, repo: str, files: dict, *, branch: str = "main", gated: bool = False,
                   tokens: tuple = (), licence: str = "mit") -> str:
        r = self.repos.setdefault(repo, {"commits": {}, "refs": {}, "gated": gated, "tokens": set(tokens),
                                         "licence": licence})
        commit = hashlib.sha1(f"{repo}:{len(r['commits'])}".encode() + b"".join(sorted(files.values()))).hexdigest()
        r["commits"][commit] = dict(files)
        r["refs"][branch] = commit
        for path, data in files.items():
            if is_lfs(path):
                self.lfs[hashlib.sha256(data).hexdigest()] = data
        return commit

    def tag(self, repo: str, name: str, commit: str):
        self.repos[repo]["refs"][name] = commit

    def revoke_links(self):
        self.revoked.update(self.issued)

    def _commit(self, r: dict, rev: str) -> str | None:
        return r["refs"].get(rev) or (rev if rev in r["commits"] else None)

    def _tree(self, files: dict) -> list:
        """HfApi.list_repo_tree's entries, as RepoFile.__init__ reads them: path, size, oid and, for an LFS file,
        lfs = {oid: the content's sha256, size, pointerSize}, where the top-level oid is the pointer file's Git blob id."""
        out, dirs = [], set()
        for path in sorted(files):
            data = files[path]
            for i in range(1, path.count("/") + 1):
                dirs.add("/".join(path.split("/")[:i]))
            entry = {"type": "file", "oid": git_oid(data), "size": len(data), "path": path}
            if is_lfs(path):
                sha = hashlib.sha256(data).hexdigest()
                pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {len(data)}\n".encode()
                entry.update(oid=git_oid(pointer), lfs={"oid": sha, "size": len(data), "pointerSize": len(pointer)})
            out.append(entry)
        out += [{"type": "directory", "oid": git_oid(d.encode()), "size": 0, "path": d} for d in sorted(dirs)]
        return out

    def _info(self, repo: str, r: dict, commit: str, blobs: bool) -> dict:
        """HfApi.model_info's reply, the fields ModelInfo.__init__ reads: "sha" is the commit, "siblings" lists the
        files (with size, blobId and lfs {size, sha256, pointerSize} when ?blobs=true), "gated" is False or
        "auto"/"manual", "cardData" holds the card's metadata, "safetensors" {parameters, total} the Hub's count."""
        files = r["commits"][commit]
        params: dict[str, int] = {}
        for path, data in files.items():
            if path.endswith(".safetensors"):
                (n,) = struct.unpack("<Q", data[:8])
                for name, t in json.loads(data[8:8 + n]).items():
                    if name != "__metadata__":
                        count = 1
                        for d in t["shape"]:
                            count *= d
                        params[t["dtype"]] = params.get(t["dtype"], 0) + count
        siblings = []
        for e in self._tree(files):
            if e["type"] == "file":
                s = {"rfilename": e["path"]}
                if blobs:
                    s.update(size=e["size"], blobId=e["oid"])
                    if "lfs" in e:
                        s["lfs"] = {"size": e["lfs"]["size"], "sha256": e["lfs"]["oid"], "pointerSize": e["lfs"]["pointerSize"]}
                siblings.append(s)
        config = json.loads(files["config.json"]) if "config.json" in files else {}
        info = {"_id": hashlib.md5(repo.encode()).hexdigest()[:24], "id": repo, "modelId": repo,
                "author": repo.split("/")[0], "sha": commit, "lastModified": "2026-10-01T00:00:00.000Z",
                "private": False, "disabled": False, "gated": "manual" if r["gated"] else False,
                "library_name": "transformers", "tags": ["transformers", "safetensors", f"license:{r['licence']}"],
                "cardData": {"license": r["licence"]}, "siblings": siblings, "createdAt": "2026-01-01T00:00:00.000Z",
                "config": {k: config[k] for k in ("architectures", "model_type") if k in config}}
        if params:
            info["safetensors"] = {"parameters": params, "total": sum(params.values())}
        return info

    # --- the two servers --------------------------------------------------------------

    def _hub_handler(self, h):
        url = urllib.parse.urlsplit(h.path)
        query = urllib.parse.parse_qs(url.query)
        parts = [urllib.parse.unquote(p) for p in url.path.split("/") if p]
        auth = h.headers.get("Authorization")
        with self.lock:
            self.hub_auth.append(auth)
            if self.fail_next:
                self.fail_next -= 1
                return h.reply(503, b"busy", {"Retry-After": "0"})
            if self.rate_limited:  # utils/_http.parse_ratelimit_headers reads these two
                self.rate_limited -= 1
                return h.error(429, "", "Too many requests", {"RateLimit": '"api";r=0;t=0',
                                                              "RateLimit-Policy": '"fixed window";"api";q=500;w=300'})
        if parts[:2] == ["api", "models"]:
            # HfApi.model_info: GET /api/models/<repo> or /api/models/<repo>/revision/<rev>;
            # HfApi._iter_repo_tree_pages: GET /api/models/<repo>/tree/<rev>?recursive=...&expand=...
            repo, rest = "/".join(parts[2:4]), parts[4:]
            r = self.repos.get(repo)
            if r is None:  # the Hub answers 401 to an anonymous request for a missing (or private) repo
                return h.error(404 if auth else 401, "RepoNotFound", "Repository not found")
            rev = rest[1] if len(rest) > 1 else "main"
            commit = self._commit(r, rev)
            if commit is None:
                return h.error(404, "RevisionNotFound", f"Invalid rev id: {rev}")
            if not rest or rest[0] == "revision":
                info = self._info(repo, r, commit, query.get("blobs") == ["true"])
                return self._text(h, json.dumps(info).encode(), {"Content-Type": "application/json"})
            if rest[0] == "tree":
                entries = self._tree(r["commits"][commit])
                start = int(query.get("cursor", ["0"])[0])
                page, headers = entries[start:start + self.page_size], {"Content-Type": "application/json"}
                if start + self.page_size < len(entries):  # utils/_pagination._get_next_page reads Link rel="next"
                    nxt = f"{self.endpoint}{url.path}?recursive=true&expand=false&cursor={start + self.page_size}"
                    headers["Link"] = f'<{nxt}>; rel="next"'
                return self._text(h, json.dumps(page).encode(), headers)
            return h.error(404, "", "Not found")
        if len(parts) >= 5 and parts[2] == "resolve":
            # file_download.hf_hub_url: /<repo>/resolve/<revision>/<path>. get_hf_file_metadata reads the reply:
            # X-Repo-Commit, and for an LFS file a 302 to the CDN carrying X-Linked-Etag (its sha256) and X-Linked-Size.
            repo, rev, path = "/".join(parts[:2]), parts[3], "/".join(parts[4:])
            r = self.repos.get(repo)
            if r is None:
                return h.error(404 if auth else 401, "RepoNotFound", "Repository not found")
            if r["gated"] and auth is None:
                return h.error(401, "GatedRepo", f"Access to model {repo} is restricted. You must have access to it "
                                                 "and be authenticated to access it. Please log in.")
            if r["gated"] and auth.removeprefix("Bearer ") not in r["tokens"]:
                return h.error(403, "GatedRepo", f"Access to model {repo} is restricted and you are not in the "
                                                 f"authorized list. Visit {self.endpoint}/{repo} to ask for access.")
            commit = self._commit(r, rev)
            if commit is None:
                return h.error(404, "RevisionNotFound", f"Invalid rev id: {rev}")
            data = r["commits"][commit].get(path)
            if data is None:
                return h.error(404, "EntryNotFound", "Entry not found", {"X-Repo-Commit": commit})
            if is_lfs(path):
                sha = hashlib.sha256(data).hexdigest()
                with self.lock:
                    self.resolves += 1
                    sig = f"sig{len(self.issued)}"
                    self.issued.append(sig)
                return h.reply(302, b"", {"Location": f"{self.cdn_url}/repos/{sha}?Expires=9999999999&Signature={sig}",
                                          "X-Repo-Commit": commit, "X-Linked-Etag": f'"{sha}"',
                                          "X-Linked-Size": len(data), "ETag": f'"{git_oid(data)}"'})
            headers = {"X-Repo-Commit": commit, "ETag": f'"{git_oid(data)}"'}
            return self._ranged(h, data, headers) if h.headers.get("Range") else self._text(h, data, headers)
        return h.error(404, "", "Not found")

    def _text(self, h, body: bytes, headers: dict):
        if self.gzip_text:
            body, headers = gzip.compress(body), {**headers, "Content-Encoding": "gzip"}
        h.reply(200, body, headers)

    def _cdn_handler(self, h):
        url = urllib.parse.urlsplit(h.path)
        sig = urllib.parse.parse_qs(url.query).get("Signature", [""])[0]
        with self.lock:
            self.cdn_auth.append(h.headers.get("Authorization"))
        if sig not in self.issued or sig in self.revoked:
            return h.reply(403, b"<Error><Code>AccessDenied</Code><Message>Request has expired</Message></Error>")
        data = self.lfs.get(url.path.rsplit("/", 1)[-1])
        if data is None:
            return h.reply(404, b"no such object")
        self._ranged(h, data, {}, cdn=True)

    def _ranged(self, h, data: bytes, headers: dict, cdn: bool = False):
        """Byte ranges as the CDN serves them: 206 with Content-Range for `Range: bytes=a-b`, 416 past the end."""
        rng = h.headers.get("Range")
        status, body = 200, data
        if rng:
            first, _, last = rng.removeprefix("bytes=").partition("-")
            first, last = int(first), min(int(last), len(data) - 1)
            if first >= len(data):
                return h.reply(416, b"", {"Content-Range": f"bytes */{len(data)}", **headers})
            status, body = 206, data[first:last + 1]
            headers = {**headers, "Content-Range": f"bytes {first}-{last}/{len(data)}", "Accept-Ranges": "bytes"}
            if cdn:
                with self.lock:
                    self.cdn_ranges.append((first, last))
        h.send_response(status)
        for k, v in headers.items():
            h.send_header(k, str(v))
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        with self.lock:
            cut, self.drop_after = (self.drop_after, None) if cdn and self.drop_after is not None else (None, self.drop_after)
        if cut is not None:
            body = body[:cut]  # then the connection closes, short of the Content-Length promised
        if cdn:
            with self.lock:  # counted before sending, so the client can't see the bytes before the count does
                self.cdn_bytes += len(body)
        h.wfile.write(body)
        h.wfile.flush()


class HubTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        env = mock.patch.dict(os.environ, {"VIBEHASH_HOME": str(self.tmp / "home"), "HF_HOME": str(self.tmp / "hf")})
        env.start()
        self.addCleanup(env.stop)  # restores the whole environment, including what's popped below
        for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH", "HF_ENDPOINT", "XDG_CACHE_HOME"):
            os.environ.pop(k, None)
        self.fake = FakeHub()
        self.addCleanup(self.fake.close)
        self.hub = Hub(self.fake.endpoint, backoff=0.001)
        self.cache = TensorCache(self.tmp / "cache")

    def sharded(self, **kw) -> str:
        return self.fake.add_commit("acme/sharded", sharded_files(), **kw)

    def single(self) -> str:
        return self.fake.add_commit("acme/tiny", single_files())


class ResolveTests(HubTestCase):
    def test_branch_tag_and_the_ways_people_write_a_model(self):
        old = self.single()
        new = self.fake.add_commit("acme/tiny", {**single_files(), "README.md": b"v2"})
        self.fake.tag("acme/tiny", "v1", old)
        self.fake.repos["acme/tiny"]["refs"]["refs/pr/1"] = old
        self.assertEqual(self.hub.resolve("acme/tiny"), ("acme/tiny", new))  # no revision: the default branch
        for ref in ("acme/tiny@v1", "hf:acme/tiny@v1", f"acme/tiny@{old}", "acme/tiny@refs/pr/1",
                    "https://huggingface.co/acme/tiny/tree/v1", f"https://huggingface.co/acme/tiny/commit/{old}",
                    "https://huggingface.co/acme/tiny/tree/refs%2Fpr%2F1"):
            self.assertEqual(self.hub.resolve(ref), ("acme/tiny", old), ref)
        for ref in ("hf:acme/tiny", "https://huggingface.co/acme/tiny", "https://hf.co/acme/tiny/", "acme/tiny@main"):
            self.assertEqual(self.hub.resolve(ref), ("acme/tiny", new), ref)
        with self.assertRaisesRegex(HubError, "no branch, tag or commit 'nope'"):
            self.hub.resolve("acme/tiny@nope")

    def test_names_that_could_steer_a_url_are_refused(self):
        for ref in ("../etc/passwd", "acme/../x", "a--b/c", "acme/tiny/extra", "", "acme/tiny.git", "a b/c",
                    "https://evil.example/acme/tiny", "https://huggingface.co/datasets/acme/tiny"):
            with self.assertRaises(ValueError, msg=ref):
                hub.parse_ref(ref)
        with self.assertRaises(ValueError):
            self.hub.fetch_tensor("acme/tiny", "main", "x")  # a pinned fetch takes a commit, not a branch

    def test_hf_endpoint_and_retrying_a_busy_hub(self):
        commit = self.single()
        with mock.patch.dict(os.environ, {"HF_ENDPOINT": self.fake.endpoint + "/"}):
            h = Hub(backoff=0.001)
        self.assertEqual(h.endpoint, self.fake.endpoint)
        self.fake.fail_next = 2  # 503 twice, then fine
        self.assertEqual(h.resolve("acme/tiny")[1], commit)
        self.fake.rate_limited = 1  # waits as long as the RateLimit header says (t=0, plus a second)
        with self.assertLogs("vibehash.hub", "WARNING") as logs:
            self.assertEqual(h.resolve("acme/tiny")[1], commit)
        self.assertRegex(logs.output[0], r"HTTP 429.*retrying in 1\.\d s")

    def test_gzipped_text_is_unpacked(self):
        commit = self.sharded()
        self.fake.gzip_text = True
        self.assertEqual(self.hub.resolve("acme/sharded")[1], commit)
        self.assertEqual(self.hub.describe("acme/sharded").config, CONFIG)  # checked against its Git blob id
        self.assertEqual(self.hub.fetch_tensor("acme/sharded", commit, "model.norm.weight")[2],
                         shard2()["model.norm.weight"][2])

    def test_unreachable_hub_says_so(self):
        h = Hub("http://127.0.0.1:1", retries=1, backoff=0.001)
        with self.assertRaisesRegex(HubError, "can't reach Hugging Face at http://127.0.0.1:1.*internet connection"):
            h.resolve("acme/tiny")

    def test_listing_follows_pages(self):
        commit = self.sharded()
        self.fake.page_size = 2
        files = self.hub.list_files("acme/sharded", commit)
        self.assertEqual(set(files), set(sharded_files()))
        for path, data in sharded_files().items():
            self.assertEqual(files[path].size, len(data))
            self.assertEqual(files[path].lfs_sha256, hashlib.sha256(data).hexdigest() if is_lfs(path) else None)


class FetchTests(HubTestCase):
    def test_single_file_model_reads_only_header_and_tensor(self):
        commit = self.single()
        data = single_files()[hub.SINGLE_FILE]
        (n,) = struct.unpack("<Q", data[:8])
        dtype, shape, raw = self.hub.fetch_tensor("acme/tiny", commit, "model.layers.0.self_attn.q_proj.weight")
        self.assertEqual((dtype, shape, raw), ("F8_E4M3", [32, 32], shard1()["model.layers.0.self_attn.q_proj.weight"][2]))
        self.assertEqual(self.fake.cdn_bytes, 8 + n + 1024)  # the length, the header, the tensor: nothing else
        self.assertEqual(self.fake.cdn_ranges[:2], [(0, 7), (8, 8 + n - 1)])
        # A second tensor of the same file: the header and the CDN address are reused, so it's just the tensor.
        before, resolves = self.fake.cdn_bytes, self.fake.resolves
        dtype, shape, raw = self.hub.fetch_tensor("acme/tiny", commit, "model.layers.1.mlp.experts.packed")
        self.assertEqual((dtype, shape, raw), ("F4", [8, 16], shard2()["model.layers.1.mlp.experts.packed"][2]))
        self.assertEqual((self.fake.cdn_bytes - before, self.fake.resolves), (64, resolves))
        self.assertEqual(self.hub.fetch_tensor("acme/tiny", commit, "model.layers.0.empty"), ("F32", [0], b""))

    def test_sharded_model_finds_the_shard_through_the_index(self):
        commit = self.sharded()
        shard = sharded_files()["model-00002-of-00002.safetensors"]
        (n,) = struct.unpack("<Q", shard[:8])
        dtype, shape, raw = self.hub.fetch_tensor("acme/sharded", commit, "model.norm.weight")
        self.assertEqual((dtype, shape, raw), shard2()["model.norm.weight"][:2] + (shard2()["model.norm.weight"][2],))
        self.assertEqual(self.fake.cdn_bytes, 8 + n + 64)  # shard 1 untouched; the index came from the Hub itself
        with self.assertRaises(KeyError):
            self.hub.fetch_tensor("acme/sharded", commit, "no.such.tensor")

    def test_expired_cdn_link_is_resolved_again(self):
        commit = self.sharded()
        f = self.hub.open("acme/sharded", commit, "model-00002-of-00002.safetensors")
        self.assertEqual(len(f.read(0, 8)), 8)
        self.fake.revoke_links()
        self.assertEqual(f.read(8, 8), sharded_files()["model-00002-of-00002.safetensors"][8:16])
        self.assertEqual(self.fake.resolves, 2)

    def test_the_hubs_file_headers_and_redirects_are_checked(self):
        commit = self.sharded()
        shard = "model-00002-of-00002.safetensors"
        with self.assertRaisesRegex(IntegrityError, "isn't the one pinned"):  # X-Linked-Etag
            Hub(self.fake.endpoint).open("acme/sharded", commit, shard, sha256="0" * 64).read(0, 8)
        with self.assertRaisesRegex(IntegrityError, "the Hub says .* bytes, not 10"):  # X-Linked-Size
            Hub(self.fake.endpoint).open("acme/sharded", commit, shard, size=10).read(0, 8)
        with self.assertRaisesRegex(IntegrityError, "outside"):
            self.hub.open("acme/sharded", commit, shard, size=10).read(4, 8)
        for location in ("http://cdn.example/x", "file:///etc/passwd", "ftp://cdn.example/x"):
            with self.assertRaisesRegex(HubError, "refusing a redirect"):  # no downgrade from https, no odd schemes
                hub._redirect_target("https://huggingface.co/acme/tiny/resolve/main/x", location)

    def test_malformed_headers_are_refused(self):
        t = {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
        bad = {
            "not json": b"not json",
            "not an object": b"[1, 2]",
            "duplicate key": b'{"a": %s, "a": %s}' % (json.dumps(t).encode(), json.dumps(t).encode()),
            "unknown dtype": json.dumps({"a": {**t, "dtype": "F7"}}).encode(),
            "length vs shape": json.dumps({"a": {**t, "shape": [3]}}).encode(),
            "overlap": json.dumps({"a": t, "b": {**t, "data_offsets": [4, 12]}}).encode(),
            "gap": json.dumps({"a": t, "b": {**t, "data_offsets": [12, 20]}}).encode(),
            "extra field": json.dumps({"a": {**t, "extra": 1}}).encode(),
            "bool offset": json.dumps({"a": {**t, "data_offsets": [False, 8]}}).encode(),
            "negative dim": json.dumps({"a": {**t, "shape": [-2]}}).encode(),
            "odd fp4": json.dumps({"a": {"dtype": "F4", "shape": [3], "data_offsets": [0, 1]}}).encode(),
            "bad metadata": json.dumps({"__metadata__": {"k": 1}}).encode(),
        }
        for why, raw in bad.items():
            with self.assertRaises(IntegrityError, msg=why):
                parse_header(raw)
        ok = json.dumps({"a": t}).encode()
        self.assertEqual(parse_header(ok, 8 + len(ok) + 8).tensors, {"a": ("F32", (2,), 0, 8)})
        with self.assertRaisesRegex(IntegrityError, "file is"):
            parse_header(ok, 8 + len(ok) + 9)  # the file has bytes no tensor accounts for
        with self.assertRaises(IntegrityError):
            hub.check_header_size(hub.MAX_HEADER + 1, None)
        with self.assertRaises(IntegrityError):
            hub.check_header_size(100, 50)

    def test_hostile_header_length_is_refused_before_it_is_fetched(self):
        for n in (2**40, 4096):  # far past the cap; past the end of this small file
            commit = self.fake.add_commit("acme/hostile", {hub.SINGLE_FILE: struct.pack("<Q", n) + b"{}" + b" " * 30})
            before = self.fake.cdn_bytes
            with self.assertRaises(IntegrityError):
                Hub(self.fake.endpoint).fetch_tensor("acme/hostile", commit, "x")
            self.assertEqual(self.fake.cdn_bytes - before, 8)

    def test_pickle_only_model_is_explained(self):
        commit = self.fake.add_commit("acme/old", {"pytorch_model.bin": b"\x80\x04pickle", "config.json": b"{}"})
        with self.assertRaisesRegex(HubError, "only has PyTorch pickle weights.*never does"):
            self.hub.fetch_tensor("acme/old", commit, "x")
        with self.assertRaisesRegex(HubError, "pickle"):
            build_manifest("acme/old", commit, hub=self.hub)
        self.assertTrue(self.hub.describe("acme/old").pickle_only)


class TokenTests(HubTestCase):
    def test_token_reaches_the_hub_but_never_the_cdn(self):
        commit = self.sharded()
        h = Hub(self.fake.endpoint, token="hf_s3cret", backoff=0.001)
        h.resolve("acme/sharded")
        m = build_manifest("acme/sharded", commit, hub=h)
        load_tensor(m, "model.norm.weight", hub=h, cache=self.cache)
        h.fetch_tensor("acme/sharded", commit, "model.embed_tokens.weight")
        self.assertTrue(self.fake.hub_auth and all(a == "Bearer hf_s3cret" for a in self.fake.hub_auth))
        self.assertTrue(self.fake.cdn_auth)
        self.assertEqual(set(self.fake.cdn_auth), {None})
        self.assertNotIn("hf_s3cret", repr(h))

    def test_where_the_token_comes_from(self):
        self.single()
        self.assertFalse(Hub(self.fake.endpoint).has_token)
        hf_cli = self.tmp / "hf" / "token"  # where the official `hf auth login` keeps one
        hf_cli.parent.mkdir(parents=True)
        hf_cli.write_text("hf_cli\n")
        Hub(self.fake.endpoint).model_info("acme/tiny")
        path = hub.save_token(" hf_saved\n")  # `vibehash hub login`, which beats it
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        Hub(self.fake.endpoint).model_info("acme/tiny")
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_env"}):  # and $HF_TOKEN beats both
            Hub(self.fake.endpoint).model_info("acme/tiny")
        Hub(self.fake.endpoint, token="hf_arg").model_info("acme/tiny")
        self.assertEqual(self.fake.hub_auth, ["Bearer hf_cli", "Bearer hf_saved", "Bearer hf_env", "Bearer hf_arg"])

    def test_gated_repo_says_what_to_do(self):
        commit = self.fake.add_commit("acme/gated", sharded_files(), gated=True, tokens=("hf_ok",))
        page = f"{self.fake.endpoint}/acme/gated"
        with self.assertRaises(AccessError) as cm:
            self.hub.fetch_tensor("acme/gated", commit, "model.norm.weight")
        msg = str(cm.exception)
        for words in ("accept its licence", page, "HF_TOKEN", "vibehash hub login"):
            self.assertIn(words, msg)
        with self.assertRaises(AccessError) as cm:
            Hub(self.fake.endpoint, token="hf_nope").fetch_tensor("acme/gated", commit, "model.norm.weight")
        self.assertIn("doesn't have access", str(cm.exception))
        self.assertIn(page, str(cm.exception))
        self.assertNotIn("hf_nope", str(cm.exception))
        raw = Hub(self.fake.endpoint, token="hf_ok").fetch_tensor("acme/gated", commit, "model.norm.weight")[2]
        self.assertEqual(raw, shard2()["model.norm.weight"][2])
        # The API still describes a gated model to anyone, so describe() works before the licence is accepted.
        s = self.hub.describe("acme/gated")
        self.assertTrue(s.gated)
        self.assertEqual(s.architecture, "TinyForCausalLM")
        self.assertIn("accept its licence", str(s))

    def test_missing_repo_says_what_to_check(self):
        with self.assertRaisesRegex(AccessError, "no model acme/missing .*check the name.*HF_TOKEN"):
            self.hub.resolve("acme/missing")
        with self.assertRaisesRegex(AccessError, "no model acme/missing that your token can see"):
            Hub(self.fake.endpoint, token="hf_x").resolve("acme/missing")


class ManifestTests(HubTestCase):
    def assert_manifest_matches(self, m: Manifest, files: dict):
        for name, e in m.tensors.items():
            self.assertEqual(files[e.file][e.offset:e.offset + e.length], {**shard1(), **shard2()}[name][2], name)
            self.assertEqual(hashlib.sha256(files[e.file][e.offset:e.offset + e.length]).hexdigest(), e.sha256)
        self.assertEqual(set(m.tensors), set(shard1()) | set(shard2()))

    def test_build_remote_and_local_and_round_trip(self):
        commit = self.sharded()
        files = sharded_files()
        seen = []
        m = build_manifest("acme/sharded", commit, hub=self.hub, progress=lambda *a: seen.append(a))
        self.assert_manifest_matches(m, files)
        weights = [p for p in files if p.endswith(".safetensors")]
        self.assertEqual(self.fake.cdn_bytes, sum(len(files[p]) for p in weights))  # each file streamed once
        self.assertEqual(seen[-1], (self.fake.cdn_bytes, self.fake.cdn_bytes, "checking acme/sharded"))
        self.assertEqual(set(m.files), set(weights) | {hub.INDEX_FILE, "config.json", "tokenizer.json"})
        for p in m.files:  # a weight file's sha256 is the LFS sha256 the Hub publishes
            self.assertEqual(m.files[p], {"size": len(files[p]), "sha256": hashlib.sha256(files[p]).hexdigest()})
        self.assertEqual(m.tensors["model.norm.weight"].dtype, "BF16")
        self.assertEqual(m.tensors["model.layers.1.mlp.experts.packed"].shape, (8, 16))

        data = m.to_bytes()
        self.assertEqual(Manifest.from_bytes(data, sha256=m.sha256), m)
        self.assertEqual(Manifest.from_bytes(data).to_bytes(), data)
        with self.assertRaisesRegex(IntegrityError, "isn't the one the job pins"):
            Manifest.from_bytes(data, sha256="0" * 64)
        with self.assertRaisesRegex(IntegrityError, "canonical"):
            Manifest.from_bytes(json.dumps(m.to_dict(), indent=1).encode())

        local = self.tmp / "snapshot"
        for p, b in files.items():
            (local / p).parent.mkdir(parents=True, exist_ok=True)
            (local / p).write_bytes(b)
        before = self.fake.cdn_bytes
        self.assertEqual(build_manifest("acme/sharded", commit, hub=self.hub, local_dir=local).to_bytes(), data)
        self.assertEqual(self.fake.cdn_bytes, before)  # only the listing came from the Hub
        self.assertEqual(hub.fetch_file(m, "config.json", hub=self.hub), files["config.json"])

    def test_single_file_model(self):
        commit = self.single()
        m = build_manifest("acme/tiny", commit, hub=self.hub)
        self.assert_manifest_matches(m, single_files())
        self.assertEqual(set(m.files), {hub.SINGLE_FILE, "config.json"})

    def test_a_malformed_manifest_is_refused(self):
        commit = self.single()
        d = build_manifest("acme/tiny", commit, hub=self.hub).to_dict()
        name = "model.norm.weight"
        for bad in ({"offset": 10**12}, {"length": 3}, {"sha256": "nothex"}, {"file": "other.safetensors"},
                    {"dtype": "F7"}, {"shape": [3]}, {"offset": -1}):
            broken = json.loads(json.dumps(d))
            broken["tensors"][name].update(bad)
            with self.assertRaises(IntegrityError, msg=str(bad)):
                Manifest.from_dict(broken)

    def test_content_not_matching_its_lfs_sha256_fails_the_build(self):
        commit = self.sharded()
        files = sharded_files()
        shard = "model-00002-of-00002.safetensors"
        sha = hashlib.sha256(files[shard]).hexdigest()
        tampered = bytearray(files[shard])
        tampered[-100] ^= 1
        self.fake.lfs[sha] = bytes(tampered)
        with self.assertRaisesRegex(IntegrityError, f"{shard} doesn't match the sha256 the Hub publishes"):
            build_manifest("acme/sharded", commit, hub=self.hub)
        local = self.tmp / "snapshot"
        local.mkdir()
        for p, b in files.items():
            (local / p).write_bytes(bytes(tampered) if p == shard else b)
        with self.assertRaisesRegex(IntegrityError, "doesn't match the sha256"):
            build_manifest("acme/sharded", commit, hub=self.hub, local_dir=local)

    def test_tampered_cdn_bytes_are_caught(self):
        commit = self.sharded()
        m = build_manifest("acme/sharded", commit, hub=self.hub)
        name, other = "model.layers.1.mlp.down_proj.weight", "model.norm.weight"
        e = m.tensors[name]
        sha = m.files[e.file]["sha256"]
        tampered = bytearray(self.fake.lfs[sha])
        tampered[e.offset + 3] ^= 0x40
        self.fake.lfs[sha] = bytes(tampered)
        h = Hub(self.fake.endpoint, backoff=0.001)
        raw = h.fetch_tensor("acme/sharded", commit, name)[2]
        with self.assertRaisesRegex(IntegrityError, "aren't the ones the manifest pins"):
            verify_tensor(m, name, raw)
        verify_tensor(m, other, h.fetch_tensor("acme/sharded", commit, other)[2])  # the rest of the file is fine
        with self.assertRaisesRegex(IntegrityError, "wrong bytes"):
            load_tensor(m, name, hub=h, cache=self.cache)
        self.assertEqual(list((self.cache.root / commit).iterdir()), [])  # nothing kept, not even a .part


class CacheTests(HubTestCase):
    def setUp(self):
        super().setUp()
        self.commit = self.sharded()
        self.m = build_manifest("acme/sharded", self.commit, hub=self.hub)
        self.fake.cdn_bytes, self.fake.cdn_ranges = 0, []

    def test_load_reads_only_the_tensor_and_caches_it(self):
        name = "model.layers.1.mlp.down_proj.weight"
        e = self.m.tensors[name]
        self.assertEqual(load_tensor(self.m, name, hub=self.hub, cache=self.cache),
                         ("BF16", [128, 256], shard2()[name][2]))
        self.assertEqual(self.fake.cdn_ranges, [(e.offset, e.offset + e.length - 1)])  # no header read needed
        self.assertTrue((self.cache.root / self.commit / e.sha256).is_file())
        self.assertEqual(load_tensor(self.m, name, hub=self.hub, cache=TensorCache(self.cache.root))[2], shard2()[name][2])
        self.assertEqual(self.fake.cdn_bytes, e.length)  # the second time, from disk
        default = hub.default_cache_dir()
        self.assertEqual(default, self.tmp / "home" / "cache" / "hub")  # under $VIBEHASH_HOME
        load_tensor(self.m, "model.norm.weight", hub=self.hub)
        self.assertTrue((default / self.commit / self.m.tensors["model.norm.weight"].sha256).is_file())

    def test_a_damaged_cache_file_is_fetched_again(self):
        name = "model.norm.weight"
        path = hub.fetch_tensors(self.m, [name], hub=self.hub, cache=self.cache)[name]
        path.write_bytes(b"\0" * self.m.tensors[name].length)
        with self.assertLogs("vibehash.hub", "WARNING"):
            self.assertEqual(load_tensor(self.m, name, hub=self.hub, cache=TensorCache(self.cache.root))[2],
                             shard2()[name][2])
        self.assertEqual(self.fake.cdn_bytes, 2 * self.m.tensors[name].length)

    def test_back_to_back_tensors_come_in_one_request_with_progress(self):
        names = list(shard1())
        seen = []
        paths = hub.fetch_tensors(self.m, names, hub=self.hub, cache=self.cache, label="layers 0-0",
                                  progress=lambda *a: seen.append(a))
        total = self.m.bytes_for(names)
        self.assertEqual(len(self.fake.cdn_ranges), 1)
        self.assertEqual(self.fake.cdn_bytes, total)
        self.assertEqual((seen[0], seen[-1]), ((0, total, "layers 0-0"), (total, total, "layers 0-0")))
        self.assertEqual([d for d, *_ in seen], sorted(d for d, *_ in seen))
        for name in names:
            self.assertEqual(paths[name].read_bytes(), shard1()[name][2])
        seen.clear()
        hub.fetch_tensors(self.m, names, hub=self.hub, cache=self.cache, progress=lambda *a: seen.append(a))
        self.assertEqual((len(self.fake.cdn_ranges), seen), (1, []))  # all cached: no request, nothing to report

    def test_resume_after_a_dropped_connection(self):
        name = "model.layers.1.mlp.down_proj.weight"
        e = self.m.tensors[name]
        self.fake.drop_after = 1000  # within one call: the read picks up at byte 1000
        self.assertEqual(load_tensor(self.m, name, hub=self.hub, cache=self.cache)[2], shard2()[name][2])
        self.assertEqual(self.fake.cdn_bytes, e.length)
        self.assertEqual(self.fake.cdn_ranges, [(e.offset, e.offset + e.length - 1), (e.offset + 1000, e.offset + e.length - 1)])

        # Across calls (a restart, say): the .part left behind is continued, not thrown away.
        cache = TensorCache(self.tmp / "cache2")
        self.fake.drop_after, self.fake.cdn_bytes = 1000, 0
        with self.assertRaisesRegex(HubError, "cut off"):
            load_tensor(self.m, name, hub=Hub(self.fake.endpoint, retries=0), cache=cache)
        part = cache.root / self.commit / f"{e.sha256}.part"
        self.assertEqual(part.stat().st_size, 1000)
        self.assertEqual(load_tensor(self.m, name, hub=self.hub, cache=cache)[2], shard2()[name][2])
        self.assertEqual(self.fake.cdn_bytes, e.length)
        self.assertFalse(part.exists())

    def test_disk_space_is_checked_first(self):
        names = list(shard2())
        with mock.patch("vibehash.hub.shutil.disk_usage", return_value=mock.Mock(free=1000)):
            with self.assertRaises(DiskFullError) as cm:
                hub.fetch_tensors(self.m, names, hub=self.hub, cache=self.cache)
        self.assertIn(str(self.cache.root / self.commit), str(cm.exception))
        self.assertIn(f"{self.m.bytes_for(names) / 1e6:.1f} MB", str(cm.exception))
        self.assertEqual(self.fake.cdn_bytes, 0)

    def test_usage_and_prune(self):
        other = self.single()
        m2 = build_manifest("acme/tiny", other, hub=self.hub)
        hub.fetch_tensors(self.m, list(shard2()), hub=self.hub, cache=self.cache)
        hub.fetch_tensors(m2, list(shard1()), hub=self.hub, cache=self.cache)
        usage = hub.cache_usage(self.cache.root)
        self.assertEqual(usage, {self.commit: self.m.bytes_for(shard2()), other: m2.bytes_for(shard1())})
        os.utime(self.cache.root / other, (1, 1))  # not used for the longest time
        self.assertEqual(hub.prune_cache(root=self.cache.root, max_bytes=sum(usage.values())), 0)  # under the cap
        self.assertEqual(hub.prune_cache(root=self.cache.root, max_bytes=sum(usage.values()) - 1), usage[other])
        self.assertEqual(set(hub.cache_usage(self.cache.root)), {self.commit})
        self.assertEqual(hub.prune_cache([self.commit], self.cache.root), 0)  # kept
        self.assertEqual(hub.prune_cache([], self.cache.root), usage[self.commit])
        self.assertEqual(hub.cache_usage(self.cache.root), {})


class DescribeTests(HubTestCase):
    def test_preflight_summary(self):
        commit = self.sharded()
        s = self.hub.describe("https://huggingface.co/acme/sharded")
        files = sharded_files()
        self.assertEqual((s.repo, s.commit, s.revision), ("acme/sharded", commit, "the default branch"))
        self.assertEqual((s.architecture, s.dtype, s.licence, s.gated), ("TinyForCausalLM", "bfloat16, fp8 (e4m3)", "mit", False))
        self.assertEqual(s.parameters, 2048 + 1024 + 1 + 0 + 2048 + 32768 + 128 + 32)
        self.assertEqual((s.weight_files, s.weight_bytes), (2, sum(len(b) for p, b in files.items() if p.endswith(".safetensors"))))
        self.assertEqual((s.layers, s.experts, s.experts_per_token, s.config), (2, 8, 2, CONFIG))
        self.assertTrue(s.has_tokenizer)
        self.assertFalse(s.pickle_only)
        for words in (commit, "2 safetensors files", "mixture of 8 experts (2 per token), 2 layers", "licence: mit"):
            self.assertIn(words, str(s))


if __name__ == "__main__":
    unittest.main()

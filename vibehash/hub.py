"""Pretrained weights straight from the Hugging Face Hub: each worker fetches only the tensors it needs, verified.

A job names its base model as `hf:<org>/<name>@<commit>`. The Hub keeps weights
as safetensors files (an 8-byte length, a JSON header, then the tensors back to
back) and serves them by byte range, so a worker holding layers 10-19 of a 552B
model downloads those layers' tensors and nothing else: no whole 5 GB shard,
and nothing passing through the node.

What makes that trustworthy is the manifest. The Hub publishes one sha256 per
*file* (its Git LFS object id), which can't check a slice of a file. So someone
streams every weight file once (build_manifest), checking each against the
Hub's sha256 and hashing every tensor on the way. The manifest lists each
tensor's file, byte range, dtype, shape and sha256; a job pins the manifest's
own sha256, and workers check every tensor they download against it
(verify_tensor), so a CDN, mirror or proxy that alters a byte is caught.

The HTTP side does what the official client (huggingface_hub 2.x) does, read
from its source and cited where it matters: the same endpoints and JSON shapes,
$HF_ENDPOINT, $HF_TOKEN, `resolve/<commit>/<path>` redirecting large files to a
CDN, and the token never following that redirect. Standard library only, so a
worker needs no extra packages. Tensors land in
~/.vibehash/cache/hub/<commit>/<tensor sha256>, a dropped connection or a
restart resumes where it stopped, and only bytes that hashed right are kept.

Turning raw bytes into arrays (bf16, fp8, fp4 ...) is safetensors.py's job:
this module hands over the dtype string and the bytes exactly as the file has them.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import logging
import math
import os
import random
import re
import shutil
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

from . import settings

log = logging.getLogger("vibehash.hub")

DEFAULT_ENDPOINT = "https://huggingface.co"  # huggingface_hub constants._HF_DEFAULT_ENDPOINT; $HF_ENDPOINT overrides it
HUB_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}  # whose model-page addresses parse_ref accepts
INDEX_FILE = "model.safetensors.index.json"  # constants.SAFETENSORS_INDEX_FILE: which shard holds each tensor
SINGLE_FILE = "model.safetensors"  # constants.SAFETENSORS_SINGLE_FILE: an unsharded model
MAX_HEADER = 25_000_000  # constants.SAFETENSORS_MAX_HEADER_LENGTH: a longer header is refused before it's fetched
MAX_JSON = 64 * 2**20  # an index, config or API reply bigger than this is refused (DeepSeek V3's index is ~9 MB)
MAX_PAGES = 1000  # of a file listing (1000 entries each)
CHUNK = 8 * 2**20
RETRIES = 5
MAX_REDIRECTS = 10
MANIFEST_FORMAT = "vibehash-hf-manifest/1"
# Small files a manifest pins besides the weights, when the repo has them: the architecture and the tokenizer have
# to be the same bytes on every worker too.
EXTRA_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json")
PICKLES = (".bin", ".pt", ".pth", ".ckpt", ".pkl")  # PyTorch pickles: loading one can run code, so we never do
USER_AGENT = "vibehash-hub/1 (python-urllib)"

# Bits per element of every safetensors dtype (the safetensors crate's Dtype::bitsize). Sub-byte types (F4, F6) pack
# elements, so a tensor of them must fill whole bytes. A dtype not listed here is refused: its length can't be checked.
DTYPE_BITS = {"BOOL": 8, "U8": 8, "I8": 8, "F8_E5M2": 8, "F8_E4M3": 8, "F8_E8M0": 8, "F4": 4, "F6_E2M3": 6,
              "F6_E3M2": 6, "I16": 16, "U16": 16, "F16": 16, "BF16": 16, "I32": 32, "U32": 32, "F32": 32,
              "C64": 64, "I64": 64, "U64": 64, "F64": 64}

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
_REPO_PART = re.compile(r"^\w([\w.-]{0,94}\w)?$", re.ASCII)  # huggingface_hub utils/_validators.REPO_ID_REGEX
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")
_REDIRECTS = (301, 302, 303, 307, 308)


class HubError(Exception):
    """Talking to the Hub (or its CDN) went wrong. `status` is the HTTP status, or None if it couldn't be reached."""

    def __init__(self, msg: str, status: int | None = None, *, retryable: bool | None = None,
                 retry_after: str | None = None):
        super().__init__(msg)
        self.status = status
        self.retry_after = retry_after
        self._retryable = retryable

    @property
    def retryable(self) -> bool:
        """Connection trouble, a server fault, or being asked to slow down (429), as client.NodeError has it."""
        if self._retryable is not None:
            return self._retryable
        return self.status is None or self.status >= 500 or self.status == 429


class AccessError(HubError):
    """The model is gated, private or doesn't exist. The message says what to do about it."""

    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg, status, retryable=False)


class DiskFullError(HubError):
    def __init__(self, msg: str):
        super().__init__(msg, retryable=False)


class IntegrityError(ValueError):
    """Bytes, a header or a manifest that don't check out. Never retried: whoever served them is wrong."""


# --- names ------------------------------------------------------------------------------

def check_repo(repo: str) -> str:
    """`org/name` (or a bare legacy `name`), as huggingface_hub's validate_repo_id allows: ASCII letters, digits,
    `_ . -`, at most 96 per part, no `--` or `..`. Anything else could steer a URL or a cache path somewhere else."""
    parts = repo.split("/") if isinstance(repo, str) else []
    if not 1 <= len(parts) <= 2 or not all(_REPO_PART.match(p) for p in parts) or "--" in repo or ".." in repo \
            or repo.endswith(".git"):
        raise ValueError(f"{repo!r} isn't a Hugging Face model name (it looks like organisation/model)")
    return repo


def check_commit(commit: str) -> str:
    """A pinned commit is the full 40-hex sha: a branch or tag can move, a commit can't."""
    if not isinstance(commit, str) or not _COMMIT.match(commit):
        raise ValueError(f"{commit!r} isn't a commit sha (40 hex digits); resolve the branch first (Hub.resolve)")
    return commit


def _check_path(path: str) -> str:
    """A file's path inside a repo, as the Hub or an index names it. Refuses anything that could escape a folder."""
    if not isinstance(path, str) or not 0 < len(path) <= 1024 or path.startswith("/") or "\\" in path \
            or "\0" in path or any(p in ("", ".", "..") for p in path.split("/")):
        raise IntegrityError(f"{path!r} isn't a file path inside a repo")
    return path


def _endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")


def parse_ref(ref: str, endpoint: str | None = None) -> tuple[str, str | None]:
    """A model as people write it, as (repo, revision, or None for the default branch). Accepts `org/name`,
    `org/name@rev`, `hf:org/name@rev`, and the model page's address, `https://huggingface.co/org/name`, with
    `/tree/<branch>` or `/commit/<sha>` when copied from a branch's or a commit's page."""
    text, rev = ref.strip(), None
    if text.lower().startswith(("https://", "http://")):
        url = urllib.parse.urlsplit(text)
        if (url.hostname or "").lower() not in HUB_HOSTS | {urllib.parse.urlsplit(endpoint or _endpoint()).hostname}:
            raise ValueError(f"{ref} isn't a Hugging Face address")
        parts = [urllib.parse.unquote(p) for p in url.path.split("/") if p]
        if parts[:1] in (["datasets"], ["spaces"]):
            raise ValueError(f"{ref} is a {parts[0][:-1]}, not a model")
        parts = parts[1:] if parts[:1] == ["models"] else parts
        cut = 1 if len(parts) > 1 and parts[1] in ("tree", "blob", "resolve", "commit") else 2
        repo, rest = "/".join(parts[:cut]), parts[cut:]
        if len(rest) > 1 and rest[0] in ("tree", "blob", "resolve", "commit"):
            rev = rest[1]
    else:
        text = text[3:].lstrip("/") if text.lower().startswith("hf:") else text
        repo, at, rev = text.partition("@")
        rev = rev if at else None
    if rev is not None and (not rev or len(rev) > 256 or any(c.isspace() or ord(c) < 32 for c in rev)):
        raise ValueError(f"{rev!r} isn't a branch, tag or commit")
    return check_repo(repo), rev


# --- tokens -----------------------------------------------------------------------------

def token_path() -> Path:
    """Where `vibehash hub login` keeps a Hugging Face token (save_token)."""
    return settings.home() / "hf_token"


def _hf_cli_token_path() -> Path:
    """Where the official `hf auth login` keeps one: huggingface_hub constants.HF_TOKEN_PATH."""
    if os.environ.get("HF_TOKEN_PATH"):
        return Path(os.environ["HF_TOKEN_PATH"]).expanduser()
    hf_home = os.environ.get("HF_HOME") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface"
    return Path(hf_home).expanduser() / "token"


def _clean(token: str | None) -> str | None:
    """huggingface_hub utils/_auth._clean_token: no stray whitespace or newlines; empty means none."""
    return (token or "").replace("\r", "").replace("\n", "").strip() or None


def _read_token(path: Path) -> str | None:
    try:
        return _clean(path.read_text())
    except (OSError, UnicodeDecodeError):
        return None


def find_token(token: str | None = None) -> str | None:
    """The token to send: the one given, else $HF_TOKEN (or the older $HUGGING_FACE_HUB_TOKEN, as huggingface_hub's
    _get_token_from_environment reads), else the one `vibehash hub login` saved, else the official CLI's."""
    return (_clean(token) or _clean(os.environ.get("HF_TOKEN")) or _clean(os.environ.get("HUGGING_FACE_HUB_TOKEN"))
            or _read_token(token_path()) or _read_token(_hf_cli_token_path()))


def save_token(token: str) -> Path:
    """Keep a token for later (what `vibehash hub login` calls), readable only by you."""
    token = _clean(token)
    if not token:
        raise ValueError("that's an empty token")
    path = token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    os.chmod(path, 0o600)  # in case it already existed with looser permissions
    return path


# --- HTTP plumbing ----------------------------------------------------------------------

def _origin(url: str) -> tuple[str, str, int | None]:
    """(scheme, host, port), the unit browsers and huggingface_hub (utils/_http._url_origin) trust by."""
    u = urllib.parse.urlsplit(url)
    scheme = u.scheme.lower()
    return scheme, (u.hostname or "").lower(), u.port or {"http": 80, "https": 443}.get(scheme)


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc or url


def _bare(url: str) -> str:
    """A URL fit for an error message: signed CDN addresses carry long signatures in the query string."""
    return url.split("?", 1)[0]


def _redirect_target(current: str, location: str) -> str:
    target = urllib.parse.urljoin(current, location)
    try:
        scheme = _origin(target)[0]
    except ValueError:  # a malformed port
        scheme = ""
    if scheme not in ("http", "https") or (_origin(current)[0] == "https" and scheme != "https"):
        raise HubError(f"refusing a redirect from {_host(current)} to {_bare(target)}", retryable=False)
    return target


def _int(value: str | None) -> int | None:
    return int(value) if value is not None and value.strip().isdigit() else None


def _normalize_etag(etag: str | None) -> str | None:
    """huggingface_hub file_download._normalize_etag: drop a weak marker and the quotes."""
    return etag.lstrip("W/").strip('"') if etag else None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """urllib would follow redirects itself, carrying every header (the token too) to whatever host it's sent to.
    Returning None makes a redirect come back as an HTTPError, and Hub._open follows it with headers of its choosing."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _gunzip(data: bytes, limit: int, url: str) -> bytes:
    """Asked for identity, a server may still gzip text (huggingface_hub's http_get notes the Hub does, for files like
    vocab.json). Unpacked under the same size cap, so a small download can't expand into gigabytes."""
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        out = d.decompress(data, limit + 1)
    except zlib.error:
        raise HubError(f"{_bare(url)} came gzipped but isn't valid gzip", retryable=False) from None
    if len(out) > limit or d.unconsumed_tail:
        raise HubError(f"{_bare(url)} is bigger than expected", retryable=False)
    if not d.eof:
        raise HubError(f"the download of {_bare(url)} was cut short")
    return out


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


# --- the Hub ----------------------------------------------------------------------------

@dataclass(frozen=True)
class FileInfo:
    """One file of a commit as the Hub lists it: huggingface_hub hf_api.RepoFile, from the tree API."""
    path: str
    size: int
    oid: str  # its Git blob id (sha1 of "blob <size>\0" + content); for an LFS file, the pointer's
    lfs_sha256: str | None  # for files stored with Git LFS (weights): the sha256 of the content

    @classmethod
    def from_api(cls, entry: dict) -> "FileInfo":
        try:
            lfs = entry.get("lfs")
            path, oid = _check_path(entry["path"]), entry["oid"]
            size, sha = (lfs["size"], lfs["oid"]) if lfs else (entry["size"], None)
        except (KeyError, TypeError, AttributeError):
            raise HubError("the Hub's file listing isn't in the shape expected", retryable=False) from None
        if type(size) is not int or size < 0 or not isinstance(oid, str) or not _GIT_OID.match(oid) \
                or not (sha is None or (isinstance(sha, str) and _SHA256.match(sha))):
            raise HubError(f"the Hub's file listing has a bad entry for {path}", retryable=False)
        return cls(path, size, oid, sha)


class _FileCheck:
    """Hashes a file as it streams past and checks it against what the Hub published: the LFS sha256 for weights and
    other large files, the Git blob id for small files kept in Git. Returns the content's sha256 either way."""

    def __init__(self, info: FileInfo):
        self.info, self.n = info, 0
        self.sha256 = hashlib.sha256()
        git = hashlib.sha1 if len(info.oid) == 40 else hashlib.sha256
        self.git = None if info.lfs_sha256 else git(b"blob %d\0" % info.size)

    def update(self, data) -> None:
        self.n += len(data)
        self.sha256.update(data)
        if self.git is not None:
            self.git.update(data)

    def finish(self) -> str:
        path, digest = self.info.path, self.sha256.hexdigest()
        if self.n != self.info.size:
            raise IntegrityError(f"{path} is {self.n} bytes, but the Hub lists it at {self.info.size}")
        if self.info.lfs_sha256 is not None and digest != self.info.lfs_sha256:
            raise IntegrityError(f"{path} doesn't match the sha256 the Hub publishes for it (expected "
                                 f"{self.info.lfs_sha256[:16]}…, got {digest[:16]}…): the copy is corrupt or was altered")
        if self.git is not None and self.git.hexdigest() != self.info.oid:
            raise IntegrityError(f"{path} doesn't match the Git blob id the Hub publishes for it: corrupt or altered")
        return digest


class Hub:
    """A client for the Hub's HTTP API, the parts huggingface_hub's HfApi and hf_hub_download use for reading models.

    `endpoint` defaults to $HF_ENDPOINT, then https://huggingface.co. The token (find_token) is sent as
    `Authorization: Bearer` only to requests for the endpoint's own origin (scheme, host and port): never to the CDN
    a download redirects to, nor anywhere else. That's stricter than huggingface_hub, which also trusts its other
    Hub hosts (hf.co, the staging Hub): see utils/_http._httpx2_follow_hub_redirects_with_backoff and _HUB_ORIGINS."""

    def __init__(self, endpoint: str | None = None, token: str | None = None, *, timeout: float = 30.0,
                 retries: int = RETRIES, backoff: float = 0.25):
        self.endpoint = (endpoint or _endpoint()).rstrip("/")
        if _origin(self.endpoint)[0] not in ("http", "https"):
            raise ValueError(f"HF_ENDPOINT {self.endpoint!r} isn't an http(s) address")
        self._token = find_token(token)  # never logged, never in an error message, never in repr
        self.timeout, self.retries, self.backoff = timeout, retries, backoff
        self._opener = urllib.request.build_opener(_NoRedirect)
        self._files: dict[tuple, RemoteFile] = {}
        self._headers: dict[tuple, Header] = {}
        self._maps: dict[tuple, dict[str, str]] = {}

    def __repr__(self) -> str:
        return f"Hub({self.endpoint!r}, token={'set' if self._token else 'none'})"

    @property
    def has_token(self) -> bool:
        return self._token is not None

    def _on_hub(self, url: str) -> bool:
        return _origin(url) == _origin(self.endpoint)

    # --- requests ---------------------------------------------------------------------

    def _open(self, url: str, headers: dict | None = None, *, repo: str | None = None, rev: str | None = None):
        """One GET, following redirects here rather than in urllib, so each hop gets the right headers: the Range
        header on every hop (a ranged read must stay ranged at the CDN), the token only on hops to the endpoint.
        Returns (response, the URL it came from, headers of the last response from the Hub itself or None)."""
        hub_headers = None
        for _ in range(MAX_REDIRECTS + 1):
            h = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity", **(headers or {})}  # identity: real lengths
            if self._token and self._on_hub(url):
                h["Authorization"] = f"Bearer {self._token}"
            try:
                resp = self._opener.open(urllib.request.Request(url, headers=h), timeout=self.timeout)
            except urllib.error.HTTPError as e:
                if self._on_hub(url):
                    hub_headers = e.headers
                location = e.headers.get("Location") if e.code in _REDIRECTS else None
                if not location:
                    raise self._refused(e, url, repo, rev) from None
                e.close()
                url = _redirect_target(url, location)
                continue
            except (OSError, http.client.HTTPException) as e:
                raise self._unreachable(url, e) from None
            if self._on_hub(url):
                hub_headers = resp.headers
            return resp, url, hub_headers
        raise HubError(f"too many redirects fetching {_bare(url)}", retryable=False)

    def _unreachable(self, url: str, e: BaseException) -> HubError:
        reason = getattr(e, "reason", None) or e.__class__.__name__
        if self._on_hub(url):
            return HubError(f"can't reach Hugging Face at {self.endpoint} ({reason}): check your internet connection, "
                            "or $HF_ENDPOINT if you use a mirror")
        return HubError(f"can't reach {_host(url)}, where Hugging Face keeps large files ({reason}): "
                        "check your internet connection")

    def _refused(self, e: urllib.error.HTTPError, url: str, repo: str | None, rev: str | None) -> HubError:
        """An HTTP error in plain words. The Hub says why in an X-Error-Code header, which huggingface_hub's
        utils/_http.hf_raise_for_status turns into GatedRepoError, RepositoryNotFoundError and so on."""
        status, code = e.code, e.headers.get("X-Error-Code") or ""
        retry_after = e.headers.get("Retry-After")
        reset = re.search(r";\s*t\s*=\s*(\d+)", e.headers.get("RateLimit") or "")
        if status == 429 and reset:  # the Hub's `RateLimit: "api";r=0;t=55`, read as utils/_http.parse_ratelimit_headers does
            retry_after = str(int(reset.group(1)) + 1)
        e.close()
        if self._on_hub(url) and repo:
            page = f"{self.endpoint}/{repo}"
            login = f"set HF_TOKEN (make one at {self.endpoint}/settings/tokens) or run `vibehash hub login`"
            if code == "GatedRepo" or (status == 403 and code != "RepoNotFound"):
                if not self._token:
                    return AccessError(f"{repo} needs you to accept its licence at {page}, then {login}", status)
                return AccessError(
                    f"your Hugging Face account doesn't have access to {repo} yet: accept its licence at {page} (some "
                    "authors approve requests by hand, which can take a while). If you have, check HF_TOKEN belongs to "
                    "that account and, for a fine-grained token, that it may read gated repos", status)
            if code == "RepoNotFound" or status == 401:
                # The Hub answers 401 for a private or gated repo when no token is sent, and for one that doesn't
                # exist, so as not to reveal which (hf_raise_for_status treats both as not found).
                if not self._token:
                    return AccessError(f"no model {repo} on {_host(self.endpoint)}: check the name (organisation/model, "
                                       f"as in its page's address). If it's private or gated, {login}", status)
                return AccessError(f"no model {repo} that your token can see on {_host(self.endpoint)}: check the name, "
                                   "and that the token's account has access", status)
            if code == "RevisionNotFound":
                return HubError(f"{repo} has no branch, tag or commit {rev!r}", status, retryable=False)
        if status == 404:
            return HubError(f"not found: {_bare(url)}", 404)
        if status == 416:
            return HubError(f"{_host(url)} refused a byte range of {_bare(url)} (past the end?)", 416)
        return HubError(f"{_host(url)} answered HTTP {status} for {_bare(url)}", status, retry_after=retry_after)

    def _wait(self, err: HubError, attempt: int) -> float:
        """client._backoff's schedule: exponential up to 30 s with jitter, or longer if the server asks (the Hub's
        rate limits reset every 5 minutes, and coming back sooner only spends another request)."""
        try:
            asked = min(330.0, float(err.retry_after or 0))
        except ValueError:
            asked = 0.0
        return max(asked, min(30.0, self.backoff * 2**attempt)) * random.uniform(1.0, 1.25)

    def _retrying(self, fn):
        for attempt in range(self.retries + 1):
            try:
                return fn()
            except HubError as e:
                if not e.retryable or attempt == self.retries:
                    raise
                wait = self._wait(e, attempt)
                log.log(logging.WARNING if e.status == 429 else logging.INFO, "%s; retrying in %.1f s", e, wait)
                time.sleep(wait)

    def _get(self, url: str, limit: int = MAX_JSON, *, repo: str | None = None, rev: str | None = None):
        """A whole small response (JSON, a config), refused past `limit`. Returns (body, its headers, Hub headers)."""
        def once():
            resp, _, hub_headers = self._open(url, repo=repo, rev=rev)
            with resp:
                encoding = (resp.headers.get("Content-Encoding") or "identity").lower()
                length = _int(resp.headers.get("Content-Length"))
                if length is not None and length > limit:
                    raise HubError(f"{_bare(url)} is {_gb(length)}, more than expected", retryable=False)
                try:
                    body = resp.read(limit + 1)
                except (OSError, http.client.HTTPException) as e:
                    raise self._unreachable(url, e) from None
            if len(body) > limit:
                raise HubError(f"{_bare(url)} is bigger than expected", retryable=False)
            if length is not None and len(body) != length:
                raise HubError(f"the download of {_bare(url)} was cut short")
            if encoding == "gzip":
                body = _gunzip(body, limit, url)
            elif encoding != "identity":
                raise HubError(f"{_bare(url)} came {encoding}-encoded, which isn't expected", retryable=False)
            return body, resp.headers, hub_headers
        return self._retrying(once)

    def _get_json(self, url: str, *, repo: str | None = None, rev: str | None = None):
        body, headers, _ = self._get(url, repo=repo, rev=rev)
        try:
            return json.loads(body), headers
        except (ValueError, RecursionError):
            raise HubError(f"{_bare(url)} didn't answer with JSON", retryable=False) from None

    # --- the API ----------------------------------------------------------------------

    def model_info(self, repo: str, revision: str | None = None) -> dict:
        """The Hub's description of a model, at its default branch or at `revision`: huggingface_hub HfApi.model_info,
        GET /api/models/<repo>[/revision/<rev>]. Its "sha" is the commit that revision points at."""
        url = f"{self.endpoint}/api/models/{check_repo(repo)}"
        if revision is not None:
            url += f"/revision/{urllib.parse.quote(revision, safe='')}"
        info, _ = self._get_json(url, repo=repo, rev=revision)
        if not isinstance(info, dict):
            raise HubError(f"the Hub's description of {repo} isn't in the shape expected", retryable=False)
        return info

    def resolve(self, ref: str) -> tuple[str, str]:
        """(repo, commit sha) for a model as people write it (parse_ref): a branch or tag, or no revision at all for
        the default branch, becomes the commit it points at now. Show and pin that, and the job never moves."""
        repo, rev = parse_ref(ref, self.endpoint)
        sha = self.model_info(repo, rev).get("sha")
        if not isinstance(sha, str) or not _COMMIT.match(sha):
            raise HubError(f"the Hub didn't say which commit {ref} is at", retryable=False)
        if rev is not None and _COMMIT.match(rev.lower()) and rev.lower() != sha:
            raise IntegrityError(f"asked for commit {rev}, the Hub answered with {sha}")
        return repo, sha

    def list_files(self, repo: str, commit: str) -> dict[str, FileInfo]:
        """Every file of a commit with its size and hashes: huggingface_hub HfApi.list_repo_tree(recursive=True),
        GET /api/models/<repo>/tree/<commit>?recursive=true, which pages through a `Link: <...>; rel="next"` header
        (utils/_pagination._paginate_pages). A next page off the Hub's own origin is refused, not followed."""
        url = f"{self.endpoint}/api/models/{check_repo(repo)}/tree/{check_commit(commit)}?recursive=true&expand=false"
        files: dict[str, FileInfo] = {}
        for _ in range(MAX_PAGES):
            page, headers = self._get_json(url, repo=repo, rev=commit)
            if not isinstance(page, list):
                raise HubError(f"the Hub's file listing for {repo} isn't in the shape expected", retryable=False)
            for entry in page:
                if isinstance(entry, dict) and entry.get("type") == "file":
                    info = FileInfo.from_api(entry)
                    files[info.path] = info
            nxt = _next_link(headers.get("Link"))
            if nxt is None:
                return files
            url = urllib.parse.urljoin(url, nxt)
            if not self._on_hub(url):
                raise HubError(f"the Hub's file listing pointed off the Hub, to {_host(url)}", retryable=False)
        raise HubError(f"{repo} lists more files than expected", retryable=False)

    def file_url(self, repo: str, commit: str, path: str) -> str:
        """huggingface_hub file_download.hf_hub_url: <endpoint>/<repo>/resolve/<revision>/<path>."""
        return (f"{self.endpoint}/{check_repo(repo)}/resolve/{urllib.parse.quote(commit, safe='')}/"
                f"{urllib.parse.quote(_check_path(path))}")

    def open(self, repo: str, commit: str, path: str, size: int | None = None, sha256: str | None = None) -> "RemoteFile":
        """A file to read by byte range. One per file per Hub, so the CDN address it learns is reused."""
        key = (repo, check_commit(commit), path)
        if key not in self._files:
            self._files[key] = RemoteFile(self, repo, commit, path)
        f = self._files[key]
        f.size = f.size if size is None else size
        f.sha256 = f.sha256 if sha256 is None else sha256
        return f

    def get_file(self, repo: str, commit: str, path: str, *, info: FileInfo | None = None, limit: int = MAX_JSON) -> bytes:
        """A whole small file (an index, a config), checked against the Hub's listing when `info` gives it."""
        body, _, hub_headers = self._get(self.file_url(repo, check_commit(commit), path), limit, repo=repo, rev=commit)
        served = ((hub_headers or {}).get("X-Repo-Commit") or "").lower()
        if served and served != commit:
            raise IntegrityError(f"the Hub served {path} from commit {served[:12]}, not {commit[:12]}")
        if info is not None:
            check = _FileCheck(info)
            check.update(body)
            check.finish()
        return body

    def header(self, repo: str, commit: str, path: str) -> "Header":
        key = (repo, commit, path)
        if key not in self._headers:
            self._headers[key] = read_header(self.open(repo, commit, path))
        return self._headers[key]

    def weight_map(self, repo: str, commit: str, listing: dict[str, FileInfo] | None = None) -> dict[str, str]:
        """{tensor name: the file holding it}: model.safetensors.index.json's "weight_map" when the model is sharded
        (huggingface_hub HfApi.get_safetensors_metadata reads it the same way), else everything in model.safetensors."""
        key = (repo, check_commit(commit))
        if key not in self._maps:
            index = None
            if listing is None or INDEX_FILE in listing:
                try:
                    index = self.get_file(repo, commit, INDEX_FILE, info=(listing or {}).get(INDEX_FILE))
                except HubError as e:
                    if e.status != 404:
                        raise
            if index is not None:
                self._maps[key] = parse_index(index)
            else:
                if listing is not None and SINGLE_FILE not in listing:
                    raise self.no_safetensors(repo, commit, listing)
                try:
                    names = self.header(repo, commit, SINGLE_FILE).tensors
                except HubError as e:
                    if e.status != 404:
                        raise
                    raise self.no_safetensors(repo, commit, listing) from None
                self._maps[key] = {name: SINGLE_FILE for name in names}
        return self._maps[key]

    def no_safetensors(self, repo: str, commit: str, listing: dict[str, FileInfo] | None = None) -> HubError:
        """The error for a model we can't load, saying why."""
        listing = listing if listing is not None else self.list_files(repo, commit)
        if any(p.endswith(PICKLES) for p in listing):
            return HubError(f"{repo} only has PyTorch pickle weights (.bin/.pt), no safetensors. Loading a pickle can "
                            "run any code its author put in it, so Vibe Hash never does: pick a model published as "
                            "safetensors (most are; look for model.safetensors on its Files tab)", retryable=False)
        if any(p.endswith(".safetensors") for p in listing):
            return HubError(f"{repo} has safetensors files but neither {SINGLE_FILE} nor {INDEX_FILE}, so which tensor "
                            "is where isn't said anywhere", retryable=False)
        return HubError(f"{repo} has no model weights (no .safetensors files)", retryable=False)

    def fetch_tensor(self, repo: str, commit: str, name: str) -> tuple[str, list[int], bytes]:
        """One tensor as (safetensors dtype string, shape, raw bytes), read by range: the index (if sharded), the
        file's header (two small reads, once per file), then exactly the tensor's bytes. Not checked against
        anything but the header's lengths: check it with verify_tensor, or use load_tensor, which does."""
        wm = self.weight_map(repo, commit)
        if name not in wm:
            raise KeyError(f"{repo} has no tensor {name}")
        hdr = self.header(repo, commit, wm[name])
        if name not in hdr.tensors:
            raise IntegrityError(f"{INDEX_FILE} puts {name} in {wm[name]}, but that file doesn't have it")
        dtype, shape, begin, end = hdr.tensors[name]
        return dtype, list(shape), self.open(repo, commit, wm[name]).read(hdr.data_start + begin, end - begin)

    def describe(self, ref: str) -> "ModelSummary":
        """What a model is, before anyone downloads it or pays for a job: see ModelSummary."""
        repo, rev = parse_ref(ref, self.endpoint)
        info = self.model_info(repo, rev)
        commit = info.get("sha")
        if not isinstance(commit, str) or not _COMMIT.match(commit):
            raise HubError(f"the Hub didn't say which commit {ref} is at", retryable=False)
        listing = self.list_files(repo, commit)
        config = info.get("config") if isinstance(info.get("config"), dict) else {}  # the API's public subset
        weights = sorted(p for p in listing if p.endswith(".safetensors"))
        try:
            if "config.json" in listing:
                full = json.loads(self.get_file(repo, commit, "config.json", info=listing["config.json"]))
                config = full if isinstance(full, dict) else config
            if INDEX_FILE in listing:
                weights = sorted(set(self.weight_map(repo, commit, listing).values()) & set(listing))
        except (AccessError, ValueError):
            pass  # a gated model's files need its licence accepted first; the API still describes it
        card = info.get("cardData") if isinstance(info.get("cardData"), dict) else {}
        licence = card.get("license_name") if card.get("license") == "other" else card.get("license")
        if not licence:
            licence = next((t[8:] for t in info.get("tags") or [] if isinstance(t, str) and t.startswith("license:")), None)
        dtype = config.get("torch_dtype") or config.get("dtype")
        quant = config.get("quantization_config")
        if isinstance(quant, dict) and quant.get("quant_method"):
            dtype = f"{dtype or '?'}, {quant['quant_method']}" + (f" ({quant['fmt']})" if quant.get("fmt") else "")
        st = info.get("safetensors") if isinstance(info.get("safetensors"), dict) else {}
        archs = config.get("architectures")
        # Mixture-of-experts configs name their expert count differently by family (DeepSeek, Mixtral, Qwen MoE).
        experts = next((config[k] for k in ("n_routed_experts", "num_local_experts", "num_experts")
                        if type(config.get(k)) is int), None)
        active = config.get("num_experts_per_tok") if type(config.get("num_experts_per_tok")) is int else None
        return ModelSummary(
            repo=repo, revision=rev or "the default branch", commit=commit, gated=bool(info.get("gated")),
            licence=str(licence) if licence else None,
            architecture=str(archs[0]) if isinstance(archs, list) and archs else config.get("model_type"),
            dtype=str(dtype) if dtype else None,
            parameters=st.get("total") if type(st.get("total")) is int else None,
            weight_files=len(weights), weight_bytes=sum(listing[p].size for p in weights),
            has_tokenizer="tokenizer.json" in listing,
            pickle_only=not weights and any(p.endswith(PICKLES) for p in listing),
            layers=config.get("num_hidden_layers") if type(config.get("num_hidden_layers")) is int else None,
            experts=experts, experts_per_token=active, config=config)


def _next_link(link: str | None) -> str | None:
    """The rel="next" URL of a Link header, as requests' and httpx's Response.links parse it."""
    for part in (link or "").split(","):
        m = re.match(r'\s*<([^>]*)>\s*;(.*)', part)
        if m and re.search(r'rel\s*=\s*"?next"?', m.group(2)):
            return m.group(1)
    return None


@dataclass(frozen=True)
class ModelSummary:
    """A pre-flight look at a model: enough to know what a job would train before it's posted."""
    repo: str
    revision: str  # what was asked for
    commit: str  # what that is right now: the one to pin
    gated: bool
    licence: str | None  # from the model card's metadata
    architecture: str | None  # config.json's "architectures"
    dtype: str | None  # config.json's dtype, and its quantization if any
    parameters: int | None  # as the Hub counts them from the safetensors headers
    weight_files: int
    weight_bytes: int
    has_tokenizer: bool  # a tokenizer.json, which is what training text needs
    pickle_only: bool  # weights only as PyTorch pickles, which Vibe Hash won't load
    layers: int | None = None  # config.json's num_hidden_layers: what a sharded job splits into blocks
    experts: int | None = None  # for a mixture of experts, how many per layer...
    experts_per_token: int | None = None  # ...and how many each token uses
    config: dict = field(default_factory=dict, compare=False, repr=False)  # config.json, or the API's part of it

    def __str__(self) -> str:
        params = "unknown size" if self.parameters is None else (
            f"{self.parameters / 1e9:.1f}B parameters" if self.parameters >= 1e9 else f"{self.parameters / 1e6:.0f}M parameters")
        if self.experts:
            params += f", mixture of {self.experts} experts" + (f" ({self.experts_per_token} per token)"
                                                                 if self.experts_per_token else "")
        if self.layers:
            params += f", {self.layers} layers"
        weights = ("no safetensors weights (only PyTorch pickles, which Vibe Hash won't load)" if self.pickle_only else
                   f"{self.weight_files} safetensors file{'s' * (self.weight_files != 1)}, {_gb(self.weight_bytes)}")
        return "\n".join([
            f"{self.repo} at commit {self.commit} ({self.revision})",
            f"  {self.architecture or 'unknown architecture'}, {params}, {self.dtype or 'dtype not stated'}",
            f"  weights: {weights}",
            f"  tokenizer.json: {'yes' if self.has_tokenizer else 'no'}   licence: {self.licence or 'not stated'}   "
            f"gated: {'yes: accept its licence on its page first' if self.gated else 'no'}"])


# --- ranged reads -----------------------------------------------------------------------

class RemoteFile:
    """One file of one commit, read by HTTP byte range without downloading the rest.

    For a large file the Hub answers `resolve/<commit>/<path>` with a redirect to a signed address on its CDN (the
    302 huggingface_hub's get_hf_file_metadata reads). We follow it keeping the Range header and dropping the token,
    then keep that address for later reads of the same file, as _get_metadata_or_catch_error does: a thousand tensor
    reads from one shard then cost the Hub one request, not a thousand (it rate-limits resolves). Signed addresses
    expire, so a refusal from the CDN sends us back to the Hub for a fresh one."""

    def __init__(self, hub: Hub, repo: str, commit: str, path: str, size: int | None = None, sha256: str | None = None):
        self.hub, self.repo, self.commit, self.path = hub, repo, check_commit(commit), _check_path(path)
        self.url = hub.file_url(repo, commit, path)
        self.size, self.sha256 = size, sha256  # learned from the first response when not given
        self._location: str | None = None

    def read(self, offset: int, size: int) -> bytes:
        return b"".join(self.iter_range(offset, size))

    def iter_range(self, offset: int, size: int) -> Iterator[bytes]:
        """The bytes [offset, offset + size) in pieces. A dropped connection, timeout or server fault resumes from
        the last byte received (huggingface_hub's http_get does the same) rather than starting over."""
        if offset < 0 or size < 0 or (self.size is not None and offset + size > self.size):
            raise IntegrityError(f"bytes {offset}-{offset + size} are outside {self.path} ({self.size} bytes)")
        got = failures = 0
        while got < size:
            try:
                with self._request(offset + got, size - got) as resp:
                    while got < size:
                        piece = resp.read(min(CHUNK, size - got))
                        if not piece:
                            raise HubError(f"the download of {self.path} was cut off")
                        got += len(piece)
                        failures = 0  # progress, so the retry budget starts again
                        yield piece
            except (HubError, OSError, http.client.HTTPException) as e:
                err = e if isinstance(e, HubError) else self.hub._unreachable(self._location or self.url, e)
                if not err.retryable or failures >= self.hub.retries:
                    raise err from None
                wait = self.hub._wait(err, failures)
                failures += 1
                log.info("%s; resuming at byte %d in %.1f s", err, offset + got, wait)
                time.sleep(wait)

    def _request(self, start: int, n: int):
        rng = {"Range": f"bytes={start}-{start + n - 1}"}
        if self._location is not None:
            try:
                return self._checked(self.hub._open(self._location, rng)[0], start, n)
            except HubError as e:
                if e.status not in (400, 401, 403, 404, 410):
                    raise
                self._location = None  # the signed address expired: ask the Hub for a fresh one
        resp, final, hub_headers = self.hub._open(self.url, rng, repo=self.repo, rev=self.commit)
        try:
            self._check_hub(hub_headers)
        except BaseException:
            resp.close()
            raise
        self._location = None if final == self.url else final
        return self._checked(resp, start, n)

    def _check_hub(self, h) -> None:
        """The Hub's headers on the resolve (huggingface_hub get_hf_file_metadata reads the same three): the commit
        served, and for an LFS file its size and sha256 (X-Linked-Size, X-Linked-Etag), when we know what to expect."""
        if h is None:
            return
        served = (h.get("X-Repo-Commit") or "").lower()
        if served and served != self.commit:
            raise IntegrityError(f"the Hub served {self.path} from commit {served[:12]}, not {self.commit[:12]}")
        size = _int(h.get("X-Linked-Size"))
        if size is not None and self.size is not None and size != self.size:
            raise IntegrityError(f"the Hub says {self.path} is {size} bytes, not {self.size}")
        etag = _normalize_etag(h.get("X-Linked-Etag"))
        if etag and self.sha256 and _SHA256.match(etag) and etag != self.sha256:
            raise IntegrityError(f"the Hub's sha256 for {self.path} isn't the one pinned")

    def _checked(self, resp, start: int, n: int):
        """The response must be exactly the bytes asked for: a 206 whose Content-Range and length match."""
        try:
            where = f"{_host(resp.geturl())} for {self.path}"
            if (resp.headers.get("Content-Encoding") or "identity").lower() != "identity":
                raise IntegrityError(f"{where}: sent compressed bytes for a byte range")
            if resp.status == 206:
                m = _CONTENT_RANGE.fullmatch((resp.headers.get("Content-Range") or "").strip())
                if not m or int(m[1]) != start or int(m[2]) != start + n - 1:
                    raise IntegrityError(f"{where}: sent bytes {resp.headers.get('Content-Range')!r}, "
                                         f"not {start}-{start + n - 1}")
                total = None if m[3] == "*" else int(m[3])
            elif resp.status == 200 and start == 0:  # Range ignored: fine only if that's the whole file anyway
                total = _int(resp.headers.get("Content-Length"))
                if total != n:
                    raise IntegrityError(f"{where}: ignored the byte range and would send the whole file")
            else:
                raise IntegrityError(f"{where}: answered HTTP {resp.status} to a byte-range request")
            if total is not None:
                if start + n > total or (self.size is not None and total != self.size):
                    raise IntegrityError(f"{where}: the file is {total} bytes, not {self.size}")
                self.size = total
            if _int(resp.headers.get("Content-Length")) not in (None, n):
                raise IntegrityError(f"{where}: sent {resp.headers.get('Content-Length')} bytes, not {n}")
            return resp
        except BaseException:
            resp.close()
            raise


# --- safetensors headers ----------------------------------------------------------------

@dataclass(frozen=True)
class Header:
    """A safetensors header. Offsets are relative to the data, which starts at `data_start`."""
    size: int  # bytes of JSON after the 8-byte length
    tensors: dict[str, tuple[str, tuple[int, ...], int, int]]  # name -> (dtype, shape, begin, end)
    metadata: dict[str, str]

    @property
    def data_start(self) -> int:
        return 8 + self.size

    @property
    def data_size(self) -> int:
        return max((end for *_, end in self.tensors.values()), default=0)


def tensor_nbytes(dtype: str, shape) -> int:
    bits = math.prod(shape) * DTYPE_BITS[dtype]
    if bits % 8:
        raise IntegrityError(f"a {dtype} tensor of shape {list(shape)} doesn't fill whole bytes")
    return bits // 8


def _unique(pairs):
    """json object hook: a key given twice is refused, since readers disagree on which one wins."""
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r}")
        out[k] = v
    return out


def _is_count(x) -> bool:
    return type(x) is int and x >= 0


def check_header_size(n: int, file_size: int | None, path: str = "the file") -> None:
    if n > MAX_HEADER:
        raise IntegrityError(f"{path}: its header claims {n} bytes; more than {MAX_HEADER} isn't a real model")
    if n < 2 or (file_size is not None and 8 + n > file_size):
        raise IntegrityError(f"{path}: its header length ({n}) doesn't fit the file ({file_size} bytes)")


def parse_header(raw: bytes, file_size: int | None = None, path: str = "the file") -> Header:
    """Check a header as a hostile file deserves, like the safetensors crate's validate(): JSON with no repeated key,
    known dtypes, each tensor's length equal to its dtype times its shape, the tensors back to back from the start of
    the data with no gap or overlap, and (when the file's size is known) ending exactly at the end of the file."""
    try:
        obj = json.loads(bytes(raw).decode("utf-8"), object_pairs_hook=_unique)
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise IntegrityError(f"{path}: its header isn't valid JSON ({e})") from None
    if not isinstance(obj, dict):
        raise IntegrityError(f"{path}: its header isn't a JSON object")
    meta = obj.pop("__metadata__", {})
    if not isinstance(meta, dict) or not all(isinstance(v, str) for v in meta.values()):
        raise IntegrityError(f"{path}: its __metadata__ isn't a map of strings")
    tensors = {}
    for name, t in obj.items():
        if not isinstance(t, dict) or set(t) != {"dtype", "shape", "data_offsets"}:
            raise IntegrityError(f"{path}: tensor {name!r} isn't described as dtype, shape and data_offsets")
        dtype, shape, offs = t["dtype"], t["shape"], t["data_offsets"]
        if dtype not in DTYPE_BITS:
            raise IntegrityError(f"{path}: tensor {name!r} has unknown dtype {dtype!r}")
        if not isinstance(shape, list) or not all(_is_count(d) for d in shape):
            raise IntegrityError(f"{path}: tensor {name!r} has a bad shape")
        if not isinstance(offs, list) or len(offs) != 2 or not all(_is_count(o) for o in offs) or offs[0] > offs[1]:
            raise IntegrityError(f"{path}: tensor {name!r} has bad data_offsets")
        if offs[1] - offs[0] != tensor_nbytes(dtype, shape):
            raise IntegrityError(f"{path}: tensor {name!r} is {offs[1] - offs[0]} bytes, but a {dtype} {shape} is "
                                 f"{tensor_nbytes(dtype, shape)}")
        tensors[name] = (dtype, tuple(shape), offs[0], offs[1])
    end = 0
    for name, (_, _, begin, stop) in sorted(tensors.items(), key=lambda kv: kv[1][2:]):
        if begin != end:
            raise IntegrityError(f"{path}: tensor {name!r} starts at {begin}, not {end} (a gap or an overlap)")
        end = stop
    if file_size is not None and 8 + len(raw) + end != file_size:
        raise IntegrityError(f"{path}: its tensors end at byte {8 + len(raw) + end}, but the file is {file_size} bytes")
    return Header(len(raw), tensors, meta)


def read_header(f: RemoteFile) -> Header:
    """A remote file's header in two small ranged reads: the 8-byte length, then exactly that much JSON. The length
    is checked before the second read, so a hostile file can't make us fetch gigabytes of "header"."""
    if f.size is not None and f.size < 10:
        raise IntegrityError(f"{f.path} is too small to be a safetensors file")
    (n,) = struct.unpack("<Q", f.read(0, 8))
    check_header_size(n, f.size, f.path)
    return parse_header(f.read(8, n), f.size, f.path)


def parse_index(data: bytes) -> dict[str, str]:
    """model.safetensors.index.json's "weight_map", {tensor name: shard file}, checked like any downloaded JSON."""
    try:
        wm = json.loads(data, object_pairs_hook=_unique)["weight_map"]
    except (ValueError, RecursionError, KeyError, TypeError):
        raise IntegrityError(f"{INDEX_FILE} isn't a safetensors index") from None
    if not isinstance(wm, dict) or not all(isinstance(f, str) and _check_path(f).endswith(".safetensors")
                                           for f in wm.values()):
        raise IntegrityError(f"{INDEX_FILE} has a bad weight_map")
    return wm


class _TensorHashes:
    """Reads a safetensors file as it streams past, once: parses its header from the first bytes, then hashes each
    tensor's bytes separately. Their sha256s are what let a worker check a slice it downloads by itself."""

    def __init__(self, size: int, path: str):
        self.size, self.path = size, path
        self.buf: bytearray | None = bytearray()
        self.header: Header | None = None
        self.order: list = []
        self.digests: dict[str, str] = {}
        self.i = self.pos = 0
        self.h = hashlib.sha256()

    def update(self, data: bytes) -> None:
        if self.header is None:
            self.buf += data
            if len(self.buf) < 8:
                return
            (n,) = struct.unpack("<Q", self.buf[:8])
            check_header_size(n, self.size, self.path)
            if len(self.buf) < 8 + n:
                return
            self.header = parse_header(bytes(self.buf[8:8 + n]), self.size, self.path)
            self.order = sorted(self.header.tensors.items(), key=lambda kv: kv[1][2:])
            data, self.buf = bytes(self.buf[8 + n:]), None
        mv = memoryview(data)
        while True:
            while self.i < len(self.order) and self.order[self.i][1][3] == self.pos:  # empty tensors too
                self.digests[self.order[self.i][0]] = self.h.hexdigest()
                self.i, self.h = self.i + 1, hashlib.sha256()
            if not mv:
                return
            if self.i == len(self.order):
                raise IntegrityError(f"{self.path}: bytes after the last tensor")
            take = min(len(mv), self.order[self.i][1][3] - self.pos)
            self.h.update(mv[:take])
            self.pos, mv = self.pos + take, mv[take:]

    def finish(self) -> tuple[Header, dict[str, str]]:
        if self.header is None or len(self.digests) != len(self.order):
            raise IntegrityError(f"{self.path} ended early")
        return self.header, self.digests


# --- the manifest -----------------------------------------------------------------------

@dataclass(frozen=True)
class TensorEntry:
    file: str
    dtype: str  # the safetensors dtype string, e.g. "BF16", "F8_E4M3"
    shape: tuple[int, ...]
    offset: int  # of the tensor's first byte in the file (header included), so a worker needs no header read
    length: int
    sha256: str

    def to_dict(self) -> dict:
        return {"file": self.file, "dtype": self.dtype, "shape": list(self.shape), "offset": self.offset,
                "length": self.length, "sha256": self.sha256}


@dataclass(frozen=True)
class Manifest:
    """Per-tensor hashes for one pinned (repo, commit). Serialized canonically (ledger.canonical), its sha256 is what
    a job pins:

        {"format": "vibehash-hf-manifest/1", "repo": "org/name", "commit": "<40 hex>",
         "files":   {"<path>": {"size": <bytes>, "sha256": "<hex>"}, ...},
         "tensors": {"<name>": {"file": "<path>", "dtype": "BF16", "shape": [...],
                                "offset": <first byte in the file>, "length": <bytes>, "sha256": "<hex>"}, ...}}

    "files" holds every weight file, whose sha256 is the one the Hub publishes for it (its Git LFS object id), plus
    the index and the small files listed in EXTRA_FILES that the repo has (config.json, the tokenizer), each checked
    against the Hub's listing as it was built."""
    repo: str
    commit: str
    files: dict[str, dict]
    tensors: dict[str, TensorEntry]

    def to_dict(self) -> dict:
        return {"format": MANIFEST_FORMAT, "repo": self.repo, "commit": self.commit,
                "files": {p: dict(f) for p, f in self.files.items()},
                "tensors": {n: e.to_dict() for n, e in self.tensors.items()}}

    def to_bytes(self) -> bytes:
        # Imported here, not at the top: ledger pulls in the job and model modules, which may well import this one.
        from .ledger import canonical
        return canonical(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @property
    def total_bytes(self) -> int:
        return sum(e.length for e in self.tensors.values())

    def bytes_for(self, names: Iterable[str]) -> int:
        """Download size of these tensors (identical ones are fetched once)."""
        return sum({self.tensors[n].sha256: self.tensors[n].length for n in names}.values())

    @classmethod
    def from_bytes(cls, data: bytes, sha256: str | None = None) -> "Manifest":
        """Load a manifest, checking it's the one pinned (when `sha256` is given) and in canonical form, so one
        manifest has exactly one hash."""
        if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
            raise IntegrityError("this manifest isn't the one the job pins (its sha256 differs)")
        try:
            m = cls.from_dict(json.loads(data, object_pairs_hook=_unique))
        except (ValueError, RecursionError) as e:
            raise IntegrityError(f"not a manifest: {e}") from None
        if m.to_bytes() != bytes(data):
            raise IntegrityError("the manifest isn't in canonical form")
        return m

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        """Every field checked: a manifest arrives from elsewhere, and its offsets steer downloads."""
        if not isinstance(d, dict) or d.get("format") != MANIFEST_FORMAT or set(d) != {"format", "repo", "commit",
                                                                                       "files", "tensors"}:
            raise IntegrityError(f"not a {MANIFEST_FORMAT} manifest")
        repo, commit = check_repo(d["repo"]), check_commit(d["commit"])
        files, tensors = d["files"], d["tensors"]
        if not isinstance(files, dict) or not isinstance(tensors, dict):
            raise IntegrityError("a manifest's files and tensors are maps")
        for p, f in files.items():
            _check_path(p)
            if not isinstance(f, dict) or set(f) != {"size", "sha256"} or not _is_count(f["size"]) \
                    or not isinstance(f["sha256"], str) or not _SHA256.match(f["sha256"]):
                raise IntegrityError(f"the manifest's entry for file {p} is malformed")
        out = {}
        for name, t in tensors.items():
            try:
                e = TensorEntry(t["file"], t["dtype"], tuple(t["shape"]), t["offset"], t["length"], t["sha256"])
                ok = (set(t) == {"file", "dtype", "shape", "offset", "length", "sha256"} and e.file in files
                      and e.dtype in DTYPE_BITS and all(_is_count(x) for x in e.shape) and _is_count(e.offset)
                      and _is_count(e.length) and isinstance(e.sha256, str) and bool(_SHA256.match(e.sha256))
                      and e.length == tensor_nbytes(e.dtype, e.shape)
                      and e.offset + e.length <= files[e.file]["size"])
            except (KeyError, TypeError, IntegrityError):
                ok = False
            if not ok:
                raise IntegrityError(f"the manifest's entry for tensor {name!r} is malformed")
            out[name] = e
        return cls(repo, commit, {p: dict(f) for p, f in files.items()}, out)


Progress = Callable[[int, int, str], None]  # (bytes done, bytes in all, what's happening)


def _read_local(path: Path) -> Iterator[bytes]:
    with open(path, "rb") as f:
        yield from iter(lambda: f.read(CHUNK), b"")


def build_manifest(repo: str, commit: str, *, hub: Hub | None = None, local_dir: str | Path | None = None,
                   listing: dict[str, FileInfo] | None = None, extra: Iterable[str] = EXTRA_FILES,
                   progress: Progress | None = None) -> Manifest:
    """Stream each weight file of a pinned commit once, from the Hub or from `local_dir` (a copy of the repo, as
    `hf download --local-dir` makes), checking it against the sha256 the Hub publishes and hashing every tensor.
    The Hub's listing is fetched unless `listing` is given, so a local copy is held to what the Hub published too.
    Which files make up the model is the index's call (as transformers loads it), else model.safetensors alone."""
    hub = hub or Hub()
    check_repo(repo)
    check_commit(commit)
    listing = hub.list_files(repo, commit) if listing is None else listing
    local = Path(local_dir) if local_dir is not None else None

    def stream(path: str) -> Iterator[bytes]:
        info = listing[path]
        if local is None:
            return hub.open(repo, commit, path, info.size, info.lfs_sha256).iter_range(0, info.size)
        if not (local / path).is_file():
            raise FileNotFoundError(f"{local / path} is missing from the local copy")
        if (local / path).stat().st_size != info.size:
            raise IntegrityError(f"{local / path} is {(local / path).stat().st_size} bytes, but the Hub's {path} at "
                                 f"{commit[:12]} is {info.size}")
        return _read_local(local / path)

    files: dict[str, dict] = {}

    def small(path: str) -> bytes:
        """A small file, whole, checked against the listing and pinned in `files`."""
        data = b"".join(stream(path)) if local is not None else hub.get_file(repo, commit, path)
        check = _FileCheck(listing[path])
        check.update(data)
        files[path] = {"size": len(data), "sha256": check.finish()}
        return data

    if INDEX_FILE in listing:
        wm = parse_index(small(INDEX_FILE))
        weight_files = sorted(set(wm.values()))
    elif SINGLE_FILE in listing:
        wm, weight_files = None, [SINGLE_FILE]
    else:
        raise hub.no_safetensors(repo, commit, listing)
    missing = [f for f in weight_files if f not in listing]
    if missing:
        raise IntegrityError(f"{INDEX_FILE} names {missing[0]}, which the commit doesn't have")

    total, done, label = sum(listing[f].size for f in weight_files), 0, f"checking {repo}"
    tensors: dict[str, TensorEntry] = {}
    if progress:
        progress(0, total, label)
    for path in weight_files:
        check, scan = _FileCheck(listing[path]), _TensorHashes(listing[path].size, path)
        for piece in stream(path):
            check.update(piece)
            scan.update(piece)
            done += len(piece)
            if progress:
                progress(done, total, label)
        sha = check.finish()  # the whole file is what the Hub published, so every tensor hash below is too
        header, digests = scan.finish()
        for name, (dtype, shape, begin, end) in header.tensors.items():
            if name in tensors:
                raise IntegrityError(f"tensor {name} is in both {tensors[name].file} and {path}")
            tensors[name] = TensorEntry(path, dtype, shape, header.data_start + begin, end - begin, digests[name])
        files[path] = {"size": listing[path].size, "sha256": sha}
    if wm is not None and (set(wm) != set(tensors) or any(tensors[n].file != f for n, f in wm.items())):
        raise IntegrityError(f"{INDEX_FILE} doesn't match what the shards hold")
    for path in extra:
        if path in listing and path not in files:
            if listing[path].size > MAX_JSON:
                log.warning("not pinning %s: at %s it's no small file", path, _gb(listing[path].size))
                continue
            small(path)
    return Manifest(repo, commit, files, tensors)


def verify_tensor(manifest: Manifest, name: str, raw: bytes) -> TensorEntry:
    """Check a tensor's bytes, however they were fetched, against the manifest. Returns its entry."""
    e = manifest.tensors.get(name)
    if e is None:
        raise KeyError(f"the manifest for {manifest.repo} has no tensor {name}")
    if len(raw) != e.length:
        raise IntegrityError(f"{name}: got {len(raw)} bytes, the manifest says {e.length}")
    if hashlib.sha256(raw).hexdigest() != e.sha256:
        raise IntegrityError(f"{name}: these bytes aren't the ones the manifest pins; whoever served them altered "
                             "or corrupted them")
    return e


def fetch_file(manifest: Manifest, path: str, *, hub: Hub | None = None) -> bytes:
    """One of the manifest's small files (config.json, tokenizer.json ...), checked against it."""
    f = manifest.files.get(path)
    if f is None:
        raise KeyError(f"the manifest for {manifest.repo} doesn't pin {path}")
    if f["size"] > MAX_JSON:
        raise ValueError(f"{path} is {_gb(f['size'])}: fetch weights by tensor (fetch_tensors), not whole")
    data = (hub or Hub()).get_file(manifest.repo, manifest.commit, path, limit=max(f["size"], 1))
    if len(data) != f["size"] or hashlib.sha256(data).hexdigest() != f["sha256"]:
        raise IntegrityError(f"{path} isn't the file the manifest pins")
    return data


def fetch_tensor(repo: str, commit: str, name: str, *, hub: Hub | None = None) -> tuple[str, list[int], bytes]:
    """Hub.fetch_tensor with a default Hub: one tensor by range, found through the index. Unverified."""
    return (hub or Hub()).fetch_tensor(repo, commit, name)


# --- the cache --------------------------------------------------------------------------

def default_cache_dir() -> Path:
    return settings.home() / "cache" / "hub"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    for block in _read_local(path):
        h.update(block)
    return h.hexdigest()


class _RunWriter:
    """Splits one ranged download covering several back-to-back tensors into a .part file per tensor, hashing each
    as it goes. A finished tensor is renamed into the cache only if its hash is the manifest's; one that isn't is
    deleted and the download stops."""

    def __init__(self, cache: "TensorCache", folder: Path, run: list[TensorEntry], resumed: int):
        self.cache, self.folder, self.run, self.i, self.f = cache, folder, run, 0, None
        self._start(resumed)

    def _start(self, resumed: int) -> None:
        e = self.run[self.i]
        part = self.folder / f"{e.sha256}.part"
        self.h = hashlib.sha256()
        if resumed:
            for block in _read_local(part):
                self.h.update(block)
        self.f, self.left = open(part, "ab" if resumed else "wb"), e.length - resumed
        self._seal()

    def write(self, data: bytes) -> None:
        mv = memoryview(data)
        while mv:
            if self.f is None:
                raise IntegrityError("more bytes arrived than the tensors take")
            take = min(len(mv), self.left)
            self.f.write(mv[:take])
            self.h.update(mv[:take])
            self.left, mv = self.left - take, mv[take:]
            self._seal()

    def _seal(self) -> None:
        while self.f is not None and self.left == 0:
            e = self.run[self.i]
            self.f.close()
            self.f, part = None, self.folder / f"{e.sha256}.part"
            if self.h.hexdigest() != e.sha256:
                part.unlink(missing_ok=True)
                raise IntegrityError(f"a tensor of {e.file} downloaded with the wrong bytes (sha256 isn't the "
                                     "manifest's); discarded. Whoever served it altered or corrupted it")
            os.replace(part, self.folder / e.sha256)
            self.cache._verified.add(self.folder / e.sha256)
            self.i += 1
            if self.i < len(self.run):
                self._start(0)

    def close(self) -> None:
        if self.f is not None:
            self.f.close()  # the .part stays, so the next attempt resumes from it


class TensorCache:
    """Verified tensors on disk, at <root>/<commit>/<tensor sha256> (root: ~/.vibehash/cache/hub). A download in
    progress is <sha256>.part beside it and resumes from there, even after a restart. Only a file that hashed to its
    name is ever renamed into place, and one found damaged later is fetched again."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else default_cache_dir()
        self._verified: set[Path] = set()  # files hashed by this process, so a big one isn't re-hashed on every use

    def path(self, commit: str, sha256: str) -> Path:
        if not isinstance(sha256, str) or not _SHA256.match(sha256):
            raise ValueError("bad tensor hash")
        return self.root / check_commit(commit) / sha256

    def fetch(self, manifest: Manifest, names: Iterable[str], *, hub: Hub | None = None,
              progress: Progress | None = None, label: str | None = None) -> dict[str, Path]:
        """{name: path of its verified bytes}, downloading only what isn't cached yet, by range, straight from the
        Hub. Back-to-back tensors of one file come down in one request. Checks the disk has room first."""
        names = list(names)
        wanted: dict[str, TensorEntry] = {}
        for name in names:
            if name not in manifest.tensors:
                raise KeyError(f"the manifest for {manifest.repo} has no tensor {name}")
            wanted.setdefault(manifest.tensors[name].sha256, manifest.tensors[name])
        folder = self.root / manifest.commit
        folder.mkdir(parents=True, exist_ok=True)
        os.utime(folder)  # marks the model as in use, so prune_cache drops models nobody has used for longest first
        missing = []
        for sha, e in wanted.items():
            p = folder / sha
            if p.exists() and p not in self._verified:
                if p.stat().st_size == e.length and _file_sha256(p) == sha:
                    self._verified.add(p)
                else:
                    log.warning("cached tensor %s was damaged on disk; fetching it again", p)
                    p.unlink(missing_ok=True)
            if not p.exists():
                missing.append(e)
        if missing:
            self._download(manifest, missing, folder, hub or Hub(), progress,
                           label or f"fetching {len(missing)} tensor{'s' * (len(missing) != 1)} of {manifest.repo}")
        return {name: folder / manifest.tensors[name].sha256 for name in names}

    def _part(self, folder: Path, e: TensorEntry) -> int:
        part = folder / f"{e.sha256}.part"
        size = part.stat().st_size if part.exists() else 0
        if size > e.length:  # can't be a prefix of this tensor
            part.unlink()
            return 0
        return size

    def _download(self, manifest: Manifest, missing: list[TensorEntry], folder: Path, hub: Hub,
                  progress: Progress | None, label: str) -> None:
        have = {e.sha256: self._part(folder, e) for e in missing}
        need = sum(e.length - have[e.sha256] for e in missing)
        free = shutil.disk_usage(folder).free
        if need > free:
            raise DiskFullError(f"these weights need {_gb(need)} and {_gb(free)} is free in {folder}: free some space, "
                                "or set VIBEHASH_HOME to a folder on a bigger disk")
        done = 0
        if progress:
            progress(0, need, label)
        for run in _runs(missing, have):
            e0 = run[0]
            info = manifest.files[e0.file]
            f = hub.open(manifest.repo, manifest.commit, e0.file, info["size"], info["sha256"])
            start, end = e0.offset + have[e0.sha256], run[-1].offset + run[-1].length
            writer = _RunWriter(self, folder, run, have[e0.sha256])
            try:
                for piece in f.iter_range(start, end - start):
                    writer.write(piece)
                    done += len(piece)
                    if progress:
                        progress(done, need, label)
            except OSError as err:
                if getattr(err, "errno", None) == 28:  # ENOSPC: something else filled the disk meanwhile
                    raise DiskFullError(f"the disk holding {folder} filled up while fetching weights") from None
                raise
            finally:
                writer.close()


def _runs(missing: list[TensorEntry], have: dict[str, int]) -> list[list[TensorEntry]]:
    """Group tensors into runs that are back to back in one file, each fetched with one ranged request. A partial
    download can only continue at the start of a run."""
    runs: list[list[TensorEntry]] = []
    for e in sorted(missing, key=lambda e: (e.file, e.offset, e.length)):
        last = runs[-1][-1] if runs else None
        if last is not None and last.file == e.file and last.offset + last.length == e.offset and not have[e.sha256]:
            runs[-1].append(e)
        else:
            runs.append([e])
    return runs


def fetch_tensors(manifest: Manifest, names: Iterable[str], *, hub: Hub | None = None, cache: TensorCache | None = None,
                  progress: Progress | None = None, label: str | None = None) -> dict[str, Path]:
    """A worker's batch download: TensorCache.fetch with the default cache. The paths can be memory-mapped."""
    return (cache or TensorCache()).fetch(manifest, names, hub=hub, progress=progress, label=label)


def load_tensor(manifest: Manifest, name: str, *, hub: Hub | None = None, cache: TensorCache | None = None,
                progress: Progress | None = None) -> tuple[str, list[int], bytes]:
    """One tensor of a pinned model as (dtype string, shape, raw bytes): from the cache, or downloaded by range
    (exactly its bytes: the manifest says where they are, so no header read), always checked against the manifest."""
    path = fetch_tensors(manifest, [name], hub=hub, cache=cache, progress=progress)[name]
    raw = path.read_bytes()
    e = verify_tensor(manifest, name, raw)
    return e.dtype, list(e.shape), raw


def cache_usage(root: str | Path | None = None) -> dict[str, int]:
    """Bytes on disk per cached commit (partial downloads included), for a cache size limit."""
    root = Path(root) if root is not None else default_cache_dir()
    usage = {}
    for d in (root.iterdir() if root.is_dir() else []):
        if d.is_dir() and _COMMIT.match(d.name):
            usage[d.name] = sum(p.stat().st_size for p in d.iterdir() if p.is_file())
    return usage


def prune_cache(keep_commits: Iterable[str] = (), root: str | Path | None = None, max_bytes: int | None = None) -> int:
    """Delete cached models other than `keep_commits`, the least recently used first: all of them, or with
    `max_bytes` (a --cache-gb limit) just enough to bring the cache under it. Returns the bytes freed."""
    root = Path(root) if root is not None else default_cache_dir()
    keep, usage, freed = set(keep_commits), cache_usage(root), 0
    for commit in sorted(set(usage) - keep, key=lambda c: (root / c).stat().st_mtime):
        if max_bytes is not None and sum(usage.values()) - freed <= max_bytes:
            break
        shutil.rmtree(root / commit, ignore_errors=True)
        freed += usage[commit]
    return freed

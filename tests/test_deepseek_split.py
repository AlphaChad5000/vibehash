"""vibehash/deepseek_split.py: the prep cut into units that miners run, connected by stored hand-offs.

- PlanTests: what each piece of DeepSeek-V4.1-Flash reads and writes, the work graph, and the node's Line
  (claims and their deadlines, optimistic consumption, rollback, deletion of what everyone has read).
- CodecTests: hand-offs decode to exactly what was encoded, and a commitment doesn't depend on the compression.
- PresignTests: the S3 signature is AWS's own worked example's.
- StoreTests, against FakeS3 (an S3 bucket on localhost that checks every URL's signature): big objects go up in
  parts and come down in ranges, through a link that drops every third request; an upload cut off part way resumes
  from its journal, sending only the parts not yet acknowledged; a signed URL can't be bent to another method or part.
- SplitTests (small config, CPU): a prep split into pieces of 1 and 2 layers, the engram table in several row
  ranges, writes exactly deepseek_prep's records and Merkle root, whatever order its units run in, and through the
  bucket too. No host needs trusting: an altered hand-off is caught and another host's copy used, and a withheld one
  is made again. A replay agrees with an honest unit, and finds the layer a bent one disagrees at, with an agreed
  state entering it.
"""
import datetime
import hashlib
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import torch
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise ImportError
    from vibehash import deepseek, deepseek_prep as prep, deepseek_split as split, deepseek_top as top
    from tests.test_deepseek import tiny_config, token_map_for
    from tests.test_deepseek_prep import dataset
except ImportError:
    torch = None


class FakeS3:
    """An S3 bucket on localhost, as much of one as PresignedStore uses: objects with ranged reads, multipart
    uploads, and every URL's signature checked (a 403 otherwise). `fail(method, query)` -> True answers that request
    with a 500; `log` lists what was asked: (method, key, sub-resource query, Range)."""

    def __init__(self, access="AKIDFAKE", secret="fake-secret", region="test"):
        self.access, self.secret, self.region = access, secret, region
        self.objects, self.uploads, self.log = {}, {}, []
        self.fail = lambda method, query: False
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                fake.handle(self, "GET")

            def do_PUT(self):
                fake.handle(self, "PUT")

            def do_POST(self):
                fake.handle(self, "POST")

            def do_HEAD(self):
                fake.handle(self, "HEAD")

            def do_DELETE(self):
                fake.handle(self, "DELETE")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.host = f"127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def signer(self, **kw):
        return split.Signer(self.host, self.region, self.access, self.secret, scheme="http", **kw)

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def signed(self, method, key, q):
        sub = {k: v for k, v in q.items() if not k.startswith("X-Amz-")}
        try:
            when = datetime.datetime.strptime(q["X-Amz-Date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=datetime.timezone.utc)
            url = split.presign(method, self.host, key, self.region, self.access, self.secret,
                                int(q["X-Amz-Expires"]), when, sub, "http")
        except (KeyError, ValueError):
            return False
        return url.endswith("X-Amz-Signature=" + q.get("X-Amz-Signature", "-"))

    def handle(self, h, method):
        url = urllib.parse.urlsplit(h.path)
        q = dict(urllib.parse.parse_qsl(url.query, keep_blank_values=True))
        key = urllib.parse.unquote(url.path[1:])
        body = h.rfile.read(int(h.headers.get("Content-Length") or 0))
        sub = {k: v for k, v in q.items() if not k.startswith("X-Amz-")}
        if not self.signed(method, key, q):
            return self.send(h, 403)
        self.log.append((method, key, sub, h.headers.get("Range")))
        if self.fail(method, sub):
            return self.send(h, 500)
        if method == "PUT" and "partNumber" in sub:
            parts = self.uploads.get(sub.get("uploadId"))
            if parts is None:
                return self.send(h, 404)
            parts[int(sub["partNumber"])] = body
            return self.send(h, 200, headers={"ETag": '"%s"' % hashlib.md5(body).hexdigest()})
        if method == "PUT":
            self.objects[key] = body
            return self.send(h, 200)
        if method == "POST" and "uploads" in sub:
            upload = f"upload-{len(self.log)}"
            self.uploads[upload] = {}
            return self.send(h, 200, ('<?xml version="1.0" encoding="UTF-8"?><InitiateMultipartUploadResult '
                                      'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Bucket>b</Bucket>'
                                      f'<Key>{key}</Key><UploadId>{upload}</UploadId>'
                                      '</InitiateMultipartUploadResult>').encode())
        if method == "POST":
            parts = self.uploads.get(sub.get("uploadId"))
            if parts is None:
                return self.send(h, 404)
            listed = [(int(p.find("PartNumber").text), p.find("ETag").text)
                      for p in ElementTree.fromstring(body).iter("Part")]
            if [n for n, _ in listed] != sorted(parts) or \
                    any(e != '"%s"' % hashlib.md5(parts[n]).hexdigest() for n, e in listed):
                return self.send(h, 400, b"<Error><Code>InvalidPart</Code></Error>")
            self.objects[key] = b"".join(parts[n] for n, _ in listed)
            del self.uploads[sub["uploadId"]]
            return self.send(h, 200, b"<CompleteMultipartUploadResult><Key>k</Key></CompleteMultipartUploadResult>")
        data = self.objects.get(key)
        if data is None:
            return self.send(h, 404)
        if method == "HEAD":
            return self.send(h, 200, headers={"Content-Length": str(len(data))}, head=True)
        if method == "DELETE":
            del self.objects[key]
            return self.send(h, 204)
        if h.headers.get("Range"):
            lo, hi = (int(v) for v in h.headers["Range"].removeprefix("bytes=").split("-"))
            hi = min(hi, len(data) - 1)
            return self.send(h, 206, data[lo:hi + 1], {"Content-Range": f"bytes {lo}-{hi}/{len(data)}"})
        return self.send(h, 200, data)

    @staticmethod
    def send(h, code, body=b"", headers=None, head=False):
        h.send_response(code)
        for k, v in (headers or {}).items():
            h.send_header(k, v)
        if not head:
            h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        if not head:
            h.wfile.write(body)


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class PlanTests(unittest.TestCase):
    def setUp(self):
        self.cut = top.Cut(deepseek.Config.flash(), 36)

    def test_pieces(self):
        self.assertEqual(split.pieces(36, 9), [(0, 9), (9, 18), (18, 27), (27, 36)])
        self.assertEqual(split.pieces(36, 8)[-1], (32, 36))
        self.assertEqual(split.pieces(36, bounds=[2, 8, 20]), [(0, 2), (2, 8), (8, 20), (20, 36)])

    def test_what_pieces_read_and_write(self):
        cut = self.cut
        self.assertEqual(split.needs(cut, 0, 4), {"engram": [1]})
        self.assertEqual(split.emits(cut, 0, 4), {"compress_kv": 2, "picks": 2})  # layers 4-7 reuse layer 2's
        self.assertEqual(split.needs(cut, 4, 8), {"compress_kv": 2, "picks": 2})
        self.assertEqual(split.needs(cut, 12, 16), {"compress_kv": 8, "picks": 8, "engram": [14]})
        self.assertEqual(split.emits(cut, 20, 24), {"compress_kv": 20, "index_k": 20, "candidates": 20})
        self.assertEqual(split.needs(cut, 24, 28), {"compress_kv": 20, "index_k": 20, "candidates": 20})
        self.assertEqual(split.needs(cut, 21, 24), {"compress_kv": 20, "picks": 20})
        self.assertEqual(split.emits(cut, 32, 36), {})  # the last piece: into the records
        self.assertEqual(split.needs(cut, 32, 36), {"compress_kv": 20, "index_k": 20, "candidates": 20})

    def test_row_ranges(self):
        cfg = self.cut.cfg
        ranges = split.row_ranges(cfg, 1, 15)
        self.assertEqual(len(ranges), 15)
        self.assertEqual((ranges[0][0], ranges[-1][1]), (0, cfg.engram_num_embeddings[0]))
        self.assertTrue(all(lo % split.ROWS_PER_PIECE == 0 for lo, _ in ranges))
        self.assertLess((ranges[0][1] - ranges[0][0]) * 264 / 1e9, 7.0)  # GB a range: a 256-byte row, 8 scales

    def test_line(self):
        graph = split.plan(self.cut, [[0], [1]], split.pieces(36, 9))
        line = split.Line(graph)
        self.assertEqual({u.kind for u in line.claimable()}, {"rows"})  # layer 1 needs its rows first
        for u in line.claimable():
            line.claim(u.name)
            line.done(u.name)
        first = [u.name for u in line.claimable()]
        self.assertEqual(first, ["layers-00-09-c00000", "layers-00-09-c00001"])
        self.assertEqual(line.claimable(holds=range(9, 18)), [])
        line.claim(first[0])
        line.done(first[0])  # optimistic: the next piece can read it before it's accepted
        nxt = [u.name for u in line.claimable(holds=range(9, 18))]
        self.assertEqual(nxt, ["layers-09-18-c00000"])
        line.claim(nxt[0])
        line.done(nxt[0])
        back = line.overturn(first[0])
        self.assertEqual(back[0], first[0])
        self.assertEqual(set(back[1:]), {nxt[0], "layers-18-27-c00000", "layers-27-36-c00000"})  # chunk 0 only
        self.assertEqual(line.state[nxt[0]], "pending")
        for name in (first[0], nxt[0]):
            line.claim(name)
            line.done(name)
        line.accept(first[0])
        gone = line.accept(nxt[0])
        self.assertIn(split.stream_name(0, 9), gone)  # everyone who reads it is past their window
        self.assertNotIn(split.rows_name(0, 1, 0), gone)  # its rows unit isn't accepted yet

    def test_deadlines(self):
        line = split.Line(split.plan(self.cut, [[0]], split.pieces(36, 4)))
        name = line.claimable()[0].name
        # a 64K-token chunk through 4 layers: 1.85 GB in, 1.83 GB out, at the slowest link the network takes
        seconds = split.deadline(1.85e9, 1.83e9, 4 * 65536)
        self.assertGreater(seconds, 3600)
        line.claim(name, now=100.0, seconds=seconds)
        self.assertEqual(line.expire(100.0 + 3600), [])
        self.assertEqual(line.expire(100.0 + seconds + 1), [name])  # a vanished miner's unit goes to another
        self.assertEqual(line.state[name], "pending")
        self.assertEqual(split.deadline(0, 0, 0), 600.0)

    def test_order(self):
        graph = split.plan(self.cut, [[0], [1], [2]], split.pieces(36, 4))
        seen = set()
        for u in graph.order():
            self.assertTrue(graph.deps(u.name) <= seen, u.name)
            seen.add(u.name)
        self.assertEqual(len(seen), 30 + 9 * 3)


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class CodecTests(unittest.TestCase):
    def test_round_trip(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(3, 7, 4, 64, generator=g).bfloat16()
        tensors = {"x": x, "pre": torch.rand(3, 7, 4, generator=g), "picks": torch.randint(-1, 9, (7, 5), dtype=torch.int16),
                   "mask": torch.rand(7, 9, generator=g) < 0.5, "fp8": x.float().to(torch.float8_e4m3fn),
                   "empty": torch.zeros(0, 32, dtype=torch.uint8), "ids": torch.arange(5, dtype=torch.int64)}
        back = split.decode(split.encode(tensors))
        self.assertEqual(list(back), list(tensors))
        for k, t in tensors.items():
            self.assertEqual(back[k].dtype, t.dtype, k)
            self.assertTrue(torch.equal(back[k].view(torch.uint8) if t.dtype == torch.float8_e4m3fn else back[k],
                                        t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t), k)
        self.assertEqual(split.digest(back), split.digest(tensors))

    def test_digest_ignores_the_codec(self):
        x = {"x": torch.randn(2, 64).bfloat16()}
        a = split.encode(x)
        old, split.CODEC = split.CODEC, "zlib"
        try:
            b = split.encode(x)
        finally:
            split.CODEC = old
        if old != "zlib":
            self.assertNotEqual(a, b)
        self.assertEqual(split.digest(split.decode(a)), split.digest(split.decode(b)))


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class PresignTests(unittest.TestCase):
    def test_aws_example(self):
        # AWS's worked example of a presigned GET (Signature Version 4, query-string authentication).
        when = datetime.datetime(2013, 5, 24, tzinfo=datetime.timezone.utc)
        url = split.presign("GET", "examplebucket.s3.amazonaws.com", "test.txt", "us-east-1", "AKIAIOSFODNN7EXAMPLE",
                            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", 86400, when)
        self.assertTrue(url.startswith("https://examplebucket.s3.amazonaws.com/test.txt?X-Amz-Algorithm="))
        self.assertTrue(url.endswith("X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404"))


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class StoreTests(unittest.TestCase):
    DATA = bytes(range(256)) * 41  # 10,496 bytes: 11 parts of 1 KiB

    def setUp(self):
        self.s3 = FakeS3()
        self.addCleanup(self.s3.close)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.journal = Path(tmp.name) / "journal"

    def store(self, **kw):
        return split.PresignedStore(self.s3.signer(), part=1024, journal=self.journal, wait=lambda s: None, **kw)

    def parts(self):
        return [int(e[2]["partNumber"]) for e in self.s3.log if "partNumber" in e[2]]

    def test_small_and_missing(self):
        st = self.store()
        st.put("c00000/x", b"hello")
        self.assertEqual(st.get("c00000/x"), b"hello")
        self.assertTrue(st.has("c00000/x"))
        st.delete("c00000/x")
        self.assertFalse(st.has("c00000/x"))
        with self.assertRaises(OSError) as e:  # what _get takes as "this host hasn't got it"
            st.get("c00000/x")
        e.exception.close()
        st.delete("c00000/x")  # already gone: fine
        self.assertEqual(self.parts(), [])

    def test_parts_and_ranges(self):
        st = self.store()
        st.put("big", self.DATA)
        self.assertEqual(self.parts(), list(range(1, 12)))
        self.assertEqual(self.s3.objects["big"], self.DATA)
        self.assertEqual(list(self.journal.glob("*.json")), [])  # finished: nothing left to resume
        self.s3.log.clear()
        self.assertEqual(st.get("big"), self.DATA)
        self.assertEqual(len([e for e in self.s3.log if e[0] == "GET" and e[3]]), 11)

    def test_flaky_link(self):
        count = [0]

        def fail(method, query):  # every third request fails
            count[0] += 1
            return count[0] % 3 == 0
        self.s3.fail = fail
        st = self.store()
        st.put("big", self.DATA)
        self.assertEqual(st.get("big"), self.DATA)

    def test_resumes_after_a_crash(self):
        def fail(method, query):  # the link dies after four parts
            return "partNumber" in query and int(query["partNumber"]) > 4
        self.s3.fail = fail
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.store(tries=2).put("big", self.DATA)
        e.exception.close()
        self.s3.fail = lambda method, query: False
        self.s3.log.clear()
        self.store().put("big", self.DATA)  # another process, the same journal
        self.assertEqual(self.parts(), list(range(5, 12)))  # only the parts not yet acknowledged
        self.assertEqual(self.s3.objects["big"], self.DATA)

    def test_restarts_a_lost_upload(self):
        self.s3.fail = lambda method, query: "partNumber" in query and int(query["partNumber"]) > 3
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.store(tries=1).put("big", self.DATA)
        e.exception.close()
        self.s3.fail = lambda method, query: False
        self.s3.uploads.clear()  # the bucket dropped the unfinished upload
        self.s3.log.clear()
        self.store().put("big", self.DATA)
        self.assertEqual(self.parts(), [4] + list(range(1, 12)))  # part 4 finds it gone, then all over again
        self.assertEqual(self.s3.objects["big"], self.DATA)

    def test_signature_binds_method_object_and_part(self):
        sign = self.s3.signer()
        bent = [urllib.request.Request(sign("GET", "x"), data=b"evil", method="PUT"),
                urllib.request.Request(sign("PUT", "x").replace("/x?", "/y?"), data=b"evil", method="PUT"),
                urllib.request.Request(sign("PUT", "x", {"partNumber": "1", "uploadId": "u"})
                                       .replace("partNumber=1", "partNumber=2"), data=b"evil", method="PUT")]
        for req in bent:
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(req)
            self.assertEqual(e.exception.code, 403)
            e.exception.close()
        self.assertEqual(self.s3.objects, {})


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class SplitTests(unittest.TestCase):
    """deepseek_prep's PrepTests setting: cut at 4, an engram layer (1) below it; compressed keys, index keys and
    candidate blocks cross into the records; layer 2 reuses layer 1's keys and picks across a piece boundary."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.cfg = tiny_config()
        self.state = deepseek.random_weights(self.cfg, 3)
        self.cut = top.Cut(self.cfg, 4)
        prep.save_dataset(self.tmp / "data.npz", dataset(self.cfg))
        self.seqs = prep.load_dataset(self.tmp / "data.npz")
        self.tmap = token_map_for(self.cfg)
        self.chunks = prep.blocks_of(self.seqs, 40)
        self.whole = prep.run(prep.LocalSource(self.state), prep.LocalUploader(self.tmp / "whole" / "repo"), self.seqs,
                              self.cut, self.tmp / "whole" / "work", block_tokens=40, token_map=self.tmap)

    def split(self, name, bounds, ranges=3):
        store = split.LocalStore(self.tmp / name / "store")
        m = split.run_local(prep.LocalSource(self.state), store, self.seqs, self.chunks, self.cut, bounds, self.tmap,
                            self.tmp / name / "work", ranges=ranges, align=1)
        return m, store

    def assert_same(self, m, store):
        self.assertEqual(m["root"], self.whole["root"])
        self.assertEqual([b["leaves"] for b in m["blocks"]], [b["leaves"] for b in self.whole["blocks"]])
        repo = self.tmp / "whole" / "repo"
        for c in range(len(self.chunks)):
            bin_name, meta_name = split.block_names(c)
            self.assertEqual(store.get(bin_name), (repo / bin_name).read_bytes(), c)
            self.assertEqual(store.get(meta_name), (repo / meta_name).read_bytes(), c)

    def test_one_layer_pieces(self):
        self.assertGreater(len(self.chunks), 1)
        m, store = self.split("one", split.pieces(4, 1))
        self.assert_same(m, store)
        self.assertGreater(len(split.decode(store.get(split.side_name(0, "picks", 1)))), 0)  # crossed into layer 2

    def test_two_layer_pieces(self):
        self.assert_same(*self.split("two", split.pieces(4, 2), ranges=2))

    def test_any_order(self):
        """Units claimed as a Line offers them, one chunk racing ahead of the others, and one unit overturned and
        redone after the next piece read its output: the same records and root."""
        bounds = split.pieces(4, bounds=[0, 1, 3])
        graph = split.plan(self.cut, self.chunks, bounds, 3, align=1)
        line, store = split.Line(graph), split.LocalStore(self.tmp / "order" / "store")
        work, src, last = self.tmp / "order" / "work", prep.LocalSource(self.state), {}
        overturned = False  # until then, every window is still open: nothing is accepted
        while any(s != "accepted" for s in line.state.values()):
            ready = line.claimable()
            u = max(ready, key=lambda u: (u.kind == "rows", u.chunk, u.a))  # the last chunk first, deepest first
            line.claim(u.name)
            if u.kind == "rows":
                split.run_rows(src, store, self.seqs, self.chunks, self.cut, u.a, u.b, self.tmap, work, 3, align=1)
            else:
                commit = split.run_piece(src, store, self.seqs, self.chunks, self.cut, u, self.tmap, work, ranges=3,
                                         align=1)
                if u.b == self.cut.first:
                    last[u.chunk] = commit
            line.done(u.name)
            if u.kind == "layers" and u.a == 1 and not overturned:
                overturned = True
                line.overturn(graph.producer[split.stream_name(u.chunk, 1)])  # its input's maker: redo both
                continue
            if not overturned:
                continue
            for name, s in line.state.items():  # accept whatever is done and whose inputs are accepted
                if s == "done" and all(line.state[d] == "accepted" for d in graph.deps(name)):
                    for gone in line.accept(name):
                        store.delete(gone)
        m = split.manifest(self.cut, self.seqs, self.chunks, list(last.values()))
        self.assert_same(m, store)
        self.assertFalse(store.has(split.stream_name(0, 1)))  # the frontier only: read by everyone, then deleted
        self.assertTrue(store.has(split.block_names(0)[0]))  # the records stay

    def test_hosts_need_no_trust(self):
        """A host that alters a hand-off is caught, and another host's copy is used instead; a host that withholds
        one costs only a rerun of the unit that wrote it, which anyone holding its layers can do, to the bit."""
        bounds = split.pieces(4, 2)
        _, bucket = self.split("trust", bounds)
        graph = split.plan(self.cut, self.chunks, bounds, 3, align=1)
        src, work = prep.LocalSource(self.state), self.tmp / "trust" / "work"
        made = json.loads(bucket.get(split.commit_name(0, 0)))["outputs"]
        name, records = split.stream_name(0, 2), bucket.get(split.block_names(0)[0])
        own = split.LocalStore(self.tmp / "trust" / "own")  # the writer's own copy
        own.put(name, bucket.get(name))
        bad = split.decode(bucket.get(name))
        bad["0.x"] = bad["0.x"].clone()
        bad["0.x"].view(-1)[0] += 1
        bucket.put(name, split.encode(bad))

        def last(store):
            return split.run_piece(src, store, self.seqs, self.chunks, self.cut, graph.units["layers-02-04-c00000"],
                                   self.tmap, work, ranges=3, align=1, want=made)
        with self.assertRaises(split.Tampered):
            last(bucket)
        last(split.Mirrors(bucket, own))  # the altered copy skipped
        self.assertEqual(bucket.get(split.block_names(0)[0]), records)
        bucket.delete(name)
        with self.assertRaises(split.Missing):  # withheld everywhere
            last(bucket)
        line = split.Line(graph)
        line.redo("layers-00-02-c00000")
        self.assertEqual(line.state["layers-00-02-c00000"], "pending")
        split.run_piece(src, bucket, self.seqs, self.chunks, self.cut, graph.units["layers-00-02-c00000"], self.tmap,
                        work, ranges=3, align=1)  # rerun by anyone holding layers 0-1
        self.assertEqual(split.digest(split.decode(bucket.get(name))), made[name])
        last(bucket)
        self.assertEqual(bucket.get(split.block_names(0)[0]), records)

    def test_through_the_bucket(self):
        s3 = FakeS3()
        self.addCleanup(s3.close)
        store = split.PresignedStore(s3.signer(prefix="job-1/"), part=4096, journal=self.tmp / "journal",
                                     wait=lambda s: None)
        m = split.run_local(prep.LocalSource(self.state), store, self.seqs, self.chunks, self.cut, split.pieces(4, 2),
                            self.tmap, self.tmp / "bucket" / "work", ranges=3, align=1)
        self.assert_same(m, store)
        self.assertTrue(any("partNumber" in e[2] for e in s3.log))  # streams big enough to go up in parts
        self.assertTrue(all(k.startswith("job-1/") for k in s3.objects))

    def test_replay_answers_a_challenge(self):
        bounds = split.pieces(4, 2)
        _, store = self.split("replay", bounds)
        unit = split.plan(self.cut, self.chunks, bounds, 3, align=1).units["layers-02-04-c00000"]
        commit = json.loads(store.get(split.commit_name(0, 2)))
        kw = dict(token_map=self.tmap, work=self.tmp / "replay" / "work", ranges=3, align=1,
                  want=json.loads(store.get(split.commit_name(0, 0)))["outputs"])
        src, j = prep.LocalSource(self.state), len(self.chunks[0]) - 1
        args = (src, store, self.seqs, self.chunks, self.cut, unit)
        self.assertEqual(split.replay(*args, commit, **kw), {"agrees": True, "disagree": [], "outputs": []})
        stats = commit["stats"]
        self.assertEqual(stats["tokens"], 2 * sum(len(self.seqs[i][0]) for i in self.chunks[0]))
        self.assertGreater(stats["read_bytes"], 0)
        self.assertIn("peak_gpu_gb", stats)
        bent = json.loads(json.dumps(commit))  # a holder who bent layer 3 of the chunk's last sequence
        bent["layer_hashes"][j][1] = "00" * 32
        bent["leaves"][j] = "00" * 32
        found = split.replay(*args, bent, **kw)
        self.assertFalse(found["agrees"])
        self.assertEqual(found["disagree"], [(j, 3)])
        self.assertEqual(found["outputs"], ["leaves"])
        state, h = split.state_before(*args, j, 3, **kw)  # what both sides hand the referee
        self.assertEqual(h, commit["layer_hashes"][j][0])  # agreed: layer 2's committed hash
        self.assertTrue({"x", "pre"} <= set(state))
        first, h = split.state_before(*args, j, 2, **kw)
        self.assertIsNone(h)  # the piece's input: a checked object
        self.assertTrue(torch.equal(first["x"], split.decode(store.get(split.stream_name(0, 2)))[f"{j}.x"]))

    def test_commitment_chain(self):
        _, store = self.split("chain", split.pieces(4, 1))
        made = json.loads(store.get(split.commit_name(0, 1)))["outputs"]
        read = json.loads(store.get(split.commit_name(0, 2)))["inputs"]
        for name in (split.stream_name(0, 2), split.side_name(0, "compress_kv", 1)):
            self.assertEqual(read[name], made[name], name)  # what layer 2 read is what layer 1 wrote


if __name__ == "__main__":
    unittest.main()

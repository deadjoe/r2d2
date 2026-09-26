import contextlib
import hashlib
import json

import pytest

import r2d2.fetch as fetch

BLOB = bytes(range(256)) * 400  # 100 KB


class Response:
    def __init__(self, status, body):
        self.status_code, self.body = status, body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_bytes(self, size):
        yield from (self.body[i:i + size] for i in range(0, len(self.body), size))


class Server:
    """Serves BLOB, honours Range unless told not to, and can cut a transfer short."""

    def __init__(self, ranges=True, cut=None, body=BLOB):
        self.ranges, self.cut, self.body, self.requests = ranges, cut, body, []

    @contextlib.contextmanager
    def stream(self, method, url, headers=None, **kwargs):
        headers = headers or {}
        self.requests.append(headers.get("range"))
        start = int(headers["range"][6:-1]) if self.ranges and "range" in headers else 0
        body = self.body[start:]
        if self.cut is not None:
            body, self.cut = body[:self.cut], None
        yield Response(206 if start else 200, body)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    entry = {"path": "models/repo/blob.bin", "sha256": hashlib.sha256(BLOB).hexdigest(), "bytes": len(BLOB),
             "source_repo": "x/y", "source_revision": "r", "source_url": "https://example.invalid/blob.bin"}
    manifest.write_text(json.dumps({"sets": {"default": [entry["path"]]}, "files": [entry]}))
    monkeypatch.setattr(fetch, "MANIFEST", manifest)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    root = tmp_path / "models"

    def serve(server):
        import httpx
        monkeypatch.setattr(httpx, "stream", server.stream)
        return server
    return root, root / "repo" / "blob.bin", serve


def test_an_interrupted_download_resumes_with_a_range_request(setup):
    root, dest, serve = setup
    server = serve(Server(cut=30000))
    assert fetch.fetch(["default"], root)
    assert dest.read_bytes() == BLOB
    assert server.requests == [None, "bytes=30000-"]
    assert not dest.with_name("blob.bin.part").exists()


def test_a_server_that_ignores_the_range_starts_the_file_over(setup):
    root, dest, serve = setup
    serve(Server(ranges=False))
    dest.parent.mkdir(parents=True)
    dest.with_name("blob.bin.part").write_bytes(b"junk" * 100)
    assert fetch.fetch(["default"], root)
    assert dest.read_bytes() == BLOB


def test_a_wrong_file_is_replaced_and_a_verified_one_is_not_rehashed(setup, monkeypatch):
    root, dest, serve = setup
    serve(Server())
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"x" * len(BLOB))
    assert fetch.fetch(["default"], root)
    assert dest.read_bytes() == BLOB
    monkeypatch.setattr(fetch, "sha256", lambda path: pytest.fail("rehashed a stamped file"))
    assert fetch.fetch(["default"], root, check_only=True)


def test_a_download_takes_its_real_name_only_after_its_hash_matches(setup, monkeypatch):
    root, dest, serve = setup
    serve(Server())
    real = fetch.sha256

    def hashed(path):
        assert not dest.exists(), "the app could offer an unverified file"
        return real(path)
    monkeypatch.setattr(fetch, "sha256", hashed)
    assert fetch.fetch(["default"], root)
    assert dest.read_bytes() == BLOB


def test_bad_bytes_from_the_server_fail_after_the_retries(setup):
    root, dest, serve = setup
    serve(Server(body=bytes(len(BLOB))))
    assert not fetch.fetch(["default"], root, attempts=2)
    assert not dest.exists()
    assert not dest.with_name("blob.bin.part").exists()


def test_check_only_reports_a_missing_file_without_downloading(setup):
    root, dest, serve = setup
    server = serve(Server())
    assert not fetch.fetch(["default"], root, check_only=True)
    assert server.requests == []

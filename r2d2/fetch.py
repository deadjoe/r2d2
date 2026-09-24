"""Download the model files a set needs and verify them against models-manifest.json.

    python -m r2d2.fetch                 # the default set: Q8_0 recognition + Hy-MT2
    python -m r2d2.fetch --set default,q4
    python -m r2d2.fetch --check         # verify only, download nothing

Every file comes from its pinned Hugging Face revision. A download goes to
<file>.part and an interrupted one resumes from there with an HTTP range
request, in this run or the next; a failed one is retried with backoff; a file
whose size or SHA-256 differs from the manifest is deleted and fetched again. A file hashed once is recorded with its size and mtime in
<root>/.r2d2-verified.json, so later starts check it in milliseconds; delete
that file to force a full re-hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models-manifest.json"
STAMP = ".r2d2-verified.json"


def log(message):
    print(f"[fetch] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


class Stamps:
    """Size, mtime and hash of files already verified, keyed by manifest path."""

    def __init__(self, root: Path):
        self.path = root / STAMP
        try:
            self.data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.data = {}

    def fresh(self, key: str, dest: Path, entry: dict) -> bool:
        seen = self.data.get(key)
        try:
            st = dest.stat()
        except OSError:
            return False
        return bool(seen) and seen == {"sha256": entry["sha256"], "size": st.st_size, "mtime_ns": st.st_mtime_ns}

    def record(self, key: str, dest: Path, entry: dict):
        st = dest.stat()
        self.data[key] = {"sha256": entry["sha256"], "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
        tmp.replace(self.path)


def verify(dest: Path, entry: dict) -> str | None:
    """None when the file matches the manifest, else what is wrong."""
    if not dest.is_file():
        return "missing"
    size = dest.stat().st_size
    if size != entry["bytes"]:
        return f"size {size}, expected {entry['bytes']}"
    actual = sha256(dest)
    return None if actual == entry["sha256"] else f"sha256 {actual[:12]}…, expected {entry['sha256'][:12]}…"


def download(entry: dict, dest: Path):
    """Fetch the pinned URL into <dest>.part, continuing from whatever is there."""
    import httpx

    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    if have > entry["bytes"]:
        part.unlink()
        have = 0
    headers = {"user-agent": "r2d2-fetch (+https://github.com/deadjoe/r2d2)"}
    if have:
        headers["range"] = f"bytes={have}-"
    timeout = httpx.Timeout(60, connect=30)
    with httpx.stream("GET", entry["source_url"], headers=headers, timeout=timeout,
                      follow_redirects=True) as response:
        if have and response.status_code == 200:
            have = 0  # the server ignored the range: start over
        elif have and response.status_code == 206:
            log(f"resume    {dest.name} from {gb(have)}")
        response.raise_for_status()
        last = time.monotonic()
        with part.open("ab" if have else "wb") as f:
            for chunk in response.iter_bytes(8 << 20):
                f.write(chunk)
                have += len(chunk)
                if time.monotonic() - last > 30:
                    last = time.monotonic()
                    log(f"          {dest.name} {gb(have)} / {gb(entry['bytes'])}")
    if have == entry["bytes"]:
        part.replace(dest)


def fetch(sets: list[str], root: Path, attempts: int = 6, check_only: bool = False) -> bool:
    manifest = json.loads(MANIFEST.read_text())
    entries = {f["path"]: f for f in manifest["files"]}
    wanted = []
    for name in sets:
        if name not in manifest["sets"]:
            raise SystemExit(f"unknown set {name!r}; known: {', '.join(manifest['sets'])}")
        wanted += [p for p in manifest["sets"][name] if p not in wanted]
    root.mkdir(parents=True, exist_ok=True)
    stamps = Stamps(root)
    started, fetched, failed = time.monotonic(), 0, []
    for key in wanted:
        entry = entries[key]
        dest = root / key.removeprefix("models/")
        label = f"{key.removeprefix('models/')} ({gb(entry['bytes'])})"
        if stamps.fresh(key, dest, entry):
            log(f"ok        {label}, verified before")
            continue
        t = time.monotonic()
        problem = verify(dest, entry)
        if problem is None:
            stamps.record(key, dest, entry)
            log(f"ok        {label}, sha256 checked in {time.monotonic() - t:.1f} s")
            continue
        if check_only:
            log(f"FAILED    {label}: {problem}")
            failed.append(key)
            continue
        if dest.exists():
            log(f"replace   {label}: {problem}")
            dest.unlink()
        for attempt in range(1, attempts + 1):
            t = time.monotonic()
            log(f"download  {label}, attempt {attempt}/{attempts}")
            try:
                download(entry, dest)
                problem = verify(dest, entry)
            except Exception as exc:  # network, HTTP 5xx, disk: all worth another try
                problem = f"{type(exc).__name__}: {str(exc)[:200]}"
            if problem is None:
                stamps.record(key, dest, entry)
                fetched += entry["bytes"]
                log(f"ok        {label}, downloaded and verified in {time.monotonic() - t:.0f} s")
                break
            log(f"retry     {label}: {problem}")
            if dest.exists():
                dest.unlink()  # complete but wrong: start that file over
            if attempt < attempts:
                time.sleep(min(60, 5 * 2 ** (attempt - 1)))
        else:
            failed.append(key)
    total = sum(entries[k]["bytes"] for k in wanted)
    log(f"{len(wanted) - len(failed)}/{len(wanted)} files ready ({gb(total)}), "
        f"{gb(fetched)} downloaded, {time.monotonic() - started:.0f} s"
        + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    return not failed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download and verify the model files a set needs.")
    parser.add_argument("--set", default="default", help="comma-separated sets from the manifest")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("R2D2_MODELS", ROOT / "models")))
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--check", action="store_true", help="verify only")
    args = parser.parse_args(argv)
    ok = fetch([s for s in args.set.split(",") if s], args.root, args.attempts, args.check)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

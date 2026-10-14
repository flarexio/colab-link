"""Minimal Google Drive client for the Colab VM (stdlib only).

Uses the token the runner uploads (DRIVE_TOKEN_FILE, default
/content/run/.drive_token) and the quota project in DRIVE_QUOTA_PROJECT.
The token only has drive.file scope, so it sees just the files this OAuth
client created. Files live flat inside named top-level folders: caches in
DRIVE_ROOT (default "colab-link-cache"); callers pass folder= for others
(e.g. "colab-link-jobs" for job inputs/outputs).

    python drive_vm.py ls
    python drive_vm.py put LOCAL NAME
    python drive_vm.py get NAME LOCAL
    python drive_vm.py rm NAME
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://www.googleapis.com/drive/v3"
UPLOAD = "https://www.googleapis.com/upload/drive/v3"
FOLDER_MIME = "application/vnd.google-apps.folder"
CHUNK = 64 * 1024 * 1024


def _token() -> str:
    # Re-read every call: the runner swaps in a fresh token file during long jobs.
    if os.environ.get("DRIVE_TOKEN"):
        return os.environ["DRIVE_TOKEN"]
    return open(os.environ.get("DRIVE_TOKEN_FILE", "/content/run/.drive_token")).read().strip()


def _req(method: str, url: str, data: bytes | None = None, headers: dict | None = None,
         retries: int = 5) -> urllib.request.addinfourl:
    h = {"Authorization": f"Bearer {_token()}",
         "x-goog-user-project": os.environ["DRIVE_QUOTA_PROJECT"], **(headers or {})}
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=h, method=method),
                                          timeout=300)
        except urllib.error.HTTPError as e:
            # 308 = resumable upload "keep going"; let the caller see it.
            if e.code == 308:
                return e
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"{method} {url.split('?')[0]} -> {e.code}: {e.read()[:300]!r}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _json(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    hdr = {"Content-Type": "application/json"} if body is not None else {}
    return json.loads(_req(method, url, data, hdr).read() or b"{}")


def _q(q: str) -> list[dict]:
    url = f"{API}/files?" + urllib.parse.urlencode(
        {"q": q, "fields": "files(id,name,size,modifiedTime)", "pageSize": 1000, "spaces": "drive"})
    return _json("GET", url)["files"]


_folders: dict[str, str] = {}


def root_id(folder: str | None = None) -> str:
    name = folder or os.environ.get("DRIVE_ROOT", "colab-link-cache")
    if name not in _folders:
        hits = _q(f"name='{name}' and mimeType='{FOLDER_MIME}' and trashed=false")
        _folders[name] = hits[0]["id"] if hits else _json(
            "POST", f"{API}/files", {"name": name, "mimeType": FOLDER_MIME})["id"]
    return _folders[name]


def find(name: str, folder: str | None = None) -> dict | None:
    hits = _q(f"name='{name}' and '{root_id(folder)}' in parents and trashed=false")
    return hits[0] if hits else None


def ls(folder: str | None = None) -> list[dict]:
    return _q(f"'{root_id(folder)}' in parents and trashed=false")


def rm(name: str, folder: str | None = None) -> bool:
    meta = find(name, folder)
    if meta:
        _req("DELETE", f"{API}/files/{meta['id']}").read()
    return bool(meta)


def put(local: str, name: str, folder: str | None = None) -> dict:
    """Resumable upload; replaces an existing file of the same name."""
    parent = root_id(folder)
    size = os.path.getsize(local)
    old = find(name, folder)
    if old:
        url = f"{UPLOAD}/files/{old['id']}?uploadType=resumable"
        init = _req("PATCH", url, b"{}", {"Content-Type": "application/json",
                                          "X-Upload-Content-Length": str(size)})
    else:
        init = _req("POST", f"{UPLOAD}/files?uploadType=resumable",
                    json.dumps({"name": name, "parents": [parent]}).encode(),
                    {"Content-Type": "application/json", "X-Upload-Content-Length": str(size)})
    session = init.headers["Location"]
    with open(local, "rb") as f:
        off = 0
        while off < size:
            buf = f.read(CHUNK)
            end = off + len(buf) - 1
            r = _req("PUT", session, buf, {"Content-Range": f"bytes {off}-{end}/{size}"})
            off = end + 1
    return find(name, folder)


def get(name: str, local: str, workers: int = 16, folder: str | None = None) -> int:
    """Parallel ranged download. Returns bytes written."""
    meta = find(name, folder)
    if not meta:
        raise FileNotFoundError(name)
    size = int(meta["size"])
    url = f"{API}/files/{meta['id']}?alt=media"
    with open(local, "wb") as f:
        f.truncate(size)

    def part(start):
        end = min(start + CHUNK, size) - 1
        data = _req("GET", url, headers={"Range": f"bytes={start}-{end}"}).read()
        if len(data) != end - start + 1:
            raise RuntimeError(f"short read at {start}: {len(data)}")
        fd = os.open(local, os.O_WRONLY)
        try:
            os.pwrite(fd, data, start)
        finally:
            os.close(fd)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(part, range(0, size, CHUNK)))
    return size


if __name__ == "__main__":
    cmd = sys.argv[1]
    t = time.time()
    if cmd == "ls":
        for f in ls():
            print(f"{int(f.get('size', 0)) / 1e9:8.2f} GB  {f['modifiedTime']}  {f['name']}")
    elif cmd == "put":
        meta = put(sys.argv[2], sys.argv[3])
        dt = time.time() - t
        print(f"put {sys.argv[3]}: {int(meta['size']) / 1e9:.2f} GB in {dt:.1f}s "
              f"({int(meta['size']) / 1e6 / dt:.0f} MB/s)")
    elif cmd == "rm":
        print(("removed " if rm(sys.argv[2]) else "not found: ") + sys.argv[2])
    elif cmd == "get":
        n = get(sys.argv[2], sys.argv[3])
        dt = time.time() - t
        print(f"get {sys.argv[2]}: {n / 1e9:.2f} GB in {dt:.1f}s ({n / 1e6 / dt:.0f} MB/s)")

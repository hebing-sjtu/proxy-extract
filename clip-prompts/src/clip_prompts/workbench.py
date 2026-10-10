"""A browser workbench over a judged corpus: who was accepted, who was not, and why.

Standard library only, so it runs on the same pod that holds the corpus and is
reached through `kubectl port-forward`. It binds to localhost by default, which
is what a port-forward connects to, and which keeps the override endpoint off
the pod network.

The corpus is re-scanned in the background, so a judge running alongside shows
up as it goes; the page asks for the current rows, never for a fresh scan.

Who may change a verdict is one of three modes. `open` lets anyone who reaches
the page override, which is right only while that is you over your own
port-forward. `readonly` lets nobody. `reviewers` lets the holders of a token in
a reviewers file, one `name token` per line, and records the name with the
verdict. The server cannot tell a port-forward from a local browser - both
arrive from 127.0.0.1 - so sharing the page means choosing a mode, not a host.
"""

from __future__ import annotations

import hmac
import json
import mimetypes
import re
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import quality
from .layout import Clip

MEDIA = {
    "panel": lambda clip: quality.panel_path(clip),
    "rgb": lambda clip: clip.rgb,
    "duv": lambda clip: clip.duv,
    "anchor": lambda clip: clip.anchor,
}
CHUNK = 1 << 20
MODES = ("open", "readonly", "reviewers")
REVIEWER_NAME = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")


class Reviewers:
    """`name token` lines, re-read when the file changes so adding someone needs no restart."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._stamp: float | None = None
        self._tokens: dict[str, str] = {}
        self._lock = threading.Lock()

    def _load(self) -> dict[str, str]:
        stamp = self.path.stat().st_mtime
        with self._lock:
            if stamp != self._stamp:
                tokens = {}
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if len(fields) == 2 and not fields[0].startswith("#"):
                        tokens[fields[0]] = fields[1]
                self._tokens, self._stamp = tokens, stamp
            return self._tokens

    def who(self, token: str | None) -> str | None:
        if not token:
            return None
        try:
            roster = self._load()
        except OSError:
            return None
        found = None
        for name, expected in roster.items():
            if hmac.compare_digest(token.encode(), expected.encode()):
                found = name
        return found


def add_reviewer(path: Path, name: str, *, rotate: bool = False) -> str:
    """Append a reviewer with a fresh token and return the token."""
    if not REVIEWER_NAME.match(name):
        raise ValueError(f"reviewer names are letters, digits and _.@- ; got {name!r}")
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    kept = [line for line in lines if line.split()[:1] != [name]]
    if len(kept) != len(lines) and not rotate:
        raise ValueError(f"{name} already has a token; pass --rotate to replace it")
    token = secrets.token_urlsafe(18)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([*kept, f"{name} {token}"]) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return token


class Index:
    """The latest scan of the corpus, refreshed on a timer and on request."""

    def __init__(self, root: Path, *, interval: float = 120.0, workers: int = 32) -> None:
        self.root = Path(root)
        self.interval = interval
        self.workers = workers
        self.rows: dict[str, dict] = {}
        self.scanned: float | None = None
        self.scanning = False
        self._lock = threading.Lock()
        self._wake = threading.Event()

    def refresh(self) -> None:
        with self._lock:
            if self.scanning:
                return
            self.scanning = True
        try:
            rows = quality.scan(self.root, workers=self.workers)
            with self._lock:
                self.rows = {row["clip"]: row for row in rows}
                self.scanned = time.time()
        finally:
            with self._lock:
                self.scanning = False

    def update(self, name: str) -> dict:
        row = quality.summarize(self.root / name)
        with self._lock:
            self.rows[name] = row
        return row

    def snapshot(self) -> dict:
        with self._lock:
            rows = list(self.rows.values())
            scanned, scanning = self.scanned, self.scanning
        return {
            "root": str(self.root),
            "scanned": scanned,
            "scanning": scanning,
            "prompt": quality.PROMPT_VERSION,
            "audit": quality.audit(rows)["summary"],
            "rows": rows,
        }

    def poke(self) -> None:
        self._wake.set()

    def run_forever(self) -> None:
        while True:
            try:
                self.refresh()
            except Exception as exc:  # noqa: BLE001 - a fuse hiccup must not stop the timer
                print(f"[workbench] scan failed: {exc}", flush=True)
            self._wake.wait(self.interval)
            self._wake.clear()


def page() -> bytes:
    return resources.files(__package__).joinpath("workbench.html").read_bytes()


def make_handler(index: Index, *, mode: str = "open", reviewers: Reviewers | None = None):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if (mode == "reviewers") != (reviewers is not None):
        raise ValueError("the reviewers mode needs a reviewers file, and only it does")
    root = index.root

    class Handler(BaseHTTPRequestHandler):
        server_version = "clip-workbench/1"

        def log_message(self, format, *args):
            pass

        def _clip(self, name: str) -> Clip | None:
            if not quality.CLIP_NAME.match(name):
                return None
            path = root / name
            return Clip(path) if path.is_dir() else None

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json")

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status)

        def _reviewer(self) -> str | None:
            """Who is asking: a name, "local" in open mode, or None if they may not write."""
            if mode == "open":
                return "local"
            if mode == "readonly":
                return None
            header = self.headers.get("Authorization") or ""
            token = header[7:].strip() if header.lower().startswith("bearer ") else None
            return reviewers.who(token)

        def do_GET(self):
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            try:
                if not parts:
                    return self._send(HTTPStatus.OK, page(), "text/html; charset=utf-8")
                if parts == ["api", "whoami"]:
                    return self._json({"mode": mode, "reviewer": self._reviewer()})
                if parts == ["api", "clips"]:
                    if "refresh" in parse_qs(url.query):
                        index.poke()
                    return self._json(index.snapshot())
                if len(parts) == 3 and parts[:2] == ["api", "clip"]:
                    clip = self._clip(parts[2])
                    if clip is None:
                        return self._error(HTTPStatus.NOT_FOUND, "no such clip")
                    return self._json(
                        {
                            "row": index.update(clip.name),
                            "quality": quality.read_json(quality.quality_path(clip)),
                            "override": quality.read_json(quality.override_path(clip)),
                            "report": quality.read_json(clip.report_path),
                        }
                    )
                if len(parts) == 3 and parts[0] == "media" and parts[2] in MEDIA:
                    clip = self._clip(parts[1])
                    if clip is None:
                        return self._error(HTTPStatus.NOT_FOUND, "no such clip")
                    return self._file(MEDIA[parts[2]](clip))
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            except (BrokenPipeError, ConnectionResetError):
                return None

        def do_POST(self):
            parts = [p for p in urlparse(self.path).path.split("/") if p]
            if len(parts) != 3 or parts[:2] != ["api", "override"]:
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            reviewer = self._reviewer()
            if reviewer is None:
                status = HTTPStatus.FORBIDDEN if mode == "readonly" else HTTPStatus.UNAUTHORIZED
                return self._error(status, "this workbench is read-only" if mode == "readonly"
                                   else "log in with a reviewer token to change verdicts")
            clip = self._clip(parts[2])
            if clip is None:
                return self._error(HTTPStatus.NOT_FOUND, "no such clip")
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                verdict = body.get("verdict")
                quality.set_override(
                    clip, verdict if verdict else None, str(body.get("note") or ""), reviewer
                )
            except (ValueError, AttributeError) as exc:
                return self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return self._json({"row": index.update(clip.name)})

        def _file(self, path: Path) -> None:
            if not path.is_file():
                return self._error(HTTPStatus.NOT_FOUND, f"{path.name} not written yet")
            size = path.stat().st_size
            start, end = 0, size - 1
            match = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range") or "")
            if match and (match.group(1) or match.group(2)):
                if match.group(1):
                    start = int(match.group(1))
                    end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
                else:
                    start = max(size - int(match.group(2)), 0)
                if start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_header("Content-Type", kind)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = handle.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return Handler


def serve(root: Path, *, host: str = "127.0.0.1", port: int = 8765, interval: float = 120.0,
          workers: int = 32, mode: str = "open",
          reviewers: Path | None = None) -> ThreadingHTTPServer:
    """Start the scanner and return a bound server; the caller runs `serve_forever`."""
    roster = Reviewers(reviewers) if reviewers else None
    if roster:
        roster._load()  # a missing or unreadable file fails here, not on the first save
    index = Index(Path(root).expanduser().resolve(), interval=interval, workers=workers)
    threading.Thread(target=index.run_forever, name="workbench-scan", daemon=True).start()
    server = ThreadingHTTPServer((host, port), make_handler(index, mode=mode, reviewers=roster))
    server.daemon_threads = True
    server.index = index  # type: ignore[attr-defined]
    server.mode = mode  # type: ignore[attr-defined]
    return server

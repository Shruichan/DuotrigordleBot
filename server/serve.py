#!/usr/bin/env python3
"""HTTP wrapper around the C++ dt_worker subprocess.

Runs a tiny stdlib HTTP server on localhost. Each request is forwarded as
a single JSON line to a long-lived dt_worker process and the response is
read back as a JSON line.
"""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKER = ROOT / "build" / "dt_worker"
HOST = "127.0.0.1"
PORT = 8765


class WorkerProcess:
    """Persistent dt_worker subprocess with serialized stdin/stdout requests."""

    def __init__(self, binary: Path):
        self.binary = binary
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.start()

    def start(self):
        if not self.binary.exists():
            raise FileNotFoundError(f"dt_worker binary not found at {self.binary}")
        self.proc = subprocess.Popen(
            [str(self.binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            bufsize=1,
            text=True,
        )
        ready = self.proc.stdout.readline()
        info = json.loads(ready)
        if not info.get("ready"):
            raise RuntimeError(f"worker did not signal ready: {ready!r}")
        print(f"[serve] worker ready (pid={self.proc.pid})", file=sys.stderr)

    def call(self, payload: dict) -> dict:
        line = json.dumps(payload, separators=(",", ":"))
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                print("[serve] worker died, restarting", file=sys.stderr)
                self.start()
            assert self.proc and self.proc.stdin and self.proc.stdout
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
            resp_line = self.proc.stdout.readline()
        if not resp_line:
            raise RuntimeError("worker EOF")
        return json.loads(resp_line)


def cors_headers(self):
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS, GET")
    self.send_header("Access-Control-Allow-Headers", "Content-Type")


class Handler(BaseHTTPRequestHandler):
    worker: WorkerProcess = None  # set at runtime

    def log_message(self, fmt, *args):
        print(f"[serve] {self.address_string()} {fmt % args}", file=sys.stderr)

    def do_OPTIONS(self):
        self.send_response(204)
        cors_headers(self)
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._send_json(200, {"ok": True, "worker_pid": self.worker.proc.pid if self.worker.proc else None})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/suggest", "/review"):
            self._send_json(404, {"error": "unknown endpoint"})
            return
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"invalid json: {e}"})
            return
        if self.path == "/review":
            payload["command"] = "review"
        try:
            resp = self.worker.call(payload)
        except Exception as e:
            self._send_json(500, {"error": f"worker error: {e}"})
            return
        if "error" in resp:
            self._send_json(400, resp)
        else:
            self._send_json(200, resp)

    def _send_json(self, status: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        cors_headers(self)
        self.end_headers()
        self.wfile.write(body)


def main():
    Handler.worker = WorkerProcess(WORKER)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[serve] listening on http://{HOST}:{PORT}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[serve] shutting down", file=sys.stderr)
        server.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""HTTP server with turn-2 specialist on top of the C++ greedy worker.

Routing:
  /health   - liveness
  /suggest  - if state's turn==2, use specialist; else proxy to worker

The specialist re-ranks top-K=100 greedy candidates via a small neural net.
"""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "ml"
sys.path.insert(0, str(ML))

from train_turn2 import TurnSpecialist, load_word_features  # noqa: E402

WORKER = ROOT / "build" / "dt_worker"
MODEL_PATH = ML / "turn2_specialist.pt"
HOST = "127.0.0.1"
PORT = 8765
K_SPECIALIST = 100


class Worker:
    def __init__(self, binary: Path):
        self.binary = binary
        self.lock = threading.Lock()
        self.proc = None
        self.start()

    def start(self):
        self.proc = subprocess.Popen(
            [str(self.binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, text=True, bufsize=1)
        ready = json.loads(self.proc.stdout.readline())
        assert ready.get("ready")
        print(f"[serve] worker ready (pid={self.proc.pid})", file=sys.stderr)

    def call(self, payload: dict) -> dict:
        line = json.dumps(payload, separators=(",", ":"))
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self.start()
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
            return json.loads(self.proc.stdout.readline())


class Specialist:
    PER_BOARD_DIM = 132
    NUM_BOARDS = 32
    FEATURE_DIM = NUM_BOARDS * PER_BOARD_DIM

    def __init__(self, model_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TurnSpecialist().to(self.device)
        ck = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.word_feats, _ = load_word_features()  # (G, 130)
        with (ROOT / "data" / "valid_guesses.txt").open() as f:
            self.guesses = f.read().split()
        with (ROOT / "data" / "solutions_default.txt").open() as f:
            self.sols = f.read().split()
        self.guess_idx = {w: i for i, w in enumerate(self.guesses)}
        # Cached feedback table (uint8, [G, S])
        self.fb = np.fromfile(ML / "feedback_table.bin", dtype=np.uint8).reshape(len(self.guesses), len(self.sols))
        self.sol_letters = np.zeros((len(self.sols), 5, 26), dtype=np.float32)
        for s, w in enumerate(self.sols):
            for p, c in enumerate(w):
                self.sol_letters[s, p, ord(c) - ord('A')] = 1.0
        self.sol_letters_flat = self.sol_letters.reshape(len(self.sols), 130)
        print(f"[serve] specialist loaded ({sum(p.numel() for p in self.model.parameters()):,} params, device={self.device})", file=sys.stderr)

    def parse_pattern(self, s: str) -> int:
        p = 0; m = 1
        for ch in s:
            d = 2 if ch in 'Gg' else (1 if ch in 'Yy' else 0)
            p += d * m; m *= 3
        return p

    def derive_state(self, boards_history):
        """Replay the boards' guess history to compute candidate sets per board."""
        S = len(self.sols)
        cand = np.ones((self.NUM_BOARDS, S), dtype=bool)
        solved = np.zeros(self.NUM_BOARDS, dtype=bool)
        # Find max turns
        max_turns = max(len(b["guesses"]) for b in boards_history)
        for t in range(max_turns):
            # Determine the shared guess for turn t
            shared = None
            for b in boards_history:
                if t < len(b["guesses"]):
                    shared = b["guesses"][t]; break
            if shared is None:
                continue
            g_idx = self.guess_idx.get(shared)
            if g_idx is None:
                continue
            fb_row = self.fb[g_idx]
            for bi, b in enumerate(boards_history):
                if t < len(b["guesses"]):
                    p = self.parse_pattern(b["feedback"][t])
                    if p == 242:
                        solved[bi] = True
                        cand[bi, :] = False
                    else:
                        cand[bi] &= (fb_row == p)
        return cand, solved

    def state_features(self, cand, solved) -> np.ndarray:
        out = np.zeros((self.NUM_BOARDS, self.PER_BOARD_DIM), dtype=np.float32)
        for b in range(self.NUM_BOARDS):
            if solved[b]:
                out[b, -1] = 1.0
                continue
            n = int(cand[b].sum())
            if n == 0:
                continue
            out[b, -2] = n / 2653.0
            mask = self.sol_letters_flat[cand[b]]
            out[b, :130] = (mask.sum(axis=0) > 0).astype(np.float32)
        return out.reshape(-1)

    def rerank(self, state_feat: np.ndarray, top_k_words):
        K = len(top_k_words)
        top_k_idx = [self.guess_idx[w] for w in top_k_words]
        cand_word = self.word_feats[np.array(top_k_idx)]  # (K, 130)
        ranks = np.arange(K, dtype=np.float32).reshape(K, 1) / K
        cand_feat = np.concatenate([cand_word, ranks], axis=-1)
        with torch.no_grad():
            xb = torch.from_numpy(state_feat).unsqueeze(0).to(self.device)
            cb = torch.from_numpy(cand_feat).unsqueeze(0).to(self.device)
            scores = self.model(xb, cb).squeeze(0).cpu().numpy()
        order = np.argsort(scores)  # ascending = best first
        return [top_k_words[int(i)] for i in order], scores[order].tolist()


def cors(self):
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS, GET")
    self.send_header("Access-Control-Allow-Headers", "Content-Type")


class Handler(BaseHTTPRequestHandler):
    worker: Worker = None
    specialist: Specialist | None = None

    def log_message(self, fmt, *args):
        print(f"[serve] {self.address_string()} {fmt % args}", file=sys.stderr)

    def do_OPTIONS(self):
        self.send_response(204); cors(self); self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._send_json(200, {"ok": True,
                                  "worker_pid": self.worker.proc.pid if self.worker.proc else None,
                                  "specialist_loaded": self.specialist is not None})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/suggest":
            self._send_json(404, {"error": "unknown endpoint"}); return
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"bad json: {e}"}); return

        boards = payload.get("boards", [])
        if not boards:
            self._send_json(400, {"error": "no boards"}); return
        max_turns = max(len(b.get("guesses", [])) for b in boards)
        # We're picking the (max_turns+1)-th guess. If max_turns == 1, that's TURN 2.
        use_specialist = (max_turns == 1) and (self.specialist is not None)

        # Always get top-K from worker first (we need its ranking).
        worker_payload = dict(payload)
        worker_payload["top_k"] = max(K_SPECIALIST, payload.get("top_k", 5))
        try:
            resp = self.worker.call(worker_payload)
        except Exception as e:
            self._send_json(500, {"error": f"worker: {e}"}); return
        if "error" in resp:
            self._send_json(400, resp); return

        if use_specialist:
            try:
                # Derive state from boards history
                cand, solved = self.specialist.derive_state(boards)
                state_feat = self.specialist.state_features(cand, solved)
                top_k_words = [s["word"] for s in resp["suggestions"][:K_SPECIALIST]]
                reordered, scores = self.specialist.rerank(state_feat, top_k_words)
                # Build new suggestions list in specialist order, keeping could_solve from worker
                lookup = {s["word"]: s for s in resp["suggestions"]}
                new_sugg = []
                for i, w in enumerate(reordered[:payload.get("top_k", 5)]):
                    s = dict(lookup.get(w, {"word": w, "could_solve": []}))
                    s["specialist_score"] = scores[i]
                    new_sugg.append(s)
                resp["suggestions"] = new_sugg
                resp["specialist_used"] = True
            except Exception as e:
                print(f"[serve] specialist failed, falling back to worker: {e}", file=sys.stderr)
                resp["suggestions"] = resp["suggestions"][:payload.get("top_k", 5)]
                resp["specialist_used"] = False
        else:
            resp["suggestions"] = resp["suggestions"][:payload.get("top_k", 5)]
            resp["specialist_used"] = False

        self._send_json(200, resp)

    def _send_json(self, status, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        cors(self); self.end_headers()
        self.wfile.write(body)


def main():
    Handler.worker = Worker(WORKER)
    if MODEL_PATH.exists():
        Handler.specialist = Specialist(MODEL_PATH)
    else:
        print(f"[serve] specialist model not found at {MODEL_PATH}; running worker-only", file=sys.stderr)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[serve] listening on http://{HOST}:{PORT}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[serve] shutting down", file=sys.stderr)


if __name__ == "__main__":
    main()

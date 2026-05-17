#!/usr/bin/env python3
"""Value-net-guided beam search for Duotrigordle.

Beam search: for each top-K greedy candidate guess, evaluate by Monte-Carlo
sampling joint feedback outcomes, applying them to produce new states, and
asking the value network for E[remaining guesses]. Pick min E[total guesses].

Compared to greedy: greedy maximizes 1-step entropy; this maximizes the
estimated total cost from playing g now and the trained policy onwards. The
value net was trained on greedy self-play, so plugging it into one ply of
lookahead is one step of policy iteration over greedy.
"""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


WORKER_BIN = None  # set in main()


class Worker:
    """Manages a dt_worker subprocess for fast greedy top-K queries."""
    def __init__(self, binary: Path):
        self.proc = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        ready = json.loads(self.proc.stdout.readline())
        assert ready.get("ready")

    def topk(self, boards_history, k=10, alpha=1.0):
        req = {"boards": boards_history, "top_k": k, "alpha": alpha, "mode": "auto"}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self):
        try: self.proc.stdin.close()
        except Exception: pass
        self.proc.wait()


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PATTERN_ALL_GREEN = 242
NUM_BOARDS = 32
PER_BOARD_DIM = 5 * 26 + 2
FEATURE_DIM = NUM_BOARDS * PER_BOARD_DIM


# -- ValueNet (mirror of training architecture) --

class ValueNet(nn.Module):
    def __init__(self, num_boards: int = 32, per_board_dim: int = PER_BOARD_DIM,
                 emb_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.num_boards = num_boards
        self.per_board_dim = per_board_dim
        self.board_enc = nn.Sequential(
            nn.Linear(per_board_dim, 128), nn.ReLU(),
            nn.Linear(128, emb_dim), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.num_boards, self.per_board_dim)
        emb = self.board_enc(x)
        solved = x[:, :, -1].unsqueeze(-1)
        active = 1.0 - solved
        emb_a = emb * active
        n_a = active.sum(dim=1).clamp(min=1.0)
        return self.head(torch.cat([emb_a.sum(dim=1) / n_a,
                                    emb_a.max(dim=1).values,
                                    emb_a.sum(dim=1)], dim=-1)).squeeze(-1)


# -- Wordlists + feedback table --

def load_world(pool="default"):
    sols = (DATA / f"solutions_{pool}.txt").read_text().split()
    guesses = (DATA / "valid_guesses.txt").read_text().split()
    G, S = len(guesses), len(sols)
    fb = np.fromfile(ROOT / "ml" / "feedback_table.bin", dtype=np.uint8).reshape(G, S)
    g2s = np.full(G, -1, dtype=np.int32)
    s2g = np.full(S, 0, dtype=np.int32)
    g_idx = {w: i for i, w in enumerate(guesses)}
    for s, w in enumerate(sols):
        gi = g_idx[w]
        g2s[gi] = s
        s2g[s] = gi
    # Letter-position incidence per solution (5 x 26 per word, flattened to 130)
    sol_letters = np.zeros((S, 5, 26), dtype=np.float32)
    for s, w in enumerate(sols):
        for p, c in enumerate(w):
            sol_letters[s, p, ord(c) - ord('A')] = 1.0
    return {
        "guesses": guesses, "solutions": sols,
        "G": G, "S": S, "fb": fb, "g2s": g2s, "s2g": s2g,
        "sol_letters": sol_letters.reshape(S, 130),
        "g_idx": g_idx,
    }


# -- State representation (numpy-friendly) --
# state["cand"]: (32, S) bool — True if solution is still possible for that board
# state["solved"]: (32,) bool
# state["answer_used"]: (S,) bool — sols confirmed as some board's answer


def fresh_state(world):
    return {
        "cand": np.ones((NUM_BOARDS, world["S"]), dtype=bool),
        "solved": np.zeros(NUM_BOARDS, dtype=bool),
        "answer_used": np.zeros(world["S"], dtype=bool),
    }


def state_features(world, state):
    """Compute the 32*132 feature vector for the value net."""
    out = np.zeros((NUM_BOARDS, PER_BOARD_DIM), dtype=np.float32)
    for b in range(NUM_BOARDS):
        if state["solved"][b]:
            out[b, -1] = 1.0
            continue
        c = state["cand"][b]
        n = int(c.sum())
        out[b, -2] = n / 2653.0
        if n > 0:
            # Letter-position incidence = any candidate has letter L at position p
            mask = world["sol_letters"][c]  # (n, 130)
            out[b, :130] = (mask.sum(axis=0) > 0).astype(np.float32)
    return out.reshape(-1)


def features_batch(world, states):
    """Compute features for a list of states. Returns (N, FEATURE_DIM) float32."""
    feats = np.zeros((len(states), FEATURE_DIM), dtype=np.float32)
    for i, s in enumerate(states):
        feats[i] = state_features(world, s)
    return feats


def apply_guess(world, state, g_idx, patterns):
    """Returns a new state (out-of-place)."""
    new_cand = state["cand"].copy()
    new_solved = state["solved"].copy()
    new_used = state["answer_used"].copy()
    fb_row = world["fb"][g_idx]  # (S,)
    g_sol = world["g2s"][g_idx]
    for b in range(NUM_BOARDS):
        if state["solved"][b]:
            continue
        if patterns[b] == PATTERN_ALL_GREEN:
            new_solved[b] = True
            new_cand[b, :] = False
            if g_sol >= 0:
                new_used[g_sol] = True
        else:
            new_cand[b] &= (fb_row == patterns[b])
    # Distinct-answer constraint
    if new_used.any():
        # Zero out used answers from all unsolved boards
        for b in range(NUM_BOARDS):
            if new_solved[b]:
                continue
            new_cand[b] &= ~new_used
    return {"cand": new_cand, "solved": new_solved, "answer_used": new_used}


def game_over(state):
    return state["solved"].all()


# -- Greedy (for top-K candidate generation) --

def greedy_score_all(world, state, alpha=1.0):
    """Returns scores[G] = entropy_sum + alpha * expected_solves."""
    fb = world["fb"]  # (G, S)
    S = world["S"]
    G = world["G"]

    # Per-board candidate mask & inv_size
    active_boards = np.where(~state["solved"])[0]
    if len(active_boards) == 0:
        return None, np.zeros(G, dtype=np.float64)
    cand = state["cand"][active_boards]  # (A, S)
    sizes = cand.sum(axis=1)  # (A,)
    valid = sizes > 0
    cand = cand[valid]
    sizes = sizes[valid]
    if len(sizes) == 0:
        return None, np.zeros(G, dtype=np.float64)

    expected_solves = np.zeros(S, dtype=np.float64)
    for c, sz in zip(cand, sizes):
        expected_solves += c.astype(np.float64) * (1.0 / sz)

    # Entropy: for each guess g, sum over active boards of H(partition of C_b by fb[g, :]).
    # Vectorize: for each (g, b) compute bucket counts using bincount-style.
    # Naive loop over guesses is slow but OK for a prototype.
    scores = np.zeros(G, dtype=np.float64)
    for g in range(G):
        row = fb[g]  # (S,)
        s = 0.0
        for b_idx in range(len(cand)):
            cb = cand[b_idx]
            row_b = row[cb]
            counts = np.bincount(row_b, minlength=243)
            n = sizes[b_idx]
            p = counts[counts > 0] / n
            s += -np.sum(p * np.log2(p))
        sol = world["g2s"][g]
        if sol >= 0:
            s += alpha * expected_solves[sol]
        scores[g] = s
    return active_boards, scores


def greedy_score_all_fast(world, state, alpha=1.0):
    """Vectorized greedy scoring. Returns scores[G] float64."""
    fb = world["fb"]  # (G, S)
    G, S = fb.shape

    active = ~state["solved"]
    if not active.any():
        return np.zeros(G, dtype=np.float64)
    cand = state["cand"][active]  # (A, S)
    sizes = cand.sum(axis=1)
    has_c = sizes > 0
    cand = cand[has_c]
    sizes = sizes[has_c]
    A = len(sizes)
    if A == 0:
        return np.zeros(G, dtype=np.float64)

    # Forced moves: if any board has |C|=1, the forced word should be top.
    # We'll handle forced via score override AFTER computing entropy.
    forced_words = []
    for b_idx in range(A):
        if sizes[b_idx] == 1:
            sol_idx = int(np.where(cand[b_idx])[0][0])
            forced_words.append(world["s2g"][sol_idx])

    # Vectorized entropy:
    # For each guess g, sum_b H(partition of C_b by fb[g, :]).
    # Build per-active-board indices.
    scores = np.zeros(G, dtype=np.float64)
    # Process in batches over guesses to keep memory bounded
    BATCH = 256
    for gs in range(0, G, BATCH):
        ge = min(gs + BATCH, G)
        fb_b = fb[gs:ge]  # (b, S)
        # For each (g_in_batch, board), gather fb[g, C_b] -> compute histogram
        # We'll loop over boards (A=up to 32, cheap).
        H_acc = np.zeros(ge - gs, dtype=np.float64)
        for b_idx in range(A):
            cb = cand[b_idx]
            n = sizes[b_idx]
            # patterns of guesses on this board's candidates: shape (b, n)
            pats = fb_b[:, cb]  # (b, n)
            # For each g, bincount over patterns
            # Vectorized via offset trick: pats + g * 243 then bincount
            offs = np.arange(ge - gs)[:, None] * 243
            flat = (pats + offs).ravel()
            counts = np.bincount(flat, minlength=243 * (ge - gs)).reshape(ge - gs, 243)
            p = counts / n
            # Entropy: -sum p*log2(p), where p=0 contributes 0
            with np.errstate(divide='ignore', invalid='ignore'):
                logp = np.where(p > 0, np.log2(p), 0.0)
            H_acc += -np.sum(p * logp, axis=1)
        scores[gs:ge] = H_acc

    # Bonus: alpha * expected_solves[sol]
    if alpha != 0:
        es = np.zeros(S, dtype=np.float64)
        for b_idx in range(A):
            es += cand[b_idx].astype(np.float64) / sizes[b_idx]
        g2s = world["g2s"]
        bonus = np.where(g2s >= 0, alpha * es[np.maximum(g2s, 0)], 0.0)
        scores += bonus

    # Force forced words to top by giving them a huge bonus.
    if forced_words:
        for g in forced_words:
            scores[g] += 1e9

    return scores


def greedy_topk(world, state, k=20, alpha=1.0):
    scores = greedy_score_all_fast(world, state, alpha)
    # Tie-break by ascending index.
    order = np.lexsort((np.arange(len(scores)), -scores))
    return order[:k].tolist(), scores


# -- Beam search with value-net leaf --

def sample_answer_tuples(state, n_samples, rng):
    """Sample n_samples joint answer tuples — same tuples used for all candidates
    in a beam turn (paired comparison for variance reduction).
    Returns (n_samples, NUM_BOARDS) array of sol indices (or -1 for solved boards)."""
    out = np.full((n_samples, NUM_BOARDS), -1, dtype=np.int32)
    for b in range(NUM_BOARDS):
        if state["solved"][b]:
            continue
        cand_idx = np.where(state["cand"][b])[0]
        if len(cand_idx) == 0:
            continue
        out[:, b] = cand_idx[rng.integers(0, len(cand_idx), size=n_samples)]
    return out


def outcomes_for(world, state, g_idx, answer_tuples):
    """Apply g_idx to each sampled answer tuple to produce new states."""
    fb_row = world["fb"][g_idx]
    states = []
    for sample in answer_tuples:
        patterns = np.zeros(NUM_BOARDS, dtype=np.int32)
        for b in range(NUM_BOARDS):
            if state["solved"][b] or sample[b] < 0:
                patterns[b] = PATTERN_ALL_GREEN
            else:
                patterns[b] = int(fb_row[sample[b]])
        states.append(apply_guess(world, state, g_idx, patterns))
    return states


def beam_choose(world, state, model, device, worker, boards_history,
                K=10, samples=5, alpha=1.0, rng=None):
    rng = rng or np.random.default_rng()
    resp = worker.topk(boards_history, k=K, alpha=alpha)
    top_k_words = [s["word"] for s in resp.get("suggestions", [])]
    top_k = [world["g_idx"][w] for w in top_k_words]

    # Paired sampling: same answer tuples across all candidates → cancels variance.
    answer_tuples = sample_answer_tuples(state, samples, rng)

    new_states = []
    cand_slices = []
    for g in top_k:
        s_start = len(new_states)
        new_states.extend(outcomes_for(world, state, g, answer_tuples))
        cand_slices.append((s_start, len(new_states)))

    feats = features_batch(world, new_states)
    with torch.no_grad():
        xb = torch.from_numpy(feats).to(device, non_blocking=True)
        vals = model(xb).cpu().numpy()

    best_g = top_k[0]
    best_score = float('inf')
    for i, g in enumerate(top_k):
        a, b = cand_slices[i]
        if a == b:
            continue
        ev = vals[a:b].mean()
        score = 1.0 + ev
        if score < best_score:
            best_score = score
            best_g = g
    return best_g


# -- Game runner --

def feedback_one(g_word, a_word):
    digits = [0]*5; used=[False]*5
    for i in range(5):
        if g_word[i] == a_word[i]:
            digits[i] = 2; used[i] = True
    for i in range(5):
        if digits[i] == 2: continue
        for j in range(5):
            if not used[j] and g_word[i] == a_word[j]:
                digits[i] = 1; used[j] = True; break
    p = 0; m = 1
    for d in digits: p += d * m; m *= 3
    return p


def play_game(world, answers_sol_idx, strategy, model, device, worker,
              K=10, samples=5, max_guesses=50, rng=None):
    state = fresh_state(world)
    boards_history = [{"guesses": [], "feedback": []} for _ in range(NUM_BOARDS)]
    n_used = 0
    while n_used < max_guesses and not game_over(state):
        if strategy == "greedy":
            resp = worker.topk(boards_history, k=1)
            g = world["g_idx"][resp["suggestions"][0]["word"]]
        elif strategy == "beam":
            g = beam_choose(world, state, model, device, worker, boards_history,
                            K=K, samples=samples, rng=rng)
        else:
            raise ValueError(strategy)
        # Apply
        patterns = np.zeros(NUM_BOARDS, dtype=np.int32)
        for b in range(NUM_BOARDS):
            if state["solved"][b]:
                patterns[b] = PATTERN_ALL_GREEN
            else:
                patterns[b] = int(world["fb"][g, answers_sol_idx[b]])
        # Update boards_history for worker (string form)
        g_word = world["guesses"][g]
        for b in range(NUM_BOARDS):
            if state["solved"][b]:
                continue
            # convert pattern int to GYB string
            p = patterns[b]; s = []
            for _ in range(5):
                d = p % 3; p //= 3
                s.append("G" if d == 2 else ("Y" if d == 1 else "B"))
            boards_history[b]["guesses"].append(g_word)
            boards_history[b]["feedback"].append("".join(s))
        state = apply_guess(world, state, g, patterns)
        n_used += 1
    return n_used, state["solved"].all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT / "ml" / "model.pt")
    ap.add_argument("--worker", type=Path, default=ROOT / "build" / "dt_worker")
    ap.add_argument("--strategy", choices=["greedy", "beam"], default="beam")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()

    world = load_world()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValueNet(num_boards=32, per_board_dim=PER_BOARD_DIM).to(device)
    ck = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"device: {device}, model loaded")

    worker = Worker(args.worker)

    counts = []
    failed = 0
    t0 = time.time()
    for g in range(args.games):
        rng_g = np.random.default_rng(args.seed + g)
        ans = rng_g.choice(world["S"], size=NUM_BOARDS, replace=False)
        n, ok = play_game(world, ans, args.strategy, model, device, worker,
                          K=args.K, samples=args.samples,
                          rng=np.random.default_rng(args.seed * 1000 + g))
        if ok: counts.append(n)
        else: failed += 1
        if (g + 1) % 10 == 0:
            mean = sum(counts) / max(1, len(counts))
            elapsed = time.time() - t0
            print(f"  [{g+1}/{args.games}] solved={len(counts)} mean={mean:.3f} elapsed={elapsed:.1f}s")

    worker.close()
    counts.sort()
    n_solved = len(counts)
    mean = sum(counts) / n_solved if n_solved else 0
    print(f"\n=== {args.strategy} K={args.K} samples={args.samples} ===")
    print(f"games: {args.games}  solved: {n_solved}  failed: {failed}")
    if n_solved:
        print(f"mean: {mean:.3f}  min/max: {counts[0]}/{counts[-1]}")
        from collections import Counter
        for k, v in sorted(Counter(counts).items()):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

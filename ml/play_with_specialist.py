#!/usr/bin/env python3
"""Play games using turn-2 specialist + greedy elsewhere, bench vs pure greedy."""
import argparse
import json
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_turn2 import TurnSpecialist, load_word_features
import play as P


def specialist_choose(world, state, model, device, worker, boards_history, K=100, alpha=1.0):
    resp = worker.topk(boards_history, k=K, alpha=alpha)
    top_k_words = [s["word"] for s in resp.get("suggestions", [])]
    top_k_idx = [world["g_idx"][w] for w in top_k_words]
    if len(top_k_idx) < K:
        # Pad to K with last (rare with K=100); fall back to greedy
        return top_k_idx[0]

    state_feat = P.state_features(world, state)
    cand_word = world["all_word_feats"][np.array(top_k_idx)]  # (K, 130)
    ranks = np.arange(K, dtype=np.float32).reshape(K, 1) / K
    cand_feat = np.concatenate([cand_word, ranks], axis=-1)  # (K, 131)
    with torch.no_grad():
        xb = torch.from_numpy(state_feat).unsqueeze(0).to(device)
        cb = torch.from_numpy(cand_feat).unsqueeze(0).to(device)
        scores = model(xb, cb).squeeze(0).cpu().numpy()
    pick = int(np.argmin(scores))
    return top_k_idx[pick]


def play_game(world, answers, strategy, model, device, worker, K=100,
              max_guesses=50, specialist_turns=(2,), rng=None):
    state = P.fresh_state(world)
    boards_history = [{"guesses": [], "feedback": []} for _ in range(P.NUM_BOARDS)]
    n_used = 0
    while n_used < max_guesses and not P.game_over(state):
        turn = n_used + 1
        if strategy == "specialist" and turn in specialist_turns:
            g = specialist_choose(world, state, model, device, worker, boards_history, K=K)
        else:
            resp = worker.topk(boards_history, k=1)
            g = world["g_idx"][resp["suggestions"][0]["word"]]
        # Apply
        patterns = np.zeros(P.NUM_BOARDS, dtype=np.int32)
        for b in range(P.NUM_BOARDS):
            if state["solved"][b]:
                patterns[b] = P.PATTERN_ALL_GREEN
            else:
                patterns[b] = int(world["fb"][g, answers[b]])
        g_word = world["guesses"][g]
        for b in range(P.NUM_BOARDS):
            if state["solved"][b]:
                continue
            p = patterns[b]; s = []
            for _ in range(5):
                d = p % 3; p //= 3
                s.append("G" if d == 2 else ("Y" if d == 1 else "B"))
            boards_history[b]["guesses"].append(g_word)
            boards_history[b]["feedback"].append("".join(s))
        state = P.apply_guess(world, state, g, patterns)
        n_used += 1
    return n_used, state["solved"].all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path(__file__).parent / "turn2_specialist.pt")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--specialist-turns", default="2",
                    help="comma-separated turns to use specialist (e.g. '2' or '2,3')")
    args = ap.parse_args()

    world = P.load_world()
    word_feats, _ = load_word_features()
    world["all_word_feats"] = word_feats

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TurnSpecialist().to(device)
    ck = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    specialist_turns = tuple(int(t) for t in args.specialist_turns.split(","))
    print(f"device: {device}, specialist turns: {specialist_turns}")

    worker = P.Worker(P.ROOT / "build" / "dt_worker")

    g_counts = []; s_counts = []; diffs = []
    t0 = time.time()
    for g in range(args.games):
        rng = np.random.default_rng(args.seed + g)
        ans = rng.choice(world["S"], size=P.NUM_BOARDS, replace=False)
        n_g, _ = play_game(world, ans, "greedy", model, device, worker)
        n_s, _ = play_game(world, ans, "specialist", model, device, worker,
                           K=args.K, specialist_turns=specialist_turns)
        g_counts.append(n_g); s_counts.append(n_s); diffs.append(n_s - n_g)
        if (g + 1) % 10 == 0:
            el = time.time() - t0
            mg = sum(g_counts) / len(g_counts); ms = sum(s_counts) / len(s_counts)
            w = sum(1 for d in diffs if d < 0); t = sum(1 for d in diffs if d == 0)
            l = sum(1 for d in diffs if d > 0)
            print(f"  [{g+1}/{args.games}] greedy={mg:.3f} specialist={ms:.3f}  "
                  f"wins={w} ties={t} losses={l}  ({el:.1f}s)")

    worker.close()
    mg = sum(g_counts) / len(g_counts); ms = sum(s_counts) / len(s_counts)
    print(f"\n=== {args.games} games ===")
    print(f"greedy:     {mg:.3f}")
    print(f"specialist: {ms:.3f}")
    print(f"delta:      {ms - mg:+.3f}")
    w = sum(1 for d in diffs if d < 0); t = sum(1 for d in diffs if d == 0); l = sum(1 for d in diffs if d > 0)
    print(f"wins/ties/losses: {w}/{t}/{l}")
    print("\nDiff (specialist - greedy):")
    for k, v in sorted(Counter(diffs).items()):
        print(f"  {k:+d}: {v}")


if __name__ == "__main__":
    main()

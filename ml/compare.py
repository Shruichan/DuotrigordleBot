#!/usr/bin/env python3
"""Paired comparison: greedy vs beam on the same N answer sets."""
import argparse, time, sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import play

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()

    world = play.load_world()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = play.ValueNet(num_boards=32, per_board_dim=play.PER_BOARD_DIM).to(device)
    ck = torch.load(play.ROOT / "ml" / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    worker = play.Worker(play.ROOT / "build" / "dt_worker")

    g_counts = []
    b_counts = []
    diffs = []
    t0 = time.time()
    for g in range(args.games):
        rng_g = np.random.default_rng(args.seed + g)
        ans = rng_g.choice(world["S"], size=play.NUM_BOARDS, replace=False)
        # Use deterministic seed for play RNG so paired comparison is fair.
        n_g, _ = play.play_game(world, ans, "greedy", model, device, worker)
        n_b, _ = play.play_game(world, ans, "beam", model, device, worker,
                                K=args.K, samples=args.samples,
                                rng=np.random.default_rng(args.seed * 1000 + g))
        g_counts.append(n_g); b_counts.append(n_b); diffs.append(n_b - n_g)
        if (g + 1) % 10 == 0:
            el = time.time() - t0
            avg_g = sum(g_counts) / len(g_counts); avg_b = sum(b_counts) / len(b_counts)
            wins = sum(1 for d in diffs if d < 0)
            ties = sum(1 for d in diffs if d == 0)
            losses = sum(1 for d in diffs if d > 0)
            print(f"  [{g+1}/{args.games}] greedy={avg_g:.3f} beam={avg_b:.3f} "
                  f"beam_wins={wins} ties={ties} losses={losses} ({el:.1f}s)")

    worker.close()
    avg_g = sum(g_counts) / len(g_counts)
    avg_b = sum(b_counts) / len(b_counts)
    wins = sum(1 for d in diffs if d < 0)
    ties = sum(1 for d in diffs if d == 0)
    losses = sum(1 for d in diffs if d > 0)
    print(f"\n=== Paired comparison ({args.games} games) ===")
    print(f"greedy mean: {avg_g:.3f}    beam mean: {avg_b:.3f}    delta: {avg_b - avg_g:+.3f}")
    print(f"beam wins: {wins}  ties: {ties}  losses: {losses}")
    print("\nDiff distribution (beam - greedy):")
    for k, v in sorted(Counter(diffs).items()):
        print(f"  {k:+d}: {v}")

if __name__ == "__main__":
    main()

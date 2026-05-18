#!/usr/bin/env python3
"""Train a turn-2 specialist policy.

Given the post-turn-1 state and the top-K candidate words from greedy, predict
which candidate minimizes total guesses (regression on totals, ranking head).

Inputs (from dt_export_labels):
    features.bin   float32 [N x 4224]  state features at turn 2
    candidates.bin int32   [N x K]     top-K greedy candidate guess indices
    totals.bin     float32 [N x K]     game-end total guesses if that candidate is played

The model takes (state, candidate_word) and predicts total guesses. At inference
we score top-K and pick min.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_word_features():
    """5x26 letter-position incidence per word in valid_guesses (one-hot per position)."""
    words = (DATA / "valid_guesses.txt").read_text().split()
    G = len(words)
    feats = np.zeros((G, 5, 26), dtype=np.float32)
    for i, w in enumerate(words):
        for p, c in enumerate(w):
            feats[i, p, ord(c) - ord('A')] = 1.0
    return feats.reshape(G, 130), words


class TurnSpecialist(nn.Module):
    """State encoder + candidate encoder + cross MLP scoring candidate-conditional value.

    Candidate features: word one-hot (130) + scalar rank/K + optional extras (greedy_score, boards_count, expected_solves).
    cand_feat_dim is set at construction time based on data.
    """

    K_MAX = 100

    def __init__(self, num_boards=32, per_board_dim=132, cand_feat_dim=131,
                 state_emb=128, cand_emb=64, hidden=256):
        super().__init__()
        self.num_boards = num_boards
        self.per_board_dim = per_board_dim
        self.cand_feat_dim = cand_feat_dim
        # Per-board encoder
        self.board_enc = nn.Sequential(
            nn.Linear(per_board_dim, 128), nn.GELU(),
            nn.Linear(128, state_emb), nn.GELU(),
        )
        # Word (candidate) encoder
        self.word_enc = nn.Sequential(
            nn.Linear(cand_feat_dim, 96), nn.GELU(),
            nn.Linear(96, cand_emb), nn.GELU(),
        )
        # Pool over boards (mean + max + sum_active) -> state_emb_pooled
        self.pool_proj = nn.Linear(state_emb * 3, state_emb)
        # Cross: predict scalar score per (state, candidate)
        self.head = nn.Sequential(
            nn.Linear(state_emb + cand_emb, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def encode_state(self, x):
        B = x.shape[0]
        x = x.view(B, self.num_boards, self.per_board_dim)
        emb = self.board_enc(x)  # [B, 32, state_emb]
        solved = x[:, :, -1].unsqueeze(-1)
        active = 1.0 - solved
        emb_a = emb * active
        n_a = active.sum(dim=1).clamp(min=1.0)
        pooled = torch.cat([emb_a.sum(dim=1) / n_a,
                            emb_a.max(dim=1).values,
                            emb_a.sum(dim=1)], dim=-1)
        return self.pool_proj(pooled)  # [B, state_emb]

    def forward(self, state, cand_feats):
        """state: [B, 4224]   cand_feats: [B, K, 131] (word 130 + rank/K)  -> scores [B, K]"""
        B, K, _ = cand_feats.shape
        se = self.encode_state(state)  # [B, state_emb]
        ce = self.word_enc(cand_feats)  # [B, K, cand_emb]
        se_e = se.unsqueeze(1).expand(-1, K, -1)
        x = torch.cat([se_e, ce], dim=-1)
        return self.head(x).squeeze(-1)  # [B, K]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "ml" / "labels_t2")
    ap.add_argument("--out", type=Path, default=ROOT / "ml" / "turn2_specialist.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--state-emb", type=int, default=128)
    ap.add_argument("--cand-emb", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    meta = json.loads((args.data / "meta.json").read_text())
    N = meta["num_examples"]
    F_dim = meta["feature_dim"]
    K = meta["K"]
    print(f"Loaded {N} examples, F={F_dim}, K={K}")

    feats = np.fromfile(args.data / "features.bin", dtype=np.float32).reshape(N, F_dim)
    cand_idx = np.fromfile(args.data / "candidates.bin", dtype=np.int32).reshape(N, K)
    totals = np.fromfile(args.data / "totals.bin", dtype=np.float32).reshape(N, K)

    word_feats_all, _ = load_word_features()  # [G, 130]
    word_feats_all_t = torch.from_numpy(word_feats_all)

    print(f"Mean greedy total: {totals[:,0].mean():.3f}")
    print(f"Mean best   total: {totals.min(axis=1).mean():.3f}")
    print(f"Headroom: {(totals[:,0] - totals.min(axis=1)).mean():.3f} guesses/game")
    print(f"Games with gap>=1: {(totals[:,0] - totals.min(axis=1) >= 1).sum()}/{N}")

    # Per-example candidate features [N, K, base_dim (+ extras)]
    cand_word = word_feats_all[cand_idx]  # (N, K, 130)
    ranks = np.tile(np.arange(K, dtype=np.float32) / K, (N, 1)).reshape(N, K, 1)
    parts = [cand_word, ranks]
    if meta.get("has_cand_extras"):
        extras = np.fromfile(args.data / "cand_extras.bin", dtype=np.float32).reshape(N, K, 3)
        # Normalize: greedy_score is up to ~200, boards_count up to 32, expected_solves bounded by 1
        extras_norm = extras.copy()
        extras_norm[:, :, 0] /= 200.0
        extras_norm[:, :, 1] /= 32.0
        # expected_solves already bounded
        parts.append(extras_norm)
        print(f"Using extras (dim 3); cand_feat_dim = {sum(p.shape[-1] for p in parts)}")
    cand_feats = np.concatenate(parts, axis=-1)
    cand_feat_dim = cand_feats.shape[-1]

    X = torch.from_numpy(feats)
    C = torch.from_numpy(cand_feats)
    Y = torch.from_numpy(totals)

    dataset = TensorDataset(X, C, Y)
    n_val = int(N * args.val_frac); n_train = N - n_val
    tr, va = random_split(dataset, [n_train, n_val],
                          generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(va, batch_size=args.batch, shuffle=False,
                            num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = TurnSpecialist(
        cand_feat_dim=cand_feat_dim,
        state_emb=args.state_emb,
        cand_emb=args.cand_emb,
        hidden=args.hidden,
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Two losses combined:
    #  - Huber regression on total (predicted score -> actual total guesses)
    #  - Soft cross-entropy: target = uniform over candidates that tie at minimum total.
    #    The minimum is what matters; CE directly optimizes argmin selection.
    huber = nn.SmoothL1Loss()
    ce_weight = 1.0
    huber_weight = 0.1

    log = {"config": vars(args), "epochs": []}

    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0; nseen = 0; pick_match_train = 0
        for xb, cb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            scores = model(xb, cb)  # [B, K]
            # Soft target = uniform over ties at minimum
            y_min = yb.min(dim=1, keepdim=True).values
            is_min = (yb == y_min).float()
            soft = is_min / is_min.sum(dim=1, keepdim=True).clamp(min=1.0)
            log_p = F.log_softmax(-scores, dim=1)  # negate: low score = high prob
            loss_ce = -(soft * log_p).sum(dim=1).mean()
            loss_h = huber(scores, yb)
            loss = ce_weight * loss_ce + huber_weight * loss_h
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * xb.size(0); nseen += xb.size(0)
            pred_pick = scores.argmin(dim=1)
            true_pick = yb.argmin(dim=1)
            pick_match_train += (pred_pick == true_pick).sum().item()
        train_loss = total_loss / nseen

        # Validation
        model.eval()
        val_mse = 0; val_mae = 0; val_n = 0
        # Inference-relevant metric: how often does argmin(pred) match argmin(true)?
        val_pick_match = 0
        # Also: realized totals if we follow model's pick
        val_realized = 0.0; val_greedy = 0.0; val_optimal = 0.0
        with torch.no_grad():
            for xb, cb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                cb = cb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                scores = model(xb, cb)
                val_mse += ((scores - yb) ** 2).sum().item()
                val_mae += (scores - yb).abs().sum().item()
                val_n += xb.size(0)
                pred_pick = scores.argmin(dim=1)
                true_pick = yb.argmin(dim=1)
                val_pick_match += (pred_pick == true_pick).sum().item()
                val_realized += yb.gather(1, pred_pick.unsqueeze(1)).sum().item()
                val_greedy += yb[:, 0].sum().item()
                val_optimal += yb.min(dim=1).values.sum().item()
        val_mse /= (val_n * yb.shape[1])
        val_mae /= (val_n * yb.shape[1])
        val_realized /= val_n
        val_greedy /= val_n
        val_optimal /= val_n
        sched.step()
        dt = time.time() - t0
        print(f"epoch {ep+1:>2}/{args.epochs}  loss={train_loss:.3f}  "
              f"val_mae={val_mae:.3f}  pick_acc={val_pick_match/val_n:.3f}  "
              f"realized={val_realized:.3f}  (greedy={val_greedy:.3f} optimal={val_optimal:.3f})  "
              f"({dt:.1f}s)")
        log["epochs"].append({
            "epoch": ep+1, "train_loss": train_loss, "val_mae": val_mae,
            "pick_acc": val_pick_match/val_n,
            "val_realized": val_realized,
            "val_greedy": val_greedy,
            "val_optimal": val_optimal,
        })

    torch.save({
        "state_dict": model.state_dict(),
        "config": vars(args),
        "meta": meta,
        "cand_feat_dim": cand_feat_dim,
    }, args.out)
    print(f"saved {args.out}")
    (args.out.parent / "turn2_train_log.json").write_text(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()

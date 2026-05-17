#!/usr/bin/env python3
"""Train a value network that predicts expected remaining guesses from a state.

Input:  features.bin (float32, [N x feature_dim])
        targets.bin  (float32, [N])
        meta.json
Output: model.pt  (TorchScript or state_dict)
        train_log.json
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


def load_dataset(data_dir: Path):
    meta = json.loads((data_dir / "meta.json").read_text())
    N = meta["num_examples"]
    F = meta["feature_dim"]
    feats = np.fromfile(data_dir / "features.bin", dtype=np.float32).reshape(N, F)
    targets = np.fromfile(data_dir / "targets.bin", dtype=np.float32)
    assert targets.shape == (N,), f"{targets.shape} vs {(N,)}"
    print(f"Loaded {N} examples, feature_dim={F}, mean_target={targets.mean():.3f}")
    return feats, targets, meta


class ValueNet(nn.Module):
    """Per-board encoder + global head."""

    def __init__(self, num_boards: int = 32, per_board_dim: int = 132,
                 emb_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.num_boards = num_boards
        self.per_board_dim = per_board_dim
        # Shared per-board encoder (boards are exchangeable, except for "solved" flag).
        self.board_enc = nn.Sequential(
            nn.Linear(per_board_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim),
            nn.ReLU(),
        )
        # Pool over boards + global head.
        self.head = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden),  # mean + max + sum-active boards
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 32 * 132]
        B = x.shape[0]
        x = x.view(B, self.num_boards, self.per_board_dim)
        emb = self.board_enc(x)  # [B, 32, emb_dim]
        # Use solved flag (last feature) as a mask for "active" pooling.
        solved = x[:, :, -1].unsqueeze(-1)  # [B, 32, 1]
        active_mask = 1.0 - solved
        emb_active = emb * active_mask
        # Pool: mean (over active), max, sum
        n_active = active_mask.sum(dim=1).clamp(min=1.0)
        mean_pool = emb_active.sum(dim=1) / n_active
        max_pool = emb_active.max(dim=1).values
        sum_pool = emb_active.sum(dim=1)
        pooled = torch.cat([mean_pool, max_pool, sum_pool], dim=-1)
        return self.head(pooled).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "model.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feats, targets, meta = load_dataset(args.data)
    F = meta["feature_dim"]
    per_board_dim = F // 32
    assert per_board_dim * 32 == F, "feature_dim must be divisible by num_boards (32)"

    X = torch.from_numpy(feats)
    y = torch.from_numpy(targets)

    dataset = TensorDataset(X, y)
    n_val = int(len(dataset) * args.val_frac)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(args.seed))
    print(f"train={n_train} val={n_val}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'})")
    model = ValueNet(num_boards=32, per_board_dim=per_board_dim).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss()

    # Baseline: predict global mean
    baseline_mean = y.mean().item()
    baseline_mse = ((y - baseline_mean) ** 2).mean().item()
    baseline_mae = (y - baseline_mean).abs().mean().item()
    print(f"baseline (predict mean={baseline_mean:.2f}): MSE={baseline_mse:.3f} MAE={baseline_mae:.3f}")

    log = {"epochs": [], "config": vars(args) | {"baseline_mean": baseline_mean,
                                                "baseline_mse": baseline_mse,
                                                "baseline_mae": baseline_mae}}

    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0; n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0); n += xb.size(0)
        train_loss = total_loss / n

        model.eval()
        val_mse = 0.0; val_mae = 0.0; m = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                val_mse += ((pred - yb) ** 2).sum().item()
                val_mae += (pred - yb).abs().sum().item()
                m += xb.size(0)
        val_mse /= m; val_mae /= m
        sched.step()
        dt = time.time() - t0
        print(f"epoch {ep+1:>2}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_mse={val_mse:.4f}  val_mae={val_mae:.4f}  ({dt:.1f}s)")
        log["epochs"].append({"epoch": ep+1, "train_loss": train_loss,
                              "val_mse": val_mse, "val_mae": val_mae, "time": dt})

    # Save state dict + minimal info
    torch.save({
        "state_dict": model.state_dict(),
        "meta": meta,
        "config": vars(args),
        "per_board_dim": per_board_dim,
        "num_boards": 32,
    }, args.out)
    print(f"saved {args.out}")
    (args.out.parent / "train_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(f"final val MAE: {log['epochs'][-1]['val_mae']:.3f} guesses  "
          f"(baseline {baseline_mae:.3f})")


if __name__ == "__main__":
    main()

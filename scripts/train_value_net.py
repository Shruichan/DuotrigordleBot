#!/usr/bin/env python3.10
"""
Train a tiny MLP to predict remaining_turns from compact state features.

Inputs (from dt_gen_value_data):
  data_dir/features.bin : float32 [N x 25]
  data_dir/labels.bin   : float32 [N]
  data_dir/meta.json

Output:
  out_path : binary weights file consumable by ValueNet (C++)

Binary format (little-endian floats; layout chosen so C++ can mmap):
  uint32 magic = 0xDDDDD000
  uint32 feature_dim
  uint32 num_layers     (e.g. 3 => [in, h1, h2, out])
  uint32 layer_sizes[num_layers+1]
  for each layer in order:
    float32 W[out_features x in_features]   (row-major, out major)
    float32 b[out_features]
  uint32 magic_end = 0xDDDDD001

Run:
  python3.10 scripts/train_value_net.py \
    --data data/value_train --out data/value_net.bin \
    --epochs 20 --batch 2048 --hidden 64 --lr 1e-3
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


MAGIC_START = 0xDDDDD000
MAGIC_END   = 0xDDDDD001


class ValueMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_data(data_dir: Path):
    meta = json.loads((data_dir / "meta.json").read_text())
    feat_dim = int(meta["feature_dim"])
    feats = np.fromfile(data_dir / "features.bin", dtype=np.float32).reshape(-1, feat_dim)
    labels = np.fromfile(data_dir / "labels.bin", dtype=np.float32)
    assert feats.shape[0] == labels.shape[0], (feats.shape, labels.shape)
    return feats, labels, meta


def dump_weights(model: nn.Module, feat_dim: int, hidden: int, out_path: Path):
    # Pull linear layers in order.
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    assert len(linears) == 3, f"expected 3 linears, got {len(linears)}"
    layer_sizes = [feat_dim, hidden, hidden, 1]
    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", MAGIC_START, feat_dim))
        f.write(struct.pack("<I", 3))  # num layers
        f.write(struct.pack("<IIII", *layer_sizes))
        for li, lin in enumerate(linears):
            W = lin.weight.detach().cpu().numpy().astype(np.float32)
            b = lin.bias.detach().cpu().numpy().astype(np.float32)
            assert W.shape == (layer_sizes[li+1], layer_sizes[li]), (W.shape, layer_sizes[li:li+2])
            assert b.shape == (layer_sizes[li+1],), (b.shape, layer_sizes[li+1])
            f.write(W.tobytes())
            f.write(b.tobytes())
        f.write(struct.pack("<I", MAGIC_END))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quantile", type=float, default=0.0,
                    help="If >0, train pinball/quantile loss at this tau (e.g. 0.9 = "
                         "worst-case-averse) instead of SmoothL1 on the mean.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feats, labels, meta = load_data(args.data)
    feat_dim = feats.shape[1]
    n = feats.shape[0]
    print(f"loaded {n} rows × {feat_dim} feats; label range [{labels.min():.1f}, {labels.max():.1f}], mean {labels.mean():.2f}")

    # Shuffle + split
    perm = np.random.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx = perm[:n_val]
    tr_idx  = perm[n_val:]
    Xtr, ytr = feats[tr_idx], labels[tr_idx]
    Xva, yva = feats[val_idx], labels[val_idx]
    print(f"train {len(tr_idx)} / val {len(val_idx)}")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"device: {device}")

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    yva_t = torch.from_numpy(yva).to(device)

    ds = TensorDataset(Xtr_t, ytr_t)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    model = ValueMLP(feat_dim, args.hidden).to(device)
    # Initialize final-layer bias to label mean so loss starts in a sane regime.
    with torch.no_grad():
        final = [m for m in model.net if isinstance(m, nn.Linear)][-1]
        final.bias.fill_(float(ytr.mean()))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.quantile > 0.0:
        tau = args.quantile
        def loss_fn(pred, target):
            # Pinball / quantile loss: asymmetric, penalizes under-prediction of
            # the tail more heavily at high tau -> worst-case-averse value.
            e = target - pred
            return torch.mean(torch.maximum(tau * e, (tau - 1.0) * e))
        print(f"quantile loss tau={tau}")
    else:
        loss_fn = nn.SmoothL1Loss()  # Huber: robust to outlier tail labels.

    # Naive baseline: predict mean.
    base = float(ytr.mean())
    base_l1 = float(np.abs(yva - base).mean())
    print(f"baseline (predict mean={base:.2f}): val L1 = {base_l1:.3f}")

    # Checkpoint on validation loss (works for both Huber and pinball modes).
    best_loss = float("inf")
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for xb, yb in dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
            n_batches += 1
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t)
            val_loss = float(loss_fn(pv, yva_t))
            val_l1 = float((pv - yva_t).abs().mean())
        if val_loss < best_loss:
            best_loss = val_loss
            dump_weights(model, feat_dim, args.hidden, args.out)
        print(f"  ep{ep+1:2d}  train_loss={ep_loss/n_batches:.4f}  val_loss={val_loss:.4f}  val_L1={val_l1:.3f}  (best_loss={best_loss:.4f})")

    print(f"best val loss = {best_loss:.4f}  (predict-mean baseline L1 {base_l1:.3f})")


if __name__ == "__main__":
    main()

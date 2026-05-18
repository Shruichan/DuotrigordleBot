#!/usr/bin/env python3
"""Concatenate multiple labels_* directories into a single dataset."""
import json, sys
from pathlib import Path

dirs = [Path(d) for d in sys.argv[1:-1]]
out = Path(sys.argv[-1])
out.mkdir(parents=True, exist_ok=True)

metas = [json.loads((d / "meta.json").read_text()) for d in dirs]
total_N = sum(m["num_examples"] for m in metas)
F = metas[0]["feature_dim"]; K = metas[0]["K"]
assert all(m["feature_dim"] == F and m["K"] == K for m in metas)

# Cat each bin file
for name in ("features.bin", "candidates.bin", "totals.bin"):
    with open(out / name, "wb") as o:
        for d in dirs:
            o.write((d / name).read_bytes())

(out / "meta.json").write_text(json.dumps({
    "num_examples": total_N, "feature_dim": F, "K": K,
    "target_turn": metas[0]["target_turn"],
    "alpha": metas[0]["alpha"],
    "merged_from": [str(d) for d in dirs],
}, indent=2))
print(f"merged {total_N} examples from {len(dirs)} dirs into {out}")

#!/bin/bash
set -e
cd "$(dirname "$0")/.."

GEN_PID=275307
echo "=== Waiting for v2 data gen (PID $GEN_PID) ==="
while kill -0 $GEN_PID 2>/dev/null; do
    sleep 60
    PROGRESS=$(grep -oE '[0-9]+/10000' /tmp/dt-labels-rich.log 2>/dev/null | tail -1)
    echo "  $(date +%H:%M:%S) $PROGRESS"
done
echo "Data gen done."

echo "=== Training v2 specialist (epochs=100) ==="
./ml/.venv/bin/python ml/train_turn2.py \
    --data ml/labels_t2_rich --out ml/turn2_specialist_v2.pt \
    --epochs 100 --batch 512

echo "=== Benchmark v2 on 300 games ==="
./ml/.venv/bin/python ml/play_with_specialist.py \
    --games 300 --K 100 --specialist-turns 2 \
    --model ml/turn2_specialist_v2.pt
echo "=== v2 pipeline done ==="

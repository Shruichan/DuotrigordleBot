#!/bin/bash
# Wait for both data gens to finish, then merge → train → bench
set -e
cd "$(dirname "$0")/.."

echo "=== Waiting for data gen processes... ==="
while pgrep -f "dt_export_labels.*labels_t2" > /dev/null; do
    sleep 30
    echo "  $(date +%H:%M:%S) still running ($(pgrep -af dt_export_labels | wc -l) procs)"
done
echo "Data gen done."

echo
echo "=== Merging datasets ==="
./ml/.venv/bin/python ml/merge_labels.py ml/labels_t2 ml/labels_t2_b ml/labels_t2_merged

echo
echo "=== Training specialist (epochs=80, batch=512) ==="
./ml/.venv/bin/python ml/train_turn2.py --data ml/labels_t2_merged --epochs 80 --batch 512

echo
echo "=== Benchmarking specialist on 200 games (vs greedy) ==="
./ml/.venv/bin/python ml/play_with_specialist.py --games 200 --K 100 --specialist-turns 2 \
    | tee /tmp/dt-specialist-bench.log
echo
echo "=== Done ==="

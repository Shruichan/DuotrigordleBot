#!/usr/bin/env bash
# Train worst-case-averse quantile value nets and bench them in exact-V mode.
# Goal: drive the 35-tail toward zero (at a known mean cost).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DATA="${DATA:-data/value_train_20260527}"
N="${N:-2000}"; SEED="${SEED:-7}"
LOG="${LOG:-scripts/quantile_log.txt}"; BIN=./build/dt_bench
PY=python3.10

bench() {
    local label="$1" net="$2"
    local out mean max dist
    out=$($BIN -n "$N" -s "$SEED" -S greedy --opener LITRE -a 300 \
          --la-k 12 --la-exact --value-net "$net" 2>/dev/null)
    mean=$(echo "$out" | awk '/Mean guesses/{print $3; exit}')
    max=$(echo "$out"  | awk '/Min . Max/{print $NF; exit}')
    dist=$(echo "$out" | awk '/^   3[0-9]:/{printf "%s%s ", $1, $2}')
    printf "%-22s mean=%s max=%s  %s\n" "$label" "$mean" "$max" "$dist" | tee -a "$LOG"
}

echo "==== quantile experiment $(date -Iseconds) N=$N seed=$SEED ====" | tee -a "$LOG"

for TAU in 0.80 0.90 0.95; do
    net="data/value_net_q${TAU}.bin"
    echo "--- train tau=$TAU ---"
    $PY scripts/train_value_net.py --data "$DATA" --out "$net" \
        --epochs 25 --batch 4096 --lr 1e-3 --quantile "$TAU" 2>&1 | tail -1
    bench "exactV q$TAU" "$net"
done

echo "==== quantile experiment done $(date -Iseconds) ====" | tee -a "$LOG"

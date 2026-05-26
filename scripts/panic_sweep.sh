#!/usr/bin/env bash
# Budget-aware panic-mode sweep: mean-vs-tail Pareto for the 34-finish goal.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
N="${N:-2000}"; SEED="${SEED:-7}"; LOG="${LOG:-scripts/panic_sweep_log.txt}"
BIN=./build/dt_bench

run() {
    local label="$1"; shift
    local out mean max dist
    out=$($BIN -n "$N" -s "$SEED" -S greedy --opener LITRE -a 300 "$@" 2>/dev/null)
    mean=$(echo "$out" | awk '/Mean guesses/{print $3; exit}')
    max=$(echo "$out"  | awk '/Min . Max/{print $NF; exit}')
    dist=$(echo "$out" | awk '/^   3[0-9]:/{printf "%s%s ", $1, $2}')
    printf "%-30s mean=%s max=%s  %s\n" "$label" "$mean" "$max" "$dist" | tee -a "$LOG"
}

echo "==== panic sweep $(date -Iseconds) N=$N seed=$SEED ====" | tee -a "$LOG"
run "baseline (panic off)"
run "slack0 a0"   --panic-slack 0 --panic-alpha 0
run "slack1 a0"   --panic-slack 1 --panic-alpha 0
run "slack2 a0"   --panic-slack 2 --panic-alpha 0
run "slack3 a0"   --panic-slack 3 --panic-alpha 0
run "slack4 a0"   --panic-slack 4 --panic-alpha 0
run "slack2 a10"  --panic-slack 2 --panic-alpha 10
run "slack3 a30"  --panic-slack 3 --panic-alpha 30
run "slack5 a0"   --panic-slack 5 --panic-alpha 0
echo "==== panic sweep done $(date -Iseconds) ====" | tee -a "$LOG"

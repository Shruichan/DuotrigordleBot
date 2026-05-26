#!/usr/bin/env bash
# Exact expected-V late-game gate sweep. Greedy early (banks 33s), risk-averse
# exact-V only when active boards <= gate (kills the 36/35 tail). Find Pareto.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
N="${N:-2000}"; SEED="${SEED:-7}"; NET="${NET:-data/value_net_20260527.bin}"
LOG="${LOG:-scripts/exactv_sweep_log.txt}"; BIN=./build/dt_bench

run() {
    local label="$1"; shift
    local out mean max dist
    out=$($BIN -n "$N" -s "$SEED" -S greedy --opener LITRE -a 300 "$@" 2>/dev/null)
    mean=$(echo "$out" | awk '/Mean guesses/{print $3; exit}')
    max=$(echo "$out"  | awk '/Min . Max/{print $NF; exit}')
    dist=$(echo "$out" | awk '/^   3[0-9]:/{printf "%s%s ", $1, $2}')
    printf "%-26s mean=%s max=%s  %s\n" "$label" "$mean" "$max" "$dist" | tee -a "$LOG"
}

echo "==== exact-V gate sweep $(date -Iseconds) N=$N seed=$SEED ====" | tee -a "$LOG"
run "baseline"
for GATE in 4 6 8 10 12 16; do
    run "exactV gate<=$GATE K12" --la-k 12 --la-exact --la-exact-max-active "$GATE" --value-net "$NET"
done
run "exactV always K12"  --la-k 12 --la-exact --value-net "$NET"
echo "==== exact-V gate sweep done $(date -Iseconds) ====" | tee -a "$LOG"

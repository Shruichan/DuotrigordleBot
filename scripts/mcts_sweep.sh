#!/usr/bin/env bash
# MCTS config sweep. Benches each config on the same N games / seed and appends
# a one-line result to scripts/mcts_sweep_log.txt. Baseline greedy first.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

N="${N:-2000}"
SEED="${SEED:-7}"
NET="${NET:-data/value_net_20260527.bin}"
LOG="${LOG:-scripts/mcts_sweep_log.txt}"
BIN=./build/dt_bench

run() {  # label, extra-args...
    local label="$1"; shift
    local out
    out=$($BIN -n "$N" -s "$SEED" --opener LITRE -a 300 "$@" 2>/dev/null)
    local mean max dist
    mean=$(echo "$out" | awk '/Mean guesses/{print $3; exit}')
    max=$(echo "$out"  | awk '/Min . Max/{print $NF; exit}')
    dist=$(echo "$out" | awk '/^   3[0-9]:/{printf "%s%s ", $1, $2}')
    printf "%-42s mean=%s max=%s  %s\n" "$label" "$mean" "$max" "$dist" | tee -a "$LOG"
}

echo "==== mcts sweep $(date -Iseconds)  N=$N seed=$SEED ====" | tee -a "$LOG"

run "baseline-greedy"                 -S greedy
run "mcts d2 r0  K10 ma16"            -S mcts --value-net "$NET" --mcts-k 10 --mcts-r 20 --mcts-depth 2 --mcts-max-active 16 --mcts-risk 0
run "mcts d3 r0  K10 ma16"            -S mcts --value-net "$NET" --mcts-k 10 --mcts-r 20 --mcts-depth 3 --mcts-max-active 16 --mcts-risk 0
run "mcts d2 risk3 K10 ma16"          -S mcts --value-net "$NET" --mcts-k 10 --mcts-r 20 --mcts-depth 2 --mcts-max-active 16 --mcts-risk 3
run "mcts d2 risk8 K10 ma16"          -S mcts --value-net "$NET" --mcts-k 10 --mcts-r 20 --mcts-depth 2 --mcts-max-active 16 --mcts-risk 8
run "mcts d3 risk5 K12 ma20 R40"      -S mcts --value-net "$NET" --mcts-k 12 --mcts-r 40 --mcts-depth 3 --mcts-max-active 20 --mcts-risk 5
run "mcts d2 risk15 K10 ma16"         -S mcts --value-net "$NET" --mcts-k 10 --mcts-r 20 --mcts-depth 2 --mcts-max-active 16 --mcts-risk 15
run "mcts d4 risk5 K10 ma14 R30"      -S mcts --value-net "$NET" --mcts-k 10 --mcts-r 30 --mcts-depth 4 --mcts-max-active 14 --mcts-risk 5

echo "==== sweep done $(date -Iseconds) ====" | tee -a "$LOG"

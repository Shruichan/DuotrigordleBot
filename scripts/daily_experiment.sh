#!/usr/bin/env bash
# Train a value net on greedy self-play across the real daily-MT19937 sequence
# (daily IDs 1..1500) and bench it against the daily distribution. Compares
# baseline greedy and the existing random-trained exact-V net.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_DIR="data/value_train_daily"
NET_PATH="data/value_net_daily.bin"
LOG="scripts/daily_experiment_log.txt"
PY=python3.10
BENCH=./build/dt_bench
N=2000
SEED=7  # bench seed (only matters in random mode; daily mode uses ids)

# ---- 1. generate training data on dailies 1..1500 ----
echo "==== daily experiment $(date -Iseconds) ====" | tee -a "$LOG"
echo "[1/4] gen daily IDs 1..1500 -> $DATA_DIR" | tee -a "$LOG"
./build/dt_gen_value_data --daily 1 1500 -o "$DATA_DIR" --log-every 200 \
    2>&1 | grep -E "Generating|games_ok|Done"

# ---- 2. train the value net ----
echo "[2/4] train -> $NET_PATH" | tee -a "$LOG"
$PY scripts/train_value_net.py \
    --data "$DATA_DIR" --out "$NET_PATH" \
    --epochs 40 --batch 4096 --lr 1e-3 --hidden 64 \
    2>&1 | grep -E "^loaded|^baseline|best val|ep4[0-9]" | tail -8

# ---- 3. bench: baseline / original-net / daily-net, all on daily 1..N ----
run() {
    local label="$1"; shift
    local out mean max dist
    out=$($BENCH -n "$N" --daily 1 -S greedy --opener LITRE -a 300 "$@" 2>/dev/null)
    mean=$(echo "$out" | awk '/Mean guesses/{print $3; exit}')
    max=$(echo "$out"  | awk '/Min . Max/{print $NF; exit}')
    dist=$(echo "$out" | awk '/^   3[0-9]:/{printf "%s%s ", $1, $2}')
    printf "%-32s mean=%s max=%s  %s\n" "$label" "$mean" "$max" "$dist" | tee -a "$LOG"
}

echo "[3/4] bench daily IDs 1..$N (3 configs)" | tee -a "$LOG"
run "baseline-greedy (no net)"
run "exact-V random-trained"  --la-k 12 --la-exact --value-net data/value_net_20260527.bin
run "exact-V DAILY-trained"   --la-k 12 --la-exact --value-net "$NET_PATH"

echo "[4/4] done $(date -Iseconds)" | tee -a "$LOG"

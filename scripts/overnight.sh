#!/usr/bin/env bash
# Overnight value-net training & bench pipeline.
#
# Phases:
#   1. Generate self-play training data with greedy (LITRE / alpha=300)
#   2. Train tiny MLP value head on (state -> remaining_turns)
#   3. Bench: baseline greedy   vs   greedy + value-net lookahead
#   4. Append a one-line summary to scripts/overnight_log.txt
#
# Conservative defaults: 100k games, 30 epochs, bench 2000 games × both modes.
# Override via env: GAMES, EPOCHS, BENCH_N, LA_K, LA_N.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

GAMES="${GAMES:-100000}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-4096}"
LR="${LR:-1e-3}"
HIDDEN="${HIDDEN:-64}"
BENCH_N="${BENCH_N:-2000}"
BENCH_SEED="${BENCH_SEED:-7}"
LA_K="${LA_K:-10}"
LA_N="${LA_N:-5}"
DATA_DIR="${DATA_DIR:-data/value_train_$(date +%Y%m%d)}"
NET_OUT="${NET_OUT:-data/value_net_$(date +%Y%m%d).bin}"
LOG="${LOG:-scripts/overnight_log.txt}"
PYTHON="${PYTHON:-python3.10}"

mkdir -p "$(dirname "$DATA_DIR")" "$(dirname "$NET_OUT")" "$(dirname "$LOG")"

echo "==== overnight run $(date -Iseconds) ===="
echo "games=$GAMES  epochs=$EPOCHS  hidden=$HIDDEN  batch=$BATCH  lr=$LR"
echo "bench_n=$BENCH_N  la_k=$LA_K  la_n=$LA_N"
echo "data_dir=$DATA_DIR  net=$NET_OUT  log=$LOG"

# Ensure binaries up to date.
if [[ ! -x build/dt_bench || ! -x build/dt_gen_value_data ]]; then
    echo "(rebuilding)"
    cmake --build build -j
fi

T0=$(date +%s)

echo
echo "[1/3] Generating data ($GAMES games) -> $DATA_DIR"
./build/dt_gen_value_data -n "$GAMES" -s 42 -a 300 --opener LITRE \
    -o "$DATA_DIR" --log-every 5000 2>&1 | tee "$DATA_DIR.gen.log" >/dev/null

T1=$(date +%s)
echo "    data done in $((T1 - T0))s"

echo
echo "[2/3] Training MLP -> $NET_OUT"
"$PYTHON" scripts/train_value_net.py \
    --data "$DATA_DIR" --out "$NET_OUT" \
    --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" --hidden "$HIDDEN" \
    2>&1 | tee "$DATA_DIR.train.log" | grep -E "best|val_L1|loaded|device"

T2=$(date +%s)
echo "    train done in $((T2 - T1))s"

echo
echo "[3/3] Bench: baseline   vs   value-net lookahead"

base_out=$(./build/dt_bench -n "$BENCH_N" -s "$BENCH_SEED" -S greedy --opener LITRE -a 300 2>&1)
echo "$base_out" | tail -12

echo
vn_out=$(./build/dt_bench -n "$BENCH_N" -s "$BENCH_SEED" -S greedy --opener LITRE -a 300 \
    --la-k "$LA_K" --la-n "$LA_N" --value-net "$NET_OUT" 2>&1)
echo "$vn_out" | tail -12

T3=$(date +%s)
echo "    bench done in $((T3 - T2))s"

# Pull mean + distribution one-liners for the log.
parse_mean()  { echo "$1" | awk '/Mean guesses/{print $3; exit}'; }
parse_max()   { echo "$1" | awk '/Min . Max/{print $NF; exit}'; }
parse_dist()  { echo "$1" | awk '/^   3[0-9]:/{printf "%s%s ", $1, $2}'; }

BASE_MEAN=$(parse_mean "$base_out")
BASE_MAX=$(parse_max  "$base_out")
BASE_DIST=$(parse_dist "$base_out")
VN_MEAN=$(parse_mean "$vn_out")
VN_MAX=$(parse_max   "$vn_out")
VN_DIST=$(parse_dist "$vn_out")

{
    echo "$(date -Iseconds)  games=$GAMES epochs=$EPOCHS hidden=$HIDDEN bench_n=$BENCH_N la=$LA_K/$LA_N"
    echo "  baseline   mean=$BASE_MEAN max=$BASE_MAX  $BASE_DIST"
    echo "  value-net  mean=$VN_MEAN max=$VN_MAX  $VN_DIST"
    echo "  delta_mean = $(python3 -c "print(round(float('$VN_MEAN') - float('$BASE_MEAN'), 3))")"
    echo "  elapsed: gen=$((T1-T0))s train=$((T2-T1))s bench=$((T3-T2))s"
    echo
} | tee -a "$LOG"

echo "==== done $(date -Iseconds) ===="

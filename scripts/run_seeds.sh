#!/usr/bin/env bash
# usage: scripts/run_seeds.sh [train_rows] [epochs] [lr]  (defaults: 5000 20 0.1)
# runs 5 copies of mnist with that config in parallel, then prints mean and sample std
# of their FINAL test10k accuracies. weight init is random per run, the shuffle seed is fixed
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

train_rows=${1:-5000}
epochs=${2:-20}
lr=${3:-0.1}
echo "config: train_rows $train_rows  epochs $epochs  lr $lr"

pids=()
for i in 1 2 3 4 5; do
    ./build/mnist "$train_rows" "$epochs" "$lr" > "logs/seed_$i.txt" 2>&1 &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
    echo "at least one run failed, check logs/" >&2
    exit 1
fi

for i in 1 2 3 4 5; do
    grep '^FINAL test10k' "logs/seed_$i.txt" | awk '{print $3}'
done | awk '
    { x[NR] = $1; sum += $1 }
    END {
        if (NR != 5) { print "expected 5 FINAL lines, got " NR > "/dev/stderr"; exit 1 }
        mean = sum / NR
        for (i = 1; i <= NR; i++) { ss += (x[i] - mean) ^ 2 }
        printf "mean %.4f  std %.4f  (n=%d)\n", mean, sqrt(ss / (NR - 1)), NR
    }'

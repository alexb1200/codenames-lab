#!/bin/sh
# Run experiments in order, e.g. ./run.sh e01_transfer e02_model_tournament
# With no arguments, runs all of them. Each writes results/<name>.jsonl; timings go to results/run.log.
cd "$(dirname "$0")"
names="$*"
[ -z "$names" ] && names=$(ls experiments/e*.py | xargs -n1 basename | sed 's/\.py$//')
for name in $names; do
    start=$(date +%s)
    python3 -m "experiments.$name" || { echo "$name FAILED" >> results/run.log; exit 1; }
    echo "$name $(( $(date +%s) - start ))s" >> results/run.log
done

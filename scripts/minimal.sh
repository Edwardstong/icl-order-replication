#!/usr/bin/env bash
# The simplest full replication: one open model, three classification datasets,
# Section 3 at the paper's size (M=10, P=10, 500 test) plus one reduced Section 4
# cell (P=32, 250 dev) on AG News, then the comparison tables.
# Resumable: a cell whose summary.json exists is skipped. Override the model or
# batch size with MODEL=Qwen/Qwen2.5-1.5B BATCH=16 scripts/minimal.sh
# About 3 hours for Qwen2.5-0.5B on an M4-class laptop; roughly 3x for 1.5B.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
BATCH="${BATCH:-32}"
SLUG="${MODEL//\//__}"
mkdir -p results/logs
LOG="results/logs/minimal_${SLUG}.log"
echo "== $(date) model=$MODEL batch=$BATCH" | tee -a "$LOG"

for ds in ag_news dbpedia mmlu; do
  [ -f "data/$ds/meta.json" ] || $PY scripts/build_splits.py --dataset "$ds" 2>&1 | tee -a "$LOG"
done

for ds in ag_news dbpedia mmlu; do
  if [ -f "results/sensitivity/${SLUG}__${ds}/summary.json" ]; then
    echo "skip sensitivity $ds (done)" | tee -a "$LOG"; continue
  fi
  nice -n 10 $PY scripts/run_sensitivity.py --model "$MODEL" --dataset "$ds" --batch-size "$BATCH" 2>&1 | tee -a "$LOG"
done

if [ -f "results/search/${SLUG}__ag_news/summary.json" ]; then
  echo "skip search ag_news (done)" | tee -a "$LOG"
else
  nice -n 10 $PY scripts/run_search.py --model "$MODEL" --dataset ag_news --P 32 --n-dev 250 --batch-size "$BATCH" 2>&1 | tee -a "$LOG"
fi

$PY scripts/make_tables.py 2>&1 | tee -a "$LOG"
echo "== $(date) done; see results/tables/sensitivity.csv and search.csv" | tee -a "$LOG"

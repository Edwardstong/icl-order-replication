#!/usr/bin/env bash
# GPT-5-nano Section 3 cells through the OpenAI API, one after another.
# Resumable: finished cells are skipped, and every completion is cached, so an
# interrupted cell resumes at no extra cost. Workers per dataset keep the
# prompt tokens per minute under the account's limit (2M TPM at tier 1).
#   MODEL=gpt-5-nano scripts/run_openai_cells.sh            # ag_news dbpedia mmlu
#   scripts/run_openai_cells.sh ag_news                       # one dataset
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODEL="${MODEL:-gpt-5-nano}"
SLUG="${MODEL//\//__}"
mkdir -p results/logs
LOG="results/logs/openai_${SLUG}.log"
DATASETS=("$@"); [ ${#DATASETS[@]} -eq 0 ] && DATASETS=(ag_news dbpedia mmlu)
echo "== $(date) model=$MODEL datasets=${DATASETS[*]}" | tee -a "$LOG"
for ds in "${DATASETS[@]}"; do
  if [ -f "results/sensitivity/${SLUG}__${ds}/summary.json" ]; then
    echo "skip $ds (done)" | tee -a "$LOG"; continue
  fi
  case "$ds" in ag_news) w=24 ;; dbpedia|mmlu) w=16 ;; *) w=8 ;; esac
  echo "== $(date) start $ds workers $w" | tee -a "$LOG"
  $PY scripts/run_sensitivity.py --backend openai --model "$MODEL" --dataset "$ds" --workers "$w" 2>&1 | tee -a "$LOG" | grep -v "prefix/s\]" || true
done
echo "== $(date) all done" | tee -a "$LOG"

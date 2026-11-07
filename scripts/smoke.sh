#!/usr/bin/env bash
# End-to-end check on the smallest model: builds AG News, runs a 2x2 grid on 16
# test examples, then a tiny search. Takes a few minutes on a laptop; the first
# run also downloads Qwen2.5-0.5B (~1 GB) and AG News (~30 MB).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -f data/ag_news/meta.json ] || $PY scripts/build_splits.py --dataset ag_news
$PY scripts/run_sensitivity.py --model Qwen/Qwen2.5-0.5B --dataset ag_news --M 2 --P 2 --n-test 16 --tag smoke
$PY scripts/run_search.py --model Qwen/Qwen2.5-0.5B --dataset ag_news --M 2 --P 3 --n-dev 8 --n-test 8 --tag smoke
$PY scripts/make_tables.py

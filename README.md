# ICL order replication

Replication harness for **"Order Matters: Rethinking Prompt Construction in In-Context Learning"**
(Li, Wang, Wang, Shang; arXiv:2511.09700, Nov 2025). Course project checkpoint.

The paper asks whether *which* demonstrations go into a few-shot prompt matters more than
*what order* they are in. It measures both as standard deviations of accuracy over a grid of
M demonstration sets x P random orderings (Section 3), then shows that picking the best of P
random orderings on a dev set recovers most of an oracle's test accuracy (Section 4).

No code, prompt template, scoring rule or sampling procedure is released with the paper.
Everything below is our reconstruction; each choice is recorded in [METHODS.md](METHODS.md).

## Scope of this replication

| | Paper | Here |
|---|---|---|
| Models | 15 (0.5B to 27B, plus GPT-5-nano) | Qwen2.5-0.5B, Qwen2.5-1.5B, Llama-3.2-1B-Instruct, Gemma-2-2B; any HF causal LM works |
| Classification data | AG News, DBPedia, MMLU, NYT-Topics, NYT-Locations | AG News, DBPedia, MMLU (NYT needs an LDC license) |
| Generation data | GSM8K, MATH, MMLU-Pro | GSM8K only, and only with GPU time |
| Section 3 | M=10, P=10, 500 test | same, full size |
| Section 4 | M=10, P=128, 1000 dev, 500 test | M=10, P=32, 250 dev, 500 test (the paper's stated plateau); full size is a flag away |
| Hardware | not stated | laptops (CPU / Apple MPS) for classification; free GPU tiers for GSM8K |

## What needs the OpenAI API

Only the **GPT-5-nano** rows of Section 3: one row of Table 2 (ordering 0.0234, selection 0.0198,
r 0.85) and seven rows of Table 4 (one per dataset). The Section 4 experiments (Tables 5 to 7) use the
twelve open models only. GPT-5-nano is one of fifteen models in the paper's headline averages, so
leaving it out changes those averages little. Everything else in this harness runs on open weights
with no key.

If the team does want that row:

- **Subscription is not API credit.** A ChatGPT Plus/Pro subscription includes Codex usage when
  Codex is signed in with ChatGPT, but that quota pays for the Codex agent's own calls (its coding
  models, under 5-hour and weekly caps). It does not give API credit, and the paper's model is reached
  only through the API. The API is billed separately on platform.openai.com; the smallest top-up is $5.
  Pin the model id `gpt-5-nano` (the August 2025 model the paper used), not the newer `gpt-5.4-nano`,
  which is a different model at four times the price.
- **Cost.** One full Section 3 cell is M x P x N = 50,000 requests. List prices for gpt-5-nano
  (checked 2026-11-06): $0.05 per 1M input, $0.005 cached input, $0.40 per 1M output; the Batch API
  is about half that. AG News prompts (about 550 tokens, under the 1,024-token caching threshold)
  cost about $1.8 per cell synchronously; MMLU and DBPedia prompts are longer but mostly cached, about
  $1.2 each. All three classification cells: about $5 synchronously, about $2.50 through the Batch
  API. GSM8K is generation with a few hundred output tokens per call, about $6 to $8 per cell. The
  risk is reasoning tokens: set `reasoning_effort` to minimal and measure one small batch first;
  200 hidden reasoning tokens per call would add about $4 per cell.
- **Method difference.** We could not confirm whether gpt-5-nano returns token log-probabilities;
  third-party listings disagree. The backend should request `logprobs` once and fall back to scoring
  by generating the label text and matching it to a label name. The paper does not say how it scored
  GPT-5-nano. Record whatever the team does in METHODS.md.
- **Not implemented here.** An OpenAI backend would be a new `iclorder/openai_scorer.py` exposing the
  same `predict`-style interface and reusing `iclorder/prompts.py`, with a Batch-API path (one JSONL
  of requests per cell, results collected within 24 hours), plus a `--backend openai` flag on the two
  run scripts. The key lives in the `OPENAI_API_KEY` environment variable of whoever runs it and is
  never written into the repo.

## Why this fits on laptops

For one demonstration set and one ordering, the demonstrations are identical for every
test query. `iclorder/scorer.py` encodes that prefix once, keeps its key/value cache, and runs
only the short query tokens through the model, in batches. Classification needs no text
generation: the label is read from the next-token distribution over the label words. Each test
example then costs one forward pass of ~50 tokens on a model under 2B parameters.

Measured: 46 queries/s for Qwen2.5-0.5B on an M4 Pro (bf16, batch 32, 8-shot AG News prefixes of
~490 tokens). That makes one full Section 3 cell (50,000 evaluations) about 18 minutes, and one
reduced Section 4 cell (P=32, 250 dev + 500 test = 240,000 evaluations) about 90 minutes, for the
0.5B model. Budget roughly 3x for the 1.5B-2B models and 2x for DBPedia's 28-demonstration prefixes:
Section 3 on four models x three datasets is an overnight job on one laptop; Section 4 on two
datasets is a day or two, or a few hours split across the team.
Generation (GSM8K) needs batched decoding; use Colab / Kaggle free GPUs, ideally with vLLM.

## Quickstart

```bash
python3.13 -m venv .venv            # 3.11+ works
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q        # unit tests, no downloads
```

Gated weights (Llama 3.x, Gemma) need a Hugging Face account that accepted each model's license:
```bash
.venv/bin/hf auth login
```

Build data, then run. Every script has `--help`.
```bash
.venv/bin/python scripts/build_splits.py --dataset ag_news        # also: dbpedia, mmlu, gsm8k
.venv/bin/python scripts/run_sensitivity.py --model Qwen/Qwen2.5-0.5B --dataset ag_news
.venv/bin/python scripts/run_search.py      --model Qwen/Qwen2.5-0.5B --dataset ag_news --P 32 --n-dev 250
.venv/bin/python scripts/make_tables.py                            # results/tables/*.csv next to the paper's numbers
```
`scripts/smoke.sh` runs a tiny end-to-end check (2x2 grid, 16 examples) on Qwen2.5-0.5B.

Full-size Section 4, when GPU time allows:
```bash
.venv/bin/python scripts/run_search.py --model Qwen/Qwen2.5-0.5B --dataset ag_news --P 128 --n-dev 1000 --batch-size 64
```

## Layout

```
iclorder/            the library (specs, data, prompts, scorer, metrics, search, runner, parse)
scripts/             build_splits, run_sensitivity, run_search, make_tables, smoke.sh
tests/               unit tests for every formula and sampling rule (pytest)
paper/               the paper's Tables 2 and 4 as CSV, and the model-id mapping
docs/PAPER_NOTES.md  the paper's definitions and procedure, extracted, with its internal inconsistencies
METHODS.md           every detail the paper leaves unstated and the choice we froze
data/                materialized splits (gitignored; rebuilt by build_splits.py, seeded)
results/             one folder per run with summary.json (kept) and .npy arrays (gitignored)
```

Each run folder is self-contained: `summary.json` holds the config, the metrics, the environment
and timing; `perms.json` and `demo_sets.json` hold the exact orderings and example ids, so any
cell can be recomputed. Team members run disjoint (model, dataset) slices and merge the folders.

## Replication structure

Each phase has a command, an output folder and the paper number it is compared with. Phases 2 and 3
are the checkpoint; 4 and 5 are optional extensions.

| Phase | What | How | Output | Paper target |
|---|---|---|---|---|
| 0 Setup | venv, tests, smoke run | Quickstart above | passing `pytest`, `results/*__smoke` | none |
| 1 Data | balanced 1,000 dev + 500 test + demo pool per dataset | `scripts/build_splits.py --dataset <d>` | `data/<d>/{demo_pool,dev,test}.jsonl`, `meta.json` | Table 1 sizes |
| 2 Section 3 | sensitivity grid, M=10 x P=10 x 500, 4 models x 3 datasets (12 cells) | `scripts/run_sensitivity.py --model <m> --dataset <d>` | `results/sensitivity/<m>__<d>/summary.json` (ordering, selection, r) | Table 4 cells; Table 2 rows (ours average 3 datasets, theirs 7); classification mean r = 1.09 |
| 3 Section 4 | best-of-P ordering search, P=32, 250 dev, AG News + DBPedia, 4 models | `scripts/run_search.py --model <m> --dataset <d> --P 32 --n-dev 250` | `results/search/<m>__<d>/summary.json` (highest-dev, average, oracle, fraction, curves over P and dev size) | over 98% of oracle; plateau at 16 to 32 orderings and about 250 dev examples |
| 4 Generation | GSM8K Section 3 on two models, GPU time | same `run_sensitivity.py` on Colab / Kaggle | `results/sensitivity/<m>__gsm8k/` | Table 4 GSM8K cells; generation mean r = 1.46 |
| 5 GPT-5-nano | Section 3 on three datasets through the OpenAI API | backend not written; see the section above | `results/sensitivity/gpt-5-nano__<d>/` | Table 2 row, Table 4 rows |
| 6 Report | tables and write-up | `scripts/make_tables.py` | `results/tables/sensitivity.csv`, `search.csv` | side by side with `paper/*.csv` |

Rough wall-clock for phase 2 on one M4-class laptop, from the measured 46 queries/s on the 0.5B model:
0.5B about 1.5 h for three datasets, 1.5B about 4.5 h, Llama-3.2-1B about 3.5 h, Gemma-2-2B about 6 h.
Phase 3 on two datasets is about 90 minutes per 0.5B cell and 3 to 6 h per larger-model cell.

Report outline: (1) the claim and the two sensitivity definitions; (2) what we ran against what the
paper ran, with the frozen choices from METHODS.md; (3) Section 3 cells beside the paper's Table 4,
and whether selection sensitivity is at least ordering sensitivity in ours; (4) Section 4 fraction of
oracle and the two curves; (5) discrepancies and limits: no GPT-5-nano, no NYT, reduced P and dev,
the paper's internal inconsistencies; (6) reproducibility: seeds, commands, run folders.

## Team split, suggested

| Who | Runs |
|---|---|
| A | build all splits; Section 3 on Qwen2.5-0.5B and 1.5B, three datasets |
| B | Section 3 on Llama-3.2-1B-Instruct and Gemma-2-2B, three datasets |
| C | Section 4 (P=32, dev 250) on AG News and DBPedia, four models |
| D | GSM8K on Colab: build splits, Section 3 on two models; write up |

## Status

The shortest path to both claims is in [docs/ROADMAP.md](docs/ROADMAP.md); `scripts/minimal.sh` runs it.

- [x] Library, scripts, 24 unit tests, 2 model-loading integration tests (`ICL_RUN_MODEL_TESTS=1 pytest tests/integration`)
- [x] Smoke test: Qwen2.5-0.5B on AG News on Apple MPS, 46 queries/s (numbers at the bottom of METHODS.md)
- [ ] Section 3, four models x three datasets
- [ ] Section 4, reduced
- [ ] GSM8K on GPU time
- [ ] Report: our cells next to the paper's Table 4, curves, list of choices and discrepancies

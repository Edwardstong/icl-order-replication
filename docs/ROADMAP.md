# Roadmap: the simplest replication

Goal: reproduce the paper's two claims with the least compute, then widen only if time allows.

1. Selection sensitivity is at least as large as ordering sensitivity on average (Section 3; the
   paper's classification mean r is 1.09).
2. Picking the best of P random orderings on a dev set recovers over 98% of the oracle test
   accuracy, with a plateau near 16 to 32 orderings and about 250 dev examples (Section 4).

Exact numbers will not match: the paper's prompt template, sampling and scoring are unstated
(METHODS.md). What can be checked is the sign and size of r per cell, the mean r, and the
fraction of oracle.

## Why three datasets, not one

In the paper every small model has ordering > selection on AG News (r below 1) and the opposite on
DBPedia and MMLU. One AG News cell therefore cannot show the headline result; three datasets on one
model can. The targets for the ungated models:

| Model | Dataset | Paper ordering | Paper selection | Paper r |
|---|---|---|---|---|
| Qwen2.5-0.5B | AG News | 0.04581 | 0.03704 | 0.81 |
| Qwen2.5-0.5B | DBPedia | 0.00835 | 0.01783 | 2.14 |
| Qwen2.5-0.5B | MMLU | 0.00996 | 0.01384 | 1.39 |
| Qwen2.5-1.5B | AG News | 0.00911 | 0.00729 | 0.80 |
| Qwen2.5-1.5B | DBPedia | 0.02336 | 0.02324 | 0.99 |
| Qwen2.5-1.5B | MMLU | 0.01051 | 0.01358 | 1.29 |

(r here is selection / ordering computed from the Table 4 cells; the paper's Table 2 r of 1.1 for
both models averages over seven datasets.)

## Steps

| Step | What | Command | Time (M4-class laptop) | Done when |
|---|---|---|---|---|
| 1 | Build DBPedia and MMLU splits (AG News exists) | `scripts/build_splits.py --dataset dbpedia`, `--dataset mmlu` | 5 min, one download each | `data/<d>/meta.json` |
| 2 | Section 3, Qwen2.5-0.5B x 3 datasets, full size | `scripts/run_sensitivity.py --model Qwen/Qwen2.5-0.5B --dataset <d> --batch-size 32` | about 1.5 h total | three `results/sensitivity/Qwen__Qwen2.5-0.5B__<d>/summary.json` |
| 3 | Section 4, Qwen2.5-0.5B x AG News, reduced | `scripts/run_search.py --model Qwen/Qwen2.5-0.5B --dataset ag_news --P 32 --n-dev 250 --batch-size 32` | about 1.5 h | `results/search/.../summary.json` |
| 4 | Tables | `scripts/make_tables.py` | seconds | `results/tables/sensitivity.csv`, `search.csv` beside the paper's numbers |
| 5 | Second model: repeat 2 for Qwen2.5-1.5B | `MODEL=Qwen/Qwen2.5-1.5B BATCH=16 scripts/minimal.sh` | about 5 h, overnight | six Section 3 cells, mean r over them |
| 6 | Write-up | outline in README | an afternoon | report |

`scripts/minimal.sh` runs steps 1 to 4 in one go and is resumable (done cells are skipped).

## Success criteria for the checkpoint

- Step 2: r below 1 on AG News and above 1 on DBPedia and MMLU, as in the paper, and mean r over
  the three above 1. Report our cells next to the paper's in one table.
- Step 3: highest-dev ordering reaches at least 98% of the oracle test accuracy; the curve over P
  flattens by 16 to 32; the curve over dev size flattens by about 250.
- Any disagreement is a finding, not a failure: say which unstated choice (template, scoring,
  sampling) could explain it.

## Only if time allows

- Gated models (Llama-3.2-1B-Instruct, Gemma-2-2B): `hf auth login` first, then the same script
  with `MODEL=...`. Adds two Table 4 rows.
- Section 4 on DBPedia (about 3 h for 0.5B).
- GSM8K (generation): build the split here, run on Colab or Kaggle GPU time; shows the
  generation-side r (paper mean 1.46).
- GPT-5-nano row: about $5 of API credit through the Batch API; needs the backend described in the
  README's OpenAI section.

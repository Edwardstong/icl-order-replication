# Paper notes: arXiv:2511.09700 v1

"Order Matters: Rethinking Prompt Construction in In-Context Learning". Warren Li, Yiqian Wang,
Zihan Wang, Jingbo Shang. Submitted 12 Nov 2025. cs.CL. CC BY 4.0. Read from the arXiv HTML
rendering; the table values were then checked against the PDF text (see paper/README.md).

## Claim

Selection sensitivity and ordering sensitivity are of the same size (selection ~14% larger on
average: 0.02251 vs 0.01970). r = selection / ordering averages 1.09 on classification and 1.46
on generation. Choosing the best of P random orderings on a 1,000-example dev set recovers over
98% (classification, 16-32 orderings) and over 95% (generation, 64-100) of the oracle test accuracy.
Orderings tuned on one dataset transfer poorly to another (mean transfer ratio 0.798, Table 7).

## Definitions (Sec. 3.2)

a_{i,j} = Acc(S_i, pi_j | D_test), S_i the i-th demonstration set (i = 1..M), pi_j the j-th permutation
(j = 1..P), applied to the set's default ordering.

- Ordering sensitivity  sigma_M = (1/M) sum_i std_j a_{i,j}      (row-wise std, averaged)
- Selection sensitivity sigma_P = (1/P) sum_j std_i a_{i,j}      (column-wise std, averaged)
- r = sigma_P / sigma_M; r > 1 selection dominates, r < 1 ordering dominates.
- Default ordering: classification groups by label, labels alphabetical, examples alphabetical within
  label; generation sorts all examples alphabetically.
- M = 10, P = 10 "for the computational consideration". N_test = 500.

## Section 4 procedure (Sec. 4.1, Algorithm 1)

P = 128 permutations "generated uniformly at random", M = 10 sets, N_dev = 1000, N_test = 500.
Reported: highest-dev (ordering with the best dev accuracy, then evaluated on test), average over
all P, and max over P on test (oracle). Dev sweep 50 to 1000 examples, plateau "after around 250".
The text picks the best ordering by dev accuracy averaged over the M sets; Algorithm 1 picks it per set.

## Data (Table 1)

| Dataset | Task | \|C\| | k | Source |
|---|---|---|---|---|
| AG News | classification | 4 | 8 | Zhang et al. 2015 |
| NYT-Topics | classification | 9 | 18 | LDC2008T19 (licensed) |
| NYT-Locations | classification | 10 | 20 | LDC2008T19 (licensed) |
| DBPedia | classification | 14 | 28 | Auer et al. 2007 |
| MMLU | classification | 4 | 8 | Hendrycks et al. 2020 |
| GSM8K | generation | - | 8 | Cobbe et al. 2021 |
| MMLU-Pro | generation | - | 8 | Wang et al. 2024 |
| MATH | generation | - | 8 | Hendrycks et al. 2021 |

Every dataset: 1,000 dev, 500 test. Classification balanced by oversampling underrepresented labels.
Not stated: splits used, how examples were sampled, demonstration pool, MMLU's four labels.

## Models

Open weights (ids printed in Tables 5-7): Qwen/Qwen2.5-{0.5B,1.5B,3B,7B}; google/gemma-2b,
google/gemma-2-2b, google/gemma-7b; meta-llama/Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct,
Llama-3.1-8B-Instruct; deepseek-ai/DeepSeek-R1-Distill-Qwen-{1.5B,7B}. Tables 2-4 also include
Gemma-2 9B, Gemma-3 27B and GPT-5-nano (no ids). Section 4 tables use the 12 open models only
(Table 6 drops google/gemma-7b). Not stated: inference framework, precision, decoding, prompt
template, GPUs, how GPT-5-nano was accessed. "We used OpenAI's GPT-o3 and o4 models to assist
with writing code and editing the paper" (authoring only).

## Where the target numbers live

`paper/table2_sensitivity_by_model.csv`, `paper/table4_sensitivity_by_dataset.csv`. Spot-check
against the PDF before quoting.

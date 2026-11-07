# Replicating "Order Matters: Rethinking Prompt Construction in In-Context Learning"

Replication report for arXiv:2511.09700 (Li, Wang, Wang, Shang, November 2025).
Course project checkpoint. Team: *[names]*. Course: *[course]*. Date: *[date]*.

Code, data recipes, run folders and this report live in one repository; every number in
Section 4 is produced by `scripts/fill_report.py` from the run folders under `results/`, so the
report and the code cannot drift apart.

## 1. In-context learning

A large language model can perform a task it was never fine-tuned for if the prompt shows it a
few worked examples first. This is **in-context learning (ICL)**, or few-shot prompting: the prompt
is k *demonstrations*, each an input with its answer, followed by the test input, and the model's
continuation is read as the prediction. No weights change. For a classification task with four
labels, an 8-shot prompt looks like

```
Article: <news text>
Topic: Sports

Article: <news text>
Topic: Business

... (6 more)

Article: <test news text>
Topic:
```

and the prediction is whichever label word the model finds most likely next.

ICL is how most practitioners adapt a model: it needs no training data pipeline, no GPU for
training, and it works through an API. Its weakness is that the result depends on how the
prompt was built. Two choices are known to move accuracy by several points: **which** examples
are shown (selection) and the **order** they are shown in (ordering). Earlier work found that
some orderings of the same examples are near state of the art and others near chance (Lu et al.,
2022), that models favour labels that appear late in the prompt or often (Zhao et al., 2021),
and that retrieving examples similar to the test input helps (Liu et al., 2022; Rubin et al.,
2022). Most of that work treats selection as the main lever and ordering as noise to be
calibrated away.

## 2. Why this paper, and why we want its result

The paper asks a plain question: **is ordering really the smaller effect?** It measures both on
the same footing. For a model and dataset it draws M = 10 demonstration sets and P = 10 random
orderings and evaluates every pair on 500 test examples, giving an accuracy grid a[i, j]. Then

- **ordering sensitivity** = mean over sets i of the standard deviation of a[i, :] over orderings;
- **selection sensitivity** = mean over orderings j of the standard deviation of a[:, j] over sets;
- **r** = selection sensitivity / ordering sensitivity. r > 1 means selection matters more.

Across 15 models and 7 datasets the paper finds the two are of the same size: selection 0.0225
against ordering 0.0197 on average, r = 1.09 on classification and 1.46 on generation. Ordering is
not noise; it is roughly half the prompt-construction effect.

The paper then shows that ordering is **cheap to exploit**. Sample P random orderings of a fixed
demonstration set, score each on a small dev set, keep the best. With 16 to 32 orderings and about
250 dev examples the chosen ordering recovers over 98% of the oracle test accuracy (the best
ordering in hindsight) on classification, and over 95% on generation. That is a practical recipe
with no retriever, no embeddings and no training: a few hundred extra forward passes per task.

We want this result for two reasons. First, if it holds on small open models it changes how we
would build prompts for any project: spend a dev set on ordering search before spending effort on
example retrieval. Second, the paper releases no code, no prompt template and no sampling
procedure, so a replication is also a test of whether the claim survives someone else's
reasonable choices. A claim that only holds under one unstated template is a weaker claim.

## 3. How we make it work without GPUs

### 3.1 The cost problem and the trick that removes it

One Section 3 cell is M x P x N_test = 50,000 prompt evaluations, each an 8-shot prompt of about
500 tokens. Done naively that is 25 million tokens through the model per cell, and the paper used
14 open models up to 27B parameters on GPUs. We have laptops.

Two observations make the classification experiments fit on a laptop.

1. **The demonstrations are the same for every test example.** For a fixed (set, ordering) the
   prompt is `prefix + query`, and only the short query changes. We run the prefix through the model
   once, keep its key/value cache, and score all 500 queries against that cache in batches. Each
   test example then costs a forward pass over about 50 tokens instead of 550. `iclorder/scorer.py`
   does this; an integration test checks that the cached log-probabilities match full-prompt
   scoring to within 1e-3.
2. **Classification needs no text generation.** The label is read from the next-token distribution
   over the label words (World, Sports, Business, Technology for AG News; the answer letters A to D
   for MMLU). One forward pass, no decoding loop.

Measured on an Apple M4 Pro laptop with Qwen2.5-0.5B in bfloat16: 46 queries per second, so a full
Section 3 cell takes about 16 minutes and a reduced Section 4 cell (P = 32, 250 dev + 500 test)
about 90 minutes. Models of 1.5B to 2B parameters cost about three times that. Generation tasks
(GSM8K) still need decoding and are left to free GPU tiers.

### 3.2 What we run

| | Paper | Our replication |
|---|---|---|
| Models | 15, 0.5B to 27B plus GPT-5-nano | Qwen2.5-0.5B first; Qwen2.5-1.5B, Llama-3.2-1B-Instruct, Gemma-2-2B as time allows |
| Classification data | AG News, DBPedia, MMLU, NYT-Topics, NYT-Locations | AG News, DBPedia, MMLU (the NYT corpora need an LDC licence) |
| Generation data | GSM8K, MATH, MMLU-Pro | GSM8K only, on GPU time, optional |
| Section 3 grid | M = 10, P = 10, 500 test | identical |
| Section 4 search | P = 128, 1,000 dev, 500 test | P = 32, 250 dev, 500 test (the paper's own plateau); full size is a flag |

Three datasets matter even for one model: in the paper every small model has ordering > selection
on AG News and the opposite on DBPedia and MMLU, so a single AG News cell could not show the
headline result.

### 3.3 Choices the paper leaves open

The paper states the metric and the grid sizes but not the prompt template, the label words, how
the label is read off, the demonstration pool, how dev and test were sampled, or the seeds. Each
of these is a decision we had to make; the full table with code locations is `METHODS.md`. The ones
most likely to move numbers:

| Detail | Our choice |
|---|---|
| Template | `<input>\n<Key>: <answer>` per demonstration, blank line between, query ends with `<Key>:`; no chat template, even for instruct checkpoints |
| Label reading | argmax of next-token log-probability over the label words' first tokens; label words chosen so first tokens are distinct |
| Demonstration sets | 2 examples per label without replacement (k = 2\|C\| as in the paper), drawn from the train split minus dev/test |
| Dev / test | balanced per label, disjoint; 1,000 dev, 500 test |
| Default ordering | grouped by label alphabetically, examples alphabetical within label (as stated); random orderings are distinct uniform permutations of that |
| Standard deviation | population (ddof = 0) |
| Highest-dev rule | the paper describes it two ways; we report both: best ordering by dev accuracy pooled over sets, and per set |

We also found six internal inconsistencies in the paper (abstract vs body on GPT-5 vs GPT-5-nano,
a "most sensitive" claim contradicted by its own table, two definitions of the Section 4 rule,
missing table cells, an undefined transfer ratio); they are listed in `METHODS.md`.

### 3.4 Verification before spending compute

- 24 unit tests cover the two sensitivity formulas, the Section 4 aggregates, balanced sampling,
  permutation generation and prompt layout.
- The cache-equivalence test runs the real 0.5B model on CPU in float32 and compares cached against
  full-prompt scoring.
- The paper's Tables 2 and 4 were transcribed to CSV and all 255 values checked against the PDF
  text, so the comparison targets are exact.
- A smoke run (2 x 2 grid, 16 examples) exercises the whole pipeline in a few minutes.

## 4. Our results

What we compare is the sign and size of r per cell, the mean r, and the fraction of oracle, not
exact numbers: with the template and sampling unstated, exact agreement would be a coincidence.

<!-- RESULTS:BEGIN -->
<!-- generated by scripts/fill_report.py on 2026-11-06 17:34 PST; do not edit by hand -->

*No finished runs yet. Run `scripts/minimal.sh`, then `scripts/fill_report.py`.*
<!-- RESULTS:END -->

## 5. Discussion

*To be completed once Section 4 above is filled.* Points to cover:

- Does selection sensitivity exceed ordering sensitivity on average in our cells, as the paper
  claims? Which cells agree with the paper on which effect dominates, and which do not?
- Does the best-of-P ordering reach 98% of the oracle, and where do the curves over P and dev size
  flatten?
- Where we disagree with the paper, which unstated choice could explain it: the template, the
  log-probability reading of the label (the paper may have generated text), the label words, or
  the demonstration pool.
- Limits of this replication: one to four small models instead of 15, three datasets instead of
  seven, no generation task unless GPU time was found, reduced P and dev size in Section 4, no
  GPT-5-nano row (about $5 of API credit and a small backend would add it).

## 6. Reproducing this report

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                 # unit tests, no downloads
scripts/minimal.sh                            # splits, three Section 3 cells, one Section 4 cell, tables
.venv/bin/python scripts/fill_report.py       # rewrites Section 4 of this file and report/figures/
```

Every run folder under `results/` holds `summary.json` (configuration, metrics, environment,
timing), `perms.json` (the exact orderings) and `demo_sets.json` (the exact example ids), so any
cell can be recomputed. Gated models need `.venv/bin/hf auth login` after accepting the licence.

## References

- Li, W., Wang, Y., Wang, Z., Shang, J. (2025). Order Matters: Rethinking Prompt Construction in
  In-Context Learning. arXiv:2511.09700.
- Lu, Y., Bartolo, M., Moore, A., Riedel, S., Stenetorp, P. (2022). Fantastically Ordered Prompts and
  Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity. ACL.
- Zhao, Z., Wallace, E., Feng, S., Klein, D., Singh, S. (2021). Calibrate Before Use: Improving Few-Shot
  Performance of Language Models. ICML.
- Liu, J., Shen, D., Zhang, Y., Dolan, B., Carin, L., Chen, W. (2022). What Makes Good In-Context
  Examples for GPT-3? DeeLIO Workshop.
- Rubin, O., Herzig, J., Berant, J. (2022). Learning to Retrieve Prompts for In-Context Learning. NAACL.
- Brown, T. et al. (2020). Language Models are Few-Shot Learners. NeurIPS.

# Methods: what the paper leaves unstated, and what we do

The paper (arXiv:2511.09700 v1) releases no code and no prompt. Every row below is a decision
the replication had to make. Change a row only by editing the code it points to, bumping the
run `--tag`, and noting it here. The report lists this table.

## Prompt and scoring

| Detail | Paper | Here | Where |
|---|---|---|---|
| Prompt template | "Prompts include k demonstrations followed by a free-form test query." Nothing else. | `<input>\n<Key>: <answer>` per demonstration, blank line between, query ends with `<Key>:`. Keys: Topic (AG News), Category (DBPedia), Answer (MMLU, GSM8K). | `iclorder/prompts.py` |
| Chat template / system prompt | not stated | None. Raw completion mode for every model, instruct checkpoints included. Keeps the prompt identical across models. | `scorer.py` (tokenizer called directly) |
| Label verbalizers | not stated | One word per label, chosen so first tokens are distinct under each tokenizer; the scorer refuses to run otherwise. AG News: World, Sports, Business, Technology. DBPedia: Company, School, Artist, Athlete, Politician, Vehicle, Building, Nature, Village, Animal, Plant, Album, Film, Book. MMLU: A, B, C, D. | `specs.py` |
| Classification prediction | "accuracy"; extraction method not stated | argmax of next-token log-probability over the label words' first tokens. No generation. | `scorer.score_labels` |
| MMLU as classification | "|C| = 4", labels not named | The four answer letters A-D. Question and options are the input. | `specs.py` |
| Generation decoding | not stated | greedy, `max_new_tokens=256`, continuation cut at `\n\nQuestion:` or a triple newline | `scorer.generate`, `parse.cut_at_stop` |
| "Exact match or numeric tolerance" | tolerance not stated | number after `####` if present, else the last number in the continuation; match if abs diff <= 1e-6 + 1e-4 * abs(gold) | `parse.py` |
| BOS token | not stated | added to the prefix where the tokenizer adds one; never to queries | `scorer.encode_prefix` |

## API backend (`--backend openai`, GPT-5-nano)

| Detail | Paper | Here | Where |
|---|---|---|---|
| Prompt format | not stated | chat: system line "Reply with exactly one of these labels and nothing else: <labels>", user message = the same demonstrations-plus-query text as the local backend | `openai_scorer.OpenAIScorer.classification_instruction` |
| Label reading | not stated | first allowed label mentioned as a whole word in the completion (case and punctuation ignored); no label mentioned counts as wrong and is tallied as `unmatched` | `openai_scorer.match_label` |
| Decoding | not stated | API default; GPT-5 models accept no temperature, so cells are not bit-reproducible. `reasoning_effort=minimal`, 64 completion tokens for classification | `scripts/_common.py` flags |
| Generation (GSM8K) | not stated | system line asks for the examples' style and a final `#### <number>` line; the same parser as the local backend | `OpenAIScorer.generation_instruction`, `parse.py` |
| Caching | n/a | every completion stored by request hash in sqlite; a rerun reuses outputs, so a cell's numbers are stable after its first run | `openai_scorer._Cache` |
| Cost accounting | n/a | prompt, cached-prompt, completion and reasoning tokens summed from the API's usage field; cost at list prices in `summary.json` | `OpenAIScorer.usage_summary` |

## Sampling

| Detail | Paper | Here | Where |
|---|---|---|---|
| Demonstration pool | not stated | train split (MMLU: validation split), minus any example used in dev/test | `data.build_splits` |
| Demonstration sets (M=10) | "M distinct example sets" | classification: 2 examples per label without replacement (k = 2\|C\| makes this the natural reading); generation: k=8 uniform. Independent draws per set; identical sets redrawn. | `data.sample_demo_sets` |
| Dev / test | "held-out", N_dev=1000, N_test=500, classification "balanced across classes by oversampling underrepresented labels" | test from the eval split, dev from what is left of it; if the eval split is too small (GSM8K: 1,319 rows) dev comes from the train split. Per-label quota n // \|C\| with remainder to the first labels; oversample only when a label's pool is short. All three files disjoint. | `data.balanced_sample`, `data.build_splits` |
| Default ordering | classification: group by label, labels alphabetical, examples alphabetical within label; generation: all alphabetical | as stated; alphabetical by verbalizer then by input text | `prompts.default_order` |
| Random permutations | "uniformly at random" | uniform, duplicates rejected so P permutations are P distinct orderings; the identity is allowed | `prompts.random_permutations` |
| Same permutation across sets | notation a_{i,j} suggests pi_j is shared | pi_j is the same index permutation applied to each set's default ordering | `runner.evaluate_grid` |
| Seeds | not stated | `--seed` (default 0): demo sets use seed, permutations use seed+1, splits use their own `--seed` at build time | scripts |

## Statistics

| Detail | Paper | Here | Where |
|---|---|---|---|
| Standard deviation | population or sample not stated | population (ddof=0); `--ddof 1` available | `metrics.sensitivities` |
| Headline averages | "across all model-dataset combinations" | plain mean over run folders (make_tables) | `scripts/make_tables.py` |
| "highest-dev" rule (Sec. 4) | text: best ordering by dev accuracy averaged over the M sets; Algorithm 1: best ordering per set | both computed and reported: `highest_dev_pooled` and `highest_dev_per_set` | `search.aggregates` |
| Oracle | max test accuracy over P orderings | per set, then averaged over sets (Algorithm 1) | `search.aggregates` |
| Curves over P and dev size | figures only | recomputed from stored per-example correctness using the first p permutations / first n dev examples | `search.curve_over_p`, `curve_over_dev_size` |

## Known inconsistencies in the paper (to list in the report)

1. Abstract says GPT-5; body and limitations say GPT-5-nano only.
2. Sec. 3.3 calls GPT-5-nano "the most sensitive to ordering", but Table 2 shows three models with
   larger ordering values (DeepSeek 1.5B 0.0362, DeepSeek 7B 0.0287, Llama 3 8B 0.0277). Only its r (0.85) is lowest.
3. Sec. 3.3 says classification r lies in 0.93-1.20; Table 3's values run 0.97-1.20.
4. The Section 4 selection rule is described two ways (pooled over sets in the text, per set in Algorithm 1).
5. MMLU-Pro is listed as a dataset but absent from Tables 3 and 4; six models have blank MMLU-Pro and MATH cells in Table 6.
6. Table 7's "Transfer Ratio" is never defined.

## What would move the numbers most

Verbalizer choice and template for the small models (0.5B-2B are template-sensitive), the
log-prob versus generated-text reading of the label, and the ddof choice (ddof=1 inflates both
sensitivities by sqrt(P/(P-1)) ~ 5% at P=10 but leaves r unchanged).

## Smoke test record (2026-11-06, Apple M4 Pro, 24 GB, MPS, bfloat16)

Qwen/Qwen2.5-0.5B on AG News, transformers 5.22, torch 2.15.

| Run | Grid | Result |
|---|---|---|
| sensitivity, tag `smoke` | M=2, P=2, 16 test | mean acc 0.625; per-example correctness differs between sets and orderings |
| sensitivity, tag `timing` | M=2, P=2, 128 test, batch 32 | mean acc 0.740; ordering 0.0137, selection 0.0137; 512 evaluations in 11.2 s = 46 queries/s; mean prefix 492 tokens |
| search, tag `smoke` | M=2, P=3, 8 dev, 8 test | pooled = per-set = oracle = 0.8125, average 0.771 |
| `tests/integration/test_scorer_equivalence.py` | CPU, float32 | cached-prefix log-probs match full-prompt log-probs within 1e-3 on queries of different lengths |

Grids this small say nothing about the paper's numbers; they show the pipeline runs and the
cache is exact. At 46 queries/s a full Section 3 cell (M=10, P=10, 500 test = 50,000 evaluations)
is ~18 min for the 0.5B model on this machine; expect ~3x for 1.5B-2B models and ~2x for DBPedia's
28-demonstration prefixes.

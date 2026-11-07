# Methods: what the paper leaves unstated, and what we do

The paper (arXiv:2511.09700 v1) releases no code and no prompt. Every row below is a decision this
replication had to make for the GPT-5-nano / AG News cell. To change a row, edit the code it points
to and rerun with a new `--tag`. A changed prompt or system line changes the cache key, so the next
run pays for new requests.

## Prompt and model call

| Detail | Paper | Here | Where |
|---|---|---|---|
| Prompt template | "Prompts include k demonstrations followed by a free-form test query." Nothing else. | `Article: <text>\nTopic: <label>` per demonstration, blank line between, query ends with `Topic:` | `iclorder/prompts.py` |
| Label words | not stated. AG News's class names are World, Sports, Business, Sci/Tech | World, Sports, Business, Technology (one word each) | `iclorder/dataset.py`, `LABELS` |
| How the model is called | not stated | chat completion: one system line, "The user message is a sequence of labelled examples followed by one unlabelled item. Label the last item in the same format as the examples. Reply with exactly one of these labels and nothing else: World, Sports, Business, Technology."; the prompt is the user message | `OpenAIScorer.instruction` |
| Decoding | not stated | API default. GPT-5 models accept no temperature. `reasoning_effort=minimal` (0 reasoning tokens were used), 64 completion tokens, raised to 256 once if a request reports its budget exhausted | `scripts/run_sensitivity.py` flags, `OpenAIScorer.complete` |
| Label reading | "accuracy"; extraction method not stated | the first allowed label word mentioned in the reply, whole word, case and punctuation ignored. "Science", "Sci Tech", "Tech" are accepted for Technology. A reply with no label counts as wrong and is tallied (`unmatched`) | `match_label` |
| Model id | "GPT-5-nano" | `gpt-5-nano` | |

## Sampling

| Detail | Paper | Here | Where |
|---|---|---|---|
| Demonstration pool | not stated | the AG News train split, 120,000 articles | `dataset.build_splits` |
| Demonstration sets (M = 10) | "M distinct example sets", k = 2\|C\| = 8 | 2 examples per label without replacement, drawn independently per set; an identical set is redrawn | `dataset.sample_demo_sets` |
| Dev / test | 1,000 dev, 500 test, "balanced across classes by oversampling underrepresented labels" | test: 125 per class from the HF test split. Dev: 1,000 from what is left, kept so the data recipe matches the paper's sizes; this experiment does not use it. The three files are disjoint | `dataset.balanced_sample`, `dataset.build_splits` |
| Default ordering | "group by label, labels alphabetical, examples alphabetical within label" | sort by (label word, article text) | `prompts.default_order` |
| Random orderings (P = 10) | "generated uniformly at random" | uniform permutations of the 8 positions, duplicates rejected, the identity allowed | `prompts.random_permutations` |
| Same orderings across sets | the notation a_{i,j} suggests pi_j is shared | the same 10 permutations are applied to every set's default ordering | `scripts/run_sensitivity.py` |
| Seeds | not stated | `--seed 0`: demonstration sets use the seed, orderings use seed + 1; the data files use their own seed 0 | scripts |

## Statistics

| Detail | Paper | Here | Where |
|---|---|---|---|
| Standard deviation | population or sample not stated | population (ddof = 0). `--ddof 1` scales both sensitivities by sqrt(10/9) = 1.054 and leaves r unchanged | `metrics.sensitivities` |
| Accuracy | per (set, ordering) on 500 test examples | the same; unmatched replies count as wrong | `scripts/run_sensitivity.py` |

## Caching and cost

| Detail | Here | Where |
|---|---|---|
| Cache | every reply stored in `results/openai_cache/<model>.sqlite` under sha256(model, reasoning effort, system line, prompt, completion budget); a rerun or a resumed run reads from it | `openai_scorer._Cache` |
| Cost | prompt, cached-prompt, completion and reasoning tokens summed from the API's usage field; cost at list prices in `summary.json` | `OpenAIScorer.usage_summary` |
| Rate limits | 24 parallel requests; retries with exponential backoff on 429 and 5xx; none were needed | `OpenAIScorer.complete` |

## Inconsistencies in the paper that touch this cell

1. The abstract says GPT-5; the body and the limitations section say GPT-5-nano.
2. Section 3.3 calls GPT-5-nano "the most sensitive to ordering", but Table 2 shows three models with
   larger ordering values (DeepSeek 1.5B 0.0362, DeepSeek 7B 0.0287, Llama 3 8B 0.0277). What is lowest
   for GPT-5-nano is r (0.85): it is the model where ordering matters *most relative to* selection.
3. The paper does not say how it accessed GPT-5-nano, how it decoded, or how it read the label off.

## What would move the numbers most

The system line that restricts the reply to the four label words (a free-form reply parsed afterwards
would likely be noisier), the label word for Sci/Tech, and the ddof choice (5% on both sensitivities,
nothing on r). The default decoding of the API adds run-to-run noise that we did not measure.

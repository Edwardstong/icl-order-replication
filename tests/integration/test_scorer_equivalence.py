"""Cached-prefix scoring must match plain full-prompt scoring.

Downloads Qwen2.5-0.5B (~1 GB) on first run, so it is skipped unless
ICL_RUN_MODEL_TESTS=1. Runs on CPU in float32 so the comparison is tight.
"""
import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("ICL_RUN_MODEL_TESTS") != "1",
                                reason="set ICL_RUN_MODEL_TESTS=1 to run model-loading tests")

MODEL = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def scorer():
    from iclorder.scorer import CachedScorer
    return CachedScorer(MODEL, device="cpu", dtype="float32", batch_size=4)


def _full_prompt_logprobs(scorer, prefix, query, label_ids):
    ids = scorer.tok(prefix + query, return_tensors="pt", add_special_tokens=True).input_ids
    with torch.inference_mode():
        logits = scorer.model(input_ids=ids).logits[0, -1].float()
    return torch.log_softmax(logits, dim=-1)[label_ids].numpy()


def test_cached_scoring_matches_full_prompt(scorer):
    from iclorder.prompts import build_prefix, label_continuations, render_query
    from iclorder.specs import SPECS
    spec = SPECS["ag_news"]
    demos = [
        {"id": "a", "input": "Article: The central bank raised rates again on Tuesday.", "target": 2},
        {"id": "b", "input": "Article: The striker scored twice in the final.", "target": 1},
        {"id": "c", "input": "Article: Talks resumed between the two governments.", "target": 0},
        {"id": "d", "input": "Article: The chipmaker unveiled a faster processor.", "target": 3},
    ]
    queries = [
        {"id": "q1", "input": "Article: Shares fell after the earnings report.", "target": 2},
        {"id": "q2", "input": "Article: The team won the championship after a dramatic overtime period that lasted far longer than anyone expected.", "target": 1},
        {"id": "q3", "input": "Article: A new phone.", "target": 3},
    ]
    prefix = build_prefix(spec, demos)
    label_ids = scorer.label_token_ids(label_continuations(spec))
    state = scorer.encode_prefix(prefix)
    cached = scorer.score_labels(state, [render_query(spec, q) for q in queries], label_ids)
    full = np.stack([_full_prompt_logprobs(scorer, prefix, render_query(spec, q), label_ids) for q in queries])
    assert cached.shape == full.shape == (3, 4)
    assert np.abs(cached - full).max() < 1e-3, f"max diff {np.abs(cached - full).max()}"
    assert (cached.argmax(1) == full.argmax(1)).all()


def test_label_tokens_distinct_for_all_runnable_specs(scorer):
    from iclorder.prompts import label_continuations
    from iclorder.specs import SPECS
    for name, spec in SPECS.items():
        if spec.runnable and spec.kind == "classification":
            ids = scorer.label_token_ids(label_continuations(spec))
            assert len(set(ids)) == spec.n_labels, name

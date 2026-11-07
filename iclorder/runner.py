"""Evaluate an M x P grid of (demonstration set, permutation) on a list of examples.

Returns per-example correctness, shape (M, P, N), as uint8. Keeping the
per-example array (not just accuracy) lets Section 4's dev-size sweep be
computed afterwards without rerunning the model.
"""
from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm

from .parse import cut_at_stop, extract_number, gsm8k_gold, numeric_match
from .prompts import apply_perm, build_prefix, default_order, label_continuations, render_query
from .scorer import CachedScorer
from .specs import DatasetSpec


def evaluate_grid(scorer: CachedScorer, spec: DatasetSpec, demo_sets: list[list[dict]],
                  perms: list[tuple[int, ...]], examples: list[dict], desc: str = "",
                  max_new_tokens: int = 256) -> tuple[np.ndarray, dict]:
    M, P, N = len(demo_sets), len(perms), len(examples)
    correct = np.zeros((M, P, N), dtype=np.uint8)
    queries = [render_query(spec, r) for r in examples]
    if spec.kind == "classification":
        label_ids = scorer.label_token_ids(label_continuations(spec))
        golds = np.array([int(r["target"]) for r in examples])
    else:
        golds_str = [gsm8k_gold(str(r["target"])) for r in examples]
    prefix_tokens = []
    t0 = time.time()
    with tqdm(total=M * P, desc=desc or f"{spec.name}", unit="prefix") as bar:
        for i, demos in enumerate(demo_sets):
            ordered = default_order(spec, demos)
            for j, perm in enumerate(perms):
                prefix_text = build_prefix(spec, apply_perm(ordered, perm))
                if spec.kind == "classification":
                    state = scorer.encode_prefix(prefix_text)
                    prefix_tokens.append(state.length)
                    preds = scorer.predict(state, queries, label_ids)
                    correct[i, j] = (preds == golds).astype(np.uint8)
                else:
                    outs = scorer.generate(prefix_text, queries, max_new_tokens=max_new_tokens)
                    correct[i, j] = np.array(
                        [numeric_match(extract_number(cut_at_stop(o)), g) for o, g in zip(outs, golds_str)], dtype=np.uint8)
                bar.update(1)
    info = {
        "seconds": time.time() - t0,
        "mean_prefix_tokens": float(np.mean(prefix_tokens)) if prefix_tokens else None,
        "multi_token_labels": getattr(scorer, "multi_token_labels", None),
        "evaluations": int(M * P * N),
    }
    return correct, info

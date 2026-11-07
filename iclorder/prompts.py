"""Prompt construction: default ordering, permutations, prefix and query text.

Prompt layout (raw completion mode, no chat template):

    <demo 1 input>
    <AnswerKey>: <demo 1 answer>

    <demo 2 input>
    <AnswerKey>: <demo 2 answer>

    ...

    <query input>
    <AnswerKey>:

The model is scored on the token that follows the final "<AnswerKey>:". For
classification that is the first token of " <verbalizer>"; for generation the
model continues freely and the answer is parsed from the continuation.
"""
from __future__ import annotations

import math

import numpy as np

from .specs import DatasetSpec


def target_text(spec: DatasetSpec, rec: dict) -> str:
    if spec.kind == "classification":
        return spec.verbalizers[int(rec["target"])]
    return str(rec["target"])


def render_demo(spec: DatasetSpec, rec: dict) -> str:
    return f"{rec['input']}\n{spec.answer_key}: {target_text(spec, rec)}"


def render_query(spec: DatasetSpec, rec: dict) -> str:
    return f"{rec['input']}\n{spec.answer_key}:"


def label_continuations(spec: DatasetSpec) -> list[str]:
    """The strings whose first token is scored, one per label, with the leading space."""
    return [f" {v}" for v in spec.verbalizers]


def default_order(spec: DatasetSpec, demos: list[dict]) -> list[dict]:
    """Paper Sec. 3.2. Classification: group by label, labels alphabetical, examples
    alphabetical within a label. Generation: all examples alphabetical."""
    if spec.kind == "classification":
        return sorted(demos, key=lambda r: (spec.verbalizers[int(r["target"])], r["input"]))
    return sorted(demos, key=lambda r: r["input"])


def random_permutations(k: int, p: int, seed: int) -> list[tuple[int, ...]]:
    """P distinct permutations of range(k), drawn uniformly. The paper says only
    "uniformly at random"; we reject duplicates so P permutations are P orderings."""
    if p > math.factorial(k):
        raise ValueError(f"cannot draw {p} distinct permutations of {k} items")
    rng = np.random.default_rng(seed)
    out: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(out) < p:
        perm = tuple(int(x) for x in rng.permutation(k))
        if perm in seen:
            continue
        seen.add(perm)
        out.append(perm)
    return out


def apply_perm(items: list, perm: tuple[int, ...]) -> list:
    if sorted(perm) != list(range(len(items))):
        raise ValueError("permutation does not match the item count")
    return [items[i] for i in perm]


def build_prefix(spec: DatasetSpec, demos_ordered: list[dict]) -> str:
    return "\n\n".join(render_demo(spec, r) for r in demos_ordered) + "\n\n"


def build_full_prompt(spec: DatasetSpec, demos_ordered: list[dict], query: dict) -> str:
    return build_prefix(spec, demos_ordered) + render_query(spec, query)

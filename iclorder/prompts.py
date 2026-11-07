"""Prompt construction: default ordering, random permutations, prompt text.

Prompt layout, the same for every demonstration set and ordering:

    Article: <demo 1 text>
    Topic: <demo 1 label>

    Article: <demo 2 text>
    Topic: <demo 2 label>

    ... (8 demonstrations in all)

    Article: <test text>
    Topic:

`build_prefix` renders the demonstrations, `render_query` the last block. The model
is sent prefix + query and asked to reply with one label word.
"""
from __future__ import annotations

import math

import numpy as np

from .dataset import ANSWER_KEY, LABELS


def render_demo(rec: dict) -> str:
    return f"{rec['input']}\n{ANSWER_KEY}: {LABELS[int(rec['target'])]}"


def render_query(rec: dict) -> str:
    return f"{rec['input']}\n{ANSWER_KEY}:"


def default_order(demos: list[dict]) -> list[dict]:
    """Paper Sec. 3.2: group by label, labels alphabetical, examples alphabetical within a label."""
    return sorted(demos, key=lambda r: (LABELS[int(r["target"])], r["input"]))


def random_permutations(k: int, p: int, seed: int) -> list[tuple[int, ...]]:
    """P distinct permutations of range(k), uniform. The paper says only "uniformly at
    random"; duplicates are rejected so P permutations are P different orderings."""
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


def build_prefix(demos_ordered: list[dict]) -> str:
    return "\n\n".join(render_demo(r) for r in demos_ordered) + "\n\n"

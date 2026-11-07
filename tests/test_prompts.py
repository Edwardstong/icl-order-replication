import math

import pytest

from iclorder.dataset import K, LABELS
from iclorder.prompts import apply_perm, build_prefix, default_order, random_permutations, render_demo, render_query


def _rec(i, text, label):
    return {"id": str(i), "input": text, "target": label}


def test_default_order_groups_by_label_alphabetically():
    demos = [_rec(0, "zebra story", 0), _rec(1, "apple story", 3), _rec(2, "match report", 1), _rec(3, "bank news", 2),
             _rec(4, "alpha world", 0)]
    ordered = default_order(demos)
    assert [LABELS[r["target"]] for r in ordered] == ["Business", "Sports", "Technology", "World", "World"]
    assert [r["input"] for r in ordered[-2:]] == ["alpha world", "zebra story"]  # alphabetical within a label


def test_random_permutations_are_distinct_valid_and_seeded():
    perms = random_permutations(K, 50, seed=3)
    assert len(perms) == 50 and len(set(perms)) == 50
    assert all(sorted(p) == list(range(K)) for p in perms)
    assert perms == random_permutations(K, 50, seed=3)
    assert perms != random_permutations(K, 50, seed=4)


def test_random_permutations_refuses_impossible_count():
    with pytest.raises(ValueError):
        random_permutations(3, math.factorial(3) + 1, seed=0)


def test_apply_perm():
    assert apply_perm(["a", "b", "c"], (2, 0, 1)) == ["c", "a", "b"]
    with pytest.raises(ValueError):
        apply_perm(["a", "b"], (0, 0))


def test_prompt_text_layout():
    demos = [_rec(0, "Article: one", 0), _rec(1, "Article: two", 1)]
    assert render_demo(demos[0]) == "Article: one\nTopic: World"
    assert render_query(_rec(9, "Article: q", 2)) == "Article: q\nTopic:"
    assert build_prefix(demos) == "Article: one\nTopic: World\n\nArticle: two\nTopic: Sports\n\n"


def test_k_is_two_per_label():
    assert K == 8 == 2 * len(LABELS)

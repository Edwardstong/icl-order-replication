import math

import pytest

from iclorder.prompts import (apply_perm, build_full_prompt, build_prefix, default_order,
                              label_continuations, random_permutations, render_demo, render_query)
from iclorder.specs import SPECS

AG = SPECS["ag_news"]


def _rec(i, text, label):
    return {"id": str(i), "input": text, "target": label}


def test_default_order_groups_by_label_alphabetically():
    demos = [_rec(0, "zebra story", 0), _rec(1, "apple story", 3), _rec(2, "match report", 1), _rec(3, "bank news", 2),
             _rec(4, "alpha world", 0)]
    ordered = default_order(AG, demos)
    labels = [AG.verbalizers[r["target"]] for r in ordered]
    assert labels == ["Business", "Sports", "Technology", "World", "World"]
    assert [r["input"] for r in ordered[-2:]] == ["alpha world", "zebra story"]  # alphabetical within label


def test_generation_default_order_is_alphabetical():
    g = SPECS["gsm8k"]
    demos = [{"id": "a", "input": "Question: zzz", "target": "#### 1"}, {"id": "b", "input": "Question: aaa", "target": "#### 2"}]
    assert [r["id"] for r in default_order(g, demos)] == ["b", "a"]


def test_random_permutations_are_distinct_valid_and_seeded():
    perms = random_permutations(8, 50, seed=3)
    assert len(perms) == 50 and len(set(perms)) == 50
    assert all(sorted(p) == list(range(8)) for p in perms)
    assert perms == random_permutations(8, 50, seed=3)
    assert perms != random_permutations(8, 50, seed=4)


def test_random_permutations_refuses_impossible_count():
    with pytest.raises(ValueError):
        random_permutations(3, math.factorial(3) + 1, seed=0)


def test_apply_perm():
    assert apply_perm(["a", "b", "c"], (2, 0, 1)) == ["c", "a", "b"]
    with pytest.raises(ValueError):
        apply_perm(["a", "b"], (0, 0))


def test_prompt_text_layout():
    demos = [_rec(0, "Article: one", 0), _rec(1, "Article: two", 1)]
    query = _rec(9, "Article: q", 2)
    assert render_demo(AG, demos[0]) == "Article: one\nTopic: World"
    assert render_query(AG, query) == "Article: q\nTopic:"
    prefix = build_prefix(AG, demos)
    assert prefix == "Article: one\nTopic: World\n\nArticle: two\nTopic: Sports\n\n"
    assert build_full_prompt(AG, demos, query) == prefix + "Article: q\nTopic:"
    assert label_continuations(AG) == [" World", " Sports", " Business", " Technology"]


def test_k_follows_paper_rule():
    assert SPECS["ag_news"].k == 8
    assert SPECS["dbpedia"].k == 28
    assert SPECS["mmlu"].k == 8
    assert SPECS["gsm8k"].k == 8

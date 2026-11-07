import numpy as np
import pytest

from iclorder.data import balanced_sample, random_sample, sample_demo_sets
from iclorder.specs import SPECS


def _pool(counts):
    rows, n = [], 0
    for label, c in enumerate(counts):
        for _ in range(c):
            rows.append({"id": f"r{n}", "input": f"text {n}", "target": label})
            n += 1
    return rows


def test_balanced_sample_quotas_and_remainder():
    rows = _pool([50, 50, 50, 50])
    out = balanced_sample(rows, 10, 4, np.random.default_rng(0))
    counts = np.bincount([r["target"] for r in out], minlength=4)
    assert counts.tolist() == [3, 3, 2, 2]           # 10 // 4 = 2, remainder to the first two labels
    assert len({r["id"] for r in out}) == 10           # no duplicates when pools are big enough


def test_balanced_sample_oversamples_short_label():
    rows = _pool([50, 2, 50])
    out = balanced_sample(rows, 9, 3, np.random.default_rng(0))
    ones = [r["id"] for r in out if r["target"] == 1]
    assert len(ones) == 3 and len(set(ones)) == 2     # label 1 has 2 rows, quota 3 -> one repeat


def test_balanced_sample_missing_label_raises():
    with pytest.raises(ValueError):
        balanced_sample(_pool([5, 0, 5]), 6, 3, np.random.default_rng(0))


def test_random_sample_bounds():
    rows = _pool([4])
    assert len(random_sample(rows, 4, np.random.default_rng(0))) == 4
    with pytest.raises(ValueError):
        random_sample(rows, 5, np.random.default_rng(0))


def test_demo_sets_are_balanced_distinct_and_seeded():
    spec = SPECS["ag_news"]
    pool = _pool([30, 30, 30, 30])
    sets = sample_demo_sets(pool, spec, m=5, seed=1)
    assert len(sets) == 5
    for s in sets:
        assert len(s) == spec.k == 8
        assert np.bincount([r["target"] for r in s], minlength=4).tolist() == [2, 2, 2, 2]
    keys = {tuple(sorted(r["id"] for r in s)) for s in sets}
    assert len(keys) == 5
    again = sample_demo_sets(pool, spec, m=5, seed=1)
    assert [[r["id"] for r in s] for s in again] == [[r["id"] for r in s] for s in sets]


def test_generation_demo_sets_have_k_items():
    spec = SPECS["gsm8k"]
    pool = [{"id": f"g{i}", "input": f"Question: {i}", "target": f"#### {i}"} for i in range(40)]
    sets = sample_demo_sets(pool, spec, m=3, seed=0)
    assert all(len(s) == 8 for s in sets)

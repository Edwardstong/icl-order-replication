import numpy as np

from iclorder.search import aggregates, curve_over_dev_size, curve_over_p


def _grid(acc):
    """Turn an (M, P) accuracy matrix into (M, P, N=10) correctness with that exact mean."""
    acc = np.asarray(acc)
    M, P = acc.shape
    out = np.zeros((M, P, 10), dtype=np.uint8)
    for i in range(M):
        for j in range(P):
            out[i, j, : int(round(acc[i, j] * 10))] = 1
    return out


def test_aggregates_pooled_vs_per_set_differ_when_sets_disagree():
    dev = _grid([[0.9, 0.1, 0.5],
                 [0.1, 0.9, 0.5]])
    test = _grid([[0.8, 0.2, 0.6],
                  [0.2, 0.8, 0.6]])
    a = aggregates(dev, test)
    # pooled dev means: [0.5, 0.5, 0.5] -> argmax picks j=0 (first max) -> test mean of column 0 = 0.5
    assert a["best_perm_pooled"] == 0
    assert a["highest_dev_pooled"] == 0.5
    # per set: set 0 picks j=0 (0.8), set 1 picks j=1 (0.8) -> 0.8
    assert a["highest_dev_per_set"] == 0.8
    assert a["average"] == np.mean([0.8, 0.2, 0.6, 0.2, 0.8, 0.6])
    assert a["oracle"] == 0.8
    assert a["frac_of_oracle_per_set"] == 1.0


def test_curves_use_prefixes_of_permutations_and_dev_examples():
    dev = _grid([[0.2, 0.9], [0.3, 0.8]])
    test = _grid([[0.3, 0.7], [0.4, 0.6]])
    c = curve_over_p(dev, test, [1, 2, 4])
    assert [x["p"] for x in c] == [1, 2]            # p=4 > P is skipped
    assert c[0]["oracle"] == np.mean([0.3, 0.4])    # only permutation 0 available
    assert c[1]["oracle"] == np.mean([0.7, 0.6])
    d = curve_over_dev_size(dev, test, [5, 10, 50])
    assert [x["n_dev"] for x in d] == [5, 10]

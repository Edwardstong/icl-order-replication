import numpy as np
import pytest

from iclorder.metrics import sensitivities


def test_formulas_on_small_matrix():
    acc = np.array([[0.5, 0.7, 0.9],
                    [0.6, 0.6, 0.6]])
    s = sensitivities(acc, ddof=0)
    # row stds: std([.5,.7,.9]) = 0.1633, std([.6,.6,.6]) = 0 -> mean 0.08165
    assert s["ordering"] == pytest.approx(np.mean([np.std([0.5, 0.7, 0.9]), 0.0]))
    # column stds: std([.5,.6]) = .05, std([.7,.6]) = .05, std([.9,.6]) = .15 -> mean 0.08333
    assert s["selection"] == pytest.approx(np.mean([0.05, 0.05, 0.15]))
    assert s["r"] == pytest.approx(s["selection"] / s["ordering"])
    assert s["M"] == 2 and s["P"] == 3 and s["mean_acc"] == pytest.approx(0.65)


def test_order_invariant_sets_give_zero_ordering_sensitivity():
    acc = np.tile(np.array([[0.3], [0.8], [0.5]]), (1, 4))  # every ordering identical per set
    s = sensitivities(acc)
    assert s["ordering"] == 0.0
    assert np.isnan(s["r"])
    assert s["selection"] > 0


def test_ddof_changes_result():
    acc = np.array([[0.5, 0.7], [0.6, 0.9]])
    assert sensitivities(acc, ddof=1)["ordering"] > sensitivities(acc, ddof=0)["ordering"]


def test_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        sensitivities(np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        sensitivities(np.array([[0.1, 0.2]]))  # M must be >= 2

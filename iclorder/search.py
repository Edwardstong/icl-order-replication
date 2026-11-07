"""Section 4 aggregates.

Inputs are per-example correctness arrays of shape (M, P, N): dev and test.
The paper describes the "highest-dev" rule two ways, so both are computed:

    pooled   (Sec. 4.1 text)  j* = argmax_j mean_i dev[i, j];  report mean_i test[i, j*]
    per_set  (Algorithm 1)    j*_i = argmax_j dev[i, j];       report mean_i test[i, j*_i]
    average                   mean over all (i, j) of test[i, j]
    oracle   ("max")          mean_i max_j test[i, j]

`curve_over_p` recomputes these using only the first p permutations, and
`curve_over_dev_size` using only the first n dev examples, which gives the
paper's Figure 3 style curves without rerunning anything.
"""
from __future__ import annotations

import numpy as np


def aggregates(dev_correct: np.ndarray, test_correct: np.ndarray) -> dict[str, float]:
    dev = np.asarray(dev_correct).mean(axis=2)   # (M, P)
    test = np.asarray(test_correct).mean(axis=2)  # (M, P)
    j_pooled = int(dev.mean(axis=0).argmax())
    j_per_set = dev.argmax(axis=1)
    pooled = float(test[:, j_pooled].mean())
    per_set = float(test[np.arange(test.shape[0]), j_per_set].mean())
    average = float(test.mean())
    oracle = float(test.max(axis=1).mean())
    return {
        "highest_dev_pooled": pooled,
        "highest_dev_per_set": per_set,
        "average": average,
        "oracle": oracle,
        "frac_of_oracle_pooled": pooled / oracle if oracle else float("nan"),
        "frac_of_oracle_per_set": per_set / oracle if oracle else float("nan"),
        "best_perm_pooled": j_pooled,
        "M": int(test.shape[0]),
        "P": int(test.shape[1]),
    }


def curve_over_p(dev_correct: np.ndarray, test_correct: np.ndarray, p_values: list[int]) -> list[dict]:
    P = dev_correct.shape[1]
    out = []
    for p in p_values:
        if p < 1 or p > P:
            continue
        agg = aggregates(dev_correct[:, :p], test_correct[:, :p])
        out.append({"p": p, **agg})
    return out


def curve_over_dev_size(dev_correct: np.ndarray, test_correct: np.ndarray, sizes: list[int]) -> list[dict]:
    N = dev_correct.shape[2]
    out = []
    for n in sizes:
        if n < 1 or n > N:
            continue
        agg = aggregates(dev_correct[:, :, :n], test_correct)
        out.append({"n_dev": n, **agg})
    return out

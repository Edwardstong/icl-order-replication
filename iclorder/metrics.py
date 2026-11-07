"""Ordering sensitivity, selection sensitivity and r (paper Sec. 3.2, Eqs. 1-3).

acc[i, j] = accuracy of demonstration set i under permutation j, shape (M, P).

    ordering  (sigma_bar_M) = mean over sets i of std_j acc[i, :]
    selection (sigma_bar_P) = mean over permutations j of std_i acc[:, j]
    r = selection / ordering

The paper does not say whether std is population or sample. Default ddof=0
(population). Change in one place and record it in METHODS.md.
"""
from __future__ import annotations

import numpy as np


def sensitivities(acc: np.ndarray, ddof: int = 0) -> dict[str, float]:
    a = np.asarray(acc, dtype=float)
    if a.ndim != 2 or min(a.shape) < 2:
        raise ValueError("acc must be an (M, P) matrix with M, P >= 2")
    ordering = float(a.std(axis=1, ddof=ddof).mean())
    selection = float(a.std(axis=0, ddof=ddof).mean())
    r = float("nan") if ordering == 0 else selection / ordering
    return {"ordering": ordering, "selection": selection, "r": r, "mean_acc": float(a.mean()), "M": a.shape[0], "P": a.shape[1]}


def accuracy_matrix(correct: np.ndarray) -> np.ndarray:
    """(M, P, N) per-example correctness -> (M, P) accuracy."""
    c = np.asarray(correct)
    if c.ndim != 3:
        raise ValueError("correct must have shape (M, P, N)")
    return c.mean(axis=2)

"""Ordering sensitivity, selection sensitivity and r (paper Sec. 3.2, Eqs. 1-3).

acc[i, j] = accuracy of demonstration set i under ordering j, shape (M, P).

    ordering  = mean over sets i      of std_j acc[i, :]   (how much the order moves accuracy)
    selection = mean over orderings j of std_i acc[:, j]   (how much the choice of examples moves it)
    r = selection / ordering                               (r < 1: ordering matters more)

The paper does not say whether std is population or sample; default ddof=0 (population).
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
    return {"ordering": ordering, "selection": selection, "r": r, "mean_acc": float(a.mean()),
            "M": int(a.shape[0]), "P": int(a.shape[1])}

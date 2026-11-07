"""Answer extraction for generation tasks and the numeric-tolerance match.

The paper says only "exact match or numeric tolerance". Our rule, frozen in
METHODS.md: parse the number after '####' if present, else the last number in
the continuation; match when |pred - gold| <= 1e-6 + 1e-4 * |gold|.
"""
from __future__ import annotations

import re

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_HASH = re.compile(r"####\s*(-?[\d,]*\.?\d+)")


def gsm8k_gold(target: str) -> str:
    m = _HASH.search(target)
    if not m:
        raise ValueError("GSM8K reference answer has no '####' line")
    return m.group(1).replace(",", "")


def cut_at_stop(text: str, stops: tuple[str, ...] = ("\n\nQuestion:", "\n\n\n")) -> str:
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def extract_number(text: str) -> float | None:
    m = _HASH.search(text)
    if m:
        s = m.group(1)
    else:
        nums = _NUM.findall(text)
        if not nums:
            return None
        s = nums[-1]
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def numeric_match(pred: float | None, gold: str | float, rel_tol: float = 1e-4, abs_tol: float = 1e-6) -> bool:
    if pred is None:
        return False
    g = float(str(gold).replace(",", ""))
    return abs(pred - g) <= abs_tol + rel_tol * abs(g)

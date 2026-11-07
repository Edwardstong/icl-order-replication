"""Call the OpenAI API for one prompt at a time, read the label from the reply, cache everything.

How a prompt is sent: a short system line names the allowed labels and asks for one of them;
the demonstrations-plus-query text is the user message. The label is the first allowed label
word mentioned in the reply ("Science" is accepted for "Technology", AG News's Sci/Tech class).
A reply naming no label counts as wrong and is tallied as `unmatched`.

Every completion is stored in results/openai_cache/<model>.sqlite under a hash of the request.
A rerun of the same grid is served from the cache: no requests, no cost, the same numbers.

The key is read from OPENAI_API_KEY, or from the macOS Keychain item named below. It is never
written anywhere by this code.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "results" / "openai_cache"
KEYCHAIN_SERVICE = "icl-order-replication-openai"

# USD per 1M tokens: (input, cached input, output). List prices checked 2026-11-06.
PRICES_PER_M: dict[str, tuple[float, float, float]] = {
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5": (1.25, 0.125, 10.00),
}

# Synonyms a chat model uses for a label word. Applied only when the exact label is absent;
# a match through an alias is tallied as `alias_matched`.
LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "technology": ("science", "sci tech", "science technology", "tech", "science and technology"),
}

_WORD = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def match_label(text: str, labels: list[str], aliases: dict[str, tuple[str, ...]] | None = LABEL_ALIASES) -> int:
    """Index of the label mentioned first in `text` as a whole word (case and punctuation
    ignored), or -1 if none is mentioned. Exact label words win over aliases."""
    hay = f" {normalize(text)} "
    best, best_pos = -1, None
    for i, lab in enumerate(labels):
        pos = hay.find(f" {normalize(lab)} ")
        if pos >= 0 and (best_pos is None or pos < best_pos):
            best, best_pos = i, pos
    if best >= 0 or not aliases:
        return best
    for i, lab in enumerate(labels):
        for alt in aliases.get(normalize(lab), ()):
            pos = hay.find(f" {normalize(alt)} ")
            if pos >= 0 and (best_pos is None or pos < best_pos):
                best, best_pos = i, pos
    return best


def price_for(model_id: str) -> tuple[float, float, float] | None:
    for key in sorted(PRICES_PER_M, key=len, reverse=True):  # longest prefix wins
        if model_id.startswith(key):
            return PRICES_PER_M[key]
    return None


def _retryable(exc: Exception) -> bool:
    try:
        import openai
    except ImportError:  # pragma: no cover
        return False
    for name in ("RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"):
        cls = getattr(openai, name, None)
        if cls is not None and isinstance(exc, cls):
            return True
    status = getattr(exc, "status_code", 0) or 0
    return status in (408, 409, 429) or status >= 500


def _budget_exhausted(exc: Exception) -> bool:
    """400 'Could not finish the message because max_tokens ... was reached': hidden
    reasoning used the whole completion budget."""
    return getattr(exc, "status_code", 0) == 400 and "max_tokens" in str(exc)


def _usage_dict(resp) -> dict[str, int]:
    u = getattr(resp, "usage", None)
    ptd = getattr(u, "prompt_tokens_details", None)
    ctd = getattr(u, "completion_tokens_details", None)
    return {
        "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
        "cached_prompt_tokens": int(getattr(ptd, "cached_tokens", 0) or 0),
        "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(ctd, "reasoning_tokens", 0) or 0),
    }


def _api_key_from_keychain(service: str = KEYCHAIN_SERVICE) -> str | None:
    try:
        out = subprocess.run(["security", "find-generic-password", "-s", service, "-w"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def _make_client():
    key = os.environ.get("OPENAI_API_KEY") or _api_key_from_keychain()
    if not key:
        raise SystemExit("No OpenAI key. Either export OPENAI_API_KEY, or store it once in the macOS Keychain:\n"
                         f'  security add-generic-password -a "$USER" -s {KEYCHAIN_SERVICE} -w')
    import openai
    return openai.OpenAI(api_key=key)


class _Cache:
    """sqlite map from request hash to (completion text, usage). Safe across threads."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS completions "
                          "(key TEXT PRIMARY KEY, text TEXT NOT NULL, usage TEXT NOT NULL, created REAL NOT NULL)")
        self.conn.commit()
        self.lock = threading.Lock()

    def get(self, key: str):
        with self.lock:
            row = self.conn.execute("SELECT text, usage FROM completions WHERE key = ?", (key,)).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def put(self, key: str, text: str, usage: dict) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO completions VALUES (?, ?, ?, ?)",
                              (key, text, json.dumps(usage), time.time()))
            self.conn.commit()


class OpenAIScorer:
    def __init__(self, model_id: str = "gpt-5-nano", labels=("World", "Sports", "Business", "Technology"),
                 workers: int = 8, reasoning_effort: str | None = "minimal", max_completion_tokens: int = 64,
                 cache_dir: str | Path = CACHE_DIR, client=None, retries: int = 6):
        self.model_id = model_id
        self.labels = list(labels)
        self.workers = max(1, workers)
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.retries = retries
        self.client = client if client is not None else _make_client()
        self.cache = _Cache(Path(cache_dir) / f"{model_id.replace('/', '__')}.sqlite")
        self.usage = {k: 0 for k in ("requests", "cache_hits", "retries", "prompt_tokens", "cached_prompt_tokens",
                                     "completion_tokens", "reasoning_tokens", "unmatched", "alias_matched", "budget_raised")}
        self._lock = threading.Lock()

    @property
    def instruction(self) -> str:
        """The system line. Part of the cache key, so changing it starts a new cache."""
        return ("The user message is a sequence of labelled examples followed by one unlabelled item. "
                "Label the last item in the same format as the examples. Reply with exactly one of these "
                f"labels and nothing else: {', '.join(self.labels)}.")

    def predict(self, prompts: list[str]) -> np.ndarray:
        """Label index per prompt (-1 when the reply names no label). Requests run in parallel."""
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            outs = list(ex.map(lambda p: self.complete(p, self.instruction, self.max_completion_tokens), prompts))
        preds = np.array([match_label(o, self.labels) for o in outs], dtype=np.int64)
        exact = np.array([match_label(o, self.labels, aliases=None) for o in outs], dtype=np.int64)
        with self._lock:
            self.usage["unmatched"] += int((preds < 0).sum())
            self.usage["alias_matched"] += int(((exact < 0) & (preds >= 0)).sum())
        return preds

    def complete(self, prompt: str, system: str, max_tokens: int) -> str:
        key = hashlib.sha256(json.dumps([self.model_id, self.reasoning_effort, system, prompt, max_tokens]).encode()).hexdigest()
        hit = self.cache.get(key)
        if hit is not None:
            with self._lock:
                self.usage["cache_hits"] += 1
            return hit[0]
        kwargs = dict(model=self.model_id,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                      max_completion_tokens=max_tokens)
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        delay, raised = 1.0, False
        for attempt in range(self.retries + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                if _budget_exhausted(exc) and not raised:
                    kwargs["max_completion_tokens"] = 4 * max_tokens  # retry once with a larger budget
                    raised = True
                    with self._lock:
                        self.usage["budget_raised"] += 1
                    continue
                if attempt >= self.retries or not _retryable(exc):
                    raise
                with self._lock:
                    self.usage["retries"] += 1
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        text = resp.choices[0].message.content or ""
        used = _usage_dict(resp)
        self.cache.put(key, text, used)
        with self._lock:
            self.usage["requests"] += 1
            for k, v in used.items():
                self.usage[k] += v
        return text

    def estimated_cost_usd(self) -> float | None:
        p = price_for(self.model_id)
        if p is None:
            return None
        u = self.usage
        uncached = u["prompt_tokens"] - u["cached_prompt_tokens"]
        return (uncached * p[0] + u["cached_prompt_tokens"] * p[1] + u["completion_tokens"] * p[2]) / 1e6

    def usage_summary(self) -> dict:
        cache = self.cache.path
        if cache.is_relative_to(ROOT):
            cache = cache.relative_to(ROOT)
        return {"model": self.model_id, "reasoning_effort": self.reasoning_effort,
                "max_completion_tokens": self.max_completion_tokens, "workers": self.workers, "cache": str(cache),
                **self.usage, "estimated_cost_usd": self.estimated_cost_usd()}

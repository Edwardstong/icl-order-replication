"""OpenAI API backend: the same prompts, grids and formulas, scored through chat completions.

Differences from the local `CachedScorer`, all recorded in METHODS.md:
- the prompt is sent as a chat: a short system instruction naming the allowed
  labels, then the demonstrations-plus-query text as the user message;
- the label is read from the generated text (first allowed label mentioned as
  a whole word), not from next-token log-probabilities;
- decoding is the API default (GPT-5 models accept no temperature), so a cell
  is not bit-reproducible; every completion is cached on disk so reruns and
  resumed runs cost nothing and reuse the same outputs.

The key is read from OPENAI_API_KEY by the SDK; it is never written anywhere.
"""
from __future__ import annotations

import dataclasses
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

# USD per 1M tokens: (input, cached input, output). List prices checked 2026-11-06.
PRICES_PER_M: dict[str, tuple[float, float, float]] = {
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5": (1.25, 0.125, 10.00),
}

_WORD = re.compile(r"[a-z0-9]+")

# Synonyms a chat model uses for a label word (AG News's real class name is "Sci/Tech";
# our label word is "Technology"). Keyed by normalized label word; applied only when
# that label is in the current label set. A match through an alias is tallied.
LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "technology": ("science", "sci tech", "science technology", "tech", "science and technology"),
}


def normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def match_label(text: str, labels: list[str], aliases: dict[str, tuple[str, ...]] | None = LABEL_ALIASES) -> int:
    """Index of the label mentioned first in `text` as a whole word (after
    lower-casing and dropping punctuation), or -1 if no label is mentioned.
    Exact label words win over aliases; among aliases the earliest mention wins."""
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


def estimate_tokens(text: str) -> int:
    """Rough token count (chars / 4) for the prefix-length statistic; the API's
    usage field holds the exact prompt tokens."""
    return max(1, round(len(text) / 4))


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
    """OpenAI returns 400 'Could not finish the message because max_tokens ... was reached'
    when hidden reasoning used the whole completion budget."""
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


KEYCHAIN_SERVICE = "icl-order-replication-openai"


def _api_key_from_keychain(service: str = KEYCHAIN_SERVICE) -> str | None:
    """macOS Keychain fallback. Store the key once, hidden at the prompt, with
        security add-generic-password -a "$USER" -s icl-order-replication-openai -w
    """
    try:
        out = subprocess.run(["security", "find-generic-password", "-s", service, "-w"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def _make_client():
    key = os.environ.get("OPENAI_API_KEY") or _api_key_from_keychain()
    if not key:
        raise SystemExit("No OpenAI key: export OPENAI_API_KEY, or store it in the macOS Keychain with\n"
                         '  security add-generic-password -a "$USER" -s icl-order-replication-openai -w\n'
                         "The key is read by the OpenAI SDK and never written by this code.")
    import openai
    return openai.OpenAI(api_key=key)


class _Cache:
    """sqlite map from request hash to completion text and usage. Safe across threads."""

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


@dataclasses.dataclass
class PrefixText:
    text: str
    length: int  # estimated tokens; see estimate_tokens


class OpenAIScorer:
    """Drop-in for CachedScorer in `runner.evaluate_grid`: label_token_ids,
    encode_prefix, predict, generate, plus usage accounting."""

    def __init__(self, model_id: str, workers: int = 8, reasoning_effort: str | None = "minimal",
                 max_completion_tokens: int = 64, cache_dir: str | Path | None = None, client=None,
                 retries: int = 6):
        self.model_id = model_id
        self.workers = max(1, workers)
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.retries = retries
        self.client = client if client is not None else _make_client()
        cache_dir = Path(cache_dir) if cache_dir is not None else Path("results") / "openai_cache"
        self.cache = _Cache(cache_dir / f"{model_id.replace('/', '__')}.sqlite")
        self.labels: list[str] = []
        self.multi_token_labels: list[str] = []
        # attributes the run scripts print for any backend
        self.device, self.dtype, self.load_seconds = "openai", "api", 0.0
        self.usage = {k: 0 for k in ("requests", "cache_hits", "retries", "prompt_tokens", "cached_prompt_tokens",
                                     "completion_tokens", "reasoning_tokens", "unmatched", "alias_matched", "budget_raised")}
        self._lock = threading.Lock()

    # -- interface used by runner.evaluate_grid -------------------------------------------------

    def label_token_ids(self, continuations: list[str]) -> list[int]:
        self.labels = [c.strip() for c in continuations]
        self.multi_token_labels = []
        return list(range(len(self.labels)))

    def encode_prefix(self, text: str) -> PrefixText:
        return PrefixText(text=text, length=estimate_tokens(text))

    def classification_instruction(self) -> str:
        return ("The user message is a sequence of labelled examples followed by one unlabelled item. "
                "Label the last item in the same format as the examples. Reply with exactly one of these "
                f"labels and nothing else: {', '.join(self.labels)}.")

    @staticmethod
    def generation_instruction() -> str:
        return ("The user message is a sequence of worked examples followed by one unanswered question. "
                "Answer the last question in the same style as the examples and finish with a line of the form "
                "'#### <final numeric answer>'.")

    def predict(self, prefix: PrefixText, queries: list[str], label_ids: list[int]) -> np.ndarray:
        outs = self.complete_many([prefix.text + q for q in queries], self.classification_instruction(),
                                  self.max_completion_tokens)
        preds = np.array([match_label(o, self.labels) for o in outs], dtype=np.int64)
        exact = np.array([match_label(o, self.labels, aliases=None) for o in outs], dtype=np.int64)
        with self._lock:
            self.usage["unmatched"] += int((preds < 0).sum())
            self.usage["alias_matched"] += int(((exact < 0) & (preds >= 0)).sum())
        return preds

    def generate(self, prefix_text: str, queries: list[str], max_new_tokens: int = 256) -> list[str]:
        return self.complete_many([prefix_text + q for q in queries], self.generation_instruction(), max_new_tokens)

    # -- requests ---------------------------------------------------------------------------------

    def complete_many(self, prompts: list[str], system: str, max_tokens: int) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(lambda p: self.complete(p, system, max_tokens), prompts))

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
        delay = 1.0
        raised = False
        for attempt in range(self.retries + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                if _budget_exhausted(exc) and not raised:
                    # reasoning tokens consumed the completion budget: retry once with 4x
                    kwargs["max_completion_tokens"] = 4 * max_tokens
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

    # -- reporting --------------------------------------------------------------------------------

    def estimated_cost_usd(self) -> float | None:
        p = price_for(self.model_id)
        if p is None:
            return None
        u = self.usage
        uncached = u["prompt_tokens"] - u["cached_prompt_tokens"]
        return (uncached * p[0] + u["cached_prompt_tokens"] * p[1] + u["completion_tokens"] * p[2]) / 1e6

    def usage_summary(self) -> dict:
        return {
            "model": self.model_id, "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": self.max_completion_tokens, "workers": self.workers,
            "cache": str(self.cache.path), **self.usage, "estimated_cost_usd": self.estimated_cost_usd(),
            "note": "label read from generated text (match_label); prefix lengths are chars/4 estimates",
        }

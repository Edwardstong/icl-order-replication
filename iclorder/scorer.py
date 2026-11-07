"""Model wrapper with shared-prefix caching.

For a fixed demonstration set and ordering, the demonstrations are the same for
every test query. `encode_prefix` runs them through the model once and keeps
the key/value cache. `score_labels` then runs only the short query tokens,
batched, against copies of that cache, and reads the next-token distribution
over the label words. This removes ~95% of the tokens from every evaluation,
which is what makes the paper's grid affordable without a GPU cluster.

Works on CUDA, Apple MPS and CPU. Tested against transformers 5.x; the cache
helpers fall back to older attribute names where they differ.
"""
from __future__ import annotations

import copy
import dataclasses
import inspect
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, name: str | None = None) -> torch.dtype:
    if name:
        return getattr(torch, name)
    return torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32


@dataclasses.dataclass
class PrefixState:
    text: str
    length: int
    cache: DynamicCache


class CachedScorer:
    def __init__(self, model_id: str, device: str | None = None, dtype: str | None = None,
                 batch_size: int = 16, trust_remote_code: bool = False):
        self.model_id = model_id
        self.device = pick_device(device)
        self.dtype = pick_dtype(self.device, dtype)
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        t0 = time.time()
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.dtype, trust_remote_code=trust_remote_code)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.dtype, trust_remote_code=trust_remote_code)
        if next(model.parameters()).dtype != self.dtype:
            model = model.to(self.dtype)
        self.model = model.to(self.device).eval()
        self.load_seconds = time.time() - t0
        self._supports_logits_to_keep = "logits_to_keep" in inspect.signature(self.model.forward).parameters

    # ----- tokens -----------------------------------------------------------
    def label_token_ids(self, continuations: list[str]) -> list[int]:
        """First token of each label continuation (e.g. " World"). Must be distinct."""
        ids, multi = [], []
        for c in continuations:
            toks = self.tok(c, add_special_tokens=False).input_ids
            ids.append(int(toks[0]))
            if len(toks) > 1:
                multi.append((c, len(toks)))
        if len(set(ids)) != len(ids):
            dup = [c for c, i in zip(continuations, ids) if ids.count(i) > 1]
            raise ValueError(f"label words share a first token under {self.model_id}: {dup}. Choose other verbalizers.")
        self.multi_token_labels = multi
        return ids

    # ----- prefix cache ------------------------------------------------------
    def _new_cache(self) -> DynamicCache:
        try:
            return DynamicCache(config=self.model.config)
        except TypeError:
            return DynamicCache()

    @torch.inference_mode()
    def encode_prefix(self, text: str) -> PrefixState:
        ids = self.tok(text, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)
        out = self.model(input_ids=ids, past_key_values=self._new_cache(), use_cache=True)
        return PrefixState(text=text, length=int(ids.shape[1]), cache=out.past_key_values)

    def _expand_cache(self, cache: DynamicCache, n: int) -> DynamicCache:
        c = copy.deepcopy(cache)
        if hasattr(c, "batch_repeat_interleave"):
            c.batch_repeat_interleave(n)
            return c
        # Older transformers: rebuild layer by layer.
        new = self._new_cache()
        layers = list(zip(c.key_cache, c.value_cache))
        for idx, (k, v) in enumerate(layers):
            new.update(k.expand(n, *k.shape[1:]).contiguous(), v.expand(n, *v.shape[1:]).contiguous(), idx)
        return new

    # ----- classification scoring ------------------------------------------
    @torch.inference_mode()
    def score_labels(self, prefix: PrefixState, queries: list[str], label_ids: list[int]) -> np.ndarray:
        """Log-probabilities over the label tokens for each query, shape (n, L).

        Queries are left-padded so that every row's last real token sits at the
        final position; the pads between prefix and query are masked out and
        given dummy positions. Only the final position's logits are computed.
        """
        out = np.empty((len(queries), len(label_ids)), dtype=np.float32)
        label_t = torch.tensor(label_ids, device=self.device)
        pad = self.tok.pad_token_id
        for s in range(0, len(queries), self.batch_size):
            chunk = queries[s:s + self.batch_size]
            enc = [self.tok(q, add_special_tokens=False).input_ids for q in chunk]
            B, Q = len(enc), max(len(e) for e in enc)
            ids = torch.full((B, Q), pad, dtype=torch.long)
            qmask = torch.zeros((B, Q), dtype=torch.long)
            pos = torch.zeros((B, Q), dtype=torch.long)
            for b, e in enumerate(enc):
                n = len(e)
                ids[b, Q - n:] = torch.tensor(e)
                qmask[b, Q - n:] = 1
                pos[b, Q - n:] = prefix.length + torch.arange(n)
            attn = torch.cat([torch.ones((B, prefix.length), dtype=torch.long), qmask], dim=1)
            kwargs = {}
            if self._supports_logits_to_keep:
                kwargs["logits_to_keep"] = 1
            logits = self.model(
                input_ids=ids.to(self.device), attention_mask=attn.to(self.device), position_ids=pos.to(self.device),
                past_key_values=self._expand_cache(prefix.cache, B), use_cache=True, **kwargs,
            ).logits
            last = logits[:, -1, :].float()
            out[s:s + B] = torch.log_softmax(last, dim=-1)[:, label_t].cpu().numpy()
        return out

    def predict(self, prefix: PrefixState, queries: list[str], label_ids: list[int]) -> np.ndarray:
        return self.score_labels(prefix, queries, label_ids).argmax(axis=1)

    # ----- generation (no prefix caching; use vLLM on a GPU for scale) ------
    @torch.inference_mode()
    def generate(self, prefix_text: str, queries: list[str], max_new_tokens: int = 256) -> list[str]:
        self.tok.padding_side = "left"
        outs: list[str] = []
        for s in range(0, len(queries), self.batch_size):
            prompts = [prefix_text + q for q in queries[s:s + self.batch_size]]
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc.input_ids.shape[1]:]
            outs.extend(self.tok.batch_decode(new, skip_special_tokens=True))
        return outs

"""AG News: constants, the three data files, and demonstration-set sampling.

Files written by `build_splits` into data/ag_news/:
    demo_pool.jsonl   the train split minus anything in dev/test; demonstrations come from here
    dev.jsonl         1,000 balanced examples (the paper's dev size; unused in Section 3)
    test.jsonl        500 balanced examples, the evaluation set
    meta.json         sizes, seed, label words

Records are {"id": str, "input": str, "target": int}. The three files are disjoint and
every row is shuffled, so the first n rows of a file are a random subset. The dev file is
kept so the recipe matches the paper's split sizes; the sensitivity run reads only the demo
pool and the test file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HF_ID = "fancyzhx/ag_news"
LABELS = ("World", "Sports", "Business", "Technology")  # AG News class names; "Sci/Tech" -> one word
ANSWER_KEY = "Topic"
K = 2 * len(LABELS)  # demonstrations per prompt: 2 per label, as in the paper (k = 2|C|)


def render(ex: dict) -> tuple[str, int]:
    """One Hugging Face row -> (input text, label index)."""
    return f"Article: {ex['text'].strip()}", int(ex["label"])


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(split: str) -> list[dict]:
    from datasets import load_dataset  # imported here so the tests need no network

    return [{"id": f"{split}-{i}", "input": render(ex)[0], "target": render(ex)[1]}
            for i, ex in enumerate(load_dataset(HF_ID, split=split))]


def balanced_sample(rows: list[dict], n: int, n_labels: int, rng: np.random.Generator) -> list[dict]:
    """n rows with per-label quotas n // L (+1 for the first n % L labels), shuffled.
    A label whose pool is smaller than its quota is oversampled with replacement
    (the paper's "oversampling underrepresented labels")."""
    base, rem = divmod(n, n_labels)
    out: list[dict] = []
    for label in range(n_labels):
        quota = base + (1 if label < rem else 0)
        pool = [r for r in rows if r["target"] == label]
        if not pool:
            raise ValueError(f"label {label} has no examples to sample from")
        if len(pool) >= quota:
            idx = rng.choice(len(pool), size=quota, replace=False)
        else:
            idx = np.concatenate([np.arange(len(pool)), rng.choice(len(pool), size=quota - len(pool), replace=True)])
        out.extend(pool[i] for i in idx)
    rng.shuffle(out)
    return out


def build_splits(n_dev: int, n_test: int, seed: int, out_dir: Path) -> dict:
    """Download AG News and write demo_pool / dev / test. Returns the meta dict."""
    rng = np.random.default_rng(seed)
    demo_rows = load_split("train")
    eval_rows = load_split("test")

    test = balanced_sample(eval_rows, n_test, len(LABELS), rng)
    used = {r["id"] for r in test}
    dev = balanced_sample([r for r in eval_rows if r["id"] not in used], n_dev, len(LABELS), rng)
    used |= {r["id"] for r in dev}
    demo_pool = [r for r in demo_rows if r["id"] not in used]
    rng.shuffle(demo_pool)

    write_jsonl(out_dir / "demo_pool.jsonl", demo_pool)
    write_jsonl(out_dir / "dev.jsonl", dev)
    write_jsonl(out_dir / "test.jsonl", test)
    meta = {"dataset": "ag_news", "hf_id": HF_ID, "k": K, "labels": list(LABELS), "answer_key": ANSWER_KEY,
            "n_dev": len(dev), "n_test": len(test), "n_demo_pool": len(demo_pool), "seed": seed,
            "demo_source_split": "train", "dev_and_test_source_split": "test"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_splits(data_dir: Path) -> tuple[list[dict], list[dict], dict]:
    """(demo_pool, test, meta) from data/ag_news/."""
    d = data_dir / "ag_news"
    if not (d / "meta.json").exists():
        raise SystemExit(f"No data at {d}. Run: python scripts/build_splits.py")
    return read_jsonl(d / "demo_pool.jsonl"), read_jsonl(d / "test.jsonl"), json.loads((d / "meta.json").read_text())


def sample_demo_sets(demo_pool: list[dict], m: int, seed: int) -> list[list[dict]]:
    """M demonstration sets of K examples: 2 per label, drawn without replacement.
    Sets are drawn independently; a duplicate set is rejected and redrawn."""
    rng = np.random.default_rng(seed)
    sets: list[list[dict]] = []
    seen: set[tuple[str, ...]] = set()
    guard = 0
    while len(sets) < m:
        guard += 1
        if guard > 100 * m:
            raise RuntimeError("could not draw distinct demonstration sets; pool too small?")
        chosen: list[dict] = []
        for label in range(len(LABELS)):
            pool = [r for r in demo_pool if r["target"] == label]
            if len(pool) < 2:
                raise ValueError(f"label {label} has fewer than 2 demonstration candidates")
            idx = rng.choice(len(pool), size=2, replace=False)
            chosen.extend(pool[i] for i in idx)
        key = tuple(sorted(r["id"] for r in chosen))
        if key in seen:
            continue
        seen.add(key)
        sets.append(chosen)
    return sets

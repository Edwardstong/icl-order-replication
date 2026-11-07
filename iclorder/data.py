"""Build dev / test splits and demonstration pools, and sample demonstration sets.

Materialized layout (data/<dataset>/):
    demo_pool.jsonl   examples demonstrations are drawn from
    dev.jsonl         N_dev examples (paper default 1000)
    test.jsonl        N_test examples (paper default 500)
    meta.json         sizes, seed, k, verbalizers, which HF splits were used

Records are {"id": str, "input": str, "target": int | str}. All three files are
disjoint. Classification dev/test are balanced across labels, oversampling a
label only when its pool is too small (paper, Sec. 3.1). Rows are shuffled, so
taking the first n of a file is a random subset.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .specs import DatasetSpec


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(spec: DatasetSpec, split: str) -> list[dict]:
    from datasets import load_dataset  # imported here so tests need no network

    ds = load_dataset(spec.hf_id, spec.hf_config, split=split)
    rows = []
    for i, ex in enumerate(ds):
        text, target = spec.render(ex)
        rows.append({"id": f"{split}-{i}", "input": text, "target": target})
    return rows


def balanced_sample(rows: list[dict], n: int, n_labels: int, rng: np.random.Generator) -> list[dict]:
    """n rows with per-label quotas n // L (+1 for the first n % L labels).

    A label whose pool is smaller than its quota is oversampled with
    replacement, which is the paper's "oversampling underrepresented labels".
    """
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


def random_sample(rows: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
    if n > len(rows):
        raise ValueError(f"asked for {n} rows but only {len(rows)} available")
    idx = rng.choice(len(rows), size=n, replace=False)
    return [rows[i] for i in idx]


def _take(rows: list[dict], n: int, spec: DatasetSpec, rng: np.random.Generator) -> list[dict]:
    if spec.kind == "classification":
        return balanced_sample(rows, n, spec.n_labels, rng)
    return random_sample(rows, n, rng)


def build_splits(spec: DatasetSpec, n_dev: int, n_test: int, seed: int, out_dir: Path) -> dict:
    """Materialize demo_pool / dev / test for one dataset. Returns the meta dict."""
    rng = np.random.default_rng(seed)
    demo_rows = load_split(spec, spec.demo_split)
    eval_rows = load_split(spec, spec.eval_split)

    test = _take(eval_rows, n_test, spec, rng)
    used = {r["id"] for r in test}
    eval_left = [r for r in eval_rows if r["id"] not in used]

    if len(eval_left) >= n_dev:
        dev = _take(eval_left, n_dev, spec, rng)
        dev_source = spec.eval_split
    else:
        # Eval split too small (e.g. GSM8K test = 1,319): dev comes from the demo split.
        dev = _take(demo_rows, n_dev, spec, rng)
        dev_source = spec.demo_split
    used |= {r["id"] for r in dev}
    demo_pool = [r for r in demo_rows if r["id"] not in used]
    rng.shuffle(demo_pool)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "demo_pool.jsonl", demo_pool)
    write_jsonl(out_dir / "dev.jsonl", dev)
    write_jsonl(out_dir / "test.jsonl", test)
    meta = {
        "dataset": spec.name, "kind": spec.kind, "hf_id": spec.hf_id, "hf_config": spec.hf_config,
        "k": spec.k, "verbalizers": list(spec.verbalizers), "answer_key": spec.answer_key,
        "n_dev": len(dev), "n_test": len(test), "n_demo_pool": len(demo_pool),
        "dev_source_split": dev_source, "test_source_split": spec.eval_split, "demo_source_split": spec.demo_split,
        "seed": seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_materialized(data_dir: Path, spec: DatasetSpec) -> tuple[list[dict], list[dict], list[dict], dict]:
    d = data_dir / spec.name
    if not (d / "meta.json").exists():
        raise SystemExit(f"No materialized data at {d}. Run: python scripts/build_splits.py --dataset {spec.name}")
    meta = json.loads((d / "meta.json").read_text())
    return read_jsonl(d / "demo_pool.jsonl"), read_jsonl(d / "dev.jsonl"), read_jsonl(d / "test.jsonl"), meta


def sample_demo_sets(demo_pool: list[dict], spec: DatasetSpec, m: int, seed: int) -> list[list[dict]]:
    """M demonstration sets of k examples each.

    Classification: 2 examples per label (k = 2|C| makes this the natural reading).
    Generation: k examples drawn uniformly. Sets are drawn independently; two
    identical sets are rejected and redrawn.
    """
    rng = np.random.default_rng(seed)
    sets: list[list[dict]] = []
    seen: set[tuple[str, ...]] = set()
    guard = 0
    while len(sets) < m:
        guard += 1
        if guard > 100 * m:
            raise RuntimeError("could not draw distinct demonstration sets; pool too small?")
        if spec.kind == "classification":
            chosen: list[dict] = []
            for label in range(spec.n_labels):
                pool = [r for r in demo_pool if r["target"] == label]
                if len(pool) < 2:
                    raise ValueError(f"label {label} has fewer than 2 demonstration candidates")
                idx = rng.choice(len(pool), size=2, replace=False)
                chosen.extend(pool[i] for i in idx)
        else:
            chosen = random_sample(demo_pool, spec.k, rng)
        key = tuple(sorted(r["id"] for r in chosen))
        if key in seen:
            continue
        seen.add(key)
        sets.append(chosen)
    return sets

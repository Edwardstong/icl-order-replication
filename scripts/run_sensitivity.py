#!/usr/bin/env python
"""The experiment: ordering vs selection sensitivity of GPT-5-nano on AG News (paper Section 3).

    python scripts/run_sensitivity.py --M 2 --P 2 --n-test 16 --tag smoke   # 64 requests, under a cent
    python scripts/run_sensitivity.py                                        # paper grid: 10 x 10 x 500 = 50,000 requests

Draws M demonstration sets and P random orderings, evaluates every (set, ordering) pair on the
test examples, and reduces the M x P accuracy grid to the two sensitivities and r.
Writes results/<model>__ag_news[__tag]/: summary.json (settings, metrics, API usage, cost),
acc.csv (the M x P grid), perms.json and demo_sets.json (the exact orderings and example ids),
correct.npy (per-example correctness, gitignored).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iclorder.dataset import K, load_splits, sample_demo_sets  # noqa: E402
from iclorder.metrics import sensitivities  # noqa: E402
from iclorder.openai_scorer import OpenAIScorer  # noqa: E402
from iclorder.prompts import apply_perm, build_prefix, default_order, random_permutations, render_query  # noqa: E402

# The paper's values for this cell (arXiv:2511.09700 v1, Table 4): ordering, selection.
PAPER_TABLE4 = {"gpt-5-nano": (0.03792, 0.03026)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-5-nano", help="OpenAI model id (the paper used gpt-5-nano)")
    ap.add_argument("--M", type=int, default=10, help="demonstration sets (paper: 10)")
    ap.add_argument("--P", type=int, default=10, help="orderings per set (paper: 10)")
    ap.add_argument("--n-test", type=int, default=500, help="test examples (paper: 500)")
    ap.add_argument("--seed", type=int, default=0, help="demo sets use seed, orderings use seed + 1")
    ap.add_argument("--ddof", type=int, default=0, help="0 population std (default), 1 sample std")
    ap.add_argument("--tag", default="", help="suffix for the output folder, e.g. smoke")
    ap.add_argument("--workers", type=int, default=8, help="parallel requests (24 stayed under the tier-1 rate limit)")
    ap.add_argument("--reasoning-effort", default="minimal", help="minimal | low | medium | high; 'none' omits it")
    ap.add_argument("--max-completion-tokens", type=int, default=64)
    args = ap.parse_args()

    demo_pool, test, meta = load_splits(ROOT / "data")
    test = test[: args.n_test]
    demo_sets = sample_demo_sets(demo_pool, args.M, seed=args.seed)
    perms = random_permutations(K, args.P, seed=args.seed + 1)
    queries = [render_query(r) for r in test]
    golds = np.array([int(r["target"]) for r in test])

    effort = None if args.reasoning_effort in ("none", "") else args.reasoning_effort
    scorer = OpenAIScorer(args.model, workers=args.workers, reasoning_effort=effort,
                          max_completion_tokens=args.max_completion_tokens)
    out_dir = ROOT / "results" / (f"{args.model.replace('/', '__')}__ag_news" + (f"__{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.model}: M={args.M} sets x P={args.P} orderings x N={len(test)} test examples, k={K} demonstrations")

    correct = np.zeros((args.M, args.P, len(test)), dtype=np.uint8)
    t0 = time.time()
    with tqdm(total=args.M * args.P, unit="cell") as bar:
        for i, demos in enumerate(demo_sets):
            ordered = default_order(demos)
            for j, perm in enumerate(perms):
                prefix = build_prefix(apply_perm(ordered, perm))
                preds = scorer.predict([prefix + q for q in queries])
                correct[i, j] = (preds == golds).astype(np.uint8)
                bar.update(1)
    acc = correct.mean(axis=2)
    sens = sensitivities(acc, ddof=args.ddof)

    np.save(out_dir / "correct.npy", correct)
    with (out_dir / "acc.csv").open("w") as f:
        f.write("demo_set," + ",".join(f"ordering_{j}" for j in range(args.P)) + "\n")
        for i in range(args.M):
            f.write(f"{i}," + ",".join(f"{v:.6g}" for v in acc[i]) + "\n")
    (out_dir / "perms.json").write_text(json.dumps([list(p) for p in perms], indent=2))
    (out_dir / "demo_sets.json").write_text(json.dumps([[r["id"] for r in s] for s in demo_sets], indent=2))
    summary = {
        "model": args.model, "dataset": "ag_news", "M": args.M, "P": args.P, "n_test": len(test), "k": K,
        "seed": args.seed, "tag": args.tag, "ddof": args.ddof, **sens,
        "run_info": {"seconds": time.time() - t0, "evaluations": int(args.M * args.P * len(test))},
        "data_meta": meta,
        "env": {"python": platform.python_version(), "openai": __import__("openai").__version__,
                "platform": platform.platform()},
        "api": scorer.usage_summary(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    paper = PAPER_TABLE4.get(args.model)
    print(f"\n{'':22}{'ours':>10}" + (f"{'paper':>10}" if paper else ""))
    print(f"{'ordering sensitivity':22}{sens['ordering']:>10.4f}" + (f"{paper[0]:>10.4f}" if paper else ""))
    print(f"{'selection sensitivity':22}{sens['selection']:>10.4f}" + (f"{paper[1]:>10.4f}" if paper else ""))
    print(f"{'r = selection/ordering':22}{sens['r']:>10.2f}" + (f"{paper[1] / paper[0]:>10.2f}" if paper else ""))
    print(f"{'mean accuracy':22}{sens['mean_acc']:>10.3f}")
    a = summary["api"]
    cost = a["estimated_cost_usd"]
    print(f"\napi: {a['requests']} requests, {a['cache_hits']} from cache, {a['unmatched']} replies named no label, "
          f"{a['alias_matched']} matched via alias, est. cost {'unknown' if cost is None else f'${cost:.2f}'}, "
          f"{summary['run_info']['seconds'] / 60:.1f} min")
    print(f"wrote {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

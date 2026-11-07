"""Section 3: ordering vs selection sensitivity for one model on one dataset.

    python scripts/run_sensitivity.py --model Qwen/Qwen2.5-0.5B --dataset ag_news
    python scripts/run_sensitivity.py --model Qwen/Qwen2.5-0.5B --dataset ag_news --M 2 --P 2 --n-test 16 --tag smoke

Paper defaults: M=10 demonstration sets, P=10 permutations, 500 test examples.
Writes results/sensitivity/<model>__<dataset>[__tag]/{acc.npy, correct.npy, perms.json, demo_sets.json, summary.json}.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from _common import DATA_DIR, RESULTS_DIR, add_model_args, dump, env_info, model_slug
from iclorder.data import load_materialized, sample_demo_sets
from iclorder.metrics import accuracy_matrix, sensitivities
from iclorder.prompts import random_permutations
from iclorder.runner import evaluate_grid
from iclorder.scorer import CachedScorer
from iclorder.specs import get_spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--M", type=int, default=10, help="demonstration sets")
    ap.add_argument("--P", type=int, default=10, help="permutations per set")
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--ddof", type=int, default=0, help="0 population std (default), 1 sample std")
    args = ap.parse_args()

    spec = get_spec(args.dataset)
    demo_pool, _dev, test, meta = load_materialized(DATA_DIR, spec)
    test = test[: args.n_test]
    demo_sets = sample_demo_sets(demo_pool, spec, args.M, seed=args.seed)
    perms = random_permutations(spec.k, args.P, seed=args.seed + 1)

    out_dir = RESULTS_DIR / "sensitivity" / (f"{model_slug(args.model)}__{spec.name}" + (f"__{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer = CachedScorer(args.model, device=args.device, dtype=args.dtype, batch_size=args.batch_size)
    print(f"model loaded in {scorer.load_seconds:.1f}s on {scorer.device} ({scorer.dtype}); "
          f"M={args.M} P={args.P} N={len(test)} k={spec.k}")

    correct, info = evaluate_grid(scorer, spec, demo_sets, perms, test, desc=f"{spec.name} sensitivity",
                                  max_new_tokens=args.max_new_tokens)
    acc = accuracy_matrix(correct)
    sens = sensitivities(acc, ddof=args.ddof)

    np.save(out_dir / "correct.npy", correct)
    np.save(out_dir / "acc.npy", acc)
    dump(out_dir / "perms.json", [list(p) for p in perms])
    dump(out_dir / "demo_sets.json", [[r["id"] for r in s] for s in demo_sets])
    summary = {
        "model": args.model, "dataset": spec.name, "paper_dataset": spec.paper_name, "kind": spec.kind,
        "M": args.M, "P": args.P, "n_test": len(test), "k": spec.k, "seed": args.seed, "tag": args.tag, "ddof": args.ddof,
        "device": str(scorer.device), "dtype": str(scorer.dtype), "batch_size": args.batch_size,
        **sens, "run_info": info, "data_meta": meta, "env": env_info(),
    }
    dump(out_dir / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("ordering", "selection", "r", "mean_acc")}, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

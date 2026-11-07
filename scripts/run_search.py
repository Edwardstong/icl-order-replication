"""Section 4: pick the best of P random orderings on a dev set, evaluate on test.

    python scripts/run_search.py --model Qwen/Qwen2.5-0.5B --dataset ag_news --P 32 --n-dev 250
    python scripts/run_search.py --model Qwen/Qwen2.5-0.5B --dataset ag_news --M 2 --P 3 --n-dev 8 --n-test 8 --tag smoke

Paper defaults: M=10, P=128, N_dev=1000, N_test=500. The paper's own text says
dev performance plateaus around 250 examples and 16-32 permutations recover
most of the oracle for classification; the reduced defaults here follow that.
Writes results/search/<model>__<dataset>[__tag]/{dev_correct.npy, test_correct.npy, perms.json, demo_sets.json, summary.json}.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from _common import DATA_DIR, RESULTS_DIR, add_model_args, dump, env_info, make_scorer, model_slug, scorer_extra
from iclorder.data import load_materialized, sample_demo_sets
from iclorder.prompts import random_permutations
from iclorder.runner import evaluate_grid
from iclorder.search import aggregates, curve_over_dev_size, curve_over_p
from iclorder.specs import get_spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--P", type=int, default=32)
    ap.add_argument("--n-dev", type=int, default=250)
    ap.add_argument("--n-test", type=int, default=500)
    args = ap.parse_args()

    spec = get_spec(args.dataset)
    demo_pool, dev, test, meta = load_materialized(DATA_DIR, spec)
    dev, test = dev[: args.n_dev], test[: args.n_test]
    demo_sets = sample_demo_sets(demo_pool, spec, args.M, seed=args.seed)
    perms = random_permutations(spec.k, args.P, seed=args.seed + 1)

    out_dir = RESULTS_DIR / "search" / (f"{model_slug(args.model)}__{spec.name}" + (f"__{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer = make_scorer(args)
    print(f"model loaded in {scorer.load_seconds:.1f}s on {scorer.device} ({scorer.dtype}); "
          f"M={args.M} P={args.P} N_dev={len(dev)} N_test={len(test)} k={spec.k}")

    # One prefix per (set, permutation) serves both dev and test queries.
    examples = dev + test
    correct, info = evaluate_grid(scorer, spec, demo_sets, perms, examples, desc=f"{spec.name} search",
                                  max_new_tokens=args.max_new_tokens)
    dev_c, test_c = correct[:, :, : len(dev)], correct[:, :, len(dev):]

    np.save(out_dir / "dev_correct.npy", dev_c)
    np.save(out_dir / "test_correct.npy", test_c)
    dump(out_dir / "perms.json", [list(p) for p in perms])
    dump(out_dir / "demo_sets.json", [[r["id"] for r in s] for s in demo_sets])
    agg = aggregates(dev_c, test_c)
    summary = {
        "model": args.model, "dataset": spec.name, "paper_dataset": spec.paper_name, "kind": spec.kind,
        "M": args.M, "P": args.P, "n_dev": len(dev), "n_test": len(test), "k": spec.k, "seed": args.seed, "tag": args.tag,
        "device": str(scorer.device), "dtype": str(scorer.dtype), "batch_size": args.batch_size,
        **agg,
        "curve_over_p": curve_over_p(dev_c, test_c, [1, 2, 4, 8, 16, 32, 64, 100, 128]),
        "curve_over_dev_size": curve_over_dev_size(dev_c, test_c, [50, 100, 250, 500, 1000]),
        "run_info": info, "data_meta": meta, "env": env_info(), **scorer_extra(args, scorer),
    }
    dump(out_dir / "summary.json", summary)
    print(json.dumps({k: agg[k] for k in ("highest_dev_pooled", "highest_dev_per_set", "average", "oracle",
                                          "frac_of_oracle_pooled")}, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

"""Materialize demo_pool / dev / test for one dataset.

    python scripts/build_splits.py --dataset ag_news
    python scripts/build_splits.py --dataset mmlu --n-dev 1000 --n-test 500 --seed 0
"""
from __future__ import annotations

import argparse

from _common import DATA_DIR
from iclorder.data import build_splits
from iclorder.specs import SPECS, get_spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(SPECS))
    ap.add_argument("--n-dev", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    spec = get_spec(args.dataset)
    meta = build_splits(spec, args.n_dev, args.n_test, args.seed, DATA_DIR / spec.name)
    print(f"{spec.name}: demo_pool={meta['n_demo_pool']} dev={meta['n_dev']} (from {meta['dev_source_split']}) "
          f"test={meta['n_test']} k={meta['k']} -> {DATA_DIR / spec.name}")


if __name__ == "__main__":
    main()

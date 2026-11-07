#!/usr/bin/env python
"""Download AG News and write data/ag_news/{demo_pool,dev,test}.jsonl + meta.json (seeded, ~35 MB).

    python scripts/build_splits.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iclorder.dataset import build_splits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-dev", type=int, default=1000, help="paper: 1,000 (unused by the sensitivity run)")
    ap.add_argument("--n-test", type=int, default=500, help="paper: 500")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "ag_news")
    args = ap.parse_args()
    meta = build_splits(args.n_dev, args.n_test, args.seed, args.out)
    print(f"ag_news: demo_pool={meta['n_demo_pool']} dev={meta['n_dev']} test={meta['n_test']} k={meta['k']} -> {args.out}")


if __name__ == "__main__":
    main()

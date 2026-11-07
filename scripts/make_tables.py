"""Collect every results/*/summary.json into CSV tables next to the paper's values.

    python scripts/make_tables.py          # skips tagged runs (smoke, timing, ...)
    python scripts/make_tables.py --all    # include them

Writes results/tables/sensitivity.csv and results/tables/search.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from _common import RESULTS_DIR, ROOT

PAPER = ROOT / "paper"


def load_paper_table4() -> dict[tuple[str, str], dict]:
    out = {}
    with (PAPER / "table4_sensitivity_by_dataset.csv").open() as f:
        for row in csv.DictReader(f):
            out[(row["hf_id"], row["dataset"])] = row
            out[(row["hf_id"].split()[0], row["dataset"])] = row  # "gpt-5-nano (OpenAI API)" -> "gpt-5-nano"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="include runs that were given a --tag")
    args = ap.parse_args()
    paper = load_paper_table4()
    tables = RESULTS_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in sorted((RESULTS_DIR / "sensitivity").glob("*/summary.json")):
        d = json.loads(s.read_text())
        if d.get("tag") and not args.all:
            continue
        p = paper.get((d["model"], d["paper_dataset"]), {})
        rows.append({
            "model": d["model"], "dataset": d["dataset"], "tag": d.get("tag", ""), "M": d["M"], "P": d["P"], "n_test": d["n_test"],
            "mean_acc": round(d["mean_acc"], 4),
            "ordering": round(d["ordering"], 5), "selection": round(d["selection"], 5), "r": round(d["r"], 3),
            "paper_ordering": p.get("ordering", ""), "paper_selection": p.get("selection", ""),
            "paper_r": (round(float(p["selection"]) / float(p["ordering"]), 3) if p else ""),
            "folder": s.parent.name,
        })
    if rows:
        with (tables / "sensitivity.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"sensitivity: {len(rows)} runs -> {tables / 'sensitivity.csv'}")
    else:
        print("no sensitivity runs found")

    rows = []
    for s in sorted((RESULTS_DIR / "search").glob("*/summary.json")):
        d = json.loads(s.read_text())
        if d.get("tag") and not args.all:
            continue
        rows.append({
            "model": d["model"], "dataset": d["dataset"], "tag": d.get("tag", ""), "M": d["M"], "P": d["P"], "n_dev": d["n_dev"], "n_test": d["n_test"],
            "highest_dev_pooled": round(d["highest_dev_pooled"], 4), "highest_dev_per_set": round(d["highest_dev_per_set"], 4),
            "average": round(d["average"], 4), "oracle": round(d["oracle"], 4),
            "frac_of_oracle_pooled": round(d["frac_of_oracle_pooled"], 4), "folder": s.parent.name,
        })
    if rows:
        with (tables / "search.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"search: {len(rows)} runs -> {tables / 'search.csv'}")
    else:
        print("no search runs found")


if __name__ == "__main__":
    main()

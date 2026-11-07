#!/usr/bin/env python
"""Render the results section of REPORT.md and its figures from results/.

Reads every results/sensitivity/*/summary.json and results/search/*/summary.json
(runs that were given a --tag are skipped unless --include-tags), the paper's
tables in paper/, writes report/figures/*.png, and replaces the block between
<!-- RESULTS:BEGIN --> and <!-- RESULTS:END --> in REPORT.md. Re-run it whenever
a cell finishes; it is idempotent.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
REPORT = ROOT / "REPORT.md"
FIGDIR = ROOT / "report" / "figures"
BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"

# Colours: the dataviz reference palette, light surface. Colour follows the
# quantity (ordering = blue, selection = orange, oracle = aqua); ours vs paper is
# carried by fill, never by a second hue.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRID, AXIS, SURFACE, INK, INK2 = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#1a1a19", "#5f5e58"


def load_runs(kind: str, include_tags: bool) -> list[dict]:
    runs = []
    for f in sorted((RESULTS / kind).glob("*/summary.json")):
        s = json.loads(f.read_text())
        if s.get("tag") and not include_tags:
            continue
        s["folder"] = f.parent.name
        runs.append(s)
    runs.sort(key=lambda s: (s["model"], s["dataset"]))
    return runs


def paper_cells() -> dict[tuple[str, str], tuple[float, float]]:
    cells = {}
    with open(PAPER / "table4_sensitivity_by_dataset.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            cells[(r["hf_id"], r["dataset"])] = (float(r["ordering"]), float(r["selection"]))
            cells[(r["hf_id"].split()[0], r["dataset"])] = cells[(r["hf_id"], r["dataset"])]
    return cells


def short(model: str) -> str:
    return model.split("/")[-1]


def f4(x) -> str:
    return "-" if x is None else f"{x:.4f}"


def f3(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


# ----------------------------------------------------------------------------- Section 3

def sensitivity_table(runs: list[dict], paper: dict) -> tuple[str, dict]:
    lines = [
        "| Model | Dataset | Mean acc | Ordering (ours) | Selection (ours) | r (ours) | Ordering (paper) | Selection (paper) | r (paper) | Same side of 1? |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    ours_o, ours_s, ours_r, pap_o, pap_s, pap_r, agree = [], [], [], [], [], [], []
    for s in runs:
        cell = paper.get((s["model"], s.get("paper_dataset", s["dataset"])))
        po, ps = cell if cell else (None, None)
        pr = (ps / po) if cell and po else None
        same = "-"
        if pr is not None and s["ordering"]:
            same = "yes" if (s["r"] >= 1) == (pr >= 1) else "no"
            agree.append(same == "yes")
        ours_o.append(s["ordering"]); ours_s.append(s["selection"]); ours_r.append(s["r"])
        if cell:
            pap_o.append(po); pap_s.append(ps); pap_r.append(pr)
        ds = s["dataset"] + (f" ({s['tag']})" if s.get("tag") else "")
        lines.append(
            f"| {short(s['model'])} | {ds} | {f3(s['mean_acc'])} | {f4(s['ordering'])} | "
            f"{f4(s['selection'])} | {s['r']:.2f} | {f4(po)} | {f4(ps)} | "
            f"{'-' if pr is None else f'{pr:.2f}'} | {same} |"
        )
    n = len(runs)
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    summary = {
        "n": n,
        "ours": (mean(ours_o), mean(ours_s), mean(ours_r)),
        "paper": (mean(pap_o), mean(pap_s), mean(pap_r)),
        "agree": (sum(agree), len(agree)),
    }
    if n:
        o, s_, r = summary["ours"]
        lines.append(f"| **mean, ours ({n} cells)** | | | **{f4(o)}** | **{f4(s_)}** | **{r:.2f}** | | | | |")
        if pap_o:
            o, s_, r = summary["paper"]
            lines.append(f"| **mean, paper (same cells)** | | | | | | **{f4(o)}** | **{f4(s_)}** | **{r:.2f}** | |")
    return "\n".join(lines), summary


def fig_sensitivity(runs: list[dict], paper: dict, path: Path) -> bool:
    if not runs:
        return False
    fig, ax = plt.subplots(figsize=(max(5.5, 1.3 * len(runs) + 2), 3.6), facecolor=SURFACE)
    xs = list(range(len(runs)))
    labels = [f"{short(s['model'])}\n{s['dataset']}" for s in runs]
    mk = dict(markersize=8, linestyle="none", markeredgewidth=1.6)
    ax.plot(xs, [s["ordering"] for s in runs], marker="o", color=BLUE, label="ordering, ours", **mk)
    ax.plot(xs, [s["selection"] for s in runs], marker="o", color=ORANGE, label="selection, ours", **mk)
    po = [paper.get((s["model"], s.get("paper_dataset", s["dataset"])), (None, None))[0] for s in runs]
    ps = [paper.get((s["model"], s.get("paper_dataset", s["dataset"])), (None, None))[1] for s in runs]
    ax.plot(xs, [v if v is not None else float("nan") for v in po], marker="o", markerfacecolor="none",
            color=BLUE, label="ordering, paper", **mk)
    ax.plot(xs, [v if v is not None else float("nan") for v in ps], marker="o", markerfacecolor="none",
            color=ORANGE, label="selection, paper", **mk)
    for x, s in zip(xs, runs):  # a thin tie between the two of ours
        ax.plot([x, x], [s["ordering"], s["selection"]], color=AXIS, linewidth=1, zorder=0)
    ax.set_xticks(xs, labels)
    ax.set_ylim(bottom=0)
    style(ax, "Ordering vs selection sensitivity (std of accuracy)", "", "sensitivity")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.28))
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------- Section 4

def search_table(runs: list[dict]) -> str:
    lines = [
        "| Model | Dataset | P | N dev | Average | Highest-dev (pooled) | Highest-dev (per set) | Oracle | Fraction of oracle (pooled) | Fraction (per set) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in runs:
        ds = s["dataset"] + (f" ({s['tag']})" if s.get("tag") else "")
        lines.append(
            f"| {short(s['model'])} | {ds} | {s['P']} | {s['n_dev']} | {f3(s['average'])} | "
            f"{f3(s['highest_dev_pooled'])} | {f3(s['highest_dev_per_set'])} | {f3(s['oracle'])} | "
            f"{s['frac_of_oracle_pooled']:.3f} | {s['frac_of_oracle_per_set']:.3f} |"
        )
    return "\n".join(lines)


def _dev_key(row: dict) -> str:
    for k in ("n_dev", "dev_size", "n"):
        if k in row:
            return k
    raise KeyError(f"no dev-size key in {sorted(row)}")


def fig_curve_p(runs: list[dict], path: Path) -> bool:
    runs = [s for s in runs if s.get("curve_over_p")]
    if not runs:
        return False
    fig, axes = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 3.6), squeeze=False, facecolor=SURFACE)
    for ax, s in zip(axes[0], runs):
        c = s["curve_over_p"]
        p = [r["p"] for r in c]
        lw = dict(linewidth=2, marker="o", markersize=6)
        ax.plot(p, [r["oracle"] for r in c], color=AQUA, linestyle="--", label="oracle (max over orderings)", **lw)
        ax.plot(p, [r["highest_dev_pooled"] for r in c], color=BLUE, label="highest-dev ordering", **lw)
        ax.plot(p, [r["average"] for r in c], color=ORANGE, label="average ordering", **lw)
        ax.set_xscale("log", base=2)
        ax.set_xticks(p, [str(v) for v in p])
        ax.minorticks_off()
        style(ax, f"{short(s['model'])}, {s['dataset']}\ntest accuracy vs number of orderings sampled", "P (orderings sampled)", "test accuracy")
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return True


def fig_curve_dev(runs: list[dict], path: Path) -> bool:
    runs = [s for s in runs if s.get("curve_over_dev_size")]
    if not runs:
        return False
    fig, axes = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 3.6), squeeze=False, facecolor=SURFACE)
    for ax, s in zip(axes[0], runs):
        c = s["curve_over_dev_size"]
        k = _dev_key(c[0])
        n = [r[k] for r in c]
        ax.axhline(s["oracle"], color=AQUA, linewidth=2.5, linestyle="--", label="oracle (best ordering on test)")
        ax.plot(n, [r["highest_dev_pooled"] for r in c], color=BLUE, linewidth=2, marker="o", markersize=6,
                label="highest-dev ordering")
        ax.axhline(s["average"], color=ORANGE, linewidth=2, linestyle="--", label="average ordering")
        ax.set_xscale("log", base=2)
        ax.set_xticks(n, [str(v) for v in n])
        ax.minorticks_off()
        style(ax, f"{short(s['model'])}, {s['dataset']}\ntest accuracy vs dev-set size", "dev examples used to pick the ordering", "test accuracy")
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return True


def first_p_reaching(curve: list[dict], frac: float):
    for r in curve:
        if r["frac_of_oracle_pooled"] >= frac:
            return r["p"]
    return None


# ----------------------------------------------------------------------------- assemble

def build_block(sens: list[dict], search: list[dict], paper: dict, figs: dict[str, bool]) -> str:
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    out = [BEGIN, f"<!-- generated by scripts/fill_report.py on {now}; do not edit by hand -->", ""]
    if not sens and not search:
        out += ["*No finished runs yet. Run `scripts/minimal.sh`, then `scripts/fill_report.py`.*", END]
        return "\n".join(out)

    env = (sens or search)[0]
    out += [
        f"Runs finished so far: {len(sens)} Section 3 cell(s), {len(search)} Section 4 cell(s). "
        f"Hardware: {env.get('device')} / {str(env.get('dtype')).replace('torch.', '')}; "
        f"torch {env['env']['torch']}, transformers {env['env']['transformers']}, Python {env['env']['python']}.",
        "",
    ]

    out += ["### 4.1 Ordering versus selection sensitivity (paper Section 3, Table 4)", ""]
    if sens:
        table, summ = sensitivity_table(sens, paper)
        out += [table, ""]
        o, s_, r = summ["ours"]
        sent = [
            f"Over our {summ['n']} cell(s): mean ordering sensitivity {o:.4f}, mean selection sensitivity {s_:.4f}, "
            f"mean r = {r:.2f}, so selection {'dominates' if r > 1 else 'does not dominate'} on average."
        ]
        if summ["paper"][0] is not None:
            po, ps, pr = summ["paper"]
            sent.append(f"The paper's values for the same cells: {po:.4f}, {ps:.4f}, mean r = {pr:.2f}.")
        a, b = summ["agree"]
        if b:
            sent.append(f"In {a} of {b} cells our r falls on the same side of 1 as the paper's.")
        out += [" ".join(sent), ""]
        if figs.get("sens"):
            out += ["![Ordering vs selection sensitivity](report/figures/sensitivity.png)", ""]
        sizes = {(s["M"], s["P"], s["n_test"]) for s in sens}
        out.append("Grid size(s): " + "; ".join(f"M={m}, P={p}, {n} test examples" for m, p, n in sorted(sizes)) + ".")
        out.append("")
    else:
        out += ["*No Section 3 cell finished yet.*", ""]

    out += ["### 4.2 Picking an ordering on a dev set (paper Section 4)", ""]
    if search:
        out += [search_table(search), ""]
        for s in search:
            p98 = first_p_reaching(s.get("curve_over_p", []), 0.98)
            out.append(
                f"- {short(s['model'])} on {s['dataset']}: the highest-dev ordering recovers "
                f"{100 * s['frac_of_oracle_pooled']:.1f}% of the oracle test accuracy (pooled rule; "
                f"{100 * s['frac_of_oracle_per_set']:.1f}% per set), against {100 * s['average'] / s['oracle']:.1f}% "
                f"for a random ordering. "
                + (f"98% of the oracle is first reached at P = {p98}." if p98 else "98% of the oracle is not reached within the P tried.")
            )
        out.append("")
        if figs.get("curve_p"):
            out += ["![Accuracy vs number of orderings](report/figures/curve_over_p.png)", ""]
        if figs.get("curve_dev"):
            out += ["![Accuracy vs dev-set size](report/figures/curve_over_dev.png)", ""]
    else:
        out += ["*No Section 4 cell finished yet.*", ""]

    out += ["Run folders: " + ", ".join(f"`{s['folder']}`" for s in sens + search) + ".", END]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-tags", action="store_true", help="also use runs that were given a --tag (smoke, timing)")
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args()

    if not args.report.exists() or BEGIN not in args.report.read_text() or END not in args.report.read_text():
        sys.exit(f"{args.report} must exist and contain {BEGIN} ... {END}")

    sens = load_runs("sensitivity", args.include_tags)
    search = load_runs("search", args.include_tags)
    paper = paper_cells()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figs = {
        "sens": fig_sensitivity(sens, paper, FIGDIR / "sensitivity.png"),
        "curve_p": fig_curve_p(search, FIGDIR / "curve_over_p.png"),
        "curve_dev": fig_curve_dev(search, FIGDIR / "curve_over_dev.png"),
    }
    block = build_block(sens, search, paper, figs)
    text = args.report.read_text()
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    args.report.write_text(head + block + tail)
    print(f"{args.report.name}: {len(sens)} sensitivity + {len(search)} search run(s); figures: "
          + ", ".join(k for k, v in figs.items() if v) or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Draw the M x P accuracy grid of a run with its row and column spreads.

    python scripts/plot_grid.py                               # results/gpt-5-nano__ag_news
    python scripts/plot_grid.py results/gpt-5-nano__ag_news__smoke

Rows are demonstration sets, columns are orderings. The bars on the right are each row's std
over orderings (their mean is the ordering sensitivity); the bars below are each column's std
over sets (their mean is the selection sensitivity). Writes accuracy_grid.png into the run folder.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# One-hue sequential ramp (blue, light -> dark) and chart chrome from the dataviz reference palette.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
BAR, SURFACE, INK, INK2, MUTED, GRID = "#2a78d6", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "gpt-5-nano__ag_news"
    s = json.loads((run / "summary.json").read_text())
    with (run / "acc.csv").open() as f:
        acc = np.array([[float(v) for v in row[1:]] for row in list(csv.reader(f))[1:]])
    M, P = acc.shape
    row_std, col_std = acc.std(axis=1, ddof=s["ddof"]), acc.std(axis=0, ddof=s["ddof"])

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
    fig = plt.figure(figsize=(8.0, 6.8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.42, 0.06], height_ratios=[1, 0.36], wspace=0.08, hspace=0.1,
                          left=0.09, right=0.9, top=0.82, bottom=0.07)
    ax = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1], sharey=ax)
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax)
    cax = fig.add_subplot(gs[0, 2])
    for a in (ax, ax_r, ax_b):
        a.set_facecolor(SURFACE)
        for side in a.spines.values():
            side.set_visible(False)
        a.tick_params(which="both", length=0)

    cmap = LinearSegmentedColormap.from_list("blue_seq", RAMP)
    im = ax.imshow(acc, cmap=cmap, aspect="auto", vmin=acc.min(), vmax=acc.max())
    # a 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, P, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, M, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.set_xticks(range(P), [str(j) for j in range(P)])
    ax.set_yticks(range(M), [str(i) for i in range(M)])
    ax.set_xlabel("ordering j", labelpad=4)
    ax.set_ylabel("demonstration set i")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=INK2)
    cb.set_label("accuracy on 500 test examples" if s["n_test"] == 500 else f"accuracy on {s['n_test']} test examples",
                 color=INK2)

    ax_r.barh(range(M), row_std, height=0.72, color=BAR)
    ax_r.axvline(s["ordering"], color=INK2, linewidth=1, linestyle=(0, (3, 2)))
    ax_r.set_xlim(0, max(row_std.max(), col_std.max()) * 1.25)
    ax_r.tick_params(which="both", labelleft=False, left=False)
    ax_r.set_xlabel("std over orderings", labelpad=4)
    ax_r.xaxis.set_label_position("top")
    ax_r.xaxis.tick_top()
    ax_r.grid(axis="x", color=GRID, linewidth=0.8)
    ax_r.set_axisbelow(True)
    ax_r.text(s["ordering"], M - 0.4, f"mean = ordering\nsensitivity {s['ordering']:.4f}", color=INK2, fontsize=8,
              ha="left", va="top")

    ax_b.bar(range(P), col_std, width=0.72, color=BAR)
    ax_b.axhline(s["selection"], color=INK2, linewidth=1, linestyle=(0, (3, 2)))
    ax_b.set_ylim(0, max(row_std.max(), col_std.max()) * 1.25)
    ax_b.invert_yaxis()
    ax_b.tick_params(which="both", labelbottom=False, bottom=False)
    ax_b.set_ylabel("std over sets")
    ax_b.grid(axis="y", color=GRID, linewidth=0.8)
    ax_b.set_axisbelow(True)
    ax_b.text(P - 0.3, s["selection"], f"mean = selection\nsensitivity {s['selection']:.4f}", color=INK2, fontsize=8,
              ha="left", va="center", clip_on=False)

    fig.text(0.09, 0.955, f"{s['model']} on AG News: accuracy over {M} demonstration sets x {P} orderings",
             fontsize=11.5, color=INK, weight="bold")
    fig.text(0.09, 0.922, f"ordering sensitivity {s['ordering']:.4f}, selection sensitivity {s['selection']:.4f}, "
             f"r = {s['r']:.2f}", fontsize=9.5, color=INK2)
    fig.text(0.09, 0.894, f"mean accuracy {s['mean_acc']:.3f}, range {acc.min():.3f} to {acc.max():.3f} "
             f"over the {M * P} cells", fontsize=9.5, color=INK2)
    out = run / "accuracy_grid.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

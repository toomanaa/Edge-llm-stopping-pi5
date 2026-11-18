#!/usr/bin/env python3
"""Simple version of the C5 figure: energy per correct answer, and waiting time (typical and slowest 5%).
Run from the repo root:  python3 src/analysis/make_fig_c5_simple.py
Reads experiments/c5_deployment/results/c5_numbers.json (written by make_fig_c5.py)."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = "experiments/c5_deployment/results/"
N = json.load(open(D + "c5_numbers.json"))
GRAY, BLUE, ORANGE, INK, INK2, GRID = "#8a8984", "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
modes = [("full", "No stopping", GRAY), ("controller", "Learned stopper", BLUE), ("cap", "Token limit (210)", ORANGE)]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#b8b7b2", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1, 1.15]})
for ax in (a1, a2):
    ax.spines[["top", "right"]].set_visible(False); ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, lw=0.8)

# left: energy per correct answer
for i, (k, lab, col) in enumerate(modes):
    v, lo, hi = N[k]["J_per_correct"], N[k]["lo"], N[k]["hi"]
    a1.bar(i, v, 0.62, color=col)
    a1.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=INK, capsize=3, lw=1)
    a1.text(i, hi + 5, f"{v:.0f} J", ha="center", fontsize=9, fontweight="bold")
a1.set_xticks(range(3)); a1.set_xticklabels([m[1].replace(" (", "\n(").replace("Learned ", "Learned\n").replace("No ", "No\n") for m in modes], fontsize=8)
a1.set_ylim(0, 270); a1.set_ylabel("Joules per correct answer")
a1.set_title("Energy spent per correct answer", loc="left", fontsize=9.5, fontweight="bold")

# right: waiting time for explanation questions
L = N["explain_latency_s"]; w = 0.26
for j, (k, lab, col) in enumerate(modes):
    for g, key in enumerate(("median", "p95")):
        x = g + (j - 1) * (w + 0.02); v = L[k][key]
        a2.bar(x, v, w, color=col, label=lab if g == 0 else None)
        a2.text(x, v + 1, f"{v:.0f} s", ha="center", fontsize=8.5)
a2.set_xticks([0, 1]); a2.set_xticklabels(["Typical wait\n(half the answers are faster)", "Slowest 5% of answers"], fontsize=8)
a2.set_ylim(0, 62); a2.set_ylabel("Seconds until the answer is complete")
a2.set_title("Waiting time (explanation questions)", loc="left", fontsize=9.5, fontweight="bold")
a2.legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(D + "fig_c5_simple.pdf"); fig.savefig(D + "fig_c5_simple.png", dpi=200)
print("saved", D + "fig_c5_simple.pdf")

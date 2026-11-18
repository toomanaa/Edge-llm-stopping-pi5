#!/usr/bin/env python3
"""Figure for the battery-drain test: answers finished over time, and answers per charge. Run from the repo root.
  python3 src/analysis/make_fig_battery.py
Reads experiments/c5_deployment/battery/raw/{A,B}_runs.jsonl (A = stopping on, B = no stopping)."""
import json, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src/common")
from correctness import is_correct
ap = argparse.ArgumentParser()
ap.add_argument("--A", default="experiments/c5_deployment/battery/raw/A_runs.jsonl")
ap.add_argument("--B", default="experiments/c5_deployment/battery/raw/B_runs.jsonl")
ap.add_argument("--data", default="data/qa_v2_heldout300.jsonl")
ap.add_argument("--out", default="experiments/c5_deployment/battery/results/")
a = ap.parse_args()
Q = {j["id"]: j for j in map(json.loads, open(a.data))}
def load(f):
    R = []
    for l in open(f, encoding="utf-8", errors="replace"):
        l = l.replace("\x00", "").strip()
        if not l: continue
        try: r = json.loads(l)
        except ValueError: continue
        if all(k in r for k in ("id", "text", "t_start", "t_end")): R.append(r)
    R.sort(key=lambda r: r["t_end"])
    t0 = R[0]["t_start"]
    h = np.array([(r["t_end"] - t0) / 3600 for r in R])
    ok = np.array([bool(is_correct(r["text"], Q[r["id"].rsplit("_r", 1)[0]]["answers"], Q[r["id"].rsplit("_r", 1)[0]].get("match_mode", "any"))) for r in R])
    return h, ok
hA, okA = load(a.A); hB, okB = load(a.B)

BLUE, GRAY, INK, INK2, GRID = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#b8b7b2", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1.35, 1]})
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False); ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, lw=0.8)

# (a) answers finished as the charge runs down
for h, col, lab in ((hB, GRAY, "No stopping"), (hA, BLUE, "Stopping on")):
    x = np.concatenate([[0], h]); y = np.arange(len(x))
    ax1.step(x, y, where="post", color=col, lw=2, label=lab)
    ax1.text(x[-1] + 0.12, y[-1], f"{len(h)}", color=INK, va="center", fontsize=9, fontweight="bold")
ax1.set_xlim(0, max(hA[-1], hB[-1]) + 0.9); ax1.set_ylim(0, max(len(hA), len(hB)) * 1.08)
ax1.set_xlabel("Hours since start (pack at 100%)"); ax1.set_ylabel("Answers finished so far")
ax1.set_title("Answers finished as the battery runs down", loc="left", fontsize=9.5, fontweight="bold")
ax1.legend(frameon=False, loc="upper left")

# (b) answers on one full charge
cats = ["All answers", "Correct answers"]
nA, nB = len(hA), len(hB); cA, cB = int(okA.sum()), int(okB.sum())
xs = np.arange(2); w = 0.36
ax2.bar(xs - w / 2 - 0.01, [nA, cA], w, color=BLUE, label="Stopping on")
ax2.bar(xs + w / 2 + 0.01, [nB, cB], w, color=GRAY, label="No stopping")
for i, (va, vb) in enumerate(((nA, nB), (cA, cB))):
    ax2.text(i - w / 2 - 0.01, va + 0.015 * max(nA, nB), str(va), ha="center", fontsize=8.5)
    ax2.text(i + w / 2 + 0.01, vb + 0.015 * max(nA, nB), str(vb), ha="center", fontsize=8.5)
    ax2.text(i, max(va, vb) * 1.14, f"×{va / vb:.2f}", ha="center", fontsize=10, fontweight="bold", color=BLUE)
ax2.set_xticks(xs); ax2.set_xticklabels(cats); ax2.set_ylim(0, max(nA, nB) * 1.3)
ax2.set_ylabel("On one full charge")
ax2.set_title("Answers on one full charge", loc="left", fontsize=9.5, fontweight="bold")
fig.tight_layout()
fig.savefig(a.out + "fig_battery.pdf"); fig.savefig(a.out + "fig_battery.png", dpi=200)
json.dump(dict(answers_stopping=nA, answers_no_stopping=nB, correct_stopping=cA, correct_no_stopping=cB,
               ratio_answers=nA / nB, ratio_correct=cA / cB, hours_stopping=float(hA[-1]), hours_no_stopping=float(hB[-1])),
          open(a.out + "battery_figure_numbers.json", "w"), indent=1)
print(f"stopping on: {nA} answers, {cA} correct, {hA[-1]:.2f} h;  no stopping: {nB} answers, {cB} correct, {hB[-1]:.2f} h")

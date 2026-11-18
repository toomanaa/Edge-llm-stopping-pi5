#!/usr/bin/env python3
"""Per-question view for Contribution 1: energy with stopping against energy without stopping. Run from the repo root."""
import json, bisect, sys, numpy as np
sys.path.insert(0, "src/common")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "experiments/c1_energy_savings/raw/"; OUT = "experiments/c1_energy_savings/results/"
Q = {j["id"]: j for j in map(json.loads, open("data/qa_v2_heldout300.jsonl"))}
runs = [r for r in map(json.loads, open(D + "run300_runs.jsonl"))]
P = np.loadtxt(D + "run300_power.csv", delimiter=",", skiprows=1); ts, ws = P[:, 0].tolist(), P[:, 1].tolist()
def joules(t0, t1, max_gap=5.0):
    i = max(bisect.bisect_right(ts, t0) - 1, 0); tot = 0.0
    for k in range(i, len(ts) - 1):
        x, y = ts[k], ts[k+1]
        if x >= t1: break
        if y <= t0 or y - x > max_gap: continue
        lo, hi = max(x, t0), min(y, t1)
        if hi <= lo: continue
        wa = ws[k] + (ws[k+1]-ws[k])*(lo-x)/(y-x); wb = ws[k] + (ws[k+1]-ws[k])*(hi-x)/(y-x); tot += 0.5*(wa+wb)*(hi-lo)
    return tot
E = {}
for r in runs: E.setdefault(r["mode"], {})[r["id"]] = joules(r["t_start"], r["t_end"])
ids = sorted(E["full"]); typ = np.array([Q[i]["type"] for i in ids]); x = np.array([E["full"][i] for i in ids])
col = {"factual": "#1baf7a", "explain": "#6250d6"}; name = {"factual": "Factual questions", "explain": "Explanation questions"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.0), sharex=True, sharey=True)
for a, m, ttl in zip(ax, ("controller", "cap"), ("Learned stopper", "Token cap (210 tokens)")):
    y = np.array([E[m][i] for i in ids]); lim = [5, 400]
    a.plot(lim, lim, color="#9a9992", lw=0.9, zorder=1); a.plot(lim, [0.5*lim[0], 0.5*lim[1]], color="#9a9992", lw=0.9, ls="--", zorder=1)
    for t in ("factual", "explain"):
        s = typ == t; a.scatter(x[s], y[s], s=14, color=col[t], alpha=0.8, edgecolor="#fcfcfb", linewidth=0.5, label=name[t], zorder=2)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(*lim); a.set_ylim(*lim); a.set_title(ttl, fontsize=9.5, loc="left")
    a.set_xlabel("Energy with no stopping (J)"); a.grid(color="#e3e2dd", lw=0.5, zorder=0)
ax[0].set_ylabel("Energy with stopping (J)"); ax[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "fig_c1_per_question.pdf"); fig.savefig(OUT + "fig_c1_per_question.png", dpi=160)
for t in ("factual", "explain"):
    s = typ == t
    print(t, "median energy no stopping %.0f J; questions where the controller used under 60%% of no-stopping energy: %d of %d" % (np.median(x[s]), sum(np.array([E["controller"][i] for i in ids])[s] < 0.6 * x[s]), s.sum()))

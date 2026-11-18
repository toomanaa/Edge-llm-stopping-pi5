#!/usr/bin/env python3
"""Contribution 5: energy per correct answer, response time, projected answers per charge. Run from the repo root."""
import json, sys, bisect, numpy as np
sys.path.insert(0, "src/common")
from correctness import is_correct
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "experiments/c1_energy_savings/raw/"; OUT = "experiments/c5_deployment/results/"
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
R = {}
for r in runs:
    q = Q[r["id"]]; R.setdefault(r["mode"], {})[r["id"]] = (joules(r["t_start"], r["t_end"]), float(is_correct(r["text"], q["answers"], q.get("match_mode", "any"))), r["wall_seconds"])
ids = sorted(R["full"]); typ = np.array([Q[i]["type"] for i in ids]); n = len(ids)
E = {m: np.array([R[m][i][0] for i in ids]) for m in R}; C = {m: np.array([R[m][i][1] for i in ids]) for m in R}; T = {m: np.array([R[m][i][2] for i in ids]) for m in R}
rng = np.random.default_rng(11); B = 10000; ix = rng.integers(0, n, (B, n))
res = {}
for m in ("full", "controller", "cap"):
    pc = E[m].sum() / C[m].sum(); pcb = E[m][ix].sum(1) / C[m][ix].sum(1)
    res[m] = dict(J_per_correct=pc, lo=np.percentile(pcb, 2.5), hi=np.percentile(pcb, 97.5), J_per_q=E[m].mean(), mean_s=T[m].mean())
for m in ("controller", "cap"):
    r1 = (E["full"][ix].sum(1) / E["full"][ix].sum(1)); 
    cr = (E["full"].sum() / C["full"].sum()) / (E[m].sum() / C[m].sum()); crb = (E["full"][ix].sum(1) / C["full"][ix].sum(1)) / (E[m][ix].sum(1) / C[m][ix].sum(1))
    qr = E["full"].sum() / E[m].sum(); qrb = E["full"][ix].sum(1) / E[m][ix].sum(1)
    res[m].update(correct_ratio=cr, cr_lo=np.percentile(crb, 2.5), cr_hi=np.percentile(crb, 97.5), q_ratio=qr, qr_lo=np.percentile(qrb, 2.5), qr_hi=np.percentile(qrb, 97.5),
                  saving_per_correct=100 * (1 - 1 / cr))
ex = typ == "explain"; lat = {}
for m in R:
    t = T[m][ex]; sm = np.array([np.median(t[rng.integers(0, ex.sum(), ex.sum())]) for _ in range(4000)])
    lat[m] = dict(median=np.median(t), med_lo=np.percentile(sm, 2.5), med_hi=np.percentile(sm, 97.5), p95=np.percentile(t, 95), mean=t.mean())
res["explain_latency_s"] = lat
json.dump(res, open(OUT + "c5_numbers.json", "w"), indent=1, default=float)
print("energy per correct answer (J):", {m: (round(res[m]["J_per_correct"], 1), round(res[m]["lo"], 1), round(res[m]["hi"], 1)) for m in ("full", "controller", "cap")})
for m in ("controller", "cap"):
    d = res[m]; print(f"{m}: correct answers per joule x{d['correct_ratio']:.2f} (95% CI {d['cr_lo']:.2f}-{d['cr_hi']:.2f}); responses per joule x{d['q_ratio']:.2f} ({d['qr_lo']:.2f}-{d['qr_hi']:.2f}); energy per correct answer {d['saving_per_correct']:.1f}% lower")
print("explain-question response time (s):", {m: (round(v["median"], 1), round(v["med_lo"], 1), round(v["med_hi"], 1), round(v["p95"], 1)) for m, v in lat.items()})
col = {"full": "#8a8984", "controller": "#2a78d6", "cap": "#eb6834"}; lab = {"full": "No stopping", "controller": "Learned stopper", "cap": "Token limit (210)"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8), gridspec_kw={"width_ratios": [1, 1.25]})
a = ax[0]
for k, m in enumerate(("full", "controller", "cap")):
    d = res[m]; a.bar(k, d["J_per_correct"], width=0.58, color=col[m], zorder=2); a.errorbar(k, d["J_per_correct"], yerr=[[d["J_per_correct"]-d["lo"]], [d["hi"]-d["J_per_correct"]]], color="#0b0b0b", capsize=3, lw=1, zorder=3)
    a.text(k, d["hi"] + 4, f"{d['J_per_correct']:.0f} J", ha="center", fontsize=8.5)
a.set_xticks(range(3)); a.set_xticklabels(["No\nstopping", "Learned\nstopper", "Token limit\n(210)"], fontsize=8); a.set_ylim(0, 260); a.set_ylabel("Joules per correct answer")
a.set_title("Energy per correct answer", fontsize=9.5, loc="left"); a.grid(axis="y", color="#e3e2dd", lw=0.6, zorder=0); a.set_axisbelow(True)
b = ax[1]
for m in ("full", "controller", "cap"):
    t = np.sort(T[m][ex]); b.step(t, np.arange(1, len(t) + 1) / len(t) * 100, where="post", color=col[m], lw=1.6, label=lab[m], zorder=3)
b.axhline(50, color="#c9c8c2", lw=0.8, ls="--", zorder=1); b.text(1, 52, "median", fontsize=7.5, color="#52514e")
b.set_xlabel("Response time for explanation questions (s)"); b.set_ylabel("Questions answered by this time (%)"); b.set_xlim(0, 64); b.set_ylim(0, 101)
b.set_title("Waiting time", fontsize=9.5, loc="left"); b.grid(color="#e3e2dd", lw=0.5, zorder=0); b.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(OUT + "fig_c5_deployment.pdf"); fig.savefig(OUT + "fig_c5_deployment.png", dpi=160)

import json, sys, bisect, numpy as np
sys.path.insert(0, "src/common")
from correctness import is_correct
OUT = "experiments/c1_energy_savings/results/"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
A="experiments/c1_energy_savings/raw/"
Q = {j["id"]: j for j in map(json.loads, open("data/qa_v2_heldout300.jsonl"))}
runs = [r for r in map(json.loads, open(A + "run300_runs.jsonl"))]
P = np.loadtxt(A + "run300_power.csv", delimiter=",", skiprows=1); ts, ws = P[:, 0].tolist(), P[:, 1].tolist()
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
    q = Q[r["id"]]; R.setdefault(r["mode"], {})[r["id"]] = (joules(r["t_start"], r["t_end"]), is_correct(r["text"], q["answers"], q.get("match_mode", "any")))
ids = sorted(R["full"]); rng = np.random.default_rng(7); B = 10000; ix = rng.integers(0, len(ids), (B, len(ids)))
E = {m: np.array([R[m][i][0] for i in ids]) for m in R}; C = {m: np.array([R[m][i][1] for i in ids], float) for m in R}
out = {}
for m in R:
    me = np.percentile(E[m][ix].mean(1), [2.5, 97.5]); ma = np.percentile(C[m][ix].mean(1), [2.5, 97.5])
    out[m] = dict(E=E[m].mean(), Elo=me[0], Ehi=me[1], acc=100*C[m].mean(), alo=100*ma[0], ahi=100*ma[1],
                  per_correct=E[m].sum()/C[m].sum())
for m in ("controller", "cap"):
    ratio = E[m][ix].sum(1) / E["full"][ix].sum(1); dacc = (C[m] - C["full"])[ix].mean(1)
    out[m].update(ratio=E[m].sum()/E["full"].sum(), rlo=np.percentile(ratio, 2.5), rhi=np.percentile(ratio, 97.5),
                  dacc=100*(C[m]-C["full"]).mean(), dlo=100*np.percentile(dacc, 2.5), dhi=100*np.percentile(dacc, 97.5))
json.dump(out, open(OUT+"c1_numbers.json", "w"), indent=1)
for m, d in out.items(): print(m, {k: round(v, 3) for k, v in d.items()})
# figure: two panels, one measure each (no dual axis)
col = {"full": "#8a8984", "controller": "#2a78d6", "cap": "#eb6834"}
lab = {"full": "No\nstopping", "controller": "Learned\nstopper", "cap": "Token cap\n(210 tokens)"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))
for a, (key, lo, hi, ttl, unit, fmt) in zip(ax, [("E", "Elo", "Ehi", "Energy per question", "J", "{:.0f}"), ("acc", "alo", "ahi", "Accuracy", "%", "{:.1f}")]):
    for k, m in enumerate(["full", "controller", "cap"]):
        d = out[m]; a.bar(k, d[key], width=0.58, color=col[m], zorder=2)
        a.errorbar(k, d[key], yerr=[[d[key]-d[lo]], [d[hi]-d[key]]], color="#0b0b0b", capsize=3, lw=1, zorder=3)
        a.text(k, d[hi] + (0.02 if key == "E" else 0.0) * d[hi] + (2 if key == "E" else 1.2), fmt.format(d[key]) + ("" if unit == "%" else " ") + unit, ha="center", va="bottom", fontsize=8.5, color="#0b0b0b")
    a.set_xticks(range(3)); a.set_xticklabels([lab[m] for m in ("full", "controller", "cap")], fontsize=8)
    a.set_title(ttl, fontsize=9.5, loc="left", color="#0b0b0b"); a.grid(axis="y", color="#e3e2dd", lw=0.6, zorder=0); a.set_axisbelow(True)
ax[0].set_ylim(0, 185); ax[1].set_ylim(0, 100); ax[0].set_ylabel("Joules"); ax[1].set_ylabel("Percent correct")
fig.tight_layout(); fig.savefig(OUT+"fig_c1_energy_accuracy.pdf"); fig.savefig(OUT+"fig_c1_energy_accuracy.png", dpi=160)

# by question type
for ty in ("factual", "explain"):
    sel = np.array([Q[i]["type"] == ty for i in ids])
    r = {m: (E[m][sel].sum(), 100*C[m][sel].mean(), E[m][sel].mean()) for m in R}
    ix2 = rng.integers(0, sel.sum(), (B, sel.sum()))
    sv = {m: (1 - (E[m][sel][ix2].sum(1) / E["full"][sel][ix2].sum(1)))*100 for m in ("controller", "cap")}
    print(ty, sel.sum(), {m: (round(r[m][2],1), round(r[m][1],1)) for m in r}, {m: (round(100*(1-r[m][0]/r["full"][0]),1), [round(x,1) for x in np.percentile(sv[m],[2.5,97.5])]) for m in sv})

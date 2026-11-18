#!/usr/bin/env python3
"""Repeat of the C1 energy run (cooler board). Same method as make_fig_c1.py, with file names as arguments.
usage (from the repo root):
python3 src/analysis/c1_repeat_analysis.py --runs experiments/c1_energy_savings/raw_cool/run300_cool_runs.jsonl \
  --power .../run300_cool_power.csv --temp .../run300_cool_temp.csv --first experiments/c1_energy_savings/results/c1_numbers.json \
  --first-temp experiments/c1_energy_savings/raw/run300_temp.csv --outdir experiments/c1_energy_savings/results_cool"""
import argparse, json, sys, bisect, os, re
import numpy as np
sys.path.insert(0, "src/common")
from correctness import is_correct
ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True); ap.add_argument("--power", required=True); ap.add_argument("--temp", required=True)
ap.add_argument("--first", required=True); ap.add_argument("--first-temp", required=True)
ap.add_argument("--first-runs", default="experiments/c1_energy_savings/raw/run300_runs.jsonl")
ap.add_argument("--data", default="data/qa_v2_heldout300.jsonl"); ap.add_argument("--outdir", required=True)
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
L = []
def P(s=""): print(s); L.append(s)

Q = {j["id"]: j for j in map(json.loads, open(a.data))}
runs = [r for r in map(json.loads, open(a.runs))]
Pw = np.loadtxt(a.power, delimiter=",", skiprows=1); ts, ws = Pw[:, 0].tolist(), Pw[:, 1].tolist()
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
    out[m] = dict(E=E[m].mean(), Elo=me[0], Ehi=me[1], acc=100*C[m].mean(), alo=100*ma[0], ahi=100*ma[1], per_correct=E[m].sum()/C[m].sum())
for m in ("controller", "cap"):
    ratio = E[m][ix].sum(1) / E["full"][ix].sum(1); dacc = (C[m] - C["full"])[ix].mean(1)
    out[m].update(ratio=E[m].sum()/E["full"].sum(), rlo=np.percentile(ratio, 2.5), rhi=np.percentile(ratio, 97.5),
                  dacc=100*(C[m]-C["full"]).mean(), dlo=100*np.percentile(dacc, 2.5), dhi=100*np.percentile(dacc, 97.5))
c = E["controller"][ix].sum(1) / E["cap"][ix].sum(1)
out["controller_vs_cap"] = dict(ratio=E["controller"].sum()/E["cap"].sum(), lo=np.percentile(c, 2.5), hi=np.percentile(c, 97.5))
f = json.load(open(a.first))

def temps(p):
    T, Th = [], []
    for line in open(p, errors="replace"):
        m = re.search(r"temp=([\d.]+).*throttled=(0x[0-9a-f]+)", line)
        if m: T.append(float(m.group(1))); Th.append(int(m.group(2), 16))
    T = np.array(T); Th = np.array(Th)
    return dict(n=len(T), median=float(np.median(T)), p95=float(np.percentile(T, 95)), max=float(T.max()),
                throttle_now_pct=100*float(np.mean((Th & 0xF) != 0)), any_flag_pct=100*float(np.mean(Th != 0)), flags=sorted({hex(x) for x in Th}))
t2, t1 = temps(a.temp), temps(a.first_temp)
out["temps_repeat"] = t2; out["temps_first"] = t1

P(f"Repeat run (cooler board): {len(ids)} questions, {len(runs)} runs")
P(f"{'':<22}{'first run':>16}{'cooler run':>16}")
def row(name, x, y, fmt="{:.1f}"): P(f"{name:<22}{fmt.format(x):>16}{fmt.format(y):>16}")
for m, nm in (("full", "no stopping"), ("controller", "learned stopper"), ("cap", "token cap 210")):
    row(f"{nm} J/question", f[m]["E"], out[m]["E"]); row(f"{nm} accuracy %", f[m]["acc"], out[m]["acc"])
P()
for m, nm in (("controller", "learned stopper"), ("cap", "token cap")):
    P(f"{nm}: energy saving first {100*(1-f[m]['ratio']):.1f}% ({100*(1-f[m]['rhi']):.1f}-{100*(1-f[m]['rlo']):.1f}), cooler {100*(1-out[m]['ratio']):.1f}% ({100*(1-out[m]['rhi']):.1f}-{100*(1-out[m]['rlo']):.1f})")
    P(f"{nm}: accuracy change first {f[m]['dacc']:+.1f} ({f[m]['dlo']:+.1f} to {f[m]['dhi']:+.1f}), cooler {out[m]['dacc']:+.1f} ({out[m]['dlo']:+.1f} to {out[m]['dhi']:+.1f})")
cv = out["controller_vs_cap"]; P(f"controller vs cap energy ratio, cooler: {cv['ratio']:.3f} ({cv['lo']:.3f}-{cv['hi']:.3f})  [first run 1.029 (0.958-1.102)]")
P(f"energy per correct answer: full {out['full']['per_correct']:.0f} J, controller {out['controller']['per_correct']:.0f} J  [first run {f['full']['per_correct']:.0f} -> {f['controller']['per_correct']:.0f}]")
P(f"Full-run energy cooler/first: {out['full']['E']/f['full']['E']:.3f}")
stopped = sum(1 for r in runs if r["mode"] == "controller" and r.get("stopped_early_by_probe"))
fr = [r for r in map(json.loads, open(a.first_runs))]
n1 = sum(1 for r in fr if r["mode"] == "controller" and r.get("stopped_early_by_probe"))
P(f"controller stopped early: cooler {stopped} of {len(ids)}, first {n1} of {len(ids)}")
d = {r["id"]: r for r in runs if r["mode"] == "controller"}; d1 = {r["id"]: r for r in fr if r["mode"] == "controller"}
P(f"controller text identical to first run: {sum(1 for i in d if i in d1 and d[i]['text'] == d1[i]['text'])} of {len(d)}")
P()
P(f"Board temperature  first: median {t1['median']:.1f}, 95th pct {t1['p95']:.1f}, max {t1['max']:.1f} C; throttling active in {t1['throttle_now_pct']:.1f}% of readings; flags {t1['flags']}")
P(f"Board temperature cooler: median {t2['median']:.1f}, 95th pct {t2['p95']:.1f}, max {t2['max']:.1f} C; throttling active in {t2['throttle_now_pct']:.1f}% of readings; flags {t2['flags']}")
open(os.path.join(a.outdir, "c1_repeat_output.txt"), "w").write("\n".join(L) + "\n")
json.dump(out, open(os.path.join(a.outdir, "c1_repeat_numbers.json"), "w"), indent=1)

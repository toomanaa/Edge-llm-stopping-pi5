#!/usr/bin/env python3
"""Compare the conditions run by sp_controller on the same questions.
  python3 analyze_runs.py --runs pilot_runs.jsonl --power pilot_power.csv --data qa_v2_dev40.jsonl --correctness DIR
Energy of a run = power integrated over [t_start, t_end] (window-clipped trapezoid; gaps > 5 s skipped)."""
import argparse, bisect, json, sys
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True); ap.add_argument("--power", required=True)
ap.add_argument("--data", required=True); ap.add_argument("--correctness", default="scripts/controller")
ap.add_argument("--baseline", default="full"); ap.add_argument("--boot", type=int, default=10000)
a = ap.parse_args()
sys.path.insert(0, a.correctness)
from correctness import is_correct
Q = {j["id"]: j for j in map(json.loads, open(a.data))}
runs = [json.loads(l) for l in open(a.runs)]
P = np.loadtxt(a.power, delimiter=",", skiprows=1); ts, ws = P[:, 0].tolist(), P[:, 1].tolist()
def joules(t0, t1, max_gap=5.0):
    if not (ts[0] <= t0 and t1 <= ts[-1]): return float("nan")
    i = max(bisect.bisect_right(ts, t0) - 1, 0); tot = 0.0
    for k in range(i, len(ts) - 1):
        x, y = ts[k], ts[k + 1]
        if x >= t1: break
        if y <= t0 or y - x > max_gap: continue
        lo, hi = max(x, t0), min(y, t1)
        if hi <= lo: continue
        wa = ws[k] + (ws[k+1] - ws[k]) * (lo - x) / (y - x); wb = ws[k] + (ws[k+1] - ws[k]) * (hi - x) / (y - x)
        tot += 0.5 * (wa + wb) * (hi - lo)
    return tot
R = {}
for r in runs:
    q = Q[r["id"]]
    R.setdefault(r["mode"], {})[r["id"]] = dict(
        ok=bool(is_correct(r["text"], q["answers"], q.get("match_mode", "any"))), E=joules(r["t_start"], r["t_end"]),
        lat=r["wall_seconds"], tok=r["n_tokens"], early=r["stopped_early_by_probe"], over=r.get("check_overhead_seconds", 0.0),
        checks=r.get("boundary_checks", 0), rb=r.get("n_rollbacks", 0), reason=r.get("end_reason", ""))
common = sorted(set.intersection(*[set(v) for v in R.values()]))
print(f"questions with all conditions: {len(common)}  (power covers: {sum(all(np.isfinite(R[m][i]['E']) for m in R) for i in common)})")
common = [i for i in common if all(np.isfinite(R[m][i]['E']) for m in R)]
print(f"\n{'mode':<11}{'acc %':>7}{'energy J':>10}{'J/question':>12}{'J/correct':>11}{'tokens':>8}{'med s':>7}{'p95 s':>7}{'early %':>9}{'checks':>8}{'overhead s':>11}")
rows = {}
for m in R:
    d = [R[m][i] for i in common]; E = np.array([x["E"] for x in d]); ok = np.array([x["ok"] for x in d]); lat = np.array([x["lat"] for x in d])
    rows[m] = (E, ok)
    print(f"{m:<11}{ok.mean()*100:7.1f}{E.sum():10.0f}{E.mean():12.1f}{E.sum()/max(ok.sum(),1):11.1f}{np.mean([x['tok'] for x in d]):8.0f}"
          f"{np.median(lat):7.1f}{np.percentile(lat,95):7.1f}{np.mean([x['early'] for x in d])*100:9.1f}{sum(x['checks'] for x in d):8d}{sum(x['over'] for x in d):11.2f}")
rng = np.random.default_rng(1)
def cmp(m1, m0):
    E1, o1 = rows[m1]; E0, o0 = rows[m0]; n = len(E1)
    idx = rng.integers(0, n, size=(a.boot, n))
    dacc = (o1.astype(float) - o0.astype(float)); ratio = E1[idx].sum(1) / E0[idx].sum(1)
    ci_a = np.percentile(dacc[idx].mean(1), [2.5, 97.5]); ci_e = np.percentile(ratio, [2.5, 97.5])
    print(f"{m1} vs {m0}: accuracy {dacc.mean()*100:+.1f} pp (95% CI {ci_a[0]*100:+.1f}..{ci_a[1]*100:+.1f}); "
          f"energy ratio {E1.sum()/E0.sum():.3f} (95% CI {ci_e[0]:.3f}..{ci_e[1]:.3f}), saving {(1-E1.sum()/E0.sum())*100:.1f}%")
print()
for m in R:
    if m != a.baseline and a.baseline in R: cmp(m, a.baseline)
if "controller" in R and "cap" in R: cmp("controller", "cap")
if "controller" in R:
    d = [R["controller"][i] for i in common]
    print(f"\ncontroller: {sum(x['rb'] for x in d)} KV rollbacks over {sum(x['checks'] for x in d)} checks; "
          f"time spent in checks {sum(x['over'] for x in d):.1f} s of {sum(x['lat'] for x in d):.0f} s total")

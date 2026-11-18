#!/usr/bin/env python3
"""Integrity checks for the 300-question run (Contribution 1). Run from the repo root."""
import json, re, bisect, numpy as np
D = "experiments/c1_energy_savings/raw/"
Q = {j["id"]: j for j in map(json.loads, open("data/qa_v2_heldout300.jsonl"))}
import sys; sys.path.insert(0, "src/common")
runs = [r for r in map(json.loads, open(D + "run300_runs.jsonl"))]
by = {}
for r in runs: by.setdefault(r["id"], {})[r["mode"]] = r
eq = pre = bad = 0; badl = []
for i, m in by.items():
    f = m["full"]["text"].strip()
    for k in ("controller", "cap"):
        t = m[k]["text"].strip()
        if t == f: eq += 1
        elif f.startswith(t): pre += 1
        else: bad += 1; badl.append((i, k))
print(f"1) stopped text vs no-stopping text (greedy decoding): identical {eq}, proper prefix {pre}, diverged {bad} {badl}")
print("   (possible cause: after a rollback the controller re-decodes in a different batch shape, which can change a later token through tiny numerical differences; not investigated further)")
P = np.loadtxt(D + "run300_power.csv", delimiter=",", skiprows=1); ts = P[:, 0]; gaps = np.diff(ts)
print(f"2) power samples {len(ts)}, rate {len(ts)/(ts[-1]-ts[0]):.1f} Hz, max gap {gaps.max():.2f} s, gaps > 5 s: {(gaps>5).sum()}")
ok = 0
for r in runs:
    a = bisect.bisect_left(ts, r["t_start"]); b = bisect.bisect_right(ts, r["t_end"]); seg = ts[max(a-1, 0):b+1]
    ok += (np.diff(seg).max() if len(seg) > 1 else 99) <= 0.2
print(f"   runs with no sample gap above 0.2 s inside their window: {ok}/{len(runs)}")
T = []
for l in open(D + "run300_temp.csv"):
    m = re.match(r"(\d+) temp=([\d.]+)'C throttled=(0x[0-9a-f]+)", l)
    if m: T.append((int(m.group(1)), float(m.group(2)), int(m.group(3), 16)))
tt = [x[0] for x in T]; tv = np.array([x[1] for x in T]); tf = np.array([x[2] for x in T])
print(f"3) temperature: median {np.median(tv):.1f} C, max {tv.max():.1f} C; samples with a current throttle bit set: {100*np.mean((tf&0xF)!=0):.1f}%")
for mode in ("full", "controller", "cap"):
    ix = [min(bisect.bisect_left(tt, r["t_start"]), len(tt)-1) for r in runs if r["mode"] == mode]
    print(f"   {mode:<11} mean temperature at run start {tv[ix].mean():.1f} C; throttled at start {100*np.mean((tf[ix]&0xF)!=0):.0f}%")
tok = lambda md: np.mean([r["n_tokens"]/r["wall_seconds"] for r in runs if r["mode"] == md and r["n_tokens"] > 100])
print("4) tokens per second on long runs: full %.2f, controller %.2f, cap %.2f" % (tok("full"), tok("controller"), tok("cap")))

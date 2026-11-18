#!/usr/bin/env python3
"""Held-out re-analysis of the live controller (0.99, check_every_n=2) vs the
no-stopping run. Run from the repo root:  python3 results/e2_controller/heldout_analysis.py
Needs numpy, scipy and correctness.py (scripts/controller/correctness.py)."""
import json, re, random, sys, os
import numpy as np
from collections import defaultdict
from scipy.stats import binomtest
R = sys.argv[1] if len(sys.argv) > 1 else "results"
sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else "scripts/controller")
from correctness import is_correct as is_correct_shared

gen = [json.loads(l) for l in open(f"{R}/e1_v2_1074/qa_v2_gen_merged.jsonl")]
ctl = {r["id"]: r for r in map(json.loads, open(f"{R}/e2_controller/live_full_v2.jsonl"))}
G = {g["id"]: g for g in gen}
ids = [g["id"] for g in gen]

# --- probe-training questions (re-created exactly as in the notebook) ---
BOUNDARY = re.compile(r"(?<=[.!?\n])\s+")
def plain(text, answers, mode):
    t = text.lower()
    return all(a.lower() in t for a in answers) if mode == "all" else any(a.lower() in t for a in answers)
def bounds(text):
    offs = [m.start() for m in BOUNDARY.finditer(text)]
    if not offs or offs[-1] < len(text): offs.append(len(text))
    return offs
allb = []
for r in gen:
    if not plain(r["text"], r["answers"], r.get("match_mode", "any")): continue
    offs = bounds(r["text"])
    if len(offs) < 2: continue
    for b, off in enumerate(offs):
        allb.append((r["id"], plain(r["text"][:off], r["answers"], r.get("match_mode", "any"))))
by = defaultdict(list)
for i, s in allb: by[i].append(s)
rids = list(by.keys()); random.seed(123); random.shuffle(rids)
sampled = []; sampled_ids = []
for rid in rids:
    if len(sampled) >= 4500: break
    sampled.extend(by[rid]); sampled_ids.append(rid)
print(f"re-created probe sample: {len(sampled)} boundaries, {len(sampled_ids)} responses, "
      f"{sum(1 for s in sampled if not s)} unsafe (notebook: 4500 / 170 / 1059)")
probe_ids = set(sampled_ids)
calib_ids = {f"qav2_{i:04d}" for i in list(range(0, 100)) + list(range(200, 230))}

# --- subsets ---
ans_key = lambda i: (G[i].get("match_mode", "any"), tuple(sorted(a.lower() for a in G[i]["answers"])))
removed = (probe_ids | calib_ids) & set(ids)
removed_keys = {ans_key(i) for i in removed}
subsets = {
    "all 1074": ids,
    "held-out (no probe-train, no calibration)": [i for i in ids if i not in removed],
    "strict held-out (also no shared-answer paraphrases)": [i for i in ids if i not in removed and ans_key(i) not in removed_keys],
}

# --- per-question energy from the power CSVs (window-clipped trapezoid; logging gaps are skipped) ---
import bisect
def load_power(p):
    a = np.loadtxt(p, delimiter=",", skiprows=1)
    return a[:, 0].tolist(), a[:, 1].tolist()
def joules(ts, ws, t0, t1, max_gap=5.0):
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
def energy(P, a, b):
    return joules(*P, a, b) if P[0][0] <= a and b <= P[0][-1] else np.nan
PB = load_power(f"{R}/e1_v2_1074/qa_v2_power_merged.csv")
PC = load_power(f"{R}/e2_controller/live_full_v2_power.csv")
E_base = {i: energy(PB, G[i]["t_start"], G[i]["t_end"]) for i in ids}
E_ctl = {i: energy(PC, ctl[i]["t_start"], ctl[i]["t_end"]) for i in ids}
ok = lambda i: not (np.isnan(E_base[i]) or np.isnan(E_ctl[i]))
print(f"sanity (all questions with power in that run): baseline {np.nansum(list(E_base.values())):.1f} J, controller {np.nansum(list(E_ctl.values())):.1f} J "
      f"(the earlier integration gave 246665.86 / 146313.55; the baseline figure included a 74.5 kJ logging-gap artefact on qav2_1026)")

base_ok = {i: is_correct_shared(G[i]["text"], G[i]["answers"], G[i].get("match_mode", "any")) for i in ids}
ctl_ok = {i: bool(ctl[i]["correct"]) for i in ids}
base_lat = {i: G[i]["t_end"] - G[i]["t_start"] for i in ids}
ctl_lat = {i: ctl[i]["wall_seconds"] for i in ids}
print(f"baseline correct {sum(base_ok.values())}/1074, controller correct {sum(ctl_ok.values())}/1074")

rng = np.random.default_rng(2026)
def boot(f, n, B=10000):
    idx = rng.integers(0, n, size=(B, n)); v = np.array([f(r) for r in idx])
    return np.percentile(v, [2.5, 97.5])
out = []
for name, S in subsets.items():
    n = len(S)
    b = np.array([base_ok[i] for i in S]); c = np.array([ctl_ok[i] for i in S])
    both = int((b & c).sum()); bo = int((b & ~c).sum()); co = int((~b & c).sum()); nei = int((~b & ~c).sum())
    mc = binomtest(bo, bo + co, 0.5).pvalue if bo + co else 1.0
    Sx = [i for i in S if ok(i)]
    eb = np.array([E_base[i] for i in Sx]); ec = np.array([E_ctl[i] for i in Sx])
    early = np.array([ctl[i]["stopped_early_by_probe"] for i in S])
    d = c.astype(float) - b.astype(float)
    ci_d = boot(lambda r: d[r].mean(), n)
    ci_e = boot(lambda r: ec[r].sum() / eb[r].sum(), len(Sx))
    bc = np.array([base_ok[i] for i in Sx]); cc_ = np.array([ctl_ok[i] for i in Sx])
    lb = np.array([base_lat[i] for i in S]); lc = np.array([ctl_lat[i] for i in S])
    lines = [
        f"== {name}: n = {n} questions ({len(Sx)} with power in both runs)",
        f"accuracy: no-stop {b.mean()*100:.1f}% ({b.sum()}), controller {c.mean()*100:.1f}% ({c.sum()}); diff {d.mean()*100:+.1f} pp, 95% CI [{ci_d[0]*100:+.1f}, {ci_d[1]*100:+.1f}]",
        f"2x2 (rows no-stop, cols controller): correct/correct {both}, correct->wrong {bo}, wrong->correct {co}, wrong/wrong {nei}",
        f"McNemar exact p = {mc:.3g} (discordant pairs {bo}+{co})",
        f"energy (kJ): no-stop {eb.sum()/1000:.1f}, controller {ec.sum()/1000:.1f}; ratio {ec.sum()/eb.sum():.3f} (reduction {(1-ec.sum()/eb.sum())*100:.1f}%), 95% CI of ratio [{ci_e[0]:.3f}, {ci_e[1]:.3f}]",
        f"energy per correct answer (J): no-stop {eb.sum()/max(bc.sum(),1):.0f}, controller {ec.sum()/max(cc_.sum(),1):.0f}",
        f"early-stop rate: {early.mean()*100:.1f}% ({early.sum()})",
        f"latency median/p95 (s): no-stop {np.median(lb):.2f}/{np.percentile(lb,95):.2f}, controller {np.median(lc):.2f}/{np.percentile(lc,95):.2f}",
        ""]
    out += lines
txt = "\n".join(out); print(txt)
open(f"{R}/e2_controller/heldout_analysis.txt", "w").write(txt)

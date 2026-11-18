#!/usr/bin/env python3
"""Battery-drain confirmation (Pi 4, FNIRSI meter): answers and correct answers per charge, with and without stopping.
Run from the repo root:
  python3 src/analysis/battery_analysis.py --data data/qa_v2_heldout300.jsonl \
     --A-runs A_runs.jsonl --A-power A_power.txt --A-name "stopping on" \
     --B-runs B_runs.jsonl --B-power B_power.txt --B-name "no stopping"
Several power files of one run (if the logger was restarted) can be given comma-separated.
Meter columns: timestamp sample voltage_V current_A dp dn temp energy_Ws capacity_As (space separated, ~100 Hz)."""
import argparse, json, sys, io
import numpy as np, pandas as pd
sys.path.insert(0, "src/common")
from correctness import is_correct

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
for k in "AB":
    ap.add_argument(f"--{k}-runs"); ap.add_argument(f"--{k}-power"); ap.add_argument(f"--{k}-name", default=k)
ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--max-gap", type=float, default=5.0)
a = ap.parse_args()
Q = {}
for l in open(a.data):
    j = json.loads(l); Q[j["id"]] = j

def read_meter(f):
    """Read one meter file, tolerant of a cut-off or NUL-filled tail after a power loss."""
    t, v, c = [], [], []; bad = 0
    with open(f, "rb") as fh:
        for line in fh:
            p = line.replace(b"\x00", b"").split()
            if len(p) != 9: bad += 1; continue
            try: t.append(float(p[0])); v.append(float(p[2])); c.append(float(p[3]))
            except ValueError: bad += 1
    print(f"  read {f}: {len(t)} samples, {bad} unreadable lines skipped")
    return pd.DataFrame({"timestamp": t, "voltage_V": v, "current_A": c})

def load_power(files):
    parts = [read_meter(f) for f in files.split(",")]
    df = pd.concat(parts).sort_values("timestamp").drop_duplicates("timestamp")
    t = df.timestamp.values; w = df.voltage_V.values * df.current_A.values
    dt = np.diff(t); seg = 0.5 * (w[1:] + w[:-1]) * dt
    gap = dt > a.max_gap; seg = np.where(gap, 0.0, seg)
    cum = np.concatenate([[0.0], np.cumsum(seg)])           # joules since first sample, gaps skipped
    gaps = [(t[i], t[i + 1]) for i in np.where(gap)[0]]
    return t, w, cum, float(dt[gap].sum()), int(gap.sum()), gaps

def cum_at(t, w, cum, x):
    i = np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 2)
    frac = np.clip((x - t[i]) / (t[i + 1] - t[i]), 0, 1)
    wa = w[i]; wb = w[i] + (w[i + 1] - w[i]) * frac
    return cum[i] + 0.5 * (wa + wb) * frac * (t[i + 1] - t[i])

def run_stats(tag, runs, power, name):
    t, w, cum, gap_s, n_gaps, gaps = load_power(power)
    R = []; bad = 0
    for l in open(runs, encoding="utf-8", errors="replace"):
        l = l.replace("\x00", "").strip()
        if not l: continue
        try: r = json.loads(l)
        except ValueError: bad += 1; continue
        if all(k in r for k in ("id", "text", "t_start", "t_end")):
            R.append(r)
        else: bad += 1
    print(f"  read {runs}: {len(R)} answers, {bad} unreadable lines skipped")
    ok = np.array([bool(is_correct(r["text"], Q[r["id"].rsplit("_r", 1)[0]]["answers"], Q[r["id"].rsplit("_r", 1)[0]].get("match_mode", "any"))) for r in R])
    t0 = np.array([r["t_start"] for r in R]); t1 = np.array([r["t_end"] for r in R])
    inside = (t0 >= t[0]) & (t1 <= t[-1])
    for g0, g1 in gaps: inside &= ~((t0 < g1) & (t1 > g0))     # answers overlapping a meter gap get no energy value
    n_lost = int((~inside).sum())
    E = np.where(inside, cum_at(t, w, cum, t1) - cum_at(t, w, cum, t0), np.nan)
    mean_p = cum[-1] / (t[-1] - t[0] - gap_s); tot_E = cum[-1] + mean_p * gap_s   # gaps filled at the mean power
    dur_h = (t[-1] - t[0]) / 3600
    run_h = (t1.max() - t0.min()) / 3600
    typ = np.array([Q[r["id"].rsplit("_r", 1)[0]]["type"] for r in R])
    print(f"\n=== Run {tag}: {name} ===")
    print(f"meter log: {dur_h:.2f} h, {len(t)} samples, {n_gaps} gaps > {a.max_gap:.0f} s ({gap_s:.0f} s lost)")
    print(f"energy delivered by pack: measured {cum[-1]/3600:.1f} Wh, with gaps filled at mean power {tot_E/3600:.1f} Wh ({tot_E/1000:.0f} kJ); mean power {mean_p:.2f} W")
    print(f"answers without an energy value (overlap a meter gap): {n_lost}")
    print(f"answers completed: {len(R)}   correct: {int(ok.sum())}   accuracy {ok.mean()*100:.1f}%   (factual {np.mean(ok[typ=='factual'])*100:.1f}%, explain {np.mean(ok[typ=='explain'])*100:.1f}%)")
    print(f"answers per hour {len(R)/run_h:.1f}; J per answer {np.nanmean(E):.1f}; J per correct answer {np.nanmean(E)*len(R)/ok.sum():.1f} (answers with a value)")
    print(f"share of energy inside answers: {np.nanmean(E)*len(R)/tot_E*100:.1f}%   (the rest is waiting/idle between answers and start-up)")
    return dict(n=len(R), c=int(ok.sum()), tot_E=tot_E, dur=dur_h, ok=ok, E=E, typ=typ, ids=[r["id"] for r in R], texts={r["id"]: r["text"] for r in R})

res = {}
for k in "AB":
    if getattr(a, f"{k}_runs"):
        res[k] = run_stats(k, getattr(a, f"{k}_runs"), getattr(a, f"{k}_power"), getattr(a, f"{k}_name"))
if len(res) == 2:
    A, B = res["A"], res["B"]; rng = np.random.default_rng(1)
    print("\n=== Comparison: A (first) vs B (second) ===")
    print(f"answers per charge:          {A['n']} vs {B['n']}   ratio x{A['n']/B['n']:.2f}")
    print(f"correct answers per charge:  {A['c']} vs {B['c']}   ratio x{A['c']/B['c']:.2f}")
    print(f"time on one charge:          {A['dur']:.2f} h vs {B['dur']:.2f} h")
    print(f"energy delivered:            {A['tot_E']/3600:.1f} Wh vs {B['tot_E']/3600:.1f} Wh   (if these differ, the charges were not equal; use the per-Wh rows)")
    print(f"answers per Wh:              {A['n']/(A['tot_E']/3600):.2f} vs {B['n']/(B['tot_E']/3600):.2f}   ratio x{(A['n']/A['tot_E'])/(B['n']/B['tot_E']):.2f}")
    print(f"correct answers per Wh:      {A['c']/(A['tot_E']/3600):.2f} vs {B['c']/(B['tot_E']/3600):.2f}   ratio x{(A['c']/A['tot_E'])/(B['c']/B['tot_E']):.2f}")
    # bootstrap over QUESTIONS (the list repeats the same 300 questions, so answers of one question are resampled together)
    def base(i): return i.rsplit("_r", 1)[0]
    def cluster_boot(pairs, B=5000):
        """pairs: list of (base_id, value). Returns the bootstrap distribution of the mean value, resampling base ids."""
        d = {}
        for k, v in pairs: d.setdefault(k, []).append(v)
        ks = list(d); S = np.array([sum(d[k]) for k in ks]); N = np.array([len(d[k]) for k in ks], float)
        idx = rng.integers(0, len(ks), size=(B, len(ks)))
        return S[idx].sum(1) / N[idx].sum(1), len(ks)
    for lab, X in (("A", A), ("B", B)):
        dist, nq = cluster_boot([(base(i), float(o)) for i, o in zip(X["ids"], X["ok"])])
        lo, hi = np.percentile(dist, [2.5, 97.5]); print(f"accuracy {lab}: {X['ok'].mean()*100:.1f}%  (95% CI over {nq} distinct questions {lo*100:.1f}..{hi*100:.1f}; the two runs cover different parts of the repeated list, so compare them with the paired line below)")
    ia = {i: o for i, o in zip(A["ids"], A["ok"])}; ib = {i: o for i, o in zip(B["ids"], B["ok"])}
    com = sorted(set(ia) & set(ib))
    if len(com):
        dist, nq = cluster_boot([(base(i), float(ia[i]) - float(ib[i])) for i in com])
        lo, hi = np.percentile(dist, [2.5, 97.5]); dm = np.mean([float(ia[i]) - float(ib[i]) for i in com])
        print(f"accuracy change (stopping on minus no stopping) on the {len(com)} answers both runs gave ({nq} distinct questions): {dm*100:+.1f} pp (95% CI over questions {lo*100:+.1f}..{hi*100:+.1f})")
    same = sum(1 for i in com if A["texts"].get(i) == B["texts"].get(i))
    print(f"identical answer text in both runs for {same} of {len(com)} common answers (stopping on = shorter text, so this is mostly 0)")
    print("\nNote: a single drain per condition. Claim 'confirmed' only if the ratio is near the projected 1.9x / 1.75x;")
    print("report the measured value and say it is one run per condition (repeat with swapped order if time allows).")

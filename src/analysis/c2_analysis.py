#!/usr/bin/env python3
"""C2: what does one stopping check cost? Compares, on the same questions with no stopping,
   full (no checks), check_single (single-pass check at every n-th sentence end), check_reencode (server-style check:
   the text so far is read again at every check). Run from the repo root:
  python3 src/analysis/c2_analysis.py --runs experiments/c2_check_cost/raw/c2_runs.jsonl --power experiments/c2_check_cost/raw/c2_power.csv
Energy of a run = power integrated over [t_start, t_end] (window-clipped trapezoid; gaps > 5 s skipped)."""
import argparse, bisect, json, sys
sys.path.insert(0, "src/common")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True); ap.add_argument("--power", required=True)
ap.add_argument("--data", default="data/qa_v2_heldout300.jsonl"); ap.add_argument("--out", default="experiments/c2_check_cost/results/")
ap.add_argument("--boot", type=int, default=10000)
a = ap.parse_args()
Q = {j["id"]: j for j in map(json.loads, open(a.data))}
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
for l in open(a.runs, encoding="utf-8", errors="replace"):
    l = l.replace("\x00", "").strip()
    if not l: continue
    try: r = json.loads(l)
    except ValueError: continue
    R.setdefault(r["id"], {})[r["mode"]] = r
MODES = ("full", "check_single", "check_reencode")
ids = sorted(i for i, d in R.items() if all(m in d for m in MODES))
for i in ids:
    for m in MODES: R[i][m]["E"] = joules(R[i][m]["t_start"], R[i][m]["t_end"])
ids = [i for i in ids if all(np.isfinite(R[i][m]["E"]) for m in MODES)]
typ = np.array([Q[i]["type"] for i in ids])
print(f"questions with all three modes and power data: {len(ids)}  (explain {int((typ=='explain').sum())}, factual {int((typ=='factual').sum())})")
same_tok = sum(R[i]["full"]["n_tokens"] == R[i][m]["n_tokens"] for i in ids for m in MODES[1:]); same_txt = sum(R[i]["full"]["text"] == R[i][m]["text"] for i in ids for m in MODES[1:])
print(f"same number of tokens as no-checks: {same_tok} of {2*len(ids)} runs; identical text: {same_txt} of {2*len(ids)}  (the checks do not change what is generated)")
E = {m: np.array([R[i][m]["E"] for i in ids]) for m in MODES}
T = {m: np.array([R[i][m]["wall_seconds"] for i in ids]) for m in MODES}
NC = {m: np.array([R[i][m]["boundary_checks"] for i in ids]) for m in MODES}
rng = np.random.default_rng(1); n = len(ids); idx = rng.integers(0, n, size=(a.boot, n))
def ci(x): return np.percentile(x, [2.5, 97.5])
res = {"n_questions": n}
print(f"\n{'mode':<16}{'J/question':>12}{'s/question':>12}{'checks/question':>17}")
for m in MODES: print(f"{m:<16}{E[m].mean():12.1f}{T[m].mean():12.1f}{NC[m].mean():17.1f}")
print()
for m in MODES[1:]:
    dE = E[m] - E["full"]; dT = T[m] - T["full"]; nc = NC[m]
    pct = dE.sum() / E["full"].sum() * 100; pct_ci = ci(dE[idx].sum(1) / E["full"][idx].sum(1) * 100)
    perc = dE.sum() / nc.sum(); perc_ci = ci(dE[idx].sum(1) / nc[idx].sum(1))
    tpc = dT.sum() / nc.sum(); tpc_ci = ci(dT[idx].sum(1) / nc[idx].sum(1))
    tpct = dT.sum() / T["full"].sum() * 100; tpct_ci = ci(dT[idx].sum(1) / T["full"][idx].sum(1) * 100)
    own = sum(sum(R[i][m]["check_seconds"]) for i in ids) / max(nc.sum(), 1)
    print(f"{m}: extra energy {pct:.1f}% of no-checks energy (95% CI {pct_ci[0]:.1f}..{pct_ci[1]:.1f}); "
          f"{perc:.2f} J per check ({perc_ci[0]:.2f}..{perc_ci[1]:.2f}); extra time {tpct:.1f}% ({tpct_ci[0]:.1f}..{tpct_ci[1]:.1f}), "
          f"{tpc:.3f} s per check ({tpc_ci[0]:.3f}..{tpc_ci[1]:.3f}); checks timed inside the program: {own:.3f} s each")
    # a second, more precise estimate: time spent inside the checks (timed in the program) times the power of that run
    est = np.array([sum(R[i][m]["check_seconds"]) * R[i][m]["E"] / R[i][m]["wall_seconds"] for i in ids])
    est_pct = est.sum() / E["full"].sum() * 100; est_ci = ci(est[idx].sum(1) / E["full"][idx].sum(1) * 100)
    print(f"    from the timers inside the program: {est_pct:.2f}% of no-checks energy (95% CI {est_ci[0]:.2f}..{est_ci[1]:.2f}); {est.sum()/max(nc.sum(),1):.3f} J per check")
    res_extra = dict(timer_energy_pct=est_pct, timer_energy_pct_ci=est_ci.tolist(), timer_J_per_check=est.sum() / max(nc.sum(), 1))
    for t in ("explain", "factual"):
        s = typ == t
        if s.any() and nc[s].sum() > 0:
            print(f"    {t}: {dE[s].sum()/E['full'][s].sum()*100:.1f}% extra energy, {dE[s].sum()/nc[s].sum():.2f} J per check ({int(nc[s].sum())} checks)")
    res[m] = dict(extra_energy_pct=pct, extra_energy_pct_ci=pct_ci.tolist(), J_per_check=perc, J_per_check_ci=perc_ci.tolist(),
                  extra_time_pct=tpct, extra_time_pct_ci=tpct_ci.tolist(), s_per_check=tpc, checks=int(nc.sum()), **res_extra)
den = (E["check_single"] - E["full"]).sum()
ratio = (E["check_reencode"] - E["full"]).sum() / den if den > 0 else float("nan")
print(f"\nserver-style check energy / single-pass check energy: x{ratio:.1f}" + ("" if den > 0 else "  (single-pass extra energy is not above zero; use the timer-based estimate)"))
res["reencode_over_single"] = ratio
# time per check against text length
D = {}
for m in MODES[1:]:
    tok = np.array([x for i in ids for x in R[i][m]["check_tokens"]], float); sec = np.array([x for i in ids for x in R[i][m]["check_seconds"]], float)
    D[m] = (tok, sec)
    if len(tok) > 2:
        b, a0 = np.polyfit(tok, sec, 1)
        print(f"{m}: time per check = {a0*1000:.0f} ms + {b*1000:.2f} ms per token of text (straight-line fit over {len(tok)} checks, {int(tok.min())}-{int(tok.max())} tokens); median {np.median(sec)*1000:.0f} ms")
        res[m]["fit_ms_intercept"] = a0 * 1000; res[m]["fit_ms_per_token"] = b * 1000; res[m]["check_tokens_range"] = [int(tok.min()), int(tok.max())]
json.dump(res, open(a.out + "c2_numbers.json", "w"), indent=1)
# figure
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#b8b7b2", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})
for ax in (a1, a2): ax.spines[["top", "right"]].set_visible(False); ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, lw=0.8)
for k, (m, col, lab) in enumerate((("check_single", BLUE, "Single-pass\n(our design)"), ("check_reencode", ORANGE, "Server-style\n(text re-read)"))):
    v = res[m]["extra_energy_pct"]; lo, hi = res[m]["extra_energy_pct_ci"]
    a1.bar(k, v, 0.6, color=col); a1.errorbar(k, v, yerr=[[v - lo], [hi - v]], color=INK, capsize=3, lw=1)
    a1.text(k, hi + max(res["check_reencode"]["extra_energy_pct"], 5) * 0.04, f"{v:.1f}%", ha="center", fontweight="bold")
a1.set_xticks([0, 1]); a1.set_xticklabels(["Single-pass\n(our design)", "Server-style\n(text re-read)"], fontsize=8)
a1.set_ylabel("Extra energy from the checks\n(% of energy without checks)"); a1.set_ylim(0, res["check_reencode"]["extra_energy_pct_ci"][1] * 1.2)
a1.set_title("What the checks cost in energy", loc="left", fontsize=9.5, fontweight="bold")
for m, col, lab in (("check_reencode", ORANGE, "Server-style"), ("check_single", BLUE, "Single-pass")):
    tok, sec = D[m]
    a2.scatter(tok, sec, s=5, color=col, alpha=0.25, lw=0)
    edges = np.linspace(tok.min(), tok.max(), 9); mid, med = [], []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        s = (tok >= lo_) & (tok <= hi_)
        if s.sum() >= 3: mid.append(tok[s].mean()); med.append(sec[s].mean())
    a2.plot(mid, med, color=col, lw=2, label=lab)
a2.set_yscale("log"); a2.set_xlabel("Length of the text when the check runs (tokens)"); a2.set_ylabel("Time for one check (s, log scale)")
a2.set_title("Time per check against text length", loc="left", fontsize=9.5, fontweight="bold"); a2.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.32), title="Average per length bin", title_fontsize=8)
fig.tight_layout(); fig.savefig(a.out + "fig_c2_check_cost.pdf"); fig.savefig(a.out + "fig_c2_check_cost.png", dpi=200)
print("saved", a.out + "fig_c2_check_cost.pdf")

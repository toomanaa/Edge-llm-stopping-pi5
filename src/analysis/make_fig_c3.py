#!/usr/bin/env python3
"""Offline test: price-based stopping vs fixed threshold vs token caps, from stored 4-bit hidden states.
Stop at a check if  predicted_remaining_tokens > lam * (1 - p_cal)   (lam in tokens per lost correct answer)."""
import json, sys, numpy as np
from collections import defaultdict
sys.path.insert(0, "src/common")
from correctness import is_correct
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

G = {j["id"]: j for j in map(json.loads, open("data/gen_slim.jsonl"))}
idx_all = [json.loads(l) for l in open("data/states_index.jsonl")]
X = np.fromfile("states.f32", dtype=np.float32).reshape(-1, 2048); assert len(X) == len(idx_all)
idx = idx_all
held = set()
for f in ("data/qa_v2_dev40.jsonl", "data/qa_v2_heldout300.jsonl"): held |= {json.loads(l)["id"] for l in open(f)}
ok_full = {i: is_correct(G[i]["text"], G[i]["answers"], G[i].get("match_mode", "any")) for i in G}
rows = []
for r in idx:
    g = G[r["id"]]; s = is_correct(g["text"][: r["off"]], g["answers"], g.get("match_mode", "any"))
    rows.append((r["id"], r["boundary"], s, r["n_tok_prefix"], r["n_tok_total"]))
ids = np.array([r[0] for r in rows]); safe = np.array([r[2] for r in rows], int)
ntp = np.array([r[3] for r in rows], float); ntt = np.array([r[4] for r in rows], float)
usable = np.array([ok_full[i] for i in ids]); is_held = np.array([i in held for i in ids])
train_pool = usable & ~is_held
tr_ids = sorted(set(ids[train_pool]))
# same split/seed recipe as train_probe_v3.py (seed 1 chosen there); retrained here so results can differ slightly
best = None
for seed in range(5):
    rng = np.random.default_rng(seed); order = list(tr_ids); rng.shuffle(order)
    val_ids = set(order[: int(0.15 * len(order))]); isval = np.array([i in val_ids for i in ids])
    tr = train_pool & ~isval; va = train_pool & isval
    sc = StandardScaler().fit(X[tr])
    clf = MLPClassifier((64,), max_iter=500, random_state=seed, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15).fit(sc.transform(X[tr]), safe[tr])
    vap = average_precision_score(safe[va], clf.predict_proba(sc.transform(X[va]))[:, 1])
    if best is None or vap > best[0]: best = (vap, seed, sc, clf, val_ids, isval, tr, va)
vap, seed, sc, clf, val_ids, isval, tr, va = best
print(f"probe: seed {seed}, validation AP {vap:.3f}; validation answers {len(val_ids)}")
p_raw = clf.predict_proba(sc.transform(X))[:, 1]
# remaining-length head: ridge on log(1+remaining tokens), trained on ALL non-held, non-validation answers (correct or not)
rem = np.maximum(ntt - ntp, 0); fit = ~is_held & ~isval
sc2 = StandardScaler().fit(X[fit]); rg = Ridge(alpha=3000.0).fit(sc2.transform(X[fit]), np.log1p(rem[fit]))
rem_hat = np.expm1(rg.predict(sc2.transform(X))).clip(0, 800)
tm = is_held
print(f"remaining-length head: held-out correlation (log) {np.corrcoef(np.log1p(rem[tm]), np.log1p(rem_hat[tm]))[0,1]:.2f}, median abs error {np.median(np.abs(rem[tm]-rem_hat[tm])):.0f} tokens (median true {np.median(rem[tm]):.0f})")
# Platt calibration of the probe on the validation rows only
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
cal = LogisticRegression(C=100.0).fit(lg(p_raw[va]).reshape(-1, 1), safe[va])
p_cal = cal.predict_proba(lg(p_raw).reshape(-1, 1))[:, 1]
print(f"calibration (validation rows): raw p>=0.99 has safe rate {100*safe[va][p_raw[va]>=0.99].mean():.1f}% ; held-out {100*safe[tm][p_raw[tm]>=0.99].mean():.1f}%")

by = defaultdict(list)
for k, r in enumerate(rows): by[r[0]].append(k)
for i in by: by[i].sort(key=lambda k: rows[k][1])
def n_total(i): return rows[by[i][0]][4] if by.get(i) else max(1, int(len(G[i]["text"].split()) * 1.3))
held_ids = sorted(i for i in held if i in G); val_list = sorted(i for i in val_ids if i in G)
def outcome(i, k):  # stop at row k (or None = never)
    tot = n_total(i)
    if k is None: return (ok_full[i], tot, tot)
    return (is_correct(G[i]["text"][: idx[k]["off"]], G[i]["answers"], G[i].get("match_mode", "any")), rows[k][3], tot)
def run(rule, id_list):
    out = []
    for i in id_list:
        hit = None; streak = 0
        for j, k in enumerate(by.get(i, [])):
            if j % 2 == 1:      # check at every 2nd boundary
                ok = rule(k)
                streak = streak + 1 if ok else 0
                if ok and streak >= rule.need: hit = k; break
        out.append(outcome(i, hit))
    return np.array(out, float)
def mk(f, need=1, cap=None):
    def rule(k): return f(k) or (cap is not None and ntp[k] >= cap)
    rule.need = need; return rule
def summ(a): return 100 * a[:, 0].mean(), 100 * a[:, 1].sum() / a[:, 2].sum()
def cap_run(T, id_list):
    out = []
    for i in id_list:
        k = next((k for k in by.get(i, []) if rows[k][3] >= T), None); out.append(outcome(i, k))
    return np.array(out, float)


import json as _j
def summ(a): return 100 * a[:, 0].mean(), 100 * a[:, 1].sum() / a[:, 2].sum()
def cap_run(T, id_list):
    out = []
    for i in id_list:
        k = next((k for k in by.get(i, []) if rows[k][3] >= T), None); out.append(outcome(i, k))
    return np.array(out, float)
def fixed_run(t, id_list):
    out = []
    for i in id_list:
        hit = None
        for j, k in enumerate(by.get(i, [])):
            if j % 2 == 1 and p_raw[k] >= t: hit = k; break
        out.append(outcome(i, hit))
    return np.array(out, float)
THRS = [0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]; CAPS = [60, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 500]
capH = {T: cap_run(T, held_ids) for T in CAPS}; thrH = {t: fixed_run(t, held_ids) for t in THRS}
base = 100 * np.mean([ok_full[i] for i in held_ids])
cc = np.array([[summ(capH[T])[1], summ(capH[T])[0]] for T in CAPS]); o = np.argsort(cc[:, 0]); cx, cy = cc[o, 0], cc[o, 1]
rng = np.random.default_rng(3); B = 5000; n = len(held_ids); ix = rng.integers(0, n, (B, n))
def gap(a, ix_):
    tok = 100 * a[ix_, 1].sum(1) / a[ix_, 2].sum(1); acc = 100 * a[ix_, 0].mean(1)
    capacc = np.empty(len(ix_)); 
    for b in range(len(ix_)):
        c2 = np.array([[100 * capH[T][ix_[b], 1].sum() / capH[T][ix_[b], 2].sum(), 100 * capH[T][ix_[b], 0].mean()] for T in CAPS]); oo = np.argsort(c2[:, 0])
        capacc[b] = np.interp(tok[b], c2[oo, 0], c2[oo, 1])
    return acc - capacc
res = {}
for t in THRS:
    a = thrH[t]; g = gap(a, ix); acc, tok = summ(a)
    res[t] = (acc, tok, np.interp(tok, cx, cy), g.mean(), np.percentile(g, 2.5), np.percentile(g, 97.5))
    print(f"threshold {t:<6} accuracy {acc:5.1f}  tokens {tok:5.1f}%  cap curve at same tokens {res[t][2]:5.1f}  gap {res[t][3]:+4.1f} pp (95% CI {res[t][4]:+.1f} to {res[t][5]:+.1f})")
print("caps:", {T: (round(summ(capH[T])[0], 1), round(summ(capH[T])[1], 1)) for T in CAPS}); print("no-stopping accuracy", round(base, 1), "n answers", n)
OUT = "experiments/c3_cap_comparison/results/"; import os; os.makedirs(OUT, exist_ok=True)
_j.dump({"thresholds": {str(t): dict(acc=r[0], tokens=r[1], cap_curve_acc=r[2], gap=r[3], lo=r[4], hi=r[5]) for t, r in res.items()},
         "caps": {str(T): dict(acc=summ(capH[T])[0], tokens=summ(capH[T])[1]) for T in CAPS}, "no_stop_acc": base, "n": n}, open(OUT + "c3_offline_curves.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, a = plt.subplots(figsize=(4.6, 3.3))
a.plot(cx, cy, "-o", color="#eb6834", ms=3.5, lw=1.4, label="Token limit (60 to 500 tokens)", zorder=3)
tx = [res[t][1] for t in THRS]; ty = [res[t][0] for t in THRS]
a.plot(tx, ty, "-s", color="#2a78d6", ms=3.5, lw=1.4, label="Learned stopper (threshold 0.7 to 0.999)", zorder=3)
a.axhline(base, color="#8a8984", lw=0.9, ls="--"); a.text(3, base + 0.8, "no stopping", fontsize=8, color="#52514e")
a.scatter([res[0.99][1]], [res[0.99][0]], s=60, facecolor="none", edgecolor="#0b0b0b", lw=1.1, zorder=4); a.annotate("threshold 0.99", (res[0.99][1], res[0.99][0]), (res[0.99][1] + 8, res[0.99][0] - 9), fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.7, color="#52514e"))
a.set_xlabel("Tokens generated (% of no stopping)"); a.set_ylabel("Accuracy (%)"); a.set_xlim(0, 105); a.set_ylim(40, 82); a.grid(color="#e3e2dd", lw=0.5, zorder=0)
a.legend(frameon=False, fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(OUT + "fig_c3_curves.pdf"); fig.savefig(OUT + "fig_c3_curves.png", dpi=160)

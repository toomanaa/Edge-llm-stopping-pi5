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

LAMS = [380, 550, 800, 1200, 2000, 3000, 4500, 7000, 12000]
THRS = [0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]
CAPS = [75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400]
pol = {}
for t in THRS: pol[f"fixed {t}"] = (lambda t: mk(lambda k: p_raw[k] >= t))(t)
for L in LAMS:
    pol[f"price lam={L}"] = (lambda L: mk(lambda k: rem_hat[k] > L * (1 - p_cal[k])))(L)
    pol[f"price+2 lam={L}"] = (lambda L: mk(lambda k: rem_hat[k] > L * (1 - p_cal[k]), need=2))(L)
    pol[f"price+cap250 lam={L}"] = (lambda L: mk(lambda k: rem_hat[k] > L * (1 - p_cal[k]), cap=250))(L)
for t in (0.99, 0.995): pol[f"fixed+2 {t}"] = (lambda t: mk(lambda k: p_raw[k] >= t, need=2))(t)
V = {n: run(r, val_list) for n, r in pol.items()}; H = {n: run(r, held_ids) for n, r in pol.items()}
capV = {T: cap_run(T, val_list) for T in CAPS}; capH = {T: cap_run(T, held_ids) for T in CAPS}
base_v = 100 * np.mean([ok_full[i] for i in val_list]); base_h = 100 * np.mean([ok_full[i] for i in held_ids])
print(f"\nvalidation answers {len(val_list)} (no-stop acc {base_v:.1f}%), held-out answers {len(held_ids)} (no-stop acc {base_h:.1f}%)")
# choose on validation: fewest tokens with acc loss <= 8 pp ; per family
fam = defaultdict(list)
for n in pol: fam[n.split(" ")[0] if not n.startswith("price") else n.split(" ")[0]].append(n)
print("\nchosen on validation (fewest tokens with accuracy loss <= 8 pp), reported on held-out:")
print(f"{'policy':<26}{'val acc':>8}{'val tok%':>9}{'HELD acc':>10}{'HELD tok%':>11}")
chosen = {}
for f, names in fam.items():
    ok = [n for n in names if base_v - summ(V[n])[0] <= 8] or [max(names, key=lambda n: summ(V[n])[0])]
    n = min(ok, key=lambda n: summ(V[n])[1]); chosen[f] = n
    print(f"{n:<26}{summ(V[n])[0]:8.1f}{summ(V[n])[1]:9.1f}{summ(H[n])[0]:10.1f}{summ(H[n])[1]:11.1f}")
okc = [T for T in CAPS if base_v - summ(capV[T])[0] <= 8] or CAPS
Tc = min(okc, key=lambda T: summ(capV[T])[1]); print(f"{'cap '+str(Tc):<26}{summ(capV[Tc])[0]:8.1f}{summ(capV[Tc])[1]:9.1f}{summ(capH[Tc])[0]:10.1f}{summ(capH[Tc])[1]:11.1f}")
print("\nHELD-OUT curves (information; sorted by tokens)")
for f in ("fixed", "price", "price+2", "price+cap250"):
    print(f"-- {f}")
    for n in sorted([n for n in pol if (n.split(" ")[0] == f)], key=lambda n: summ(H[n])[1]):
        print(f"   {n:<26} acc {summ(H[n])[0]:5.1f}  tokens {summ(H[n])[1]:5.1f}%")
print("-- caps")
for T in CAPS: print(f"   cap {T:<22} acc {summ(capH[T])[0]:5.1f}  tokens {summ(capH[T])[1]:5.1f}%")
# paired bootstrap: policy vs the cap curve interpolated at the same token share
ct = np.array([summ(capH[T])[1] for T in CAPS]); rng = np.random.default_rng(0)
def vs_cap(name, B=2000):
    a = H[name]; n = len(a); res = []
    for _ in range(B):
        ix = rng.integers(0, n, n); tok = 100 * a[ix, 1].sum() / a[ix, 2].sum(); acc = 100 * a[ix, 0].mean()
        cc = np.array([[100 * capH[T][ix, 0].mean(), 100 * capH[T][ix, 1].sum() / capH[T][ix, 2].sum()] for T in CAPS])
        o = np.argsort(cc[:, 1]); res.append(acc - np.interp(tok, cc[o, 1], cc[o, 0]))
    return np.mean(res), np.percentile(res, 2.5), np.percentile(res, 97.5)
print("\naccuracy difference vs the cap curve at the same token share (positive = better than the caps), 95% bootstrap CI:")
for f, n in chosen.items():
    m, lo, hi = vs_cap(n); print(f"   {n:<26} {m:+5.1f} pp  ({lo:+.1f}..{hi:+.1f})")

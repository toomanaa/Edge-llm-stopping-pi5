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


# hopeless-answer head: P(answer is ever correct at full length | state at this boundary), trained on non-held, non-validation answers
okf = np.array([ok_full[i] for i in ids], int); fit = ~is_held & ~isval
bnd = np.array([r[1] for r in rows])
sc3 = StandardScaler().fit(X[fit]); hc = LogisticRegression(C=0.01, max_iter=300).fit(sc3.transform(X[fit]), okf[fit])
p_ever = hc.predict_proba(sc3.transform(X))[:, 1]
print("ever-correct head AUC held-out:", end=" ")
from sklearn.metrics import roc_auc_score
for b in (1, 3, 5):
    m = tm & (bnd == b); print(f"boundary {b}: {roc_auc_score(okf[m], p_ever[m]):.2f}", end="  ")
print()
FIX = 0.99
def mk2(tau, minb, fixed=FIX, cap=None):
    def rule(k): return p_raw[k] >= fixed
    def abort(k): return bnd[k] >= minb and p_ever[k] < tau
    return rule, abort
def run2(tau, minb, id_list, fixed=FIX):
    rule, abort = mk2(tau, minb, fixed); out = []
    for i in id_list:
        hit = None
        for j, k in enumerate(by.get(i, [])):
            if j % 2 == 1 and (rule(k) or abort(k)): hit = k; break
        if hit is None: out.append(outcome(i, None)); continue
        if abort(hit) and not rule(hit):   # aborted: answer is whatever we have (counted by prefix correctness)
            out.append(outcome(i, hit))
        else: out.append(outcome(i, hit))
    return np.array(out, float)
base_v = 100 * np.mean([ok_full[i] for i in val_list]); base_h = 100 * np.mean([ok_full[i] for i in held_ids])
CAPS = [75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400]
capH = {T: cap_run(T, held_ids) for T in CAPS}; capV = {T: cap_run(T, val_list) for T in CAPS}
def summ(a): return 100 * a[:, 0].mean(), 100 * a[:, 1].sum() / a[:, 2].sum()
print(f"\nvalidation {len(val_list)} answers (no-stop {base_v:.1f}%), held-out {len(held_ids)} (no-stop {base_h:.1f}%)")
print("fixed 0.99 alone  held-out acc %.1f tokens %.1f" % summ(run2(0.0, 99, held_ids)))
res = []
for minb in (3, 5, 7):
    for tau in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        v = summ(run2(tau, minb, val_list)); h = summ(run2(tau, minb, held_ids)); res.append((minb, tau, v, h))
        print(f"abort if P(ever correct)<{tau:<4} after boundary>={minb}:  val acc {v[0]:5.1f} tok {v[1]:5.1f} | HELD acc {h[0]:5.1f} tok {h[1]:5.1f}")
ok = [r for r in res if base_v - r[2][0] <= 8]; ch = min(ok, key=lambda r: r[2][1]) if ok else None
if ch: print("chosen on validation:", ch[:2], " HELD acc %.1f tok %.1f" % ch[3])
ct = np.array([summ(capH[T])[1] for T in CAPS]); ca = np.array([summ(capH[T])[0] for T in CAPS])
for r in res:
    pass
if ch:
    a = run2(ch[1], ch[0], held_ids); rng = np.random.default_rng(0); n = len(a); d = []
    for _ in range(2000):
        ix = rng.integers(0, n, n); tok = 100 * a[ix, 1].sum() / a[ix, 2].sum(); acc = 100 * a[ix, 0].mean()
        cc = np.array([[100 * capH[T][ix, 0].mean(), 100 * capH[T][ix, 1].sum() / capH[T][ix, 2].sum()] for T in CAPS]); o = np.argsort(cc[:, 1])
        d.append(acc - np.interp(tok, cc[o, 1], cc[o, 0]))
    print(f"vs cap curve at same tokens: {np.mean(d):+.1f} pp ({np.percentile(d,2.5):+.1f}..{np.percentile(d,97.5):+.1f})")
print("-- caps"); [print(f"   cap {T}: acc {summ(capH[T])[0]:.1f} tok {summ(capH[T])[1]:.1f}") for T in CAPS]

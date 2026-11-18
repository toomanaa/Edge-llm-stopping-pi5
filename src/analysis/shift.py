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


import itertools
def fixed_rule(t): return mk(lambda k: p_raw[k] >= t)
def summ(a): return 100 * a[:, 0].mean(), 100 * a[:, 1].sum() / a[:, 2].sum()
def cap_run(T, id_list):
    out = []
    for i in id_list:
        k = next((k for k in by.get(i, []) if rows[k][3] >= T), None); out.append(outcome(i, k))
    return np.array(out, float)
THRS = [0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]; CAPS = [75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 500, 600, 800]
# held-out answers that have sentence ends (the ones a stopper can act on)
pool = [i for i in held_ids if by.get(i)]
tot = {i: n_total(i) for i in pool}; order = sorted(pool, key=lambda i: tot[i]); h = len(order) // 2
short, long_ = order[:h], order[h:]
print(f"pool {len(pool)} answers; 'short' half: full length {min(tot[i] for i in short)}-{max(tot[i] for i in short)} tokens (median {np.median([tot[i] for i in short]):.0f}); 'long' half: {min(tot[i] for i in long_)}-{max(tot[i] for i in long_)} (median {np.median([tot[i] for i in long_]):.0f})")
def table(ids):
    return ({t: run(fixed_rule(t), ids) for t in THRS}, {T: cap_run(T, ids) for T in CAPS})
def study(tune, test, tag, budget):
    tf, tc = table(tune); ef, ec = table(test)
    base_t = 100 * np.mean([ok_full[i] for i in tune]); base_e = 100 * np.mean([ok_full[i] for i in test])
    # tune: the setting with the FEWEST tokens whose accuracy loss on the tuning half is <= budget points
    okf = [t for t in THRS if base_t - summ(tf[t])[0] <= budget]; okc = [T for T in CAPS if base_t - summ(tc[T])[0] <= budget]
    t = min(okf, key=lambda t: summ(tf[t])[1]) if okf else THRS[-1]; T = min(okc, key=lambda T: summ(tc[T])[1]) if okc else CAPS[-1]
    af, ac = summ(tf[t]), summ(tc[T]); bf, bc = summ(ef[t]), summ(ec[T])
    print(f"{tag} (budget {budget} pp): tuned threshold {t} / cap {T}")
    print(f"   on tuning half : threshold loss {base_t-af[0]:4.1f} pp tokens {af[1]:5.1f}% | cap loss {base_t-ac[0]:4.1f} pp tokens {ac[1]:5.1f}%")
    print(f"   on OTHER half  : threshold loss {base_e-bf[0]:4.1f} pp tokens {bf[1]:5.1f}% | cap loss {base_e-bc[0]:4.1f} pp tokens {bc[1]:5.1f}%")
for b in (3, 5):
    study(short, long_, "tune on SHORT answers, test on LONG", b); study(long_, short, "tune on LONG answers, test on SHORT", b)
# how spread out is the length at which an answer first becomes correct?
est = {}
for i in pool:
    sk = [rows[k][3] for k in by[i] if rows[k][2]]
    if sk and ok_full[i]: est[i] = min(sk)
es = list(est.values()); print('answers with a first-correct point:', len(es))
import numpy as np
cor = np.corrcoef([est[i] for i in est], [tot[i] for i in est])[0,1]; print('correlation between first-correct tokens and full length: %.2f' % cor)
print('share of answers already correct within 100 / 150 / 200 / 300 tokens:', [round(100*np.mean([e<=T for e in es])) for T in (100,150,200,300)])
print("\ntokens needed before the answer is first correct (answers correct at full length):", {q: int(np.percentile(es, q)) for q in (10, 25, 50, 75, 90)})

# ceiling: oracle semantic stop (first correct sentence end) vs the cheapest cap with the same accuracy as no stopping
tot_all = sum(tot[i] for i in pool)
orc = sum(est.get(i, tot[i]) for i in pool) / tot_all * 100
base = np.mean([ok_full[i] for i in pool])
print(f"\nceiling on the {len(pool)} checkable held-out answers (no-stop accuracy {100*base:.1f}%):")
print(f"  oracle semantic stop, zero accuracy loss: {orc:.1f}% of tokens")
for T in (200, 250, 300, 350, 400, 500, 600, 800):
    a = cap_run(T, pool); print(f"  cap {T}: accuracy {100*a[:,0].mean():.1f}% tokens {100*a[:,1].sum()/a[:,2].sum():.1f}%")
a = run(fixed_rule(0.99), pool); print(f"  learned controller 0.99: accuracy {100*a[:,0].mean():.1f}% tokens {100*a[:,1].sum()/a[:,2].sum():.1f}%")

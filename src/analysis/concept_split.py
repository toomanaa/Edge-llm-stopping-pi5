#!/usr/bin/env python3
"""Concept-level split test (answers a reviewer concern: the same concept asked in different wordings in train and test).
Retrains the probe using ONLY answers whose concept (the set of accepted answers / required keywords) never appears in the
held-out sets, then replays the stored held-out answers. Compare with the normal probe (trained on a random split of questions).
Run from the repo root:  python3 src/analysis/concept_split.py   (needs states.f32 in the repo root; not in git)"""
import json, sys, numpy as np
from collections import defaultdict
sys.path.insert(0, "src/common")
from correctness import is_correct
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

G = {j["id"]: j for j in map(json.loads, open("data/gen_slim.jsonl"))}
idx_all = [json.loads(l) for l in open("data/states_index.jsonl")]
X = np.fromfile("states.f32", dtype=np.float32).reshape(-1, 2048); assert len(X) == len(idx_all)
idx = idx_all
held = set()
for f in ("data/qa_v2_dev40.jsonl", "data/qa_v2_heldout300.jsonl"): held |= {json.loads(l)["id"] for l in open(f)}
concept = lambda i: tuple(sorted(G[i]["answers"]))
ok_full = {i: is_correct(G[i]["text"], G[i]["answers"], G[i].get("match_mode", "any")) for i in G}
rows = []
for r in idx:
    g = G[r["id"]]; s = is_correct(g["text"][: r["off"]], g["answers"], g.get("match_mode", "any"))
    rows.append((r["id"], r["boundary"], s, r["n_tok_prefix"], r["n_tok_total"]))
ids = np.array([r[0] for r in rows]); safe = np.array([r[2] for r in rows], int)
ntp = np.array([r[3] for r in rows], float)
usable = np.array([ok_full[i] for i in ids]); is_held = np.array([i in held for i in ids])
held_concepts = {concept(i) for i in held if i in G}
shares = np.array([(concept(i) in held_concepts) for i in ids])
pool_random = usable & ~is_held                      # what the paper's probe used
pool_concept = pool_random & ~shares                 # concept never seen in held-out
for nm, pl in (("random split (paper)", pool_random), ("concept-disjoint", pool_concept)):
    a = sorted(set(ids[pl])); print(f"{nm}: {len(a)} training-pool answers, {len({concept(i) for i in a})} concepts, {int(pl.sum())} boundaries")

by = defaultdict(list)
for k, r in enumerate(rows): by[r[0]].append(k)
for i in by: by[i].sort(key=lambda k: rows[k][1])
def outcome(i, k):
    tot = rows[by[i][0]][4] if by.get(i) else max(1, int(len(G[i]["text"].split()) * 1.3))
    if k is None: return (ok_full[i], tot, tot)
    return (is_correct(G[i]["text"][: idx[k]["off"]], G[i]["answers"], G[i].get("match_mode", "any")), rows[k][3], tot)
held_ids = sorted(i for i in held if i in G)
def summ(a): return 100 * a[:, 0].mean(), 100 * a[:, 1].sum() / a[:, 2].sum()
def cap_run(T):
    return np.array([outcome(i, next((k for k in by.get(i, []) if rows[k][3] >= T), None)) for i in held_ids], float)
CAPS = [60, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 500]
cc = np.array([summ(cap_run(T)) for T in CAPS]); o = np.argsort(cc[:, 1])
def gap(a): return a[:, 0].mean() * 100 - np.interp(100 * a[:, 1].sum() / a[:, 2].sum(), cc[o, 1], cc[o, 0])
def policy(p, thr):
    out = []
    for i in held_ids:
        hit = next((k for j, k in enumerate(by.get(i, [])) if j % 2 == 1 and p[k] >= thr), None); out.append(outcome(i, hit))
    return np.array(out, float)
def boot(a, n=2000):
    rng = np.random.default_rng(0); g = []
    for _ in range(n):
        s = rng.integers(0, len(a), len(a)); g.append(gap(a[s]))
    return np.percentile(g, [2.5, 97.5])
def train(pool, group):
    best = None
    grp = sorted({group(i) for i in set(ids[pool])})
    for seed in range(5):
        rng = np.random.default_rng(seed); order = list(grp); rng.shuffle(order)
        vg = set(order[: int(0.15 * len(order))]); isval = np.array([group(i) in vg for i in ids])
        tr = pool & ~isval; va = pool & isval
        sc = StandardScaler().fit(X[tr])
        clf = MLPClassifier((64,), max_iter=500, random_state=seed, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15).fit(sc.transform(X[tr]), safe[tr])
        vap = average_precision_score(safe[va], clf.predict_proba(sc.transform(X[va]))[:, 1])
        if best is None or vap > best[0]: best = (vap, seed, sc, clf)
    vap, seed, sc, clf = best
    return vap, seed, clf.predict_proba(sc.transform(X))[:, 1]
tmm = is_held & usable
for nm, pool in (("normal probe (random split of questions)", pool_random), ("concept-disjoint probe", pool_concept)):
    vap, seed, p = train(pool, (lambda i: i) if pool is pool_random else concept)
    print(f"\n== {nm}: seed {seed}, validation AP {vap:.3f}; held-out boundary AP {average_precision_score(safe[tmm], p[tmm]):.3f}")
    for thr in (0.9, 0.95, 0.99, 0.995, 0.999):
        a = policy(p, thr); ac, tk = summ(a); lo, hi = boot(a)
        print(f"   thr {thr:<6} accuracy {ac:5.1f}  tokens {tk:5.1f}%   vs token limits {gap(a):+.1f} pp (95% CI {lo:+.1f}..{hi:+.1f})")
print("\nno stopping accuracy %.1f on %d held-out answers" % (100 * np.mean([ok_full[i] for i in held_ids]), len(held_ids)))

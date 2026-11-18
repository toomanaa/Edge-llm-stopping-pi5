#!/usr/bin/env python3
"""Retrain the stopping probe on hidden states taken from the deployed 4-bit model, using every
sentence boundary of every training answer, and compare it with the current probe on held-out questions.

Needs: states.f32 + index.jsonl (from extract_states), the generation file with answers, the shared
correctness.py, scikit-learn, numpy.

  python3 train_probe_v3.py --states states.f32 --index index.jsonl --gen qa_v2_gen_merged.jsonl \
      --heldout qa_v2_dev40.jsonl qa_v2_heldout300.jsonl --old-probe probe_weights.npz \
      --correctness scripts/controller --out probe_weights_v3.npz
"""
import argparse, json, sys
from collections import defaultdict
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--states", required=True); ap.add_argument("--index", required=True)
ap.add_argument("--gen", required=True); ap.add_argument("--heldout", nargs="+", required=True)
ap.add_argument("--old-probe", required=True); ap.add_argument("--correctness", default="scripts/controller")
ap.add_argument("--out", default="probe_weights_v3.npz"); ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--boot", type=int, default=5000)
ap.add_argument("--max-loss", type=float, default=0.08, help="largest accuracy loss allowed on the validation answers when choosing threshold and minimum length")
a = ap.parse_args()
sys.path.insert(0, a.correctness)
from correctness import is_correct

G = {j["id"]: j for j in map(json.loads, open(a.gen))}
idx = [json.loads(l) for l in open(a.index)]
D = 2048
X = np.fromfile(a.states, dtype=np.float32).reshape(-1, D)
assert len(X) == len(idx), (len(X), len(idx))
held = set()
for f in a.heldout:
    held |= {json.loads(l)["id"] for l in open(f)}
ok_full = {i: is_correct(G[i]["text"], G[i]["answers"], G[i].get("match_mode", "any")) for i in G}
rows = []
for r in idx:
    g = G[r["id"]]
    safe = is_correct(g["text"][: r["off"]], g["answers"], g.get("match_mode", "any"))
    rows.append((r["id"], r["boundary"], safe, r["n_tok_prefix"], r["n_tok_total"]))
ids = np.array([r[0] for r in rows]); safe = np.array([r[2] for r in rows], dtype=int)
usable = np.array([ok_full[i] for i in ids])         # same convention as before: responses that end correct
is_held = np.array([i in held for i in ids])
train_pool = usable & ~is_held; test_mask = usable & is_held
print(f"boundaries: {len(rows)}; usable (answers correct at full length): {usable.sum()}; "
      f"training pool {train_pool.sum()} from {len(set(ids[train_pool]))} answers; held-out {test_mask.sum()} from {len(set(ids[test_mask]))} answers")
print(f"unsafe share: training pool {100*(1-safe[train_pool].mean()):.1f}%, held-out {100*(1-safe[test_mask].mean()):.1f}%")

class OldProbe:
    def __init__(s, p):
        d = np.load(p); s.m, s.s, s.W0, s.b0, s.W1, s.b1 = d["scaler_mean"], d["scaler_scale"], d["W0"].astype(float), d["b0"].astype(float), d["W1"].astype(float), d["b1"].astype(float)
    def proba(s, X):
        h = np.maximum(0, ((X - s.m) / s.s) @ s.W0 + s.b0); return (1 / (1 + np.exp(-(h @ s.W1 + s.b1))))[:, 0]
class SkProbe:
    def __init__(s, sc, clf): s.sc, s.clf = sc, clf
    def proba(s, X): return s.clf.predict_proba(s.sc.transform(X))[:, 1]

# deterministic grouped split of the training answers (sorted before shuffling: no hash-order dependence)
tr_ids = sorted(set(ids[train_pool]))
best = None
for seed in range(a.seeds):
    rng = np.random.default_rng(seed); order = list(tr_ids); rng.shuffle(order)
    n_val = int(0.15 * len(order)); val_ids = set(order[:n_val])
    isval = np.array([i in val_ids for i in ids])
    tr = train_pool & ~isval; va = train_pool & isval
    sc = StandardScaler().fit(X[tr])
    clf = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", max_iter=500, random_state=seed,
                        early_stopping=True, validation_fraction=0.15, n_iter_no_change=15).fit(sc.transform(X[tr]), safe[tr])
    p = SkProbe(sc, clf)
    vap = average_precision_score(safe[va], p.proba(X[va])); tap = average_precision_score(safe[test_mask], p.proba(X[test_mask]))
    print(f"seed {seed}: train {tr.sum()} val {va.sum()}  val AP {vap:.3f}  held-out AP {tap:.3f}")
    if best is None or vap > best[0]: best = (vap, seed, p, sc, clf, val_ids)
vap, seed, new, sc, clf, val_ids = best
old = OldProbe(a.old_probe)
print(f"\nchosen seed {seed} (by validation AP {vap:.3f}; held-out answers were not used to choose)")
np.savez(a.out, scaler_mean=sc.mean_, scaler_scale=sc.scale_, activation=np.array("relu"), out_activation=np.array("logistic"),
         n_layers=np.array(2), W0=clf.coefs_[0].astype(np.float32), b0=clf.intercepts_[0].astype(np.float32),
         W1=clf.coefs_[1].astype(np.float32), b1=clf.intercepts_[1].astype(np.float32))
print("wrote", a.out)

Xt, yt = X[test_mask], safe[test_mask]
po, pn = old.proba(Xt), new.proba(Xt)
print(f"\nheld-out boundary-level AP: old probe {average_precision_score(yt, po):.3f}   new probe {average_precision_score(yt, pn):.3f}")
for t in (0.99, 0.97, 0.95, 0.9):
    for name, p in (("old", po), ("new", pn)):
        m = p >= t
        print(f"  p >= {t}: {name}  fires on {m.sum():5d} boundaries, precision {100*yt[m].mean() if m.any() else float('nan'):5.1f}%, recall {100*(m & (yt == 1)).sum()/(yt == 1).sum():5.1f}%")

# ---- replay of the controller from the stored hidden states (checks at every 2nd boundary, as in the live runs) ----
# energy proxy: generated tokens (the live pilot's measured energy matched this proxy within ~1.3%)
by = defaultdict(list)
for k, r in enumerate(rows): by[r[0]].append(k)
for i in by: by[i].sort(key=lambda k: rows[k][1])
def n_total(i): return rows[by[i][0]][4] if by.get(i) else max(1, int(len(G[i]["text"].split()) * 1.3))
earliest_safe_tok = {}
for i in by:
    sk = [rows[k][3] for k in by[i] if rows[k][2]]
    earliest_safe_tok[i] = min(sk) if sk else None
def replay(prob_by_row, thr, floor, id_list):
    out = []
    for i in id_list:
        tot = n_total(i); g = G[i]; hit = None
        for j, k in enumerate(by.get(i, [])):
            if j % 2 == 1 and prob_by_row[k] >= thr and rows[k][3] >= floor: hit = k; break
        if hit is None: out.append((ok_full[i], tot, tot, False, 0.0)); continue
        used = rows[hit][3]; okk = is_correct(g["text"][: idx[hit]["off"]], g["answers"], g.get("match_mode", "any"))
        es = earliest_safe_tok.get(i); wasted = max(0, used - es) if (okk and es is not None) else 0
        out.append((okk, used, tot, not okk, wasted))
    a = np.array(out, float); return a   # columns: correct, used, total, false_stop, wasted
def summarize(a):
    return dict(acc=100 * a[:, 0].mean(), tok=100 * a[:, 1].sum() / a[:, 2].sum(), false=int(a[:, 3].sum()), stops=int((a[:, 1] < a[:, 2]).sum()),
                waste=100 * a[:, 4].sum() / a[:, 2].sum())
def cap_replay(T, id_list):
    out = []
    for i in id_list:
        tot = n_total(i); g = G[i]; hit = next((k for k in by.get(i, []) if rows[k][3] >= T), None)
        if hit is None: out.append((ok_full[i], tot, tot, False, 0.0)); continue
        okk = is_correct(g["text"][: idx[hit]["off"]], g["answers"], g.get("match_mode", "any")); out.append((okk, rows[hit][3], tot, not okk, 0.0))
    return np.array(out, float)

p_old = np.zeros(len(rows)); p_new = np.zeros(len(rows))
p_old[:] = old.proba(X); p_new[:] = new.proba(X)
held_ids = sorted(i for i in held if i in G)
val_list = sorted(i for i in val_ids if i in G)
THRS = (0.9, 0.95, 0.99, 0.995); FLOORS = (0, 40, 80)
# 1) choose threshold and minimum length on the VALIDATION answers only (held-out answers play no part)
print(f"\nchoosing threshold and minimum length on {len(val_list)} validation answers: fewest tokens with accuracy loss <= {a.max_loss*100:.0f} pp")
base_val = np.mean([ok_full[i] for i in val_list]); cfgs = []
for t in THRS:
    for f in FLOORS:
        r = summarize(replay(p_new, t, f, val_list)); cfgs.append((t, f, r))
        print(f"  thr {t:<6} min {f:>3} tokens: accuracy {r['acc']:5.1f}% (no stopping {100*base_val:.1f}%), tokens used {r['tok']:5.1f}%, false stops {r['false']}/{r['stops']}")
okc = [c for c in cfgs if base_val * 100 - c[2]["acc"] <= a.max_loss * 100]
chosen = min(okc, key=lambda c: c[2]["tok"]) if okc else max(cfgs, key=lambda c: c[2]["acc"])
print(f"chosen on validation: threshold {chosen[0]}, minimum {chosen[1]} tokens")
# 2) report on the held-out answers: the chosen configuration, then the whole grid for information only
print(f"\nHELD-OUT answers: {len(held_ids)} (of which {sum(1 for i in held_ids if by.get(i))} have extracted sentence ends); no stopping accuracy {100*np.mean([ok_full[i] for i in held_ids]):.1f}%")
print(f"{'policy':<34}{'acc %':>7}{'tokens %':>10}{'stops':>7}{'false stops':>13}{'wasted tok %':>14}")
def line(name, a_):
    r = summarize(a_); print(f"{name:<34}{r['acc']:7.1f}{r['tok']:10.1f}{r['stops']:7d}{r['false']:13d}{r['waste']:14.1f}")
sel_new = replay(p_new, chosen[0], chosen[1], held_ids)
line(f"NEW probe (chosen) {chosen[0]}/{chosen[1]}", sel_new)
line(f"OLD probe, same setting", replay(p_old, chosen[0], chosen[1], held_ids))
line("OLD probe 0.99 (as in the live pilot)", replay(p_old, 0.99, 0, held_ids))
print("--- information only, not used for any choice ---")
for t in THRS:
    for f in FLOORS: line(f"new {t} / min {f}", replay(p_new, t, f, held_ids))
caps = {}
for T in (100, 150, 200, 250, 300): caps[T] = cap_replay(T, held_ids); line(f"cap {T} tokens", caps[T])
rng = np.random.default_rng(0)
def paired(name, a1, a0):
    n = len(a1); ix = rng.integers(0, n, size=(a.boot, n)); d = (a1[:, 0] - a0[:, 0])[ix].mean(1); e = a1[:, 1][ix].sum(1) / a0[:, 1][ix].sum(1)
    print(f"{name}: accuracy {100*(a1[:,0]-a0[:,0]).mean():+.1f} pp (95% CI {100*np.percentile(d,2.5):+.1f}..{100*np.percentile(d,97.5):+.1f}); "
          f"tokens ratio {a1[:,1].sum()/a0[:,1].sum():.3f} (CI {np.percentile(e,2.5):.3f}..{np.percentile(e,97.5):.3f})")
print()
paired("NEW chosen vs OLD 0.99", sel_new, replay(p_old, 0.99, 0, held_ids))
paired("NEW chosen vs cap 200", sel_new, caps[200])
paired("OLD 0.99 vs cap 200", replay(p_old, 0.99, 0, held_ids), caps[200])

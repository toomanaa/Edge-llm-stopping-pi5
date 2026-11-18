#!/usr/bin/env python3
"""Signal validity (paper Section 6.3): learned probe vs entropy baseline, on the SAME test boundaries.
Hidden states are the 4-bit model's own states (states.f32, extracted at sentence ends with the controller's extractor).
Answers are split into train / validation / test by topic (answers sharing the same required keywords always stay together),
70/15/15, 5 seeds. Only answers outside the live-run sets (dev, held-out) are used. Run from the repo root:
  python3 src/analysis/probe_vs_entropy.py        (needs states.f32 in the repo root; not in git)"""
import json, sys, numpy as np
from collections import defaultdict
sys.path.insert(0, "src/common")
from correctness import is_correct
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

G = {j["id"]: j for j in map(json.loads, open("data/gen_slim.jsonl"))}
idx = [json.loads(l) for l in open("data/states_index.jsonl")]
X = np.fromfile("states.f32", dtype=np.float32).reshape(-1, 2048); assert len(X) == len(idx)
H = {}
for l in open("experiments/earlier_experiments/e1_v2_1074/qa_v2_labels.jsonl"):
    r = json.loads(l); H[(r["id"], r["boundary"])] = r["entropy_topk"]
live = set()
for f in ("data/qa_v2_dev40.jsonl", "data/qa_v2_heldout300.jsonl"): live |= {json.loads(l)["id"] for l in open(f)}
ok_full = {i: is_correct(g["text"], g["answers"], g.get("match_mode", "any")) for i, g in G.items()}
rows = []; keepi = []
nb = defaultdict(int)
for r in idx: nb[r["id"]] += 1
for k, r in enumerate(idx):
    i = r["id"]
    if i in live or not ok_full[i] or nb[i] < 2 or (i, r["boundary"]) not in H: continue
    g = G[i]; safe = is_correct(g["text"][: r["off"]], g["answers"], g.get("match_mode", "any"))
    rows.append((i, tuple(sorted(g["answers"])), int(safe), H[(i, r["boundary"])])); keepi.append(k)
X = X[keepi]; ids = np.array([r[0] for r in rows]); topic = [r[1] for r in rows]
y = np.array([r[2] for r in rows]); ent = np.array([r[3] for r in rows])
topics = sorted(set(topic)); tarr = np.array([topics.index(t) for t in topic])
print(f"{len(rows)} boundaries from {len(set(ids))} answers ({len(topics)} topics); {int((1-y).sum())} unsafe ({100*(1-y).mean():.1f}%)")

def entropy_curve(yy, ee):
    """stop when entropy < T: precision and recall of the 'safe' prediction over a sweep of T"""
    Ts = np.unique(np.quantile(ee, np.linspace(0.02, 1.0, 60)))
    P, R = [], []
    for T in Ts:
        s = ee <= T
        if s.sum() == 0: continue
        P.append(yy[s].mean()); R.append(s.sum() / yy.sum() if False else (yy[s].sum() / yy.sum()))
    return np.array(R), np.array(P)

res = []
for seed in range(5):
    rng = np.random.default_rng(seed); order = rng.permutation(len(topics))
    n = len(topics); ntr, nva = int(0.7 * n), int(0.15 * n)
    grp = {t: ("tr" if j < ntr else "va" if j < ntr + nva else "te") for j, t in enumerate(order)}
    sp = np.array([grp[t] for t in tarr])
    tr, va, te = sp == "tr", sp == "va", sp == "te"
    sc = StandardScaler().fit(X[tr])
    clf = MLPClassifier((64,), max_iter=500, random_state=seed, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15).fit(sc.transform(X[tr]), y[tr])
    pv = clf.predict_proba(sc.transform(X[va]))[:, 1]; pt = clf.predict_proba(sc.transform(X[te]))[:, 1]
    # entropy as a score for "safe" (lower entropy = more confident): AP of -entropy on the same test boundaries
    ap_e = average_precision_score(y[te], -ent[te])
    pr, rc, _ = precision_recall_curve(y[te], pt); er, ep = entropy_curve(y[te], ent[te])
    res.append(dict(seed=seed, vap=average_precision_score(y[va], pv), tap=average_precision_score(y[te], pt), eap=ap_e, n=int(te.sum()), neg=int((1 - y[te]).sum()), base=float(y[te].mean()), pr=pr, rc=rc, er=er, ep=ep))
    r_ = res[-1]; print(f"seed {seed}: test boundaries {r_['n']} ({r_['neg']} unsafe, base precision {r_['base']:.2f}); probe AP {r_['tap']:.3f}  entropy AP {r_['eap']:.3f}  (val AP {r_['vap']:.3f})")
t = np.array([r["tap"] for r in res]); e = np.array([r["eap"] for r in res])
print(f"probe AP {t.mean():.3f} +/- {t.std():.3f} (range {t.min():.3f}-{t.max():.3f});  entropy AP on the same test boundaries {e.mean():.3f} +/- {e.std():.3f} (range {e.min():.3f}-{e.max():.3f})")
print("probe beats entropy in", int((t > e).sum()), "of 5 seeds; mean base precision (share of safe boundaries) %.2f" % np.mean([r["base"] for r in res]))
best = max(res, key=lambda r: r["vap"])
# figure
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORG, GRAY, TXT, GRID = "#2a78d6", "#eb6834", "#8a8984", "#2f2e2c", "#e4e3df"
fig, ax = plt.subplots(figsize=(5.2, 4)); ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
for r in res:
    ax.plot(r["rc"], r["pr"], color=BLUE, alpha=0.25, lw=1)
ax.plot(best["rc"], best["pr"], color=BLUE, lw=2.2, label="Probe (best of five seeds)")
ax.plot(best["er"], best["ep"], color=ORG, lw=1.8, ls="--", marker=".", ms=4, label="Entropy baseline (same boundaries)")
ax.set_xlabel("Recall", color=TXT); ax.set_ylabel("Precision", color=TXT); ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.4, 1.02)
ax.legend(loc="lower left", frameon=False, fontsize=9); fig.tight_layout()
fig.savefig("experiments/earlier_experiments/probe/fig_probe_vs_entropy.pdf"); fig.savefig("experiments/earlier_experiments/probe/fig_probe_vs_entropy.png", dpi=200)

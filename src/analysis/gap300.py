import json,sys,re,numpy as np
sys.path.insert(0,"src/common")
from correctness import is_correct
A="experiments/c1_energy_savings/raw/"
Q={j["id"]:j for j in map(json.loads,open("data/qa_v2_heldout300.jsonl"))}
full={r["id"]:r for r in map(json.loads,open(A+"run300_runs.jsonl")) if r["mode"]=="full"}
B=re.compile(r'(?<!\d\.)(?<=[.!?])\s+')
# validate the token estimate (chars fraction x total tokens) against exact counts from the extraction index
idx={}
for l in open("data/states_index.jsonl"):
    r=json.loads(l); idx.setdefault(r["id"],[]).append(r)
errs=[]
for i,f in full.items():
    if i in idx and abs(idx[i][0]["n_tok_total"]-f["n_tokens"])<=3:
        for r in idx[i]:
            errs.append(abs(r["n_tok_prefix"]-f["n_tokens"]*r["off"]/len(f["text"])))
print(f"token estimate check on {len(errs)} boundaries of {len({i for i in full if i in idx})} answers whose live text length matches the stored one: median error {np.median(errs):.1f} tokens, p90 {np.percentile(errs,90):.1f}")
rows=[]
for i,f in full.items():
    q=Q[i]; t=f["text"]; N=f["n_tokens"]; mm=q.get("match_mode","any")
    ok=is_correct(t,q["answers"],mm)
    offs=[m.start() for m in B.finditer(t)]
    pts=[(N*o/len(t), is_correct(t[:o],q["answers"],mm)) for o in offs]   # (est tokens at boundary, prefix correct)
    first=next((p for p,s in pts if s),None)
    rows.append(dict(id=i,N=N,ok=ok,first=first,pts=pts,type=q["type"]))
tot=sum(r["N"] for r in rows); base=100*np.mean([r["ok"] for r in rows])
orc=sum(r["first"] if (r["ok"] and r["first"] is not None) else r["N"] for r in rows)
print(f"\nALL {len(rows)} live-run questions; no-stopping accuracy {base:.1f}%")
print(f"oracle semantic stop (stop at first correct sentence end): accuracy {base:.1f}% (no loss), tokens {100*orc/tot:.1f}%  -> saves {100-100*orc/tot:.1f}%")
def cap(T):
    u=0;c=0
    for r in rows:
        k=next(((p,s) for p,s in r["pts"] if p>=T),None)
        if k is None: u+=r["N"]; c+=r["ok"]
        else: u+=k[0]; c+=k[1]
    return 100*c/len(rows),100*u/tot
print("fixed caps (stop at the first sentence end after T tokens):")
best=None
for T in (100,150,200,210,250,300,350,400,500,600,800):
    a,t=cap(T); print(f"  cap {T:<4} accuracy {a:5.1f}%  tokens {t:5.1f}%"+("   <- first cap with no accuracy loss" if best is None and a>=base-1e-9 else ""))
    if best is None and a>=base-1e-9: best=(T,t)
print(f"\nzero-loss comparison: oracle saves {100-100*orc/tot:.1f}% ; cheapest no-loss cap ({best[0]} tokens) saves {100-best[1]:.1f}%")
ex=[r for r in rows if r["type"]=="explain"]
tx=sum(r["N"] for r in ex); ox=sum(r["first"] if (r["ok"] and r["first"] is not None) else r["N"] for r in ex)
print(f"explain questions only ({len(ex)}): oracle saves {100-100*ox/tx:.1f}% of their tokens")

# does the controller adapt to the answer? compare where it stopped with where the answer first became correct
from scipy.stats import spearmanr
A="experiments/c1_energy_savings/raw/"
ctl={r["id"]:r for r in map(json.loads,open(A+"run300_runs.jsonl")) if r["mode"]=="controller"}
cp={r["id"]:r for r in map(json.loads,open(A+"run300_runs.jsonl")) if r["mode"]=="cap"}
ex=[r for r in rows if r["type"]=="explain" and r["ok"] and r["first"] is not None]
x=[r["first"] for r in ex]; yc=[ctl[r["id"]]["n_tokens"] for r in ex]; yk=[cp[r["id"]]["n_tokens"] for r in ex]
print(f"\nexplain answers that are correct at full length: {len(ex)}")
print(f"  rank correlation, tokens when the answer first becomes correct vs tokens used:  controller {spearmanr(x,yc)[0]:.2f}   cap {spearmanr(x,yk)[0]:.2f}")
print(f"  answers the controller stopped early: {sum(ctl[r['id']]['stopped_early_by_probe'] for r in ex)} of {len(ex)}")
nreq=[len(Q[r['id']]['answers']) for r in ex]
print("  required-concept counts among them:", {k:nreq.count(k) for k in sorted(set(nreq))})
easy=[r for r in ex if r["first"]<=100]; hard=[r for r in ex if r["first"]>200]
for nm,g in (("early-correct (<=100 tokens)",easy),("late-correct (>200 tokens)",hard)):
    print(f"  {nm}: n={len(g)}  median tokens used  controller {np.median([ctl[r['id']]['n_tokens'] for r in g]):.0f}  cap {np.median([cp[r['id']]['n_tokens'] for r in g]):.0f}  oracle {np.median([r['first'] for r in g]):.0f}")

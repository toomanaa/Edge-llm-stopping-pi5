#!/usr/bin/env python3
"""Unfamiliar-prompts test: how often does the controller stop on prompts unlike its training data,
and how much of the answer does it cut?  Only measured quantities are used (no manual labels).
usage: python3 src/analysis/failopen_analysis.py --runs experiments/fail_open/raw/fo_runs.jsonl
         --prompts data/failopen_prompts.jsonl --indomain experiments/c1_energy_savings/raw/run300_runs.jsonl
         --outdir experiments/fail_open/results"""
import argparse, json, csv, collections, os, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--indomain", default=None, help="C1 run file (299 held-out questions) for comparison")
ap.add_argument("--outdir", default=".")
a = ap.parse_args()

def load(p):
    out = []
    for line in open(p, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line: continue
        try: out.append(json.loads(line))
        except Exception: pass
    return out

cat = {r["id"]: r["type"] for r in load(a.prompts)}
runs = collections.defaultdict(dict)
for r in load(a.runs): runs[r["id"]][r["mode"]] = r
pairs = {i: m for i, m in runs.items() if "full" in m and "controller" in m}
lines = []
def P(s=""): print(s); lines.append(s)

n = len(pairs)
P(f"Prompts with both runs: {n}")
stopped = {i: m for i, m in pairs.items() if m["controller"].get("stopped_early_by_probe")}
P(f"Stopped by the probe: {len(stopped)} of {n} ({100*len(stopped)/max(n,1):.1f}%)")
P(f"Full run hit the token limit (800): {sum(1 for m in pairs.values() if m['full']['end_reason']=='max_tokens')}")
P("Controller text is the start of the full text (should be 0 failures): "
  f"{sum(1 for m in stopped.values() if not m['full']['text'].startswith(m['controller']['text']))}")

tf = sum(m["full"]["n_tokens"] for m in pairs.values()); tc = sum(m["controller"]["n_tokens"] for m in pairs.values())
P(f"Tokens: full {tf}, controller {tc}  -> {100*(1-tc/max(tf,1)):.1f}% fewer")

# how much of each stopped answer was cut (share of the words of the full answer)
share, cutw, open_fence = [], [], collections.Counter()
for i, m in stopped.items():
    kept, full = m["controller"]["text"], m["full"]["text"]
    k, f_ = len(kept.split()), len(full.split())
    share.append((f_ - k) / max(f_, 1)); cutw.append(f_ - k)
    if kept.count("```") % 2 == 1: open_fence[cat.get(i, "?")] += 1
if share:
    P(f"Share of words cut in stopped answers: median {100*statistics.median(share):.0f}%, "
      f"more than half cut in {sum(1 for s in share if s > 0.5)} of {len(share)}")
    P(f"Words cut: median {int(statistics.median(cutw))}, min {min(cutw)}, max {max(cutw)}")
    P(f"Stopped inside an unfinished code block: {sum(open_fence.values())} answers "
      f"(code: {open_fence.get('code',0)} of {sum(1 for i in stopped if cat.get(i)=='code')} stopped code answers)")

P(); P(f"{'category':<16}{'n':>4}{'stopped':>9}{'rate %':>8}{'tokens saved %':>16}")
bycat = collections.defaultdict(list)
for i, m in pairs.items(): bycat[cat.get(i, "?")].append(m)
rows = {}
for c, ms in bycat.items():
    s = sum(1 for m in ms if m["controller"].get("stopped_early_by_probe"))
    f_ = sum(m["full"]["n_tokens"] for m in ms); c_ = sum(m["controller"]["n_tokens"] for m in ms)
    rows[c] = dict(n=len(ms), stopped=s, tokens_saved_pct=100*(1-c_/max(f_, 1)))
    P(f"{c:<16}{len(ms):>4}{s:>9}{100*s/len(ms):>8.1f}{rows[c]['tokens_saved_pct']:>16.1f}")

ind = None
if a.indomain and os.path.exists(a.indomain):
    im = collections.defaultdict(dict)
    for r in load(a.indomain): im[r["id"]][r["mode"]] = r
    ims = [m["controller"] for m in im.values() if "controller" in m]
    k = sum(1 for r in ims if r.get("stopped_early_by_probe"))
    ind = dict(n=len(ims), stopped=k, rate=100*k/max(len(ims), 1))
    P(); P(f"For comparison, held-out quiz questions (C1 run): stopped {k} of {len(ims)} ({ind['rate']:.1f}%)")

mx = [max(m["controller"]["p_safe"]) for i, m in pairs.items() if i not in stopped and m["controller"].get("p_safe")]
if mx: P(f"Not-stopped answers: highest probe score reached, median {sorted(mx)[len(mx)//2]:.3f}, max {max(mx):.3f} (threshold 0.99)")

os.makedirs(a.outdir, exist_ok=True)
with open(os.path.join(a.outdir, "failopen_stopped_answers.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["id", "category", "question", "kept_text", "cut_text"])
    for i, m in sorted(stopped.items()):
        kept, full = m["controller"]["text"], m["full"]["text"]
        w.writerow([i, cat.get(i, "?"), m["controller"]["question"], kept, full[len(kept):]])
open(os.path.join(a.outdir, "failopen_output.txt"), "w").write("\n".join(lines) + "\n")
json.dump(dict(n=n, stopped=len(stopped), tokens_full=tf, tokens_controller=tc, by_category=rows, indomain=ind,
               median_share_cut=statistics.median(share) if share else None,
               more_than_half_cut=sum(1 for s in share if s > 0.5), open_code_block=dict(open_fence)),
          open(os.path.join(a.outdir, "failopen_numbers.json"), "w"), indent=1)

# figure: stopped / not stopped by group
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Liberation Serif"
order = ["code", "creative", "chat", "math", "translation", "lists_format", "writing_tasks", "opinion_advice", "odd_inputs", "mixed_domain"]
names = {"lists_format": "lists format", "writing_tasks": "writing tasks", "opinion_advice": "advice and opinion",
         "odd_inputs": "odd inputs", "mixed_domain": "general how-to"}
order = [c for c in order if c in rows]
fig, ax = plt.subplots(figsize=(6.2, 3.3))
for i, c in enumerate(order):
    s = rows[c]["stopped"]; t = rows[c]["n"]
    ax.barh(i, s, color="#eb6834", edgecolor="white", linewidth=1.5, height=0.7, label="stopped" if i == 0 else None)
    ax.barh(i, t - s, left=s, color="#d3d2cd", edgecolor="white", linewidth=1.5, height=0.7, label="not stopped" if i == 0 else None)
ax.set_yticks(range(len(order))); ax.set_yticklabels([names.get(c, c) for c in order], color="#2f2e2c", fontsize=9)
ax.invert_yaxis(); ax.set_xlim(0, 10); ax.set_xlabel("prompts (10 per group)", color="#52514e", fontsize=9)
ax.tick_params(colors="#52514e", labelsize=9)
for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color("#e4e3df"); ax.xaxis.grid(True, color="#e4e3df", linewidth=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(a.outdir, "fig_failopen.pdf")); fig.savefig(os.path.join(a.outdir, "fig_failopen.png"), dpi=200)

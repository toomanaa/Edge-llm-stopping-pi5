# Contribution 4: potential of semantic stopping
Offline analysis of the saved answers (from the repo root):
- `python3 src/analysis/gap300.py`: oracle semantic stop vs fixed caps on all 299 live-run answers; whether the
  controller's stopping point follows where the answer first becomes correct.
- `python3 src/analysis/shift.py`: spread of the first-correct point; cap vs threshold when answer lengths shift.
- `python3 src/analysis/variants.py`, `hopeless.py`, `price_policy.py`: alternative detectors, abort rule and
  price-based rule (need `states.f32`).
Outputs are in `results/`. Key numbers: oracle saves 42.5% of tokens at no accuracy loss; the cheapest no-loss cap
(500 tokens) saves 5.6%; rank correlation of stopping point with first-correct point is 0.36 (controller) vs 0.14 (cap).
These oracle numbers use the answer key and are upper bounds.

**Unseen topics.** `src/analysis/concept_split.py` retrains the probe on answers whose topic never appears in the held-out sets and replays the held-out answers (output: `results/concept_split_output.txt`; needs `states.f32`). Held-out boundary AP 0.926 (normal probe 0.945); at threshold 0.99, +0.5 points against the token-limit curve (normal probe +0.9).

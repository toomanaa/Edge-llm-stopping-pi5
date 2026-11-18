# Contribution 3: comparison with tuned token limits

**Question.** Does the learned stopper beat a simple token limit at the same energy?

**Live comparison.** `../c1_energy_savings/` (controller vs a 210-token limit chosen to match the controller's average
token use): energy ratio 1.029 (95% CI 0.958-1.102), accuracy +0.7 points (-2.0 to +3.3).

**Offline curves** (`results/`, produced by `python3 src/analysis/make_fig_c3.py` from the repo root; needs `states.f32`):
both methods swept over their dial (token limit 60-500, threshold 0.7-0.999) and replayed on the 338 held-out answers,
with generated tokens as the energy measure.
- `c3_output.txt`: accuracy gap to the token-limit curve at the same tokens, with 95% bootstrap intervals
  (threshold 0.99: +0.9 points, -1.2 to +3.0; all thresholds: -0.1 to +1.1, every interval includes zero).
- `c3_offline_curves.json`: all curve points.
- `fig_c3_curves.{pdf,png}`: the figure used in the paper.

`price_vs_cap_output.txt` is an earlier offline test of a price-based stopping rule against the same caps (no gain).

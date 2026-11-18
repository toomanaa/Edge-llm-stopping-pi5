# Contribution 5: deployment-level efficiency and responsiveness

Derived from the 299-question run in `../c1_energy_savings/` (no new hardware run).

Run from the repo root: `python3 src/analysis/make_fig_c5.py`

- **Energy per correct answer** (total energy / number of correct answers): no stopping 193 J (95% CI 168-221),
  learned stopper 111 J (95-129, 42.7% lower), token limit 109 J (94-125, 43.8% lower). Correct answers per joule:
  x1.75 (1.62-1.88) for the learned stopper, x1.78 (1.68-1.87) for the token limit.
- **Response time, 160 explanation questions** (median, 95th percentile): no stopping 36.4 s (CI 33.9-38.2), 52.6 s;
  learned stopper 16.3 s (14.6-18.2), 34.1 s; token limit 17.6 s (17.4-17.8), 19.7 s.
- **Projected answers per charge**: x1.89 responses (1.76-2.04) and x1.75 correct responses, assuming board
  energy dominates battery draw. The battery test in `battery/` measured x1.77 and x1.68.

Files in `results/`: `c5_numbers.json`, `c5_output.txt`, `fig_c5_deployment.{pdf,png}`.

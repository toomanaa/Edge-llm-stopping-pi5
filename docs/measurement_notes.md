# Measurement notes
- Energy of a run = trapezoid integral of the 20 Hz power samples clipped to the run's own [start, end]; sample gaps
  above 5 s are skipped. Implemented in `src/analysis/analyze_runs.py`.
- Accuracy = `src/common/correctness.py` (required points present in the text; word-boundary matching).
- Confidence intervals: percentile bootstrap over questions; the runs of one question are resampled together.
- See `ENERGY_CORRECTION.md` for the correction applied to earlier energy numbers in `experiments/earlier_experiments/`.

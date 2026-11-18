# Energy integration correction (no-stopping baseline and fixed-budget sweep)

The earlier `joules()` helper integrated, for a window starting at `t0`, the trapezoid between the
last sample before `t0` and the first sample after it. Normally that segment is about 0.05 s long.
Response `qav2_1026` starts right after a 18,984 s (5.3 h) gap in `qa_v2_power_merged.csv`, so the
helper added about 74.5 kJ to that one response. The window-clipped version (`joules()` in `src/analysis/analyze_runs.py`) integrates only inside [t0, t1] and skips segments between samples more than 5 s apart.

| quantity | before (1,074-question run) | after (1,071 questions) |
|---|---|---|
| no-stopping energy | 246,665.86 J | 172,382.8 J |
| controller energy (1,069 questions with power) | 146,313.55 J | 146,293.8 J |
| reduction on the 1,069 shared questions | 40.6% | 15.0% |
| fixed max_tokens 100 / 200 / 500 | 151.1 / 177.3 / 238.0 kJ | 76.8 / 103.0 / 163.7 kJ |
| max_tokens 200 accuracy | 73.8% | 73.8% (790 of 1,071) |

The "after" column covers the 1,071 questions of the final set (three questions of the first run were removed from the dataset). `heldout_analysis.py` recomputes the energy values from the raw timestamps and power CSVs.

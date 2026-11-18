# Energy integration correction (no-stopping baseline and fixed-budget sweep)

The earlier `joules()` helper integrated, for a window starting at `t0`, the trapezoid between the
last sample before `t0` and the first sample after it. Normally that segment is about 0.05 s long.
Response `qav2_1026` starts right after a 18,984 s (5.3 h) gap in `qa_v2_power_merged.csv`, so the
helper added about 74.5 kJ to that one response. The window-clipped version (`scripts/analysis/`)
integrates only inside [t0, t1] and skips segments between samples more than 5 s apart.

| quantity | before | after |
|---|---|---|
| no-stopping energy, 1,074 questions | 246,665.86 J | 172,406.5 J |
| controller energy (1,072 questions with power) | 146,313.55 J | 146,311.9 J |
| reduction on the 1,072 shared questions | 40.6% | 15.0% |
| fixed max_tokens 100 / 200 / 500 | 151.1 / 177.3 / 238.0 kJ | 76.8 / 103.0 / 163.8 kJ |
| max_tokens 200 accuracy | 73.8% | 73.8% (unchanged) |

Only `qav2_1026` rows changed in `qa_v2_labels.jsonl` (`joules_so_far`, `joules_total`).
`heldout_analysis.py` recomputes everything from the raw timestamps and power CSVs.

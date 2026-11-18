# C5 battery-drain test (Raspberry Pi 4)

Question: does stopping early give more answers per battery charge, as the energy-per-answer numbers predict?

## Setup
- Raspberry Pi 4 (4 GB), llama.cpp commit 89e0aa6fd362617d9073e0dafc18e41241521572 built natively, Llama-3.2-1B-Instruct Q4_K_M
  (SHA-256 6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83, identical on the Pi 5 and the Pi 4).
- Power: Anker 20,000 mAh pack -> FNIRSI FNB48S meter (logged at about 100 samples/s over its USB data port) -> Pi 4.
- Questions: `data/battery_questions.jsonl` = the held-out questions repeated 5 times (ids `_r1` ... `_r5`, a different fixed order each round); identical list for both runs.
  The pack was empty before the list ran out in both runs.
- Run A (stopping on): `sp_controller --modes controller --threshold 0.99 --check-every-n 2 --probe probe_v3.txt`, started 2026-11-11 07:49 with the pack at 100%.
- Run B (no stopping): same command with `--modes full`, started 2026-11-11 21:36 with the pack at 100%.
- Each run ran until the pack cut the Pi 4 off. Wi-Fi and one SSH session were on in both runs; nothing else ran.
- The meter logger crashed several times with USB timeouts (3 times in run A, 7 in run B). A watchdog restarted it (from 14:17 in run A, from the start in run B), so the meter logs are split into several files. Gaps longer than 5 s are skipped in the integration.
  Run A lost 2,978 s of meter data (86 of 875 answers without an energy value; its pack energy uses the gap filled at the average power); run B lost 34 s.

## Result (`results/battery_output.txt`, `src/analysis/battery_analysis.py`)
| | Stopping on | No stopping | Ratio |
|---|---|---|---|
| Time on one charge | 8.58 h | 9.00 h | |
| Energy delivered by the pack | 60.3 Wh | 62.5 Wh | |
| Answers per charge | 875 | 493 | x1.77 |
| Correct answers per charge | 641 | 381 | x1.68 |
| Answers per Wh | 14.5 | 7.9 | x1.84 |
| Correct answers per Wh | 10.6 | 6.1 | x1.74 |
| Energy per answer | 245 J | 450 J | 46% less |
| Energy per correct answer | 334 J | 582 J | 43% less |
| Mean power | 7.02 W | 6.94 W | |

Accuracy change, stopping on minus no stopping, on the 493 answers both runs gave (299 distinct questions): -4.1 points (95% CI -6.4 to -2.0, bootstrap over questions).
In both runs 98.7-98.9% of the energy was used inside answers (almost no idle time).

## Limits
One drain per condition (no interval on the ratios). The two runs were on different times of day (A by day, B at night). The board was never throttled in either run (`vcgencmd get_throttled` = 0x0 in all readings); mean temperature was 71.6 C in run A and 69.9 C in run B, maximum 77.9 C and 74.5 C. The question list repeats the same 299 questions, so the answers are not independent.
Run order was always A then B; a swapped-order repeat has not been done.

## Raw files
`A_runs.jsonl`, `B_runs.jsonl`, the temperature logs and stderr logs are in `raw/`. The 100 Hz meter files (about 350 MB) are too large for git and are deposited separately (SHA-256 in `raw/SHA256SUMS_meter.txt`).

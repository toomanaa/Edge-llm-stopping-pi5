# Contribution 1: measured energy savings on a Raspberry Pi 5

**Question.** How much energy does early stopping save on the device, and what does it cost in accuracy?

**Setup.** Llama-3.2-1B-Instruct, 4-bit (Q4_K_M), pinned llama.cpp (see `environment/`), greedy decoding,
Raspberry Pi 5. 299 held-out questions (139 factual, 160 explanation with two required points each; none used to
train or tune the probe). Each question was answered three ways, back to back:

1. `full`: no stopping (up to 800 tokens)
2. `controller`: single-pass learned stopper, threshold 0.99, check at every 2nd sentence end
3. `cap`: stop at the first sentence end after 210 tokens (chosen to match the controller's average token use)

900 runs in total, about 4 hours. Board power was logged at 20 Hz; the energy of a run is the power integrated
over that run's own start and end times (gaps above 5 s skipped), so the cost of the checks is included.

**Command used on the Pi** (from `~/e1`):

```
nohup python3 power_logger.py --out out/run300_power.csv --hz 20 > out/run300_logger.log 2>&1 &
sleep 30
nohup sh -c 'while true; do echo "$(date +%s) $(vcgencmd measure_temp) $(vcgencmd get_throttled)"; sleep 5; done' > out/run300_temp.csv 2>&1 &
nohup ./sp_controller --model models/llama-3.2-1b-instruct-q4_k_m.gguf --data qa_v2_heldout300.jsonl \
  --out out/run300_runs.jsonl --modes full,controller,cap --threshold 0.99 --check-every-n 2 \
  --cap-tokens 210 --probe probe_v3.txt > out/run300_stderr.txt 2>&1 &
```

**Raw files** (`raw/`): `run300_runs.jsonl` (897 lines), `run300_power.csv` (276,304 samples), `run300_temp.csv`.

**Results** (`results/`): `run300_analysis.txt`, `c1_numbers.json`, `fig_c1_energy_accuracy.{pdf,png}`, `integrity_checks.txt`.

| Mode | Accuracy | Energy per question | Saving vs no stopping | Accuracy change |
|---|---|---|---|---|
| No stopping | 76.6% | 148.0 J | - | - |
| Learned stopper | 70.6% | 78.1 J | 47.2% (43.1-51.0) | -6.0 points (-8.7 to -3.3) |
| Token cap (210) | 69.9% | 75.9 J | 48.7% (46.4-50.8) | -6.7 points (-9.7 to -4.0) |

Controller vs cap: accuracy +0.7 points (-2.0 to +3.3), energy ratio 1.029 (0.957-1.102). 95% percentile bootstrap
over questions. By type: factual questions are unchanged (95.0% accuracy, about 14 J, 139 questions); on the 160 explanation
questions energy falls from 264 J to 134 J and accuracy from 60.6% to 49.4%.

**Known conditions and limits.**
- The board ran hot (median 80.7 C, max 84 C); a throttle bit was set in 7.2% of temperature samples. The three
  runs of a question happen back to back, so they share nearly the same conditions.
- In 1 of 299 questions the controller's text diverged from the no-stopping text after a rollback (see `integrity_checks.txt`).
- One model, one board, one question family.
- Checking overhead (about 2%) is estimated from the timed checks multiplied by average power, not from a separate
  energy difference.

**Note on intervals.** `analyze_runs.py` prints the controller accuracy change as -8.7 to -3.7 and `make_fig_c1.py`
as -8.7 to -3.3. Both are 95% bootstrap intervals over questions with different resampling and random seeds. The paper
uses the numbers from `make_fig_c1.py` (`results/c1_numbers.json`).

## Repeat at a lower board temperature
Same 299 questions, modes, program and command as above (`raw_cool/run300_cool_*`), run in an air-conditioned room (air conditioner at 21 C, ceiling fan, no extra cooling of the board), 2026-11-13 14:10 to 17:30. Board median 73.0 C (max 76.3 C), no throttling in any reading (first run: median 80.7 C, throttling in 7.2%).
Results (`results_cool/c1_repeat_output.txt`, `c1_repeat_numbers.json`; script `src/analysis/c1_repeat_analysis.py`): energy per question no stopping 139.7 J, learned stopper 73.7 J, token cap 71.6 J (first run 148.0, 78.1, 75.9). Learned stopper saving 47.2% (43.1-51.0), accuracy change -6.0 points (-8.7 to -3.3), controller vs cap energy ratio 1.030 (0.960-1.105). Output is greedy, so the text is identical to the first run (299 of 299 controller answers), and the repeat tests the energy measurement, not the accuracy. Room temperature was not measured.



# C2: what does one stopping check cost?

Question: how much energy and time does a stopping check cost, for a server-style check (the text so far is read again) and for our single-pass check (stored hidden states)?

## Setup
- Raspberry Pi 5 (wall power, active cooler), Llama-3.2-1B-Instruct Q4_K_M, pinned llama.cpp 89e0aa6fd362617d9073e0dafc18e41241521572, 20 Hz board power.
- 119 held-out questions (80 explanation, 39 factual; one of the original 120 was removed, see `data/README.md`; `data/c2_ids_A.txt`, `data/c2_ids_B.txt`, drawn with a fixed seed from `data/qa_v2_heldout300.jsonl`).
- Every question was run to its natural end three times in a row, with no stopping:
  `full` (no checks), `check_single` (single-pass check at every 2nd sentence end), `check_reencode` (server-style check at every 2nd sentence end:
  the prompt and the text so far are read again in a separate context with an empty cache). The mode order was `full, check_single, check_reencode` for list A and the reverse for list B.
- The checks never stop generation, so all three runs produce identical tokens (240 of 240 comparisons identical text). Extra energy is from the checks alone.
- Program: `src/controller/sp_controller.cpp` (modes `check_single` and `check_reencode`; every check is timed and its text length recorded).

```
nohup python3 power_logger.py --out out/c2_power.csv --hz 20 > out/c2_logger.log 2>&1 &
nohup sh -c 'while true; do echo "$(date +%s) $(vcgencmd measure_temp) $(vcgencmd get_throttled)"; sleep 5; done' > out/c2_temp.csv 2>&1 &
./sp_controller_c2 --model models/llama-3.2-1b-instruct-q4_k_m.gguf --data qa_v2_heldout300.jsonl --out out/c2_runs.jsonl \
  --modes full,check_single,check_reencode --threshold 0.99 --check-every-n 2 --probe probe_v3.txt --ids c2_ids_A.txt
./sp_controller_c2 ... --modes check_reencode,check_single,full ... --ids c2_ids_B.txt
```
Started 2026-11-12 11:45, finished about 15:15. Analysis: `python3 src/analysis/c2_analysis.py --runs experiments/c2_check_cost/raw/c2_runs.jsonl --power experiments/c2_check_cost/raw/c2_power.csv`.

## Result (`results/c2_output.txt`, `results/c2_numbers.json`, `results/fig_c2_check_cost.pdf`)
| | Single-pass | Server-style |
|---|---|---|
| Energy per check | 0.76 J (0.69-0.84) | 43.6 J (41.1-46.0) |
| Time per check | 0.10 s | 5.4 s |
| Extra energy (% of the energy without checks) | 2.0% (1.8-2.2) | 116% (107-125) |
| Time per check at 70 / 790 tokens of text | about 0.15 / 0.17 s | about 1.2 / 14 s |

560 checks, 95% bootstrap intervals over questions. A server-style check costs about 57 times the extra energy of a single-pass check, and its time grows linearly with the text (about 18 ms per token).
A single-pass check takes either about 0.5 ms (stored state) or about 0.15 s (a few tokens are re-read because the sentence boundary falls inside a token; 378 of 560 checks).
In the live 299-question controller run of C1 the checks used 1.97% of the energy (610 checks, 0.76 J each).

## Limits
One board and one model. The server-style check is emulated inside the same program (same prefill work as an embeddings request, without HTTP and JSON handling), so it is a lower bound on the cost of a real server.
The board ran warm (median 77.4 C, maximum 82.3 C) with no active throttling flag in any temperature reading; all three modes of a question ran back to back, so they share the same conditions.

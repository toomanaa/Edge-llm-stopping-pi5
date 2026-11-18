# Edge LLM stopping: measured energy of early stopping on a Raspberry Pi 5

Code, data and results for the paper on stopping LLM generation early on edge hardware.

**Central claim.** Semantic early stopping can be implemented efficiently on resource-constrained
edge devices and delivers substantial energy and latency reductions. A well-tuned token limit remains
competitive, and an oracle analysis shows significant room for more precise stopping decisions.

## The five contributions and where to find them

| # | Contribution | Where | Status |
|---|---|---|---|
| 1 | Hardware-measured energy savings: 47.2% less energy, accuracy -6.0 points; repeated on a cooler board with the same result | `experiments/c1_energy_savings/` (`raw/`, `raw_cool/`) | done, reproducible from raw files |
| 2 | Single-pass stopping controller; cost of one check: 0.76 J (about 2% extra energy) vs 43.6 J for a server-style check | `src/controller/`, `experiments/c2_check_cost/` | done |
| 3 | Comparison with tuned token limits (energy ratio 1.03, 95% CI 0.96-1.10) | `experiments/c1_energy_savings/` (cap 210 run), `experiments/c3_cap_comparison/` | done |
| 4 | Potential of semantic stopping (oracle saves 42.5% of tokens vs 5.6% for the best no-loss cap) | `experiments/c4_semantic_potential/` | done offline |
| 5 | Deployment level: energy per correct answer 193 J -> 111 J; median explanation response 36 s -> 16 s; battery test on a Raspberry Pi 4: 875 vs 493 answers per charge | derived from the C1 run; battery test in `experiments/c5_deployment/battery/` | done (one discharge per condition) |
| - | Behaviour on prompts unlike the training data (100 prompts: 46 stopped, 40 cut before the task was done) | `experiments/fail_open/` | done |

## Layout

```
data/                 questions (dev 39, held-out 299), generated answers, sentence-end index
src/controller/       sp_controller.cpp (single-pass controller; modes full, controller, cap, check_single, check_reencode), probe.hpp
src/probe/            hidden-state extraction, probe training/export
src/common/           answer checker (correctness.py), power logger
src/analysis/         analysis and figure scripts (run from the repo root)
probe/                v1 probe weights and the output of the v3 training run
experiments/          one folder per contribution, with raw files and results
environment/          pinned versions and setup notes
docs/                 measurement notes
```

## Reproduce Contribution 1 (no hardware needed)

From the repo root, with Python 3, numpy, scipy, matplotlib and scikit-learn:

```
python3 src/analysis/analyze_runs.py --runs experiments/c1_energy_savings/raw/run300_runs.jsonl \
   --power experiments/c1_energy_savings/raw/run300_power.csv --data data/qa_v2_heldout300.jsonl --correctness src/common
python3 src/analysis/make_fig_c1.py          # table numbers and the figure
python3 src/analysis/c1_integrity_checks.py  # consistency, power-log and temperature checks
```

## Files that are not in this repo yet

Copy these from the Pi (`~/e1`) before publishing:
- `probe_weights_v3.npz` and `probe_v3.txt` (the probe used in the 300-question run)
- `out/run300_stderr.txt`
- `out/extract_stderr.txt`
- the SHA-256 of `models/llama-3.2-1b-instruct-q4_k_m.gguf`

`states.f32` (about 80 MB, hidden states of all stored answers) is not included. Regenerate it with
`src/probe/extract_states.cpp`, or host it separately. Some analysis scripts (`price_policy.py`,
`variants.py`, `shift.py`, `hopeless.py`) read it from the repo root.

## License
See `LICENSE`.

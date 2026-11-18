# Edge LLM stopping: measured energy of early stopping on a Raspberry Pi 5

Code, data and results for the paper on stopping LLM generation early on edge hardware.
The model is Llama-3.2-1B-Instruct (Q4_K_M) running on llama.cpp. Energy was measured with a power meter on a Raspberry Pi 5; the battery test used a Raspberry Pi 4.

A small probe reads the model's hidden state at sentence ends and decides whether the answer so far is already enough. The controller is written in C++ on top of llama.cpp and checks every second sentence end in a single pass.

## Contributions and where to find them

| # | Contribution | Where |
|---|---|---|
| 1 | Hardware-measured energy savings: 47.2% less energy, accuracy -6.0 points; repeated on a cooler board with the same result | `experiments/c1_energy_savings/` (`raw/`, `raw_cool/`) |
| 2 | Single-pass stopping controller; cost of one check 0.76 J (about 2% extra energy) vs 43.6 J for a server-style check | `src/controller/`, `experiments/c2_check_cost/` |
| 3 | Comparison with tuned token limits (energy ratio 1.03, 95% CI 0.96-1.10) | `experiments/c1_energy_savings/` (cap 210 run), `experiments/c3_cap_comparison/` |
| 4 | Headroom of semantic stopping: an oracle saves 42.5% of tokens vs 5.6% for the best token limit that loses no accuracy | `experiments/c4_semantic_potential/` |
| 5 | Deployment: energy per correct answer 193 J -> 111 J; median wait for an explanation 36 s -> 16 s; battery test on a Raspberry Pi 4: 875 vs 493 answers per charge | derived from the C1 run; battery test in `experiments/c5_deployment/battery/` |
| - | Behavior on 100 prompts unlike the training data: 46 stopped early; for 29 of these more than half of the words were cut | `experiments/fail_open/` |

## Layout

```
data/                 questions (39 dev, 299 held-out, the 1,071-question set), 100 unfamiliar prompts, generated answers, sentence-end index
src/controller/       sp_controller.cpp (single-pass controller; modes full, controller, cap, check_single, check_reencode), probe.hpp
src/probe/            hidden-state extraction, probe training and export
src/common/           answer checker (correctness.py), power logger
src/analysis/         analysis and figure scripts (run from the repo root)
probe/                probe weights (v3 is the one used in the paper) and the output of the v3 training run
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

## Large files

Two large files are attached to the release `data-v1` of this repository instead of the folders:
- the meter files of the battery test (12 files, logged by the power meter)
- `states.f32` (about 80 MB, hidden states of all stored answers). Some analysis scripts (`price_policy.py`, `variants.py`, `shift.py`, `hopeless.py`) read it from the repo root. It can also be regenerated with `src/probe/extract_states.cpp`.

Checksums of the files in this repository are in `SHA256SUMS.txt`. Check them on Linux or macOS with `sha256sum -c SHA256SUMS.txt`. Checksums of the release files are in the release notes.

## Model

Llama-3.2-1B-Instruct, Q4_K_M, file `llama-3.2-1b-instruct-q4_k_m.gguf`
SHA-256: `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83`

## License
See `LICENSE`.

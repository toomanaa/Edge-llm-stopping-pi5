# Data

## The three datasets of the paper
| Dataset | File | Size | Made by |
|---|---|---|---|
| Smart-home commands | `commands.jsonl` | 333 commands (100 with a self-correction) | `src/analysis/earlier/make_dataset.py` |
| Factual questions | `qa_304.jsonl` | 304 questions | `src/analysis/earlier/make_qa_dataset.py` |
| Factual and explain-why questions | `qa_v2_all_questions.jsonl` | 1,071 questions (431 factual, 640 explain-why) | `src/analysis/earlier/make_qa_v2.py` |

All three are made from templates and fixed lists with a fixed random seed. Running a script from the repo root
writes the same file again. No language model produced any of the data while the experiments ran.

Fields: `commands.jsonl` has `id, utterance, ground_truth, family, has_correction`; `qa_304.jsonl` has
`id, question, answers`; `qa_v2_all_questions.jsonl` has `id, question, type, answers, match_mode`
(`any`: one answer string is enough; `all`: every required concept must appear).

## Splits and other files used by the experiments
- `qa_v2_dev40.jsonl` (39 questions), `qa_v2_heldout300.jsonl` (299 questions): the two sets used for live runs (never used to train the probe).
- `qa_v2_calib.jsonl`: calibration questions used to choose settings.
- `battery_questions.jsonl`: the question list of the battery test (the 299 held-out questions, five orders, 1,495 lines).
- `failopen_prompts.jsonl`: the 100 prompts that are unlike the training data (made by `src/analysis/make_failopen_prompts.py`).
- `gen_slim.jsonl`: stored answers (1,071) used to build probe training data and for offline replays.
- `states_index.jsonl`: one row per sentence end (answer id, boundary number, character offset, token counts);
  row k matches row k of `states.f32` (2,048 floats per row; not included, see the root README).



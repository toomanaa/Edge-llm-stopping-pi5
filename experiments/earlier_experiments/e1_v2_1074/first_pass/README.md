# First-pass labels (plain substring matching)

`qa_v2_labels_first_pass.jsonl` was produced by
`label_qa_v2_boundaries_first_pass.py` (exact substring matching for both
question types) on `../qa_v2_gen_merged.jsonl`. It was regenerated from the
saved model outputs and gives 874 correct responses and 13,569 hard
boundaries, 3,623 of them unsafe. `report_first_pass.txt` is its report.

The corrected labels are `../qa_v2_labels.jsonl` (shared `correctness.py`:
word-form matching for explain questions, word-boundary matching for factual
ones): 877 correct responses, 13,568 hard boundaries, 3,507 unsafe.

`changes_first_pass_vs_corrected.json` lists the differences. Among hard
boundaries present in both passes, 153 changed from unsafe to safe and 90
from safe to unsafe (all explain questions). 8 responses left the correct
set and 11 entered it. The net change in unsafe boundaries is 116.

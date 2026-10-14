# Canonical replay data schema

One row represents one synchronized IMU sample.

| Column | Unit | Required | Meaning |
|---|---|---:|---|
| `timestamp_s` | s | yes | monotonic time from replay start |
| `ax`, `ay`, `az` | m/s² | yes | tri-axial acceleration |
| `gx`, `gy`, `gz` | deg/s | yes | tri-axial angular rate |
| `fog_label` | 0/1 | evaluation | ground-truth FoG state |
| `turn_label` | text | evaluation | `left`, `right`, or `straight` |
| `scenario_id` | text | optional | scenario grouping key |

Store only synthetic or approved de-identified data. Raw participant data must not be committed. Document sensor placement and label-generation assumptions in the dataset's accompanying notes.

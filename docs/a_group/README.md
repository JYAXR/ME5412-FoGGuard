# A-group held-out test

The held-out evaluation covers the three delivered FoG models, the current-direction
model, and their fixed offline perception adapter. The fusion reference is
`bilateral_rf`, selected before TEST evaluation. This task does not implement
motion feedback, cue control, preturn intent or final 3D visualization.

## Protocol and integrity

`config/fog_selected.json` defines disjoint TRAIN/development/TEST participants.
`config/a_group_test.json` fixes the scoring convention. The first phase below
creates an exclusive pre-score lock of all four weights, thresholds, split,
perception configuration and relevant inference/evaluation source bytes. It checks
the model fitting and threshold-selection participant metadata before extracting
any TEST features. No model fitting or parameter search is performed.

```bash
python -X utf8 scripts/evaluate_a_group_test.py lock --fog-bundle artifacts/fog-selected --direction-run artifacts/a2-current --lock artifacts/a_group_test_lock.json
python -X utf8 scripts/evaluate_a_group_test.py score --fog-bundle artifacts/fog-selected --direction-run artifacts/a2-current --lock artifacts/a_group_test_lock.json --output-dir reports/a_group_test_NEW
```

Use Python 3.12 with `requirements-a2.txt`, `requirements-a1-deep.txt`, and an
editable project installation. The raw file defaults to the original FoG-STAR
CSV; its SHA-256 must match both training configurations. Weights are shared
separately. Supply the A2 training directory containing `current/model.joblib`
and `results.json`. Preserve the lock and output directory.

## Interpretation

- Primary FoG method comparison uses the intersection of record/window identities
  across all three methods. Native support, participant metrics, missing support,
  causal raw-time coverage and raw-event overlap are reported separately.
- FoG window truth is the existing >=50% positive-sample label of a completed
  window. Raw-time/event metrics use only time after a result becomes available,
  capped at 0.5 s, rather than crediting the preceding input interval.
- Direction targets use the existing 80% purity and endpoint rule. Known-target
  accuracy counts unknown predictions as errors; accepted-only accuracy is always
  accompanied by coverage. Unknown target windows and invalid sensor windows have
  separate denominators. Macro-F1 averages the three named directions for the
  primary score and all four classes for the full-stream score.
- Fused FoG and direction are scored at their own source-arrival grids. Timer
  rows are not independent classification samples. Fused stable FoG is compared
  with trailing-window truth; it is not a claim of instant raw-sample detection.
- Runtime CSV files never contain truth. Annotation columns are joined after
  inference and stored in separate evaluation CSV files.
- The test timeline uses the first record by sorted identity and its first 30 s,
  fixed before scoring. It is illustrative, not a selected success case.

The lock prevents changed artifacts from being scored under the same protocol.
An identical repeat can reproduce arithmetic; it is not a fresh independent test.
Once scores are opened, these participants are considered consumed for model
selection. Further tuning requires another untouched cohort for a new final test.
The small held-out cohort supports a course-project generalization check, not
clinical validation or a precise population-wide performance estimate.

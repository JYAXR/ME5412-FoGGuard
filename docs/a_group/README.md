# A-group held-out test

The held-out evaluation covers the three delivered FoG models, the current-direction
model, and their fixed offline perception adapter. The fusion reference is
`bilateral_rf`, selected before TEST evaluation. This task does not implement
motion feedback, cue control, preturn intent or final 3D visualization.

## Locked TEST results

Four held-out participants provide 16 records and 1,605 matched FoG windows,
including 253 positive windows. The fixed weights and thresholds were not refit.

| FoG model | Accuracy | Precision | Recall | FoG F1 |
|---|---:|---:|---:|---:|
| Ankle-history RF | 72.96% | 0.346 | 0.806 | 0.485 |
| Bilateral RF | 78.69% | 0.413 | 0.838 | 0.554 |
| Ankle CNN | 85.42% | 0.535 | 0.581 | 0.557 |

Always predicting non-FoG gives 84.24% accuracy and zero FoG recall/F1 on this
cohort. Therefore accuracy alone is not a suitable primary FoG result. The RF
models retain higher recall with substantial false alarms; the CNN has fewer
false alarms and more missed FoG. The primary fusion reference remains the
preselected bilateral RF; the TEST ranking is not used to switch models.

| Current direction, same 676 valid known-target windows | Raw model | Fixed fusion |
|---|---:|---:|
| Accuracy, rejection counted as error | 81.07% | 51.78% |
| Three-class macro-F1 | 0.844 | 0.592 |
| Accepted prediction coverage | 86.69% | 54.73% |
| Accuracy among accepted predictions only | 93.52% | 94.59% |

The direction grid contains 1,605 windows: 398 invalid sensor windows are excluded,
and 531 valid unknown-target windows enter the separate four-class result.
The 676-window known-target denominator is not all recorded time. An always-straight
baseline scores 56.66% on that denominator. Fixed fusion increases rejection and
does not improve overall accuracy; accepted-only accuracy must not hide this loss.

At the 1,605 bilateral-source arrivals, the stable FoG event output scores F1 0.523,
precision 0.382 and recall 0.830 against trailing-window targets. Its 0.10 hysteresis
can keep positive state through score dips and increase false alarms. It is not
an accuracy improvement over the raw bilateral model. Average precision in the
machine-readable stable-FoG block describes the underlying raw score ranking,
not a newly learned event score.

All three FoG F1 values are below their development-validation values. FoG
prevalence also differs (23.34% development versus 15.76% TEST); cohort composition
and persistent false positives limit how directly the scores can be compared.
The higher raw direction TEST score is an estimate on different held-out people,
not evidence of a model change or optimization.

English slides should use [the TEST presentation package](presentation/README.md)
for final effect tables and [development figures](../a1/figures/README.md) to explain
method selection. Aggregate counts: [results.json](results.json). Integrity evidence:
[pre-score lock](TEST_LOCK.json) and [independent verification](INDEPENDENT_VERIFICATION.json).

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
python -X utf8 scripts/verify_a_group_test.py --run reports/a_group_test_NEW --lock artifacts/a_group_test_lock.json --raw data/raw/fogstar/sensor_data.csv --output reports/a_group_test_NEW/INDEPENDENT_VERIFICATION.json
python -X utf8 scripts/build_a_group_test_report.py --run reports/a_group_test_NEW --output-dir reports/a_group_test_figures_NEW
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

# Aggregate data dictionary

`model_comparison.csv` contains one row per trained configuration, evaluated on the same matched development-validation windows. It contains aggregate results, not individual patient signals.

| Column | Definition |
|---|---|
| `method` | Stable model identifier: ankle_history_rf, bilateral_rf or ankle_cnn |
| `threshold` | Decision threshold applied to the uncalibrated FoG score |
| `windows` | Number of matched validation windows |
| `positive_windows` | Windows with at least 50% FoG-labeled samples |
| `TP / FP / FN / TN` | True positive, false positive, false negative and true negative window counts |
| `precision` | TP / (TP + FP) |
| `recall` | TP / (TP + FN) |
| `fog_f1` | 2 TP / (2 TP + FP + FN) |
| `average_precision` | Average precision from window-level FoG scores and binary labels |
| `truth_events` | Continuous raw sample-level FoG intervals in the evaluated recordings |
| `covered_raw_events` | Raw FoG events with any positive-duration alarm overlap; not full-event detection or event F1 |
| `supported_events` | Raw FoG events with any valid decision-time support |
| `true_alarm_s` | Alarm duration overlapping raw FoG labels, in seconds |
| `false_alarm_s` | Alarm duration overlapping non-FoG labels, in seconds |
| `time_recall` | Raw FoG duration covered by alarms / total labeled FoG duration |
| `time_precision` | Alarm duration overlapping FoG / total alarm duration |
| `observed_s` | Total observed duration in the evaluated recordings, in seconds |
| `support_s` | Total duration supported by available model decisions, in seconds |
| `support_fraction` | support_s / observed_s |
| `raw_records` | Number of evaluated raw recordings, including records without predictions |

Time-based metrics use predictions from their actual availability at window end, held for at most 0.5 s. Missing support is not a negative prediction. Overlapping intervals are integrated without double counting. Window-level and time-based metrics have different denominators.

## Interpretation

The comparison uses 4 development-validation participants and 20 recordings. Different sensor inputs, threshold-selection policies and checkpoint selection are part of each configuration. These results do not establish statistically significant architecture superiority or performance on a new final test set.

The real-signal timeline is an illustrative recording selected independently of prediction quality. Its shaded intervals use sample-level labels; the window-level confusion matrices use the 50% window-label rule.

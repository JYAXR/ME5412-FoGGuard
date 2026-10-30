# A-group TEST presentation assets

Four English figures are available as PNG for slides and editable SVG. The fixed
evaluation uses 4 held-out participants, 16 records and 1,605 matched FoG windows.
Models, thresholds and fusion rules were locked before scoring.

| Figure | Slide caption |
|---|---|
| [FoG comparison](01_fog_test.png) · [SVG](01_fog_test.svg) | On held-out TEST participants, the RF methods retain higher FoG recall with substantial false alarms; the CNN trades recall for precision. No threshold was tuned on TEST. |
| [FoG confusion matrices](02_fog_confusion.png) · [SVG](02_fog_confusion.svg) | Identical supported windows expose both false positives and missed freezes. Accuracy must be interpreted alongside FoG precision, recall and F1. |
| [Direction and fusion](03_direction_test.png) · [SVG](03_direction_test.svg) | Raw current-direction accuracy is 81.07% on 676 valid known-target windows. Fixed confidence gating and confirmation reduce coverage to 54.73% and accuracy to 51.78%. |
| [TEST process timeline](04_test_timeline.png) · [SVG](04_test_timeline.svg) | The first TEST record and its first 30 seconds are shown using a rule fixed before scoring. Trailing-window targets expose false positive event persistence and direction rejection. |

[Open the local report](index.html) · [FoG aggregate CSV](fog_test_metrics.csv) ·
[Direction aggregate CSV](direction_test_metrics.csv) · [File hashes](MANIFEST.json)

Use the comparison and direction figure for the main result slides. The timeline
is a process/failure example, not a selected success case. The FoG confusion figure
supports questions about false alarms and misses. No raw IMU, record identities
or per-window patient CSVs are published in this package.

Direction accuracy includes rejected known targets as errors; accepted-only
accuracy always needs its coverage denominator. Excluded invalid sensor windows
are counted separately. Current direction does not establish future intention,
and score recovery does not establish a successful forward step.

Full scoring conventions, baselines and limitations: [held-out report](../README.md).

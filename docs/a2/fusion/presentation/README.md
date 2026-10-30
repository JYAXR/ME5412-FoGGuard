# English perception figures

These figures describe the implemented A1+A2 adapter and an illustrative real-IMU
development replay. PNG files are suitable for slides; SVG files are editable.

| Asset | Slide caption |
|---|---|
| [Pipeline PNG](pipeline.png) / [SVG](pipeline.svg) | Independent FoG and current-direction models are aligned by record and source availability; only fresh, distinct observations advance event confirmation. |
| [Real timeline PNG](real_timeline.png) / [SVG](real_timeline.svg) | A real-IMU development record shows FoG scores, confirmed/recovery phases and raw versus smoothed current direction on the decision clock. |
| [Derived figure data](timeline_data.csv) | The displayed relative-time predictions; identifiers and raw IMU are omitted. |
| [Figure manifest](MANIFEST.json) | Runtime/source hashes, display selection rule and figure-file hashes. |

Open [the local figure page](index.html) for both figures. Reproduce them with
`scripts/plot_perception.py`; the command is in [the handoff](../README.md).

## Development evidence

The A2 model was rebuilt from the fixed TRAIN/development split using the original
data hash and configuration. It reproduces the checked-in direction results:

| Current direction, known development targets | Result |
|---|---:|
| Evaluated windows | 1,130 |
| Accuracy, counting rejected known targets as errors | 66.99% |
| Three-class macro-F1 | 0.720 |
| Non-rejected prediction coverage | 76.46% |
| Accuracy among accepted predictions only | 87.62% |

Feature/threshold selection used this development set. The final TEST participants
were not scored; 87.62% is not overall accuracy. These are **raw direction-model**
scores, not the performance of confidence-gated, smoothed direction after fusion.
Fusion adds a 0.60 confidence gate and two-observation direction confirmation, so
its unknown rate and delays differ. Separate fused classification scoring is not
claimed by this runtime-only replay.

| Sensor-only A1+A2 development replay | Result |
|---|---:|
| Participants / records | 4 / 20 |
| New valid A1 bilateral FoG observations | 2,014 |
| New valid A2 current-direction observations | 1,921 |
| Fused decision rows, including clock/expiry rows | 2,130 |
| Confirmed model events | 42 |

The 42 events are model-generated episodes, not 42 verified true freezes. Decision
row counts are unweighted behavior counts, not time coverage or accuracy. Missing
FoG is unknown; missing/rejected direction is not straight. `recovery` describes
score hysteresis, not measured forward movement.

The illustrated record is chosen by a fixed prediction-only rule, documented in
the manifest; it is not a representative performance estimate. The offline replay
uses model-supplied window-end availability, without measured computation latency.
There is no motion-response model, rescue action or patient outcome in these plots.

Aggregate provenance and verification: [results](../results.json).

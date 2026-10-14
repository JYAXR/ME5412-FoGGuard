# Architecture and interface rules

## Data flow

`WindowReplay` emits immutable `SensorWindow` objects. The FoG and turning modules consume the same window independently. Their results are joined into `ControllerContext`; the cueing controller returns one `CueCommand`. The UI displays outputs but does not own detection or control logic. Evaluation calls the same modules used by the UI.

## Boundary rules

- Shared contracts live only in `types.py` and `interfaces.py`.
- Every probability/confidence is in `[0, 1]`; timestamps and durations are seconds.
- Accelerometer input uses m/s²; gyroscope input uses degrees/s; axes are x, y, z.
- A model must be deterministic for identical input/configuration.
- Missing, stale, or low-confidence input must produce `CueType.NONE`.
- Models may add evidence fields but may not rename shared fields without an interface issue.

## MVP integration sequence

1. Load and validate canonical CSV.
2. Produce overlapping two-second windows every 0.5 seconds.
3. Run FoG and turn baselines on every window.
4. Join predictions and run the cueing policy.
5. Render probability, intent, and cue timelines.
6. Compare predicted and labelled windows using the evaluation module.

## Deferred beyond three weeks

Live hardware streaming, deep learning deployment, mobile apps, personalized clinical tuning, closed-loop actuation, and human-subject validation are explicitly out of scope.

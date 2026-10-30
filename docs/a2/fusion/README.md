# A1 + A2 perception handoff

The adapter combines A1 FoG predictions with A2 **current-direction** predictions.
It exports a label-free `perception_runtime.csv` for B1 control and B2 integration.
The current-direction model can be rebuilt from the checked-in training source;
the weights are shared separately and are not stored in Git.

## Rebuild the direction model

Use Python 3.12 and the repository root. Keep the original FoG-STAR file and
`config/fog_selected.json` unchanged; the trainer checks the dataset hash and
participant separation. Each command needs a new output directory.

```bash
python -m pip install -r requirements-a2.txt
python -m pip install -e .
python -X utf8 scripts/train_a2_turning.py --task current --output reports/a2_current_NEW
```

This fits random forests using TRAIN participants and selects the feature set
and rejection threshold on development validation. The selected current model
uses 12 gravity/rotation features and 300 trees. Its outputs include explicit
`unknown`; a zero additional rejection threshold does not remove the learned
unknown class. The preturn models are not inputs to this adapter.

## Real-data development replay

Provide the A1 model bundle and rebuilt A2 run directory:

```bash
python -X utf8 scripts/validate_perception.py --raw data/raw/fogstar/sensor_data.csv --fog-bundle artifacts/fog-selected --method bilateral_rf --direction-run reports/a2_current_NEW --output-dir reports/a2_perception_NEW
python -X utf8 scripts/plot_perception.py --runtime reports/a2_perception_NEW/perception_runtime.csv --output-dir reports/a2_perception_figures_NEW
```

Open the figure directory's `index.html`. The replay loads sensor channels and
identifiers only, retains the fixed four development participants and does not
featurize or score TEST. It saves the two source runtime files, a clock, the fused
runtime, behavior summary, and input/model/config/source/output hashes.
Participant filtering follows chunked CSV parsing; this is not a claim that the
CSV bytes of TEST participants were never parsed.

The models supply window-end availability for offline replay. This is an offline
timing convention, not measured live computation latency. Delayed availability
is supported and tested separately.

## Fuse already saved predictions

```bash
python -X utf8 scripts/fuse_perception.py --fog reports/a1_run/runtime.csv --direction reports/a2_current_NEW/current/direction_runtime.csv --output-dir reports/a2_fused_NEW
```

An optional `--clock` CSV contains `subject_id,session_id,task_id,timestamp_s`.
Without it the timeline covers source arrivals and expiries; a clock also exposes
initial unsupported time and records with no predictions. Raw replay constructs
that clock automatically. Neither command creates feedback or cue commands.

## Time, state and quality contract

| Rule | Implemented behavior |
|---|---|
| Record identity | Match all three keys; reset state between records |
| Availability | Use only results whose `available_at_s <= decision time` |
| Source lifetime | Valid only while `decision time < source window_end_s + 0.5`; processing delay consumes this lifetime |
| Arrival at expiry | A new complete result replaces the expiring result at the same tick |
| Missing/invalid/expired FoG | `fog_state=unknown`, score/threshold/`is_fog` missing, `event_phase=safe_none` |
| Missing/invalid/expired direction | Direction becomes `unknown`; it does not erase a separately valid FoG event |
| FoG confirmation | Two distinct positive source observations; held/timer/direction rows never add evidence |
| FoG recovery | Three distinct negative observations, with 0.10 score hysteresis; this is model-score recovery, not a measured step |
| Direction | Confidence >= 0.60 plus two distinct matching observations; unknown withdraws immediately |
| Time/key errors | Reject non-finite, missing, duplicate or reversed per-source times and availability before input completion |
| Truth | Whitelist runtime columns; ignore annotation columns rather than feeding them to decisions |
| Preturn | Reject non-current tasks and nonzero horizons |

Strict lifetime gating is deliberately conservative: a repeatedly delayed 0.5 s
stream can leave unknown gaps and prevent confirmation. No interpolation closes
those gaps. Timestamps are normalized to nanosecond precision to avoid duplicate
ticks from harmless CSV float representations.

`available_at_s` is the **decision** time in the fused file. `window_end_s` is a
compatibility alias of that decision time for the existing B1 CSV reader; it is
not either source's input end. Original input ranges and availability are retained
as `fog_window_start_s`, `fog_window_end_s`, `fog_available_at_s` and the analogous
`direction_*` fields. Each source also has age, status, freshness, model/version
provenance, and an `*_observation_is_new` flag.

`input_fresh` requires both source inputs to be valid and fresh. A fresh rejected
direction can still be unknown; B1 must also check `turn` and `turn_confidence`.
`is_fog` is nullable. `stable_fog_detected=false` on a missing input is accompanied
by explicit unknown state and must not be interpreted as a negative observation.
No cue fields are generated by perception.

The adapter reuses the existing event and direction state-machine methods;
`types.py` and `interfaces.py` are unchanged. The legacy integrator and demos
keep their existing behavior. B1 can read this export with `replay_guidance`,
but real feedback, valid-observation no-response timing, and four-module acceptance
remain B1/B2 work. B1 must not use arbitrary held/timer rows as new sensor evidence.

## Evidence

See [development figures and bounded results](presentation/README.md). Their
module scores refer to development selection, and their behavior counts are not
accuracy. The separate [locked held-out TEST report](../../a_group/README.md)
scores raw and fused outputs independently and documents coverage losses. It
provides the final English effect tables; live latency and receiver review remain
pending.

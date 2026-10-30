# UI and animation handoff

Start with EXAMPLES.json. Every example contains timeline.json in the same schema. The first six are constructed rule probes; the seventh contains only de-identified cue commands from a real-data model-output replay. It contains no raw sensor streams, true FoG labels or subject/session/task identities.

## Examples

| Example | Content | Origin |
|---|---|---|
| 01_fallback | Forward → alternating march → forward → stop | Constructed rule fixture |
| 02_fading | Forward → gradual fade → stop | Constructed rule fixture |
| 03_left_turn | Left-turn targets and a later fallback | Constructed rule fixture |
| 04_right_turn | Right-turn targets and a later fallback | Constructed rule fixture |
| 05_refreeze | Fade begins, confirmed FoG returns, cue intensity is restored | Constructed rule fixture |
| 06_feedback_gap | Missing feedback withdraws cues; later valid evidence permits re-confirmation | Constructed rule fixture |
| 07_real_output_replay | Cue commands driven by real model outputs and a movement proxy | Real-data derived output replay |

All examples use the same minimal JSON fields for UI playback. Full experimental CSVs remain local and can be reproduced using the saved code and configuration; they are not required by the playback UI.

## JSON format

Top-level metadata identifies the source, whether it is constructed, the source command CSV hash, and that observed footfalls are unavailable. `frames` is an ordered array:

```json
{
  "time_s": 3.1,
  "episode_id": "episode_1",
  "action": "start",
  "state": "GUIDING",
  "reason": "confirmed_event_started",
  "terminal": false,
  "visual": {"enabled": true, "opacity": 1.0, "targets": []},
  "audio": {"enabled": true, "volume": 0.6, "cadence_bpm": 80.0, "mode": "metronome"}
}
```

The snippet illustrates the schema. For actual mode names and target values, read the saved timeline. Each target has `target_id`, `foot`, `x_m`, `y_m`, and `yaw_deg`.

## Playback behavior

1. Advance through frames in their saved order, including frames with the same timestamp. Time is in seconds from the first source decision row of that example.
2. A `start` opens one episode. An `update` changes an existing episode. A `hold` continues it and must not restart audio or animation. `none` does not create an episode.
3. Use `visual.enabled` and `audio.enabled` as enable flags. On stop, remove targets and cancel scheduled sound for that episode.
4. Apply the stored opacity and volume. Use the stored cadence and audio mode. March target changes illustrate alternating prompts; they are not measured footfalls.
5. Process terminal rows, including a final stop at the same timestamp as a prior frame. Do not infer new states or rerun models in the UI.

Target coordinates are illustrative local cue geometry. These files do not contain a full human skeleton or measured body trajectory. If a character walks in the animation, it is an illustrative animation chosen by the designer.

Display source labels: `Simulation` for the first six and `Real-data output replay` for the last. In all cases, label the footprints as suggested targets.

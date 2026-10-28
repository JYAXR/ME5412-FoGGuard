"""Causality, boundary resets, startup coverage and streaming parity."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from fogguard.data.fogstar import FogStarWindow
from fogguard.modeling.a1_context import CONTEXT_VARIANT, ContextFoG, context_matrix, history_links
from fogguard.modeling.a1_fog import A1FogModel, a1_matrix


def windows(count=12):
    rng = np.random.default_rng(5412)
    names = tuple(f"{p}_{s}_{a}" for p in ("ankleL", "back")
                  for s in ("acc", "gyro") for a in "xyz")
    samples = rng.normal(size=(120 + 30 * (count - 1), 12))
    return [FogStarWindow(samples[i * 30:i * 30 + 120], names, 60,
                           i * 0.5, i * 0.5 + 119 / 60, "1", "1", "1") for i in range(count)]


def model(ws):
    values, names, _ = context_matrix(ws)
    forest = RandomForestClassifier(n_estimators=8, random_state=5412).fit(
        values, np.arange(len(ws)) % 2)
    return ContextFoG(forest, CONTEXT_VARIANT, ws[0].channel_names, names, 0.3)


def test_context_matches_lagged_sensor_features_and_keeps_startup_rows():
    ws = windows()
    current, original = a1_matrix(ws, "baseline_2s_v1")
    values, schema, links = context_matrix(ws, base=(current, original))
    assert values.shape == (len(ws), 278)
    assert len(schema) == len(set(schema)) == 278
    np.testing.assert_array_equal(values[:, :132], current)
    assert links[:4, 0].tolist() == [-1] * 4
    assert links[8].tolist() == [4, 0]
    rms = original.index("ankleL_acc_x__rms")
    lag = schema.index("ankleL_acc_x__rms__lag_4s")
    delta = schema.index("ankleL_acc_x__rms__current_minus_lag_4s")
    assert values[8, lag] == current[0, rms]
    assert values[8, delta] == current[8, rms] - current[0, rms]
    assert values[0, lag] == current[0, rms] and values[0, delta] == 0
    assert values[0, schema.index("history_available__lag_4s")] == 0


@pytest.mark.parametrize("change", ["subject_id", "session_id", "task_id", "gap"])
def test_history_resets_at_record_boundaries_and_gaps(change):
    ws = windows(18)
    if change == "gap":
        ws[9:] = [replace(w, start_s=w.start_s + 2, end_s=w.end_s + 2) for w in ws[9:]]
    else:
        ws[9:] = [replace(w, **{change: "2"}) for w in ws[9:]]
    links = history_links(ws)
    assert links[8].tolist() == [4, 0]
    assert links[9].tolist() == [-1, -1]
    assert links[13].tolist() == [9, -1]
    assert links[17].tolist() == [13, 9]


def test_labels_and_future_windows_do_not_change_completed_features():
    ws = windows()
    original = context_matrix(ws[:9])[0]
    changed = [replace(w, fog_label=1, fog_fraction=1, direction_label="left", activity_label=7)
               for w in ws]
    changed[9:] = [replace(w, samples=w.samples * 1000) for w in changed[9:]]
    np.testing.assert_array_equal(original, context_matrix(changed)[0][:9])
    altered = list(ws)
    altered[0] = replace(ws[0], samples=ws[0].samples * 3)
    assert not np.array_equal(original[8, 132:], context_matrix(altered)[0][8, 132:])


def test_roundtrip_streaming_and_future_raw_samples(tmp_path):
    ws = windows()
    detector = model(ws)
    path = tmp_path / "context.joblib"
    detector.save(path)
    loaded = A1FogModel.load(path)
    batch = loaded.predict_windows(ws)
    stream = loaded.new_stream()
    actual = pd.concat([stream.push(w) for w in ws], ignore_index=True)
    pd.testing.assert_frame_equal(batch, actual)
    assert len(batch) == len(ws) and (batch.available_at_s == batch.window_end_s).all()
    raw = np.vstack([ws[0].samples, *(w.samples[-30:] for w in ws[1:])])
    frame = pd.DataFrame(raw, columns=ws[0].channel_names)
    frame["timestamp"] = np.arange(len(raw)) / 60
    before = loaded.predict_frame(frame).iloc[:9]
    frame.loc[360:, ws[0].channel_names[0]] = np.nan
    frame["fog"], frame["activity"] = 1, 7
    pd.testing.assert_frame_equal(before, loaded.predict_frame(frame).iloc[:9])
    assert not any(c.startswith("truth_") for c in batch)


def test_invalid_time_or_channel_schema_rejected_without_stream_mutation():
    ws = windows()
    detector = model(ws)
    stream = detector.new_stream()
    stream.push(ws[0])
    with pytest.raises(ValueError, match="chronological"):
        stream.push(ws[0])
    assert len(stream.windows) == 1
    with pytest.raises(ValueError, match="regular"):
        history_links([replace(ws[0], end_s=2.5)])
    with pytest.raises(ValueError, match="channel"):
        stream.push(replace(ws[1], channel_names=ws[1].channel_names[::-1]))
    assert len(stream.windows) == 1

"""Behavioral acceptance of asynchronous sources, expiry and truth separation."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fogguard.context.fogstar import FogStarIntegrationConfig
from fogguard.cueing.replay import replay_guidance
from fogguard.perception import PerceptionConfig, fuse_perception


def sources(ends=(2, 2.5, 3), scores=(0.9, 0.9, 0.9)):
    common = pd.DataFrame({"subject_id": 1, "session_id": 1, "task_id": 1,
                           "window_start_s": np.array(ends) - 1.9,
                           "window_end_s": ends, "available_at_s": ends,
                           "model_version": "fixture-v1", "prediction_source": "constructed"})
    fog = common.assign(fog_probability=scores, fog_threshold=0.5,
                        is_fog=np.array(scores) >= 0.5)
    direction = common.assign(direction="left", direction_confidence=0.9,
                              direction_task="current", horizon_s=0.0)
    return fog, direction


def clock(times):
    return pd.DataFrame({"subject_id": 1, "session_id": 1, "task_id": 1, "timestamp_s": times})


def at(frame, time):
    return frame.loc[frame.available_at_s.eq(time)].iloc[0]


def test_held_rows_never_confirm_fog_or_direction():
    fog, direction = sources()
    result = fuse_perception(fog, direction, clock=clock([2.1, 2.2, 2.3]))
    assert at(result, 2.3).event_phase == "suspected"
    assert at(result, 2.3).turn == "unknown"
    assert at(result, 2.5).event_phase == "confirmed"
    assert at(result, 2.5).turn == "left"
    assert result.fog_observation_is_new.sum() == 3


def test_delayed_direction_is_not_backfilled_or_row_joined():
    fog, direction = sources()
    direction.available_at_s += 0.1
    result = fuse_perception(fog, direction)
    assert at(result, 2).direction_status == "missing"
    assert at(result, 2.1).turn == "unknown"
    # A 0.1 s availability delay leaves a gap before the next 0.5 s update.
    # Confirmation cannot bridge that unknown period.
    assert at(result, 2.6).turn == "unknown"
    # A1 expires at 2.5, not 2.6, and A2 expires at 3.5, not 3.6.
    assert at(result, 3.5).direction_status == "stale"
    assert not at(result, 3.5).input_fresh


def test_expiry_withdraws_fog_and_direction_without_an_extra_input():
    result = fuse_perception(*sources())
    end = result.iloc[-1]
    assert end.available_at_s == 3.5
    assert end.event_phase == "safe_none" and end.fog_state == "unknown"
    assert pd.isna(end.fog_probability) and pd.isna(end.is_fog)
    assert end.turn == "unknown" and not end.stable_fog_detected


def test_missing_direction_does_not_erase_fog_event():
    fog, direction = sources()
    result = fuse_perception(fog, direction.iloc[:0])
    assert at(result, 2.5).event_phase == "confirmed"
    assert at(result, 2.5).turn == "unknown"
    assert not result.input_fresh.any()
    assert not replay_guidance(result).guidance_action.eq("start").any()


def test_missing_fog_is_unknown_while_direction_can_be_known():
    fog, direction = sources()
    result = fuse_perception(fog.iloc[:0], direction)
    assert at(result, 2.5).turn == "left"
    assert result.event_phase.eq("safe_none").all()
    assert result.fog_probability.isna().all()


def test_processing_delay_consumes_the_source_lifetime():
    fog, direction = sources()
    fog.available_at_s += 0.6
    result = fuse_perception(fog, direction)
    assert not result.fog_input_fresh.any()
    assert not result.stable_fog_detected.any()


def test_expiry_breaks_confirmation_across_gaps():
    fog, direction = sources(ends=(2, 3, 3.5))
    result = fuse_perception(fog, direction)
    assert at(result, 3).event_phase == "suspected"
    assert at(result, 3).turn == "unknown"
    assert at(result, 3.5).event_phase == "confirmed"


def test_invalid_direction_withdraws_immediately():
    fog, direction = sources()
    direction["input_valid"] = [True, True, False]
    result = fuse_perception(fog, direction)
    assert at(result, 2.5).turn == "left"
    assert at(result, 3).turn == "unknown"
    assert at(result, 3).direction_status == "invalid"


def test_invalid_fog_never_counts_as_recovery():
    fog, direction = sources()
    fog["input_valid"] = [True, True, False]
    result = fuse_perception(fog, direction)
    assert at(result, 3).event_phase == "safe_none"
    assert at(result, 3).fog_state == "unknown"


def test_hysteresis_and_three_observation_recovery():
    fog, direction = sources(ends=(2, 2.5, 3, 3.5, 4, 4.5),
                             scores=(0.9, 0.9, 0.45, 0.1, 0.1, 0.1))
    result = fuse_perception(fog, direction)
    assert at(result, 3).event_phase == "confirmed"
    assert at(result, 4).event_phase == "recovery"
    assert at(result, 4.5).event_phase == "normal"


def test_cross_record_state_and_keys_are_isolated():
    fog, direction = sources()
    fog2, direction2 = sources(ends=(2,), scores=(0.9,))
    fog2["task_id"] = direction2["task_id"] = 2
    result = fuse_perception(pd.concat([fog, fog2]), pd.concat([direction, direction2]))
    second = result.loc[result.task_id.eq("2")]
    assert not second.stable_fog_detected.any()
    assert second.turn.eq("unknown").all()


def test_truth_and_future_observations_cannot_change_prefix():
    fog, direction = sources()
    expected = fuse_perception(fog, direction)
    fog["truth_fog"] = [1, 0, 1]
    direction["target"] = "right"
    fog.loc[2, ["fog_probability", "is_fog"]] = [0.1, False]
    direction.loc[2, "direction"] = "right"
    actual = fuse_perception(fog, direction)
    pd.testing.assert_frame_equal(expected.loc[expected.available_at_s < 3],
                                  actual.loc[actual.available_at_s < 3])
    assert "truth_fog" not in actual and "target" not in actual


@pytest.mark.parametrize("kind", ["duplicate", "reversed", "future", "preturn", "missing_key"])
def test_rejects_unsafe_sources(kind):
    fog, direction = sources()
    if kind == "duplicate":
        fog = pd.concat([fog, fog.iloc[-1:]])
    elif kind == "reversed":
        direction = direction.iloc[::-1]
    elif kind == "future":
        fog.loc[0, "available_at_s"] = 1
    elif kind == "preturn":
        direction["direction_task"] = "preturn"
    else:
        fog.loc[0, "task_id"] = None
    with pytest.raises(ValueError):
        fuse_perception(fog, direction)


def test_clock_exposes_initial_unsupported_time_and_missing_record():
    fog, direction = sources()
    ticks = pd.concat([clock([0, 1, 2]), clock([0, 1]).assign(task_id=2)])
    result = fuse_perception(fog, direction, clock=ticks)
    assert at(result.loc[result.task_id.eq("1")], 0).fog_state == "unknown"
    assert result.loc[result.task_id.eq("2"), "fog_status"].eq("missing").all()


def test_b1_cues_use_actual_decision_time_and_withdraw_on_expiry():
    fog, direction = sources(ends=(2, 2.4, 2.8))
    fog.available_at_s += 0.1
    direction.available_at_s += 0.1
    config = PerceptionConfig(events=FogStarIntegrationConfig(
        confirmation_windows=1, direction_confirmation_windows=1, turn_horizon_s=0,
    ))
    result = replay_guidance(fuse_perception(fog, direction, config=config))
    starts = result.loc[result.guidance_action.eq("start")]
    assert len(starts) == 1
    assert starts.iloc[0].guidance_timestamp_s == 2.1
    assert (starts.guidance_timestamp_s >= starts.available_at_s).all()
    assert starts.input_fresh.all()
    assert not result.iloc[-1].guidance_visual_enabled


def test_config_never_allows_lifetime_above_half_second():
    with pytest.raises(ValueError):
        PerceptionConfig(source_max_hold_s=1)


def test_saved_runtime_cli_roundtrip_and_no_overwrite(tmp_path):
    fog, direction = sources()
    fog.to_csv(tmp_path / "fog.csv", index=False)
    direction.to_csv(tmp_path / "direction.csv", index=False)
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/fuse_perception.py"),
               "--fog", str(tmp_path / "fog.csv"), "--direction", str(tmp_path / "direction.csv"),
               "--output-dir", str(tmp_path / "run")]
    subprocess.run(command, cwd=root, check=True, capture_output=True)
    actual = pd.read_csv(tmp_path / "run/perception_runtime.csv")
    assert actual.iloc[-1].fog_state == "unknown"
    assert (tmp_path / "run/MANIFEST.json").exists()
    assert subprocess.run(command, cwd=root, capture_output=True, check=False).returncode != 0

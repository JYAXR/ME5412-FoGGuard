import json
import sys
from dataclasses import replace

import pandas as pd
import pytest

from fogguard.cueing.contracts import GuidanceConfig
from fogguard.cueing.replay import replay_guidance, summarize_guidance
from fogguard.cueing.scenarios import build_guidance_cases
from fogguard.data.responses import load_step_responses


@pytest.fixture(scope="module")
def cases():
    return build_guidance_cases(load_step_responses("data/demo/simulated_step_responses.csv"))


@pytest.mark.parametrize("name", ["normal", "normal_right_turn", "single_candidate"])
def test_negative_cases_never_start(cases, name):
    events, _ = cases[name]
    result = replay_guidance(events)
    assert not result.guidance_action.eq("start").any()
    assert not result.guidance_visual_enabled.any()


@pytest.mark.parametrize("name", ["straight_fog", "left_fog", "right_fog", "short_dip"])
def test_one_episode_and_one_stop_for_confirmed_core_cases(cases, name):
    events, _ = cases[name]
    report = summarize_guidance(replay_guidance(events))
    assert report["start_commands"] == report["stop_commands"] == 1
    assert report["unclosed_commands"] == 0


def test_stationary_is_an_observation_not_an_automatic_veto(cases):
    events, _ = cases["voluntary_stop_observation"]
    result = replay_guidance(events)
    assert result.guidance_action.eq("start").sum() == 1
    assert events.truth_fog_label.sum() == 0


def test_replay_is_deterministic_and_truth_independent(cases):
    events, responses = cases["successful_recovery"]
    first = replay_guidance(events, responses=responses)
    changed = events.copy()
    changed["truth_fog_label"] = 1 - changed.truth_fog_label
    changed["truth_extra"] = "must not read"
    second = replay_guidance(changed, responses=responses)
    fields = [name for name in first if name.startswith("guidance_")]
    pd.testing.assert_frame_equal(first[fields], second[fields])
    pd.testing.assert_frame_equal(first, replay_guidance(events, responses=responses))


def test_future_response_outcomes_cannot_change_earlier_commands(cases):
    events, responses = cases["successful_recovery"]
    prefix = events.loc[events.window_end_s.le(7.0)]
    changed = responses.copy()
    changed.loc[changed.timestamp_s.gt(7.0), "reliability"] = 0.0
    first = replay_guidance(prefix, responses=responses, finalize=False)
    second = replay_guidance(prefix, responses=changed, finalize=False)
    pd.testing.assert_frame_equal(first, second)


def test_successful_recovery_fades_and_stops_earlier_than_fixed(cases):
    events, responses = cases["successful_recovery"]
    config = GuidanceConfig()
    guided = replay_guidance(events, config=config, responses=responses)
    fixed = replay_guidance(events, config=replace(config, enable_response_policy=False),
                            responses=responses)
    assert "FADING" in set(guided.guidance_control_state)
    assert "simulated_recovery_fade_completed" in set(guided.guidance_reason)
    # Both use exactly the same upstream input; no efficacy claim from this duration difference.
    assert summarize_guidance(guided)["active_duration_s"] <= (
        summarize_guidance(fixed)["active_duration_s"]
    )


def test_refreeze_during_fade_restores_same_command(cases):
    events, responses = cases["refreeze_during_fade"]
    result = replay_guidance(events, responses=responses)
    restored = result.loc[result.guidance_reason.eq("fog_reconfirmed_restore_intensity")]
    assert len(restored) == 1
    assert restored.iloc[0].guidance_action == "update"
    assert restored.iloc[0].guidance_audio_volume == GuidanceConfig().volume
    assert result.guidance_action.eq("start").sum() == 1


def test_interrupted_recovery_is_not_reported_as_successful_fade(cases):
    events, responses = cases["recovery_interrupted"]
    result = replay_guidance(events, responses=responses)
    assert "FADING" in set(result.guidance_control_state)
    interrupted = result.loc[result.window_end_s.eq(9.0)].iloc[0]
    assert interrupted.guidance_stable_steps == 0
    assert interrupted.guidance_control_state == "EXIT_OBSERVE"
    assert "simulated_recovery_fade_completed" not in set(result.guidance_reason)
    assert result.guidance_action.eq("start").sum() == 1
    assert summarize_guidance(result)["unclosed_commands"] == 0


@pytest.mark.parametrize("name", ["unknown_direction", "stale_input"])
def test_unsafe_rows_revoke_targets_and_audio(cases, name):
    events, _ = cases[name]
    result = replay_guidance(events)
    unsafe = result.loc[result.guidance_control_state.eq("SAFE_NONE")]
    assert not unsafe.guidance_audio_enabled.any()
    assert not unsafe.guidance_visual_enabled.any()
    assert unsafe.guidance_targets.eq("[]").all()
    assert unsafe.guidance_action.eq("stop").any()


def test_provided_response_timestamps_are_used_once_and_delay_not_added_twice(cases):
    events, responses = cases["response_delayed"]
    result = replay_guidance(events, responses=responses)
    before = result.loc[result.window_end_s.lt(14.0)]
    assert before.guidance_stable_steps.max() == 0
    at_available = result.loc[result.window_end_s.eq(14.0)].iloc[0]
    assert at_available.guidance_stable_steps == 1
    assert at_available.guidance_reason == "simulated_forward_step_accepted"
    assert result.guidance_stable_steps.max() == 2


def test_missing_response_falls_back_without_a_countdown(cases):
    events, responses = cases["response_missing"]
    result = replay_guidance(events, responses=responses)
    assert "RESET_CUE" not in set(result.guidance_control_state)
    assert result.guidance_stable_steps.max() == 0
    assert "unavailable_fixed_fallback" in set(result.guidance_response_source)


def test_persistent_no_response_has_a_bounded_failure_and_does_not_restart(cases):
    events, responses = cases["persistent_no_response"]
    result = replay_guidance(events, responses=responses)
    assert result.guidance_action.eq("start").sum() == 1
    failure = result.loc[result.guidance_reason.eq("episode_timeout")].iloc[0]
    assert failure.guidance_action == "stop"
    assert failure.guidance_control_state == "FAILED"
    assert not failure.guidance_terminal


def test_end_of_stream_always_emits_a_censored_stop(cases):
    events, _ = cases["straight_fog"]
    prefix = events.iloc[:10]
    result = replay_guidance(prefix)
    assert result.iloc[-1].guidance_reason == "replay_ended"
    assert result.iloc[-1].guidance_terminal
    assert summarize_guidance(result)["unclosed_commands"] == 0


def test_false_boolean_strings_do_not_trigger(cases):
    events, _ = cases["straight_fog"]
    events = events.copy()
    events["input_fresh"] = "False"
    result = replay_guidance(events)
    assert not result.guidance_action.eq("start").any()


def test_missing_columns_and_ambiguous_response_association_are_rejected(cases):
    events, responses = cases["response_success"]
    with pytest.raises(ValueError, match="missing columns"):
        replay_guidance(events.drop(columns="input_fresh"))
    other = events.copy()
    other["subject_id"] = "different-subject"
    with pytest.raises(ValueError, match="exactly one"):
        replay_guidance(pd.concat([events, other]), responses=responses)
    with pytest.raises(ValueError, match="no event match"):
        replay_guidance(cases["straight_fog"][0], responses=responses)


def test_session_boundary_cancels_old_episode_and_does_not_reuse_command_id(cases):
    events, _ = cases["straight_fog"]
    first = events.iloc[:10].copy()
    second = first.copy()
    second["subject_id"] = "second-subject"
    result = replay_guidance(pd.concat([first, second], ignore_index=True))
    starts = result.loc[result.guidance_action.eq("start")]
    assert starts.guidance_command_id.nunique() == 2
    assert result.guidance_action.eq("stop").sum() == 2
    assert summarize_guidance(result)["unclosed_commands"] == 0


def test_json_targets_are_finite_and_schemas_identified(cases):
    result = replay_guidance(cases["right_fog"][0])
    for value in result.guidance_targets:
        targets = json.loads(value)
        assert "NaN" not in value
        assert all(target["foot"] in {"left", "right"} for target in targets)
    assert set(result.guidance_schema_version) == {"guidance-proposal-v1"}


def test_command_produces_self_contained_review_and_versioned_results(tmp_path, monkeypatch):
    from scripts.run_guidance_demo import main

    monkeypatch.setattr(sys, "argv", [
        "run_guidance_demo.py", "--predictions", "docs/examples/robot2_predictions.csv",
        "--output-dir", str(tmp_path),
    ])
    main()
    report = json.loads((tmp_path / "guidance_summary.json").read_text())
    assert report["cases"]["supplied"]["input_rows"] == 20
    assert report["cases"]["supplied"]["unclosed_commands"] == 0
    assert len(report["guidance_code_sha256"]) == 64
    assert len(report["cases"]["supplied"]["event_input_sha256"]) == 64
    assert "__GUIDANCE_DATA__" not in (tmp_path / "guidance_review.html").read_text()
    assert (tmp_path / "guidance_supplied.csv").exists()
    assert (tmp_path / "guidance_supplied_fixed.csv").exists()
    assert (tmp_path / "guidance_supplied_no_fade.csv").exists()


def test_exported_spacing_and_fade_references_change_only_the_named_factor(cases, tmp_path):
    from scripts.run_guidance_demo import run_case

    config = GuidanceConfig(enable_adaptive_spacing=True)
    events, responses = cases["adaptive_spacing"]
    _, summary = run_case("adaptive_spacing", events, responses, config, tmp_path)
    references = summary["single_factor_references"]
    assert set(references) == {"no_fade", "no_spacing_adjustment"}
    for reference in references.values():
        changed = {key for key, value in config.as_dict().items()
                   if value != reference["effective_config"][key]}
        assert changed == {reference["changed_setting"]}
        assert reference["summary"]["unclosed_commands"] == 0
    fixed_spacing = pd.read_csv(tmp_path / references["no_spacing_adjustment"]["timeline_file"])
    adjusted = pd.read_csv(tmp_path / "guidance_adaptive_spacing.csv")
    assert fixed_spacing.guidance_step_length_m.eq(config.step_length_m).all()
    assert adjusted.guidance_step_length_m.max() > config.step_length_m
    pd.testing.assert_frame_equal(
        fixed_spacing[events.columns], adjusted[events.columns],
    )

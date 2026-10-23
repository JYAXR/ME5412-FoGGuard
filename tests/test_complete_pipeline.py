import pandas as pd
import pytest

from fogguard.complete import complete_cases, run_complete_pipeline, strategy_configs
from fogguard.cueing.march import MarchGuidanceController
from fogguard.cueing.replay import replay_guidance
from fogguard.evaluation.automation2 import audit_commands


@pytest.mark.parametrize("case", list(complete_cases()))
def test_all_integrated_cases_close_commands_and_preserve_safety(case):
    events, responses = complete_cases()[case]
    for config in strategy_configs().values():
        result = replay_guidance(events, config=config, responses=responses,
                                 controller_type=MarchGuidanceController)
        assert not audit_commands(result)
        stopped = result.loc[result.guidance_action.eq("stop")]
        assert not stopped.guidance_visual_enabled.any()
        assert not stopped.guidance_audio_enabled.any()
        invalid = result.loc[result.guidance_control_state.eq("SAFE_NONE")]
        assert not invalid.guidance_visual_enabled.any()
        assert not invalid.guidance_audio_enabled.any()


def test_complete_march_recovery_is_not_triggered_by_truth_or_stationary_steps():
    events, responses = complete_cases()["march_recovery"]
    result = replay_guidance(events, config=strategy_configs()["march"], responses=responses,
                             controller_type=MarchGuidanceController)
    marching = result.loc[result.guidance_control_state.eq("MARCH_FALLBACK")]
    assert not marching.empty
    assert marching.guidance_stable_steps.eq(0).all()
    assert result.guidance_reason.eq("march_completed_return_forward").sum() == 1
    assert result.guidance_reason.eq("simulated_recovery_fade_completed").sum() == 1
    events = events.drop(columns=[name for name in events if name.startswith("truth_")])
    changed = replay_guidance(events, config=strategy_configs()["march"], responses=responses,
                              controller_type=MarchGuidanceController)
    fields = [name for name in result if name.startswith("guidance_")]
    pd.testing.assert_frame_equal(result[fields], changed[fields])


def test_external_robot2_handoff_runs_without_truth_and_without_downloads():
    candidates = pd.read_csv("docs/examples/robot2_predictions.csv")
    events, commands = run_complete_pipeline(candidates)
    assert len(events) == 20
    assert commands.guidance_action.eq("start").sum() == 2
    assert commands.guidance_action.eq("stop").sum() == 2
    assert not audit_commands(commands)

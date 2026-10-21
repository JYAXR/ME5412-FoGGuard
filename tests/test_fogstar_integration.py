import pandas as pd

from fogguard.context import FogStarIntegrationConfig, FogStarTemporalIntegrator
from fogguard.evaluation import evaluate_fogstar_integration


def _predictions(
    probabilities: list[float],
    *,
    direction: str = "straight",
    subject: str = "1",
) -> pd.DataFrame:
    end = [2.0 + 0.5 * index for index in range(len(probabilities))]
    return pd.DataFrame(
        {
            "window_start_s": [value - 2.0 for value in end],
            "window_end_s": end,
            "fog_probability": probabilities,
            "fog_threshold": 0.5,
            "is_fog": [value >= 0.5 for value in probabilities],
            "direction": direction,
            "direction_confidence": 0.9,
            "state": ["fog" if value >= 0.5 else direction for value in probabilities],
            "subject_id": subject,
            "session_id": "1",
            "task_id": "1",
        }
    )


def test_temporal_integration_confirms_merges_and_recovers_one_event() -> None:
    raw = _predictions([0.1, 0.8, 0.8, 0.35, 0.8, 0.2, 0.2, 0.2])

    result = FogStarTemporalIntegrator().process(raw)

    assert result["event_phase"].tolist() == [
        "normal",
        "suspected",
        "confirmed",
        "recovery",
        "confirmed",
        "recovery",
        "recovery",
        "normal",
    ]
    assert result["event_id"].dropna().nunique() == 1
    assert result["stable_fog_detected"].sum() == 2
    assert (result["cue"] != "none").sum() == 1
    assert result.loc[0, "turn"] == "unknown"
    assert set(result.loc[1:, "turn"]) == {"straight"}


def test_single_positive_window_never_reaches_controller() -> None:
    result = FogStarTemporalIntegrator().process(_predictions([0.1, 0.9, 0.1, 0.1]))

    assert "confirmed" not in set(result["event_phase"])
    assert not result["stable_fog_detected"].any()
    assert set(result["cue"]) == {"none"}


def test_csv_boolean_strings_do_not_turn_false_into_positive() -> None:
    raw = _predictions([0.9, 0.9])
    raw["is_fog"] = "False"

    result = FogStarTemporalIntegrator().process(raw)

    assert set(result["event_phase"]) == {"normal"}
    assert not result["stable_fog_detected"].any()


def test_other_direction_maps_to_unknown_and_blocks_cue() -> None:
    raw = _predictions([0.8, 0.8, 0.8], direction="other")
    raw["yaw_turn"] = "straight"
    raw["yaw_turn_confidence"] = 1.0
    result = FogStarTemporalIntegrator().process(raw)

    assert result["direction"].tolist() == ["other", "other", "other"]
    assert set(result["turn"]) == {"unknown"}
    assert result["stable_fog_detected"].any()
    assert set(result["cue"]) == {"none"}
    assert result.loc[1, "cue_reason"] == "Turning context unavailable"


def test_yaw_fallback_requires_agreement_with_known_model_class() -> None:
    raw = _predictions([0.1, 0.1, 0.1])
    raw["direction_confidence"] = 0.5
    raw["yaw_turn"] = "straight"
    raw["yaw_turn_confidence"] = 0.9

    result = FogStarTemporalIntegrator(
        FogStarIntegrationConfig(use_yaw_agreement_fallback=True)
    ).process(raw)

    assert result.loc[0, "turn"] == "unknown"
    assert set(result.loc[1:, "turn"]) == {"straight"}
    assert set(result["turn_source"]) == {"yaw_agreement_fallback"}


def test_stale_input_enters_safe_none_and_resets_confirmation() -> None:
    raw = _predictions([0.8, 0.8, 0.8])
    raw["received_at_s"] = raw["window_end_s"]
    raw.loc[1, "received_at_s"] += 2.0

    result = FogStarTemporalIntegrator().process(raw)

    assert result.loc[1, "event_phase"] == "safe_none"
    assert result.loc[1, "integration_reason"] == "stale_input"
    assert not result.loc[1, "input_fresh"]
    assert result.loc[2, "event_phase"] == "suspected"
    assert set(result["cue"]) == {"none"}


def test_stationary_proxy_preserves_raw_fog_but_suppresses_effective_event() -> None:
    raw = _predictions([0.8, 0.8, 0.8])
    raw["motion_state"] = "stationary"
    raw["motion_confidence"] = 0.9
    raw["dynamic_accel_rms_mps2"] = 0.05

    result = FogStarTemporalIntegrator().process(raw)

    assert result["is_fog"].all()
    assert result["fog_suppressed"].all()
    assert not result["effective_model_fog"].any()
    assert not result["stable_fog_detected"].any()
    assert set(result["turn"]) == {"unknown"}
    assert set(result["cue"]) == {"none"}


def test_event_evaluation_reports_detection_delay_and_false_event() -> None:
    detected = _predictions([0.1, 0.8, 0.8, 0.8, 0.1, 0.1, 0.1], subject="1")
    detected["truth_fog_label"] = [0, 1, 1, 1, 0, 0, 0]
    detected["truth_turn_label"] = "straight"
    false_event = _predictions([0.1, 0.8, 0.8, 0.1, 0.1, 0.1], subject="2")
    false_event["truth_fog_label"] = 0
    false_event["truth_turn_label"] = "straight"
    integrated = FogStarTemporalIntegrator().process(
        pd.concat([detected, false_event], ignore_index=True)
    )

    report = evaluate_fogstar_integration(integrated)

    assert report["n_truth_events"] == 1
    assert report["n_detected_events"] == 1
    assert report["n_false_positive_events"] == 1
    assert report["mean_confirmation_delay_s"] == 0.5
    assert 0.0 <= report["direction_macro_f1_when_accepted"] <= 1.0

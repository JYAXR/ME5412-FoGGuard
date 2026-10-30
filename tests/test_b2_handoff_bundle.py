"""Reject incomplete, misaligned and tampered saved handoffs."""

import json

import numpy as np
import pandas as pd
import pytest

from fogguard.cueing.proxy_replay import replay_proxy_guidance
from fogguard.evaluation.handoff_bundle import audit_case, seal, verify_manifest
from fogguard.motion.proxy_feedback import GYRO, estimate_motion_feedback
from fogguard.perception import fuse_perception


@pytest.fixture
def saved():
    t = np.arange(12 * 60) / 60
    sensors = pd.DataFrame({"timestamp": t, "subjectID": 1, "sessionID": 1, "taskID": 1,
                            **{c: 0. for c in GYRO}})
    f = estimate_motion_feedback(sensors)
    ends = np.arange(2, 12, .5)
    common = {"subject_id": "1", "session_id": "1", "task_id": "1",
              "window_start_s": ends - 119 / 60, "window_end_s": ends,
              "available_at_s": ends, "model_version": "fixture",
              "prediction_source": "constructed_test"}
    fog = pd.DataFrame(common).assign(fog_probability=.9, fog_threshold=.5, is_fog=True)
    direction = pd.DataFrame(common).assign(direction="straight", direction_confidence=.9,
                                            direction_task="current", horizon_s=0.)
    ticks = np.unique(np.round(np.r_[np.arange(0, 12, .1), f.feedback_available_at_s], 9))
    clock = pd.DataFrame({"subject_id": "1", "session_id": "1", "task_id": "1",
                          "timestamp_s": ticks})
    p = fuse_perception(fog, direction, clock=clock)
    p["cue_test_input_origin"] = "constructed_rule_fixture"
    return p, f, replay_proxy_guidance(p, f, strategy="march")


def test_saved_lifecycle_and_non_step_metrics(saved):
    result = audit_case(*saved, "march")
    assert result["status"] == "PASS" and result["starts"] == result["stops"] == 1
    assert result["fallback_entries"] == result["timeout_returns"] == 1
    assert result["step_accuracy"] is None and result["forward_success_rate"] is None


@pytest.mark.parametrize("damage", ["label", "identity", "clock", "unclosed", "fake_step"])
def test_reject_corrupt_saved_commands(saved, damage):
    p, f, c = saved
    c = c.copy()
    if damage == "label":
        c["truth_fog_label"] = 1
    elif damage == "identity":
        c.loc[0, "subject_id"] = "other"
    elif damage == "clock":
        c.loc[3, "guidance_timestamp_s"] = -1
    elif damage == "unclosed":
        c.loc[c.guidance_action.eq("stop"), "guidance_action"] = "hold"
    else:
        c.loc[0, "guidance_stable_steps"] = 4
    with pytest.raises((ValueError, AssertionError)):
        audit_case(p, f, c, "march")


def test_missing_required_fields_are_not_accepted(saved):
    p, f, c = saved
    with pytest.raises(ValueError, match="Schema"):
        audit_case(p, f.drop(columns="feedback_source"), c, "march")


@pytest.mark.parametrize("source", ["raw", "perception", "feedback"])
def test_reversed_clocks_are_rejected(saved, source):
    p, f, c = saved
    if source == "raw":
        raw = pd.DataFrame({"timestamp": np.arange(60)[::-1] / 60,
                            "subjectID": 1, "sessionID": 1, "taskID": 1,
                            **{channel: 0. for channel in GYRO}})
        with pytest.raises(ValueError, match="strictly increasing"):
            estimate_motion_feedback(raw)
    else:
        if source == "perception":
            p = p.iloc[::-1].reset_index(drop=True)
        else:
            f = f.iloc[::-1].reset_index(drop=True)
        with pytest.raises(ValueError, match="Clock"):
            audit_case(p, f, c, "march")


def test_manifest_detects_tampering_and_unlisted_files(tmp_path):
    data = tmp_path / "commands.csv"
    data.write_text("original", encoding="utf-8")
    seal(tmp_path)
    assert verify_manifest(tmp_path) == 1
    data.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        verify_manifest(tmp_path)
    data.write_text("original", encoding="utf-8")
    (tmp_path / "extra.csv").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="Unlisted"):
        verify_manifest(tmp_path)


def test_manifest_rejects_escape_paths(tmp_path):
    (tmp_path / "MANIFEST.json").write_text(json.dumps(
        {"files_sha256": {"../outside.csv": "not-a-valid-hash"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        verify_manifest(tmp_path)

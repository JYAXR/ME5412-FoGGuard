"""Sampling isolation and independent receiver assertions, without trained weights."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fogguard.evaluation.randomized_plan import (
    CHANNELS,
    build_plan,
    file_hash,
    mutate_sensors,
    select_clips,
)
from fogguard.evaluation.randomized_receiver import decisions_equal, random_sources, receiver_issues
from fogguard.perception import fuse_perception

ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return json.loads((ROOT / "config/b2_randomized_evaluation.json").read_text("utf-8"))


def raw_fixture():
    times = np.arange(4200) / 60
    return pd.concat([
        pd.DataFrame({"subjectID": person, "sessionID": 1, "taskID": 1,
                      "timestamp": times, "fog": (times > 25).astype(int),
                      "activity": 1, **{name: np.sin(times) for name in CHANNELS}})
        for person in (1, 5, 7)], ignore_index=True)


def test_sampling_reproducible_excludes_test_and_no_scored_overlap():
    first = list(select_clips(raw_fixture(), protocol(), [1, 5]))
    second = list(select_clips(raw_fixture(), protocol(), [1, 5]))
    assert [(k, start) for k, start, *_ in first] == [(k, start) for k, start, *_ in second]
    assert {int(k[0]) for k, *_ in first} == {1, 5}
    for key in {k for k, *_ in first}:
        intervals = sorted((low, high) for k, _, low, high, _ in first if k == key)
        assert len(intervals) <= 2
        assert all(intervals[i][1] <= intervals[i + 1][0] for i in range(len(intervals)-1))


def test_plan_labels_separate_and_hashes(tmp_path):
    raw = tmp_path / "raw.csv"
    raw_fixture().to_csv(raw, index=False)
    p = protocol()
    p["raw_sha256"] = file_hash(raw)
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps(p), encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"split": {"validation": [1, 5], "test": [7]}}), "utf-8")
    target = tmp_path / "plan"
    manifest = build_plan(raw, config, split, target)
    assert manifest["models_consulted"] is False
    assert manifest["scored_windows"] == manifest["clips"] * 40
    for path in (target / "clips").glob("*_sensors.csv"):
        assert not {"fog", "activity", "truth_fog_label"} & set(pd.read_csv(path))
    for name, digest in manifest["csv_sha256"].items():
        assert file_hash(target / name) == digest
    with pytest.raises(FileExistsError):
        build_plan(raw, config, split, target)


def test_mutations_are_seeded_sensor_only_and_do_not_change_original():
    raw = raw_fixture().loc[lambda x: x.subjectID.eq(1)].drop(columns=["fog", "activity"])
    before = raw.copy()
    a = mutate_sensors(raw, "noise_low", 12, protocol())
    b = mutate_sensors(raw, "noise_low", 12, protocol())
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(raw, before)
    assert not a[CHANNELS].equals(raw[CHANNELS])
    assert a.timestamp.equals(raw.timestamp)
    gap = mutate_sensors(raw, "right_gyro_gap", 12, protocol())
    assert gap.ankleR_gyro_x.isna().sum() == 120
    with pytest.raises(ValueError, match="annotation"):
        mutate_sensors(raw.assign(fog=0), "clean", 1, protocol())


@pytest.mark.parametrize("seed", [3, 12, 20261030])
def test_receiver_properties_and_reject_corrupted_output(seed):
    fog, direction, clock = random_sources(seed, observations=10)
    result = fuse_perception(fog, direction, clock=clock)
    assert not receiver_issues(fog, direction, result)
    damaged = result.copy()
    index = damaged.index[damaged.fog_status.eq("fresh")][0]
    damaged.loc[index, "fog_input_fresh"] = False
    assert any(i["code"] == "fog_freshness" for i in receiver_issues(fog, direction, damaged))


def test_random_receiver_no_patient_labels_or_scores():
    a = random_sources(24, observations=8)
    b = random_sources(24, observations=8)
    for x, y in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(x, y)
        assert not any(name.startswith("truth_") for name in x)
    assert a[0].prediction_source.eq("constructed_random_receiver_fixture").all()


def test_receiver_accepts_nanosecond_clock_normalization():
    fog, direction, clock = random_sources(33, observations=8)
    for frame in (fog, direction):
        for col in ("window_start_s", "window_end_s", "available_at_s"):
            frame[col] += 1/3
    clock["timestamp_s"] += 1/3
    assert not receiver_issues(fog, direction, fuse_perception(fog, direction, clock=clock))


def test_record_comparison_allows_zero_dtype_but_rejects_changed_values():
    assert decisions_equal(pd.DataFrame({"confidence": [0, 0]}),
                           pd.DataFrame({"confidence": [0.0, 0.0]}))
    assert not decisions_equal(pd.DataFrame({"confidence": [0, 0]}),
                               pd.DataFrame({"confidence": [0.0, 0.01]}))

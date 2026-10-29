"""Causal rate conversion, missing data, label separation and native-time denominators."""

import ast
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fogguard.data.stanford_external import (
    DELAY_S,
    FILTER,
    SOURCE_GYRO,
    canonical_clock,
    project_labels,
    read_workbook,
    resample_gyro,
)
from fogguard.evaluation.stanford_external import record_metrics
from fogguard.modeling.a1_bilateral import DUAL


def test_causal_resampling_matches_direct_fir_and_has_no_future_tail():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(1280, 6))
    time = 4 + np.arange(len(x)) / 128
    frame, _, _ = resample_gyro(time, x)
    np.testing.assert_allclose(np.diff(frame.timestamp), 1 / 60, atol=1e-12)
    prefix, _, _ = resample_gyro(time[:800], x[:800])
    pd.testing.assert_frame_equal(frame.iloc[:len(prefix)].reset_index(drop=True), prefix)
    assert frame.timestamp.iloc[-1] + DELAY_S <= time[-1] + 1e-12
    for j in [0, 1, 70, len(frame) - 1]:
        n = (j + 20) * 32
        source = np.arange(max(0, (n - 640 + 14) // 15), n // 15 + 1)
        expected = FILTER[n - source * 15] @ np.rad2deg(x[source])
        np.testing.assert_allclose(frame.loc[j, list(DUAL)].to_numpy(float), expected, atol=1e-10)


def test_units_and_missing_right_do_not_synthesize_an_ankle():
    time = np.arange(1280) / 128
    x = np.ones((len(time), 6))
    good, _, _ = resample_gyro(time, x)
    np.testing.assert_allclose(good[list(DUAL)].to_numpy(float), 180 / np.pi, rtol=0.003)
    x[400:420, 3:] = np.nan
    bad, _, _ = resample_gyro(time, x)
    np.testing.assert_array_equal(good[list(DUAL[:3])], bad[list(DUAL[:3])])
    assert bad[list(DUAL[3:])].isna().any(axis=1).any()


def test_clock_gaps_split_and_non128_clock_is_rejected():
    t = np.r_[np.arange(128) / 128, 3 + np.arange(128) / 128]
    _, blocks = canonical_clock(t)
    assert blocks == [(0, 128), (128, 256)]
    output, _, _ = resample_gyro(t, np.ones((256, 6)))
    assert not output.timestamp.between(1, 3).any()
    with pytest.raises(ValueError, match="128Hz"):
        canonical_clock(np.arange(128) / 100)
    # Source record 45 stores a rounded interval, accumulating clock rounding error.
    rounded = 206.11962 + np.arange(3751) * 0.007812
    canonical, _ = canonical_clock(rounded)
    np.testing.assert_allclose(canonical, rounded[0] + np.arange(3751) / 128, atol=1e-12)
    assert max(abs(canonical - rounded)) == pytest.approx(0.001875)


def test_labels_are_previous_native_cell_not_filtered_or_interpolated():
    t = np.arange(128) / 128
    y = np.zeros(128)
    y[20] = 1
    np.testing.assert_array_equal(project_labels(t, y, [19.5 / 128, 20.5 / 128, 21.5 / 128]), [0, 1, 0])
    with pytest.raises(ValueError, match="binary"):
        project_labels(t, y + 2, [0.5])


def test_native_events_use_delayed_availability_and_keep_unknown_records():
    t = np.arange(1280) / 128
    labels = ((t >= 4) & (t < 5)).astype(int)
    prediction = pd.DataFrame({"available_at_s": [4 + DELAY_S], "is_fog": [True]})
    metrics, events, _ = record_metrics(t, labels, prediction)
    assert metrics["true_alarm_s"] == pytest.approx(0.5)
    assert events[0]["first_overlap_delay_s"] == pytest.approx(DELAY_S)
    empty = prediction.iloc[:0]
    missing, events, _ = record_metrics(t, labels, empty)
    assert missing["truth_events"] == 1 and missing["supported_events"] == 0
    assert missing["support_s"] == 0 and events[0]["longest_unalarmed_s"] == pytest.approx(1)


def test_sensor_xlsx_reader_does_not_decode_target_data_cells(tmp_path):
    names = ["subject_ID", "time", *SOURCE_GYRO, "freeze_label"]
    letters = list("ABCDEFGHI")
    header = "".join(f'<c r="{c}1" t="inlineStr"><is><t>{n}</t></is></c>' for c, n in zip(letters, names, strict=True))
    cells = "".join(f'<c r="{c}2"><v>{v}</v></c>' for c, v in zip(letters, [1, 0, 1, 2, 3, 4, 5, 6, "TARGET_NOT_NUMERIC"], strict=True))
    path = tmp_path / "trial.xlsx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
                   f'<row r="1">{header}</row><row r="2">{cells}</row></sheetData></worksheet>')
    sensors = read_workbook(path, labels=False)
    assert "freeze_label" not in sensors and sensors[SOURCE_GYRO[0]][0] == 1


def test_external_evaluation_has_no_fit_or_threshold_selection():
    path = Path(__file__).parents[1] / "scripts/evaluate_stanford_external.py"
    source = path.read_text("utf-8")
    calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)]
    assert not [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr in {"fit", "partial_fit", "fit_transform"}]
    assert "choose_threshold" not in source

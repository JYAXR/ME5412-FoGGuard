import numpy as np
import pandas as pd
import pytest

from fogguard.motion.candidate_peaks import candidate_motion_peaks


def frame():
    t = np.arange(240)/60
    df = pd.DataFrame({"timestamp": t})
    for side in ("ankleL", "ankleR"):
        for axis in "xyz":
            df[f"{side}_gyro_{axis}"] = 0.0
    return df


def test_motion_peaks_are_side_specific_and_label_independent():
    data = frame()
    data.loc[60:64, "ankleL_gyro_x"] = 50
    data.loc[120:124, "ankleR_gyro_y"] = 55
    result = candidate_motion_peaks(data)
    assert result.side.tolist() == ["left", "right"]
    assert result.source.eq("offline_unverified_motion_peak").all()
    pd.testing.assert_frame_equal(result, candidate_motion_peaks(data.assign(fog=1, activity=7)))


def test_constant_signal_and_missing_data_do_not_produce_events():
    data = frame()
    data.loc[100:110, "ankleL_gyro_x"] = np.nan
    assert candidate_motion_peaks(data).empty


def test_invalid_clock_rejected():
    data = frame()
    data.loc[10, "timestamp"] = data.loc[9, "timestamp"]
    with pytest.raises(ValueError, match="timestamps"):
        candidate_motion_peaks(data)


def test_missing_segment_cannot_form_peak_across_gap():
    data = frame()
    data.loc[119, "ankleL_gyro_x"] = np.nan
    data.loc[115:118, "ankleL_gyro_x"] = 80
    data.loc[120:123, "ankleL_gyro_x"] = 80
    assert candidate_motion_peaks(data).empty

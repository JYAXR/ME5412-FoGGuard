"""Offline exploratory ankle motion peaks; never a verified footfall detector."""
from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def candidate_motion_peaks(frame: pd.DataFrame, *, height=20.0, prominence=10.0,
                           separation_s=0.3, smooth_samples=5):
    """Return per-ankle angular-speed peaks, with no labels or controller access.

    Centered smoothing and peak selection use future samples, so output is offline
    only. A peak is not an independently verified step or forward movement.
    """
    times = frame.timestamp.to_numpy(dtype=float)
    if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("timestamps must be finite and strictly increasing")
    if height <= 0 or prominence <= 0 or separation_s <= 0 or smooth_samples < 1:
        raise ValueError("parameters must be positive")
    result = []
    for side, prefix in (("left", "ankleL"), ("right", "ankleR")):
        values = frame[[f"{prefix}_gyro_{axis}" for axis in "xyz"]].to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1)
        # Never join across a missing sample or a recording gap.
        breaks = np.r_[True, (~valid[1:]) | (~valid[:-1]) | (np.diff(times) > 1.5 / 60), True]
        boundaries = np.flatnonzero(breaks)
        for start, end in pairwise(boundaries):
            if end-start < max(60, smooth_samples) or not valid[start:end].all():
                continue
            magnitude = np.linalg.norm(values[start:end], axis=1)
            smoothed = np.convolve(magnitude, np.ones(smooth_samples)/smooth_samples, mode="valid")
            peaks, props = find_peaks(smoothed, height=height, prominence=prominence,
                                      distance=int(np.ceil(separation_s * 60)))
            for peak, amplitude in zip(peaks, props["peak_heights"]):
                idx = start + int(peak) + smooth_samples//2
                result.append({"timestamp_s": float(times[idx]), "side": side,
                               "angular_speed_deg_s": float(amplitude),
                               "source": "offline_unverified_motion_peak"})
    return pd.DataFrame(result, columns=["timestamp_s", "side", "angular_speed_deg_s", "source"]).sort_values(
        ["timestamp_s", "side"], ignore_index=True)

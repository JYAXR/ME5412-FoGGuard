"""Post-hoc temporal opportunity and rotation-invariant signal descriptions, no model fit."""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

NAMES = tuple(f"{side}_{field}" for side in ("left", "right") for field in
              ("vector_rms", "gait_power", "freeze_power", "freeze_ratio", "gait_peak_hz"))


def invariant_summaries(samples):
    samples = np.asarray(samples, float)
    if samples.ndim != 3 or samples.shape[1:] != (120, 6) or not np.isfinite(samples).all():
        raise ValueError("Expected finite N x 120 x 6 completed gyro windows")
    dynamic = sosfiltfilt(butter(4, (0.5, 20), fs=60, btype="bandpass", output="sos"), samples, axis=1)
    spectrum = abs(np.fft.rfft(dynamic, axis=1)) ** 2 / 120**2
    f = np.fft.rfftfreq(120, 1 / 60)
    gait, freeze = (f >= 0.5) & (f < 3), (f >= 3) & (f <= 8)
    values = []
    for start in (0, 3):
        power = spectrum[:, :, start:start + 3].sum(axis=2)
        g, z = power[:, gait].sum(axis=1), power[:, freeze].sum(axis=1)
        rms = np.sqrt(np.mean(np.sum(dynamic[:, :, start:start + 3] ** 2, axis=2), axis=1))
        peak = np.where(g > 1e-12, f[gait][power[:, gait].argmax(axis=1)], 0)
        values.extend([rms, g, z, z / np.maximum(g, 1e-12), peak])
    return np.column_stack(values)


def summarize_windows(windows):
    return np.vstack([invariant_summaries(np.stack([w.samples for w in windows[a:a + 256]])) for a in range(0, len(windows), 256)])


def event_opportunities(events, predictions):
    result = []
    for event in events.itertuples():
        frame = predictions.loc[predictions.record_id.eq(event.record_id)].sort_values("available_at_s")
        start = frame.available_at_s.to_numpy(float)
        stop = np.minimum(start + 0.5, np.r_[start[1:], np.inf]) if len(start) else np.array([])
        overlap = np.maximum(0, np.minimum(stop, event.end_s) - np.maximum(start, event.start_s))
        timely = overlap > 1e-6
        labels = frame.truth_fog_label.to_numpy(bool)
        begins = frame.window_start_s.to_numpy(float)
        points = begins[:, None] + np.arange(120)[None, :] / 60
        own_fraction = ((points >= event.start_s - 1e-9) & (points < event.end_s - 1e-9)).mean(axis=1)
        own = own_fraction >= 0.5
        score = frame.fog_probability.to_numpy(float)
        actual = frame.is_fog.to_numpy(bool)
        alarm = float(overlap[actual].sum())
        if not np.isclose(alarm, event.alarm_s, atol=1e-6, rtol=0):
            raise ValueError("Original native event overlap changed")
        result.append({**event._asdict(), "timely_windows": int(timely.sum()),
            "timely_weak_positive_windows": int((timely & labels).sum()),
            "timely_self_majority_windows": int((timely & own).sum()),
            "label_reference_overlap_s": float(overlap[labels].sum()),
            "label_reference_covered": bool((timely & labels).any()),
            "self_majority_reference_covered": bool((timely & own).any()),
            "max_timely_score": float(score[timely].max()) if timely.any() else None,
            "max_timely_weak_positive_score": float(score[timely & labels].max()) if (timely & labels).any() else None,
            "missed_despite_self_majority": bool(not event.covered and (timely & own).any())})
    return pd.DataFrame(result)


def window_groups(fractions):
    x = np.asarray(fractions, float)
    return np.where(x >= 1 - 1e-12, "pure_fog", np.where(x <= 1e-12, "pure_nonfog", "mixed"))


def distribution_table(source, target):
    reference = source.loc[source.window_group.eq("pure_fog")]
    rows = []
    for person, group in target.loc[target.window_group.eq("pure_fog")].groupby("source_person", sort=True):
        for name in NAMES:
            low, high = reference[name].quantile([0.01, 0.99])
            q1, median, q3 = reference[name].quantile([0.25, 0.5, 0.75])
            value = float(group[name].median())
            rows.append({"source_person": int(person), "feature": name, "pure_fog_windows": len(group),
                "train_pure_fog_windows": len(reference), "train_median": median, "target_median": value,
                "train_q01": low, "train_q99": high, "outside_train_q01_q99_fraction": float(((group[name] < low) | (group[name] > high)).mean()),
                "median_difference_in_train_iqr": (value - median) / max(q3 - q1, 1e-12)})
    return pd.DataFrame(rows)


def select_failures(events):
    result = []
    for person in (6, 7):
        missed = events.loc[events.source_person.eq(person) & ~events.covered]
        timely = missed.loc[missed.timely_self_majority_windows.gt(0)]
        pool = timely if len(timely) else missed
        if len(pool):
            row = pool.sort_values(["duration_s", "record_id", "start_s"], ascending=[False, True, True]).iloc[0]
            result.append({"source_person": person, "record_id": int(row.record_id), "start_s": float(row.start_s),
                "end_s": float(row.end_s), "selection": "longest_missed_with_timely_self_majority" if len(timely) else "longest_missed_no_timely_self_majority"})
    return result

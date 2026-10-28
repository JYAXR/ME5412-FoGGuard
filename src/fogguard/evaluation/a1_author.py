"""Descriptive diagnostics; activity truth is never a runtime input."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def annotate(windows, predictions, role):
    frame = predictions.copy()
    frame["truth_fog_label"] = [w.fog_label for w in windows]
    frame["truth_fog_fraction"] = [w.fog_fraction for w in windows]
    frame["truth_activity"] = [w.activity_label if w.activity_label is not None else "mixed" for w in windows]
    frame["data_role"] = role
    return frame


def metrics(frame):
    y, p = frame.truth_fog_label.to_numpy(int), frame.is_fog.to_numpy(bool)
    return {"windows": len(frame), "positive_windows": int(y.sum()),
            "TP": int(((y == 1) & p).sum()), "FP": int(((y == 0) & p).sum()),
            "FN": int(((y == 1) & ~p).sum()), "TN": int(((y == 0) & ~p).sum()),
            "precision": float(precision_score(y, p, zero_division=0)),
            "recall": float(recall_score(y, p, zero_division=0)),
            "fog_f1": float(f1_score(y, p, zero_division=0)),
            "average_precision": float(average_precision_score(y, frame.fog_probability)) if y.sum() else None}


def grouped_diagnostics(frame, column):
    rows = []
    for key, group in frame.groupby(column, dropna=False, sort=False):
        row = {column: key, **metrics(group)}
        negatives = row["FP"] + row["TN"]
        row["negative_windows"] = negatives
        row["false_positive_window_rate"] = row["FP"] / negatives if negatives else None
        rows.append(row)
    return pd.DataFrame(rows)


def matched_frames(frames):
    """Use exact record/sample-time keys; float serialization rounded below 1us."""
    keys = ["subject_id", "session_id", "task_id", "window_start_s", "window_end_s"]
    prepared = {}
    for name, frame in frames.items():
        frame = frame.copy()
        frame[keys[:3]] = frame[keys[:3]].astype(str)
        frame[keys[3:]] = frame[keys[3:]].round(6)
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate prediction window")
        prepared[name] = frame.set_index(keys, drop=False)
    common = None
    for frame in prepared.values():
        common = frame.index if common is None else common.intersection(frame.index, sort=False)
    if common is None or not len(common):
        raise ValueError("No common windows")
    result = {name: frame.loc[common].reset_index(drop=True) for name, frame in prepared.items()}
    truth = next(iter(result.values())).truth_fog_label.to_numpy(int)
    if any(not np.array_equal(frame.truth_fog_label, truth) for frame in result.values()):
        raise ValueError("Matched window truth differs")
    return result

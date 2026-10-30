"""Scoring helpers for frozen real-clip inference; no training or threshold changes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def attach_to_truth(truth, predictions, *, kind):
    """Left-join planned windows; missing decisions remain unavailable, not negative."""
    keys = ["window_start_s", "window_end_s"]
    truth = truth.copy()
    for col in keys:
        truth[col] = truth[col].round(6)
    if truth.duplicated(keys).any():
        raise ValueError("Duplicate truth window identity")
    if predictions is None or predictions.empty:
        result = truth.copy()
        result["supported"] = False
        result["predicted"] = None
        return result, 0
    if any(c.startswith("truth_") for c in predictions.columns):
        raise ValueError("Runtime output must not contain truth fields")
    pred = predictions.copy()
    for col in keys:
        pred[col] = pred[col].round(6)
    if pred.duplicated(keys).any():
        raise ValueError("Duplicate prediction window identity")
    value = "is_fog" if kind == "fog" else "direction"
    if value not in pred:
        raise ValueError(f"Missing {value} in runtime output")
    if "input_valid" in pred:
        pred = pred.loc[pred.input_valid.eq(True)].copy()
    pred = pred.loc[pred[value].notna()].copy()
    if kind == "fog":
        if not pred[value].isin([True, False, 0, 1]).all():
            raise ValueError("FoG decisions must be boolean")
    elif not pred[value].isin(["straight", "left", "right", "unknown"]).all():
        raise ValueError("Unknown direction encoding")
    # Out-of-grid predictions can occur after gaps; count explicitly, never backfill.
    matched_keys = pd.MultiIndex.from_frame(truth[keys])
    predicted_keys = pd.MultiIndex.from_frame(pred[keys])
    outside = int((~predicted_keys.isin(matched_keys)).sum())
    # Warmup rows precede the first scored output and are expected, not alignment failures.
    outside_scored = int((~predicted_keys.isin(matched_keys) &
                          pred.window_end_s.ge(truth.window_end_s.min() - 1e-6) &
                          pred.window_end_s.le(truth.window_end_s.max() + 1e-6)).sum())
    result = truth.merge(pred[keys + [value]], how="left", on=keys, validate="one_to_one")
    result["supported"] = result[value].notna()
    result["predicted"] = result[value]
    result.attrs["warmup_or_other_outside_grid"] = outside - outside_scored
    return result, outside_scored


def fog_metrics(frame):
    """Classification on supported windows; report coverage and excluded truth."""
    valid = frame.loc[frame.supported].copy()
    y = valid.truth_fog_label.astype(int).to_numpy()
    p = valid.predicted.astype(int).to_numpy()
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    return {"planned_windows": len(frame), "supported_windows": len(valid),
            "coverage": len(valid) / len(frame) if len(frame) else None,
            "unsupported_positive_windows": int((~frame.supported &
                                                   frame.truth_fog_label.eq(1)).sum()),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2*tp / (2*tp + fp + fn) if 2*tp + fp + fn else None,
            "accuracy": (tp + tn) / len(valid) if len(valid) else None}


def direction_metrics(frame):
    """Unknown predictions count as errors for known true directions."""
    valid = frame.loc[frame.supported].copy()
    known = valid.loc[valid.truth_direction.ne("unknown")]
    accepted = known.loc[known.predicted.ne("unknown")]
    return {"planned_windows": len(frame), "supported_windows": len(valid),
            "input_coverage": len(valid) / len(frame) if len(frame) else None,
            "known_target_windows": len(known), "accepted_known_windows": len(accepted),
            "known_accuracy_including_rejection": float(known.predicted.eq(
                known.truth_direction).mean()) if len(known) else None,
            "known_accepted_coverage": len(accepted) / len(known) if len(known) else None,
            "accepted_only_accuracy": float(accepted.predicted.eq(
                accepted.truth_direction).mean()) if len(accepted) else None,
            "per_class": {name: {"n": int(known.truth_direction.eq(name).sum()),
                                  "recall": float(known.loc[known.truth_direction.eq(name),
                                                            "predicted"].eq(name).mean())
                                  if known.truth_direction.eq(name).any() else None}
                          for name in ("straight", "left", "right")}}


def event_metrics(sample_truth, attached, score_start, score_stop):
    """Any-overlap event coverage on raw timestamps; conditional delays only."""
    observed = sample_truth.loc[sample_truth.timestamp.ge(score_start - 1e-8) &
                                sample_truth.timestamp.lt(score_stop - 1e-8)].copy()
    labels = observed.fog.to_numpy(int)
    times = observed.timestamp.to_numpy(float)
    supported = attached.loc[attached.supported].sort_values("window_end_s")
    ends = supported.window_end_s.to_numpy(float)
    decisions = supported.predicted.astype(int).to_numpy()
    # A result applies only after its window end and for at most 0.5 s.
    indexes = np.searchsorted(ends, times + 1e-6, side="right") - 1
    valid = indexes >= 0
    if len(ends):
        valid &= (times - ends[np.maximum(indexes, 0)]) < 0.5 - 1e-6
    alarm = np.zeros(len(times), dtype=bool)
    if len(ends):
        alarm[valid] = decisions[indexes[valid]] == 1
    event_rows = []
    active = np.flatnonzero(labels == 1)
    if len(active):
        split = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1)
        for positions in split:
            start, stop = int(positions[0]), int(positions[-1])
            hits = positions[alarm[positions]]
            censored = start == 0 and bool(sample_truth.loc[
                sample_truth.timestamp.lt(score_start - 1e-8), "fog"].tail(1).eq(1).any())
            preexisting = start > 0 and bool(alarm[start - 1])
            delay = (float(times[hits[0]] - times[start]) if len(hits) and
                     not censored and not preexisting else None)
            event_rows.append({"start_s": float(times[start]), "end_s": float(times[stop] + 1/60),
                               "covered": bool(len(hits)), "left_censored": censored,
                               "preexisting_alarm": preexisting, "delay_s": delay,
                               "any_decision_support": bool(valid[positions].any())})
    return event_rows

"""Data feasibility and handoff audits; no new detector or response claims."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from fogguard.data.fogstar import sensor_columns

RAW_KEYS = ["subjectID", "sessionID", "taskID"]
KEYS = ["subject_id", "session_id", "task_id"]
SENSOR_SETS = {
    "fog_baseline": ("ankleL", "back"),
    "bilateral_ankles": ("ankleL", "ankleR"),
    "bilateral_and_back": ("ankleL", "ankleR", "back"),
}


def _longest_run(mask: np.ndarray, timestamps: np.ndarray) -> int:
    longest = current = 0
    for i, valid in enumerate(mask):
        if not valid:
            current = 0
        elif i and timestamps[i] - timestamps[i - 1] > 1.5 / 60:
            current = 1
        else:
            current += 1
        longest = max(longest, current)
    return longest


def audit_records(raw: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Include every raw recording, even when there are no supported windows.

    Sensor availability uses finite raw samples (no interpolation). This is
    separate from actual baseline window support, which is counted from OOF.
    """
    counts = predictions.groupby(KEYS).size().to_dict()
    rows = []
    for key, group in raw.groupby(RAW_KEYS, sort=True):
        times = group.timestamp.to_numpy(float)
        row = dict(zip(KEYS, map(int, key)))
        row.update(samples=len(group), duration_s=float(times[-1] - times[0] + 1 / 60),
                   fog_samples=int(group.fog.sum()),
                   oof_windows=int(counts.get(tuple(map(int, key)), 0)))
        for name, positions in SENSOR_SETS.items():
            channels = list(sensor_columns(positions))
            valid = np.isfinite(group[channels].to_numpy(float)).all(axis=1)
            row[f"{name}_finite_fraction"] = float(valid.mean())
            row[f"{name}_longest_raw_run_samples"] = _longest_run(valid, times)
            row[f"{name}_has_2s_raw_run"] = _longest_run(valid, times) >= 120
        rows.append(row)
    return pd.DataFrame(rows)


def preturn_candidates(raw: pd.DataFrame, horizons=(0.5, 1.0)) -> pd.DataFrame:
    """Audit positive targets before annotated walk->turn transitions.

    Future activity defines evaluation targets only. The input interval ends
    at onset-horizon and contains 120 observed samples, all annotated walking.
    These positives alone are NOT a training/accuracy dataset: negatives and a
    held-out evaluation must be added before claiming intent prediction.
    """
    rows = []
    channels = list(sensor_columns(("ankleL", "back")))
    for key, group in raw.groupby(RAW_KEYS, sort=True):
        times = group.timestamp.to_numpy(float)
        activity = group.activity.to_numpy(int)
        for i in range(1, len(group)):
            if activity[i] not in (6, 7) or activity[i - 1] != 1:
                continue
            onset = float(times[i])
            for horizon in horizons:
                decision = onset - horizon
                stop = int(np.searchsorted(times, decision + 1e-8, side="right"))
                start = stop - 120
                reason = "eligible"
                if start < 0:
                    reason = "insufficient_history"
                else:
                    chunk = group.iloc[start:stop]
                    t = times[start:stop]
                    if decision - t[-1] > 1.5 / 60 or np.any(np.diff(t) > 1.5 / 60):
                        reason = "signal_gap"
                    elif not np.all(activity[start:stop] == 1):
                        reason = "history_not_all_walking"
                    elif not np.isfinite(chunk[channels].to_numpy(float)).all():
                        reason = "missing_sensor"
                row = dict(zip(KEYS, map(int, key)))
                row.update(onset_s=onset, decision_time_s=decision, horizon_s=float(horizon),
                           target_direction="right" if activity[i] == 6 else "left",
                           eligible=reason == "eligible", reason=reason,
                           input_start_s=float(times[start]) if start >= 0 else None,
                           input_end_s=float(times[stop - 1]) if stop else None)
                rows.append(row)
    return pd.DataFrame(rows)


def frozen_folds(report: dict, predictions: pd.DataFrame) -> dict:
    """Check and freeze existing subject assignments without selecting new folds."""
    all_subjects = set(predictions.subject_id.astype(int))
    seen = set()
    folds = report["validation"]["fold_subjects"]
    for fold in folds:
        train, test = (set(map(int, fold[name])) for name in ("train", "test"))
        if train & test or seen & test or train | test != all_subjects:
            raise ValueError("Subject leakage, repeated test subject, or incomplete fold")
        seen |= test
    if seen != all_subjects:
        raise ValueError("Some subjects were never held out")
    return {"schema": "fogguard.frozen-subject-folds.v1",
            "random_state": report["validation"]["random_state"],
            "method": report["validation"]["method"], "folds": folds,
            "threshold_rule": "Training-fold OOB only; never tune on held-out labels",
            "limitation": "OOB threshold selection is not subject-group inner validation; "
                          "use grouped inner validation for new tuning experiments"}


def split_runtime_truth(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Physically remove truth from inference input, retain it in a keyed audit file."""
    truth = [name for name in predictions if name.startswith("truth_")]
    key = KEYS + ["window_start_s", "window_end_s"]
    return predictions.drop(columns=truth).copy(), predictions[key + truth].copy()


def window_metrics(predictions: pd.DataFrame) -> dict:
    y, p = predictions.truth_fog_label.astype(int), predictions.is_fog.astype(int)
    return {
        "windows": len(predictions),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y, p, labels=[0, 1]).ravel().tolist(),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "fog_f1": float(f1_score(y, p, zero_division=0)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y, predictions.fog_probability)) if y.nunique() > 1 else None,
    }

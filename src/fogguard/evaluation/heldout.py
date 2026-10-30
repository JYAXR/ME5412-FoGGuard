"""Fixed held-out metrics and pre-score integrity checks; no model fitting."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from fogguard.evaluation.selected_fog import metrics
from fogguard.perception import KEYS, PerceptionConfig
from fogguard.selected_fog import METHODS, sha
from fogguard.turning.learned import CURRENT_CLASSES

WINDOW_KEYS = KEYS + ["window_start_s", "window_end_s"]


def assert_split(split):
    groups = [set(split[name]) for name in ("train", "validation", "test")]
    if any(not g for g in groups) or any(groups[i] & groups[j]
                                        for i in range(3) for j in range(i)):
        raise ValueError("TRAIN, development and TEST must be nonempty disjoint participant sets")


def validate_fog_membership(model, split):
    assert_split(split)
    if set(model.fit_participants) != set(split["train"]):
        raise ValueError("Model fitting participants differ from locked TRAIN")
    if (not model.threshold_participants
            or not set(model.threshold_participants) <= set(split["train"] + split["validation"])):
        raise ValueError("Threshold selection includes unapproved participants")


def source_paths(root):
    names = ["config/a_group_test.json", "config/fog_selected.json", "config/a2_turning.json",
             "scripts/evaluate_a_group_test.py", "src/fogguard/evaluation/heldout.py",
             "src/fogguard/evaluation/selected_fog.py", "src/fogguard/evaluation/intervals.py",
             "src/fogguard/selected_fog.py", "src/fogguard/data/fogstar.py",
             "src/fogguard/modeling/fogstar_baseline.py", "src/fogguard/turning/learned.py",
             "src/fogguard/perception.py", "src/fogguard/context/fogstar.py"]
    names.extend(p.relative_to(root).as_posix() for p in
                 sorted((root / "src/fogguard/modeling").glob("a1_*.py")))
    return names


def frozen_inputs(root, bundle, direction_run):
    files = {name: root / name for name in source_paths(root)}
    files.update({f"models/{m}": bundle / m / "model.joblib" for m in METHODS})
    files["models/current_direction"] = direction_run / "current/model.joblib"
    files["a1_bundle_manifest"] = bundle / "MANIFEST.json"
    files["a2_training_manifest"] = direction_run / "results.json"
    return {name: sha(path) for name, path in files.items()}


def verify_lock(lock, current):
    if lock.get("schema") != "fogguard.a_group.test_lock.v1":
        raise ValueError("Unsupported test lock")
    if lock["files_sha256"] != current:
        changed = sorted(k for k in set(lock["files_sha256"]) | set(current)
                         if lock["files_sha256"].get(k) != current.get(k))
        raise ValueError(f"Locked source/config/model changed: {changed}")
    if lock["perception_config"] != asdict(PerceptionConfig()):
        raise ValueError("Perception configuration differs from lock")


def window_index(frame):
    result = frame.copy()
    result[KEYS] = result[KEYS].astype(str)
    result[WINDOW_KEYS[3:]] = result[WINDOW_KEYS[3:]].round(6)
    if result.duplicated(WINDOW_KEYS).any():
        raise ValueError("Duplicate record/window identity")
    return result.set_index(WINDOW_KEYS, drop=False)


def attach_fog_truth(runtime, windows):
    truth = pd.DataFrame([{**dict(zip(KEYS, (w.subject_id, w.session_id, w.task_id), strict=True)),
                           "window_start_s": w.start_s, "window_end_s": w.end_s,
                           "truth_fog_label": w.fog_label, "truth_fog_fraction": w.fog_fraction}
                          for w in windows])
    r, t = window_index(runtime), window_index(truth)
    if set(r.index) != set(t.index):
        raise ValueError("Sensor-only predictions and annotation windows differ")
    result = runtime.copy()
    for name in ("truth_fog_label", "truth_fog_fraction"):
        result[name] = t.loc[r.index, name].to_numpy()
    return result


def binary_report(frame):
    report = metrics(frame)
    n = report["windows"]
    report.update(accuracy=(report["TP"] + report["TN"]) / n if n else None,
                  always_non_fog_accuracy=(report["TN"] + report["FP"]) / n if n else None,
                  specificity=report["TN"] / (report["TN"] + report["FP"])
                  if report["TN"] + report["FP"] else None,
                  confusion_classes=["non_fog", "fog"],
                  confusion_matrix=[[report["TN"], report["FP"]], [report["FN"], report["TP"]]])
    return report


def _direction_scores(truth, prediction):
    if len(truth):
        p, r, f, n = precision_recall_fscore_support(
            truth, prediction, labels=CURRENT_CLASSES, zero_division=0,
        )
        matrix = confusion_matrix(truth, prediction, labels=CURRENT_CLASSES).tolist()
    else:
        p, r, f, n = (np.zeros(4) for _ in range(4))
        matrix = np.zeros((4, 4), dtype=int).tolist()
    accepted = np.asarray(prediction) != "unknown"
    return {
        "windows": len(truth), "accuracy": float(np.mean(np.asarray(truth) == prediction))
        if len(truth) else None,
        "macro_f1": float(f.mean()) if len(truth) else None,
        "three_class_macro_f1": float(f[:3].mean()) if len(truth) else None,
        "prediction_coverage": float(accepted.mean()) if len(truth) else None,
        "accepted_accuracy": float(np.mean(np.asarray(truth)[accepted] == prediction[accepted]))
        if accepted.any() else None,
        "classes": CURRENT_CLASSES,
        "confusion_matrix": matrix,
        "per_class": {c: {"precision": float(a), "recall": float(b), "f1": float(v),
                          "support": int(s)} for c, a, b, v, s in
                      zip(CURRENT_CLASSES, p, r, f, n, strict=True)},
    }


def direction_report(frame, prediction_column="direction"):
    valid = frame.input_valid.to_numpy(bool) & frame.target.notna().to_numpy()
    known = valid & frame.target.ne("unknown").to_numpy()
    truth, predicted = frame.target.to_numpy(), frame[prediction_column].to_numpy()
    primary = _direction_scores(truth[known], predicted[known])
    primary["macro_f1"] = primary.pop("three_class_macro_f1")
    primary["always_straight_accuracy"] = float(np.mean(truth[known] == "straight")) if known.any() else None
    per_person = {}
    for key, group in frame.groupby("subject_id", sort=True):
        mask = known & frame.subject_id.eq(key).to_numpy()
        per_person[str(key)] = _direction_scores(truth[mask], predicted[mask])
        per_person[str(key)]["macro_f1"] = per_person[str(key)].pop("three_class_macro_f1")
    return {"primary_known_targets": primary,
            "full_stream_including_unknown": _direction_scores(truth[valid], predicted[valid]),
            "total_grid_windows": len(frame), "missing_sensor_windows": int((~valid).sum()),
            "unknown_target_windows": int((valid & ~known).sum()), "per_subject": per_person}


def align_direction_truth(runtime, meta):
    r, m = window_index(runtime), window_index(meta)
    if set(r.index) != set(m.index):
        raise ValueError("Direction runtime and target grids differ")
    if not np.array_equal(r.input_valid.to_numpy(), m.loc[r.index, "input_valid"].to_numpy()):
        raise ValueError("Direction runtime and truth validity disagree")
    result = runtime.copy()
    result["target"] = m.loc[r.index, "target"].to_numpy()
    return result


def at_source_arrivals(perception, source):
    """Evaluate one row per source arrival, never arbitrary timer/expiry ticks."""
    key = KEYS + ["available_at_s"]
    p, s = perception.copy(), source.copy()
    for frame in (p, s):
        frame[KEYS] = frame[KEYS].astype(str)
        frame["available_at_s"] = frame.available_at_s.round(9)
        if frame.duplicated(key).any():
            raise ValueError("Duplicate source/decision tick")
    p = p.set_index(key)
    wanted = pd.MultiIndex.from_frame(s[key])
    if not wanted.isin(p.index).all():
        raise ValueError("Missing fused decision at a source arrival")
    return p.loc[wanted].reset_index()

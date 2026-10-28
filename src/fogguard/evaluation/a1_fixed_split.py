"""Immutable participant split, validation-only thresholds and fixed sensor stress."""

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score

from fogguard.modeling.a1_posture import search_classifier


def check_split(split, people):
    sets = [set(map(int, split[r])) for r in ("train", "validation", "test")]
    if any(len(v) != len(split[r]) for v, r in zip(sets, ("train", "validation", "test"), strict=True)):
        raise ValueError("Duplicate participant within split")
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)) or set.union(*sets) != set(map(int, people)):
        raise ValueError("Participant overlap or incomplete split")
    if any(not v for v in sets):
        raise ValueError("Empty participant split")


def role_frame(raw, split, role):
    if role not in ("train", "validation", "test"):
        raise ValueError("Unknown data role")
    result = raw.loc[raw.subjectID.astype(int).isin(split[role])].copy()
    if set(result.subjectID.astype(int)) != set(split[role]):
        raise ValueError("Missing expected role participants")
    return result


def choose_threshold(labels, scores, config):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    if set(labels) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Validation requires both classes and finite scores")
    grid = config["threshold_grid"]
    rows = []
    for threshold in np.linspace(grid["start"], grid["stop"], grid["count"]):
        decision = scores >= threshold
        rows.append({"threshold": float(threshold), "fog_f1": float(f1_score(labels, decision, zero_division=0)),
                     "balanced_accuracy": float(balanced_accuracy_score(labels, decision))})
    return {"selected": max(rows, key=lambda r: (r["fog_f1"], r["balanced_accuracy"], r["threshold"])),
            "threshold_scores": rows, "average_precision": float(average_precision_score(labels, scores))}


def classifier_for(model_id, config):
    from fogguard.modeling.a1_multimodel import make_classifier

    if model_id in ("rf_2s", "rf_context", "rf_ankle"):
        return make_classifier("rf", config)
    if model_id in ("extra_2s", "extra_posture"):
        return search_classifier({"family": "extra", "params": config["extra"]}, config["random_state"])
    family = {"svm_2s": "svm", "histgb_2s": "hist_gb", "cnn_2s": "cnn"}.get(model_id)
    if family:
        return make_classifier(family, config)
    raise ValueError("Unknown fixed-split model")


def training_noise_statistics(raw, channels, train_people):
    if set(raw.subjectID.astype(int)) != set(train_people):
        raise ValueError("Noise statistics may use TRAIN people only")
    result = {}
    for channel in channels:
        values = raw[channel].to_numpy(float)
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError("Training channel has no observed samples")
        result[channel] = max(float(values.std()), 1e-6)
    return result


def perturb_sensors(frame, scenario, noise_stats, seed):
    """Preserve all timestamps/labels; no parameters inferred from TEST signals."""
    result = frame.copy()
    channels = list(noise_stats)
    if scenario == "clean":
        return result
    if scenario in ("noise_005", "noise_015"):
        scale = 0.05 if scenario == "noise_005" else 0.15
        rng = np.random.default_rng(seed + (5 if scenario == "noise_005" else 15))
        for channel in channels:
            result[channel] = result[channel] + rng.normal(0, noise_stats[channel] * scale, len(result))
    elif scenario == "rotation_15deg":
        axis = np.ones(3) / np.sqrt(3)
        skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        angle = np.deg2rad(15)
        rotation = np.eye(3) * np.cos(angle) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * skew
        for position in ("ankleL", "back"):
            for sensor in ("acc", "gyro"):
                cols = [f"{position}_{sensor}_{a}" for a in "xyz"]
                result[cols] = result[cols].to_numpy(float) @ rotation.T
    elif scenario == "gyro_bias_3dps":
        for position in ("ankleL", "back"):
            result[f"{position}_gyro_z"] += 3.0
    elif scenario in ("back_dropout_1s", "ankle_dropout_1s"):
        position = "back" if scenario == "back_dropout_1s" else "ankleL"
        columns = [f"{position}_{s}_{a}" for s in ("acc", "gyro") for a in "xyz"]
        for indices in result.groupby(["subjectID", "sessionID", "taskID"], sort=False).groups.values():
            times = result.loc[indices, "timestamp"]
            relative = times - times.iloc[0]
            phase = np.mod(relative - 5, 10)
            missing = (relative >= 5 - 1e-9) & (phase < 1 - 1e-9)
            result.loc[indices[missing.to_numpy()], columns] = np.nan
    else:
        raise ValueError("Unknown predeclared stress scenario")
    return result

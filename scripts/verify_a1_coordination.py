"""Independent proxy formulas, raw inventories and frozen participant-OOF audit."""

import argparse
import ast
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json
from scipy.signal import butter, sosfiltfilt
from verify_a1_bilateral import check_frame, inventory

from fogguard.evaluation.a1_author import matched_frames, metrics
from fogguard.evaluation.a1_event import EventSupport
from fogguard.evaluation.a1_fixed_split import choose_threshold, role_frame
from fogguard.modeling.a1_bilateral import anchored_gyro_windows
from fogguard.modeling.a1_coordination import CoordinationFoG, coordination_matrix
from fogguard.modeling.a1_fog import a1_matrix


def independent_features(samples):
    filtered = sosfiltfilt(butter(4, [0.5, 20], fs=60, btype="bandpass", output="sos"), samples, axis=0)
    envelopes = [np.sqrt(np.square(filtered[:, section]).sum(axis=1)) for section in (slice(0, 3), slice(3, 6))]

    def corr(lag):
        start, stop = max(0, -lag), min(120, 120 - lag)
        a, b = envelopes[0][start:stop], envelopes[1][start + lag:stop + lag]
        if min(np.std(a) * np.sqrt(len(a)), np.std(b) * np.sqrt(len(b))) <= 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    lags = sorted(range(-30, 31), key=lambda v: (abs(v), v))
    scores = np.array([corr(lag) for lag in lags])
    peak = int(np.argmax(scores))
    powers = np.square(np.abs(np.fft.rfft(filtered, axis=0)))
    left, right = powers[:, :3].sum(axis=1), powers[:, 3:].sum(axis=1)
    f = np.arange(61) / 2
    low, high = (f >= 0.5) & (f < 3), (f >= 3) & (f <= 8)
    el, er = [np.square(e).mean() for e in envelopes]
    asym = (el - er) / (el + er + 1e-12)
    band_asym = [(left[m].sum() - right[m].sum()) / (left[m].sum() + right[m].sum() + 1e-12) for m in (low, high)]
    norms = np.linalg.norm(left[low]), np.linalg.norm(right[low])
    cosine = np.clip(np.sum(left[low] * right[low]) / np.prod(norms), 0, 1) if min(norms) > 1e-12 else 0
    difference = abs(f[low][left[low].argmax()] - f[low][right[low].argmax()]) if min(left[low].sum(), right[low].sum()) > 1e-12 else 0
    return np.array([asym, abs(asym), abs(envelopes[0] - envelopes[1]).mean() / (sum(e.mean() for e in envelopes) + 1e-12),
        corr(0), scores[peak], abs(lags[peak]) / 60, (corr(15) + corr(-15)) / 2,
        (corr(30) + corr(-30)) / 2, *band_asym, cosine, difference])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    args = parser.parse_args()
    run, root = args.run_dir, Path(__file__).resolve().parents[1]
    config, split, lock, protocol = [json.loads((run / name).read_text("utf-8")) for name in ("config.json", "split.json", "MODEL_LOCK.json", "protocol.json")]
    complete = json.loads((run / "evaluation/EVALUATION_COMPLETED.json").read_text("utf-8"))
    start = json.loads((run / "evaluation/VALIDATION_STARTED.json").read_text("utf-8"))
    check_lock(run, lock)
    check_lock(run / "evaluation", complete)
    if sha(args.raw) != protocol["raw_sha256"] or complete["model_lock_sha256"] != sha(run / "MODEL_LOCK.json"):
        raise ValueError("Raw data or model lock differs")
    if datetime.fromisoformat(lock["locked_utc"]) >= datetime.fromisoformat(start["started_utc"]):
        raise ValueError("Validation before lock")
    if protocol["data_roles_constructed"] != ["train"] or protocol["validation_windows_before_lock"] or complete["test_windows_constructed"]:
        raise ValueError("Unexpected data exposure")
    for name, digest in protocol["source_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError(f"Changed source: {name}")
    tree = ast.parse((root / "scripts/evaluate_a1_coordination.py").read_text("utf-8"))
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {"fit", "fit_transform", "partial_fit"} for n in ast.walk(tree)):
        raise ValueError("Fitting in evaluation")
    raw = pd.read_csv(args.raw)
    raws = {role: role_frame(raw, split, role) for role in ("train", "validation")}
    del raw
    windows, banks, expected = {}, {}, {}
    for role, role_raw in raws.items():
        windows[role] = anchored_gyro_windows(role_raw, require_right=True)
        banks[role] = coordination_matrix(windows[role])
        expected[role] = inventory(role_raw).loc[lambda x: x.right_complete].reset_index(drop=True)
        independent = np.vstack([independent_features(w.samples) for w in windows[role]])
        np.testing.assert_allclose(independent, banks[role]["coordination_rf"][0][:, 66:], atol=1e-8, rtol=1e-9)
        original, names = a1_matrix(windows[role], "baseline_2s_v1")
        np.testing.assert_array_equal(original, banks[role]["coordination_rf"][0][:, :66])
        if banks[role]["bilateral_control"][1] != names:
            raise ValueError("Control feature schema differs")
    np.testing.assert_allclose(pd.read_csv(run / "train_coordination_features.csv"), banks["train"]["coordination_rf"][0][:, 66:], atol=1e-10, rtol=1e-10)
    plans = json.loads((run / "fold_plan.json").read_text("utf-8"))
    reference = Path(config["reference_run"])
    if plans != json.loads((reference / "fold_plan.json").read_text("utf-8")):
        raise ValueError("Participant plans changed")
    groups, y = expected["train"].subject_id.to_numpy(int), expected["train"].truth_fog_label.to_numpy(int)
    actual = tuple(sorted(map(int, set(groups))))
    all_frames, frames, checked = {}, {}, 0
    for name in config["models"]:
        oof = pd.read_csv(run / name / "oof_predictions.csv", float_precision="round_trip")
        val = pd.read_csv(run / "evaluation" / name / "predictions.csv", float_precision="round_trip")
        checked += check_frame(oof, expected["train"]) + check_frame(val, expected["validation"])
        values = banks["train"][name][0]
        metadata = json.loads((run / name / "fit_metadata.json").read_text("utf-8"))
        seen = np.zeros(len(groups), int)
        for plan, saved in zip(plans, metadata["folds"], strict=True):
            fit, held = [np.flatnonzero(np.isin(groups, plan[k])) for k in ("fit_people", "held_out_people")]
            if set(plan["fit_people"]) & set(plan["held_out_people"]) or set(plan["fit_people"]) | set(plan["held_out_people"]) != set(split["train"]):
                raise ValueError("Participant leakage")
            if saved["actual_fit_people"] != sorted(map(int, set(groups[fit]))) or set(y[fit]) != {0, 1}:
                raise ValueError("Actual fit people/classes differ")
            classifier = joblib.load(run / name / f"fold_{plan['fold']}.joblib")
            np.testing.assert_allclose(classifier.predict_proba(values[held])[:, 1], oof.iloc[held].fog_probability, atol=1e-7, rtol=0)
            seen[held] += 1
        if not np.all(seen == 1):
            raise ValueError("Incomplete OOF")
        model = CoordinationFoG.load(run / name / "model.joblib")
        if model.fit_participants != actual or model.threshold_participants != actual or metadata["fit_people"] != list(actual) or metadata["validation_people_used"] or metadata["test_people_used"]:
            raise ValueError("Fit/calibration people differ")
        selected = choose_threshold(y, oof.fog_probability.to_numpy(), config)["selected"]
        np.testing.assert_allclose(model.threshold, selected["threshold"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(model.predict_windows(windows["validation"]).fog_probability, val.fog_probability, atol=1e-7, rtol=0)
        sensors = raws["validation"].iloc[:180].drop(columns=["fog", "activity", "fog_severity"])
        pd.testing.assert_frame_equal(model.predict_frame(sensors), model.predict_frame(sensors.assign(fog=1, activity=3)))
        runtime = pd.read_csv(run / "evaluation" / name / "runtime.csv")
        if any(c.startswith("truth_") for c in runtime):
            raise ValueError("Truth in runtime")
        if name == "bilateral_control":
            for current, path in [(oof, reference / "bilateral_rf/oof_predictions.csv"), (val, reference / "evaluation/bilateral_rf/predictions.csv")]:
                previous = pd.read_csv(path, float_precision="round_trip")
                np.testing.assert_allclose(current.fog_probability, previous.fog_probability, atol=1e-12, rtol=0)
                np.testing.assert_array_equal(current.is_fog, previous.is_fog)
        frames[name], all_frames[name, "validation_own"], all_frames[name, "train_oof_calibration"] = val, val, oof
    for name in ("previous_author", "reference_rf_context", "reference_rf_ankle", "left_natural"):
        frames[name] = pd.read_csv(run / "evaluation" / name / "predictions.csv", float_precision="round_trip")
        all_frames[name, "validation_own"] = frames[name]
    all_frames.update({(name, "all_common"): frame for name, frame in matched_frames(frames).items()})
    table = pd.read_csv(run / "evaluation/comparison.csv")
    for row in table.itertuples():
        frame = all_frames[row.model, row.pool]
        for field, value in metrics(frame).items():
            if value is not None and not np.isclose(getattr(row, field), value, atol=1e-10, rtol=0):
                raise ValueError(f"Metric differs: {field}")
        if row.pool != "all_common":
            role = "train" if row.pool == "train_oof_calibration" else "validation"
            support = EventSupport.build(raws[role], frame, allowed=split[role])
            for field, value in support.metrics(frame.is_fog).items():
                if not np.isclose(getattr(row, field), value, atol=1e-6, rtol=0):
                    raise ValueError(f"Raw event/time differs: {field}")
    own = table.loc[table.pool.eq("validation_own")].set_index("model")
    a, b = own.loc["bilateral_control"], own.loc["coordination_rf"]
    conditions = {"matched_f1_higher": bool(b.fog_f1 > a.fog_f1), "false_alarm_seconds_not_higher": bool(b.false_alarm_s <= a.false_alarm_s + 1e-6),
        "fog_time_recall_not_lower": bool(b.time_recall + 1e-9 >= a.time_recall), "decision_support_not_lost": bool(b.decision_support_fraction + 1e-9 >= a.decision_support_fraction)}
    gate = json.loads((run / "evaluation/development_gate.json").read_text("utf-8"))
    if gate["conditions"] != conditions or gate["passed"] != all(conditions.values()):
        raise ValueError("Development gate differs")
    result = {"status": "PASS", "verified_new_model_window_rows": checked, "independent_proxy_rows": sum(map(len, windows.values())),
        "fold_models_rescored": 6, "independent_retraining": False, "old_control_train_and_validation_reproduced": True,
        "feature_prefix_and_independent_formulas": "PASS", "raw_inventory_people_threshold_time_lock": "PASS",
        "model_files": len(lock["files"]), "evaluation_files": len(complete["files"]), "test_windows_constructed": 0,
        "model_lock_sha256": sha(run / "MODEL_LOCK.json")}
    write_json(run / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

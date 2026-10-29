"""Audit locked participant folds, independent representation references and native-time scores."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json
from verify_a1_bilateral import check_frame, inventory
from verify_stanford_external import integral
from verify_stanford_failure import direct_descriptors

from fogguard.evaluation.a1_author import metrics
from fogguard.evaluation.a1_event import EventSupport
from fogguard.evaluation.a1_fixed_split import choose_threshold, role_frame
from fogguard.modeling.a1_bilateral import anchored_gyro_windows
from fogguard.modeling.a1_invariant import InvariantFoG, feature_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--diagnostic-reference", type=Path, default=Path("reports/a1_external_failure_v1_20261029"))
    args = parser.parse_args()
    run, root, diagnostic = args.run_dir, Path(__file__).resolve().parents[1], args.diagnostic_reference
    config, split, protocol, lock = [json.loads((run / n).read_text("utf-8")) for n in ("config.json", "split.json", "protocol.json", "MODEL_LOCK.json")]
    output = run / "evaluation"
    completed, started = [json.loads((output / n).read_text("utf-8")) for n in ("EVALUATION_COMPLETED.json", "EVALUATION_STARTED.json")]
    check_lock(run, lock)
    check_lock(output, completed)
    assert started["model_lock_sha256"] == completed["model_lock_sha256"] == sha(run / "MODEL_LOCK.json")
    assert datetime.fromisoformat(lock["locked_utc"]) < datetime.fromisoformat(started["started_utc"])
    assert protocol["data_roles_constructed"] == ["train"] and protocol["test_windows_constructed"] == completed["test_windows_constructed"] == 0
    assert sha(args.raw) == protocol["raw_sha256"] and completed["target_refits_or_calibrations"] == 0
    for name, digest in protocol["source_sha256"].items():
        assert sha(root / name) == digest, name
    diagnostic_lock = json.loads((diagnostic / "DIAGNOSTIC_LOCK.json").read_text("utf-8"))
    check_lock(diagnostic, diagnostic_lock)
    prior_verification = json.loads((diagnostic / "verification.json").read_text("utf-8"))
    assert prior_verification["status"] == "PASS" and prior_verification["diagnostic_lock_sha256"] == sha(diagnostic / "DIAGNOSTIC_LOCK.json")
    for name, digest in diagnostic_lock["source_sha256"].items():
        assert sha(root / name) == digest
    source_reference = pd.read_csv(diagnostic / "train_invariant_summaries.csv", float_precision="round_trip")
    target_reference = pd.read_csv(diagnostic / "external_invariant_summaries.csv", float_precision="round_trip")
    feature_names = [n for n in source_reference if n.startswith(("left_", "right_"))]
    raw = pd.read_csv(args.raw)
    raws = {r: role_frame(raw, split, r) for r in ("train", "validation")}
    del raw
    windows = {r: anchored_gyro_windows(f, require_right=True) for r, f in raws.items()}
    expected = {r: inventory(f).loc[lambda x: x.right_complete].reset_index(drop=True) for r, f in raws.items()}
    independent_val = np.array([direct_descriptors(w.samples) for w in windows["validation"]])
    np.testing.assert_allclose(feature_matrix(windows["validation"], "invariant_rf")[0], independent_val, atol=1e-8, rtol=1e-10)
    np.testing.assert_array_equal(source_reference.source_person, expected["train"].subject_id)
    np.testing.assert_allclose(source_reference.truth_fog_fraction, expected["train"].truth_fog_fraction, atol=1e-12)
    groups, y = expected["train"].subject_id.to_numpy(int), expected["train"].truth_fog_label.to_numpy(int)
    plans = json.loads((run / "fold_plan.json").read_text("utf-8"))
    reference = Path(config["reference_run"])
    assert plans == json.loads((reference / "fold_plan.json").read_text("utf-8"))
    assert split == json.loads((reference / "split.json").read_text("utf-8"))
    models, frames = {}, {}
    checked = 0
    for name in config["models"]:
        values, names = feature_matrix(windows["train"], name)
        np.testing.assert_allclose(values, np.load(run / name / "train_features.npy"), atol=1e-10, rtol=1e-10)
        if name == "invariant_rf":
            np.testing.assert_allclose(values, source_reference[feature_names], atol=1e-8, rtol=1e-10)
        oof = pd.read_csv(run / name / "oof_predictions.csv", float_precision="round_trip")
        val = pd.read_csv(output / "validation" / name / "predictions.csv", float_precision="round_trip")
        checked += check_frame(oof, expected["train"]) + check_frame(val, expected["validation"])
        metadata = json.loads((run / name / "fit_metadata.json").read_text("utf-8"))
        seen = np.zeros(len(groups), int)
        for plan, saved in zip(plans, metadata["folds"], strict=True):
            fit, held = [np.flatnonzero(np.isin(groups, plan[k])) for k in ("fit_people", "held_out_people")]
            assert not set(plan["fit_people"]) & set(plan["held_out_people"])
            assert set(plan["fit_people"]) | set(plan["held_out_people"]) == set(split["train"])
            assert saved["actual_fit_people"] == sorted(map(int, set(groups[fit])))
            forest = joblib.load(run / name / f"fold_{plan['fold']}.joblib")
            np.testing.assert_allclose(forest.predict_proba(values[held])[:, 1], oof.iloc[held].fog_probability, atol=1e-7, rtol=0)
            seen[held] += 1
        assert np.all(seen == 1)
        model = InvariantFoG.load(run / name / "model.joblib")
        assert model.feature_names == names and model.fit_participants == model.threshold_participants == tuple(sorted(map(int, set(groups))))
        assert not metadata["validation_people_used"] and not metadata["test_people_used"]
        for key, value in {**config["rf"], "random_state": config["random_state"], "max_features": "sqrt"}.items():
            assert model.classifier.get_params()[key] == value
        choice = choose_threshold(y, oof.fog_probability.to_numpy(), config)
        np.testing.assert_allclose(model.threshold, choice["selected"]["threshold"], atol=1e-12)
        np.testing.assert_allclose(model.predict_windows(windows["validation"]).fog_probability, val.fog_probability, atol=1e-7, rtol=0)
        sensor = raws["validation"].iloc[:180].drop(columns=["fog", "activity", "fog_severity"])
        pd.testing.assert_frame_equal(model.predict_frame(sensor), model.predict_frame(sensor.assign(fog=1, activity=6)))
        if name == "axis_control":
            for current, path in ((oof, reference / "bilateral_rf/oof_predictions.csv"), (val, reference / "evaluation/bilateral_rf/predictions.csv")):
                old = pd.read_csv(path, float_precision="round_trip")
                np.testing.assert_allclose(current.fog_probability, old.fog_probability, atol=1e-12, rtol=0)
                np.testing.assert_array_equal(current.is_fog, old.is_fog)
        models[name], frames[name] = model, val
    for row in pd.read_csv(output / "validation_comparison.csv").itertuples():
        frame = frames[row.model]
        for field, value in {**metrics(frame), **EventSupport.build(raws["validation"], frame, allowed=split["validation"]).metrics(frame.is_fog)}.items():
            if value is not None:
                np.testing.assert_allclose(getattr(row, field), value, atol=1e-6, rtol=0)
    external = Path(config["external_reference"])
    assert started["external_input_lock_sha256"] == sha(external / "INPUT_LOCK.json")
    assert started["external_evaluation_sha256"] == sha(external / "evaluation/EVALUATION_COMPLETED.json")
    for folder, file in ((external, "INPUT_LOCK.json"), (external / "evaluation", "EVALUATION_COMPLETED.json")):
        check_lock(folder, json.loads((folder / file).read_text("utf-8")))
    old = pd.read_csv(external / "evaluation/predictions.csv", float_precision="round_trip")
    replay = json.loads((external / "evaluation/replay_data.json").read_text("utf-8"))
    native_checked = 0
    for name in config["models"]:
        folder = output / "external" / name
        frame = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
        for field in ("subject_id", "session_id", "task_id", "record_id", "window_start_s", "window_end_s", "available_at_s", "truth_fog_label", "truth_fog_fraction"):
            np.testing.assert_array_equal(frame[field], old[field])
        scores = old.fog_probability.to_numpy() if name == "axis_control" else models[name].classifier.predict_proba(target_reference[feature_names].to_numpy())[:, 1]
        np.testing.assert_allclose(frame.fog_probability, scores, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(frame.is_fog, frame.fog_probability >= models[name].threshold)
        records, events = pd.read_csv(folder / "by_record.csv"), pd.read_csv(folder / "events.csv")
        for row in records.itertuples():
            part = frame.loc[frame.record_id.eq(row.record_id)]
            data = np.load(external / f"inputs/record_{row.record_id:03}.npz")
            t = data["native_time"]
            truth = next(r for r in replay if r["record_id"] == row.record_id)["intervals"]["truth"]
            labels = np.array([any(a - 1e-9 <= x < b - 1e-9 for a, b in truth) for x in t], int)
            start = part.available_at_s.to_numpy()
            stop = np.minimum(start + 0.5, np.r_[start[1:], t[-1] + 1 / 128])
            truth_duration = integral(t, labels, stop) - integral(t, labels, start)
            support = integral(t, np.ones(len(t)), stop) - integral(t, np.ones(len(t)), start)
            alarm = part.is_fog.to_numpy(bool)
            np.testing.assert_allclose([row.truth_s, row.support_s, row.true_alarm_s, row.false_alarm_s],
                [labels.sum() / 128, support.sum(), truth_duration[alarm].sum(), (support - truth_duration)[alarm].sum()], atol=1e-6, rtol=0)
            record_events = events.loc[events.record_id.eq(row.record_id)]
            assert len(record_events) == len(truth)
            for event, (a, b) in zip(record_events.itertuples(), truth, strict=True):
                overlaps = np.maximum(0, np.minimum(stop, b) - np.maximum(start, a))
                np.testing.assert_allclose([event.start_s, event.end_s, event.alarm_s], [a, b, overlaps[alarm].sum()], atol=1e-6, rtol=0)
                assert bool(event.covered) == bool(np.any(overlaps[alarm] > 0))
                native_checked += 1
        summary = pd.read_csv(output / "external_comparison.csv").set_index("model").loc[name]
        for field, value in metrics(frame).items():
            if value is not None:
                np.testing.assert_allclose(summary[field], value, atol=1e-10)
        for field in ("observed_s", "truth_s", "support_s", "truth_events", "covered_raw_events", "true_alarm_s", "false_alarm_s"):
            np.testing.assert_allclose(summary[field], records[field].sum(), atol=1e-6)
        if name == "axis_control":
            original = json.loads((external / "evaluation/summary.json").read_text("utf-8"))
            for field, value in original["native128_time_metrics"].items():
                np.testing.assert_allclose(summary[field], value, atol=1e-6)
    for file in output.rglob("runtime.csv"):
        assert not any(c.startswith("truth_") for c in pd.read_csv(file, nrows=0).columns)
    table = pd.read_csv(output / "validation_comparison.csv").set_index("model")
    a, b = table.loc["axis_control"], table.loc["invariant_rf"]
    conditions = {"matched_f1_higher": bool(b.fog_f1 > a.fog_f1), "false_alarm_seconds_not_higher": bool(b.false_alarm_s <= a.false_alarm_s + 1e-6),
        "fog_time_recall_not_lower": bool(b.time_recall + 1e-9 >= a.time_recall), "decision_support_not_lost": bool(b.decision_support_fraction + 1e-9 >= a.decision_support_fraction)}
    gate = json.loads((output / "development_gate.json").read_text("utf-8"))
    assert gate["conditions"] == conditions and gate["passed"] == all(conditions.values())
    write_json(run / "verification.json", {"status": "PASS", "source_prediction_rows_checked": checked,
        "fold_models_rescored": 6, "independent_validation_feature_rows": len(independent_val),
        "reused_independently_verified_train_feature_rows": len(source_reference), "reused_independently_verified_external_feature_rows": len(target_reference),
        "native_event_rows_independently_checked": native_checked, "external_prediction_rows_checked": 2 * len(old),
        "external_candidate_rows_rescored": len(old), "external_control_scores_compared_to_original_lock": len(old),
        "old_66d_control_reproduced": True, "test_windows_constructed": 0, "independent_retraining": False,
        "model_lock_sha256": sha(run / "MODEL_LOCK.json"), "evaluation_lock_sha256": sha(output / "EVALUATION_COMPLETED.json"),
        "diagnostic_reference_lock_sha256": sha(diagnostic / "DIAGNOSTIC_LOCK.json"),
        "verifier_sha256": {n: sha(root / "scripts" / n) for n in ("verify_a1_invariant.py", "verify_stanford_failure.py", "verify_stanford_external.py")}})
    print("PASS: independent VAL formulas, verified TRAIN/external references, six folds, thresholds, control, native time/events, truth-free runtime", flush=True)


if __name__ == "__main__":
    main()

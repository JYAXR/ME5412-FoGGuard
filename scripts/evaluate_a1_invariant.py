"""Locked source VALIDATION and already-inspected external development recheck; no fit."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from report_a1_author_cnn import event_summary
from run_a1_experiment import sha, write_json

from fogguard.data.stanford_external import DELAY_S
from fogguard.evaluation.a1_author import annotate, grouped_diagnostics, metrics
from fogguard.evaluation.a1_fixed_split import role_frame
from fogguard.evaluation.stanford_external import aggregate, record_metrics
from fogguard.modeling.a1_bilateral import DUAL, anchored_gyro_windows
from fogguard.modeling.a1_invariant import InvariantFoG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    args = parser.parse_args()
    run, root = args.run_dir, Path(__file__).resolve().parents[1]
    output = run / "evaluation"
    if output.exists():
        parser.error("Use a run without prior evaluation")
    config, split, lock, protocol = [json.loads((run / n).read_text("utf-8")) for n in ("config.json", "split.json", "MODEL_LOCK.json", "protocol.json")]
    check_lock(run, lock)
    if sha(args.raw) != protocol["raw_sha256"]:
        raise ValueError("Raw data changed")
    for path, digest in protocol["source_sha256"].items():
        if sha(root / path) != digest:
            raise ValueError(f"Source changed: {path}")
    reference, external = Path(config["reference_run"]), Path(config["external_reference"])
    for folder, filename in ((reference / "evaluation", "EVALUATION_COMPLETED.json"), (external, "INPUT_LOCK.json"), (external / "evaluation", "EVALUATION_COMPLETED.json")):
        check_lock(folder, json.loads((folder / filename).read_text("utf-8")))
    output.mkdir()
    write_json(output / "EVALUATION_STARTED.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
        "model_lock_sha256": sha(run / "MODEL_LOCK.json"), "external_already_inspected": True,
        "external_input_lock_sha256": sha(external / "INPUT_LOCK.json"),
        "external_evaluation_sha256": sha(external / "evaluation/EVALUATION_COMPLETED.json")})
    raw = pd.read_csv(args.raw)
    validation = role_frame(raw, split, "validation")
    del raw
    windows = anchored_gyro_windows(validation, require_right=True)
    models = {name: InvariantFoG.load(run / name / "model.joblib") for name in config["models"]}
    rows = []
    for name, model in models.items():
        frame = annotate(windows, model.predict_windows(windows), "validation")
        if name == "axis_control":
            old = pd.read_csv(reference / "evaluation/bilateral_rf/predictions.csv", float_precision="round_trip")
            np.testing.assert_allclose(frame.fog_probability, old.fog_probability, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(frame.is_fog, old.is_fog)
        dest = output / "validation" / name
        dest.mkdir(parents=True)
        frame.to_csv(dest / "predictions.csv", index=False)
        frame.drop(columns=[c for c in frame if c.startswith("truth_")]).to_csv(dest / "runtime.csv", index=False)
        for field in ("subject_id", "truth_activity", "task_id"):
            grouped_diagnostics(frame, field).to_csv(dest / f"by_{field}.csv", index=False)
        rows.append({"model": name, "scope": "source_validation_development", **metrics(frame), **event_summary(validation, frame, dest, root)})
    table = pd.DataFrame(rows)
    table.to_csv(output / "validation_comparison.csv", index=False)
    own = table.set_index("model")
    base, candidate = own.loc["axis_control"], own.loc["invariant_rf"]
    conditions = {"matched_f1_higher": bool(candidate.fog_f1 > base.fog_f1),
        "false_alarm_seconds_not_higher": bool(candidate.false_alarm_s <= base.false_alarm_s + 1e-6),
        "fog_time_recall_not_lower": bool(candidate.time_recall + 1e-9 >= base.time_recall),
        "decision_support_not_lost": bool(candidate.decision_support_fraction + 1e-9 >= base.decision_support_fraction)}
    write_json(output / "development_gate.json", {"conditions": conditions, "passed": all(conditions.values()), "default_replaced": False})
    print(table[["model", "precision", "recall", "fog_f1", "false_alarm_s", "time_recall"]].to_string(index=False), flush=True)
    old_predictions = pd.read_csv(external / "evaluation/predictions.csv", float_precision="round_trip")
    replay = json.loads((external / "evaluation/replay_data.json").read_text("utf-8"))
    external_rows = []
    for name in config["models"]:
        dest = output / "external" / name
        dest.mkdir(parents=True)
        frames, records, events = [], [], []
        for item in pd.read_csv(external / "input_inventory.csv").itertuples():
            data = np.load(external / f"inputs/record_{item.record_id:03}.npz")
            saved = old_predictions.loc[old_predictions.record_id.eq(item.record_id)].reset_index(drop=True)
            if name == "axis_control":
                frame = saved.copy()  # Original frozen 66D scores; source control was reproduced above.
            else:
                sensors = pd.DataFrame(data["gyro"], columns=DUAL)
                sensors.insert(0, "timestamp", data["logical_time"])
                for field, value in (("subjectID", item.subject_id), ("sessionID", item.session_id), ("taskID", item.task_id)):
                    sensors[field] = value
                target_windows = anchored_gyro_windows(sensors, require_right=True)
                frame = models[name].predict_windows(target_windows)
                np.testing.assert_allclose(frame.window_end_s, saved.window_end_s, atol=1e-8, rtol=0)
                frame["available_at_s"] = frame.window_end_s + DELAY_S
                for field in ("truth_fog_fraction", "truth_fog_label"):
                    frame[field] = saved[field].to_numpy()
                frame["record_id"] = item.record_id
                frame["prediction_source"] = "train_only_model_inspected_stanford_development_recheck"
            record = next(r for r in replay if r["record_id"] == item.record_id)
            t = data["native_time"]
            labels = np.zeros(len(t), int)
            for a, b in record["intervals"]["truth"]:
                labels[(t >= a - 1e-9) & (t < b - 1e-9)] = 1
            native, native_events, _ = record_metrics(t, labels, frame)
            records.append({"record_id": item.record_id, "source_person": item.source_person, **native, **metrics(frame)})
            events.extend({"record_id": item.record_id, "source_person": item.source_person, **e} for e in native_events)
            frame["source_person"] = item.source_person
            frames.append(frame)
        frame, records = pd.concat(frames, ignore_index=True), pd.DataFrame(records)
        frame.to_csv(dest / "predictions.csv", index=False)
        frame.drop(columns=[c for c in frame if c.startswith("truth_")]).to_csv(dest / "runtime.csv", index=False)
        records.to_csv(dest / "by_record.csv", index=False)
        pd.DataFrame(events).to_csv(dest / "events.csv", index=False)
        people = []
        for p, group in records.groupby("source_person", sort=True):
            people.append({"source_person": int(p), **aggregate(group.to_dict("records")), **metrics(frame.loc[frame.source_person.eq(p)])})
        pd.DataFrame(people).to_csv(dest / "by_person.csv", index=False)
        external_rows.append({"model": name, "scope": "inspected_stanford_development_recheck", **metrics(frame), **aggregate(records.to_dict("records"))})
        print(f"{name}: external development recheck complete ({len(frame)} windows / 60 records)", flush=True)
    pd.DataFrame(external_rows).to_csv(output / "external_comparison.csv", index=False)
    check_lock(run, lock)
    write_json(output / "EVALUATION_COMPLETED.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_lock_sha256": sha(run / "MODEL_LOCK.json"), "test_windows_constructed": 0, "target_refits_or_calibrations": 0,
        "external_scope": "Already inspected development recheck, not independent validation",
        "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print(pd.DataFrame(external_rows)[["model", "precision", "recall", "fog_f1", "covered_raw_events", "false_alarm_s", "time_recall"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

"""Once-only frozen-model zero-shot scoring of all locked Stanford records."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json

from fogguard.data.stanford_external import DELAY_S, canonical_clock, project_labels, read_workbook
from fogguard.evaluation.a1_author import metrics
from fogguard.evaluation.stanford_external import aggregate, record_metrics
from fogguard.modeling.a1_bilateral import DUAL, anchored_gyro_windows
from fogguard.modeling.a1_coordination import CoordinationFoG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/stanford_external_v1_20261029"))
    args = parser.parse_args()
    run, root = args.run_dir, Path(__file__).resolve().parents[1]
    output = run / "evaluation"
    if output.exists():
        parser.error("Use a run without prior external scores")
    config, protocol, lock = [json.loads((run / n).read_text("utf-8")) for n in ("config.json", "protocol.json", "INPUT_LOCK.json")]
    check_lock(run, lock)
    for path, digest in protocol["source_sha256"].items():
        if sha(root / path) != digest:
            raise ValueError(f"Source changed: {path}")
    if sha(config["model"]) != config["model_sha256"]:
        raise ValueError("Frozen model changed")
    model = CoordinationFoG.load(Path(config["model"]))
    if model.variant != "bilateral_control" or not np.isclose(model.threshold, 0.23, atol=1e-12):
        raise ValueError("Unexpected primary model or threshold")
    output.mkdir()
    (output / "records").mkdir()
    write_json(output / "EVALUATION_STARTED.json", {"started_utc": datetime.now(timezone.utc).isoformat(), "input_lock_sha256": sha(run / "INPUT_LOCK.json"), "target_labels_first_decoded_in_this_phase": True})
    inventory = pd.read_csv(run / "input_inventory.csv")
    all_predictions, rows, all_events, replay = [], [], [], []
    for item in inventory.itertuples():
        data = np.load(run / "inputs" / f"record_{item.record_id:03}.npz")
        sensors = pd.DataFrame(data["gyro"], columns=DUAL)
        sensors.insert(0, "timestamp", data["logical_time"])
        ids = {"subjectID": item.subject_id, "sessionID": item.session_id, "taskID": item.task_id}
        for key, value in ids.items():
            sensors[key] = value
        windows = anchored_gyro_windows(sensors, require_right=True) if len(sensors) else []
        if windows:
            predictions = model.predict_windows(windows)  # sensor-only; target labels not loaded yet.
            predictions["available_at_s"] = predictions.window_end_s + DELAY_S
            predictions["resampling_delay_s"] = DELAY_S
            predictions["prediction_source"] = "frozen_fogstar_model_zero_shot_stanford_unaligned_axes"
        else:
            predictions = pd.DataFrame(columns=["subject_id", "session_id", "task_id", "window_start_s", "window_end_s", "available_at_s", "fog_probability", "fog_threshold", "is_fog", "resampling_delay_s"])
        truth = read_workbook(args.source_dir / item.filename, labels=True)
        native_time, _ = canonical_clock(truth["time"])
        np.testing.assert_array_equal(native_time, data["native_time"])
        if not np.all(truth["subject_ID"] == item.source_person) or not np.isin(truth["freeze_label"], [0, 1]).all():
            raise ValueError("Invalid external identity/labels")
        fractions = [project_labels(native_time, truth["freeze_label"], w.start_s + np.arange(120) / 60).mean() for w in windows]
        predictions["truth_fog_fraction"] = fractions
        predictions["truth_fog_label"] = np.asarray(fractions) >= 0.5
        predictions["record_id"] = item.record_id
        predictions.to_csv(output / "records" / f"record_{item.record_id:03}_predictions.csv", index=False)
        predictions.drop(columns=[c for c in predictions if c.startswith("truth_")]).to_csv(output / "records" / f"record_{item.record_id:03}_runtime.csv", index=False)
        native, events, intervals = record_metrics(native_time, truth["freeze_label"], predictions)
        rows.append({"record_id": item.record_id, "source_person": item.source_person, "filename": item.filename,
            "windows": len(predictions), **native, **(metrics(predictions) if len(predictions) else {})})
        all_events.extend({"record_id": item.record_id, "source_person": item.source_person, **event} for event in events)
        all_predictions.append(predictions)
        points = sensors.iloc[::4]
        mags = [np.sqrt(np.square(points[list(DUAL[a:a + 3])].to_numpy(float)).sum(axis=1)) for a in (0, 3)]
        replay.append({"record_id": item.record_id, "source_person": item.source_person, "filename": item.filename,
            "native_start_s": float(native_time[0]), "native_stop_s": float(native_time[-1] + 1 / 128), "intervals": intervals,
            "decisions": [[float(p.available_at_s), min(float(p.available_at_s + 0.5), float(native_time[-1] + 1 / 128)), float(p.fog_probability)] for p in predictions.itertuples()],
            "signals": [[round(float(t), 5), *[round(float(m[i]), 3) if np.isfinite(m[i]) else None for m in mags]] for i, t in enumerate(points.timestamp)]})
        print(f"External record {item.record_id}/{len(inventory)} evaluated (no fit)", flush=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    predictions.drop(columns=[c for c in predictions if c.startswith("truth_")]).to_csv(output / "runtime.csv", index=False)
    records = pd.DataFrame(rows)
    records.to_csv(output / "by_record.csv", index=False)
    pd.DataFrame(all_events).to_csv(output / "events.csv", index=False)
    people = []
    for person, blocks in records.groupby("source_person", sort=True):
        p = predictions.loc[predictions.subject_id.astype(int).eq(1000 + person)]
        people.append({"source_person": int(person), "records": len(blocks), **aggregate(blocks.to_dict("records")), **(metrics(p) if len(p) else {"windows": 0})})
    pd.DataFrame(people).to_csv(output / "by_person.csv", index=False)
    summary = {"scope": "Unit/rate-adapted zero-shot domain transfer; anatomical axes UNVERIFIED", "axis_alignment_verified": False,
        "source_records": len(records), "source_people": len(people), "records_with_predictions": int((records.windows > 0).sum()),
        "window_metrics": metrics(predictions), "native128_time_metrics": aggregate(rows), "resampling_delay_s": DELAY_S,
        "model_threshold": model.threshold, "model_sha256": sha(config["model"]), "fitted_or_calibrated_on_external": False}
    write_json(output / "summary.json", summary)
    write_json(output / "replay_data.json", replay)
    check_lock(run, lock)
    write_json(output / "EVALUATION_COMPLETED.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_lock_sha256": sha(run / "INPUT_LOCK.json"), "model_sha256": sha(config["model"]),
        "external_refits": 0, "threshold_changes": 0,
        "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

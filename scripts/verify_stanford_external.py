"""Verify immutable sources, causal FIR samples, frozen scores and native-time integrals."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json

from fogguard.data.stanford_external import (
    DELAY_S,
    FILTER,
    SOURCE_GYRO,
    canonical_clock,
    read_workbook,
)
from fogguard.modeling.a1_bilateral import DUAL, anchored_gyro_windows
from fogguard.modeling.a1_coordination import CoordinationFoG


def integral(time, weights, at):
    """Independent piecewise-native-cell antiderivative, including observation gaps."""
    at = np.asarray(at, float)
    i = np.searchsorted(time, at, side="right") - 1
    safe = np.maximum(i, 0)
    before = np.r_[0, np.cumsum(weights / 128)]
    result = before[safe] + weights[safe] * np.clip(at - time[safe], 0, 1 / 128)
    return np.where(i >= 0, result, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/stanford_external_v1_20261029"))
    args = parser.parse_args()
    run, root = args.run_dir, Path(__file__).resolve().parents[1]
    config, protocol, lock = [json.loads((run / n).read_text("utf-8")) for n in ("config.json", "protocol.json", "INPUT_LOCK.json")]
    complete, started = [json.loads((run / "evaluation" / n).read_text("utf-8")) for n in ("EVALUATION_COMPLETED.json", "EVALUATION_STARTED.json")]
    check_lock(run, lock)
    check_lock(run / "evaluation", complete)
    if datetime.fromisoformat(lock["locked_utc"]) >= datetime.fromisoformat(started["started_utc"]):
        raise ValueError("External labels/scores before input lock")
    if complete["input_lock_sha256"] != sha(run / "INPUT_LOCK.json") or sha(config["model"]) != config["model_sha256"]:
        raise ValueError("Model or input lock changed")
    for path, digest in protocol["source_sha256"].items():
        if sha(root / path) != digest:
            raise ValueError("Source changed")
    model = CoordinationFoG.load(Path(config["model"]))
    np.testing.assert_allclose(model.threshold, 0.23, atol=1e-12)
    inventory = pd.read_csv(run / "input_inventory.csv")
    records = pd.read_csv(run / "evaluation/by_record.csv")
    events = pd.read_csv(run / "evaluation/events.csv")
    window_count, fir_rows = 0, 0
    for item in inventory.itertuples():
        path = args.source_dir / item.filename
        if sha(path) != item.source_sha256:
            raise ValueError("Workbook SHA changed")
        raw = read_workbook(path, labels="both")
        time, blocks = canonical_clock(raw["time"])
        saved = np.load(run / "inputs" / f"record_{item.record_id:03}.npz")
        np.testing.assert_array_equal(time, saved["native_time"])
        gyro = np.column_stack([raw[c] for c in SOURCE_GYRO])
        logical = saved["logical_time"]
        for a, b in blocks:
            candidates = np.flatnonzero((logical >= time[a] + DELAY_S - 1e-9) & (logical + DELAY_S <= time[b - 1] + 1e-9))
            for index in candidates[np.unique(np.linspace(0, len(candidates) - 1, min(5, len(candidates))).astype(int))] if len(candidates) else []:
                k = round((logical[index] - time[a] + DELAY_S) * 60)
                n = k * 32
                j = np.arange(max(0, (n - 640 + 14) // 15), n // 15 + 1)
                if len(j) and a + j[-1] >= b:
                    raise ValueError("FIR used future tail")
                expected = FILTER[n - j * 15] @ np.rad2deg(gyro[a + j])
                np.testing.assert_allclose(expected, saved["gyro"][index], atol=1e-8, rtol=1e-10, equal_nan=True)
                fir_rows += 1
        sensors = pd.DataFrame(saved["gyro"], columns=DUAL)
        sensors.insert(0, "timestamp", logical)
        for name, value in zip(("subjectID", "sessionID", "taskID"), (item.subject_id, item.session_id, item.task_id), strict=True):
            sensors[name] = value
        windows = anchored_gyro_windows(sensors, require_right=True) if len(sensors) else []
        pred = pd.read_csv(run / "evaluation/records" / f"record_{item.record_id:03}_predictions.csv", float_precision="round_trip")
        window_count += len(pred)
        if len(windows) != len(pred):
            raise ValueError("Window support differs")
        if windows:
            np.testing.assert_allclose(model.predict_windows(windows).fog_probability, pred.fog_probability, atol=1e-7, rtol=0)
            np.testing.assert_allclose(pred.available_at_s - pred.window_end_s, DELAY_S, atol=1e-10)
            for w, p in zip(windows, pred.itertuples(), strict=True):
                queries = w.start_s + np.arange(120) / 60
                owner = np.searchsorted(time, queries + 1e-9, side="right") - 1
                fraction = raw["freeze_label"][owner].mean()
                np.testing.assert_allclose(fraction, p.truth_fog_fraction, atol=1e-12)
                if int(fraction >= 0.5) != p.truth_fog_label or bool(p.fog_probability >= model.threshold) != p.is_fog:
                    raise ValueError("Window truth or decision differs")
        runtime = pd.read_csv(run / "evaluation/records" / f"record_{item.record_id:03}_runtime.csv")
        if any(c.startswith("truth_") for c in runtime):
            raise ValueError("Target truth in runtime")
        start = pred.available_at_s.to_numpy(float)
        stop = np.minimum(start + 0.5, np.r_[start[1:], time[-1] + 1 / 128]) if len(start) else np.array([])
        active = pred.is_fog.to_numpy(bool)
        positive = integral(time, raw["freeze_label"], stop) - integral(time, raw["freeze_label"], start)
        duration = integral(time, np.ones(len(time)), stop) - integral(time, np.ones(len(time)), start)
        row = records.loc[records.record_id.eq(item.record_id)].iloc[0]
        np.testing.assert_allclose([row.true_alarm_s, row.false_alarm_s, row.support_s, row.truth_s],
            [positive[active].sum(), (duration[active] - positive[active]).sum(), duration.sum(), raw["freeze_label"].sum() / 128], atol=1e-6, rtol=0)
        yes = np.flatnonzero(raw["freeze_label"])
        pieces = np.split(yes, np.flatnonzero((np.diff(yes) > 1) | (np.diff(time[yes]) > 1.5 / 128)) + 1) if len(yes) else []
        record_events = events.loc[events.record_id.eq(item.record_id)]
        if len(pieces) != len(record_events):
            raise ValueError("Native raw event omitted")
        for cells, event in zip(pieces, record_events.itertuples(), strict=True):
            a, b = time[cells[0]], time[cells[-1]] + 1 / 128
            covered = np.maximum(0, np.minimum(stop, b) - np.maximum(start, a))
            np.testing.assert_allclose([event.start_s, event.end_s, event.alarm_s, event.support_s], [a, b, covered[active].sum(), covered.sum()], atol=1e-6, rtol=0)
    summary = json.loads((run / "evaluation/summary.json").read_text("utf-8"))
    from fogguard.evaluation.a1_author import metrics

    all_predictions = pd.read_csv(run / "evaluation/predictions.csv", float_precision="round_trip")
    for field, value in metrics(all_predictions).items():
        if value is not None:
            np.testing.assert_allclose(summary["window_metrics"][field], value, atol=1e-10)
    for field in ("observed_s", "truth_s", "support_s", "true_alarm_s", "false_alarm_s", "truth_events", "covered_raw_events", "supported_events"):
        np.testing.assert_allclose(summary["native128_time_metrics"][field], records[field].sum(), atol=1e-6)
    if len(inventory) != 60 or len(records) != 60 or inventory.source_person.nunique() != 7:
        raise ValueError("Frozen dataset denominator changed")
    result = {"status": "PASS", "source_records_sha_verified": len(inventory), "source_people": 7,
        "window_rows_rescored": window_count, "direct_fir_rows_checked": fir_rows, "native_events_checked": len(events),
        "independent_native_antiderivative_and_window_truth": "PASS", "delay_and_runtime_truth_separation": "PASS",
        "input_files": len(lock["files"]), "evaluation_files": len(complete["files"]), "external_refits": 0,
        "axis_alignment_verified": False, "input_lock_sha256": sha(run / "INPUT_LOCK.json"), "model_sha256": sha(config["model"])}
    write_json(run / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

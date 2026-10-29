"""Independently map decision ownership and count continuous raw disagreement cells."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("reports/a1_coordination_v1_20261029"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    args = parser.parse_args()
    audit, run, root = args.audit_dir, args.run_dir, Path(__file__).resolve().parents[1]
    complete = json.loads((audit / "AUDIT_COMPLETED.json").read_text("utf-8"))
    check_lock(audit, complete)
    if sha(args.raw) != complete["raw_sha256"] or sha(run / "MODEL_LOCK.json") != complete["source_model_lock_sha256"] or sha(run / "evaluation/EVALUATION_COMPLETED.json") != complete["source_evaluation_sha256"]:
        raise ValueError("Source hashes changed")
    for name, digest in complete["source_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError("Audit source changed")
    raw = pd.read_csv(args.raw)
    allowed = json.loads((run / "split.json").read_text("utf-8"))["validation"]
    raw = raw.loc[raw.subjectID.isin(allowed)]
    frames = [pd.read_csv(run / "evaluation" / name / "predictions.csv", float_precision="round_trip") for name in ("bilateral_control", "coordination_rf")]
    events = pd.read_csv(audit / "events.csv")
    records = pd.read_csv(audit / "records.csv")
    summary = json.loads((audit / "summary.json").read_text("utf-8"))
    payload = json.loads((audit / "replay_data.json").read_text("utf-8"))
    count = 0
    totals = {k: 0.0 for k in ("base_true_s", "candidate_true_s", "base_false_s", "candidate_false_s", "lost_fog_s", "recovered_fog_s", "removed_false_alarm_s", "new_false_alarm_s")}
    partition = {k: 0.0 for k in ("onset", "interior", "ending")}
    expected_pieces = []
    for key, signal in raw.groupby(["subjectID", "sessionID", "taskID"], sort=False):
        time, truth = signal.timestamp.to_numpy(float), signal.fog.to_numpy(bool)
        np.testing.assert_allclose(np.diff(time), 1 / 60, atol=1e-6, rtol=0)
        states = []
        for frame in frames:
            block = frame.loc[np.all(frame[["subject_id", "session_id", "task_id"]].to_numpy(int) == key, axis=1)]
            ends = block.window_end_s.to_numpy(float)
            owner = np.searchsorted(ends, time + 1e-6, side="right") - 1
            state = np.full(len(time), -1, int)
            if len(ends):
                safe = np.maximum(owner, 0)
                valid = (owner >= 0) & (time - ends[safe] < 0.5 - 1e-6)
                state[valid] = block.is_fog.to_numpy(int)[safe[valid]]
            states.append(state)
        a, b = states
        masks = {"lost_fog": truth & (a == 1) & (b == 0), "recovered_fog": truth & (a == 0) & (b == 1),
                 "removed_false_alarm": ~truth & (a == 1) & (b == 0), "new_false_alarm": ~truth & (a == 0) & (b == 1)}
        for kind, mask in masks.items():
            active = np.flatnonzero(mask)
            groups = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1) if len(active) else []
            for cells in groups:
                expected_pieces.append((*map(int, key), kind, time[cells[0]], time[cells[-1]] + 1 / 60, len(cells) / 60))
        expected = {"base_true_s": np.sum(truth & (a == 1)) / 60, "candidate_true_s": np.sum(truth & (b == 1)) / 60,
            "base_false_s": np.sum(~truth & (a == 1)) / 60, "candidate_false_s": np.sum(~truth & (b == 1)) / 60,
            "lost_fog_s": np.sum(truth & (a == 1) & (b == 0)) / 60, "recovered_fog_s": np.sum(truth & (a == 0) & (b == 1)) / 60,
            "removed_false_alarm_s": np.sum(~truth & (a == 1) & (b == 0)) / 60, "new_false_alarm_s": np.sum(~truth & (a == 0) & (b == 1)) / 60}
        rec = records.loc[np.all(records[["subject_id", "session_id", "task_id"]].to_numpy(int) == key, axis=1)]
        if len(rec) != 1:
            raise ValueError("Raw record omitted or duplicated")
        for name, value in expected.items():
            np.testing.assert_allclose(rec.iloc[0][name], value, atol=1e-6, rtol=0)
            totals[name] += value
        indices = np.flatnonzero(truth)
        runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(indices) else []
        saved = events.loc[np.all(events[["subject_id", "session_id", "task_id"]].to_numpy(int) == key, axis=1)]
        if len(saved) != len(runs):
            raise ValueError("Raw events omitted")
        for cells, row in zip(runs, saved.itertuples(), strict=True):
            count += 1
            np.testing.assert_allclose([row.start_s, row.end_s], [time[cells[0]], time[cells[-1]] + 1 / 60], atol=1e-6, rtol=0)
            for name, state in zip(("base", "candidate"), states, strict=True):
                alarm = state[cells] == 1
                np.testing.assert_allclose(getattr(row, name + "_alarm_s"), alarm.sum() / 60, atol=1e-6, rtol=0)
                missing = np.flatnonzero(~alarm)
                pieces = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1) if len(missing) else []
                np.testing.assert_allclose(getattr(row, name + "_longest_unalarmed_s"), max(map(len, pieces), default=0) / 60, atol=1e-6, rtol=0)
            onset = time[cells] < row.start_s + min(3, row.duration_s) - 1e-6
            ending = ~onset & (time[cells] >= max(row.end_s - 3, row.start_s) - 1e-6)
            for zone, mask in (("onset", onset), ("ending", ending), ("interior", ~onset & ~ending)):
                loss = np.sum(mask & (a[cells] == 1) & (b[cells] == 0)) / 60
                np.testing.assert_allclose(getattr(row, zone + "_lost_s"), loss, atol=1e-6, rtol=0)
                partition[zone] += loss
    for name, value in totals.items():
        np.testing.assert_allclose(summary[name], value, atol=1e-6, rtol=0)
    for name, value in partition.items():
        np.testing.assert_allclose(summary["lost_partition_s"][name], value, atol=1e-6, rtol=0)
    if count != len(events) or len(payload["records"]) != len(records) or len(records) != summary["raw_records"]:
        raise ValueError("Replay/data denominator differs")
    pieces = pd.read_csv(audit / "disagreement_segments.csv")
    expected = pd.DataFrame(expected_pieces, columns=["subject_id", "session_id", "task_id", "kind", "start_s", "end_s", "duration_s"])
    if list(expected.kind) != list(pieces.kind):
        raise ValueError("Disagreement segment kind/order differs")
    numeric = [c for c in expected if c != "kind"]
    np.testing.assert_allclose(expected[numeric], pieces[numeric], atol=1e-6, rtol=0)
    cases, selected, used = json.loads((audit / "selected_cases.json").read_text("utf-8")), [], set()
    keys = ["subject_id", "session_id", "task_id", "start_s", "end_s"]
    for label, field in (("largest_fog_time_loss", "alarm_s"), ("largest_other_recall_drop", "recall")):
        ranked = events.assign(delta=events["base_" + field] - events["candidate_" + field]).sort_values(["delta", *keys[:-1]], ascending=[False, True, True, True, True])
        row = next(r for r in ranked.to_dict("records") if tuple(r[k] for k in keys[:-1]) not in used and r["delta"] > 1e-6)
        used.add(tuple(row[k] for k in keys[:-1]))
        selected.append({"case": label, **{k: row[k] for k in keys}})
    for kind in ("removed_false_alarm", "new_false_alarm"):
        row = expected.loc[expected.kind.eq(kind)].sort_values(["duration_s", *keys[:-1]], ascending=[False, True, True, True, True]).iloc[0]
        selected.append({"case": "longest_" + kind, **{k: row[k] for k in keys}})
    for actual, wanted in zip(cases, selected, strict=True):
        if actual["case"] != wanted["case"]:
            raise ValueError("Fixed case selection differs")
        np.testing.assert_allclose([actual[k] for k in keys], [wanted[k] for k in keys], atol=1e-6, rtol=0)
    result = {"status": "PASS", "raw_records": len(records), "raw_samples": len(raw), "raw_events": count,
        "independent_sample_ownership_and_event_counts": "PASS", "partitions_and_longest_unalarmed_runs": "PASS",
        "all_segments_and_fixed_case_selection": "PASS",
        "audit_files_checked": len(complete["files"]), "model_retrained_or_rescored": False, "new_test_windows_constructed": 0,
        "audit_completion_sha256": sha(audit / "AUDIT_COMPLETED.json"), "verifier_source_sha256": sha(Path(__file__))}
    write_json(audit / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

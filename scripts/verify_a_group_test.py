"""Independently recount held-out scores and labels without production metric code."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["subject_id", "session_id", "task_id"]
METHODS = ["ankle_history_rf", "bilateral_rf", "ankle_cnn"]
CLASSES = ["straight", "left", "right", "unknown"]


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binary(frame, expected):
    truth, prediction = frame.truth_fog_label.to_numpy(int), frame.is_fog.to_numpy(bool)
    tp, fp = int(((truth == 1) & prediction).sum()), int(((truth == 0) & prediction).sum())
    fn, tn = int(((truth == 1) & ~prediction).sum()), int(((truth == 0) & ~prediction).sum())
    assert [tp, fp, fn, tn] == [expected[k] for k in ("TP", "FP", "FN", "TN")]
    assert np.isclose(expected["accuracy"], (tp + tn) / len(truth))
    assert np.isclose(expected["fog_f1"], 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0)


def direction(frame, field, expected):
    valid = frame.input_valid & frame.target.ne("unknown")
    known = frame.loc[valid]
    matrix = [[int((known.target.eq(a) & known[field].eq(b)).sum()) for b in CLASSES]
              for a in CLASSES]
    assert matrix == expected["confusion_matrix"]
    scores = []
    for i in range(3):
        tp = matrix[i][i]
        denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
        scores.append(2 * tp / denominator if denominator else 0)
    assert np.isclose(expected["macro_f1"], np.mean(scores))
    correct = int(known.target.eq(known[field]).sum())
    accepted = int(known[field].ne("unknown").sum())
    assert expected["windows"] == len(known)
    assert np.isclose(expected["accuracy"], correct / len(known))
    assert np.isclose(expected["prediction_coverage"], accepted / len(known))
    assert np.isclose(expected["accepted_accuracy"], correct / accepted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh verification output")
    results = read(args.run / "results.json")
    lock = read(args.lock)
    manifest = read(args.run / "MANIFEST.json")
    assert manifest["lock_sha256"] == digest(args.lock) == results["lock_sha256"]
    assert digest(args.raw) == lock["raw_sha256"]
    for name, value in manifest["files"].items():
        assert digest(args.run / name) == value, name
    root = Path(__file__).resolve().parents[1]
    for name, value in lock["files_sha256"].items():
        if name.startswith(("src/", "scripts/", "config/")):
            assert digest(root / name) == value, name
    people = lock["split"]["test"]
    raw = pd.concat([chunk.loc[chunk.subjectID.isin(people)] for chunk in pd.read_csv(
        args.raw, usecols=["subjectID", "sessionID", "taskID", "timestamp", "fog", "activity"],
        chunksize=50000)], ignore_index=True)
    groups = {tuple(map(str, key)): group.reset_index(drop=True) for key, group in
              raw.groupby(["subjectID", "sessionID", "taskID"], sort=False)}
    tested = []
    for method in METHODS:
        evaluated = pd.read_csv(args.run / method / "evaluation.csv", dtype={k: str for k in KEYS})
        runtime = pd.read_csv(args.run / method / "runtime.csv", dtype={k: str for k in KEYS})
        assert not any(c.startswith("truth_") or c == "target" for c in runtime)
        pd.testing.assert_frame_equal(runtime, evaluated[runtime.columns], check_dtype=False)
        binary(evaluated, results["a1_native_support"][method])
        for key, group in evaluated.groupby(KEYS, sort=False):
            samples = groups[key]
            times = samples.timestamp.to_numpy()
            for row in group.itertuples():
                start = int(np.argmin(abs(times - row.window_start_s)))
                stop = int(np.argmin(abs(times - row.window_end_s))) + 1
                assert stop - start == 120
                fraction = float(samples.fog.iloc[start:stop].mean())
                assert int(fraction >= 0.5) == row.truth_fog_label
                assert np.isclose(fraction, row.truth_fog_fraction)
        tested.append(len(evaluated))
    common_keys = KEYS + ["window_start_s", "window_end_s"]
    indexed = {}
    for method in METHODS:
        frame = pd.read_csv(args.run / method / "evaluation.csv", dtype={k: str for k in KEYS})
        frame[common_keys[3:]] = frame[common_keys[3:]].round(6)
        indexed[method] = frame.set_index(common_keys)
    shared = set.intersection(*(set(f.index) for f in indexed.values()))
    for expected in results["a1_common_windows"]:
        binary(indexed[expected["method"]].loc[list(shared)], expected)
    direction_eval = pd.read_csv(args.run / "direction_evaluation.csv", dtype={k: str for k in KEYS})
    label_checks = 0
    mapping = {1: "straight", 6: "right", 7: "left"}
    for key, group in direction_eval.groupby(KEYS, sort=False):
        samples = groups[key]
        times = samples.timestamp.to_numpy()
        for row in group.itertuples():
            start = int(np.argmin(abs(times - row.window_start_s)))
            labels = [mapping.get(int(a), "unknown") for a in samples.activity.iloc[start:start + 120]]
            counts = {name: labels.count(name) for name in CLASSES}
            top = max(counts, key=counts.get)
            expected = top if counts[top] >= 96 and labels[-1] == top else "unknown"
            assert expected == row.target
            label_checks += 1
    direction(direction_eval, "direction", results["a2_current_direction"]["primary_known_targets"])
    fused_direction = pd.read_csv(args.run / "direction_fused_evaluation.csv")
    direction(fused_direction, "turn",
              results["fusion"]["direction_on_a2_arrival_grid"]["primary_known_targets"])
    binary(pd.read_csv(args.run / "fog_fused_evaluation.csv"),
           results["fusion"]["stable_fog_on_a1_arrival_grid"])
    runtime = pd.read_csv(args.run / "perception_runtime.csv")
    assert not any(c.startswith("truth_") or c == "target" for c in runtime)
    assert runtime.window_end_s.eq(runtime.available_at_s).all()
    report = {"schema": "fogguard.a_group.independent_verification.v1", "status": "PASS",
              "run_manifest_sha256": digest(args.run / "MANIFEST.json"),
              "lock_sha256": digest(args.lock), "source_sha256": digest(Path(__file__)),
              "artifact_hashes_checked": len(manifest["files"]),
              "raw_fog_window_label_checks": sum(tested),
              "raw_direction_window_label_checks": label_checks,
              "matched_windows": len(shared), "confusion_counts_and_metrics": "PASS",
              "runtime_truth_separation": "PASS",
              "scope": "Independent hash/label/confusion recount; no model retraining or retuning"}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

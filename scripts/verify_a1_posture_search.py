"""Audit nested search from every saved inner score; no independent retraining."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score


def verify_candidates(audit, frame, config):
    ids = [c["id"] for c in config["candidates"]]
    if [c["candidate_id"] for c in audit["candidates"]] != ids:
        raise ValueError("Missing/reordered candidate configurations")
    if set(frame.subject_id.astype(int).astype(str)) != set(audit["training_people"]):
        raise ValueError("Wrong inner-score people")
    y = frame.truth_fog_label.to_numpy(int)
    for row, candidate in zip(audit["candidates"], config["candidates"], strict=True):
        if row["features"] != candidate["features"]:
            raise ValueError("Candidate feature schema changed")
        score = frame[f"score__{candidate['id']}"].to_numpy(float)
        if not (np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()):
            raise ValueError("Invalid inner score")
        options, grid = row["threshold_scores"], config["threshold_grid"]
        expected = np.linspace(grid["start"], grid["stop"], grid["count"])
        if len(options) != len(expected):
            raise ValueError("Wrong threshold grid size")
        for option, threshold in zip(options, expected, strict=True):
            if abs(option["threshold"] - threshold) > 1e-12:
                raise ValueError("Threshold grid changed")
            decision = score >= threshold
            if (abs(f1_score(y, decision, zero_division=0) - option["fog_f1"]) > 1e-12
                    or abs(balanced_accuracy_score(y, decision) - option["balanced_accuracy"]) > 1e-12):
                raise ValueError("Inner threshold metric mismatch")
        best = max(options, key=lambda r: (r["fog_f1"], r["balanced_accuracy"], r["threshold"]))
        if best != row["selected_threshold"]:
            raise ValueError("Wrong candidate threshold")
        if abs(average_precision_score(y, score) - row["average_precision"]) > 1e-12:
            raise ValueError("Wrong inner AP")
    best = max(enumerate(audit["candidates"]), key=lambda pair: (
        pair[1]["selected_threshold"]["fog_f1"], pair[1]["average_precision"], -pair[0]))[1]
    if best["candidate_id"] != audit["selected_candidate_id"]:
        raise ValueError("Wrong training-only configuration choice")


def verify_groups(audit, train, forbidden):
    if set(audit["training_people"]) != train or train & forbidden:
        raise ValueError("Wrong training people")
    seen = set()
    for fold in audit["inner_folds"]:
        a, b = set(fold["train"]), set(fold["validation"])
        if a & b or a | b != train or (a | b) & forbidden or b & seen:
            raise ValueError("Leaking inner split")
        seen |= b
    if seen != train:
        raise ValueError("Incomplete inner validation")


def main():
    from verify_a1_multimodel import verify_predictions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    config = protocol["config"]
    audits = json.loads((root / "fold_audit.json").read_text("utf-8"))
    if len(audits) != len(protocol["outer_folds"]):
        raise ValueError("Incomplete outer folds")
    for i, (audit, frozen) in enumerate(zip(audits, protocol["outer_folds"], strict=True), 1):
        if audit["outer_train"] != frozen["train"] or audit["outer_test"] != frozen["test"]:
            raise ValueError("Frozen outer people changed")
        verify_groups(audit["selection"], set(frozen["train"]), set(frozen["test"]))
        verify_candidates(audit["selection"], pd.read_csv(root / f"outer_{i}_inner_scores.csv", float_precision="round_trip"), config)
    deployment = json.loads((root / "deployment_selection.json").read_text("utf-8"))
    all_people = {p for f in protocol["outer_folds"] for p in f["test"]}
    verify_groups(deployment, all_people, set())
    verify_candidates(deployment, pd.read_csv(root / "deployment_inner_scores.csv", float_precision="round_trip"), config)
    comparison = json.loads((root / "comparison.json").read_text("utf-8"))
    canonical = None
    for variant in config["outputs"]:
        directory = root / variant
        frame = pd.read_csv(directory / "oof_predictions.csv", float_precision="round_trip")
        thresholds = {}
        for i, audit in enumerate(audits, 1):
            name = audit["selection"]["selected_candidate_id"] if variant == "inner_selected" else variant
            choice = next(r for r in audit["selection"]["candidates"] if r["candidate_id"] == name)
            people = set(map(int, audit["outer_test"]))
            part = frame.loc[frame.subject_id.isin(people)]
            if not (part.candidate_id.eq(name).all() and part.feature_version.eq(choice["features"]).all()
                    and part.held_out_fold.eq(i).all()):
                raise ValueError("Outer configuration assigned using wrong people")
            thresholds.update({p: choice["selected_threshold"]["threshold"] for p in people})
        verify_predictions(frame, next(r for r in comparison if r["variant"] == variant), thresholds)
        keys = frame[["subject_id", "session_id", "task_id", "window_start_s", "window_end_s", "truth_fog_label"]]
        if canonical is not None:
            pd.testing.assert_frame_equal(keys, canonical)
        canonical = keys
        if not np.array_equal(frame.available_at_s, frame.window_end_s):
            raise ValueError("Backdated decision")
        runtime = pd.read_csv(directory / "a2_runtime.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(runtime, frame.drop(columns=[n for n in frame if n.startswith("truth_")]))
        saved = json.loads((directory / "deployment_config.json").read_text("utf-8"))
        name = deployment["selected_candidate_id"] if variant == "inner_selected" else variant
        choice = next(r for r in deployment["candidates"] if r["candidate_id"] == name)
        if saved["candidate"]["id"] != name or saved["threshold"] != choice["selected_threshold"]["threshold"]:
            raise ValueError("Wrong deployment configuration")
    diagnostic = pd.read_csv(root / "training_only_diagnosis.csv", float_precision="round_trip")
    first = audits[0]["selection"]
    if set(diagnostic.subject_id.astype(int).astype(str)) != set(first["training_people"]):
        raise ValueError("Test people entered training diagnosis")
    threshold = first["candidates"][0]["selected_threshold"]["threshold"]
    y, decision = diagnostic.truth_fog_label.to_numpy(int), diagnostic.inner_score.to_numpy() >= threshold
    expected = np.select([(y == 1) & decision, (y == 0) & decision, (y == 1) & ~decision],
                          ["TP", "FP", "FN"], default="TN")
    if not np.array_equal(diagnostic.outcome, expected):
        raise ValueError("Training diagnosis not based on fresh inner scores")
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    for name, digest in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Output changed: {name}")
    if hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest() != manifest["protocol_sha256"]:
        raise ValueError("Protocol changed")
    result = {"status": "PASS", "variants": len(comparison), "windows_per_variant": len(canonical),
              "configurations_per_selection": len(config["candidates"]),
              "selection_runs_checked": len(audits) + 1, "training_diagnosis_people": len(first["training_people"]),
              "hashes_verified": len(manifest["files"]),
              "scope": "Saved scores, selection, groups and hashes; no independent retraining"}
    (root / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

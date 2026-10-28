"""Independently check saved multi-model decisions, metrics, groups and output hashes.

This verifies artifacts, not an independent retraining or a proof of every fit.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def verify_predictions(frame, expected, threshold_by_subject):
    decision = frame.is_fog.map({True: True, False: False, "True": True, "False": False})
    if decision.isna().any():
        raise ValueError("Invalid boolean decision")
    score = frame.fog_probability.to_numpy(float)
    threshold = frame.fog_threshold.to_numpy(float)
    if not (np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()):
        raise ValueError("Invalid score")
    if not np.array_equal(decision, score >= threshold):
        raise ValueError("Score/threshold decision mismatch")
    expected_threshold = frame.subject_id.astype(int).map(threshold_by_subject).to_numpy(float)
    if not np.allclose(threshold, expected_threshold, atol=1e-12, rtol=0):
        raise ValueError("Threshold assigned to wrong participant")
    truth = frame.truth_fog_label.to_numpy(int)
    for name, value in {
        "fog_f1": f1_score(truth, decision, zero_division=0),
        "auc": roc_auc_score(truth, score),
        "average_precision": average_precision_score(truth, score),
    }.items():
        if abs(float(value) - expected[name]) > 1e-12:
            raise ValueError(f"Saved aggregate metric mismatch: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    comparison = json.loads((root / "comparison.json").read_text("utf-8"))
    config = protocol["config"]
    if {r["family"] for r in comparison} != set(config["models"]):
        raise ValueError("Some declared models did not complete")
    canonical_inner = None
    canonical_keys = None
    for family in config["models"]:
        directory = root / family
        folds = json.loads((directory / "fold_audit.json").read_text("utf-8"))
        if len(folds) != len(protocol["outer_folds"]):
            raise ValueError("Missing outer folds")
        thresholds, inner_signature = {}, []
        for audited, expected_fold in zip(folds, protocol["outer_folds"], strict=True):
            train, test = set(audited["outer_train"]), set(audited["outer_test"])
            if train & test or audited["outer_train"] != expected_fold["train"]:
                raise ValueError("Outer group mismatch")
            if audited["outer_test"] != expected_fold["test"]:
                raise ValueError("Outer test people changed")
            validation_seen = set()
            signatures = []
            for inner in audited["inner"]["folds"]:
                a, b = set(inner["train"]), set(inner["validation"])
                if a & b or a | b != train or b & validation_seen or (a | b) & test:
                    raise ValueError("Inner group leakage or incomplete partition")
                validation_seen |= b
                signatures.append((inner["train"], inner["validation"]))
            if validation_seen != train:
                raise ValueError("Incomplete training-participant inner coverage")
            inner_signature.append(signatures)
            options = audited["inner"]["threshold_scores"]
            best = max(options, key=lambda r: (r["fog_f1"], r["balanced_accuracy"], r["threshold"]))
            if best != audited["inner"]["selected"]:
                raise ValueError("Declared threshold objective not followed")
            for subject in test:
                if int(subject) in thresholds:
                    raise ValueError("Repeated outer test person")
                thresholds[int(subject)] = best["threshold"]
        if canonical_inner is not None and inner_signature != canonical_inner:
            raise ValueError("Algorithms used different inner participant splits")
        canonical_inner = inner_signature
        frame = pd.read_csv(directory / "oof_predictions.csv")
        expected = next(r for r in comparison if r["family"] == family)
        verify_predictions(frame, expected, thresholds)
        keys = frame[["subject_id", "session_id", "task_id", "window_start_s", "window_end_s",
                      "truth_fog_label"]]
        if canonical_keys is not None:
            pd.testing.assert_frame_equal(keys, canonical_keys)
        canonical_keys = keys
        runtime = pd.read_csv(directory / "a2_runtime.csv")
        if any(c.startswith("truth_") for c in runtime):
            raise ValueError("Truth leaked into handoff runtime file")
        pd.testing.assert_frame_equal(runtime, frame.drop(columns=[
            c for c in frame if c.startswith("truth_")]))
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    for relative, expected_hash in manifest["files"].items():
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if digest != expected_hash:
            raise ValueError(f"Generated file hash changed: {relative}")
    result = {"status": "PASS", "models": len(comparison),
              "windows_per_model": len(canonical_keys), "same_inner_groups": True,
              "runtime_truth_columns": 0, "hashes_verified": len(manifest["files"]),
              "scope": "Saved artifact verification; no independent retraining"}
    (root / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

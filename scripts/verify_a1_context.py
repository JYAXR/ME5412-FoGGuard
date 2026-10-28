"""Independently reconstruct context links and audit saved groups/metrics/hashes.

This does not independently retrain or reproduce the whole feature extractor.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify_history(frame):
    """Rebuild references from CSV keys/times without importing model link code."""
    keys = list(zip(frame.subject_id, frame.session_id, frame.task_id, strict=True))
    times = frame.window_start_s.to_numpy(float)
    ends = frame.window_end_s.to_numpy(float)
    if not (np.isfinite(times).all() and np.isfinite(ends).all()
            and np.allclose(ends - times, 119 / 60, atol=1e-6, rtol=0)):
        raise ValueError("Invalid window timing")
    expected = np.full((len(frame), 2), -1, dtype=int)
    first = 0
    for i in range(len(frame)):
        if i and keys[i] == keys[i - 1] and times[i] <= times[i - 1]:
            raise ValueError("Duplicate/reversed context rows")
        if i and (keys[i] != keys[i - 1] or abs(times[i] - times[i - 1] - 0.5) > 1e-6):
            first = i
        for j, lag in enumerate((4, 8)):
            if i - first >= lag:
                expected[i, j] = i - lag
    actual = frame[["lag_2s_row", "lag_4s_row"]].to_numpy(float)
    if not np.array_equal(actual, expected):
        raise ValueError("Wrong/future/cross-boundary history reference")
    return expected


def main():
    from verify_a1_multimodel import verify_predictions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--previous-rf", type=Path)
    args = parser.parse_args()
    root = args.run_dir
    protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    comparison = json.loads((root / "comparison.json").read_text("utf-8"))
    variants = protocol["config"]["variants"]
    if [r["variant"] for r in comparison] != variants:
        raise ValueError("Incomplete declared variants")
    trace = pd.read_csv(root / "history_links.csv")
    links = verify_history(trace)
    signatures = None
    canonical = None
    for variant in variants:
        directory = root / variant
        audits = json.loads((directory / "fold_audit.json").read_text("utf-8"))
        if len(audits) != len(protocol["outer_folds"]):
            raise ValueError("Missing outer folds")
        thresholds, signature = {}, []
        for audit, frozen in zip(audits, protocol["outer_folds"], strict=True):
            if audit["outer_train"] != frozen["train"] or audit["outer_test"] != frozen["test"]:
                raise ValueError("Frozen outer groups changed")
            train, test = set(frozen["train"]), set(frozen["test"])
            if train & test:
                raise ValueError("Outer group leakage")
            seen, inner_signature = set(), []
            for fold in audit["inner"]["folds"]:
                a, b = set(fold["train"]), set(fold["validation"])
                if a & b or (a | b) != train or (a | b) & test or b & seen:
                    raise ValueError("Inner group leakage")
                seen |= b
                inner_signature.append((fold["train"], fold["validation"]))
            if seen != train:
                raise ValueError("Incomplete inner groups")
            signature.append(inner_signature)
            options = audit["inner"]["threshold_scores"]
            grid = protocol["config"]["threshold_grid"]
            expected_grid = np.linspace(grid["start"], grid["stop"], grid["count"])
            if len(options) != len(expected_grid) or not np.allclose(
                    [o["threshold"] for o in options], expected_grid, rtol=0, atol=1e-12):
                raise ValueError("Threshold grid changed")
            selected = max(options, key=lambda r: (r["fog_f1"], r["balanced_accuracy"], r["threshold"]))
            if selected != audit["inner"]["selected"]:
                raise ValueError("Wrong threshold objective")
            for p in test:
                if int(p) in thresholds:
                    raise ValueError("Repeated outer test person")
                thresholds[int(p)] = selected["threshold"]
        if signatures is not None and signatures != signature:
            raise ValueError("Variants used different inner splits")
        signatures = signature
        frame = pd.read_csv(directory / "oof_predictions.csv")
        verify_predictions(frame, next(r for r in comparison if r["variant"] == variant), thresholds)
        columns = ["subject_id", "session_id", "task_id", "window_start_s", "window_end_s"]
        pd.testing.assert_frame_equal(frame[columns], trace[columns])
        if not np.array_equal(frame.available_at_s, frame.window_end_s):
            raise ValueError("Decision availability backdated")
        for i, name in enumerate(("history_2s_available", "history_4s_available")):
            if not np.array_equal(frame[name].to_numpy(), links[:, i] >= 0):
                raise ValueError("History flag mismatch")
        keys = frame[[*columns, "truth_fog_label"]]
        if canonical is not None:
            pd.testing.assert_frame_equal(keys, canonical)
        canonical = keys
        runtime = pd.read_csv(directory / "a2_runtime.csv")
        if any(c.startswith("truth_") for c in runtime):
            raise ValueError("Runtime truth leak")
        pd.testing.assert_frame_equal(runtime, frame.drop(columns=[c for c in frame if c.startswith("truth_")]))
    reproduced = "NOT_RUN (previous RF not supplied)"
    if args.previous_rf:
        old = pd.read_csv(args.previous_rf)
        current = pd.read_csv(root / variants[0] / "oof_predictions.csv")
        columns = ["subject_id", "session_id", "task_id", "window_start_s", "window_end_s",
                   "fog_threshold", "is_fog", "held_out_fold"]
        pd.testing.assert_frame_equal(old[columns], current[columns], check_exact=True)
        # Parallel tree probability summation can differ at machine precision.
        difference = float(np.max(np.abs(old.fog_probability - current.fog_probability)))
        if not np.allclose(old.fog_probability, current.fog_probability, rtol=0, atol=1e-12):
            raise ValueError("Previous baseline RF scores not reproduced within 1e-12")
        reproduced = {"status": "PASS", "maximum_absolute_score_difference": difference,
                      "score_absolute_tolerance": 1e-12,
                      "thresholds_decisions_keys_folds": "exactly equal"}
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    if hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest() != manifest["protocol_sha256"]:
        raise ValueError("Protocol hash changed")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed output: {name}")
    result = {"status": "PASS", "history_rows_checked": len(trace), "same_inner_groups": True,
              "full_history_rows": int((links[:, 1] >= 0).sum()),
              "startup_rows_preserved": int((links[:, 1] < 0).sum()),
              "hashes_verified": len(manifest["files"]), "previous_rf_reproduction": reproduced,
              "scope": "Saved artifacts and independently reconstructed links, not independent retraining"}
    (root / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

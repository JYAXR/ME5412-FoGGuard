"""Verify source-preserving causal routing, raw channel support and event metrics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    from run_a1_fallback import validate_mixed_windows
    from verify_a1_event import old_interval_check

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    if hashlib.sha256(args.raw.read_bytes()).hexdigest() != protocol["raw_sha256"]:
        raise ValueError("Wrong raw source")
    audits = json.loads((root / "fold_audit.json").read_text("utf-8"))
    if len(audits) != len(protocol["outer_folds"]):
        raise ValueError("Missing outer folds")
    thresholds = {}
    for i, (audit, expected) in enumerate(zip(audits, protocol["outer_folds"], strict=True), 1):
        if audit["outer_train"] != expected["train"] or audit["outer_test"] != expected["test"]:
            raise ValueError("Changed outer people")
        train, test, seen = set(expected["train"]), set(expected["test"]), set()
        for fold in audit["inner"]["folds"]:
            a, b = set(fold["train"]), set(fold["validation"])
            if a & b or a | b != train or (a | b) & test or b & seen:
                raise ValueError("Fallback inner leakage")
            seen |= b
        if seen != train:
            raise ValueError("Missing inner validation people")
        choice = max(audit["inner"]["threshold_scores"], key=lambda r: (r["fog_f1"], r["balanced_accuracy"], r["threshold"]))
        if choice != audit["inner"]["selected"]:
            raise ValueError("Wrong training-only threshold objective")
        thresholds.update({int(p): (choice["threshold"], i) for p in test})
    primary = pd.read_csv(root / "primary_only/oof_predictions.csv", float_precision="round_trip")
    ankle = pd.read_csv(root / "ankle_oof_predictions.csv", float_precision="round_trip")
    combined = pd.read_csv(root / "availability_fallback/oof_predictions.csv", float_precision="round_trip")
    ids = ["subject_id", "session_id", "task_id"]
    keys = [*ids, "window_start_s", "window_end_s"]
    retained = combined.loc[combined.sensor_route.eq("primary_ankle_back"), primary.columns]
    pd.testing.assert_frame_equal(primary.sort_values(keys).reset_index(drop=True),
        retained.sort_values(keys).reset_index(drop=True), check_dtype=False, check_exact=True)
    allowed_fallback = []
    for key, group in ankle.groupby(ids, sort=False):
        mask = np.ones(len(primary), bool)
        for name, value in zip(ids, key, strict=True):
            mask &= primary[name].eq(value)
        ends = primary.loc[mask, "window_end_s"].to_numpy(float)
        # Same-sample CSV noise is matched at the existing audit's 1us precision.
        for row in group.itertuples():
            fresh = (ends <= row.window_end_s + 1e-6) & (row.window_end_s < ends + 0.5 - 1e-6)
            if not fresh.any():
                allowed_fallback.append((*key, row.window_start_s, row.window_end_s))
    expected = pd.DataFrame(allowed_fallback, columns=keys).sort_values(keys).reset_index(drop=True)
    used = combined.loc[combined.sensor_route.eq("fallback_ankle")].copy()
    pd.testing.assert_frame_equal(expected, used[keys].sort_values(keys).reset_index(drop=True), check_dtype=False)
    a = used.merge(ankle, on=keys, suffixes=("_routed", "_original"), validate="one_to_one")
    for field in ("fog_probability", "fog_threshold", "is_fog", "held_out_fold", "truth_fog_label"):
        if not np.array_equal(a[field + "_routed"], a[field + "_original"]):
            raise ValueError("Routing changed ankle prediction")
    for row in ankle.itertuples():
        threshold, fold = thresholds[int(row.subject_id)]
        if (abs(threshold - row.fog_threshold) > 1e-12 or row.held_out_fold != fold
                or row.is_fog != (row.fog_probability >= row.fog_threshold)):
            raise ValueError("Ankle decision/fold mismatch")
    raw = pd.read_csv(args.raw)
    results = json.loads((root / "comparison.json").read_text("utf-8"))
    for variant, frame in [("primary_only", primary), ("availability_fallback", combined)]:
        validate_mixed_windows(raw, frame)
        replay = frame.assign(event_fog=frame.is_fog)
        actual = old_interval_check(raw, replay, Path(__file__).resolve().parents[1])
        metrics = next(r for r in results if r["variant"] == variant)
        for name, value in actual.items():
            if abs(value - metrics[name]) > 1e-6:
                raise ValueError(f"Raw interval mismatch: {name}")
        runtime = pd.read_csv(root / variant / "a2_runtime.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(runtime, frame.drop(columns=[n for n in frame if n.startswith("truth_")]))
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    for name, digest in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed file: {name}")
    result = {"status": "PASS", "primary_rows_preserved": len(primary), "added_fallback_rows": len(used),
              "combined_rows_validated": len(combined), "raw_records": raw.groupby(["subjectID", "sessionID", "taskID"]).ngroups,
              "hashes_verified": len(manifest["files"]),
              "scope": "Saved score/routing/raw-support/interval verification; no independent retraining"}
    (root / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

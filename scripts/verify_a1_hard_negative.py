"""Verify saved three-level participant isolation and pure-negative weighting."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from verify_a1_multimodel import verify_predictions


def check_mining(audit, training, forbidden):
    if set(audit["training_people"]) != training or training & forbidden:
        raise ValueError("Wrong training people or validation/test leakage")
    seen = set()
    for fold in audit.get("mining_folds", []):
        a, b = set(fold["train"]), set(fold["validation"])
        if a & b or a | b != training or (a | b) & forbidden or seen & b:
            raise ValueError("Leaking mining fold")
        seen |= b
    if "mining_folds" in audit and seen != training:
        raise ValueError("Incomplete mining coverage")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    cfg = protocol["config"]
    results = json.loads((root / "comparison.json").read_text("utf-8"))
    keys_reference = None
    weighted_rows = 0
    for variant in cfg["variants"]:
        directory = root / variant
        audits = json.loads((directory / "fold_audit.json").read_text("utf-8"))
        if len(audits) != len(protocol["outer_folds"]):
            raise ValueError("Missing outer folds")
        thresholds = {}
        for i, (outer, expected) in enumerate(zip(audits, protocol["outer_folds"], strict=True), 1):
            if outer["outer_train"] != expected["train"] or outer["outer_test"] != expected["test"]:
                raise ValueError("Frozen outer people changed")
            train, test = set(expected["train"]), set(expected["test"])
            check_mining(outer["outer_training_audit"], train, test)
            inner_seen = set()
            for inner in outer["inner"]["folds"]:
                a, b = set(inner["train"]), set(inner["validation"])
                if a & b or a | b != train or inner_seen & b or (a | b) & test:
                    raise ValueError("Leaking inner validation")
                check_mining(inner["training_audit"], a, b | test)
                inner_seen |= b
            if inner_seen != train:
                raise ValueError("Incomplete inner groups")
            best = max(outer["inner"]["threshold_scores"],
                       key=lambda r: (r["fog_f1"], r["balanced_accuracy"], r["threshold"]))
            if best != outer["inner"]["selected"]:
                raise ValueError("Wrong threshold objective")
            thresholds.update({int(p): best["threshold"] for p in test})
            if variant == "hard_negative":
                weights = pd.read_csv(directory / f"outer_{i}_training_weights.csv")
                if set(weights.subject_id.astype(str)) != train:
                    raise ValueError("Training weight rows contain wrong people")
                hard = (weights.truth_label.eq(0) & weights.truth_fraction.eq(0)
                        & weights.mining_score.ge(cfg["mining_threshold"]))
                expected_weight = np.where(hard, cfg["negative_weight"], 1.0)
                if not np.array_equal(weights.sample_weight, expected_weight):
                    raise ValueError("Positive/mixed window weighted or mining condition changed")
                if int(hard.sum()) != outer["outer_training_audit"]["weighted_negative_windows"]:
                    raise ValueError("Hard-negative count mismatch")
                weighted_rows += len(weights)
        frame = pd.read_csv(directory / "oof_predictions.csv")
        metrics = next(r for r in results if r["variant"] == variant)
        verify_predictions(frame, metrics, thresholds)
        keys = frame[["subject_id", "session_id", "task_id", "window_start_s", "window_end_s",
                      "truth_fog_label"]]
        if keys_reference is not None:
            pd.testing.assert_frame_equal(keys, keys_reference)
        keys_reference = keys
        runtime = pd.read_csv(directory / "a2_runtime.csv")
        if any(n.startswith("truth_") for n in runtime):
            raise ValueError("Runtime truth leak")
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    for path, digest in manifest["files"].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed output: {path}")
    result = {"status": "PASS", "three_level_group_isolation": True,
              "outer_training_weight_rows_checked": weighted_rows,
              "hashes_verified": len(manifest["files"]),
              "scope": "Saved artifacts, not an independent retraining"}
    (root / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

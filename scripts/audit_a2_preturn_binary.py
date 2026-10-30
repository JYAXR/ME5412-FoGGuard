"""Diagnostic validation threshold curves; never changes a fitted model or threshold."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_a2_preturn_binary import evaluate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    manifest = json.loads((args.run / "results.json").read_text("utf-8"))
    audit = {"purpose": "Validation diagnostic only; no refitting or threshold selection",
             "manifest_sha256": digest(args.run / "results.json"), "results": {}}
    keys = ["subject_id", "session_id", "task_id", "window_start_s", "window_end_s"]
    args.output.mkdir(parents=True)
    for horizon in manifest["config"]["horizons_s"]:
        directory = args.run / f"preturn_{horizon:.1f}s"
        runtime, truth = directory / "runtime.csv", directory / "evaluation_truth.csv"
        meta = pd.read_csv(runtime).merge(pd.read_csv(truth), on=keys,
                                          validate="one_to_one", how="left", sort=False)
        scores = meta.turn_score.to_numpy()
        rows = []
        for threshold in manifest["config"]["thresholds"]:
            metrics = evaluate(meta, scores, threshold)
            rows.append({"threshold": threshold,
                         "precision": metrics["walking_binary"]["precision"],
                         "recall": metrics["walking_binary"]["recall"],
                         "f1": metrics["walking_binary"]["f1"],
                         "false_alarms_per_minute": metrics["full_stream"]["false_alarms_per_minute"],
                         "negative_alarm_fraction": metrics["full_stream"]["negative_alarm_fraction"]})
        review = pd.DataFrame(rows)
        review.to_csv(args.output / f"thresholds_{horizon:.1f}s.csv", index=False)
        above_half = [r for r in rows if r["recall"] >= 0.5]
        result = {"runtime_sha256": digest(runtime), "truth_sha256": digest(truth),
                  "fitted_model_threshold": manifest["results"][str(horizon)]["threshold"],
                  "diagnostic_points": rows,
                  "minimum_false_alarm_point_with_recall_at_least_half":
                      min(above_half, key=lambda r: r["false_alarms_per_minute"])
                      if above_half else None}
        audit["results"][str(horizon)] = result
        selected = next(r for r in rows if np.isclose(r["threshold"], result["fitted_model_threshold"]))
        print(horizon, "TRAIN-selected threshold:", selected, flush=True)
        print(horizon, "Diagnostic >=50% recall:",
              result["minimum_false_alarm_point_with_recall_at_least_half"], flush=True)
    (args.output / "threshold_audit.json").write_text(json.dumps(audit, indent=2), "utf-8")


if __name__ == "__main__":
    main()

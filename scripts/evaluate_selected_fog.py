"""Read-only matched development evaluation of the three retained models, without refitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fogguard.evaluation.intervals import EventSupport, merge
from fogguard.evaluation.selected_fog import annotate, grouped_diagnostics, metrics
from fogguard.selected_fog import METHODS, check_lock, load_selected, sha, windows_for, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/fog_selected.json"))
    parser.add_argument("--reference-run", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    config = json.loads(args.config.read_text("utf-8"))
    check_lock(args.bundle, json.loads((args.bundle / "MANIFEST.json").read_text("utf-8")))
    if sha(args.raw) != config["raw_sha256"]:
        raise ValueError("Different raw dataset")
    raw = pd.read_csv(args.raw)
    validation = raw.loc[raw.subjectID.isin(config["split"]["validation"])].copy()
    del raw
    observed = sum(sum(b - a for a, b in merge(zip(g.timestamp, g.timestamp + 1 / 60, strict=True)))
                   for _, g in validation.groupby(["subjectID", "sessionID", "taskID"]))
    args.output.mkdir(parents=True)
    rows, first, checked = [], None, 0
    for method in METHODS:
        windows = windows_for(validation, method)
        model = load_selected(args.bundle / method / "model.joblib", method)
        frame = annotate(windows, model.predict_windows(windows), "previously_inspected_validation")
        keys = ["subject_id", "session_id", "task_id", "window_start_s", "window_end_s", "truth_fog_fraction", "truth_fog_label"]
        if first is not None:
            np.testing.assert_allclose(frame[keys].astype(float), first[keys].astype(float), atol=1e-6, rtol=0)
        first = frame
        difference = None
        if args.reference_run:
            ref = pd.read_csv(args.reference_run / method / "reference_validation.csv", float_precision="round_trip")
            np.testing.assert_allclose(frame[keys].astype(float), ref[keys].astype(float), atol=1e-6, rtol=0)
            np.testing.assert_allclose(frame.fog_probability, ref.fog_probability, atol=1e-7, rtol=0)
            np.testing.assert_array_equal(frame.is_fog, ref.is_fog)
            difference = float(np.max(abs(frame.fog_probability.to_numpy() - ref.fog_probability.to_numpy())))
            checked += len(frame)
        dest = args.output / method
        dest.mkdir()
        frame.to_csv(dest / "predictions.csv", index=False)
        frame.drop(columns=[c for c in frame if c.startswith("truth_")]).to_csv(dest / "runtime.csv", index=False)
        grouped_diagnostics(frame, "subject_id").to_csv(dest / "by_subject.csv", index=False)
        support = EventSupport.build(validation, frame, allowed=config["split"]["validation"])
        rows.append({"method": method, "threshold": model.threshold, **metrics(frame), **support.metrics(frame.is_fog),
            "observed_s": observed, "support_s": float(support.duration.sum()),
            "support_fraction": float(support.duration.sum()) / observed,
            "raw_records": validation.groupby(["subjectID", "sessionID", "taskID"]).ngroups,
            "reference_max_score_difference": difference})
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "comparison.csv", index=False)
    root = Path(__file__).resolve().parents[1]
    sources = [root / "src/fogguard/selected_fog.py", *sorted((root / "src/fogguard/modeling").glob("a1_*.py")),
               root / "src/fogguard/evaluation/intervals.py", root / "src/fogguard/evaluation/selected_fog.py", Path(__file__)]
    write_json(args.output / "VERIFICATION.json", {"status": "PASS", "scope": "Preserved selected-model development scores, not a new blind test",
        "matched_windows": len(first), "positive_windows": int(first.truth_fog_label.sum()),
        "reference_prediction_rows_checked": checked, "models_refitted": 0, "test_windows_constructed": 0,
        "config_sha256": sha(args.config), "bundle_manifest_sha256": sha(args.bundle / "MANIFEST.json"),
        "source_sha256": {p.relative_to(root).as_posix(): sha(p) for p in sources},
        "files": {p.relative_to(args.output).as_posix(): sha(p) for p in args.output.rglob("*") if p.is_file()}})
    print(table[["method", "precision", "recall", "fog_f1", "false_alarm_s", "time_recall", "covered_raw_events"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

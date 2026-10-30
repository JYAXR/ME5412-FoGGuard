"""Lock existing A-group artifacts, then score the reserved participants once."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fogguard.evaluation.heldout import (
    align_direction_truth,
    assert_split,
    at_source_arrivals,
    attach_fog_truth,
    binary_report,
    direction_report,
    frozen_inputs,
    validate_fog_membership,
    verify_lock,
)
from fogguard.evaluation.intervals import EventSupport, merge
from fogguard.evaluation.selected_fog import matched_frames
from fogguard.perception import KEYS, PerceptionConfig, fuse_perception, summarize_perception
from fogguard.selected_fog import METHODS, check_lock, load_selected, sha, windows_for, write_json
from fogguard.turning.learned import RAW_KEYS, DirectionModel, current_target, iter_raw_windows


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["lock", "score"])
    parser.add_argument("--fog-bundle", type=Path, required=True)
    parser.add_argument("--direction-run", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = read(root / "config/fog_selected.json")
    protocol = read(root / "config/a_group_test.json")
    a2_training = read(args.direction_run / "results.json")
    assert_split(config["split"])
    if a2_training["split"] != config["split"] or a2_training["test_scored"]:
        raise ValueError("Direction training must use the fixed split without TEST scoring")
    if sha(args.raw) != config["raw_sha256"] or config["raw_sha256"] != a2_training["raw_sha256"]:
        raise ValueError("Dataset differs from fixed training source")
    check_lock(args.fog_bundle, read(args.fog_bundle / "MANIFEST.json"))
    frozen = frozen_inputs(root, args.fog_bundle, args.direction_run)
    models = {m: load_selected(args.fog_bundle / m / "model.joblib", m) for m in METHODS}
    for model in models.values():
        validate_fog_membership(model, config["split"])
    direction_model = DirectionModel.load(args.direction_run / "current/model.joblib")
    if direction_model.horizon_s != 0:
        raise ValueError("Only existing current-direction weights may be evaluated")
    if args.mode == "lock":
        if args.lock.exists():
            parser.error("Lock exists; preserve the pre-score artifact")
        lock = {
            "schema": "fogguard.a_group.test_lock.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                                   text=True).strip(),
            "raw_sha256": config["raw_sha256"], "split": config["split"],
            "protocol": protocol, "files_sha256": frozen,
            "perception_config": asdict(PerceptionConfig()),
            "fog_models": {m: {"threshold": model.threshold,
                               "fit_participants": list(model.fit_participants),
                               "threshold_participants": list(model.threshold_participants)}
                           for m, model in models.items()},
            "direction_model": {"horizon_s": direction_model.horizon_s,
                                "rejection_threshold": direction_model.rejection_threshold,
                                "feature_indices": list(direction_model.feature_indices)
                                if direction_model.feature_indices is not None else None},
            "test_scored_at_lock_creation": False,
            "test_policy": "No fitting, threshold search or method selection after TEST scoring",
        }
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation: never replace an existing pre-score lock.
        with args.lock.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(lock, handle, ensure_ascii=False, indent=2)
        print("Locked source/config/model hashes before TEST feature extraction", flush=True)
        return
    if args.output_dir is None or args.output_dir.exists():
        parser.error("Scoring requires a fresh --output-dir")
    lock = read(args.lock)
    verify_lock(lock, frozen)
    if protocol != lock["protocol"] or config["split"] != lock["split"]:
        raise ValueError("Protocol or split differs from pre-score lock")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "EVALUATION_STARTED.json", {
        "lock_sha256": sha(args.lock), "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_labels_exposed_by_this_run": True,
    })
    channels = sorted(set(direction_model.channels).union(
        *(set(model.channel_names) for model in models.values())))
    columns = ["timestamp", *RAW_KEYS, *channels, "fog", "activity"]
    people = config["split"]["test"]
    raw = pd.concat([chunk.loc[chunk.subjectID.isin(people)] for chunk in
                     pd.read_csv(args.raw, usecols=columns, chunksize=50000)], ignore_index=True)
    if set(raw.subjectID.unique()) != set(people):
        raise ValueError("Reserved participants are missing from the dataset")
    sensors = raw.drop(columns=["fog", "activity"])
    observed = sum(sum(b - a for a, b in merge(zip(g.timestamp, g.timestamp + 1 / 60, strict=True)))
                   for _, g in raw.groupby(RAW_KEYS, sort=False))
    frames, native, truth_files = {}, {}, []
    for method, model in models.items():
        print(f"TEST sensor-only inference: {method}", flush=True)
        runtime = model.predict_frame(sensors)
        # Build annotation targets separately, after label-free inference.
        evaluated = attach_fog_truth(runtime, windows_for(raw, method))
        evaluated["data_role"] = "locked_participant_heldout_test"
        frames[method] = evaluated
        support = EventSupport.build(raw, evaluated, allowed=people,
                                     forbidden=config["split"]["train"] + config["split"]["validation"])
        native[method] = {**binary_report(evaluated), **support.metrics(evaluated.is_fog),
                          "threshold": model.threshold, "observed_s": observed,
                          "supported_s": float(support.duration.sum()),
                          "support_fraction": float(support.duration.sum()) / observed,
                          "per_subject": {str(p): binary_report(g) for p, g in
                                          evaluated.groupby("subject_id", sort=True)}}
        dest = args.output_dir / method
        dest.mkdir()
        runtime.to_csv(dest / "runtime.csv", index=False)
        evaluated.to_csv(dest / "evaluation.csv", index=False)
        truth_files.append(f"{method}/evaluation.csv")
    common = matched_frames(frames)
    comparison = [{"method": m, "threshold": models[m].threshold, **binary_report(common[m])}
                  for m in METHODS]
    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (dict, list))}
                  for row in comparison]).to_csv(args.output_dir / "fog_comparison.csv", index=False)
    print("TEST sensor-only inference: current direction", flush=True)
    direction = direction_model.predict_frame(sensors)
    purity = read(root / "config/a2_turning.json")["current_label_purity"]
    targets = []
    for row, _, group, start, stop in iter_raw_windows(raw, direction_model.channels):
        row["target"] = current_target(group.activity.to_numpy()[start:stop], purity)
        targets.append(row)
    direction_eval = align_direction_truth(direction, pd.DataFrame(targets))
    direction.to_csv(args.output_dir / "direction_runtime.csv", index=False)
    direction_eval.to_csv(args.output_dir / "direction_evaluation.csv", index=False)
    truth_files.append("direction_evaluation.csv")
    primary = protocol["fusion_fog_method"]
    fog_runtime = frames[primary].drop(columns=[c for c in frames[primary] if c.startswith("truth_")
                                              or c == "data_role"])
    clocks = []
    for key, group in sensors.groupby(RAW_KEYS, sort=False):
        times = group.timestamp.to_numpy()
        indices = sorted({0, len(times) - 1, *range(119, len(times), 30)})
        clocks.extend([{**dict(zip(KEYS, key, strict=True)), "timestamp_s": float(times[i])}
                       for i in indices])
    perception = fuse_perception(fog_runtime, direction, clock=pd.DataFrame(clocks))
    perception.to_csv(args.output_dir / "perception_runtime.csv", index=False)
    fog_arrivals = at_source_arrivals(perception, fog_runtime)
    stable = frames[primary].copy()
    stable["is_fog"] = fog_arrivals.stable_fog_detected.to_numpy(bool)
    stable_support = EventSupport.build(raw, stable, allowed=people)
    fused_direction = direction_eval.copy()
    fused_direction["turn"] = at_source_arrivals(perception, direction).turn.to_numpy()
    fused_direction.to_csv(args.output_dir / "direction_fused_evaluation.csv", index=False)
    truth_files.append("direction_fused_evaluation.csv")
    stable.to_csv(args.output_dir / "fog_fused_evaluation.csv", index=False)
    truth_files.append("fog_fused_evaluation.csv")
    summary = {
        "schema": "fogguard.a_group.test_results.v1", "partition": "participant_heldout_test",
        "participants": people, "records": raw.groupby(RAW_KEYS).ngroups,
        "models_refitted": 0, "thresholds_changed": False,
        "primary_fusion_fog_method": primary, "lock_sha256": sha(args.lock),
        "a1_common_windows": comparison, "a1_native_support": native,
        "a2_current_direction": direction_report(direction_eval),
        "fusion": {"behavior": summarize_perception(perception),
                   "stable_fog_on_a1_arrival_grid": {**binary_report(stable),
                       **stable_support.metrics(stable.is_fog)},
                   "direction_on_a2_arrival_grid": direction_report(fused_direction, "turn")},
        "scope": "Locked participant-held-out classification and offline fusion; "
                 "not preturn intent, measured step response, clinical efficacy or live latency",
    }
    write_json(args.output_dir / "results.json", summary)
    # Check again after inference: evaluation must not mutate weights/config/code.
    verify_lock(lock, frozen_inputs(root, args.fog_bundle, args.direction_run))
    outputs = {p.relative_to(args.output_dir).as_posix(): sha(p)
               for p in args.output_dir.rglob("*") if p.is_file()}
    write_json(args.output_dir / "MANIFEST.json", {
        "schema": "fogguard.a_group.test_run.v1", "lock_sha256": sha(args.lock),
        "runtime_truth_separate": True, "evaluation_only_files": truth_files,
        "model_hashes_unchanged": True, "files": outputs,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
    })
    print(pd.DataFrame(comparison)[["method", "accuracy", "precision", "recall", "fog_f1"]]
          .to_string(index=False), flush=True)
    print(json.dumps(summary["a2_current_direction"]["primary_known_targets"], indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Replay existing A1/A2 models on fixed development people, without training."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from fogguard.perception import KEYS, fuse_perception, summarize_perception
from fogguard.selected_fog import METHODS, check_lock, load_selected, sha, write_json
from fogguard.turning.learned import RAW_KEYS, DirectionModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/fog_selected.json"))
    parser.add_argument("--fog-bundle", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default="bilateral_rf")
    parser.add_argument("--direction-run", type=Path, required=True,
                        help="A2 training directory containing results.json and current/model.joblib")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a fresh output directory")
    config = json.loads(args.config.read_text("utf-8"))
    direction_manifest = json.loads((args.direction_run / "results.json").read_text("utf-8"))
    raw_hash = sha(args.raw)
    if raw_hash != config["raw_sha256"] or raw_hash != direction_manifest["raw_sha256"]:
        raise ValueError("Raw dataset must match the fixed split and A2 training source")
    if config["split"] != direction_manifest["split"] or direction_manifest["test_scored"]:
        raise ValueError("A1/A2 fixed splits or TEST-use policy disagree")
    check_lock(args.fog_bundle,
               json.loads((args.fog_bundle / "MANIFEST.json").read_text("utf-8")))
    fog_path = args.fog_bundle / args.method / "model.joblib"
    direction_path = args.direction_run / "current/model.joblib"
    fog_model = load_selected(fog_path, args.method)
    direction_model = DirectionModel.load(direction_path)
    if direction_model.horizon_s != 0:
        raise ValueError("Only the current-direction model can be fused")
    channels = sorted(set(fog_model.channel_names) | set(direction_model.channels))
    # No annotation columns are even loaded by this runtime route. TEST samples
    # are discarded by participant before retaining/featurizing the input.
    columns = ["timestamp", *RAW_KEYS, *channels]
    people = config["split"]["validation"]
    sensors = pd.concat([c.loc[c.subjectID.isin(people)] for c in
                         pd.read_csv(args.raw, usecols=columns, chunksize=50000)], ignore_index=True)
    print(f"Loaded {len(sensors)} development samples; labels and TEST excluded", flush=True)
    fog = fog_model.predict_frame(sensors)
    direction = direction_model.predict_frame(sensors)
    # Clock includes record start/end and every 0.5 s input-grid boundary.
    clocks = []
    for key, group in sensors.groupby(RAW_KEYS, sort=False):
        times = group.timestamp.to_numpy()
        indices = sorted({0, len(times) - 1, *range(119, len(times), 30)})
        clocks.extend([{**dict(zip(KEYS, key, strict=True)), "timestamp_s": float(times[i])}
                       for i in indices])
    clock = pd.DataFrame(clocks)
    perception = fuse_perception(fog, direction, clock=clock)
    summary = summarize_perception(perception)
    summary.update(partition="development_validation", participants=people,
                   test_scored=False, truth_loaded=False, fog_method=args.method,
                   feedback_implemented=False, receiver_acceptance="pending",
                   availability_assumption="Offline model outputs use window-end availability; "
                                           "live computation latency is not measured")
    args.output_dir.mkdir(parents=True)
    for name, frame in (("fog_runtime", fog), ("direction_runtime", direction),
                        ("clock", clock), ("perception_runtime", perception)):
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
    write_json(args.output_dir / "summary.json", summary)
    root = Path(__file__).resolve().parents[1]
    write_json(args.output_dir / "MANIFEST.json", {
        "schema": "fogguard.perception.validation.v1",
        "partition": "development_validation", "participants": people,
        "raw_sha256": raw_hash, "config_sha256": sha(args.config),
        "model_sha256": {"fog": sha(fog_path), "direction": sha(direction_path)},
        "a2_training_manifest_sha256": sha(args.direction_run / "results.json"),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                               text=True).strip(),
        "source_sha256": {name: sha(root / name) for name in (
            "scripts/validate_perception.py", "src/fogguard/perception.py",
            "src/fogguard/context/fogstar.py", "src/fogguard/turning/learned.py",
            "src/fogguard/selected_fog.py", "src/fogguard/modeling/a1_fog.py",
            "src/fogguard/modeling/a1_bilateral.py", "src/fogguard/data/fogstar.py",
        )},
        "files": {p.name: sha(p) for p in args.output_dir.iterdir() if p.is_file()},
        "scope": "Real-IMU development inference and causal fusion; no accuracy or feedback claims",
    })
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

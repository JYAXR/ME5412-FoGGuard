"""Freeze random real-signal clips and a separate annotation table before inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_KEYS = ["subjectID", "sessionID", "taskID"]
CHANNELS = [f"{sensor}_{kind}_{axis}" for sensor in ("ankleL", "back")
            for kind in ("acc", "gyro") for axis in "xyz"] + [f"ankleR_gyro_{a}" for a in "xyz"]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_clips(raw, protocol, allowed_people):
    """Uniformly shuffled anchors per record; score intervals cannot overlap."""
    if protocol["partition"] != "validation":
        raise ValueError("This robustness protocol uses development participants, not locked TEST")
    rng = np.random.default_rng(protocol["seed"])
    total = protocol["warmup_samples"] + protocol["scoring_samples"]
    if protocol["warmup_samples"] < 360 or protocol["scoring_samples"] < 120:
        raise ValueError("Require six seconds of warmup and at least one scored window")
    for keys, record in raw.groupby(RAW_KEYS, sort=True):
        if int(keys[0]) not in set(allowed_people):
            continue
        record = record.reset_index(drop=True)
        times = record.timestamp.to_numpy(float)
        if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("Raw record timestamps must be finite and increasing")
        candidates = np.arange(0, len(record) - total + 1, protocol["anchor_step_samples"])
        rng.shuffle(candidates)
        selected = []
        for begin in candidates:
            chunk = record.iloc[int(begin):int(begin) + total].copy().reset_index(drop=True)
            if not np.all(np.abs(np.diff(chunk.timestamp.to_numpy(float)) - 1/60) <= 1e-4):
                continue
            lower = float(chunk.timestamp.iloc[protocol["warmup_samples"] - 1])
            upper = lower + protocol["scoring_samples"] / 60
            if any(lower < b and upper > a for a, b in selected):
                continue
            selected.append((lower, upper))
            yield keys, int(begin), lower, upper, chunk
            if len(selected) == protocol["clips_per_record"]:
                break


def build_plan(raw_path, protocol_path, split_path, output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(protocol_path.read_text("utf-8"))
    split = json.loads(split_path.read_text("utf-8"))["split"]
    actual = file_hash(raw_path)
    if actual != protocol["raw_sha256"]:
        raise ValueError("Raw dataset differs from frozen protocol")
    needed = ["timestamp", *RAW_KEYS, *CHANNELS, "fog", "activity"]
    raw = pd.read_csv(raw_path, usecols=needed)
    # No TRAIN/TEST samples enter clip selection. No model outputs influence sampling.
    raw = raw.loc[raw.subjectID.isin(split["validation"])].copy()
    folder = output / "clips"
    folder.mkdir()
    selected = []
    for index, (keys, begin, lower, upper, chunk) in enumerate(
        select_clips(raw, protocol, split["validation"])
    ):
        name = f"clip_{index:03d}"
        sensors = chunk.drop(columns=["fog", "activity"])
        truth = chunk[[*RAW_KEYS, "timestamp", "fog", "activity"]].copy()
        sensors.to_csv(folder / f"{name}_sensors.csv", index=False)
        truth.to_csv(folder / f"{name}_truth.csv", index=False)
        scored = truth.loc[truth.timestamp.ge(lower) & truth.timestamp.lt(upper)]
        rows = []
        for start in range(0, len(chunk) - 119, 30):
            end = start + 120
            timestamp = float(chunk.timestamp.iloc[end - 1])
            if not lower - 1e-8 <= timestamp < upper - 1e-8:
                continue
            activity = chunk.activity.iloc[start:end].to_numpy(int)
            mapped = np.array([{1: "straight", 6: "right", 7: "left"}.get(int(a), "unknown")
                               for a in activity])
            labels, counts = np.unique(mapped, return_counts=True)
            top = str(labels[counts.argmax()])
            direction = top if counts.max() / 120 >= 0.8 and mapped[-1] == top else "unknown"
            rows.append({"clip": name, "window_start_s": float(chunk.timestamp.iloc[start]),
                         "window_end_s": timestamp,
                         "truth_fog_label": int(chunk.fog.iloc[start:end].mean() >= 0.5),
                         "truth_direction": direction})
        pd.DataFrame(rows).to_csv(folder / f"{name}_window_truth.csv", index=False)
        selected.append({"clip": name, "subject_id": int(keys[0]), "session_id": int(keys[1]),
                         "task_id": int(keys[2]), "source_row_start": begin,
                         "score_start_s": lower, "score_stop_s": upper,
                         "samples": len(chunk), "scored_windows": len(rows),
                         "positive_windows": sum(r["truth_fog_label"] for r in rows),
                         "missing_sensor_samples": int(sensors[CHANNELS].isna().any(axis=1).sum()),
                         "fog_sample_fraction": float(scored.fog.mean()),
                         "activities": sorted(map(int, scored.activity.unique()))})
    if not selected:
        raise ValueError("No full-length development clips available")
    pd.DataFrame(selected).to_csv(output / "sample_manifest.csv", index=False)
    digest = {p.relative_to(output).as_posix(): file_hash(p)
              for p in sorted(output.rglob("*.csv"))}
    manifest = {"schema": "fogguard.b2.randomized_plan.v1", "protocol": protocol,
                "allowed_people": split["validation"], "forbidden_TEST_people": split["test"],
                "raw_sha256": actual, "protocol_sha256": file_hash(protocol_path),
                "split_sha256": file_hash(split_path), "models_consulted": False,
                "clips": len(selected), "records": len({(r["subject_id"], r["session_id"],
                                                         r["task_id"]) for r in selected}),
                "scored_windows": sum(r["scored_windows"] for r in selected),
                "positive_windows": sum(r["positive_windows"] for r in selected),
                "csv_sha256": digest}
    (output / "PLAN.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    return manifest


def mutate_sensors(frame, kind, seed, protocol):
    """Preserve original truth separately. Noise is sensitivity, not new clinical truth."""
    if {"fog", "activity", "fog_severity"} & set(frame):
        raise ValueError("Sensor mutations must not receive annotation columns")
    result = frame.copy()
    rng = np.random.default_rng(seed)
    if kind in {"noise_low", "noise_high"}:
        for column in CHANNELS:
            scale = protocol[kind]["acc_g_std" if "_acc_" in column else "gyro_deg_s_std"]
            result[column] += rng.normal(0, scale, len(result))
    elif kind == "right_gyro_gap":
        start = len(result) // 2
        result.loc[start:start+119, [f"ankleR_gyro_{a}" for a in "xyz"]] = np.nan
    elif kind == "missing_left_gyro_column":
        result = result.drop(columns="ankleL_gyro_y")
    elif kind == "reversed_time":
        result = result.iloc[::-1].reset_index(drop=True)
    elif kind == "duplicate_time":
        result.loc[len(result)//2, "timestamp"] = result.loc[len(result)//2-1, "timestamp"]
    elif kind != "clean":
        raise ValueError("Unknown perturbation")
    return result

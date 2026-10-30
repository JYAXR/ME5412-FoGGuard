"""Run the published frozen models on preselected development clips and perturbations."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from fogguard.evaluation.heldout import at_source_arrivals
from fogguard.evaluation.randomized_model_scoring import (
    attach_to_truth,
    direction_metrics,
    event_metrics,
    fog_metrics,
)
from fogguard.evaluation.randomized_plan import file_hash, mutate_sensors
from fogguard.perception import fuse_perception
from fogguard.selected_fog import METHODS, check_lock, load_selected
from fogguard.turning.learned import DirectionModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = (*METHODS, "current_direction")
INVALID_INPUTS = {"missing_left_gyro_column", "reversed_time", "duplicate_time"}


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")


def load_frozen(bundle, direction_path):
    """Check published hashes before loading trusted joblib, never re-fit."""
    release_lock = json.loads((ROOT / "docs/a_group/TEST_LOCK.json").read_text("utf-8"))
    check_lock(bundle, json.loads((bundle / "MANIFEST.json").read_text("utf-8")))
    paths = {method: bundle / method / "model.joblib" for method in METHODS}
    paths["current_direction"] = direction_path
    hashes = {name: file_hash(path) for name, path in paths.items()}
    for name, digest in hashes.items():
        if digest != release_lock["files_sha256"]["models/" + name]:
            raise ValueError(f"{name} differs from A-group TEST locked model")
    models = {method: load_selected(paths[method], method) for method in METHODS}
    models["current_direction"] = DirectionModel.load(direction_path)
    if models["current_direction"].horizon_s != 0:
        raise ValueError("Only the current-direction model is valid")
    return models, hashes


def check_plan(plan):
    manifest = json.loads((plan / "PLAN.json").read_text("utf-8"))
    if manifest.get("schema") != "fogguard.b2.randomized_plan.v1":
        raise ValueError("Expected a frozen B2 sampling plan")
    if manifest["protocol"]["partition"] != "validation":
        raise ValueError("Unexpected TEST or training partition")
    for name, expected in manifest["csv_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe plan path")
        if file_hash(plan / relative) != expected:
            raise ValueError(f"Plan input changed: {name}")
    return manifest


def valid_source_windows(frame, runtime, channels):
    """Independently reject emitted decisions based on missing/gapped input."""
    valid = runtime
    if "input_valid" in valid:
        valid = valid.loc[valid.input_valid.eq(True)]
    if valid.empty:
        return 0
    if not set(channels) <= set(frame.columns):
        return len(valid)
    times = frame.timestamp.to_numpy(float)
    values = frame[list(channels)].to_numpy(float)
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        return len(valid)
    bad = 0
    for row in valid.itertuples():
        start = int(np.searchsorted(times, row.window_start_s - 1e-6, side="left"))
        stop = int(np.searchsorted(times, row.window_start_s + 1e-6, side="right"))
        good = False
        for idx in range(start, stop):
            chunk = times[idx:idx+120]
            signal = values[idx:idx+120]
            if (len(chunk) == 120 and abs(chunk[0] - row.window_start_s) <= 1e-6
                    and abs(chunk[-1] - row.window_end_s) <= 1e-6
                    and np.isfinite(signal).all()
                    and np.all(np.abs(np.diff(chunk) - 1/60) <= 1e-4)
                    and row.available_at_s + 1e-8 >= row.window_end_s):
                good = True
                break
        if not good:
            bad += 1
    return bad


def compare_release_example(models, package):
    """All four exact weights must reproduce the published sensor-only example."""
    example = pd.read_csv(package / "examples/synthetic_imu.csv")
    result = []
    for name, model in models.items():
        actual = model.predict_frame(example)
        expected = pd.read_csv(package / "examples" / (
            "direction_expected.csv" if name == "current_direction" else name + "_expected.csv"))
        if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
            raise ValueError(f"{name} synthetic example schema or row count mismatch")
        for col in actual:
            a, b = actual[col], expected[col]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float),
                                           rtol=0, atol=1e-7, equal_nan=True)
            elif not a.fillna("<missing>").astype(str).equals(b.fillna("<missing>").astype(str)):
                raise ValueError(f"{name} synthetic example differs in {col}")
        result.append({"model": name, "rows": len(actual), "status": "PASS"})
    return result


def run(plan, package, output):
    manifest = check_plan(plan)
    if output.exists():
        raise ValueError("Preserve existing evidence: choose a new output directory")
    torch.set_num_threads(1)
    models, hashes = load_frozen(package / "fog-selected", package / "a2-current/current/model.joblib")
    smoke = compare_release_example(models, package)
    output.mkdir(parents=True)
    predictions_dir = output / "runtime"
    predictions_dir.mkdir()
    sample_table = pd.read_csv(plan / "sample_manifest.csv")
    protocol = manifest["protocol"]
    aggregate, events, runs, alignments, fused_rows = [], [], [], [], []
    for clip_index, sample in enumerate(sample_table.itertuples()):
        sensor = pd.read_csv(plan / "clips" / f"{sample.clip}_sensors.csv")
        sample_truth = pd.read_csv(plan / "clips" / f"{sample.clip}_truth.csv")
        window_truth = pd.read_csv(plan / "clips" / f"{sample.clip}_window_truth.csv")
        if {"fog", "activity", "fog_severity"} & set(sensor):
            raise ValueError("Truth present in inference sensor input")
        for condition_index, condition in enumerate(protocol["perturbations"]):
            seed = protocol["seed"] + clip_index * 1009 + condition_index * 131
            signal = mutate_sensors(sensor, condition, seed, protocol)
            runtimes = {}
            for name, model in models.items():
                started = time.perf_counter()
                runtime, exception = None, None
                try:
                    runtime = model.predict_frame(signal)
                except (ValueError, KeyError, IndexError, TypeError) as error:
                    exception = f"{type(error).__name__}: {error}"
                elapsed = time.perf_counter() - started
                bad = valid_source_windows(signal, runtime,
                    model.channels if name == "current_direction" else model.channel_names
                ) if runtime is not None else 0
                count = len(runtime) if runtime is not None else 0
                rejected = exception is not None or count == 0
                status = ("REJECTED_EXPECTED" if condition in INVALID_INPUTS and rejected
                          else "INVALID_INPUT_ACCEPTED" if condition in INVALID_INPUTS
                          else "NO_VALID_OUTPUT" if rejected else "INVALID_SOURCE_OUTPUT" if bad
                          else "OUTPUT")
                runs.append({"clip": sample.clip, "condition": condition, "model": name,
                             "status": status, "output_rows": count,
                             "invalid_source_outputs": bad, "seconds": elapsed,
                             "exception": exception or ""})
                if condition in INVALID_INPUTS:
                    if runtime is not None and not runtime.empty:
                        runtime.to_csv(predictions_dir / f"{sample.clip}_{condition}_{name}.csv", index=False)
                    continue
                if runtime is not None:
                    runtime.to_csv(predictions_dir / f"{sample.clip}_{condition}_{name}.csv", index=False)
                    runtimes[name] = runtime
                attached, outside = attach_to_truth(window_truth, runtime,
                    kind="direction" if name == "current_direction" else "fog")
                runs[-1]["scored_supported_windows"] = int(attached.supported.sum())
                if not attached.supported.any() and runs[-1]["status"] == "OUTPUT":
                    runs[-1]["status"] = "NO_SCORED_OUTPUT"
                attached["clip"] = sample.clip
                attached["condition"] = condition
                attached["model"] = name
                aggregate.append(attached)
                alignments.append({"clip": sample.clip, "condition": condition, "model": name,
                                   "out_of_grid_scored_predictions": outside})
                if name != "current_direction":
                    events.extend({"clip": sample.clip, "condition": condition, "model": name, **row}
                                  for row in event_metrics(sample_truth, attached,
                                                           sample.score_start_s, sample.score_stop_s))
            # Evaluate original and fused direction on the same source-arrival grid.
            if condition == "clean" and {"bilateral_rf", "current_direction"} <= set(runtimes):
                fog, direction = runtimes["bilateral_rf"], runtimes["current_direction"]
                if len(fog) and len(direction):
                    fused = fuse_perception(fog, direction)
                    at_direction = at_source_arrivals(fused, direction)
                    # Model window_end is input end; fuse_perception alias is decision time.
                    at_direction["window_start_s"] = direction.window_start_s.to_numpy()
                    at_direction["window_end_s"] = direction.window_end_s.to_numpy()
                    at_direction["input_valid"] = direction.input_valid.to_numpy()
                    at_direction["direction"] = at_direction["turn"]
                    scored, outside = attach_to_truth(window_truth, at_direction, kind="direction")
                    scored["clip"], scored["condition"], scored["model"] = sample.clip, condition, "fused_direction"
                    fused_rows.append(scored)
                    alignments.append({"clip": sample.clip, "condition": condition,
                                       "model": "fused_direction", "out_of_grid_scored_predictions": outside})
        print(f"Completed {clip_index + 1}/{len(sample_table)} clips", flush=True)
    table = pd.concat([*aggregate, *fused_rows], ignore_index=True)
    table.to_csv(output / "scored_windows.csv", index=False)
    pd.DataFrame(runs).to_csv(output / "run_status.csv", index=False)
    pd.DataFrame(events).to_csv(output / "event_details.csv", index=False)
    pd.DataFrame(alignments).to_csv(output / "alignment.csv", index=False)
    metrics = []
    for (name, condition), frame in table.groupby(["model", "condition"], sort=True):
        value = direction_metrics(frame) if "direction" in name else fog_metrics(frame)
        metrics.append({"model": name, "condition": condition, **value})
    paired = []
    for name, frame in table.groupby("model"):
        clean = frame.loc[frame.condition.eq("clean") & frame.supported].copy()
        for condition in ("noise_low", "noise_high", "right_gyro_gap"):
            other = frame.loc[frame.condition.eq(condition) & frame.supported].copy()
            keys = ["clip", "window_start_s", "window_end_s"]
            pair = clean.merge(other, on=keys, suffixes=("_clean", "_perturbed"), validate="one_to_one")
            if pair.empty:
                continue
            truth_col = "truth_direction" if "direction" in name else "truth_fog_label"
            if not pair[f"{truth_col}_clean"].equals(pair[f"{truth_col}_perturbed"]):
                raise ValueError("Paired target changed")
            if "direction" in name:
                pair = pair.loc[pair.truth_direction_clean.ne("unknown")]
            paired.append({"model": name, "condition": condition, "paired_windows": len(pair),
                           "changed_predictions": int(pair.predicted_clean.ne(pair.predicted_perturbed).sum()),
                           "clean_accuracy": float(pair.predicted_clean.eq(pair[f"{truth_col}_clean"]).mean())
                           if len(pair) else None,
                           "perturbed_accuracy": float(pair.predicted_perturbed.eq(pair[f"{truth_col}_clean"]).mean())
                           if len(pair) else None})
    _write_json(output / "metrics.json", metrics)
    pd.DataFrame(paired).to_csv(output / "paired_perturbations.csv", index=False)
    run_frame = pd.DataFrame(runs)
    anomaly = run_frame.status.isin(["INVALID_INPUT_ACCEPTED", "INVALID_SOURCE_OUTPUT"])
    no_runtime = run_frame.status.eq("NO_VALID_OUTPUT") & run_frame.condition.isin(
        ["clean", "noise_low", "noise_high"])
    no_scored = run_frame.status.eq("NO_SCORED_OUTPUT") & run_frame.condition.isin(
        ["clean", "noise_low", "noise_high"])
    counts = run_frame.groupby(["model", "condition", "status"]).size().reset_index(name="n")
    counts.to_csv(output / "run_status_summary.csv", index=False)
    result = {"schema": "fogguard.b2.randomized_model_evaluation.v1",
              "scope": "Previously inspected development participants; no retraining; not independent TEST",
              "model_hashes": hashes, "sample_plan_sha256": file_hash(plan / "PLAN.json"),
              "seed": protocol["seed"], "sample_clips": len(sample_table), "planned_windows": manifest["scored_windows"],
              "truth_runtime_separated": True, "synthetic_example_check": smoke,
              "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
              "scipy": scipy.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__,
              "torch": torch.__version__,
              "abnormal_input_accepted_or_invalid_output_runs": int(anomaly.sum()),
              "invalid_emitted_windows": int(run_frame.invalid_source_outputs.sum()),
              "no_runtime_rows_in_clean_or_noise_runs": int(no_runtime.sum()),
              "no_scored_output_in_clean_or_noise_runs": int(no_scored.sum()),
              "out_of_grid_scored_predictions": int(pd.DataFrame(alignments).out_of_grid_scored_predictions.sum()),
              "metrics": metrics, "paired": paired,
              "limitations": ["Development sample selected after prior model development; not fresh blind test",
                              "Windows are correlated and clips are not population-representative",
                              "Time latency is offline convention, not measured processing or hardware latency",
                              "Noise levels are engineering perturbations, not validated real sensor noise",
                              "Model accuracy is not cue effectiveness or patient treatment efficacy"]}
    _write_json(output / "RESULTS.json", result)
    print(json.dumps({"clips": len(sample_table), "runs": len(runs), "invalid_source": int(anomaly.sum()),
                      "no_scored_output": int(no_scored.sum())}, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.package, args.output_dir)

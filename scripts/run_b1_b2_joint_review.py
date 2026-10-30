"""Replay real A1/A2 outputs with a movement proxy and compare cueing strategies."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from fogguard.cueing.proxy_replay import replay_proxy_guidance
from fogguard.evaluation.automation2 import audit_commands
from fogguard.evaluation.randomized_plan import file_hash
from fogguard.evaluation.randomized_receiver import receiver_issues
from fogguard.motion.proxy_feedback import GYRO, MotionProxyConfig, estimate_motion_feedback
from fogguard.perception import fuse_perception

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("fixed", "countdown", "march")


def _sha(path):
    return file_hash(path)


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")


def _check_inputs(plan, model_run, package):
    frozen = json.loads((plan / "PLAN.json").read_text("utf-8"))
    model_result = json.loads((model_run / "RESULTS.json").read_text("utf-8"))
    lock = json.loads((ROOT / "docs/a_group/TEST_LOCK.json").read_text("utf-8"))
    if frozen.get("schema") != "fogguard.b2.randomized_plan.v1":
        raise ValueError("Expected a fixed randomized plan")
    if frozen["protocol"]["partition"] != "validation":
        raise ValueError("The joint replay accepts this development sample only")
    if _sha(plan / "PLAN.json") != model_result["sample_plan_sha256"]:
        raise ValueError("Model run and sample plan do not match")
    for relative, expected in frozen["csv_sha256"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or _sha(plan / path) != expected:
            raise ValueError(f"Sample input hash mismatch: {relative}")
    for name, expected in model_result["model_hashes"].items():
        path = (package / "a2-current/current/model.joblib" if name == "current_direction" else
                package / "fog-selected" / name / "model.joblib")
        if _sha(path) != expected or expected != lock["files_sha256"]["models/" + name]:
            raise ValueError(f"The published model hash changed: {name}")
    return frozen, model_result


def _clock(sensors, feedback):
    """A decision tick every 0.1 seconds and at every real proxy availability."""
    last = float(sensors.timestamp.iloc[-1])
    values = np.r_[sensors.timestamp.iloc[::6].to_numpy(float),
                   feedback.feedback_available_at_s.to_numpy(float), last]
    values = np.unique(np.round(values[values <= last + 1e-9], 9))
    return pd.DataFrame({"subject_id": feedback.subject_id.iloc[0],
                         "session_id": feedback.session_id.iloc[0],
                         "task_id": feedback.task_id.iloc[0], "timestamp_s": values})


def _finite_duration(commands):
    active = commands.loc[~commands.guidance_terminal]
    times = active.guidance_timestamp_s.to_numpy(float)
    elapsed = np.diff(times, append=times[-1])
    if (elapsed < 0).any():
        raise ValueError("Command time reversed")
    return float(np.sum(elapsed * active.guidance_visual_enabled.to_numpy(bool)))


def _one_case(sensor, fog, direction, config, folder, *, case, origin):
    for frame in (sensor, fog, direction):
        if {"fog", "activity", "fog_severity"} & set(frame) or any(
            str(name).startswith("truth_") for name in frame):
            raise ValueError("Runtime input unexpectedly contains labels")
    feedback = estimate_motion_feedback(sensor, config)
    clock = _clock(sensor, feedback)
    fog = fog.copy()
    if "input_valid" not in fog:
        fog["input_valid"] = True
    perception = fuse_perception(fog, direction, clock=clock)
    # End-of-record is an observation boundary, not unlimited future holding.
    perception = perception.loc[perception.available_at_s.le(
        float(sensor.timestamp.iloc[-1]) + 1e-8)].reset_index(drop=True)
    issues = receiver_issues(fog, direction, perception)
    spoofed = fuse_perception(fog.assign(truth_fog_label=1),
                              direction.assign(truth_direction="right"), clock=clock)
    spoofed = spoofed.loc[spoofed.available_at_s.le(
        float(sensor.timestamp.iloc[-1]) + 1e-8)].reset_index(drop=True)
    truth_invariance = spoofed.equals(perception)
    folder.mkdir(parents=True)
    perception.to_csv(folder / "perception_runtime.csv", index=False)
    feedback.to_csv(folder / "feedback_runtime.csv", index=False)
    results = []
    for strategy in STRATEGIES:
        commands = replay_proxy_guidance(perception, feedback, strategy=strategy,
                                         proxy_config=config, scenario_id=case)
        commands.to_csv(folder / f"commands_{strategy}.csv", index=False)
        lifecycle = audit_commands(commands)
        unsafe = commands.guidance_visual_enabled & (
            ~commands.input_fresh | commands.turn.eq("unknown"))
        if strategy != "fixed":
            unsafe |= commands.guidance_visual_enabled & ~commands.proxy_feedback_valid_now
        starts = commands.loc[commands.guidance_action.eq("start")]
        stops = commands.loc[commands.guidance_action.eq("stop")]
        results.append({"case": case, "origin": origin, "strategy": strategy,
                        "starts": len(starts), "stops": len(stops),
                        "active_s": _finite_duration(commands),
                        "fallback_entries": int(commands.guidance_reason.isin(
                            ["valid_no_motion_enter_march", "valid_no_motion_enter_countdown"]).sum()),
                        "fallback_timeout_returns": int(commands.guidance_reason.eq(
                            "fallback_timeout_return_forward_unverified").sum()),
                        "proxy_fade_entries": int(commands.guidance_reason.eq(
                            "perception_recovery_with_motion_proxy_begin_fade").sum()),
                        "actual_steps_claimed": int((commands.guidance_stable_steps != 0).sum()),
                        "lifecycle_errors": len(lifecycle), "unsafe_active_rows": int(unsafe.sum()),
                        "stop_reasons": json.dumps(stops.guidance_reason.value_counts().to_dict()),
                        "receiver_errors": len(issues), "truth_invariant": truth_invariance})
    info = {"case": case, "origin": origin, "sensor_samples": len(sensor),
            "perception_rows": len(perception), "receiver_errors": issues,
            "truth_invariant": truth_invariance,
            "feedback_counts": feedback.feedback_state.value_counts().to_dict(),
            "feedback_source": "derived_proxy", "feedback_kind": "movement_only",
            "actual_step_or_foot_position_truth": "NOT_AVAILABLE"}
    _write(folder / "case_summary.json", info)
    return results, info


def _synthetic_fixture():
    """Deterministic example: persistent confirmed event and valid stationary gyro."""
    ends = np.arange(2, 22, 0.5)
    common = {"subject_id": "fixture", "session_id": "1", "task_id": "1",
              "window_start_s": ends - 119/60, "window_end_s": ends,
              "available_at_s": ends, "model_version": "fixture-v1",
              "prediction_source": "constructed_fixture"}
    fog = pd.DataFrame(common).assign(fog_probability=0.9, fog_threshold=0.5, is_fog=True)
    direction = pd.DataFrame(common).assign(direction="straight", direction_confidence=0.9,
                                             direction_task="current", horizon_s=0.0, input_valid=True)
    times = np.arange(22 * 60) / 60
    sensors = pd.DataFrame({"timestamp": times, "subjectID": "fixture", "sessionID": "1", "taskID": "1"})
    for name in GYRO:
        sensors[name] = 0.0
    return sensors, fog, direction


def run(plan, model_run, package, output):
    if output.exists():
        raise ValueError("Use a fresh output directory; never replace existing evidence")
    frozen, model_result = _check_inputs(plan, model_run, package)
    config_path = ROOT / "config/b1_motion_feedback.json"
    config = MotionProxyConfig.load(config_path)
    output.mkdir(parents=True)
    samples = pd.read_csv(plan / "sample_manifest.csv")
    all_results, all_info, input_hashes, truth_files = [], [], {}, []
    for index, sample in enumerate(samples.itertuples()):
        sensor_path = plan / "clips" / f"{sample.clip}_sensors.csv"
        truth_path = plan / "clips" / f"{sample.clip}_truth.csv"
        fog_path = model_run / "runtime" / f"{sample.clip}_clean_bilateral_rf.csv"
        direction_path = model_run / "runtime" / f"{sample.clip}_clean_current_direction.csv"
        for name, path in (("sensor", sensor_path), ("fog", fog_path), ("direction", direction_path),
                           ("truth_evaluation_only", truth_path)):
            input_hashes[f"{sample.clip}/{name}"] = _sha(path)
        sensor, fog, direction = pd.read_csv(sensor_path), pd.read_csv(fog_path), pd.read_csv(direction_path)
        result, info = _one_case(sensor, fog, direction, config, output / "real" / sample.clip,
                                 case=sample.clip, origin="real_imu_model_inference_and_gyro_proxy")
        truth = pd.read_csv(truth_path)
        truth.insert(0, "case", sample.clip)
        truth_files.append(truth)
        all_results.extend(result)
        all_info.append(info)
        if (index + 1) % 4 == 0:
            print(f"Real records reviewed: {index+1}/{len(samples)}", flush=True)
    sensor, fog, direction = _synthetic_fixture()
    synthetic_results, synthetic_info = _one_case(sensor, fog, direction, config,
        output / "synthetic" / "no_movement", case="synthetic_no_movement", origin="synthetic_fixture")
    table = pd.DataFrame([*all_results, *synthetic_results])
    table.to_csv(output / "comparison_by_case.csv", index=False)
    pd.concat(truth_files, ignore_index=True).to_csv(output / "evaluation_truth.csv", index=False)
    summary = table.groupby(["origin", "strategy"], as_index=False).agg(
        cases=("case", "count"), starts=("starts", "sum"), stops=("stops", "sum"),
        active_s=("active_s", "sum"), fallback_entries=("fallback_entries", "sum"),
        fallback_timeout_returns=("fallback_timeout_returns", "sum"),
        proxy_fade_entries=("proxy_fade_entries", "sum"),
        actual_steps_claimed=("actual_steps_claimed", "sum"),
        lifecycle_errors=("lifecycle_errors", "sum"), unsafe_active_rows=("unsafe_active_rows", "sum"),
        receiver_errors=("receiver_errors", "sum"), truth_invariant=("truth_invariant", "all"))
    summary.to_csv(output / "comparison_summary.csv", index=False)
    failures = table.loc[(table.lifecycle_errors != 0) | (table.unsafe_active_rows != 0)
                         | (table.receiver_errors != 0) | ~table.truth_invariant]
    feedback_counts = {state: sum(info["feedback_counts"].get(state, 0) for info in all_info)
                       for state in ("movement_evidence", "no_movement_evidence", "unknown")}
    source_names = ["config/b1_motion_feedback.json", "src/fogguard/motion/proxy_feedback.py",
                    "src/fogguard/cueing/proxy_replay.py", "scripts/run_b1_b2_joint_review.py",
                    "src/fogguard/evaluation/randomized_receiver.py", "src/fogguard/perception.py",
                    "src/fogguard/context/fogstar.py", "src/fogguard/cueing/guidance.py",
                    "src/fogguard/cueing/contracts.py", "src/fogguard/cueing/targets.py",
                    "src/fogguard/evaluation/automation2.py"]
    result = {"schema": "fogguard.b1_b2.joint_review.v1", "real_cases": len(samples),
              "synthetic_cases": 1, "strategies": list(STRATEGIES),
              "status": "ENGINEERING_CHECKS_PASS" if failures.empty else "ENGINEERING_CHECKS_FAIL",
              "engineering_failures": len(failures), "real_feedback_counts": feedback_counts,
              "real_case_summaries": all_info, "synthetic_case_summary": synthetic_info,
              "comparison": json.loads(summary.to_json(orient="records")),
              "proxy_config": config.as_dict(), "source_sha256": {name: _sha(ROOT/name) for name in source_names},
              "input_sha256": input_hashes, "model_hashes": model_result["model_hashes"],
              "sampling_plan_sha256": _sha(plan / "PLAN.json"), "python": platform.python_version(),
              "model_data_partition": frozen["protocol"]["partition"],
              "limits": ["Gyro proxy is uncalibrated and does not establish actual steps or forward motion",
                         "Timer advances only on distinct valid low-motion windows after cue activation",
                         "March targets alternate as prompts; no actual alternating footfalls are asserted",
                         "Return to forward occurs on fallback timeout and is explicitly unverified",
                         "Adaptive spacing is unavailable without actual foot placement",
                         "Real input is previously inspected development data, not a blind TEST",
                         "Software outputs are not verified patient recovery or clinical efficacy"]}
    _write(output / "RESULTS.json", result)
    files = {str(p.relative_to(output)).replace("\\", "/"): _sha(p)
             for p in sorted(output.rglob("*")) if p.is_file()}
    _write(output / "MANIFEST.json", {"schema": "fogguard.b1_b2.output_manifest.v1", "files_sha256": files})
    print(json.dumps({"status": result["status"], "real_cases": len(samples),
                      "failures": len(failures), "feedback_counts": feedback_counts}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outcome = run(args.plan, args.model_run, args.package, args.output_dir)
    raise SystemExit(int(outcome["status"] != "ENGINEERING_CHECKS_PASS"))

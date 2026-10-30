"""Audit and standardize saved B1/B2 outputs without running model decisions."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["subject_id", "session_id", "task_id"]
STRATEGIES = ["fixed", "feedback", "countdown", "march", "no_fade"]
REQUIRED = {
    "perception_runtime.csv": KEYS + ["available_at_s", "input_fresh", "turn",
        "event_phase", "fog_observation_is_new", "fog_model_version",
        "direction_model_version", "cue_test_input_origin"],
    "feedback_runtime.csv": KEYS + ["feedback_available_at_s", "feedback_window_start_s",
        "feedback_window_end_s", "feedback_state", "feedback_valid", "feedback_source",
        "feedback_kind", "feedback_reason", "observed_duration_s"],
    "commands.csv": KEYS + ["guidance_timestamp_s", "guidance_action",
        "guidance_reason", "guidance_visual_enabled", "guidance_audio_enabled",
        "guidance_targets", "guidance_terminal", "proxy_no_motion_valid_s",
        "proxy_observation_is_new", "proxy_feedback_valid_now", "guidance_stable_steps"],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def verify_manifest(root):
    """Verify a bounded manifest; never follow a path outside the bundle."""
    root = Path(root).resolve()
    files = json.loads((root / "MANIFEST.json").read_text("utf-8"))["files_sha256"]
    for name, digest in files.items():
        target = (root / name).resolve()
        if not target.is_relative_to(root) or not target.is_file() or sha256(target) != digest:
            raise ValueError(f"Manifest mismatch: {name}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p != root / "MANIFEST.json"}
    if actual - set(files):
        raise ValueError(f"Unlisted files: {sorted(actual - set(files))[:3]}")
    return len(files)


def seal(root):
    files = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p != root / "MANIFEST.json"}
    write_json(root / "MANIFEST.json", {"schema": "fogguard.b2.handoff.manifest.v1",
                                        "files_sha256": files})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_case(perception, feedback, commands, strategy):
    """Independent saved-row checks, with explicit non-evaluable step metrics."""
    frames = {"perception_runtime.csv": perception, "feedback_runtime.csv": feedback,
              "commands.csv": commands}
    for name, frame in frames.items():
        _require(not frame.empty and set(REQUIRED[name]) <= set(frame), f"Schema: {name}")
        _require(not frame[KEYS].isna().any().any(), f"Missing identity: {name}")
        _require(len(frame[KEYS].drop_duplicates()) == 1, f"Mixed records: {name}")
        _require(not any(c.startswith("truth_") or c in {"fog", "activity", "fog_severity"}
                         for c in frame), f"Runtime labels: {name}")
        _require(frame[KEYS].astype(str).iloc[0].equals(
            perception[KEYS].astype(str).iloc[0]), f"Record mismatch: {name}")
    for frame, col in [(perception, "available_at_s"),
                       (feedback, "feedback_available_at_s")]:
        values = frame[col].to_numpy(float)
        _require(np.isfinite(values).all() and (np.diff(values) > 0).all(), f"Clock: {col}")
    _require(feedback.feedback_source.eq("derived_proxy").all()
             and feedback.feedback_kind.eq("movement_only").all(), "Proxy provenance")
    _require((feedback.feedback_valid == feedback.feedback_state.ne("unknown")).all(),
             "Feedback validity/state mismatch")
    _require(feedback.loc[~feedback.feedback_valid, "observed_duration_s"].eq(0).all(),
             "Unknown feedback accumulated observation")
    active = commands.loc[~commands.guidance_terminal].reset_index(drop=True)
    _require(len(active) == len(perception), "Decision row count")
    for col in perception:
        pd.testing.assert_series_equal(active[col], perception[col].reset_index(drop=True),
                                       check_names=False, check_dtype=False)
    t = active.guidance_timestamp_s.to_numpy(float)
    _require(np.isfinite(t).all() and (np.diff(t) > 0).all(), "Command clock")
    _require(np.allclose(t, perception.available_at_s, rtol=0, atol=1e-8), "Clock alignment")
    on = active.guidance_visual_enabled
    _require(not (on & (~active.input_fresh | active.turn.eq("unknown"))).any(),
             "Active cue on stale/unknown perception")
    if strategy != "fixed":
        _require(not (on & ~active.proxy_feedback_valid_now).any(), "Active cue without feedback")
    _require((active.guidance_audio_enabled == on).all(), "Audio/visual mismatch")
    _require(commands.guidance_stable_steps.eq(0).all(), "Invented step completion")
    _require(not commands.iloc[-1].guidance_visual_enabled
             and not commands.iloc[-1].guidance_audio_enabled, "Missing terminal withdrawal")
    starts = commands.loc[commands.guidance_action.eq("start")]
    stops = commands.loc[commands.guidance_action.eq("stop")]
    _require(set(starts.guidance_command_id) == set(stops.guidance_command_id)
             and starts.guidance_command_id.is_unique and stops.guidance_command_id.is_unique,
             "Unclosed/duplicate command lifecycle")
    counter = active.proxy_no_motion_valid_s.to_numpy(float)
    previous = np.r_[0., counter[:-1]]
    inc = counter > previous + 1e-7
    _require(active.loc[inc, "proxy_observation_is_new"].all()
             and active.loc[inc, "proxy_feedback_valid_now"].all()
             and active.loc[inc, "proxy_feedback_state"].eq("no_movement_evidence").all(),
             "Invalid no-response timer increment")
    for row in active.loc[inc].itertuples(index=False):
        f = feedback.loc[np.isclose(feedback.feedback_available_at_s,
                                   row.guidance_timestamp_s, rtol=0, atol=1e-7)]
        _require(len(f) == 1 and f.feedback_window_start_s.iloc[0]
                 >= row.guidance_cue_started_at_s - 1e-7, "Pre-cue observation counted")
    dt = np.diff(t, append=t[-1])
    eligible = active.input_fresh & active.stable_fog_detected & active.turn.ne("unknown")
    delay = None
    if len(starts) and eligible.any():
        delay = float(starts.guidance_timestamp_s.iloc[0]
                      - active.loc[eligible, "guidance_timestamp_s"].iloc[0])
    return {"status": "PASS", "rows": len(commands), "starts": len(starts),
            "stops": len(stops), "active_s": float(np.sum(dt * on)),
            "first_prompt_delay_s": delay,
            "delay_definition": "First start minus first fresh confirmed FoG with known turn; "
                                "null if no start/eligible input; not physical onset latency",
            "max_valid_low_motion_s": float(counter.max()),
            "control_state_changes": int(active.guidance_control_state.ne(
                active.guidance_control_state.shift()).sum() - 1),
            "new_valid_timer_increments": int(inc.sum()),
            "fallback_entries": int(active.guidance_reason.isin(
                ["valid_no_motion_enter_march", "valid_no_motion_enter_countdown"]).sum()),
            "timeout_returns": int(active.guidance_reason.eq(
                "fallback_timeout_return_forward_unverified").sum()),
            "step_accuracy": None, "forward_success_rate": None,
            "unavailable_reason": "No independently labelled steps or measured positions"}


def build_bundle(run, output, *, plan=None):
    """Copy saved data into a fresh private bundle; preserve original CSV bytes."""
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh output directory; prior evidence is preserved")
    verified = verify_manifest(run)
    result = json.loads((run / "RESULTS.json").read_text("utf-8"))
    saved_metrics = pd.read_csv(run / "runs.csv")
    if plan is not None:
        plan = Path(plan).resolve()
        _require(sha256(plan / "PLAN.json") == result["plan_sha256"], "Sampling plan mismatch")
    output.mkdir(parents=True)
    rows, fields, checks = [], [], []
    for kind in ("real", "constructed"):
        for source in sorted((run / kind).iterdir()):
            if not source.is_dir():
                continue
            target = output / kind / source.name
            target.mkdir(parents=True)
            p = pd.read_csv(source / "perception_runtime.csv")
            f = pd.read_csv(source / "feedback_runtime.csv")
            for name in ("perception_runtime.csv", "feedback_runtime.csv"):
                shutil.copyfile(source / name, target / name)
            for name, frame in [("perception_runtime.csv", p), ("feedback_runtime.csv", f)]:
                fields.extend({"file": name, "field": c, "dtype": str(frame[c].dtype),
                               "required": c in REQUIRED[name]} for c in frame)
            metrics = []
            for strategy in STRATEGIES:
                c = pd.read_csv(source / f"commands_{strategy}.csv")
                stats = audit_case(p, f, c, strategy)
                metrics.append({"strategy": strategy, **stats})
                checks.append({"case": source.name, "origin": kind, "strategy": strategy, **stats})
                if strategy == "march":
                    shutil.copyfile(source / "commands_march.csv", target / "commands.csv")
                    fields.extend({"file": "commands.csv", "field": col,
                                   "dtype": str(c[col].dtype), "required": col in REQUIRED["commands.csv"]}
                                  for col in c)
                else:
                    (target / "comparisons").mkdir(exist_ok=True)
                    shutil.copyfile(source / f"commands_{strategy}.csv",
                                    target / "comparisons" / f"commands_{strategy}.csv")
            truth = None
            if kind == "real" and plan is not None:
                truth = plan / "clips" / f"{source.name}_truth.csv"
                _require(sha256(truth) == result["input_sha256"][
                    f"{source.name}/truth_evaluation_only"], "Evaluation truth mismatch")
                y = pd.read_csv(truth)
                for raw, key in zip(["subjectID", "sessionID", "taskID"], KEYS, strict=True):
                    _require(y[raw].astype(str).eq(str(p[key].iloc[0])).all(), "Truth record mismatch")
                shutil.copyfile(truth, target / "evaluation_truth.csv")
            write_json(target / "metrics.json", {"primary_strategy": "march", "strategies": metrics,
                "saved_evaluation_metrics": json.loads(saved_metrics.loc[
                    saved_metrics.case.eq(source.name)].to_json(orient="records")),
                "truth_status": "separate_evaluation_only" if truth else "not_provided",
                "source_kind": kind, "derived_proxy_not_measured_steps": True})
            write_json(target / "config.json", {"motion": result["motion_config"],
                "guidance": result["base_guidance_config"], "primary_strategy": "march",
                "return_rule": "timeout_return_to_try_forward_not_verified_step_completion"})
            seal(target)
            rows.append({"case": source.name, "origin": kind,
                         "path": target.relative_to(output).as_posix(),
                         "evaluation_truth": truth is not None, "strategies": 5})
    for name in ("runs.csv", "summary.csv", "paired_comparisons.csv"):
        shutil.copyfile(run / name, output / name)
    pd.DataFrame(rows).to_csv(output / "EXPORT_INVENTORY.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "CASE_AUDIT.csv", index=False)
    pd.DataFrame(fields).drop_duplicates().to_csv(output / "FIELD_CATALOG.csv", index=False)
    write_json(output / "metrics.json", {"cases": len(rows), "strategy_runs": len(checks),
        "command_rows": sum(c["rows"] for c in checks), "engineering_audit": "PASS",
        "model_classification": "Use the separate locked A-group report and model-update evaluation",
        "new_independent_test": False, "step_accuracy": None,
        "forward_success_rate": None, "clinical_benefit": None,
        "not_implemented": "Observed alternating footstep completion",
        "per_case_metrics": "real/*/metrics.json and constructed/*/metrics.json"})
    write_json(output / "PROVENANCE.json", {"schema": "fogguard.b2.handoff.v1",
        "source_manifest_sha256": sha256(run / "MANIFEST.json"),
        "source_files_verified": verified, "model_hashes": result["model_hashes"],
        "source_sha256": result["source_sha256"], "plan_sha256": result["plan_sha256"],
        "exporter_sha256": sha256(__file__), "cases": len(rows), "runs": len(checks),
        "scope": "Private derived records; do not commit patient-level exports",
        "not_implemented": ["independently observed alternating footstep completion"],
        "not_evaluable": ["step accuracy", "forward displacement success", "clinical benefit"],
        "not_run": ["final industrial-design playback acceptance", "new-machine reproduction"]})
    (output / "README.md").write_text(
        "# B2 standard saved-output bundle\n\n"
        "Private derived records: do not upload this directory to GitHub. No raw IMU or weights.\n"
        "Each case contains perception_runtime.csv, feedback_runtime.csv, commands.csv (march), "
        "metrics.json, config.json and MANIFEST.json. Other strategies are in comparisons/.\n"
        "Real cases optionally contain evaluation_truth.csv, copied separately; controllers never "
        "read it. Constructed cases have no patient truth. Original identities and seconds are "
        "preserved locally. Feedback uses non-overlapping window endpoints, perception uses its "
        "decision clock, commands follow that clock; do not join by row number.\n"
        "feedback_state uses movement_evidence/no_movement_evidence/unknown, not verified steps.\n"
        "Fallback returns on timeout; observed alternating-step completion is not implemented.\n"
        "Use EXPORT_INVENTORY.csv, FIELD_CATALOG.csv, CASE_AUDIT.csv and PROVENANCE.json.\n"
        "Verify with: python scripts/build_b2_handoff.py --verify <this-directory>\n",
        encoding="utf-8")
    seal(output)
    files = verify_manifest(output)
    return {"status": "ENGINEERING_CHECKS_PASSED_WITH_LIMITS", "cases": len(rows),
            "strategy_runs": len(checks), "files_verified": files,
            "command_rows": sum(c["rows"] for c in checks)}

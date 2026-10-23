"""One-command offline evaluation, preserving all upstream modules and defaults.

Run from the repository root after pip install -e . . Optional real-data mode
requires subject-held-out predictions plus their matching training report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from fogguard.context import FogStarIntegrationConfig
from fogguard.cueing import AdaptiveCueingController
from fogguard.cueing.contracts import GuidanceConfig
from fogguard.cueing.replay import replay_guidance, summarize_guidance
from fogguard.cueing.scenarios import build_guidance_cases
from fogguard.data.responses import load_step_responses
from fogguard.detection import FreezeIndexDetector
from fogguard.evaluation.automation2 import (
    SEQUENCE,
    audit_commands,
    evaluate_timeline,
)
from fogguard.evaluation.automation2_acceptance import acceptance_summary, check_acceptance
from fogguard.fogstar_pipeline import integrate_fogstar_predictions
from fogguard.motion import AccelerationMotionRecognizer
from fogguard.pipeline import replay_csv

DEFAULT_SCENARIOS = {
    name: Path("data/demo") / filename for name, filename in (
        ("normal", "normal_gait.csv"), ("straight_fog", "straight_fog.csv"),
        ("turning_fog", "turning_fog.csv"), ("right_turning_fog", "right_turning_fog.csv"),
        ("normal_right_turn", "normal_right_turn.csv"),
        ("voluntary_stop", "voluntary_stop.csv"), ("fog_recovery", "fog_recovery.csv"),
    )
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sensor_candidates(sensor: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Explicit adapter for synthetic FreezeIndex results, NOT a FoG-STAR model.

    Copies detector/turn outputs; labels are carried as annotations only.
    Subject/session identify synthetic files and cannot be read as participants.
    """
    result = sensor.copy()
    result["subject_id"] = "synthetic-demo"
    result["session_id"] = scenario
    result["task_id"] = scenario
    result["scenario_id"] = scenario
    result["is_fog"] = result.fog_detected
    result["direction"] = result.turn
    result["direction_confidence"] = result.turn_confidence
    result["fog_threshold"] = 0.21
    result["source_dataset"] = "synthetic_freeze_index_adapter"
    return result


def validate_oof_provenance(predictions: pd.DataFrame, report: dict) -> None:
    """Reject obvious subject leakage; paired report is declared provenance.

    Report structure alone cannot prove predictions were not subsequently altered;
    input/report hashes are recorded, and local training reproduction is preferable.
    """
    folds = report["validation"]["fold_subjects"]
    tested = set()
    for fold in folds:
        train = set(map(str, fold["train"]))
        test = set(map(str, fold["test"]))
        if train & test or tested & test:
            raise ValueError("Subject leakage or repeated held-out subjects in training report")
        tested.update(test)
    if tested != set(predictions.subject_id.astype(str)):
        raise ValueError("OOF subjects do not match the training report")
    if report["dataset"]["windows"] != len(predictions):
        raise ValueError("OOF window count does not match the training report")


def evaluate_case(name, events, responses, config, destination, rows, all_issues,
                  *, variants=True):
    directory = destination / "timelines" / name
    directory.mkdir(parents=True, exist_ok=True)
    events.to_csv(directory / "events.csv", index=False)
    truth = events.truth_fog_label.astype(int)
    errors = events.loc[(events.is_fog.astype(int) != truth)
                        | (events.stable_fog_detected.astype(int) != truth)].copy()
    errors["raw_error"] = ["FP" if p and not t else "FN" if t and not p else "correct"
                           for t, p in zip(errors.truth_fog_label.astype(int),
                                           errors.is_fog, strict=True)]
    errors.to_csv(directory / "recognition_error_windows.csv", index=False)
    if responses is not None:
        responses.to_csv(directory / "simulated_responses.csv", index=False)
    settings = {
        "fixed": replace(config, enable_response_policy=False, enable_fading=False,
                         enable_adaptive_spacing=False),
        "response": config,
    }
    if variants:
        settings.update(no_fade=replace(config, enable_fading=False),
                        spacing=replace(config, enable_adaptive_spacing=True))
    results = {}
    for strategy, effective in settings.items():
        commands = replay_guidance(events, config=effective, responses=responses)
        commands.to_csv(directory / f"{strategy}.csv", index=False)
        issues = audit_commands(commands)
        all_issues.extend({"case": name, "strategy": strategy, **item} for item in issues)
        metrics = evaluate_timeline(events, commands)
        summary = summarize_guidance(commands)
        row = {"case": name, "strategy": strategy,
               "source": str(events.source_dataset.iloc[0]) if "source_dataset" in events
               else "supplied_model_candidates",
               "response_available": responses is not None,
               **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))},
               "raw_tp": metrics["raw"]["tp"], "raw_fp": metrics["raw"]["fp"],
               "raw_fn": metrics["raw"]["fn"], "raw_tn": metrics["raw"]["tn"],
               "stable_tp": metrics["stable"]["tp"], "stable_fp": metrics["stable"]["fp"],
               "stable_fn": metrics["stable"]["fn"], "stable_tn": metrics["stable"]["tn"],
               "active_duration_s": summary["active_duration_s"],
               "stop_commands": summary["stop_commands"], "audit_failures": len(issues),
               "unclosed_commands": summary["unclosed_commands"]}
        rows.append(row)
        results[strategy] = {"config": effective.as_dict(), "metrics": metrics,
                             "lifecycle": summary, "audit_issues": issues}
        if name == "real_oof":
            participants = []
            for subject, group in events.groupby("subject_id", sort=False):
                subject_commands = commands.loc[commands.subject_id.eq(subject)]
                entry = evaluate_timeline(group, subject_commands)
                participants.append({"subject_id": subject, **{
                    key: value for key, value in entry.items()
                    if not isinstance(value, (dict, list))},
                    **{f"raw_{k}": v for k, v in entry["raw"].items()}})
            pd.DataFrame(participants).to_csv(
                destination / f"real_subjects_{strategy}.csv", index=False)
    (directory / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/automation2"))
    parser.add_argument("--oof", type=Path)
    parser.add_argument("--training-report", type=Path)
    args = parser.parse_args()
    if bool(args.oof) != bool(args.training_report):
        parser.error("--oof and --training-report must be supplied together")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    config = GuidanceConfig.load(root / "config/guidance.yaml")
    rows = []
    issues = []
    inputs = {}
    # Seven committed IMU scenarios -> original detector/turner -> original
    # temporal integration -> original guidance controller. No truth decision path.
    for name, path in DEFAULT_SCENARIOS.items():
        inputs[str(path)] = sha256(path)
        sensor = replay_csv(
            path, detector=FreezeIndexDetector(threshold=0.21),
            motion_recognizer=AccelerationMotionRecognizer(stationary_rms_threshold_mps2=0.15),
            controller=AdaptiveCueingController(min_confidence=0.21),
        )
        candidates = sensor_candidates(sensor, name)
        for suppressed in (False, True):
            integration = FogStarIntegrationConfig(suppress_stationary_fog=suppressed)
            events = integrate_fogstar_predictions(candidates, config=integration)
            suffix = "_suppression_on" if suppressed else ""
            evaluate_case(f"sensor_{name}{suffix}", events, None, config, output,
                          rows, issues, variants=False)
    response_path = root / "data/demo/simulated_step_responses.csv"
    inputs[str(response_path.relative_to(root))] = sha256(response_path)
    for name, (events, responses) in build_guidance_cases(
        load_step_responses(response_path)
    ).items():
        evaluate_case(f"constructed_{name}", events, responses, config, output,
                      rows, issues)
    handoff_path = root / "docs/examples/robot2_predictions.csv"
    inputs[str(handoff_path.relative_to(root))] = sha256(handoff_path)
    candidates = pd.read_csv(handoff_path)
    if "scenario_id" not in candidates:
        candidates["scenario_id"] = "handoff"
    if "task_id" not in candidates:
        candidates["task_id"] = "default"
    handoff = integrate_fogstar_predictions(candidates)
    # The handoff is an interface fixture; without independent truth do not
    # manufacture accuracy numbers. It is exercised by the original demo/tests.
    handoff_commands = replay_guidance(handoff, config=config)
    handoff_commands.to_csv(output / "handoff_commands.csv", index=False)
    issues.extend({"case": "handoff", "strategy": "response", **item}
                  for item in audit_commands(handoff_commands))
    real_status = "not_run; no real-data metrics claimed"
    if args.oof:
        predictions = pd.read_csv(args.oof, dtype={"subject_id": str, "session_id": str,
                                                  "task_id": str})
        report = json.loads(args.training_report.read_text(encoding="utf-8"))
        validate_oof_provenance(predictions, report)
        inputs[str(args.oof)] = sha256(args.oof)
        inputs[str(args.training_report)] = sha256(args.training_report)
        predictions["scenario_id"] = "real_oof"
        predictions["source_dataset"] = "FoG-STAR subject-held-out predictions"
        # Batch order only; this does not invent, remove or interpolate windows.
        predictions = predictions.sort_values(SEQUENCE + ["window_end_s"], kind="stable")
        events = integrate_fogstar_predictions(predictions)
        evaluate_case("real_oof", events, None, config, output, rows, issues, variants=False)
        from fogguard.evaluation.fogstar_integration import evaluate_fogstar_integration

        legacy = evaluate_fogstar_integration(events)
        (output / "real_upstream_overlap_metrics.json").write_text(
            json.dumps(legacy, indent=2, ensure_ascii=False), encoding="utf-8")
        real_status = "run; no simulated responses; policy adaptation cannot be assessed here"
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
    (output / "audit_failures.json").write_text(
        json.dumps(issues, indent=2, ensure_ascii=False), encoding="utf-8")
    acceptance = check_acceptance(output)
    gate = acceptance_summary(acceptance)
    (output / "acceptance.json").write_text(
        json.dumps({**gate, "checks": acceptance}, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(acceptance).to_csv(output / "acceptance.csv", index=False, encoding="utf-8-sig")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                         capture_output=True, text=True, check=False)
    code_files = sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
    manifest = {
        "schema": "automation2-evaluation-v1", "base_commit": git.stdout.strip() or None,
        "platform": platform.platform(), "python": sys.version,
        "versions": {name: version(name) for name in
                     ("numpy", "pandas", "scikit-learn", "scipy", "PyYAML")},
        "input_sha256": inputs,
        "code_sha256": {str(path.relative_to(root)): sha256(path) for path in code_files},
        "guidance_config": config.as_dict(),
        "integration_config": FogStarIntegrationConfig().__dict__,
        "real_data_status": real_status,
        "handoff_lifecycle": summarize_guidance(handoff_commands),
        "case_strategy_rows": len(rows), "command_audit_failures": len(issues),
        "case_names": list(dict.fromkeys(row["case"] for row in rows)),
        "acceptance": gate,
        "scope": "Offline engineering evaluation, not treatment efficacy",
        "interpretation": [
            "Synthetic IMU adapter uses FreezeIndex, not a trained FoG-STAR model.",
            "Constructed candidates assess state transitions, not recognition accuracy.",
            "No response source means response and fixed policies have equivalent behavior.",
            "Fixed disables the whole response policy; no_fade changes only fading.",
            "Spacing changes only enable_adaptive_spacing relative to response.",
            "Terminal stops are censored endings, never successful recovery.",
            "False-window active counts include intended recovery observation tails.",
            "Strict onset matching and upstream overlap matching are reported separately.",
            "Do not tune defaults on this evaluation and then claim independent testing.",
        ]}
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluated {len(rows)} case/strategy pairs; audit failures={len(issues)}")
    print(f"Acceptance: {gate['counts']}")
    print(f"Results: {output}")
    return 1 if issues or not gate["regression_gate_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Check event-rule selection, participant isolation, causal replay and saved hashes."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def verify_selection(selection, config, people):
    if set(selection["selection_people"]) != set(people):
        raise ValueError("Wrong event-selection people")
    grid = config["threshold_grid"]
    expected = {(float(t), int(c)) for t in np.linspace(grid["start"], grid["stop"], grid["count"])
                for c in config["confirmation_choices"]}
    options = selection["options"]
    if len(options) != len(expected) or {(r["threshold"], r["confirmation_windows"]) for r in options} != expected:
        raise ValueError("Missing/changed candidate grid")
    control = max((r for r in options if r["confirmation_windows"] == 1),
                  key=lambda r: (r["fog_f1"], r["balanced_accuracy"], r["threshold"]))
    if control != selection["control"]:
        raise ValueError("Wrong F1 control selection")
    budget = control["false_alarm_s"] * config["false_duration_budget_multiplier"]
    if abs(budget - selection["false_alarm_budget_s"]) > 1e-9:
        raise ValueError("Wrong training duration budget")
    eligible = [r for r in options if r["false_alarm_s"] <= budget + 1e-6 and (
        not config["require_control_time_coverage"] or r["true_alarm_s"] >= control["true_alarm_s"] - 1e-6)]
    if len(eligible) != selection["feasible_count"]:
        raise ValueError("Wrong feasible count")
    chosen = max(eligible, key=lambda r: (r["covered_raw_events"], r["true_alarm_s"], -r["false_alarm_s"],
                                         -r["confirmation_windows"], r["threshold"]))
    if chosen != selection["selected"]:
        raise ValueError("Wrong event-objective choice")


def independent_gate(frame):
    decision, count, previous = [], 0, None
    for r in frame.itertuples():
        key = (r.subject_id, r.session_id, r.task_id)
        if previous is None or key != previous[0] or abs(r.window_end_s - previous[1] - 0.5) > 1e-6:
            count = 0
        count = count + 1 if r.fog_probability >= r.fog_threshold else 0
        decision.append(count >= r.event_confirmation_windows)
        previous = key, r.window_end_s
    return np.asarray(decision, dtype=bool)


def old_interval_check(raw, frame, repository):
    spec = importlib.util.spec_from_file_location("independent_old_intervals", repository / "recognition/diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    truth_events = covered = false_events = supported = 0
    true_s = false_s = 0.0
    for key, signal in raw.groupby(["subjectID", "sessionID", "taskID"], sort=False):
        mask = np.ones(len(frame), bool)
        for name, value in zip(["subject_id", "session_id", "task_id"], key, strict=True):
            mask &= frame[name].eq(value)
        part = frame.loc[mask].copy()
        part["is_fog"] = part.event_fog
        events, alarms, summary = module.event_rows(signal, part)
        truth_events += len(events)
        covered += sum(r["covered"] for r in events)
        supported += sum(r["model_support_s"] > 1e-6 for r in events)
        false_events += sum(not r["overlaps_truth"] for r in alarms)
        true_s += summary["true_alarm_s"]
        false_s += summary["false_alarm_s"]
    return {"truth_events": truth_events, "covered_raw_events": covered, "supported_events": supported,
            "false_alarm_events": false_events, "true_alarm_s": true_s, "false_alarm_s": false_s}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    cfg = protocol["config"]
    audits = json.loads((root / "fold_audit.json").read_text("utf-8"))
    if len(audits) != len(protocol["outer_folds"]):
        raise ValueError("Incomplete outer folds")
    for i, (audit, frozen) in enumerate(zip(audits, protocol["outer_folds"], strict=True), 1):
        train, test = set(frozen["train"]), set(frozen["test"])
        if audit["outer_train"] != frozen["train"] or audit["outer_test"] != frozen["test"] or train & test:
            raise ValueError("Wrong outer people")
        seen = set()
        for fold in audit["inner_folds"]:
            a, b = set(fold["train"]), set(fold["validation"])
            if a & b or a | b != train or (a | b) & test or b & seen:
                raise ValueError("Inner group leakage")
            seen |= b
        if seen != train:
            raise ValueError("Incomplete inner validation")
        inner = pd.read_csv(root / f"outer_{i}_inner_scores.csv")
        if set(inner.subject_id.astype(int).astype(str)) != train:
            raise ValueError("Inner scoring contains test people")
        verify_selection(audit["selection"], cfg, train)
    deploy = json.loads((root / "deployment_selection.json").read_text("utf-8"))
    all_people = {p for f in protocol["outer_folds"] for p in f["test"]}
    verify_selection(deploy["selection"], cfg, all_people)
    seen = set()
    for fold in deploy["inner_folds"]:
        a, b = set(fold["train"]), set(fold["validation"])
        if a & b or a | b != all_people or b & seen:
            raise ValueError("Deployment inner group leakage")
        seen |= b
    if seen != all_people:
        raise ValueError("Incomplete deployment inner coverage")
    raw = pd.read_csv(args.raw, usecols=["subjectID", "sessionID", "taskID", "timestamp", "fog"])
    if hashlib.sha256(args.raw.read_bytes()).hexdigest() != protocol["raw_sha256"]:
        raise ValueError("Raw input differs")
    repository = Path(__file__).resolve().parents[1]
    for i, selection in enumerate([a["selection"] for a in audits] + [deploy["selection"]], 1):
        scores = pd.read_csv(root / (f"outer_{i}_inner_scores.csv" if i <= len(audits)
                                    else "deployment_inner_scores.csv"))
        if set(scores.subject_id.astype(int).astype(str)) != set(selection["selection_people"]):
            raise ValueError("Wrong saved policy-score participants")
        training_raw = raw.loc[raw.subjectID.astype(int).astype(str).isin(selection["selection_people"])]
        for name in ("control", "selected"):
            choice = selection[name]
            replay = scores.copy()
            replay["fog_threshold"] = choice["threshold"]
            replay["event_confirmation_windows"] = choice["confirmation_windows"]
            replay["event_fog"] = independent_gate(replay)
            checked = old_interval_check(training_raw, replay, repository)
            for metric in ["truth_events", "covered_raw_events", "supported_events", "true_alarm_s", "false_alarm_s"]:
                if abs(checked[metric] - choice[metric]) > 1e-6:
                    raise ValueError(f"Training selected/control interval mismatch: {metric}")
    results = json.loads((root / "comparison.json").read_text("utf-8"))
    reference = None
    for variant in ("window_f1", "event_objective"):
        directory = root / variant
        policy = json.loads((directory / "event_policy.json").read_text("utf-8"))
        choice = deploy["selection"]["control" if variant == "window_f1" else "selected"]
        if policy["threshold"] != choice["threshold"] or policy["confirmation_windows"] != choice["confirmation_windows"]:
            raise ValueError("Deployment policy differs from declared selection")
        frame = pd.read_csv(directory / "oof_predictions.csv")
        if not np.array_equal(frame.is_fog, frame.fog_probability.ge(frame.fog_threshold)):
            raise ValueError("Raw threshold decision changed")
        if not np.array_equal(frame.event_fog, independent_gate(frame)):
            raise ValueError("Confirmation replay mismatch")
        if not np.array_equal(frame.available_at_s, frame.window_end_s):
            raise ValueError("Backdated event decision")
        for fold_no, audit in enumerate(audits, 1):
            choice = audit["selection"]["control" if variant == "window_f1" else "selected"]
            part = frame.loc[frame.subject_id.astype(int).astype(str).isin(audit["outer_test"])]
            if not (part.held_out_fold.eq(fold_no).all()
                    and np.allclose(part.fog_threshold, choice["threshold"], atol=1e-12, rtol=0)
                    and part.event_confirmation_windows.eq(choice["confirmation_windows"]).all()):
                raise ValueError("Wrong held-out policy assignment")
        columns = ["subject_id", "session_id", "task_id", "window_start_s", "window_end_s",
                   "truth_fog_label", "fog_probability"]
        if reference is not None:
            pd.testing.assert_frame_equal(reference, frame[columns], check_exact=True)
        reference = frame[columns]
        runtime = pd.read_csv(directory / "a2_runtime.csv")
        pd.testing.assert_frame_equal(runtime, frame.drop(columns=[c for c in frame if c.startswith("truth_")]))
        expected = next(r for r in results if r["variant"] == variant)
        y = frame.truth_fog_label
        actual = {"fog_f1": f1_score(y, frame.event_fog, zero_division=0),
                  "precision": precision_score(y, frame.event_fog, zero_division=0),
                  "recall": recall_score(y, frame.event_fog, zero_division=0),
                  "auc_of_same_raw_scores": roc_auc_score(y, frame.fog_probability),
                  "ap_of_same_raw_scores": average_precision_score(y, frame.fog_probability),
                  **old_interval_check(raw, frame, repository)}
        for name, value in actual.items():
            if abs(float(value) - expected[name]) > 1e-6:
                raise ValueError(f"Independent metric mismatch: {name}")
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    for name, digest in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Output hash changed: {name}")
    if hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest() != manifest["protocol_sha256"]:
        raise ValueError("Protocol changed")
    result = {"status": "PASS", "windows_per_variant": len(reference), "folds": len(audits),
              "candidate_rules_checked_per_selection": len(deploy["selection"]["options"]),
              "training_control_selected_interval_checks": (len(audits) + 1) * 2,
              "hashes_verified": len(manifest["files"]), "raw_interval_crosscheck": "PASS",
              "scope": "Saved scores/policies and original interval diagnostics; no independent retraining"}
    (root / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

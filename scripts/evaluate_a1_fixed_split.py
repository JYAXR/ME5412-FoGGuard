"""Separate, read-only frozen-model TEST evaluation; no fitting or threshold tuning."""

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from run_a1_event import legacy_events
from run_a1_experiment import sha, write_json
from run_a1_fallback import validate_mixed_windows
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fogguard.data.fogstar import iter_fogstar_windows
from fogguard.evaluation.a1_event import EventSupport, merge
from fogguard.evaluation.a1_fixed_split import perturb_sensors, role_frame
from fogguard.modeling.a1_fallback import route_available_predictions
from fogguard.modeling.a1_fixed_split import (
    feature_bank,
    feature_kind,
    prediction_frame,
    scores_for,
)
from fogguard.modeling.a1_fog import A1FogModel


def check_lock(root, lock):
    for name, digest in lock["files"].items():
        if sha(root / name) != digest:
            raise ValueError(f"Frozen training artifact changed: {name}")


def with_truth(windows, predictions, model_id, scenario):
    result = predictions.copy()
    result["truth_fog_label"] = [w.fog_label for w in windows]
    result["truth_fog_fraction"] = [w.fog_fraction for w in windows]
    result["data_role"], result["stress_scenario"] = "test", scenario
    result["prediction_source"] = "fixed_split_test"
    result["sensor_route"] = "fallback_ankle" if model_id == "rf_ankle" else "primary_ankle_back"
    return result


def summarize(raw, frame, scenario, model_id, output, repository):
    directory = output / scenario / model_id
    directory.mkdir(parents=True)
    # Truth labels/time are checked against the original, unperturbed record.
    # Sensor finiteness was checked on the transformed windows during extraction.
    validation_input = raw.copy()
    # Do not claim missing original channels have become available: perturbations
    # never restore missing inputs and the common raw audit checks original data.
    validated = validate_mixed_windows(validation_input, frame)
    frame.to_csv(directory / "predictions.csv", index=False)
    frame.drop(columns=[c for c in frame if c.startswith("truth_")]).to_csv(directory / "runtime.csv", index=False)
    allowed = sorted(raw.subjectID.astype(int).astype(str).unique())
    support = EventSupport.build(raw, frame, allowed=allowed)
    events = support.metrics(frame.is_fog)
    checked = legacy_events(raw, frame.assign(event_fog=frame.is_fog), repository, directory)
    for name in ("truth_events", "covered_raw_events", "supported_events", "true_alarm_s", "false_alarm_s"):
        if abs(events[name] - checked[name]) > 1e-6:
            raise ValueError(f"Original interval cross-check mismatch: {name}")
    observed = 0.0
    for _, group in raw.groupby(["subjectID", "sessionID", "taskID"]):
        t = group.timestamp.to_numpy(float)
        observed += sum(b - a for a, b in merge(zip(t, t + 1 / 60, strict=True)))
    y, prediction = frame.truth_fog_label.to_numpy(int), frame.is_fog.to_numpy(bool)
    row = {"scenario": scenario, "model": model_id, "windows": validated,
           "positive_windows": int(y.sum()), "precision": float(precision_score(y, prediction, zero_division=0)),
           "recall": float(recall_score(y, prediction, zero_division=0)),
           "fog_f1": float(f1_score(y, prediction, zero_division=0)),
           "auc": float(roc_auc_score(y, frame.fog_probability)) if len(set(y)) == 2 else None,
           "average_precision": float(average_precision_score(y, frame.fog_probability)) if y.sum() else None,
           **events, "false_alarm_events": checked["false_alarm_events"],
           "median_new_alarm_delay_s": checked["median_new_alarm_delay_s"],
           "new_alarm_delay_cases": checked["new_alarm_delay_cases"],
           "observed_s": observed, "decision_support_s": float(support.duration.sum()),
           "decision_support_fraction": float(support.duration.sum()) / observed,
           "fallback_rows": int(frame.sensor_route.eq("fallback_ankle").sum()),
           "raw_records": raw.groupby(["subjectID", "sessionID", "taskID"]).ngroups,
           "records_with_predictions": frame.groupby(["subject_id", "session_id", "task_id"]).ngroups}
    individual = []
    for person in sorted(raw.subjectID.unique()):
        part = frame.loc[frame.subject_id.astype(int).eq(int(person))]
        actual, predicted = part.truth_fog_label, part.is_fog
        individual.append({"subject_id": int(person), "windows": len(part), "positive_windows": int(actual.sum()),
                           "TP": int((actual.eq(1) & predicted).sum()), "FP": int((actual.eq(0) & predicted).sum()),
                           "FN": int((actual.eq(1) & ~predicted).sum()), "TN": int((actual.eq(0) & ~predicted).sum()),
                           "fog_f1": float(f1_score(actual, predicted, zero_division=0)) if len(part) else None})
    pd.DataFrame(individual).to_csv(directory / "participant_metrics.csv", index=False)
    write_json(directory / "metrics.json", row)
    return row


def plots(output, rows, selected_primary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir()

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    clean = rows.loc[rows.scenario.eq("clean")]
    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    x = np.arange(len(clean))
    for i, field in enumerate(("precision", "recall", "fog_f1")):
        bars = ax.bar(x + (i - 1) * 0.25, clean[field], width=0.24, label=field)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax.set_xticks(x, clean.model, rotation=25, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("TEST window metric, conditional on available windows")
    ax.set_title(f"Fixed 14/4/4 participants; primary selected on validation: {selected_primary}")
    ax.legend()
    save(fig, "clean_test_models")
    matrix = rows.pivot(index="model", columns="scenario", values="fog_f1")
    matrix = matrix[["clean", *[c for c in matrix if c != "clean"]]]
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    shown = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu")
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if value > 0.65 else "black")
    ax.set_yticks(range(len(matrix)), matrix.index)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
    ax.set_title("Frozen model/threshold stress test; dropout can change window support")
    fig.colorbar(shown, ax=ax, label="Conditional window F1")
    save(fig, "stress_f1")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for model in (selected_primary, "rf_ankle", "joint_validation_selected"):
        part = rows.loc[rows.model.eq(model)]
        axes[0].plot(part.scenario, part.time_recall, marker="o", label=model)
        axes[1].plot(part.scenario, part.decision_support_fraction, marker="o", label=model)
    for ax, title in zip(axes, ("Raw FoG time recall", "Prediction time support / observed time"), strict=True):
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=40, labelsize=8)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    save(fig, "coverage_under_stress")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root, repository = args.run_dir, Path(__file__).resolve().parents[1]
    output = args.output_dir or root / "test_results"
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh test output directory; do not overwrite prior tests")
    lock = json.loads((root / "TRAINING_LOCK.json").read_text("utf-8"))
    check_lock(root, lock)
    config = json.loads((root / "frozen_config.json").read_text("utf-8"))
    training_protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    for name, digest in training_protocol["source_sha256"].items():
        if sha(repository / name) != digest:
            raise ValueError(f"Model/scoring implementation changed after training: {name}")
    if sha(args.raw) != training_protocol["raw_sha256"]:
        raise ValueError("Raw source differs from training split")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(repository / "artifacts/matplotlib-a1-fixed-test-cache"))
    write_json(output / "TEST_EVALUATION_STARTED.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "training_lock_sha256": sha(root / "TRAINING_LOCK.json"), "evaluator_sha256": sha(Path(__file__)),
        "test_people": config["split"]["test"], "selected_primary_before_testing": lock["selection"]["primary_model"],
        "scenarios_fixed_before_testing": config["stress_scenarios"], "new_fitting_or_selection": False})
    raw = pd.read_csv(args.raw)
    test_raw = role_frame(raw, config["split"], "test")
    del raw
    noise_stats = json.loads((root / "training_noise_stats.json").read_text("utf-8"))
    models = {name: A1FogModel.load(root / name / "model.joblib") for name in config["models"]}
    for model in models.values():
        if (set(model.fit_participants) != set(config["split"]["train"])
                or set(model.threshold_participants) != set(config["split"]["validation"])):
            raise ValueError("Loaded model trained or selected on wrong people")
    winner = lock["selection"]["primary_model"]
    joint_dir = output / "joint_models"
    joint_dir.mkdir()
    shutil.copyfile(root / winner / "model.joblib", joint_dir / "primary_model.joblib")
    shutil.copyfile(root / "rf_ankle/model.joblib", joint_dir / "ankle_model.joblib")
    results = []
    for scenario in config["stress_scenarios"]:
        print(f"Frozen TEST evaluation: {scenario}", flush=True)
        transformed = perturb_sensors(test_raw, scenario, noise_stats, config["stress_seed"])
        for column in ("timestamp", "fog", "activity", "subjectID", "sessionID", "taskID"):
            pd.testing.assert_series_equal(test_raw[column], transformed[column])
        banks, windows_by_kind = {}, {}
        for kind, positions in [("primary", ("ankleL", "back")), ("ankle", ("ankleL",))]:
            windows = list(iter_fogstar_windows(transformed, sensor_positions=positions, max_interpolation_gap_s=0))
            if not windows:
                raise ValueError("Entire TEST input has no valid windows; retain as no-support result explicitly")
            windows_by_kind[kind] = windows
            banks[kind] = feature_bank(windows, ankle=kind == "ankle")
        predictions = {}
        for name, model in models.items():
            kind = "ankle" if name == "rf_ankle" else "primary"
            windows = windows_by_kind[kind]
            values, names = banks[kind][feature_kind(name)]
            if names != model.feature_names:
                raise ValueError("TEST features differ from frozen model")
            score = scores_for(model.classifier, name, values)
            prediction = prediction_frame(windows, score, model.threshold, name)
            frame = with_truth(windows, prediction, name, scenario)
            predictions[name] = frame
            results.append(summarize(test_raw, frame, scenario, name, output, repository))
        combined = route_available_predictions(predictions[winner], predictions["rf_ankle"])
        results.append(summarize(test_raw, combined, scenario, "joint_validation_selected", output, repository))
        write_json(output / "all_metrics.json", results)
        print(f"  complete: {len(models)} models plus preselected joint; thresholds unchanged", flush=True)
    check_lock(root, lock)
    table = pd.DataFrame(results)
    table.to_csv(output / "all_metrics.csv", index=False)
    clean = table.loc[table.scenario.eq("clean")]
    clean.to_csv(output / "clean_test_comparison.csv", index=False)
    plots(output, table, winner)
    images = "".join(f'<img src="figures/{p.name}" style="width:100%" alt="{p.stem}">' for p in sorted((output / "figures").glob("*.png")))
    columns = ["model", "windows", "precision", "recall", "fog_f1", "covered_raw_events", "truth_events",
               "false_alarm_events", "time_precision", "time_recall", "decision_support_fraction"]
    (output / "index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>A1 固定划分重训测试</title>'
        '<body style="max-width:1200px;margin:32px auto;padding:20px;font-family:system-ui;line-height:1.7">'
        f'<h1>A1：先分训练/验证/测试，再重训</h1><p>测试人 {config["split"]["test"]}；'
        f'验证预先选定主模型 {winner}。同一旧队列的新内部划分，不是独立外部验证。'
        '模型与阈值在测试前锁定。缺测时窗口池改变，应同时检查全部测试事件/时间和未知支持。</p>'
        + clean[columns].to_html(index=False, float_format=lambda n: f"{n:.4f}") + images + '</body></html>', encoding="utf-8")
    write_json(output / "TEST_COMPLETED.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "training_lock_sha256": sha(root / "TRAINING_LOCK.json"), "frozen_files_unchanged": True,
        "test_people": config["split"]["test"], "selected_primary_unchanged": winner,
        "evaluations": len(results), "test_driven_refits": 0,
        "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print(clean[columns].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

"""Build a reproducible pre-work package from real OOF and separate simulated control."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import platform
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1]
                                       / "artifacts" / "matplotlib-readiness-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fogguard.complete import complete_cases, run_complete_pipeline
from fogguard.cueing.march import MarchConfig, MarchGuidanceController
from fogguard.cueing.replay import replay_guidance
from fogguard.evaluation.automation2 import audit_commands
from fogguard.evaluation.readiness import (
    KEYS,
    RAW_KEYS,
    SENSOR_SETS,
    audit_records,
    frozen_folds,
    preturn_candidates,
    split_runtime_truth,
    window_metrics,
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def save_figure(figure, directory, name):
    figure.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    figure.savefig(directory / f"{name}.svg", bbox_inches="tight")
    plt.close(figure)


def select_cases(predictions):
    """Deterministic longest consecutive TP/FP/FN runs, not cherry-picked metrics."""
    candidates = []
    for key, group in predictions.groupby(KEYS, sort=True):
        group = group.sort_values("window_end_s").copy()
        truth, pred = group.truth_fog_label.astype(bool), group.is_fog.astype(bool)
        group["outcome"] = np.select([truth & pred, ~truth & pred, truth & ~pred],
                                     ["TP", "FP", "FN"], default="TN")
        breaks = (group.outcome.ne(group.outcome.shift())
                  | group.window_end_s.diff().gt(0.75)).cumsum()
        for _, run in group.groupby(breaks):
            if run.outcome.iloc[0] != "TN":
                candidates.append({"key": list(map(int, key)),
                                   "outcome": run.outcome.iloc[0], "windows": len(run),
                                   "start_s": float(run.window_start_s.iloc[0]),
                                   "end_s": float(run.window_end_s.iloc[-1])})
    candidates.sort(key=lambda item: (-item["windows"], item["key"], item["start_s"]))
    return [next(item for item in candidates if item["outcome"] == outcome)
            for outcome in ("TP", "FP", "FN")]


def plot_real_case(raw, predictions, commands, case, figures):
    key = case["key"]
    signal = raw.loc[(raw[RAW_KEYS].to_numpy() == key).all(axis=1)]
    left, right = case["start_s"] - 3, case["end_s"] + 3
    signal = signal.loc[signal.timestamp.between(left, right)]
    windows = predictions.loc[(predictions[KEYS].to_numpy() == key).all(axis=1)]
    windows = windows.loc[windows.window_end_s.between(left, right)]
    controls = commands.loc[(commands[KEYS].to_numpy().astype(int) == key).all(axis=1)]
    controls = controls.loc[controls.window_end_s.between(left, right)]
    origin = float(signal.timestamp.iloc[0])
    figure, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for position in ("ankleL", "back"):
        columns = [f"{position}_acc_{axis}" for axis in "xyz"]
        magnitude = np.linalg.norm(signal[columns].to_numpy(float), axis=1)
        axes[0].plot(signal.timestamp - origin, magnitude, label=position, lw=0.8)
    axes[0].set_ylabel("Acceleration (g)")
    axes[0].legend(loc="upper right", ncol=2)
    t = windows.window_end_s - origin
    axes[1].step(t, windows.fog_probability, where="post", label="OOF FoG score")
    axes[1].step(t, windows.fog_threshold, where="post", label="Train-fold threshold")
    axes[1].plot(signal.timestamp - origin, signal.fog, color="0.5", lw=0.8,
                 label="Sample FoG truth (evaluation only)")
    axes[1].set_ylabel("Score / truth")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right", fontsize=8)
    direction_map = {"other": 0, "unknown": 0, "straight": 1, "left": 2, "right": 3}
    axes[2].step(t, windows.direction.map(direction_map), where="post",
                 label="Current direction prediction")
    axes[2].set_yticks(range(4), ["reject", "straight", "left", "right"])
    axes[2].set_ylabel("Direction")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[3].step(controls.window_end_s - origin,
                 controls.stable_fog_detected.astype(int), where="post", label="Confirmed FoG")
    axes[3].step(controls.window_end_s - origin,
                 controls.guidance_visual_enabled.astype(int), where="post", label="Visual cue")
    axes[3].set_ylabel("Enabled (0/1)")
    axes[3].set_xlabel("Time from displayed segment start (s); outputs start at window END")
    axes[3].legend(loc="upper right", fontsize=8)
    figure.suptitle(f"Real held-out case {case['outcome']}: P{key[0]:02d}/S{key[1]}/T{key[2]}"
                    " | no measured step-response input")
    for axis in axes:
        axis.grid(alpha=0.2)
    name = f"real_{case['outcome'].lower()}"
    save_figure(figure, figures, name)
    signal.to_csv(figures.parent / f"{name}_signals.csv", index=False)
    windows.to_csv(figures.parent / f"{name}_windows.csv", index=False)
    controls.to_csv(figures.parent / f"{name}_commands.csv", index=False)


def generate(raw_path, prediction_path, report_path, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory to preserve previous evidence")
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir()
    raw, predictions = pd.read_csv(raw_path), pd.read_csv(prediction_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = report["dataset"]["md5"]
    if hashlib.md5(raw_path.read_bytes()).hexdigest() != expected:
        raise ValueError("Raw input does not match training report")
    records = audit_records(raw, predictions)
    turns = preturn_candidates(raw)
    records.to_csv(output / "record_support.csv", index=False)
    turns.to_csv(output / "preturn_positive_candidates.csv", index=False)
    folds = frozen_folds(report, predictions)
    write_json(output / "frozen_folds.json", folds)
    runtime, truth = split_runtime_truth(predictions)
    runtime.to_csv(output / "runtime_predictions.csv", index=False)
    truth.to_csv(output / "evaluation_truth.csv", index=False)
    events, commands = run_complete_pipeline(runtime)
    events["available_at_s"] = events.window_end_s
    events["direction_task"] = "current"
    events["prediction_source"] = "subject_held_out_oof"
    events["feedback_state"] = "not_estimated"
    events["feedback_valid"] = False
    events["feedback_source"] = "unavailable"
    fold_map = {int(subject): i for i, fold in enumerate(folds["folds"], 1)
                for subject in fold["test"]}
    events["held_out_fold"] = events.subject_id.astype(int).map(fold_map)
    events.to_csv(output / "a_to_b_runtime.csv", index=False)
    commands.to_csv(output / "real_commands.csv", index=False)
    errors = audit_commands(commands)
    if errors:
        raise ValueError(f"Real replay command audit errors: {errors[:3]}")
    if commands.guidance_audio_mode.eq("march_alternating").any():
        raise ValueError("Real replay fabricated a measured step-response source")
    metrics = window_metrics(predictions)
    subject_metrics = [{"subject_id": int(key), **window_metrics(group)}
                       for key, group in predictions.groupby("subject_id")]
    pd.DataFrame(subject_metrics).to_csv(output / "subject_metrics.csv", index=False)
    activity_metrics = []
    for activity, group in predictions.groupby("truth_activity_label", dropna=False):
        activity_metrics.append({"activity": None if pd.isna(activity) else int(activity),
                                 **window_metrics(group)})
    pd.DataFrame(activity_metrics).to_csv(output / "activity_metrics.csv", index=False)
    selected = select_cases(predictions)
    for case in selected:
        plot_real_case(raw, predictions, commands, case, figures)

    # Explicit engineering sample: its responses remain simulated throughout.
    simulated_events, responses = complete_cases()["march_recovery"]
    simulated = replay_guidance(simulated_events, responses=responses, config=MarchConfig(),
                                controller_type=MarchGuidanceController)
    if audit_commands(simulated):
        raise ValueError("Simulated command lifecycle audit failed")
    simulated.to_csv(output / "simulated_march_commands.csv", index=False)
    responses.to_csv(output / "simulated_responses.csv", index=False)
    simulated_events.to_csv(output / "simulated_candidates_events.csv", index=False)
    states = list(dict.fromkeys(simulated.guidance_control_state))
    figure, axis = plt.subplots(figsize=(12, 4), layout="constrained")
    axis.step(simulated.window_end_s, simulated.guidance_control_state.map(
        {name: i for i, name in enumerate(states)}), where="post")
    axis.set_yticks(range(len(states)), states)
    for t in responses.timestamp_s:
        axis.axvline(t, alpha=0.12, color="green")
    axis.set_xlabel("Replay time (s); vertical lines = preset simulated responses")
    axis.set_title("Engineering-only march / forward / recovery state transitions")
    axis.grid(alpha=0.2)
    save_figure(figure, figures, "simulated_switch")
    figure, axis = plt.subplots(figsize=(6, 5), layout="constrained")
    tn, fp, fn, tp = metrics["confusion_matrix_tn_fp_fn_tp"]
    matrix = np.array([[tn, fp], [fn, tp]])
    axis.imshow(matrix, cmap="Blues")
    for (i, j), count in np.ndenumerate(matrix):
        axis.text(j, i, str(count), ha="center", va="center",
                  color="white" if count > matrix.max() / 2 else "black", fontsize=18)
    axis.set_xticks([0, 1], ["Non-FoG", "FoG"])
    axis.set_yticks([0, 1], ["Non-FoG", "FoG"])
    axis.set_xlabel("Held-out prediction")
    axis.set_ylabel("Window truth")
    axis.set_title(f"Reproduced baseline | FoG F1 = {metrics['fog_f1']:.3f}")
    save_figure(figure, figures, "baseline_confusion")
    counts = [int(records[f"{name}_has_2s_raw_run"].sum()) for name in SENSOR_SETS]
    figure, axis = plt.subplots(figsize=(9, 4), layout="constrained")
    bars = axis.barh(["Left ankle + back", "Both ankles", "Both ankles + back"], counts)
    axis.bar_label(bars, labels=[f"{value}/{len(records)}" for value in counts], padding=4)
    axis.set_xlim(0, len(records) * 1.14)
    axis.set_xlabel("Records with >=120 consecutive finite raw samples (no interpolation)")
    axis.set_title("Sensor support audit; not a model accuracy result")
    save_figure(figure, figures, "sensor_support")
    usable = turns.loc[turns.eligible]
    if not usable.empty:
        example = usable.iloc[0]
        key = example[KEYS].to_numpy(int)
        signal = raw.loc[(raw[RAW_KEYS].to_numpy() == key).all(axis=1)]
        signal = signal.loc[signal.timestamp.between(example.input_start_s - 0.5,
                                                     example.onset_s + 1)]
        figure, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, layout="constrained")
        axes[0].plot(signal.timestamp - example.onset_s, signal.back_gyro_z, lw=0.8)
        axes[0].set_ylabel("Back gyro z (deg/s)")
        axes[1].step(signal.timestamp - example.onset_s, signal.activity, where="post")
        axes[1].set_ylabel("Activity truth code")
        for axis in axes:
            axis.axvspan(example.input_start_s - example.onset_s,
                         example.input_end_s - example.onset_s, alpha=0.12)
            axis.axvline(-example.horizon_s, color="orange", label="Input cutoff")
            axis.axvline(0, color="red", label="Annotated turn onset")
            axis.grid(alpha=0.2)
        axes[0].legend(loc="upper left")
        axes[1].set_xlabel("Time relative to annotated turn onset (s)")
        figure.suptitle("Preturn feasibility sample; NO intent classifier trained or validated")
        save_figure(figure, figures, "preturn_feasibility")
        signal.to_csv(output / "preturn_example_signals.csv", index=False)
    turn_summary = [{"horizon_s": float(h), "candidates": len(group),
                     "eligible": int(group.eligible.sum()),
                     "eligible_subjects": int(group.loc[group.eligible].subject_id.nunique()),
                     "reasons": {str(k): int(v) for k, v in group.reason.value_counts().items()}}
                    for h, group in turns.groupby("horizon_s")]
    summary = {
        "schema": "fogguard.algorithm-readiness.v1", "scope": "pre-work, not a new algorithm",
        "raw_samples": len(raw), "raw_columns": list(raw.columns),
        "subjects": int(raw.subjectID.nunique()), "records": len(records),
        "records_without_oof": int(records.oof_windows.eq(0).sum()),
        "median_timestamp_delta": float(raw.timestamp.diff().median()),
        "timestamp_note": "Raw values behave as seconds at 60Hz; source README says ms",
        "sensor_support_no_interpolation": dict(zip(SENSOR_SETS, counts)),
        "baseline_metrics": metrics, "preturn": turn_summary, "selected_cases": selected,
        "step_event_ground_truth_present": False, "forward_displacement_truth_present": False,
        "cue_log_present": False, "public_video_file_listed": False,
        "source_record": "https://zenodo.org/records/17838806",
        "step_feedback_status": "NOT_IMPLEMENTED; cannot score real step failure without truth",
        "preturn_status": "LABEL_FEASIBILITY_ONLY; positive candidates need negative examples",
        "real_command_audit_errors": len(errors),
        "simulated_march_source": "constructed candidates and simulated responses",
        "simulated_march_observed": bool(simulated.guidance_audio_mode.eq(
            "march_alternating").any()),
        "python": platform.python_version(),
    }
    write_json(output / "summary.json", summary)
    manifest = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "inputs": {"raw_sha256": digest(raw_path), "oof_sha256": digest(prediction_path),
                   "training_report_sha256": digest(report_path)},
        "source_sha256": {name: digest(Path(name)) for name in (
            "scripts/audit_algorithm_readiness.py", "src/fogguard/evaluation/readiness.py",
            "scripts/train_fogstar_baseline.py", "src/fogguard/data/fogstar.py",
            "src/fogguard/modeling/fogstar_baseline.py")},
        "files": {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()},
    }
    write_json(output / "manifest.json", manifest)
    image_blocks = "".join(f'<figure><img src="figures/{p.name}" style="width:100%">'
                           f'<figcaption>{html.escape(p.stem)}</figcaption></figure>'
                           for p in sorted(figures.glob("*.png")))
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            '<title>FoGGuard 前置核查</title><body style="max-width:1100px;margin:32px auto;'
            'padding:0 20px;font-family:system-ui;line-height:1.7">'
            '<h1>FoGGuard 算法前置核查与 A1 基线</h1>'
            '<p>真实数据识别与构造提示切换分别展示；本包未实现新迈步反馈算法。</p>'
            '<p><a href="../recognition/真实数据冻结识别验收.html">交互查看真实数据与错误案例</a>'
            ' · <a href="../demo/index.html">查看独立工程仿真</a></p>'
            f'<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>'
            f'{image_blocks}</body></html>')
    (output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.raw, args.predictions, args.training_report, args.output)

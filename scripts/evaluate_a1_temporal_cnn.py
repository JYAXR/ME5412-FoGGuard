"""Read-only evaluation of TRAIN-frozen CNN/GRU on the development VALIDATION set."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from report_a1_author_cnn import ACTIVITIES, event_summary
from run_a1_experiment import sha, write_json

from fogguard.evaluation.a1_author import annotate, grouped_diagnostics, matched_frames, metrics
from fogguard.evaluation.a1_fixed_split import role_frame
from fogguard.modeling.a1_author_cnn import gyro_windows
from fogguard.modeling.a1_fog import A1FogModel


def plots(folder, run, frames, comparison, raw, config, gate):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = folder / "figures"
    figures.mkdir()

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    colors = {"cnn_control": "#4790a4", "cnn_gru": "#ce7d3c"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for name in config["models"]:
        epochs = []
        for fold in range(3):
            history = pd.read_csv(run / name / f"fold_{fold}/checkpoint_history.csv")
            audit = json.loads((run / name / f"fold_{fold}/fit_audit.json").read_text("utf-8"))
            axes[0].plot(history.epoch, history.checkpoint_loss, color=colors[name], alpha=0.45 + fold * 0.2,
                         linestyle=["-", "--", ":"][fold], label=f"{name} fold{fold}")
            epochs.append(audit["best_epoch"])
        axes[1].plot(range(3), epochs, "o-", color=colors[name], label=name)
        axes[1].axhline(np.median(epochs), color=colors[name], linestyle="--", alpha=0.6)
    axes[0].set(xlabel="Epoch", ylabel="Checkpoint BCE + L2", title="Only inner TRAIN participants select epochs")
    axes[1].set(xticks=range(3), xlabel="Outer fold", ylabel="Selected inner epoch", title="Median epoch fixes final TRAIN14 fit")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.15)
    save(fig, "nested_training")
    common = comparison.loc[comparison.pool.eq("all_common")]
    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    x = np.arange(len(common))
    for i, field in enumerate(("precision", "recall", "fog_f1")):
        bars = ax.bar(x + (i - 1) * 0.25, common[field], width=0.24, label=field)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, common.model, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"VALIDATION development | {common.windows.iloc[0]} common windows; historical references use different selection")
    ax.legend(ncol=3)
    save(fig, "matched_validation")
    selected = comparison.loc[comparison.pool.eq("validation_own") & comparison.model.isin(config["models"])].set_index("model").loc[config["models"]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    for ax, field, title in zip(axes, ("fog_f1", "false_alarm_s", "time_recall"),
                                 ("Window F1", "False alarm seconds", "FoG-time recall"), strict=True):
        bars = ax.bar(config["models"], selected[field], color=list(colors.values()))
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_title(title)
        ax.set_ylim(0, max(float(selected[field].max()) * 1.18, 0.1))
    fig.suptitle(f"Predeclared development gate: {'PASS' if gate['passed'] else 'FAIL'} | no auto-adoption")
    save(fig, "event_tradeoff")
    fig, ax = plt.subplots(figsize=(9, 4.3), layout="constrained")
    codes = [3, 6, 7, 1]
    for i, name in enumerate(config["models"]):
        part = frames[name]
        rates = []
        for code in codes:
            rows = part.loc[part.truth_activity.astype(str).eq(str(code)) & part.truth_fog_label.eq(0)]
            rates.append(float(rows.is_fog.mean()) if len(rows) else 0)
        bars = ax.bar(np.arange(4) + (i - 0.5) * 0.35, rates, width=0.34, label=name, color=colors[name])
        ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xticks(range(4), [ACTIVITIES[c] for c in codes])
    ax.set_ylim(0, 1)
    ax.set(ylabel="FP / activity's non-FoG windows", title="VALIDATION errors; activity truth only for analysis, never a runtime gate")
    ax.legend()
    save(fig, "activity_errors")
    cases = json.loads((Path(config["reference_run"]) / "report/selected_cases.json").read_text("utf-8"))
    case = next(c for c in cases if c["role"] == "validation")
    key = case["key"]
    signal = raw.loc[(raw[["subjectID", "sessionID", "taskID"]].to_numpy(int) == key).all(axis=1)]
    origin = float(signal.timestamp.iloc[0])
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, layout="constrained")
    for c in ("ankleL_gyro_x", "ankleL_gyro_y", "ankleL_gyro_z"):
        axes[0].plot(signal.timestamp - origin, signal[c], label=c, lw=0.7)
    axes[0].set_ylabel("Gyro deg/s")
    for ax in axes:
        ax.fill_between(signal.timestamp - origin, 0, 1, where=signal.fog.eq(1),
                        transform=ax.get_xaxis_transform(), color="#aeca9b", alpha=0.35, label="Raw FoG truth")
        ax.grid(alpha=0.15)
    for ax, name in zip(axes[1:], [*config["models"], "reference_author_cnn"], strict=True):
        frame = frames[name]
        part = frame.loc[(frame[["subject_id", "session_id", "task_id"]].to_numpy(int) == key).all(axis=1)]
        ends = part.window_end_s.to_numpy() - origin
        stops = np.minimum(ends + 0.5, np.r_[ends[1:], signal.timestamp.iloc[-1] - origin + 1 / 60])
        ax.hlines(part.fog_probability, ends, stops, label=name, color=colors.get(name, "#69845a"))
        ax.axhline(part.fog_threshold.iloc[0], color="#a94d41", linestyle="--", label="TRAIN-OOF cutoff" if name != "reference_author_cnn" else "Historical VAL cutoff")
        ax.set(ylabel="FoG score", ylim=(-0.05, 1.05))
    for ax in axes:
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Record-relative seconds | outputs held <=0.5s after window END; gaps unknown")
    fig.suptitle(f"Same previous failure case, chosen before this training: {key}; history does not read truth")
    save(fig, "fixed_failure_replay")
    write_json(folder / "replay_case.json", case)
    fig, ax = plt.subplots(figsize=(12, 3.6), layout="constrained")
    ax.axis("off")
    boxes = ["Gyro only / 60Hz\n2s fragments ending\nt-4s, t-2s, t",
             "Same CNN encoder\n24 values / fragment\nindependent demean",
             "Single-direction GRU32\n1/2/3 real fragments\npacked lengths",
             "Dense48 / dropout\nFoG logit + cutoff\nTRAIN-only selection"]
    for i, label in enumerate(boxes):
        ax.text((i + 0.5) / 4, 0.55, label, ha="center", va="center", fontsize=11,
                transform=ax.transAxes, bbox={"boxstyle": "round,pad=0.65", "fc": "#e8f1f4", "ec": "#4790a4"})
        if i < 3:
            ax.annotate("", ((i + 1.02) / 4, 0.55), ((i + 0.98) / 4, 0.55), xycoords="axes fraction", arrowprops={"arrowstyle": "->"})
    ax.text(0.5, 0.15, "First decision after current 2s window; every 0.5s thereafter. Record/gap resets. No future frames or label gate.",
            ha="center", transform=ax.transAxes, fontsize=10)
    ax.set_title("Controlled temporal experiment: CNN uses current fragment; CNN+GRU adds bounded past")
    save(fig, "architecture")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    args = parser.parse_args()
    run, root = args.run_dir, Path(__file__).resolve().parents[1]
    output = run / "evaluation"
    if output.exists():
        parser.error("Preserve previous evaluation; use a fresh run")
    lock = json.loads((run / "MODEL_LOCK.json").read_text("utf-8"))
    check_lock(run, lock)
    protocol = json.loads((run / "protocol.json").read_text("utf-8"))
    if sha(args.raw) != protocol["raw_sha256"]:
        raise ValueError("Raw data changed")
    for path, digest in protocol["source_sha256"].items():
        if sha(root / path) != digest:
            raise ValueError(f"Training/evaluation source changed: {path}")
    config = json.loads((run / "config.json").read_text("utf-8"))
    split = json.loads((run / "split.json").read_text("utf-8"))
    output.mkdir()
    write_json(output / "VALIDATION_STARTED.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
        "model_lock_sha256": sha(run / "MODEL_LOCK.json"), "model_locked_utc": lock["locked_utc"]})
    raw = pd.read_csv(args.raw)
    validation, train = role_frame(raw, split, "validation"), role_frame(raw, split, "train")
    del raw
    windows = gyro_windows(validation)
    frames, comparison = {}, []
    for name in config["models"]:
        model = A1FogModel.load(run / name / "model.joblib")
        frames[name] = annotate(windows, model.predict_windows(windows), "validation")
    reference = Path(config["reference_run"])
    reference_report = json.loads((reference / "report/REPORT_COMPLETED.json").read_text("utf-8"))
    check_lock(reference / "report", reference_report)
    for name in ("author_cnn", "reference_rf_context", "reference_rf_ankle"):
        frames[name if name.startswith("reference_") else f"reference_{name}"] = pd.read_csv(reference / name / "predictions.csv")
    for name, frame in frames.items():
        directory = output / name
        directory.mkdir()
        frame.to_csv(directory / "predictions.csv", index=False)
        frame.drop(columns=[c for c in frame if c.startswith("truth_")]).to_csv(directory / "runtime.csv", index=False)
        comparison.append({"model": name, "pool": "validation_own", **metrics(frame), **event_summary(validation, frame, directory, root)})
        for column in ("subject_id", "truth_activity", "task_id"):
            grouped_diagnostics(frame, column).to_csv(directory / f"by_{column}.csv", index=False)
    for name in config["models"]:
        frame = pd.read_csv(run / name / "train_oof/predictions.csv")
        directory = output / f"train_oof_{name}"
        directory.mkdir()
        comparison.append({"model": name, "pool": "train_oof_calibration", **metrics(frame), **event_summary(train, frame, directory, root)})
        for column in ("subject_id", "truth_activity"):
            grouped_diagnostics(frame, column).to_csv(directory / f"by_{column}.csv", index=False)
    for name, frame in matched_frames(frames).items():
        comparison.append({"model": name, "pool": "all_common", **metrics(frame)})
        frame.to_csv(output / f"common_{name}.csv", index=False)
    table = pd.DataFrame(comparison)
    table.to_csv(output / "comparison.csv", index=False)
    own = table.loc[table.pool.eq("validation_own")].set_index("model")
    baseline, candidate = own.loc["cnn_control"], own.loc["cnn_gru"]
    conditions = {"f1_higher": bool(candidate.fog_f1 > baseline.fog_f1),
                  "false_alarm_seconds_not_higher": bool(candidate.false_alarm_s <= baseline.false_alarm_s + 1e-6),
                  "fog_time_recall_not_lower": bool(candidate.time_recall + 1e-9 >= baseline.time_recall)}
    gate = {"conditions": conditions, "passed": all(conditions.values()), "default_replaced": False,
            "scope": "previously inspected development validation, not external generalization"}
    write_json(output / "development_gate.json", gate)
    os.environ.setdefault("MPLCONFIGDIR", str(root / "artifacts/matplotlib-a1-temporal-cache"))
    plots(output, run, frames, table, validation, config, gate)
    columns = ["model", "pool", "windows", "precision", "recall", "fog_f1", "covered_raw_events",
               "truth_events", "false_alarm_s", "time_recall", "decision_support_fraction"]
    images = "".join(f'<h2>{p.stem}</h2><img src="figures/{p.name}" style="width:100%" alt="{p.stem}">'
                     for p in sorted((output / "figures").glob("*.png")))
    (output / "index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>A1 因果 CNN＋GRU</title>'
        '<body style="max-width:1200px;margin:30px auto;padding:20px;font-family:system-ui;line-height:1.8">'
        '<h1>A1：同输入因果 CNN＋GRU 对照</h1><p>两模型在TRAIN14内部按人选轮数和OOF阈值，锁定后才评价VALIDATION4。'
        '本轮没有新候选TEST成绩。验证队列已被研究，不是盲测；OOF成绩用于阈值校准，不是独立测试。</p>'
        f'<p>预定研发条件：{gate}。不自动替换默认。历史参考选轮数/阈值方式不同，主实验对照是本轮CNN。</p>'
        '<p>validation_own为各模型全部可用窗口，all_common为五模型相同窗口。原始事件分母包含缺测未知；'
        '活动真值只用于分组，不能直接排除站立冻结。窗口标签、事件任意重叠和时间Recall是不同指标。</p>'
        + table[columns].to_html(index=False, float_format=lambda v: f"{v:.4f}", na_rep="—") + images + '</body></html>', encoding="utf-8")
    check_lock(run, lock)
    write_json(output / "EVALUATION_COMPLETED.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_lock_sha256": sha(run / "MODEL_LOCK.json"), "validation_people": split["validation"],
        "test_windows_constructed": 0, "validation_driven_refits": 0,
        "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print(table[columns].to_string(index=False), flush=True)
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()

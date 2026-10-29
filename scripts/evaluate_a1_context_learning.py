"""Read-only validation of isolated objective changes frozen using TRAIN only."""

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


def development_gates(table):
    rows = table.loc[table.pool.eq("validation_own")].set_index("model")
    base, result = rows.loc["control"], {}
    for name in ("person_balanced", "activity_aux"):
        row = rows.loc[name]
        conditions = {"f1_higher": bool(row.fog_f1 > base.fog_f1),
                      "false_alarm_seconds_not_higher": bool(row.false_alarm_s <= base.false_alarm_s + 1e-6),
                      "fog_time_recall_not_lower": bool(row.time_recall + 1e-9 >= base.time_recall)}
        result[name] = {"conditions": conditions, "passed": all(conditions.values())}
    return {"candidates": result, "default_replaced": False, "scope": "Previously inspected development validation, not external generalization"}


def plots(output, run, frames, table, raw, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / "figures"
    folder.mkdir()
    names, colors = config["models"], ["#408a9f", "#b77c37", "#6e9b52"]

    def save(fig, name):
        fig.savefig(folder / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(folder / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, name, color in zip(axes, names, colors, strict=True):
        for fold in range(3):
            history = pd.read_csv(run / name / f"fold_{fold}/checkpoint_history.csv")
            ax.plot(history.epoch, history.checkpoint_loss, color=color,
                    linestyle=["-", "--", ":"][fold], label=f"fold{fold}")
        ax.set(xlabel="Epoch", ylabel="Unweighted FoG BCE + L2", title=name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
    fig.suptitle("TRAIN participants only | identical FoG-only checkpoint rule")
    save(fig, "nested_training")
    weights = pd.read_csv(run / "person_balanced/person_weights.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    x = np.arange(len(weights))
    axes[0].bar(x, weights.fit_windows, color=colors[0])
    axes[0].set(title="TRAIN windows per participant", ylabel="Overlapping window count")
    axes[1].bar(x - 0.18, weights.fit_windows / weights.fit_windows.sum(), width=0.35, label="Uniform windows")
    axes[1].bar(x + 0.18, weights.total_weight / weights.total_weight.sum(), width=0.35, label="Equal-person loss")
    axes[1].set(title="Loss weight share; no held-out statistics", ylabel="Share of total")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x, weights.subject_id, rotation=45)
        ax.set_xlabel("TRAIN participant")
    save(fig, "person_weights")
    own = table.loc[table.pool.eq("validation_own") & table.model.isin(names)].set_index("model").loc[names]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    x = np.arange(3)
    for i, field in enumerate(("precision", "recall", "fog_f1")):
        bars = ax.bar(x + (i - 1) * 0.25, own[field], width=0.24, label=field)
        ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xticks(x, names)
    ax.set(ylim=(0, 1.08), title=f"VALIDATION | {own.windows.iloc[0]} identical windows, one-factor arms")
    ax.legend(ncol=3)
    save(fig, "validation_comparison")
    common = table.loc[table.pool.eq("all_common")]
    fig, ax = plt.subplots(figsize=(11, 4.7), layout="constrained")
    bars = ax.bar(common.model, common.fog_f1, color=colors + ["#8a9193"] * (len(common) - 3))
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.tick_params(axis="x", rotation=20)
    ax.set(ylim=(0, 1), ylabel="FoG F1", title=f"{common.windows.iloc[0]} common windows | historical references use different training/selection")
    save(fig, "matched_historical")
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    for ax, field, title in zip(axes, ("covered_raw_events", "false_alarm_s", "time_recall"),
                                 ("Events with any overlap / 10", "False alarm seconds", "FoG-time recall"), strict=True):
        bars = ax.bar(names, own[field], color=colors)
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.tick_params(axis="x", rotation=15)
        ax.set(title=title, ylim=(0, max(float(own[field].max()) * 1.2, 0.1)))
    save(fig, "event_tradeoffs")
    fig, ax = plt.subplots(figsize=(10, 4.3), layout="constrained")
    codes = [3, 6, 7, 5, 1]
    for i, (name, color) in enumerate(zip(names, colors, strict=True)):
        rates = []
        for code in codes:
            part = frames[name].loc[frames[name].truth_activity.astype(str).eq(str(code)) & frames[name].truth_fog_label.eq(0)]
            rates.append(float(part.is_fog.mean()) if len(part) else 0)
        bars = ax.bar(np.arange(len(codes)) + (i - 1) * 0.25, rates, width=0.24, label=name, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xticks(range(len(codes)), [ACTIVITIES[c] for c in codes])
    ax.set(ylim=(0, 1), ylabel="FP / activity's non-FoG windows", title="Truth-stratified development errors | no runtime activity veto")
    ax.legend(ncol=3)
    save(fig, "activity_errors")
    old_config = json.loads((Path(config["reference_run"]) / "config.json").read_text("utf-8"))
    cases = json.loads((Path(old_config["reference_run"]) / "report/selected_cases.json").read_text("utf-8"))
    case = next(c for c in cases if c["role"] == "validation")
    key = case["key"]
    signal = raw.loc[(raw[["subjectID", "sessionID", "taskID"]].to_numpy(int) == key).all(axis=1)]
    origin = signal.timestamp.iloc[0]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, layout="constrained")
    for c in ("ankleL_gyro_x", "ankleL_gyro_y", "ankleL_gyro_z"):
        axes[0].plot(signal.timestamp - origin, signal[c], lw=0.7, label=c)
    axes[0].set_ylabel("Gyro deg/s")
    for ax in axes:
        ax.fill_between(signal.timestamp - origin, 0, 1, where=signal.fog.eq(1),
                        transform=ax.get_xaxis_transform(), color="#aac79b", alpha=0.35, label="Raw FoG truth")
        ax.grid(alpha=0.15)
    for ax, name, color in zip(axes[1:], names, colors, strict=True):
        frame = frames[name]
        part = frame.loc[(frame[["subject_id", "session_id", "task_id"]].to_numpy(int) == key).all(axis=1)]
        end = part.window_end_s.to_numpy() - origin
        stop = np.minimum(end + 0.5, np.r_[end[1:], signal.timestamp.iloc[-1] - origin + 1 / 60])
        ax.hlines(part.fog_probability, end, stop, label=name, color=color)
        ax.axhline(part.fog_threshold.iloc[0], linestyle="--", color="#a14943", label="TRAIN-OOF threshold")
        ax.set(ylabel="FoG score", ylim=(-0.05, 1.05))
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Record-relative seconds | scores begin at window END; gaps unknown")
    fig.suptitle(f"Same preselected failure record {key}, all three arms; no success-case reselection")
    save(fig, "fixed_failure_replay")
    write_json(output / "replay_case.json", case)
    fig, ax = plt.subplots(figsize=(11, 4.1), layout="constrained")
    ax.axis("off")
    for x, y, label in [(0.16, 0.55, "Gyro-only history\n3 causal fragments\nTRAIN / runtime"),
                        (0.48, 0.55, "Shared CNN-GRU32\nsame 14536 parameters\nsame random seeds"),
                        (0.81, 0.78, "FoG head\nBCE: uniform OR\nequal-person weights"),
                        (0.81, 0.28, "Activity head (TRAIN)\n7-class CE x 0 OR 0.2\nnever a runtime gate")]:
        ax.text(x, y, label, ha="center", va="center", transform=ax.transAxes, fontsize=11,
                bbox={"boxstyle": "round,pad=0.65", "fc": "#e8f1f4", "ec": colors[0]})
    for start, end in [((0.29, 0.55), (0.34, 0.55)), ((0.62, 0.58), (0.68, 0.78)), ((0.62, 0.52), (0.68, 0.28))]:
        ax.annotate("", end, start, xycoords="axes fraction", arrowprops={"arrowstyle": "->"})
    ax.set_title("Isolated loss changes | equal-person and activity auxiliary are separate arms")
    save(fig, "architecture")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    args = parser.parse_args()
    run, root = args.run_dir, Path(__file__).resolve().parents[1]
    output = run / "evaluation"
    if output.exists():
        parser.error("Use a run without existing evaluation")
    lock = json.loads((run / "MODEL_LOCK.json").read_text("utf-8"))
    check_lock(run, lock)
    protocol = json.loads((run / "protocol.json").read_text("utf-8"))
    if sha(args.raw) != protocol["raw_sha256"]:
        raise ValueError("Raw data changed")
    for path, digest in protocol["source_sha256"].items():
        if sha(root / path) != digest:
            raise ValueError(f"Source changed after fitting: {path}")
    config, split = [json.loads((run / name).read_text("utf-8")) for name in ("config.json", "split.json")]
    output.mkdir()
    write_json(output / "VALIDATION_STARTED.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
        "model_lock_sha256": sha(run / "MODEL_LOCK.json"), "model_locked_utc": lock["locked_utc"]})
    raw = pd.read_csv(args.raw)
    validation, train = role_frame(raw, split, "validation"), role_frame(raw, split, "train")
    del raw
    windows = gyro_windows(validation)
    frames = {name: annotate(windows, A1FogModel.load(run / name / "model.joblib").predict_windows(windows), "validation")
              for name in config["models"]}
    reference = Path(config["reference_run"])
    complete = json.loads((reference / "evaluation/EVALUATION_COMPLETED.json").read_text("utf-8"))
    check_lock(reference / "evaluation", complete)
    for old, new in [("cnn_gru", "previous_gru"), ("reference_author_cnn", "previous_author"),
                     ("reference_rf_context", "reference_rf_context"), ("reference_rf_ankle", "reference_rf_ankle")]:
        frames[new] = pd.read_csv(reference / "evaluation" / old / "predictions.csv")
    rows = []
    for name, frame in frames.items():
        dest = output / name
        dest.mkdir()
        frame.to_csv(dest / "predictions.csv", index=False)
        frame.drop(columns=[c for c in frame if c.startswith("truth_")]).to_csv(dest / "runtime.csv", index=False)
        rows.append({"model": name, "pool": "validation_own", **metrics(frame), **event_summary(validation, frame, dest, root)})
        for column in ("subject_id", "truth_activity", "task_id"):
            grouped_diagnostics(frame, column).to_csv(dest / f"by_{column}.csv", index=False)
    for name in config["models"]:
        frame = pd.read_csv(run / name / "train_oof/predictions.csv")
        dest = output / f"train_oof_{name}"
        dest.mkdir()
        rows.append({"model": name, "pool": "train_oof_calibration", **metrics(frame), **event_summary(train, frame, dest, root)})
        grouped_diagnostics(frame, "subject_id").to_csv(dest / "by_subject_id.csv", index=False)
    for name, frame in matched_frames(frames).items():
        frame.to_csv(output / f"common_{name}.csv", index=False)
        rows.append({"model": name, "pool": "all_common", **metrics(frame)})
    table = pd.DataFrame(rows)
    table.to_csv(output / "comparison.csv", index=False)
    gates = development_gates(table)
    write_json(output / "development_gates.json", gates)
    os.environ.setdefault("MPLCONFIGDIR", str(root / "artifacts/matplotlib-a1-context-learning-cache"))
    plots(output, run, frames, table, validation, config)
    columns = ["model", "pool", "windows", "precision", "recall", "fog_f1", "covered_raw_events", "truth_events", "false_alarm_s", "time_recall"]
    images = "".join(f'<h2>{p.stem}</h2><img src="figures/{p.name}" style="width:100%" alt="{p.stem}">'
                     for p in sorted((output / "figures").glob("*.png")))
    (output / "index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>A1 均衡与活动辅助</title>'
        '<body style="max-width:1250px;margin:30px auto;padding:20px;font-family:system-ui;line-height:1.8">'
        '<h1>A1：参与者均衡、活动辅助的独立对照</h1><p>三组相同参数、输入、人员折、优化器和FoG-only检查点；'
        '候选分别只改变训练loss中的一项。TRAIN14内部选轮数和OOF阈值，模型锁定后才评价VALIDATION4；没有新候选TEST成绩。</p>'
        '<p>验证集已被开发使用，不是盲测；OOF成绩参与阈值校准。辅助活动只作TRAIN目标，不能从真值站立/转向筛选运行结果。'
        '窗口F1、事件任意重叠、报警时长和冻结时间覆盖分别报告；缺测事件不删除。</p>'
        f'<p>预定研发条件：{gates}。即使通过也不自动替换默认或证明外部泛化。历史参考另列，共同窗口比较不混分母。</p>'
        + table[columns].to_html(index=False, float_format=lambda v: f"{v:.4f}", na_rep="—") + images + '</body></html>', encoding="utf-8")
    check_lock(run, lock)
    write_json(output / "EVALUATION_COMPLETED.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_lock_sha256": sha(run / "MODEL_LOCK.json"), "validation_people": split["validation"],
        "validation_driven_refits": 0, "test_windows_constructed": 0,
        "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print(table[columns].to_string(index=False), flush=True)
    print(json.dumps(gates, indent=2), flush=True)


if __name__ == "__main__":
    main()

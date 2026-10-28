"""Report locked TRAIN-OOF diagnostics and VALIDATION development results."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_event import legacy_events
from run_a1_experiment import sha, write_json

from fogguard.evaluation.a1_author import grouped_diagnostics, matched_frames, metrics
from fogguard.evaluation.a1_event import EventSupport, merge
from fogguard.evaluation.a1_fixed_split import role_frame

ACTIVITIES = {0: "Unlabeled", 1: "Walk", 2: "Sit", 3: "Stand", 4: "Sit-to-stand",
              5: "Stand-to-sit", 6: "Right turn", 7: "Left turn", "mixed": "Mixed"}
MODELS = ["rf_gyro_2s", "author_cnn", "reference_rf_context", "reference_rf_ankle"]


def event_summary(raw, frame, directory, root):
    support = EventSupport.build(raw, frame, allowed=raw.subjectID.unique())
    result = support.metrics(frame.is_fog)
    checked = legacy_events(raw, frame.assign(event_fog=frame.is_fog), root, directory)
    for name in ("truth_events", "covered_raw_events", "supported_events", "true_alarm_s", "false_alarm_s"):
        if abs(result[name] - checked[name]) > 1e-6:
            raise ValueError(f"Event interval disagreement: {name}")
    observed = sum(sum(b - a for a, b in merge(zip(g.timestamp, g.timestamp + 1 / 60, strict=True)))
                   for _, g in raw.groupby(["subjectID", "sessionID", "taskID"]))
    return {**result, "false_alarm_events": checked["false_alarm_events"],
            "median_new_alarm_delay_s": checked["median_new_alarm_delay_s"],
            "new_alarm_delay_cases": checked["new_alarm_delay_cases"],
            "observed_s": observed, "decision_support_s": float(support.duration.sum()),
            "decision_support_fraction": float(support.duration.sum()) / observed,
            "raw_records": raw.groupby(["subjectID", "sessionID", "taskID"]).ngroups}


def figures(output, run, frames, comparison, raw_train, raw_val, tables):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / "figures"
    folder.mkdir()

    def save(fig, name):
        fig.savefig(folder / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(folder / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    history = pd.read_csv(run / "author_cnn/training_history.csv")
    metadata = json.loads((run / "author_cnn/fit_metadata.json").read_text("utf-8"))
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.plot(history.epoch, history.train_loss, label="TRAIN loss (dropout active)")
    ax.plot(history.epoch, history.validation_loss, label="VALIDATION loss (dropout off)")
    ax.axvline(metadata["best_epoch"], linestyle="--", color="#d18c20", label="Restored validation checkpoint")
    ax.set(xlabel="Epoch", ylabel="BCE + kernel L2", title="Author CNN architecture | development loss, no TEST")
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "training_curve")
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    same = comparison.loc[comparison.pool.eq("all_four_common")]
    x = np.arange(len(same))
    for i, field in enumerate(("precision", "recall", "fog_f1")):
        bars = ax.bar(x + (i - 1) * 0.25, same[field], width=0.24, label=field)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x, same.model, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"VALIDATION | {same.windows.iloc[0]} identical windows; development comparison")
    ax.legend(ncol=3)
    save(fig, "matched_validation")
    activity = tables["train_oof_rf_context", "truth_activity"].copy()
    activity["label"] = activity.truth_activity.map(lambda v: ACTIVITIES.get(int(v), str(v)) if str(v) != "mixed" else "Mixed")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    bars = axes[0].bar(activity.label, activity.FP, color="#d27649")
    axes[0].bar_label(bars, fontsize=8, padding=2)
    axes[0].set_title("TRAIN person-held-out RF | false positive windows")
    axes[0].set_ylabel("Window count, overlapping windows")
    axes[1].bar(activity.label, activity.false_positive_window_rate, color="#508fa4")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("FP / all non-FoG windows in each activity")
    for ax in axes:
        ax.tick_params(axis="x", rotation=40)
    save(fig, "train_oof_error_activities")
    fig, ax = plt.subplots(figsize=(12, 3.3), layout="constrained")
    ax.axis("off")
    boxes = ["Left ankle gyro\n120 x 3 / 60Hz\nper-window demean",
             "Causal Conv\n16 k9 -> 16 k7\npool4 / dropout",
             "Causal Conv\n24 k5 / pool3\ndropout / 24 k5",
             "Global average\nDense48 / dropout\n8353 parameters",
             "FoG score\nVAL-only threshold\navailable at window end"]
    for i, label in enumerate(boxes):
        ax.text((i + 0.5) / 5, 0.5, label, ha="center", va="center", fontsize=10,
                transform=ax.transAxes, bbox={"boxstyle": "round,pad=0.7", "fc": "#e7f1f5", "ec": "#508fa4"})
        if i < 4:
            ax.annotate("", ((i + 1.03) / 5, 0.5), ((i + 0.97) / 5, 0.5), xycoords="axes fraction",
                        arrowprops={"arrowstyle": "->", "color": "#224655"})
    ax.set_title("Published architecture, independently trained | split/training adaptations documented")
    save(fig, "architecture")
    # Worst FP record of each declared reference, not a hand-picked success case.
    cases = []
    for role, name, raw in [("train_oof", "train_oof_rf_context", raw_train),
                            ("validation", "reference_rf_context", raw_val)]:
        ref = frames[name].copy()
        ref["fp"] = ref.truth_fog_label.eq(0) & ref.is_fog
        ids = ["subject_id", "session_id", "task_id"]
        counts = ref.groupby(ids).fp.sum().sort_values(ascending=False, kind="stable")
        key = tuple(map(int, counts.index[0]))
        signal = raw.loc[(raw[["subjectID", "sessionID", "taskID"]].to_numpy(int) == key).all(axis=1)]
        origin = float(signal.timestamp.iloc[0])
        selected_names = [name] if role == "train_oof" else MODELS
        fig, axes = plt.subplots(1 + len(selected_names), 1, figsize=(12, 3 + 2 * len(selected_names)),
                                 sharex=True, layout="constrained")
        for channel in ("ankleL_gyro_x", "ankleL_gyro_y", "ankleL_gyro_z"):
            axes[0].plot(signal.timestamp - origin, signal[channel], lw=0.7, label=channel)
        axes[0].set_ylabel("Gyro (deg/s)")
        for ax in axes:
            ax.fill_between(signal.timestamp - origin, 0, 1, where=signal.fog.eq(1),
                            transform=ax.get_xaxis_transform(), color="#b7cfad", alpha=0.35, label="Raw FoG truth")
            ax.grid(alpha=0.15)
        for ax, model in zip(axes[1:], selected_names, strict=True):
            part = frames[model].loc[(frames[model][ids].to_numpy(int) == key).all(axis=1)]
            if len(part):
                end = part.window_end_s.to_numpy() - origin
                stops = np.minimum(end + 0.5, np.r_[end[1:], signal.timestamp.iloc[-1] - origin + 1 / 60])
                ax.hlines(part.fog_probability, end, stops, color="#267c9a", label=model)
                ax.axhline(part.fog_threshold.iloc[0], color="#c66a38", linestyle="--", label="Frozen threshold")
            ax.set_ylabel("FoG score")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8, loc="upper right")
        axes[0].legend(ncol=4, fontsize=8, loc="upper right")
        axes[-1].set_xlabel("Seconds from record start | score held <=0.5s after window END; gaps remain unknown")
        fig.suptitle(f"{role}: RF_context worst-FP record P{key[0]}/S{key[1]}/T{key[2]} | {int(counts.iloc[0])} FP windows")
        save(fig, f"{role}_worst_fp")
        cases.append({"role": role, "reference": name, "key": key, "reference_fp_windows": int(counts.iloc[0]),
                      "selection": "Most false-positive windows per raw record; deterministic key-order tie break"})
    write_json(output / "selected_cases.json", cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    args = parser.parse_args()
    root, run = Path(__file__).resolve().parents[1], args.run_dir
    output = run / "report"
    if output.exists():
        parser.error("Use a run without an existing report")
    lock = json.loads((run / "MODEL_LOCK.json").read_text("utf-8"))
    check_lock(run, lock)
    protocol = json.loads((run / "protocol.json").read_text("utf-8"))
    if sha(args.raw) != protocol["raw_sha256"]:
        raise ValueError("Raw hash differs")
    config = json.loads((run / "split_config.json").read_text("utf-8"))
    raw = pd.read_csv(args.raw)
    raw_train, raw_val = role_frame(raw, config["split"], "train"), role_frame(raw, config["split"], "validation")
    del raw
    output.mkdir()
    os.environ.setdefault("MPLCONFIGDIR", str(root / "artifacts/matplotlib-a1-author-cache"))
    names = MODELS + ["train_oof_rf_context"]
    frames = {name: pd.read_csv(run / name / "predictions.csv") for name in names}
    rows, tables = [], {}
    for name, frame in frames.items():
        directory = output / name
        directory.mkdir()
        signal = raw_train if name == "train_oof_rf_context" else raw_val
        rows.append({"model": name, "pool": "own_support", **metrics(frame),
                     **event_summary(signal, frame, directory, root)})
        for column in ("subject_id", "truth_activity", "task_id"):
            table = grouped_diagnostics(frame, column)
            table.to_csv(directory / f"by_{column}.csv", index=False)
            tables[name, column] = table
    for pool, chosen in [("gyro_pair_common", MODELS[:2]), ("all_four_common", MODELS)]:
        for name, frame in matched_frames({n: frames[n] for n in chosen}).items():
            frame.to_csv(output / f"{pool}_{name}.csv", index=False)
            rows.append({"model": name, "pool": pool, **metrics(frame)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "comparison.csv", index=False)
    figures(output, run, frames, comparison, raw_train, raw_val, tables)
    images = "".join(f'<h2>{p.stem}</h2><img src="figures/{p.name}" style="width:100%" alt="{p.stem}">'
                     for p in sorted((output / "figures").glob("*.png")))
    columns = ["model", "pool", "windows", "positive_windows", "precision", "recall", "fog_f1",
               "covered_raw_events", "truth_events", "false_alarm_s", "decision_support_fraction"]
    (output / "index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>A1 作者 CNN 架构复现与误报诊断</title><body style="max-width:1200px;margin:30px auto;padding:20px;font-family:system-ui;line-height:1.8">'
        '<h1>A1：公开 CNN 架构复现与误报诊断</h1><p>训练14人内按人留出诊断；验证4人用于研发、checkpoint和阈值。'
        '本轮没有构造或评分测试集。这里不是新独立测试，不表示临床有效或提前预测。</p>'
        '<p>gyro_pair_common 是相同陀螺仪窗口的直接对照；all_four_common 是四模型共同窗口；'
        'own_support 输入支持不同，不能直接按窗口F1排名。事件/时间统计覆盖各角色全部原始记录，缺测保留未知。'
        '动作真值只用于事后分组，不进入推理；Stand标签不能等同于主观主动停步。重叠窗口数不是独立事件数。</p>'
        + comparison[columns].to_html(index=False, float_format=lambda v: f"{v:.4f}", na_rep="—")
        + '<p>作者训练脚本与本实现有明确差异，见仓库协议；不加载作者已有权重，不宣称严格复现论文分数。'
        '最差误报记录按固定规则选择，保留失败；检测延迟是离线条件指标，完全漏检另列。</p>' + images + '</body></html>', encoding="utf-8")
    write_json(output / "REPORT_COMPLETED.json", {"model_lock_sha256": sha(run / "MODEL_LOCK.json"),
        "test_windows_constructed": 0, "test_scores_reported": 0,
        "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    check_lock(run, lock)
    print(comparison[columns].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

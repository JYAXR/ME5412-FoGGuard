"""Render locked invariant ablation results, fixed failure cases and training selection."""

import argparse
import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-reference", type=Path, default=Path("reports/a1_external_failure_v1_20261029"))
    args = parser.parse_args()
    run, output = args.run_dir, args.output_dir
    if output.exists():
        parser.error("Use a new presentation directory")
    verification = json.loads((run / "verification.json").read_text("utf-8"))
    assert verification["status"] == "PASS"
    assert verification["model_lock_sha256"] == sha(run / "MODEL_LOCK.json")
    assert verification["evaluation_lock_sha256"] == sha(run / "evaluation/EVALUATION_COMPLETED.json")
    check_lock(args.diagnostic_reference, json.loads((args.diagnostic_reference / "DIAGNOSTIC_LOCK.json").read_text("utf-8")))
    assert verification["diagnostic_reference_lock_sha256"] == sha(args.diagnostic_reference / "DIAGNOSTIC_LOCK.json")
    for folder, file in ((run, "MODEL_LOCK.json"), (run / "evaluation", "EVALUATION_COMPLETED.json")):
        check_lock(folder, json.loads((folder / file).read_text("utf-8")))
    config = json.loads((run / "config.json").read_text("utf-8"))
    external = Path(config["external_reference"])
    check_lock(external / "evaluation", json.loads((external / "evaluation/EVALUATION_COMPLETED.json").read_text("utf-8")))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "artifacts/matplotlib-invariant-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    output.mkdir(parents=True)
    names, labels, colors = config["models"], ["Per-axis 66D", "Invariant 10D"], ["#40859a", "#bf7842"]
    tables = {scope: pd.read_csv(run / "evaluation" / f"{scope}_comparison.csv").set_index("model") for scope in ("validation", "external")}
    gate = json.loads((run / "evaluation/development_gate.json").read_text("utf-8"))
    charts = []

    def save(fig, name, caption):
        for suffix in ("png", "svg"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        charts.append((name, caption))

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")
    boxes = [(0.14, 0.65, "Same bilateral gyro\n120 samples / 60Hz\n2s input, 0.5s step"),
             (0.48, 0.81, "Control: 66 per-axis features\nExisting statistics + band powers"),
             (0.48, 0.39, "Candidate: 10 invariant features\nPer ankle: vector RMS, summed 3-axis\ngait/freeze power, ratio, gait peak"),
             (0.84, 0.65, "Same RF training policy\nTRAIN participant OOF threshold\nLock before evaluation")]
    for x, y, label in boxes:
        ax.text(x, y, label, ha="center", va="center", fontsize=10, transform=ax.transAxes,
                bbox={"boxstyle": "round,pad=0.8", "facecolor": "#edf3f5", "edgecolor": colors[0]})
    for start, end in (((0.26, 0.67), (0.32, 0.80)), ((0.26, 0.60), (0.32, 0.40)), ((0.65, 0.80), (0.72, 0.67)), ((0.65, 0.40), (0.72, 0.60))):
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": "#45636e"})
    fig.suptitle("One representation ablation | no normalization, augmentation, history or threshold tuning on target", fontsize=12)
    fig.text(0.5, 0.10, "Invariant to independent rigid rotations of each ankle's axes; amplitude, placement and label shift remain possible.", ha="center", fontsize=10)
    fig.text(0.5, 0.03, "66D -> 10D also removes information. Fixed max_features=sqrt gives 8 versus 3 node candidates; this is not a pure axis-calibration experiment.", ha="center", fontsize=9)
    save(fig, "representation_pipeline", "实现流程：只替换表征，训练及阈值规则保持一致；10维压缩也可能损失信息。")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, scope, title in zip(axes, ("validation", "external"), ("FoG-STAR VALIDATION | development", "Stanford | inspected development recheck"), strict=True):
        table = tables[scope]
        x = np.arange(4)
        for i, (name, label, color) in enumerate(zip(names, labels, colors, strict=True)):
            bars = ax.bar(x + (i - 0.5) * 0.36, table.loc[name, ["precision", "recall", "fog_f1", "average_precision"]].astype(float), width=0.35, label=label, color=color)
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xticks(x, ["Precision", "Recall", "FoG F1", "AP"])
        ax.set(title=f"{title}\n{int(table.windows.iloc[0])} same overlapping windows", ylim=(0, 1.12))
        ax.legend(fontsize=9)
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
    fig.text(0.02, 0.01, "Within-cohort matched comparison only. Source VALIDATION and Stanford were previously inspected; neither panel is a new blind test.", fontsize=9)
    save(fig, "matched_window_metrics", "相同窗口的精确率、召回率、F1及AP；两个人群分别比较，不跨数据集宣称提升。")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for col, scope in enumerate(("validation", "external")):
        table = tables[scope]
        for row, field, ylabel in ((0, "time_recall", "Native FoG-time recall"), (1, "false_alarm_s", "False alarm seconds")):
            ax = axes[row, col]
            bars = ax.bar(labels, table.loc[names, field], color=colors)
            ax.bar_label(bars, fmt="%.3f" if row == 0 else "%.1f", padding=4)
            ax.set(ylabel=ylabel, ylim=(0, 1.10 if row == 0 else max(table[field]) * 1.24))
            ax.set_title(f"{'Source VALIDATION' if scope == 'validation' else 'Inspected Stanford'} | {field}")
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
    fig.text(0.02, 0.01, "All raw records/events retained. Native decision time includes +1/6s Stanford adapter delay; no backdating or missing-input filling.", fontsize=9)
    save(fig, "native_time_tradeoffs", "全部原始记录的冻结时间召回与误报秒数，保留适配延迟及未知输入。")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, field in zip(axes, ("fog_f1", "recall"), strict=True):
        x = np.arange(7)
        for i, (name, label, color) in enumerate(zip(names, labels, colors, strict=True)):
            people = pd.read_csv(run / "evaluation/external" / name / "by_person.csv")
            bars = ax.bar(x + (i - 0.5) * 0.36, people[field], width=0.35, label=label, color=color)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        ax.set_xticks(x, [f"P{p}" for p in range(1, 8)])
        ax.set(title=f"All seven Stanford participants | window {field}", ylim=(0, 1.12))
        ax.legend(fontsize=9)
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
    fig.text(0.02, 0.01, "Already inspected source; no participant exclusion or target calibration. Improvement in P6/P7 alone would not establish overall improvement.", fontsize=9)
    save(fig, "external_all_people", "全部7名外部参与者的F1和Recall，不只展示P6/P7。")

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for name, label, color in zip(names, labels, colors, strict=True):
        choice = json.loads((run / name / "threshold_selection.json").read_text("utf-8"))
        scores = pd.DataFrame(choice["threshold_scores"])
        ax.plot(scores.threshold, scores.fog_f1, color=color, label=f"{label} TRAIN OOF")
        ax.axvline(choice["selected"]["threshold"], color=color, linestyle="--", label=f"{label} selected {choice['selected']['threshold']:.2f}")
    ax.set(title="Threshold selected only from participant-held-out TRAIN OOF", xlabel="Threshold", ylabel="TRAIN OOF FoG F1 (used for calibration)")
    ax.legend(fontsize=9)
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
    fig.text(0.02, 0.01, "0.05..0.95 fixed grid; F1 / balanced-accuracy / higher-threshold tie rule. Calibration scores are not independent test performance.", fontsize=9)
    save(fig, "train_threshold_selection", "算法过程展示：阈值只从TRAIN按人OOF选择，选好后不据验证/外部结果改变。")

    selections = json.loads((args.diagnostic_reference / "selected_failures.json").read_text("utf-8"))
    replay = json.loads((external / "evaluation/replay_data.json").read_text("utf-8"))
    for case in selections:
        record = next(r for r in replay if r["record_id"] == case["record_id"])
        start, stop = case["start_s"], case["end_s"]
        points = np.array([[t + 1 / 6, np.nan if a is None else a, np.nan if b is None else b] for t, a, b in record["signals"]])
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, layout="constrained", gridspec_kw={"height_ratios": [1.3, 1, 1, 0.8]})
        limits = (start - 4, stop + 4)
        visible = (points[:, 0] >= limits[0]) & (points[:, 0] <= limits[1])
        axes[0].plot(points[visible, 0], points[visible, 1], color=colors[0], lw=0.9, label="Left gyro")
        axes[0].plot(points[visible, 0], points[visible, 2], color="#8a72b4", lw=0.9, label="Right gyro")
        axes[0].set(ylabel="Gyro magnitude (deg/s)")
        axes[0].legend(fontsize=8)
        for i, (name, label, color) in enumerate(zip(names, labels, colors, strict=True)):
            frame = pd.read_csv(run / "evaluation/external" / name / "predictions.csv")
            frame = frame.loc[frame.record_id.eq(case["record_id"])].sort_values("available_at_s")
            rows = list(frame.itertuples())
            intervals = []
            for j, row in enumerate(rows):
                a = row.available_at_s
                b = min(a + 0.5, rows[j + 1].available_at_s if j + 1 < len(rows) else record["native_stop_s"])
                if b < limits[0] or a > limits[1]:
                    continue
                axes[i + 1].plot([a, b], [row.fog_probability] * 2, lw=1.5, color=color)
                if row.is_fog:
                    intervals.append((a, b - a))
            threshold = frame.fog_threshold.iloc[0]
            axes[i + 1].axhline(threshold, color=color, linestyle="--", label=f"{label} cutoff {threshold:.2f}")
            axes[i + 1].set(ylabel="Saved FoG score", ylim=(-0.03, 1.03))
            axes[i + 1].legend(fontsize=8)
            axes[3].broken_barh(intervals, (1 - i - 0.17, 0.34), facecolors=color)
        axes[3].broken_barh([(a, b - a) for a, b in record["intervals"]["truth"]], (1.83, 0.34), facecolors="#c4a34e")
        axes[3].set(yticks=[0, 1, 2], yticklabels=["10D alarm", "66D alarm", "Native truth"], xlabel="Availability time (s); gyro preview shown at logical time + 1/6s")
        for ax in axes:
            ax.axvspan(start, stop, color="#ddbc65", alpha=0.2)
            ax.set_xlim(limits)
        fig.suptitle(f"Fixed previous diagnostic case: P{case['source_person']} / record {case['record_id']} / {stop - start:.3f}s event", fontsize=12)
        fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
        fig.text(0.02, 0.025, "Cases fixed before this experiment, not selected by candidate scores. Shading marks the selected event; bottom row includes neighboring truth.", fontsize=9)
        fig.text(0.02, 0.005, "15Hz magnitude preview only; models use full 60Hz completed windows. Each model uses its own TRAIN-selected threshold.", fontsize=9)
        save(fig, f"fixed_failure_p{case['source_person']}", f"固定旧案例P{case['source_person']}：同一真实输入、各自TRAIN阈值及实际可用时间下的报警对比。")
    fields = ["windows", "precision", "recall", "fog_f1", "covered_raw_events", "false_alarm_s", "time_recall"]
    intro = f'<h1>A1 旋转不变表征对照</h1><p>开发门槛：<b>{"通过" if gate["passed"] else "未通过"}</b>。仅替换66维逐轴表征为10维合成摘要；默认模型未替换。</p><p>FoG-STAR验证及Stanford都已用于研发；本页不提供新盲测、转向意图、迈步成功或临床疗效结论。候选TRAIN阈值0.37，对照0.23；未做目标拟合、调阈值或方向搜索。</p>'
    for scope, title in (("validation", "FoG-STAR 验证集（开发）"), ("external", "Stanford（已查看来源研发复查）")):
        intro += f'<h2>{title}</h2>' + tables[scope][fields].round(4).to_html()
    intro += '<p>事件覆盖只要求正时间重叠，不是一对一事件F1；同窗F1提升必须同时查看冻结时间漏报、误报时间与逐人差异。相关窗口不是独立样本。</p>'
    sections = ''.join(f'<section><h2>{html.escape(caption)}</h2><a href="{name}.svg">SVG</a> · <a href="{name}.png">PNG</a><img src="{name}.png" alt="{html.escape(caption)}"></section>' for name, caption in charts)
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>A1 invariant experiment</title><style>body{max-width:1140px;margin:30px auto;padding:0 20px;color:#19323b;background:#f5f8fa;font:16px/1.7 system-ui}section{background:white;border:1px solid #d9e2e6;padding:20px;margin:25px 0;border-radius:10px}h1{font-size:30px}h2{font-size:18px}img{width:100%}table{border-collapse:collapse;font-size:14px}td,th{padding:8px}a{color:#286f88}</style>' + intro + sections + '</html>'
    (output / "index.html").write_text(page, "utf-8")
    write_json(output / "REPORT_LOCK.json", {"model_lock_sha256": sha(run / "MODEL_LOCK.json"), "evaluation_lock_sha256": sha(run / "evaluation/EVALUATION_COMPLETED.json"),
        "verification_sha256": sha(run / "verification.json"), "reporter_sha256": sha(Path(__file__)), "png_svg_pairs": len(charts),
        "files": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print(f"Rendered {len(charts)} PNG/SVG pairs and HTML", flush=True)


if __name__ == "__main__":
    main()

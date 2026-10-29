"""Render verified, axis-unaligned external transfer results without refitting."""

import argparse
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
    args = parser.parse_args()
    run, output, root = args.run_dir, args.output_dir, Path(__file__).resolve().parents[1]
    if output.exists():
        parser.error("Use a new report directory")
    if json.loads((run / "verification.json").read_text("utf-8"))["status"] != "PASS":
        raise ValueError("Verify external artifacts first")
    check_lock(run, json.loads((run / "INPUT_LOCK.json").read_text("utf-8")))
    check_lock(run / "evaluation", json.loads((run / "evaluation/EVALUATION_COMPLETED.json").read_text("utf-8")))
    os.environ.setdefault("MPLCONFIGDIR", str(root / "artifacts/matplotlib-stanford-external-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True)
    summary = json.loads((run / "evaluation/summary.json").read_text("utf-8"))
    people, records, events = [pd.read_csv(run / "evaluation" / name) for name in ("by_person.csv", "by_record.csv", "events.csv")]

    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3.2))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.8, bottom=0.15)
    ax.axis("off")
    for i, label in enumerate(["60 fixed source trials\n7 people / bilateral gyro\n128Hz, rad/s", "Unit + causal rate adapter\nrad/s to deg/s, 128 to 60Hz\nKnown 1/6s filter delay", "Frozen FoG-STAR RF\n66 features, cutoff 0.23\nNo target fitting/calibration", "Native128Hz evaluation\nAll raw records and events\nUnknown support retained"]):
        x = 0.12 + i * 0.255
        ax.text(x, 0.55, label, ha="center", va="center", fontsize=9, transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.75", "facecolor": "#edf4f7", "edgecolor": "#438da0"})
        if i < 3:
            ax.annotate("", xy=(x + 0.15, 0.55), xytext=(x + 0.113, 0.55), xycoords="axes fraction", arrowprops={"arrowstyle": "->"})
    fig.suptitle("Zero-shot external domain transfer | anatomical axis alignment UNVERIFIED", fontsize=12)
    fig.text(0.5, 0.08, "Original xyz kept before scoring; no rotation search, no new clinical or step-success claim", ha="center", fontsize=10)
    save(fig, "external_pipeline")

    metrics = summary["window_metrics"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.15, top=0.8)
    fields = ["precision", "recall", "fog_f1", "average_precision"]
    bars = ax.bar(["Precision", "Recall", "FoG F1", "AP"], [metrics[k] for k in fields], color="#438da0")
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set(ylim=(0, 1.08), ylabel="Window metric")
    fig.suptitle(f"External cohort | {metrics['windows']} overlapping windows\nFixed 0.23 cutoff; unit/rate adapted, axes not harmonized", fontsize=11)
    save(fig, "external_window_metrics")

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.81)
    x = np.arange(len(people))
    for i, field in enumerate(("precision", "recall", "fog_f1")):
        bars = ax.bar(x + (i - 1) * 0.25, people[field], width=0.24, label=field)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    ax.set_xticks(x, [f"P{p}" for p in people.source_person])
    ax.set(ylim=(0, 1.12), ylabel="Window metric")
    ax.legend(ncol=3, fontsize=9)
    fig.suptitle("All seven external participants | same frozen model, no target-based selection", fontsize=11)
    save(fig, "external_by_person")

    native = summary["native128_time_metrics"]
    truth_supported = events.support_s.sum()
    known_negative = native["support_s"] - truth_supported
    parts = [[native["true_alarm_s"], truth_supported - native["true_alarm_s"], native["truth_s"] - truth_supported],
             [native["false_alarm_s"], known_negative - native["false_alarm_s"], native["observed_s"] - native["truth_s"] - known_negative]]
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.8))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.17, top=0.8, hspace=0.65)
    for ax, values, label in zip(axes, parts, ("Raw FoG", "Raw non-FoG"), strict=True):
        left = 0
        for value, color, name in zip(values, ("#be7f39", "#438da0", "#c9cdd0"), ("Alarm", "Supported, no alarm", "Unknown"), strict=True):
            ax.barh([label], [max(0, value)], left=left, color=color, label=name)
            if value > sum(values) * 0.06:
                ax.text(left + value / 2, 0, f"{value:.1f}s", ha="center", va="center", fontsize=9)
            left += value
        ax.set(xlim=(0, max(sum(values), 1)), xlabel="Native observed seconds")
    axes[0].legend(ncol=3, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.02))
    fig.suptitle("Native128Hz truth | alarms begin after the explicit resampling delay", fontsize=11)
    save(fig, "external_time_support")

    replay = json.loads((run / "evaluation/replay_data.json").read_text("utf-8"))
    selected = [1, int(records.sort_values(["truth_s", "record_id"], ascending=[False, True]).iloc[0].record_id)]
    write_json(output / "selected_records.json", {"rule": "First manifest trial and most native FoG seconds; no score-based selection", "record_ids": sorted(set(selected))})
    for record_id in sorted(set(selected)):
        record = next(r for r in replay if r["record_id"] == record_id)
        points = np.array([[t, a if a is not None else np.nan, b if b is not None else np.nan] for t, a, b in record["signals"]])
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.1, top=0.85, hspace=0.27)
        if len(points):
            axes[0].plot(points[:, 0], points[:, 1], lw=0.7, label="Adapted left gyro magnitude")
            axes[0].plot(points[:, 0], points[:, 2], lw=0.7, label="Adapted right gyro magnitude")
        for a, b, probability in record["decisions"]:
            axes[1].plot([a, b], [probability, probability], color="#438da0", lw=1)
        axes[1].axhline(0.23, color="#be7f39", linestyle="--", label="Frozen threshold 0.23")
        intervals = record["intervals"]
        axes[2].broken_barh([(a, b - a) for a, b in intervals["supported"]], (0.85, 0.3), facecolors="#d4d7da")
        axes[2].broken_barh([(a, b - a) for a, b in intervals["alarm"]], (0.85, 0.3), facecolors="#438da0")
        axes[2].broken_barh([(a, b - a) for a, b in intervals["truth"]], (-0.15, 0.3), facecolors="#8c7aa5")
        for ax in axes:
            for a, b in intervals["truth"]:
                ax.axvspan(a, b, color="#8c7aa5", alpha=0.08)
            ax.set_xlim(record["native_start_s"], record["native_stop_s"])
            ax.grid(alpha=0.15)
        axes[0].set_ylabel("deg/s\n(display 15Hz)")
        axes[0].legend(fontsize=8)
        axes[1].set(ylim=(0, 1), ylabel="Available score")
        axes[1].legend(fontsize=8)
        axes[2].set(yticks=[0, 1], yticklabels=["Native FoG", "Model alarm"], ylim=(-0.5, 1.5), xlabel="Source time (s); score availability includes 1/6s delay")
        fig.suptitle(f"External record {record_id}, P{record['source_person']} | axes UNVERIFIED\nAdapted signals at logical time; probability at actual availability; gray=supported non-alarm", fontsize=11)
        save(fig, f"external_record_{record_id:03}")
    images = "".join(f'<h2>{p.stem}</h2><img src="{p.name}" alt="{p.stem}">' for p in sorted(output.glob("*.png")))
    (output / "index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>A1 Stanford 外部迁移</title><style>body{max-width:1200px;margin:30px auto;padding:0 20px;font:16px/1.7 sans-serif}img{max-width:100%}td,th{padding:6px}.table{overflow:auto}</style>'
        '<h1>A1：Stanford 零样本外部迁移</h1><p>全部60条试验、7人；固定FoG-STAR双踝RF与0.23阈值，无目标训练/校准。单位和采样已适配，'
        '解剖轴系未核实统一；原xyz原样保留，没有旋转试分。本结果包含安装/任务/人群/标注等域差异，不能称完全同分布验证或临床效果。</p>'
        '<p>预测在逻辑窗口末端加1/6秒后才可用；原始事件保留128Hz，不删无支持事件。事件任意重叠不等于准确率或一对一事件F1。'
        '没有转向意图、真实落脚或提示响应结论。</p>'
        f'<pre>{json.dumps(summary, indent=2, ensure_ascii=False)}</pre>'
        '<h2>全部参与者</h2><div class="table">' + people.to_html(index=False, float_format=lambda x: f"{x:.4f}") + '</div>'
        '<h2>全部原始记录</h2><div class="table">' + records.to_html(index=False, float_format=lambda x: f"{x:.4f}") + '</div>' + images + '</html>', encoding="utf-8")
    write_json(output / "provenance.json", {"input_lock_sha256": sha(run / "INPUT_LOCK.json"),
        "evaluation_completion_sha256": sha(run / "evaluation/EVALUATION_COMPLETED.json"), "report_source_sha256": sha(Path(__file__)),
        "files": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print(f"Saved {len(list(output.glob('*.png')))} PNG/SVG pairs and HTML", flush=True)


if __name__ == "__main__":
    main()

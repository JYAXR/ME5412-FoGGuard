"""Build Chinese slide-ready figures and aggregate data from saved A2 evidence.

No retraining, threshold selection or TEST scoring. Raw/per-window patient
data are read locally for a fixed-rule example and are not copied to the pack.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Patch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from fogguard.turning.learned import KEYS

NAVY = "#17324D"
BLUE = "#4273DF"
TEAL = "#0F9488"
ORANGE = "#D18A32"
RED = "#C75E64"
MUTED = "#65758B"
GRID = "#E6EBF1"
BACKGROUND = "#F7F9FC"
COLORS = {"straight": TEAL, "left": BLUE, "right": ORANGE, "unknown": "#CBD3DF"}
LABELS = {"straight": "直走", "left": "左转", "right": "右转", "unknown": "不确定"}
SCOPE = "14 人训练 · 4 人开发验证 · 最终 TEST 未评分"
CAPTIONS = {
    "01_overview": "在 1130 个方向明确的有效窗口上，重力投影随机森林准确率为 66.99%，三类 macro-F1 为 71.97%。拒识计为错误；这些是开发验证结果。",
    "02_confusion_main": "主评价混淆矩阵：行是真实方向，列为预测，格内为窗口数及行百分比。右转有 170/357 个窗口被拒识，是主要限制之一。",
    "02b_confusion_full": "补充评价包含 791 个不确定真值，共 1921 个有效窗口。四类准确率为 68.66%，四类 macro-F1 为 65.58%，与主三类评价范围不同。",
    "03_class_and_people": "直走、左转和右转召回率分别为 89.78%、59.12% 和 41.18%。四名验证参与者准确率差异明显，不能将窗口数等同于独立受试者数。",
    "04_real_timeline": "固定规则选取首条包含直走与转向的验证记录，未按成绩筛选。预测从输入窗口末点才可用，最多保持 0.5 秒；图中保留拒识、错误与边界滞后。",
    "05_preturn_feasibility": "提前预测作为可行性研究：训练内选定阈值后，0.5 秒与 1 秒模型仅检出 0/41 与 2/41 个转向事件；降低阈值提高召回伴随约 8 次/分钟误报。",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure():
    plt.rcParams.update({
        "font.family": "Microsoft YaHei", "axes.unicode_minus": False,
        "font.size": 12, "text.color": NAVY, "axes.labelcolor": NAVY,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "path", "savefig.facecolor": "white",
    })


def canvas(title, subtitle, tag):
    fig = plt.figure(figsize=(40/3, 7.5), facecolor="white")
    fig.text(0.055, 0.94, title, fontsize=23, weight="bold", va="center")
    fig.text(0.055, 0.883, subtitle, fontsize=11, color=MUTED)
    fig.text(0.95, 0.943, f"FoGGuard / A2 / {tag}", ha="right", fontsize=9, color=MUTED)
    return fig


def footer(fig, text):
    fig.text(0.055, 0.055, text, fontsize=9, color=MUTED, va="center", linespacing=1.7)


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=240)
    fig.savefig(output / f"{name}.svg")
    plt.close(fig)


def tidy(ax, horizontal=False):
    ax.set_axisbelow(True)
    ax.grid(axis="x" if horizontal else "y", color=GRID, linewidth=0.8)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


def overview(current, output):
    primary = current["primary_known_targets"]
    baseline = current["yaw_baseline"]["primary_known_targets"]
    full_rf = current["candidate_results"][0]["metrics"]["primary_known_targets"]
    fig = canvas("当前方向识别：A2 主要成果", SCOPE + " ｜ 主评价 n = 1130", "01")
    cards = [("主准确率", primary["accuracy"], "757 / 1130 个窗口正确"),
             ("三类 macro-F1", primary["macro_f1"], "直走、左转、右转的 F1 等权平均"),
             ("明确方向预测覆盖率", primary["prediction_coverage"], "864 / 1130 个窗口未拒识")]
    for x, (label, value, note) in zip([0.055, 0.36, 0.665], cards, strict=True):
        fig.add_artist(FancyBboxPatch((x, 0.65), 0.28, 0.17,
                       boxstyle="round,pad=0.009,rounding_size=0.012",
                       transform=fig.transFigure, fc=BACKGROUND, ec=GRID, linewidth=0.8))
        fig.text(x + 0.018, 0.785, label, fontsize=11, color=MUTED)
        fig.text(x + 0.018, 0.704, f"{value:.2%}", fontsize=32, color=TEAL, weight="bold")
        fig.text(x + 0.018, 0.667, note, fontsize=9, color=MUTED)
    ax = fig.add_axes([0.085, 0.22, 0.57, 0.34])
    x = np.arange(3)
    acc = [baseline["accuracy"], full_rf["accuracy"], primary["accuracy"]]
    f1 = [baseline["macro_f1"], full_rf["macro_f1"], primary["macro_f1"]]
    for offset, values, color, label in [(-0.18, acc, TEAL, "准确率"),
                                        (0.18, f1, BLUE, "三类 macro-F1")]:
        bars = ax.bar(x + offset, np.array(values) * 100, width=0.32, color=color, label=label)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width()/2, value*100 + 1.7,
                    f"{value:.2%}", ha="center", fontsize=10, weight="bold")
    ax.set(ylim=(0, 100), yticks=[0, 25, 50, 75, 100], ylabel="百分比（%）",
           xticks=x, xticklabels=["角速度阈值基线", "完整特征 RF\n216 维", "重力投影 RF\n12 维 · 最终方案"])
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.21), ncol=2, frameon=False, fontsize=11)
    tidy(ax)
    fig.text(0.715, 0.52, "可解释的轻量实现", fontsize=15, weight="bold")
    fig.text(0.715, 0.415, "背部 + 左踝 IMU\n12 维重力投影特征\n随机森林：300 棵树\n2 秒窗口，每 0.5 秒更新", fontsize=12,
             linespacing=1.9, va="center")
    fig.text(0.715, 0.235, "相较角速度规则基线\nmacro-F1 +5.96 个百分点\n准确率 +0.44 个百分点", color=TEAL,
             fontsize=11, linespacing=1.7)
    footer(fig, "主评价仅含真实方向明确的窗口；模型输出“不确定”仍计为错误。\n候选方案使用开发验证集选择；本图不表示独立盲测或系统联调成绩。")
    save(fig, output, "01_overview")


def confusion(current, output, full=False):
    report = current["full_stream_including_unknown"] if full else current["primary_known_targets"]
    matrix = np.array(report["confusion_matrix"])
    if not full:
        matrix = matrix[:3]
    normalized = matrix / matrix.sum(1, keepdims=True) * 100
    title = "补充混淆矩阵：包含不确定真值" if full else "主评价混淆矩阵：错误与拒识可追溯"
    subtitle = ("完整有效流 n = 1921 ｜ 其中不确定真值 791 个窗口" if full else
                "明确方向真值 n = 1130 ｜ 格内：窗口数 / 该真实类别百分比")
    fig = canvas(title, subtitle, "02b" if full else "02")
    ax = fig.add_axes([0.12, 0.22, 0.60, 0.59])
    cmap = LinearSegmentedColormap.from_list("a2_blues", ["#F2F5FA", "#3669B5", NAVY])
    ax.imshow(normalized, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for (i, j), count in np.ndenumerate(matrix):
        value = normalized[i, j]
        ax.text(j, i, f"{count}\n{value:.2f}%", ha="center", va="center",
                fontsize=16, color="white" if value >= 38 else NAVY, linespacing=1.7)
    classes = ["straight", "left", "right", "unknown"]
    ax.set(xticks=range(4), xticklabels=[LABELS[c] for c in classes],
           yticks=range(len(matrix)),
           yticklabels=[f"{LABELS[c]}  n={matrix[i].sum()}" for i, c in enumerate(classes[:len(matrix)])],
           xlabel="预测类别", ylabel="真实类别")
    ax.tick_params(length=0, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.77, 0.75, "完整流评价" if full else "主结果", fontsize=16, weight="bold")
    fig.text(0.77, 0.695, "准确率", fontsize=12, color=MUTED)
    fig.text(0.77, 0.625, f"{report['accuracy']:.2%}", fontsize=26, color=TEAL, weight="bold")
    fig.text(0.77, 0.44, ("四类 macro-F1\n65.58%\n\n此范围包含其他活动\n及混合窗口的拒识真值。" if full else
                        "右转中 170 / 357\n被预测为不确定。\n\n主要限制是右转\n漏检与拒识较多。"),
             fontsize=12, linespacing=1.7, va="center")
    footer(fig, ("四类评价包含真实不确定状态，与主三类评价的样本范围和 macro-F1 定义不同。" if full else
                 "对应主准确率 66.99% 和三类 macro-F1 71.97%。不确定预测不从分母中删除。") +
           "\n" + SCOPE + "；成绩属于开发验证。")
    save(fig, output, "02b_confusion_full" if full else "02_confusion_main")


def class_people(current, output):
    primary = current["primary_known_targets"]
    fig = canvas("逐类表现与跨人差异", "明确方向窗口：直走 499 · 左转 274 · 右转 357", "03")
    ax = fig.add_axes([0.07, 0.23, 0.49, 0.54])
    x = np.arange(3)
    for offset, name, label, color in [(-0.24, "precision", "精确率", BLUE),
                                       (0, "recall", "召回率", TEAL),
                                       (0.24, "f1", "F1", "#8799B5")]:
        values = [primary["per_class"][c][name] * 100 for c in ["straight", "left", "right"]]
        bars = ax.bar(x + offset, values, width=0.21, color=color, label=label)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x()+bar.get_width()/2, value+1.6, f"{value:.1f}", ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=["直走", "左转", "右转"], ylim=(0, 108),
           ylabel="百分比（%）", yticks=[0, 25, 50, 75, 100])
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.14), ncol=3, frameon=False, fontsize=11)
    tidy(ax)
    ax2 = fig.add_axes([0.72, 0.23, 0.23, 0.54])
    subjects = list(current["per_subject"].items())
    values = [m["accuracy"]*100 for _, m in subjects]
    ax2.barh(np.arange(4), values, color=[ORANGE, BLUE, BLUE, BLUE], height=0.5)
    for i, value in enumerate(values):
        ax2.text(value+1.5, i, f"{value:.2f}%", va="center", fontsize=10)
    ax2.set(yticks=np.arange(4), yticklabels=[f"参与者 {s}\nn={m['windows']}" for s, m in subjects],
            xlim=(0, 115), xticks=[0, 50, 100], xlabel="准确率（%）")
    ax2.invert_yaxis()
    ax2.set_title("逐人准确率", loc="left", pad=22, fontsize=14, weight="bold")
    tidy(ax2, horizontal=True)
    footer(fig, "左：P/R/F1 均在 1130 个明确方向窗口上计算；未知预测计入漏检。\n右：同一模型在四名开发验证参与者上的表现；样本量不等、窗口有重叠，跨人稳定性仍需改进。")
    save(fig, output, "03_class_and_people")


def lane(ax, times, labels, widths, y=0.15, height=0.7):
    for c in ["straight", "left", "right", "unknown"]:
        intervals = [(float(t), float(w)) for t, label, w in zip(times, labels, widths, strict=True)
                     if label == c and w > 0]
        if intervals:
            ax.broken_barh(intervals, (y, height), facecolors=COLORS[c], edgecolors="none")


def timeline(raw_path, runtime, truth, output):
    merged = runtime.merge(truth, on=KEYS + ["window_start_s", "window_end_s"],
                           validate="one_to_one", how="left", sort=False)
    selected = None
    for key, group in merged.groupby(KEYS, sort=True):
        known = set(group.loc[group.input_valid, "target"])
        if "straight" in known and ({"left", "right"} & known):
            selected = tuple(map(int, key))
            break
    if selected is None:
        raise ValueError("No eligible validation timeline")
    wanted = ["timestamp", "subjectID", "sessionID", "taskID", "activity",
              "back_gyro_x", "ankleL_gyro_x", "ankleL_gyro_y", "ankleL_gyro_z"]
    chunks = []
    for chunk in pd.read_csv(raw_path, usecols=wanted, chunksize=50000):
        mask = np.ones(len(chunk), dtype=bool)
        for c, v in zip(["subjectID", "sessionID", "taskID"], selected, strict=True):
            mask &= chunk[c].eq(v).to_numpy()
        if mask.any():
            chunks.append(chunk.loc[mask])
    raw = pd.concat(chunks, ignore_index=True)
    mask = np.ones(len(merged), dtype=bool)
    for c, v in zip(KEYS, selected, strict=True):
        mask &= merged[c].eq(v).to_numpy()
    model = merged.loc[mask].copy()
    origin = raw.timestamp.iloc[0]
    end = min(60.0, float(raw.timestamp.iloc[-1] - origin))
    raw = raw.loc[raw.timestamp <= origin + end]
    model = model.loc[model.available_at_s <= origin + end]
    t = raw.timestamp.to_numpy() - origin
    fig = canvas("真实记录：信号、真值与模型输出对齐", "固定规则选例；保留全部错误与拒识，不按预测质量挑选", "04")
    axes = [fig.add_axes([0.15, 0.58, 0.80, 0.21]),
            fig.add_axes([0.15, 0.43, 0.80, 0.08]),
            fig.add_axes([0.15, 0.31, 0.80, 0.08]),
            fig.add_axes([0.15, 0.19, 0.80, 0.08])]
    axes[0].plot(t, raw.back_gyro_x, color=BLUE, linewidth=0.9, label="背部 x 轴角速度")
    axes[0].set_ylabel("角速度\n（°/s）", fontsize=11)
    tidy(axes[0])
    axes[0].legend(loc="upper right", frameon=False, fontsize=10)
    activity_map = {1: "straight", 6: "right", 7: "left"}
    sample_labels = [activity_map.get(int(a), "unknown") for a in raw.activity]
    lane(axes[1], t, sample_labels, np.minimum(1/60, end-t))
    ends = model.available_at_s.to_numpy() - origin
    widths = np.minimum(0.5, end-ends)
    lane(axes[2], ends, model.target.to_list(), widths)
    lane(axes[3], ends, model.direction.to_list(), widths)
    for i, label in zip([1, 2, 3], ["样本级\n活动真值", "两秒窗口\n评价真值", "当前方向\n模型预测"], strict=True):
        axes[i].set(yticks=[], ylim=(0, 1))
        axes[i].set_ylabel(label, rotation=0, ha="right", va="center", labelpad=15, fontsize=10)
        axes[i].spines["left"].set_visible(False)
        if i != 3:
            axes[i].set_xticks([])
    warmup = max(0, float(model.available_at_s.iloc[0] - origin))
    for ax in axes:
        ax.set_xlim(0, end)
        ax.axvspan(0, warmup, color="#E6EBF1", alpha=0.65, zorder=0)
    axes[0].set_xticklabels([])
    axes[3].set_xlabel("记录内时间（秒）", fontsize=11)
    patches = [Patch(facecolor=COLORS[c], label=LABELS[c] if c != "unknown" else "不确定 / 其他活动")
               for c in COLORS]
    fig.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.53, 0.088), ncol=4,
               frameon=False, fontsize=10)
    footer(fig, "原始轴信号仅作展示，模型使用窗口内重力投影。灰底起始段为窗口尚未完成；窗口输出从末点最多保持 0.5 秒。\n样本级标签与两秒窗口标签定义不同，因此边界可能不一致；本图不是提前转向预测。")
    save(fig, output, "04_real_timeline")
    return {"record_key": dict(zip(KEYS, selected, strict=True)),
            "selection": "First sorted validation record with valid straight and left/right targets; no prediction-quality filtering",
            "shown_duration_s": end, "hold_limit_s": 0.5,
            "raw_values_or_per_window_outputs_in_pack": False}


def feasibility(binary, audit, output):
    fig = canvas("提前预测：有实验支撑的可行性研究", "六种候选组合 · 模型与阈值在 TRAIN 参与者隔离 OOF 中选择", "05")
    ax = fig.add_axes([0.12, 0.48, 0.32, 0.28])
    horizons = ["0.5", "1.0"]
    detected = [binary["results"][h]["validation"]["full_stream"]["detected_events"] for h in horizons]
    ax.barh([0, 1], detected, height=0.42, color=TEAL, label="已检出转向")
    ax.barh([0, 1], 41-np.array(detected), left=detected, height=0.42,
            color="#E2E7EF", label="未检出")
    for i, n in enumerate(detected):
        ax.text(42, i, f"{n}/41", va="center", fontsize=15, weight="bold", color=TEAL)
    ax.set(yticks=[0, 1], yticklabels=["至少提前 0.5 秒", "至少提前 1.0 秒"],
           xlim=(0, 50), xticks=[0, 20, 40], xlabel="可评价转向事件数")
    ax.invert_yaxis()
    ax.set_title("固定阈值下的转向检出", loc="left", pad=18, fontsize=14, weight="bold")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.08, -0.46), ncol=2, frameon=False, fontsize=10)
    tidy(ax, horizontal=True)
    fig.text(0.075, 0.25, "二分类准确率约 91%，\n但始终预测“无转向”也有 91.13%。\n左右方向正确事件：两项均为 0/41。", fontsize=12,
             linespacing=1.9, color=NAVY)
    ax2 = fig.add_axes([0.58, 0.25, 0.36, 0.51])
    ax2.fill_between([0, 1], 50, 100, color="#DFF1E9", alpha=0.9)
    for h, color, marker in [("0.5", BLUE, "o"), ("1.0", ORANGE, "s")]:
        points = audit["results"][h]["diagnostic_points"]
        points = sorted(points, key=lambda p: -p["threshold"])
        ax2.plot([p["false_alarms_per_minute"] for p in points],
                 [p["recall"]*100 for p in points], color=color, label=f"{h} 秒：诊断曲线", linewidth=2)
        m = binary["results"][h]["validation"]
        ax2.scatter([m["full_stream"]["false_alarms_per_minute"]],
                    [m["walking_binary"]["recall"]*100], s=60, marker=marker, color=color,
                    edgecolors=NAVY, zorder=4)
    ax2.axvline(1, color=MUTED, linestyle="--", linewidth=1)
    ax2.axhline(50, color=MUTED, linestyle="--", linewidth=1)
    ax2.set(xlim=(-0.1, 10), ylim=(-2, 103), xlabel="完整流误报次数 / 分钟", ylabel="转向召回率（%）")
    ax2.set_title("提高召回率的误报代价", loc="left", pad=18, fontsize=14, weight="bold")
    ax2.text(1.3, 55, "研究目标：召回 ≥ 50%\n误报 ≤ 1 次/分钟", fontsize=9, color=MUTED)
    ax2.legend(loc="upper left", frameon=False, fontsize=10)
    tidy(ax2)
    footer(fig, "阈值曲线只作验证集诊断，没有用于重新选择发布阈值；实心标记是 TRAIN 选择的工作点。\n降低阈值可获得 63–78% 召回率，但伴随约 8 次/分钟误报。当前不支持接入正式提示决策。")
    save(fig, output, "05_preturn_feasibility")


def export_tables(current, binary, audit, output):
    data = output / "data"
    data.mkdir()
    primary, full = current["primary_known_targets"], current["full_stream_including_unknown"]
    headline = [
        ("primary_accuracy", primary["accuracy"], 1130, "明确方向真值；预测拒识计错"),
        ("primary_macro_f1", primary["macro_f1"], 1130, "直走/左转/右转三类等权"),
        ("prediction_coverage", primary["prediction_coverage"], 1130, "明确方向中未拒识比例"),
        ("accepted_accuracy", primary["accepted_accuracy"], 864, "仅未拒识样本；不能替代主准确率"),
        ("full_stream_accuracy", full["accuracy"], 1921, "包含791个不确定真值"),
        ("full_stream_macro_f1", full["macro_f1"], 1921, "包含unknown的四类等权"),
    ]
    pd.DataFrame(headline, columns=["metric", "value", "denominator_windows", "scope"]).to_csv(
        data / "headline_metrics.csv", index=False, encoding="utf-8-sig")
    rows = []
    for source, scope in [(primary, "known_direction_targets"), (full, "full_stream")]:
        for c, values in source["per_class"].items():
            rows.append({"scope": scope, "class": c, "class_cn": LABELS[c], **values})
    pd.DataFrame(rows).to_csv(data / "per_class.csv", index=False, encoding="utf-8-sig")
    rows = [{"subject_id": key, "windows": m["windows"], "accuracy": m["accuracy"],
             "macro_f1": m["macro_f1"], "coverage": m["prediction_coverage"]}
            for key, m in current["per_subject"].items()]
    pd.DataFrame(rows).to_csv(data / "per_subject.csv", index=False, encoding="utf-8-sig")
    rows = [{"method": "yaw_rule", "features": None,
             **{k: current["yaw_baseline"]["primary_known_targets"][k] for k in ["accuracy", "macro_f1"]}}]
    for candidate in current["candidate_results"]:
        rows.append({"method": candidate["feature_set"], "features": 216 if candidate["feature_set"] == "full" else 12,
                     **{k: candidate["metrics"]["primary_known_targets"][k] for k in ["accuracy", "macro_f1"]}})
    pd.DataFrame(rows).to_csv(data / "method_comparison.csv", index=False, encoding="utf-8-sig")
    for name, report in [("confusion_main", primary), ("confusion_full", full)]:
        matrix = np.array(report["confusion_matrix"])
        labels = report["classes"]
        if name == "confusion_main":
            matrix, labels = matrix[:3], labels[:3]
        pd.DataFrame(matrix, index=labels, columns=report["classes"]).rename_axis("truth").to_csv(
            data / f"{name}.csv", encoding="utf-8-sig")
    rows = []
    for h in ["0.5", "1.0"]:
        report = binary["results"][h]
        metrics = report["validation"]
        rows.append({"minimum_lead_s": float(h), "accuracy": metrics["walking_binary"]["accuracy"],
                     "turn_recall": metrics["walking_binary"]["recall"],
                     "precision": metrics["walking_binary"]["precision"],
                     "F1": metrics["walking_binary"]["f1"], "AUROC": metrics["walking_binary"]["AUROC"],
                     **{k: metrics["full_stream"][k] for k in ["false_alarms_per_minute", "detected_events",
                                                              "event_count", "correct_direction_events"]},
                     "selected_threshold": report["threshold"], "selection": "TRAIN participant OOF"})
    pd.DataFrame(rows).to_csv(data / "preturn_feasibility.csv", index=False, encoding="utf-8-sig")
    rows = [{"horizon_s": float(h), **point, "scope": "validation diagnostic only"}
            for h, result in audit["results"].items() for point in result["diagnostic_points"]]
    pd.DataFrame(rows).to_csv(data / "preturn_threshold_diagnostic.csv", index=False, encoding="utf-8-sig")
    (data / "README.md").write_text("""# 汇总数据字段说明

- headline_metrics：主指标及其样本范围；value 是 0–1 比例。
- method_comparison：同一明确方向范围的方法对比；accuracy / macro_f1 是 0–1 比例。
- per_class：分别提供 known_direction_targets（1130 个窗口）和 full_stream（1921 个窗口）的 P/R/F1；support 是计数。
- per_subject：四名开发验证参与者的窗口数、准确率、macro-F1 和覆盖率。
- confusion_main / confusion_full：行是真值，列是预测，值为窗口计数。
- preturn_feasibility：二分类可行性结果，召回率、精确率和准确率为比例，事件数为整数，误报率单位为次/分钟。
- preturn_threshold_diagnostic：固定模型在开发验证上的诊断点，未用于重新挑选发布阈值。

主三类与完整四类评价范围不同，不应混合计算。未包含患者原始数据或逐窗输出。
""", "utf-8")


def write_readme(output):
    text = """# A2 展示结果包

主线：**当前方向识别是主要成果；提前预测是有完整实验支撑的可行性研究。**

## 可直接引用的数据

| 指标 | 数值 | 评价范围 |
|---|---:|---|
| 当前方向主准确率 | **66.99%（757/1130）** | 真实方向明确；预测拒识仍计为错误 |
| 当前方向三类 macro-F1 | **71.97%** | 直走 / 左转 / 右转的 F1 等权平均 |
| 直走 / 左转 / 右转召回率 | **89.78% / 59.12% / 41.18%** | 同一主评价范围 |
| 预测覆盖率 | 76.46%（864/1130） | 明确方向中未拒识比例 |
| 未拒识样本准确率 | 87.62%（757/864） | 选择性准确率，不能替代主准确率 |
| 完整有效流四类准确率 / macro-F1 | 68.66% / 65.58% | 1921 个有效窗口，包含 791 个不确定真值 |

数据为 FoG-STAR 开发验证：14 人训练、4 人验证；TEST 未评分。窗口为 120 点、60 Hz，每 30 点更新。
模型为背部和左踝重力投影 12 维特征 + 300 棵树随机森林。当前方向候选使用开发验证选择，
以上不是独立盲测；存在跨人差异和较多右转拒识，不能据此宣称系统已完成四模块联调。
方向窗口真值要求某类活动占比至少 80%，且末样本仍为该类；其他活动或明显混合窗口保留不确定。

## 推荐展示顺序与图注

1. `01_overview.png`：方法与主结果，放在 A2 第一张结果页。
2. `02_confusion_main.png`：解释左右转与不确定输出，主范围一致。
3. `03_class_and_people.png`：逐类与逐人结果，说明右转和跨人稳定性的限制。
4. `04_real_timeline.png`：真实信号、样本级标签、窗口评价标签、模型输出的时间对齐。
5. `05_preturn_feasibility.png`：提前预测结论，建议放独立研究页或补充页。
6. `02b_confusion_full.png`：完整四类混淆矩阵，用于答辩补充。

所有图同时提供 PNG（3200×1800）和可缩放 SVG。PNG 适合直接插入 PPT；SVG 适合无损缩放。
`A2展示报告.html` 是包含图表的单文件离线报告；`data/` 内 CSV 是图表背后的汇总数据，可用 Excel 打开。
CSV 中准确率、P/R/F1 和覆盖率以 0–1 比例保存；计数、时间和误报率保留原单位，详见 data/README.md。
没有包含患者原始数据、逐窗文件或模型权重。

## 40 秒汇报稿

“A2 的主要成果是当前方向识别。我们使用背部和左踝 IMU，基于窗口内重力投影构造 12 维特征，
用随机森林识别直走、左转、右转和不确定。在四名参与者的开发验证数据上，明确方向窗口准确率为
66.99%，三类 macro-F1 为 71.97%，拒识也计为错误。直走识别较好，右转和跨人稳定性仍有提升空间。
我们另外完成了严格提前 0.5 秒及 1 秒的预测可行性研究，比较短时特征、双踝特征和两类模型。
在受限误报下转向召回仅为 0% 和 4.88%，因此目前保留为研究结论，不接入正式提示决策。”

## 提前预测的数据口径

| 任务 | 准确率 | 转向检出 | 完整流误报/分钟 |
|---|---:|---:|---:|
| 至少提前 0.5 秒 | 91.13% | 0/41 | 0.000 |
| 至少提前 1.0 秒 | 91.34% | 2/41（4.88%） | 0.131 |

始终预测“无转向”就有 91.13% 准确率。两项的左右方向正确事件均为 0/41。
模型及阈值在训练人员内部交叉验证选择；该四人验证集此前已被检查过，仍属于开发验证。
阈值诊断显示，提高召回会带来约 8 次/分钟误报，不支持将表面高准确率解释为有效预测。

## 图注

"""
    for name, caption in CAPTIONS.items():
        text += f"### {name}\n\n{caption}\n\n![{name}]({name}.png)\n\n"
    text += """## 来源与复现

FoG-STAR v3，DOI: https://doi.org/10.5281/zenodo.17838806，CC BY 4.0。
原始结果来自 A2 保存的方向评价、二分类评价和阈值诊断；具体输入、文件哈希及选例规则见 MANIFEST.json。
仓库内可用 `python -X utf8 scripts/build_a2_presentation.py --output reports/a2_presentation_NEW` 重新生成。
这是算法结果展示包，不包含 A1 感知融合、运动反馈或临床效果验证。
"""
    (output / "README.md").write_text(text, "utf-8")


def write_html(output):
    sections = []
    for name in ["01_overview", "02_confusion_main", "03_class_and_people", "04_real_timeline", "05_preturn_feasibility"]:
        source = base64.b64encode((output / f"{name}.svg").read_bytes()).decode()
        sections.append(f'<section><img alt="{html.escape(CAPTIONS[name])}" '
                        f'src="data:image/svg+xml;base64,{source}"><p>{html.escape(CAPTIONS[name])}</p></section>')
    body = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FoGGuard · A2 展示结果</title><style>
*{box-sizing:border-box}body{margin:0;background:#f1f4f8;color:#17324d;font-family:'Microsoft YaHei','Noto Sans SC',sans-serif}
main{max-width:1180px;margin:40px auto;padding:0 24px}header{margin:0 0 30px}h1{font-size:34px;margin:12px 0}header p{line-height:1.9;color:#65758b}
small{letter-spacing:2px;color:#0f9488}section{background:white;border:1px solid #e6ebf1;border-radius:16px;overflow:hidden;margin:26px 0;break-inside:avoid}
img{display:block;width:100%;height:auto}section p{font-size:14px;line-height:1.8;padding:0 30px 22px;color:#65758b}
footer{padding:20px 0 40px;font-size:13px;line-height:1.9;color:#65758b}@media print{main{margin:0;padding:0}body{background:white}section{page-break-after:always;border:0}}
</style><main><header><small>FOGGUARD / A2 / DEVELOPMENT VALIDATION</small><h1>当前方向识别与提前预测研究</h1>
<p>主要成果：当前方向识别 66.99% 准确率、71.97% 三类 macro-F1。<br>14 人训练、4 人开发验证；最终 TEST 未评分。提前预测保留为可行性研究。</p></header>
""" + "".join(sections) + """<footer>图表可直接插入 PPT，汇总 CSV 与逐图讲稿见展示包 README。<br>
数据来源：FoG-STAR v3，DOI 10.5281/zenodo.17838806，CC BY 4.0。开发验证结果不表示独立测试或临床效果。</footer></main></html>"""
    (output / "A2展示报告.html").write_text(body, "utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    if args.zip and args.zip.exists():
        parser.error("ZIP already exists")
    configure()
    source_paths = [Path("docs/a2/results.json"), Path("docs/a2/binary_results.json"),
                    Path("docs/a2/binary_threshold_audit.json")]
    direction, binary, audit = [json.loads(p.read_text("utf-8")) for p in source_paths]
    if direction["test_scored"] or binary["test_scored"]:
        raise ValueError("Unexpected TEST results in development presentation")
    current = direction["results"]["current"]
    runtime_path = Path("reports/a2_final_20261030/current/direction_runtime.csv")
    truth_path = Path("reports/a2_final_20261030/current/evaluation_truth.csv")
    raw_path = Path("data/raw/fogstar/sensor_data.csv")
    runtime, truth = pd.read_csv(runtime_path), pd.read_csv(truth_path)
    merged = runtime.merge(truth, on=KEYS + ["window_start_s", "window_end_s"], validate="one_to_one")
    primary = merged.loc[merged.input_valid & merged.target.isin(["straight", "left", "right"])]
    expected = current["primary_known_targets"]
    assert len(primary) == expected["windows"]
    assert np.isclose(accuracy_score(primary.target, primary.direction), expected["accuracy"])
    assert np.isclose(f1_score(primary.target, primary.direction, labels=["straight", "left", "right"],
                               average="macro", zero_division=0), expected["macro_f1"])
    actual = confusion_matrix(primary.target, primary.direction,
                              labels=["straight", "left", "right", "unknown"])
    np.testing.assert_array_equal(actual, expected["confusion_matrix"])
    args.output.mkdir(parents=True)
    overview(current, args.output)
    confusion(current, args.output)
    confusion(current, args.output, full=True)
    class_people(current, args.output)
    selection = timeline(raw_path, runtime, truth, args.output)
    feasibility(binary, audit, args.output)
    export_tables(current, binary, audit, args.output)
    write_readme(args.output)
    write_html(args.output)
    inputs = [*source_paths, runtime_path, truth_path]
    manifest = {"schema": "fogguard.a2.presentation.v1", "language": "zh-CN",
                "scope": SCOPE, "metrics_verified_against_saved_predictions": True,
                "raw_sha256": direction["raw_sha256"],
                "source_sha256": {p.as_posix(): sha(p) for p in inputs},
                "build_script_sha256": sha(Path(__file__)),
                "timeline": selection, "captions": CAPTIONS,
                "files_sha256": {p.relative_to(args.output).as_posix(): sha(p)
                                 for p in sorted(args.output.rglob("*")) if p.is_file()}}
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    if args.zip:
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED) as package:
            for p in sorted(args.output.rglob("*")):
                if p.is_file():
                    package.write(p, "A2_展示结果包/" + p.relative_to(args.output).as_posix())
    print(f"Saved figures, CSVs, captions, offline HTML and provenance to {args.output}")


if __name__ == "__main__":
    main()

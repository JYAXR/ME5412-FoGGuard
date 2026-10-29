"""Render verified post-hoc external diagnostics as PPT-ready PNG/SVG and local HTML."""

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
    parser.add_argument("--external-run", type=Path, default=Path("reports/a1_stanford_external_v1r1_20261029"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run, external, output = args.run_dir, args.external_run, args.output_dir
    if output.exists():
        parser.error("Use a new report directory")
    verification = json.loads((run / "verification.json").read_text("utf-8"))
    if verification["status"] != "PASS" or verification["diagnostic_lock_sha256"] != sha(run / "DIAGNOSTIC_LOCK.json"):
        raise ValueError("Verify diagnostics first")
    check_lock(run, json.loads((run / "DIAGNOSTIC_LOCK.json").read_text("utf-8")))
    check_lock(external / "evaluation", json.loads((external / "evaluation/EVALUATION_COMPLETED.json").read_text("utf-8")))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "artifacts/matplotlib-stanford-failure-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    output.mkdir(parents=True)
    colors = ["#397f93", "#bb623d", "#bfc5c9"]
    charts = []

    def save(fig, name, caption):
        for suffix in ("png", "svg"):
            fig.savefig(output / f"{name}.{suffix}", dpi=190, facecolor="white")
        plt.close(fig)
        charts.append((name, caption))

    people = pd.read_csv(run / "opportunity_by_person.csv")
    events = pd.read_csv(run / "event_opportunities.csv")
    source = pd.read_csv(run / "train_invariant_summaries.csv")
    target = pd.read_csv(run / "external_invariant_summaries.csv")
    predictions = pd.read_csv(external / "evaluation/predictions.csv")
    selected = json.loads((run / "selected_failures.json").read_text("utf-8"))
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    bottom = np.zeros(7)
    fields = ("model_covered", "missed_with_timely_self_majority", "missed_without_timely_self_majority")
    labels = ("Model alarm overlaps event", "Missed: timely self-majority window exists", "Missed: no timely self-majority window")
    for field, label, color in zip(fields, labels, colors, strict=True):
        bars = ax.bar([f"P{x}" for x in people.source_person], people[field], bottom=bottom, color=color, label=label)
        ax.bar_label(bars, labels=[str(x) if x >= 3 else "" for x in people[field]], label_type="center", fontsize=10)
        bottom += people[field]
    ax.set(ylabel="Raw native FoG events", ylim=(0, 67), title="78 of 80 missed events had a timely self-majority window")
    ax.legend(loc="upper left", fontsize=9)
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
    fig.text(0.02, 0.015, "211 events / all 60 records; saved cutoff 0.23. Opportunity uses event truth retrospectively; it is not a model or a ceiling.", fontsize=9)
    save(fig, "event_opportunities", "全部211个原始事件：漏掉的80次中，78次有事件自身占至少60/120点的窗口在事件仍发生时可用。机会由事后标签计算，不是算法成绩。")

    bins = pd.cut(events.duration_s, [0, 1, 2, 5, np.inf], labels=["<=1s", "1-2s", "2-5s", ">5s"])
    grouped = events.assign(duration_bin=bins).groupby("duration_bin", observed=False).agg(events=("covered", "size"), covered=("covered", "sum"), opportunity_miss=("missed_despite_self_majority", "sum"))
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    bottom = np.zeros(4)
    for values, label, color in zip((grouped.covered, grouped.opportunity_miss, grouped.events - grouped.covered - grouped.opportunity_miss), labels, colors, strict=True):
        bars = ax.bar(grouped.index.astype(str), values, bottom=bottom, color=color, label=label)
        ax.bar_label(bars, labels=[str(x) if x >= 3 else "" for x in values], label_type="center")
        bottom += values.to_numpy()
    ax.set(ylabel="Raw native FoG events", ylim=(0, 120), title="Longer events are also missed despite timely input opportunity")
    ax.legend(fontsize=8)
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
    fig.text(0.02, 0.015, "Duration bins are descriptive; any overlap counts as covered. No model fitting, rotation rescoring or threshold search.", fontsize=9)
    save(fig, "event_durations", "按事件时长保留所有分母；2–5秒事件中有40次具备及时窗口仍全程漏检，超过5秒有11次。不能仅归因于短事件。")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    for ax, group_name, title in zip(axes, ("pure_fog", "pure_nonfog"), ("All 120 truth samples are FoG", "All 120 truth samples are non-FoG"), strict=True):
        groups = [target.loc[target.source_person.eq(p) & target.window_group.eq(group_name), "fog_probability"] for p in range(1, 8)]
        ax.boxplot(groups, tick_labels=[f"P{p}\nn={len(g)}" for p, g in enumerate(groups, 1)], showfliers=False, widths=0.6)
        ax.axhline(0.23, color=colors[1], linestyle="--", label="Fixed cutoff 0.23")
        ax.set(title=title, ylim=(-0.03, 1.03), ylabel="Saved frozen-model score")
        ax.legend(fontsize=9)
    fig.suptitle("P6/P7 scores stay low even in pure FoG windows")
    fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
    fig.text(0.02, 0.015, "Overlapping windows are correlated. Mixed windows excluded here only; all remain in original evaluation. Boxes: Q1/median/Q3, 1.5-IQR whiskers.", fontsize=8)
    save(fig, "saved_scores_by_purity", "纯FoG窗口仍可低分：P6中位0.067、P7中位0.098，均低于固定0.23阈值。窗口相关；P6只有18个纯FoG窗口。")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    train = source.loc[source.window_group.eq("pure_fog")]
    groups = [train] + [target.loc[target.source_person.eq(p) & target.window_group.eq("pure_fog")] for p in range(1, 8)]
    for ax, field, title, unit in zip(axes.flat,
            ("left_vector_rms", "right_vector_rms", "left_freeze_ratio", "right_freeze_ratio"),
            ("Left ankle: vector RMS", "Right ankle: vector RMS", "Left: 3-8Hz / 0.5-<3Hz power", "Right: 3-8Hz / 0.5-<3Hz power"),
            ("deg/s (log scale)", "deg/s (log scale)", "Ratio (log scale)", "Ratio (log scale)"), strict=True):
        ax.boxplot([g[field] for g in groups], tick_labels=["TRAIN"] + [f"P{p}" for p in range(1, 8)], showfliers=False)
        ax.set(yscale="log", title=title, ylabel=unit)
        ax.axhline(train[field].median(), color=colors[0], linestyle=":")
    fig.suptitle("Rotation-invariant signal summaries differ across datasets / people")
    fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
    fig.text(0.02, 0.025, "Pure FoG only; TRAIN n=1154; P1..P7 n=156/164/307/794/106/18/267. Same completed 2s filter and spectral calculation.", fontsize=9)
    fig.text(0.02, 0.005, "Rigid axis rotation alone cannot remove these descriptor differences. This does not establish the cause of classification errors.", fontsize=9)
    save(fig, "invariant_signal_distributions", "旋转不变摘要仍有分布差异：源TRAIN纯FoG与全部7名外部参与者并列。差异不能由刚性轴旋转单独消除，但不是错误原因的因果证明。")

    replay = json.loads((external / "evaluation/replay_data.json").read_text("utf-8"))
    for case in selected:
        record = next(r for r in replay if r["record_id"] == case["record_id"])
        points = np.array([[t + 1 / 6, np.nan if a is None else a, np.nan if b is None else b] for t, a, b in record["signals"]])
        start, stop = case["start_s"], case["end_s"]
        limits = (max(record["native_start_s"], start - 4), min(record["native_stop_s"], stop + 4))
        frame = predictions.loc[predictions.record_id.eq(case["record_id"])].sort_values("available_at_s")
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
        visible = (points[:, 0] >= limits[0]) & (points[:, 0] <= limits[1])
        for side, color, name in ((1, colors[0], "Left"), (2, "#876cb1", "Right")):
            axes[0].plot(points[visible, 0], points[visible, side], lw=0.9, color=color, label=f"{name} gyro magnitude")
        rows = list(frame.itertuples())
        for i, row in enumerate(rows):
            a = row.available_at_s
            b = min(a + 0.5, rows[i + 1].available_at_s if i + 1 < len(rows) else np.inf)
            if b < limits[0] or a > limits[1]:
                continue
            axes[1].plot([a, b], [row.fog_probability] * 2, color=colors[0], lw=1.6)
            axes[2].plot([a, b], [row.truth_fog_fraction] * 2, color=colors[0], lw=1.5)
            t = row.window_start_s + np.arange(120) / 60
            fraction = ((t >= start - 1e-9) & (t < stop - 1e-9)).mean()
            axes[2].plot([a, b], [fraction] * 2, color=colors[1], linestyle=":", lw=1.5)
        axes[0].set(ylabel="Adapted gyro (deg/s)")
        axes[0].legend(fontsize=9)
        axes[1].axhline(0.23, color=colors[1], linestyle="--", label="Fixed cutoff 0.23")
        axes[1].set(ylabel="Saved FoG score", ylim=(-0.03, 1.03))
        axes[1].legend(fontsize=9)
        axes[2].axhline(0.5, color="#777777", linestyle="--")
        axes[2].plot([], [], color=colors[0], label="All FoG in past input window")
        axes[2].plot([], [], color=colors[1], linestyle=":", label="Selected event only in past input window")
        axes[2].set(ylabel="Input truth fraction", ylim=(-0.03, 1.08), xlabel="Availability time (s); adapted gyro displayed at logical time + 1/6s")
        axes[2].legend(fontsize=8, loc="upper right")
        for ax in axes:
            ax.axvspan(start, stop, color="#dbb85c", alpha=0.22)
            ax.set_xlim(limits)
        event = events.loc[events.record_id.eq(case["record_id"]) & np.isclose(events.start_s, start)].iloc[0]
        fig.suptitle(f"P{case['source_person']} / record {case['record_id']} | longest missed event with timely self-majority input\n{stop - start:.3f}s event; timely score max {event.max_timely_score:.3f}; no alarm overlaps selected event", fontsize=11)
        fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
        fig.text(0.02, 0.028, "Shading = native event truth. Input truth fractions are retrospective diagnostics, unavailable to the algorithm; 0.5s maximum decision holds.", fontsize=8)
        fig.text(0.02, 0.008, "Magnitude display uses saved 15Hz preview samples, not the full-rate input or a frequency-analysis plot.", fontsize=8)
        save(fig, f"failure_p{case['source_person']}", f"固定规则选出的P{case['source_person']}失败案例，record {case['record_id']}；阴影为原生真值事件，概率按真实可用时间显示。不是迈步/意图标签。")

    intro = """<h1>A1 外部失败诊断</h1><p>事后只读诊断：未训练、未旋转重评分、未调阈值；原外部F1仍为0.5657，覆盖131/211个事件。</p>
<p><b>主要发现：</b>80次完全漏报中78次具备及时的事件自身多数窗口；P7全部36次漏报均有机会。P6/P7纯FoG窗口中位分数仅0.067/0.098。旋转不变信号摘要仍有分布差异，不能单用轴方向解释这些摘要差异。</p>
<p><b>轴系：</b>两篇安装图仅支持条件推断diag(-1,+1,-1)：假设图示就是导出gyro轴并遵循右手系。未逐试验核实、未应用；不能称已完成轴校准。</p>
<p>图为PNG及SVG，可用于PPT。相关窗口不是独立样本；标签规则参考不是可部署算法或性能上限；分析不能证明错误因果或临床效果。</p>"""
    sections = "".join(f'<section><h2>{html.escape(caption)}</h2><a href="{name}.svg">SVG</a> · <a href="{name}.png">PNG</a><img src="{name}.png" alt="{html.escape(caption)}"></section>' for name, caption in charts)
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>A1 external failure diagnostic</title><style>body{max-width:1120px;margin:32px auto;padding:0 20px;background:#f6f8fa;color:#17303a;font:16px/1.7 system-ui}section{background:white;padding:22px;margin:24px 0;border:1px solid #dae2e6;border-radius:10px}h1{font-size:30px}h2{font-size:18px}img{width:100%;height:auto}a{color:#256f86}</style>' + intro + sections + '</html>'
    (output / "index.html").write_text(page, "utf-8")
    write_json(output / "REPORT_LOCK.json", {"diagnostic_lock_sha256": sha(run / "DIAGNOSTIC_LOCK.json"), "verification_sha256": sha(run / "verification.json"),
        "reporter_sha256": sha(Path(__file__)), "png_svg_pairs": len(charts), "files": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print(f"Rendered {len(charts)} PNG/SVG pairs and index.html")


if __name__ == "__main__":
    main()

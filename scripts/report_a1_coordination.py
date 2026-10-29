"""Render fixed TRAIN proxy profiles and locked validation comparisons."""

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
        parser.error("Use a new presentation directory")
    check_lock(run, json.loads((run / "MODEL_LOCK.json").read_text("utf-8")))
    check_lock(run / "evaluation", json.loads((run / "evaluation/EVALUATION_COMPLETED.json").read_text("utf-8")))
    if json.loads((run / "verification.json").read_text("utf-8"))["status"] != "PASS":
        raise ValueError("Independent verification required")
    os.environ.setdefault("MPLCONFIGDIR", str(root / "artifacts/matplotlib-a1-coordination-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True)

    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3.4))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.18)
    ax.axis("off")
    texts = ["Completed 2s window\n60Hz, both ankle gyros\nMissing input = unknown", "Existing axis features\nLeft 33 + right 33\nSame 66D control", "+12 coordination proxies\nEnergy / correlation / spectrum\n78D candidate", "Same fixed RF training\nTRAIN participant OOF cutoff\nLock before VALIDATION"]
    for i, label in enumerate(texts):
        x = 0.12 + i * 0.255
        ax.text(x, 0.55, label, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=0.8", "facecolor": "#edf5f6", "edgecolor": "#438da0"}, transform=ax.transAxes)
        if i < 3:
            ax.annotate("", xy=(x + 0.151, 0.55), xytext=(x + 0.113, 0.55), xycoords="axes fraction", arrowprops={"arrowstyle": "->"})
    fig.suptitle("One fixed feature-block ablation | no activity truth, no history or future samples", fontsize=12)
    fig.text(0.5, 0.08, "Short-window motion proxies are not measured footfalls, step success or pre-FoG prediction", ha="center", fontsize=10)
    save(fig, "feature_pipeline")

    config = json.loads((run / "config.json").read_text("utf-8"))
    reference = Path(config["reference_run"])
    cases = json.loads((reference / "input_audit/selected_pairs.json").read_text("utf-8"))
    features = pd.read_csv(run / "train_coordination_features.csv")
    spread = features.std(ddof=0).replace(0, 1)
    standardized = (features - features.mean()) / spread
    indices, labels = [], []
    for case in cases:
        for kind, field in (("non-FoG FP", "negative_index"), ("pure FoG", "positive_index")):
            indices.append(case[field])
            group = case["group"].replace("_nonfog_vs_fog", "").replace("nonfog_stand_vs_fog", "stand").replace("_", " ")
            labels.append(f"P{case['person']} {group}: {kind}")
    features.iloc[indices].assign(case=labels).to_csv(output / "fixed_train_pair_features.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 5.4))
    fig.subplots_adjust(left=0.24, right=0.92, bottom=0.25, top=0.81)
    colors = ax.imshow(standardized.iloc[indices].to_numpy(), cmap="coolwarm", vmin=-3, vmax=3, aspect="auto")
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=9)
    short = ["E asym", "|E asym|", "Env diff", "Corr 0", "Peak corr", "|Lag| s", "Corr .25s", "Corr .50s", "Gait asym", "Freeze asym", "Gait cosine", "Gait delta Hz"]
    ax.set_xticks(np.arange(12), short, rotation=40, ha="right", fontsize=9)
    fig.colorbar(colors, ax=ax, label="TRAIN z score (display clipped at +/-3)", fraction=0.04)
    fig.suptitle("Fixed previous TRAIN failure pairs | descriptive proxies, no pair reselection\nStandardization uses TRAIN only for this display; RF does not use this scaler", fontsize=11)
    save(fig, "fixed_train_pair_profiles")

    table = pd.read_csv(run / "evaluation/comparison.csv")
    own = table.loc[table.pool.eq("validation_own")].set_index("model")
    names = config["models"]
    pair = own.loc[names]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.16, top=0.84)
    x = np.arange(2)
    for i, field in enumerate(("precision", "recall", "fog_f1")):
        bars = ax.bar(x + (i - 1) * 0.25, pair[field], width=0.24, label=field)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x, ["66D bilateral control", "78D + coordination"])
    ax.set(ylim=(0, 1.12), ylabel="Window metric", title=f"VALIDATION | {int(pair.windows.iloc[0])} identical windows, fixed TRAIN-only selection")
    ax.legend(ncol=3, fontsize=9)
    save(fig, "matched_window_metrics")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.24, top=0.84, wspace=0.3)
    for ax, field, title in zip(axes, ("false_alarm_s", "time_recall", "decision_support_fraction"), ("False alarm seconds", "FoG-time recall", "Raw-time support"), strict=True):
        bars = ax.bar(["66D", "78D"], pair[field], color=["#438da0", "#be7f39"])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set(title=title, ylim=(0, max(float(pair[field].max()) * 1.2, 0.1)))
    fig.suptitle("All 20 raw VALIDATION records | missing decisions remain unknown", fontsize=12)
    save(fig, "alarm_and_support_tradeoffs")

    people = [pd.read_csv(run / "evaluation" / name / "by_subject_id.csv").set_index("subject_id") for name in names]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.16, top=0.83)
    x = np.arange(len(people[0]))
    for i, (name, person) in enumerate(zip(["66D", "78D"], people, strict=True)):
        bars = ax.bar(x + (i - 0.5) * 0.36, person.false_positive_window_rate, width=0.35, label=name)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x, [f"P{p}" for p in people[0].index])
    ax.set(ylim=(0, 1), ylabel="FP / non-FoG windows", title="VALIDATION participant false-positive rates | P10 has no positive windows")
    ax.legend()
    save(fig, "participant_false_positive_rates")

    common = table.loc[table.pool.eq("all_common") & ~table.model.eq("left_natural")]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.3, top=0.84)
    bars = ax.bar(common.model, common.fog_f1, color=["#438da0", "#be7f39", "#929797", "#929797", "#929797"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.set(ylim=(0, 1), ylabel="FoG F1", title=f"{int(common.windows.iloc[0])} common windows | historical inputs/selection differ")
    save(fig, "historical_common")
    gate = json.loads((run / "evaluation/development_gate.json").read_text("utf-8"))
    images = "".join(f'<h2>{p.stem}</h2><img src="{p.name}" alt="{p.stem}">' for p in sorted(output.glob("*.png")))
    (output / "index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>A1 左右协调对照</title>'
        '<style>body{max-width:1200px;margin:30px auto;font:16px/1.7 sans-serif;padding:0 20px}img{max-width:100%}td,th{padding:6px}</style>'
        '<h1>A1：双踝协调摘要消融</h1><p>相同双踝窗口，66维对照与新增12维的78维候选。TRAIN按人OOF选阈值、锁定后只读验证。'
        'VALIDATION已被开发使用，不是新的盲测；无新模型TEST成绩，不自动替换默认。配对固定自旧TRAIN审查，不能作为生理机制或独立分类证明。'
        '本模型不输出转向意图、真实落脚或迈步成功。原始事件任意重叠不是准确率或一对一事件F1。</p>'
        f'<p>预定开发门槛通过：{gate["passed"]}；条件：{gate["conditions"]}</p>'
        + table[["model", "pool", "windows", "precision", "recall", "fog_f1", "false_alarm_s", "time_recall"]].to_html(index=False, float_format=lambda v: f"{v:.4f}")
        + images + '</html>', encoding="utf-8")
    write_json(output / "provenance.json", {"model_lock_sha256": sha(run / "MODEL_LOCK.json"),
        "evaluation_completion_sha256": sha(run / "evaluation/EVALUATION_COMPLETED.json"), "report_source_sha256": sha(Path(__file__)),
        "kind": "Readonly display of fixed TRAIN pairs and locked metrics; no fitting/scoring", "test_windows_constructed": 0,
        "files": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print("Saved six PNG/SVG pairs, fixed TRAIN feature table, HTML and provenance", flush=True)


if __name__ == "__main__":
    main()

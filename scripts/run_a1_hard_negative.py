"""Training-only hard-negative diagnosis and one nested weighted-RF experiment."""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from run_a1_experiment import raw_audit, sha, validate_window_alignment, write_json
from sklearn.metrics import average_precision_score

from fogguard.data.fogstar import load_fogstar_windows
from fogguard.evaluation.a1_hard_negative import (
    fit_weighted,
    mining_scores,
    negative_weights,
    select_weighted_threshold,
)
from fogguard.evaluation.readiness import frozen_folds, split_runtime_truth
from fogguard.modeling.a1_fog import a1_matrix
from fogguard.modeling.a1_hard_negative import HardNegativeFoG
from fogguard.modeling.a1_multimodel import MultiModelFoG, model_scores


def training_profile(x, names, labels, fractions, groups, activities, mask, config,
                     excluded, output):
    """No outer test people enter this descriptive signal/feature comparison."""
    p, folds = mining_scores(x[mask], labels[mask], groups[mask], config, forbidden=excluded)
    weights, hard = negative_weights(labels[mask], fractions[mask], p, config)
    values = x[mask]
    features = {}
    for position, sensor, label in [("ankleL", "acc", "ankle_dynamic_rms_g"),
                                     ("back", "gyro", "back_gyro_rms_dps")]:
        indices = [names.index(f"{position}_{sensor}_{axis}__rms") for axis in "xyz"]
        features[label] = np.sqrt(np.sum(values[:, indices]**2, axis=1))
    gait = sum(values[:, names.index(f"ankleL_acc_{axis}__gait_energy")] for axis in "xyz")
    freeze = sum(values[:, names.index(f"ankleL_acc_{axis}__freeze_energy")] for axis in "xyz")
    features["ankle_energy_ratio"] = freeze / np.maximum(gait, 1e-12)
    categories = np.select([labels[mask] == 1, (labels[mask] == 0) & (fractions[mask] > 0), hard],
                            ["FoG", "mixed_negative", "hard_pure_negative"],
                            default="other_pure_negative")
    table = pd.DataFrame({"subject_id": groups[mask], "truth_fog": labels[mask],
                          "truth_fraction": fractions[mask], "activity_truth": activities[mask],
                          "mining_score": p, "weight": weights, "group": categories, **features})
    table.to_csv(output / "training_only_profile.csv", index=False)
    summaries = []
    for group, rows in table.groupby("group"):
        record = {"group": group, "windows": len(rows), "people": rows.subject_id.nunique()}
        for feature in features:
            for quantile, name in [(0.25, "q25"), (0.5, "median"), (0.75, "q75")]:
                record[f"{feature}_{name}"] = float(rows[feature].quantile(quantile))
        summaries.append(record)
    pd.DataFrame(summaries).to_csv(output / "training_profile_summary.csv", index=False)
    activity_rows = []
    for activity, rows in table.groupby("activity_truth", dropna=False):
        eligible = rows.truth_fraction.eq(0)
        activity_rows.append({"activity_truth": activity, "windows": len(rows),
                              "pure_negative_windows": int(eligible.sum()),
                              "hard_negative_windows": int(rows.group.eq("hard_pure_negative").sum()),
                              "fog_windows": int(rows.truth_fog.sum())})
    pd.DataFrame(activity_rows).to_csv(output / "training_activity_profile.csv", index=False)
    audit = {"scope": "First outer fold TRAINING people, participant-held-out mining scores",
             "training_people": sorted(set(groups[mask])), "excluded_outer_test": excluded,
             "mining_folds": folds, "hard_pure_negative_windows": int(hard.sum()),
             "training_windows": int(mask.sum()), "groups": summaries}
    write_json(output / "training_profile_audit.json", audit)
    return table


def plots(output, profile, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = output / "figures"
    directory.mkdir(exist_ok=True)

    def save(fig, name):
        fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    order = ["other_pure_negative", "hard_pure_negative", "FoG"]
    for ax, feature, title in zip(axes, ["ankle_dynamic_rms_g", "ankle_energy_ratio"],
                                  ["Ankle dynamic RMS (g)", "Ankle spectral energy ratio"], strict=True):
        rows = [profile.loc[profile.group.eq(group), feature].to_numpy() for group in order]
        ax.boxplot(rows, tick_labels=["Other pure\nnegative", "Hard pure\nnegative", "FoG"],
                   showfliers=False)
        ax.set_yscale("symlog", linthresh=0.001)
        ax.set_ylabel(title + " (symlog)")
        ax.grid(alpha=0.2, axis="y")
    fig.suptitle("TRAINING-only first outer fold | overlapping distributions, no activity gate")
    save(fig, "training_signal_profile")
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    fields = ["precision", "recall", "fog_f1", "auc", "average_precision"]
    x = np.arange(len(fields))
    for i, row in enumerate(result):
        bars = ax.bar(x + (i - 0.5) * 0.35, [row[name] for name in fields], width=0.34,
                      label=row["variant"])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, fields)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Exploratory outer-held-out metric")
    ax.legend()
    ax.set_title("Same RF, features and threshold rule | one pure-negative weighting candidate")
    save(fig, "weighting_comparison")
    fig, ax = plt.subplots(figsize=(11, 3), layout="constrained")
    ax.axis("off")
    labels = ["Training people\nonly", "Grouped mining\nheld-out scores", "Pure negatives\nscore >=0.30\nweight = 2",
              "Refit same RF\ntrain weights only", "Inner-group\nthreshold → test"]
    for i, label in enumerate(labels):
        ax.text(i, 0, label, ha="center", va="center", fontsize=10,
                bbox={"boxstyle": "round,pad=0.6", "facecolor": "#e8f0f7", "edgecolor": "#456"})
        if i < 4:
            ax.annotate("", xy=(i + 0.66, 0), xytext=(i + 0.34, 0),
                        arrowprops={"arrowstyle": "->"})
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.8, 0.8)
    ax.set_title("Hard negatives are mined inside EVERY training split; runtime remains sensor-only")
    save(fig, "training_method")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--baseline-dir", type=Path,
                        default=Path("reports/readiness_20261028_verified"))
    parser.add_argument("--config", type=Path, default=Path("config/a1_hard_negative_v1.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (args.output_dir or Path("reports") / datetime.now().strftime("a1_hard_%Y%m%d_%H%M%S"))
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(root / "src"),
               MPLCONFIGDIR=str(root / "artifacts" / "matplotlib-a1-hard-cache"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    os.environ.setdefault("MPLCONFIGDIR", env["MPLCONFIGDIR"])
    cfg = json.loads(args.config.read_text("utf-8"))
    reference = pd.read_csv(args.baseline_dir / "baseline_oof.csv")
    old_report = json.loads((args.baseline_dir / "baseline_training.json").read_text("utf-8"))
    folds = frozen_folds(old_report, reference)["folds"]
    write_json(output / "protocol.json", {"config": cfg, "outer_folds": folds,
               "raw_sha256": sha(args.raw), "reference_sha256": sha(args.baseline_dir / "baseline_oof.csv"),
               "source_sha256": {str(p.relative_to(root)): sha(p) for p in [
                   root / "scripts/run_a1_hard_negative.py", root / "scripts/run_a1_experiment.py",
                   *sorted((root / "src/fogguard").rglob("*.py"))]}})
    print("Extracting fixed-window RF features ...", flush=True)
    windows = load_fogstar_windows(args.raw, sensor_positions=tuple(cfg["sensor_positions"]),
                                   max_interpolation_gap_s=0)
    validate_window_alignment(windows, reference)
    x, names = a1_matrix(windows, "baseline_2s_v1")
    y = np.array([w.fog_label for w in windows], dtype=int)
    fractions = np.array([w.fog_fraction for w in windows])
    groups = np.array([w.subject_id for w in windows], dtype=str)
    activities = np.array([w.activity_label for w in windows], dtype=object)
    print("Describing first outer TRAINING fold only ...", flush=True)
    mask = np.isin(groups, folds[0]["train"])
    profile = training_profile(x, names, y, fractions, groups, activities, mask, cfg,
                                folds[0]["test"], output)
    result = []
    frames = {}
    for variant in cfg["variants"]:
        directory = output / variant
        directory.mkdir()
        score, threshold = np.full(len(y), np.nan), np.full(len(y), np.nan)
        assigned, seen = np.zeros(len(y), int), np.zeros(len(y), int)
        audits, chosen = [], []
        for i, fold in enumerate(folds, 1):
            print(f"{variant}: outer fold {i}/5; nested mining remains inside training", flush=True)
            train, test = (np.isin(groups, fold[name]) for name in ("train", "test"))
            if (train & test).any() or not (train | test).all():
                raise ValueError("Invalid outer partition")
            cutoff, audit = select_weighted_threshold(x[train], y[train], fractions[train],
                                                       groups[train], variant, cfg,
                                                       outer_test_people=fold["test"])
            model, fit_audit, trace = fit_weighted(x[train], y[train], fractions[train], groups[train],
                                                   variant, cfg, forbidden=fold["test"])
            score[test] = model_scores(model, "rf", x[test])
            threshold[test], assigned[test] = cutoff, i
            seen[test] += 1
            chosen.append(cutoff)
            audits.append({"outer_train": fold["train"], "outer_test": fold["test"],
                           "inner": audit, "outer_training_audit": fit_audit})
            write_json(directory / "fold_audit.json", audits)
            if trace is not None:
                pd.DataFrame({"subject_id": groups[train], "truth_label": y[train],
                              "truth_fraction": fractions[train], "mining_score": trace["score"],
                              "sample_weight": trace["weight"], "hard": trace["hard"]}).to_csv(
                                  directory / f"outer_{i}_training_weights.csv", index=False)
        if not np.all(seen == 1) or not np.isfinite(score).all():
            raise ValueError("Incomplete outer predictions")
        predictions = reference.copy()
        predictions["fog_probability"], predictions["fog_threshold"] = score, threshold
        predictions["is_fog"], predictions["held_out_fold"] = score >= threshold, assigned
        predictions["state"] = np.where(predictions.is_fog, "fog", predictions.direction)
        predictions["model_version"] = f"a1_hard_negative_v1/{variant}"
        predictions["available_at_s"] = predictions.window_end_s
        predictions["prediction_source"] = "subject_held_out_oof"
        predictions["direction_source"] = "unchanged_historical_baseline_oof"
        predictions.to_csv(directory / "oof_predictions.csv", index=False)
        runtime, truth = split_runtime_truth(predictions)
        runtime.to_csv(directory / "a2_runtime.csv", index=False)
        truth.to_csv(directory / "evaluation_truth.csv", index=False)
        report = copy.deepcopy(old_report)
        report["validation"].update(method="Frozen outer groups; inner groups; nested mining",
                                      fold_fog_thresholds=chosen,
                                      deployment_fog_threshold=float(np.median(chosen)))
        report["a1_hard_negative"] = {"variant": variant, "config": cfg}
        write_json(directory / "training_report.json", report)
        print(f"{variant}: final model and independent raw audit ...", flush=True)
        classifier, final_audit, _ = fit_weighted(x, y, fractions, groups, variant, cfg)
        write_json(directory / "deployment_training_audit.json", final_audit)
        final = HardNegativeFoG(classifier, "rf", windows[0].channel_names,
                                 float(np.median(chosen)), training_variant=variant)
        final.save(directory / "model.joblib")
        pd.testing.assert_frame_equal(final.predict_windows(windows[:3]),
                                       MultiModelFoG.load(directory / "model.joblib").predict_windows(
                                           windows[:3]))
        checked = raw_audit(args.raw, directory / "oof_predictions.csv",
                            directory / "training_report.json", directory / "recognition", root, env)
        row = {"variant": variant, **checked["metrics"],
               "average_precision": float(average_precision_score(y, score)),
               **checked["diagnostics"]["event_summary"]}
        result.append(row)
        frames[variant] = predictions
        write_json(output / "comparison.json", result)
        print(f"{variant}: F1={row['fog_f1']:.4f}, P={row['precision']:.4f}, R={row['recall']:.4f}",
              flush=True)
    pd.DataFrame(result).to_csv(output / "comparison.csv", index=False)
    plots(output, profile, result)
    table = pd.DataFrame(result)[["variant", "precision", "recall", "fog_f1", "auc",
                                  "covered_events", "false_alarm_events"]]
    table.to_csv(output / "ppt_comparison.csv", index=False)
    text = ["# A1 训练内困难负样本实验", "", table.to_string(index=False), "",
            "诊断仅使用第一折训练参与者。困难负样本从每次训练内部的参与者留出分数挖掘，",
            "不读取相应验证/测试参与者，不使用活动真值做运行时门控。",
            "只加权窗口内完全没有 FoG 标注的负样本，含少量 FoG 的边界负窗口与真实 FoG 不加负权。",
            "本轮是同一既有队列上的后续探索，不是新的外部验证。默认模型不自动替换。"]
    (output / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    links = "".join(f'<p><a href="{v}/recognition/真实数据冻结识别验收.html">{v}：原始数据核验</a></p>'
                    for v in cfg["variants"])
    images = "".join(f'<img src="figures/{p.name}" alt="{p.stem}" style="width:100%">'
                     for p in sorted((output / "figures").glob("*.png")))
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>A1 困难负样本</title>'
        '<body style="max-width:1100px;margin:32px auto;padding:0 20px;font-family:system-ui;'
        'line-height:1.7"><h1>A1：训练内困难负样本诊断与加权</h1>'
        '<p>活动真值仅作训练诊断，推理仍只读传感器。既有队列探索性结果。</p>'
        + table.to_html(index=False, float_format=lambda n: f"{n:.4f}") + links + images
        + '</body></html>', encoding="utf-8")
    write_json(output / "manifest.json", {"protocol_sha256": sha(output / "protocol.json"),
               "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print(table.to_string(index=False), flush=True)
    print(f"Open {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

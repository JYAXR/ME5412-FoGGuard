"""Run one preregistered A1 feature comparison and export an A2 handoff."""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fogguard.data.fogstar import load_fogstar_windows
from fogguard.evaluation.a1_validation import grouped_threshold, make_forest
from fogguard.evaluation.readiness import frozen_folds, split_runtime_truth, window_metrics
from fogguard.modeling.a1_fog import A1FogModel, a1_matrix


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def validate_window_alignment(windows, reference):
    if len(windows) != len(reference):
        raise ValueError("Window support changed; do not compare different sample sets silently")
    for window, row in zip(windows, reference.itertuples(), strict=True):
        if ((int(window.subject_id), int(window.session_id), int(window.task_id))
                != (int(row.subject_id), int(row.session_id), int(row.task_id))
                or abs(window.start_s - row.window_start_s) > 1e-6
                or abs(window.end_s - row.window_end_s) > 1e-6
                or window.fog_label != row.truth_fog_label):
            raise ValueError("Historical OOF/raw window key or label mismatch")


def raw_audit(raw, predictions, report, output, root, env):
    process = subprocess.run([
        sys.executable, "-X", "utf8", "recognition/evaluate.py", "--raw", str(raw),
        "--predictions", str(predictions), "--training-report", str(report),
        "--output", str(output)], cwd=root, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False)
    (output.parent / "recognition_log.txt").write_text(process.stdout + process.stderr,
                                                      encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"Independent raw-window audit failed: {output}")
    return json.loads((output / "核验与指标.json").read_text(encoding="utf-8"))


def export_plots(output, frames, feature_matrices, schemas, raw_path, cases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir()

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    variants = list(frames)
    labels = ["Historical OOB", "Grouped baseline", "Recent + trend"]
    names = ["precision", "recall", "fog_f1", "auc"]
    metrics = [window_metrics(frames[v]) for v in variants]
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    x = np.arange(len(names))
    for i, (label, row) in enumerate(zip(labels, metrics, strict=True)):
        bars = ax.bar(x + (i - 1) * 0.25, [row[name] for name in names], width=0.24, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Held-out window metric")
    ax.set_title("A1 declared feature experiment | same 9,636 windows / frozen outer subjects")
    ax.legend(ncol=3, loc="upper left", fontsize=9)
    save(fig, "window_comparison")
    raw = pd.read_csv(raw_path)
    candidate = frames["recent_trend_v1"]
    for case in cases:
        key, start, end = case["key"], case["start_s"] - 3, case["end_s"] + 3
        mask = ((candidate[["subject_id", "session_id", "task_id"]].to_numpy(int) == key)
                .all(axis=1) & candidate.window_end_s.between(start, end))
        signal = raw.loc[(raw[["subjectID", "sessionID", "taskID"]].to_numpy(int) == key)
                         .all(axis=1) & raw.timestamp.between(start, end)]
        origin = float(signal.timestamp.iloc[0])
        t = candidate.loc[mask, "window_end_s"] - origin
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, layout="constrained")
        axes[0].plot(signal.timestamp - origin, signal.ankleL_acc_z, lw=0.8,
                     label="Left ankle acceleration z")
        axes[0].set_ylabel("Acceleration (g)")
        for name in ("recent_rms", "rms_recent_minus_previous"):
            index = schemas["recent_trend_v1"].index(f"ankleL_acc_z__{name}")
            axes[1].plot(t, feature_matrices["recent_trend_v1"][mask, index], label=name)
        axes[1].set_ylabel("Dynamic amplitude (g)")
        for variant, label in zip(variants, labels, strict=True):
            frame = frames[variant].loc[mask]
            axes[2].step(t, frame.fog_probability, where="post", label=label)
            axes[3].step(t, frame.is_fog.astype(int), where="post", label=label)
        axes[2].set_ylabel("FoG score")
        axes[2].set_ylim(-0.05, 1.05)
        axes[3].plot(signal.timestamp - origin, signal.fog, color="black", lw=1,
                     label="Raw truth (evaluation only)")
        axes[3].set_ylabel("FoG decision / truth")
        axes[3].set_xlabel("Displayed segment time (s); predictions start at window END")
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend(loc="upper right", fontsize=8, ncol=2)
        fig.suptitle(f"Fixed historical {case['outcome']} case: P{key[0]:02d}/S{key[1]}/T{key[2]}"
                     " | same input for all models")
        save(fig, f"historical_case_{case['outcome'].lower()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--baseline-dir", type=Path,
                        default=Path("reports/readiness_20261028_verified"))
    parser.add_argument("--config", type=Path, default=Path("config/a1_recent_trend_v1.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (args.output_dir or Path("reports") / datetime.now().strftime("a1_%Y%m%d_%H%M%S"))
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh directory to preserve evidence")
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(root / "src"),
               MPLCONFIGDIR=str(root / "artifacts" / "matplotlib-a1-cache"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    os.environ.setdefault("MPLCONFIGDIR", env["MPLCONFIGDIR"])
    config = json.loads(args.config.read_text(encoding="utf-8"))
    baseline_report_path = args.baseline_dir / "baseline_training.json"
    baseline_oof_path = args.baseline_dir / "baseline_oof.csv"
    original_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    historical = pd.read_csv(baseline_oof_path)
    if hashlib.md5(args.raw.read_bytes()).hexdigest() != original_report["dataset"]["md5"]:
        raise ValueError("Raw data hash differs from frozen baseline")
    folds = frozen_folds(original_report, historical)["folds"]
    cases = json.loads((args.baseline_dir / "audit" / "summary.json").read_text(
        encoding="utf-8"))["selected_cases"]
    protocol = {"config": config, "outer_folds": folds, "fixed_plot_cases": cases,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                     text=True).strip(),
                "raw_sha256": sha(args.raw), "historical_oof_sha256": sha(baseline_oof_path),
                "historical_report_sha256": sha(baseline_report_path),
                "config_sha256": sha(args.config),
                "source_sha256": {str(p.relative_to(root)): sha(p) for p in [
                    root / "scripts/run_a1_experiment.py", root / "scripts/predict_a1_fog.py",
                    *sorted((root / "src/fogguard").rglob("*.py"))]}}
    # Written before new test predictions exist; no thresholds or features chosen on them.
    write_json(output / "protocol.json", protocol)
    print("Extracting fixed-window features ...", flush=True)
    windows = load_fogstar_windows(args.raw, sensor_positions=tuple(config["sensor_positions"]),
                                   max_interpolation_gap_s=config["max_interpolation_gap_s"])
    validate_window_alignment(windows, historical)
    y = np.asarray([window.fog_label for window in windows], dtype=int)
    groups = np.asarray([window.subject_id for window in windows], dtype=str)
    matrices, schemas, frames = {}, {}, {"historical_oob": historical}
    evaluation = {}
    for variant in config["variants"]:
        directory = output / variant
        directory.mkdir()
        print(f"Extracting {variant} ...", flush=True)
        x, schema = a1_matrix(windows, variant)
        matrices[variant], schemas[variant] = x, schema
        write_json(directory / "feature_schema.json", list(schema))
        probability = np.full(len(y), np.nan)
        thresholds = np.full(len(y), np.nan)
        assigned = np.zeros(len(y), dtype=int)
        seen = np.zeros(len(y), dtype=int)
        audits, chosen = [], []
        for i, fold in enumerate(folds, 1):
            train_mask, test_mask = (np.isin(groups, fold[name]) for name in ("train", "test"))
            if (train_mask & test_mask).any() or not (train_mask | test_mask).all():
                raise ValueError("Invalid frozen outer fold")
            print(f"{variant}: outer fold {i}/5, grouped inner thresholds ...", flush=True)
            threshold, audit = grouped_threshold(x[train_mask], y[train_mask],
                                                  groups[train_mask], config)
            if set(fold["test"]) & {s for f in audit["folds"]
                                     for s in f["train"] + f["validation"]}:
                raise ValueError("Outer test participant entered inner threshold selection")
            model = make_forest(config).fit(x[train_mask], y[train_mask])
            probability[test_mask] = model.predict_proba(x[test_mask])[:, 1]
            thresholds[test_mask], assigned[test_mask] = threshold, i
            seen[test_mask] += 1
            chosen.append(threshold)
            audits.append({"outer_fold": i, "outer_train": fold["train"],
                           "outer_test": fold["test"], "inner": audit})
            write_json(directory / "fold_audit.json", audits)
        if not np.all(seen == 1) or not np.isfinite(probability).all():
            raise ValueError("Outer OOF did not predict each window exactly once")
        predictions = historical.copy()
        predictions["fog_probability"], predictions["fog_threshold"] = probability, thresholds
        predictions["is_fog"] = probability >= thresholds
        predictions["state"] = np.where(predictions.is_fog, "fog", predictions.direction)
        predictions["held_out_fold"] = assigned
        predictions["available_at_s"] = predictions.window_end_s
        predictions["feature_version"], predictions["model_version"] = variant, "a1_fog_v1"
        predictions["prediction_source"] = "subject_held_out_oof"
        predictions["direction_source"] = "unchanged_historical_baseline_oof"
        predictions.to_csv(directory / "oof_predictions.csv", index=False)
        runtime, truth = split_runtime_truth(predictions)
        runtime.to_csv(directory / "a2_runtime.csv", index=False)
        truth.to_csv(directory / "evaluation_truth.csv", index=False)
        report = copy.deepcopy(original_report)
        report["validation"].update(method="Frozen outer subject CV; grouped-inner thresholds",
                                      fold_fog_thresholds=chosen,
                                      deployment_fog_threshold=float(np.median(chosen)))
        report["a1_experiment"] = {"variant": variant, "features": len(schema),
                                   "config": config, "direction": "unchanged OOF reference"}
        write_json(directory / "training_report.json", report)
        final_forest = make_forest(config).fit(x, y)
        final = A1FogModel(final_forest, variant, windows[0].channel_names, schema,
                           float(np.median(chosen)))
        final.save(directory / "model.joblib")
        # Persistence smoke check uses no labels and is not a held-out performance claim.
        pd.testing.assert_frame_equal(final.predict_windows(windows[:3]),
                                       A1FogModel.load(directory / "model.joblib").predict_windows(
                                           windows[:3]))
        write_json(directory / "feature_importance.json", dict(zip(
            schema, final_forest.feature_importances_.tolist(), strict=True)))
        print(f"{variant}: independent raw/OOF audit ...", flush=True)
        audit = raw_audit(args.raw, directory / "oof_predictions.csv",
                          directory / "training_report.json", directory / "recognition", root, env)
        evaluation[variant] = {"window": audit["metrics"],
                               "events": audit["diagnostics"]["event_summary"],
                               "thresholds": chosen, "features": len(schema)}
        frames[variant] = predictions
    old_audit = json.loads((args.baseline_dir / "recognition" / "核验与指标.json").read_text(
        encoding="utf-8"))
    evaluation["historical_oob"] = {"window": old_audit["metrics"],
                                    "events": old_audit["diagnostics"]["event_summary"],
                                    "features": 132}
    comparison, by_subject = [], []
    for variant, frame in frames.items():
        value = evaluation[variant]
        comparison.append({"variant": variant, "features": value["features"],
                           **value["window"], **value["events"]})
        for subject, group in frame.groupby("subject_id"):
            by_subject.append({"variant": variant, "subject_id": int(subject),
                               "positive_windows": int(group.truth_fog_label.sum()),
                               **window_metrics(group)})
    pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(by_subject).to_csv(output / "subject_comparison.csv", index=False)
    write_json(output / "comparison.json", evaluation)
    export_plots(output, frames, matrices, schemas, args.raw, cases)
    baseline, candidate = (evaluation[name]["window"] for name in config["variants"])
    delta = candidate["fog_f1"] - baseline["fog_f1"]
    text = ["# A1 首轮冻结识别实验", "", "本轮固定规则后只运行一个近期/趋势特征候选。",
            "两组新模型使用相同外层受试者、窗口和分组内层阈值规则；原 OOB 模型仅供参考。", "",
            "| 方案 | 特征 | Precision | Recall | FoG F1 | AUC |", "|---|---:|---:|---:|---:|---:|"]
    for row in comparison:
        text.append(f"| {row['variant']} | {row['features']} | {row['precision']:.4f} | "
                    f"{row['recall']:.4f} | {row['fog_f1']:.4f} | {row['auc']:.4f} |")
    text += ["", f"近期特征相对公平基线的 FoG F1 差值为 {delta:+.4f}。这是一轮实验结果，",
             "不据此自动替换项目默认模型，不以测试集反复选参。", "",
             "原始事件、时间覆盖及漏检见 comparison.csv；完整核验和交互页在各方案 recognition/。",
             "缺测记录仍包括在全部 106 条记录的事件覆盖中，实际模型窗口来自 95 条记录。", "",
             "模型仅输出冻结候选，不识别冻结亚型、不判断真实迈步或治疗效果。", "",
             "## A2 交接", "",
             "各方案 a2_runtime.csv 是留出预测，不含 truth_*；evaluation_truth.csv 单独保存标签。",
             "运行时 CSV 中方向是未修改的旧基线 OOF，A2 应独立实现/验证新的方向任务。",
             "model.joblib 是全量训练模型；predict_a1_fog.py 输出 FoG-only，不生成方向或真值。",
             "OOF 每折阈值只来自该折训练参与者；部署阈值是这些训练阈值的中位数。", "",
             "## 可复现性与限制", "",
             "protocol.json 在预测前记录规则、输入及源码哈希；fold_audit.json 记录内外层参与者。",
             "本次禁用短缺失插值，经逐窗核验与历史 9636 个窗口完全对齐。",
             "本数据实际可插值的短缺失数为零；此修改没有改变本次样本集。",
             "同一项目测试参与者此前已用于探索，因此不能称为全新的外部验证。",
             "图中案例在本轮训练前由历史案例固定，不替代全量及逐人结果。"]
    (output / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    table = pd.DataFrame(comparison)[["variant", "precision", "recall", "fog_f1", "auc",
                                     "covered_events", "truth_events", "time_recall"]]
    links = "".join(f'<p><a href="{v}/recognition/真实数据冻结识别验收.html">{v}：真实数据核验</a></p>'
                    for v in config["variants"])
    images = "".join(f'<figure><img style="width:100%" src="figures/{p.name}">'
                     f'<figcaption>{html.escape(p.stem)}</figcaption></figure>'
                     for p in sorted((output / "figures").glob("*.png")))
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>A1 首轮实验</title>'
        '<body style="max-width:1100px;margin:32px auto;padding:0 20px;font-family:system-ui;'
        'line-height:1.7"><h1>A1：两秒特征与近期趋势特征</h1>'
        '<p>单轮固定实验；三方案共同展示。当前模型未替换共享默认实现。</p>'
        f'{table.to_html(index=False, float_format=lambda n: f"{n:.4f}")}{links}{images}</body></html>',
        encoding="utf-8")
    write_json(output / "manifest.json", {
        "python": platform.python_version(), "protocol_sha256": sha(output / "protocol.json"),
        "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print(table.to_string(index=False), flush=True)
    print(f"Open {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

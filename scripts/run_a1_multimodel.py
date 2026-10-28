"""Preregistered same-sensor RF/SVM/boosting/CNN comparison with nested groups."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from run_a1_experiment import raw_audit, sha, validate_window_alignment, write_json
from sklearn.metrics import average_precision_score

from fogguard.data.fogstar import load_fogstar_windows
from fogguard.evaluation.a1_multimodel import inner_grouped_scores
from fogguard.evaluation.readiness import frozen_folds, split_runtime_truth
from fogguard.modeling.a1_fog import a1_matrix
from fogguard.modeling.a1_multimodel import (
    MultiModelFoG,
    make_classifier,
    model_scores,
    waveform_matrix,
)


def plots(output, frames, curves, comparison):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    figures = output / "figures"
    figures.mkdir()

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    names = ["precision", "recall", "fog_f1", "auc", "average_precision"]
    positions = np.arange(len(names))
    for i, row in enumerate(comparison):
        bars = ax.bar(positions + (i - 1.5) * 0.19, [row[n] for n in names],
                      width=0.18, label=row["family"])
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(positions, names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Held-out window metric")
    ax.set_title("Same sensor/window/outer subjects; thresholds from grouped training F1")
    ax.legend(ncol=4, loc="upper left")
    save(fig, "model_comparison")
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    for family, frame in frames.items():
        precision, recall, _ = precision_recall_curve(frame.truth_fog_label, frame.fog_probability)
        ax.plot(recall, precision, label=family)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Pooled outer-held-out precision-recall curves")
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "precision_recall")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for fold, values in enumerate(curves, 1):
        ax.plot(np.arange(1, len(values) + 1), values, label=f"Outer training {fold}")
    ax.set_xlabel("Fixed training epoch")
    ax.set_ylabel("Weighted training BCE")
    ax.set_title("CNN training loss; not test accuracy / not an early-stopping signal")
    ax.legend(fontsize=8)
    save(fig, "cnn_training_loss")
    # Architecture diagram is a description of the configured 7,473-parameter network.
    fig, ax = plt.subplots(figsize=(11, 3), layout="constrained")
    ax.axis("off")
    boxes = ["12 channels\n120 samples", "Conv16 k9\nReLU + pool2",
             "Conv32 k5\nReLU + pool2", "Conv32 k3\nReLU + avg pool", "Dropout 0.3\nLinear → score"]
    for i, label in enumerate(boxes):
        ax.text(i, 0, label, ha="center", va="center", fontsize=11,
                bbox={"boxstyle": "round,pad=0.7", "facecolor": "#e8f0f7", "edgecolor": "#456"})
        if i < len(boxes) - 1:
            ax.annotate("", xy=(i + 0.65, 0), xytext=(i + 0.35, 0),
                        arrowprops={"arrowstyle": "->"})
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.8, 0.8)
    ax.set_title("A1 small 1D-CNN: trained from scratch; same completed two-second input")
    save(fig, "cnn_architecture")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--baseline-dir", type=Path,
                        default=Path("reports/readiness_20261028_verified"))
    parser.add_argument("--config", type=Path, default=Path("config/a1_multimodel_v1.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (args.output_dir or Path("reports") / datetime.now().strftime("a1_models_%Y%m%d_%H%M%S"))
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(root / "src"),
               MPLCONFIGDIR=str(root / "artifacts" / "matplotlib-a1-models-cache"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    os.environ.setdefault("MPLCONFIGDIR", env["MPLCONFIGDIR"])
    config = json.loads(args.config.read_text(encoding="utf-8"))
    reference = pd.read_csv(args.baseline_dir / "baseline_oof.csv")
    old_report = json.loads((args.baseline_dir / "baseline_training.json").read_text("utf-8"))
    folds = frozen_folds(old_report, reference)["folds"]
    protocol = {"config": config, "outer_folds": folds,
                "raw_sha256": sha(args.raw),
                "reference_sha256": sha(args.baseline_dir / "baseline_oof.csv"),
                "config_sha256": sha(args.config),
                "source_sha256": {str(p.relative_to(root)): sha(p) for p in [
                    root / "scripts/run_a1_multimodel.py", root / "scripts/run_a1_experiment.py",
                    *sorted((root / "src/fogguard").rglob("*.py"))]}}
    write_json(output / "protocol.json", protocol)
    print("Building the same completed-window input for all four models ...", flush=True)
    windows = load_fogstar_windows(args.raw, sensor_positions=tuple(config["sensor_positions"]),
                                   max_interpolation_gap_s=0)
    validate_window_alignment(windows, reference)
    tabular, names = a1_matrix(windows, "baseline_2s_v1")
    waveforms = waveform_matrix(windows)
    labels = np.asarray([w.fog_label for w in windows], dtype=int)
    groups = np.asarray([w.subject_id for w in windows], dtype=str)
    all_results, frames, cnn_curves = [], {}, []
    for family in config["models"]:
        directory = output / family
        directory.mkdir()
        values = waveforms if family == "cnn" else tabular
        scores, thresholds = np.full(len(labels), np.nan), np.full(len(labels), np.nan)
        assigned, seen = np.zeros(len(labels), int), np.zeros(len(labels), int)
        fold_audit, chosen, inner_selections = [], [], []
        start = time.perf_counter()
        for i, fold in enumerate(folds, 1):
            print(f"{family}: outer fold {i}/5, three grouped inner fits + outer fit", flush=True)
            training, testing = (np.isin(groups, fold[n]) for n in ("train", "test"))
            if (training & testing).any() or not (training | testing).all():
                raise ValueError("Invalid outer fold")
            threshold, audit = inner_grouped_scores(values[training], labels[training],
                                                     groups[training], family, config)
            inner_people = {s for row in audit["folds"]
                            for s in row["train"] + row["validation"]}
            if inner_people & set(fold["test"]):
                raise ValueError("Outer person entered inner validation")
            model = make_classifier(family, config).fit(values[training], labels[training])
            scores[testing] = model_scores(model, family, values[testing])
            thresholds[testing], assigned[testing] = threshold, i
            seen[testing] += 1
            chosen.append(threshold)
            inner_selections.append(audit["selected"])
            row = {"outer_train": fold["train"], "outer_test": fold["test"], "inner": audit}
            if family == "cnn":
                row["outer_training_loss"] = model.loss_history_
                row["parameters"] = model.parameter_count_
                cnn_curves.append(model.loss_history_)
            fold_audit.append(row)
            write_json(directory / "fold_audit.json", fold_audit)
        if not np.all(seen == 1) or not np.isfinite(scores).all():
            raise ValueError("Each window must receive exactly one outer prediction")
        fit_time = time.perf_counter() - start
        prediction = reference.copy()
        prediction["fog_probability"], prediction["fog_threshold"] = scores, thresholds
        prediction["is_fog"], prediction["held_out_fold"] = scores >= thresholds, assigned
        prediction["state"] = np.where(prediction.is_fog, "fog", prediction.direction)
        prediction["available_at_s"] = prediction.window_end_s
        prediction["model_version"] = f"a1_multimodel_v1/{family}"
        prediction["prediction_source"] = "subject_held_out_oof"
        prediction["direction_source"] = "unchanged_historical_baseline_oof"
        prediction.to_csv(directory / "oof_predictions.csv", index=False)
        runtime, truth = split_runtime_truth(prediction)
        runtime.to_csv(directory / "a2_runtime.csv", index=False)
        truth.to_csv(directory / "evaluation_truth.csv", index=False)
        report = copy.deepcopy(old_report)
        report["validation"].update(method="Five frozen participant folds; three grouped inner F1 thresholds",
                                      fold_fog_thresholds=chosen,
                                      deployment_fog_threshold=float(np.median(chosen)))
        report["a1_multimodel"] = {"family": family, "config": config,
                                   "same_channels": list(windows[0].channel_names)}
        write_json(directory / "training_report.json", report)
        print(f"{family}: final training, saving model and independent raw audit ...", flush=True)
        final_classifier = make_classifier(family, config).fit(values, labels)
        final = MultiModelFoG(final_classifier, family, windows[0].channel_names,
                              float(np.median(chosen)))
        final.save(directory / "model.joblib")
        pd.testing.assert_frame_equal(final.predict_windows(windows[:3]),
                                       MultiModelFoG.load(directory / "model.joblib").predict_windows(
                                           windows[:3]))
        if family == "cnn":
            write_json(directory / "network.json", {"parameters": final_classifier.parameter_count_,
                                                      "architecture": str(final_classifier.network_),
                                                      "training_loss": final_classifier.loss_history_})
        else:
            write_json(directory / "feature_schema.json", list(names))
        audit = raw_audit(args.raw, directory / "oof_predictions.csv",
                          directory / "training_report.json", directory / "recognition", root, env)
        result = {"family": family, **audit["metrics"],
                  "average_precision": float(average_precision_score(labels, scores)),
                  **audit["diagnostics"]["event_summary"], "nested_training_seconds": fit_time,
                  "thresholds": chosen,
                  "mean_inner_selected_f1": float(np.mean([s["fog_f1"] for s in inner_selections]))}
        all_results.append(result)
        frames[family] = prediction
        write_json(output / "comparison.json", all_results)
        print(f"{family}: OOF F1={result['fog_f1']:.4f}, P={result['precision']:.4f}, "
              f"R={result['recall']:.4f}, AUC={result['auc']:.4f}", flush=True)
    pd.DataFrame(all_results).to_csv(output / "comparison.csv", index=False)
    versions = subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True)
    (output / "environment.txt").write_text(versions, encoding="utf-8")
    plots(output, frames, cnn_curves, all_results)
    compact = pd.DataFrame(all_results)[["family", "precision", "recall", "fog_f1", "auc",
                                        "average_precision", "covered_events", "false_alarm_events"]]
    compact.to_csv(output / "ppt_comparison.csv", index=False)
    text = ["# A1 多模型首轮结果", "", "相同 12 通道、9,636 个窗口和受试者划分。",
            "各模型一个固定配置，统一在训练参与者内部选择以 FoG F1 优先的阈值。", "",
            compact.to_string(index=False), "", "内层阈值规则与上一轮不同，不能把跨实验差异全部归因于模型。",
            "模型不自动替换原项目默认；需考虑逐人、误报和事件覆盖取舍。",
            "各模型 recognition/ 包含逐人/活动分层、全部原始事件和失败案例。",
            "CNN 为单随机种子、固定 20 轮、小型 CPU 网络，不代表所有深度学习方案。",
            "SVM 分数是 sigmoid(margin)，不代表校准概率。",
            "a2_runtime.csv 为留出预测；model.joblib 为全量训练模型，不能混用性能口径。"]
    (output / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    images = "".join(f'<img style="width:100%" src="figures/{p.name}" alt="{p.stem}">'
                     for p in sorted((output / "figures").glob("*.png")))
    links = "".join(f'<p><a href="{f}/recognition/真实数据冻结识别验收.html">{f}：数据与失败审查</a></p>'
                    for f in config["models"])
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>A1 多模型比较</title>'
        '<body style="max-width:1150px;margin:32px auto;padding:0 20px;font-family:system-ui;'
        'line-height:1.7"><h1>A1：RF / SVM / 梯度提升 / 1D-CNN</h1>'
        '<p>同输入、同划分、同阈值目标。单配置比较；不自动替换默认模型。</p>'
        + compact.to_html(index=False, float_format=lambda v: f"{v:.4f}")
        + links + images + '</body></html>', encoding="utf-8")
    write_json(output / "manifest.json", {"protocol_sha256": sha(output / "protocol.json"),
               "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print(compact.to_string(index=False), flush=True)
    print(f"Open {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

"""Nested posture-feature/model search with a sensor-only deployment artifact."""

import argparse
import copy
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from run_a1_experiment import raw_audit, sha, validate_window_alignment, write_json
from sklearn.metrics import average_precision_score

from fogguard.data.fogstar import load_fogstar_windows
from fogguard.evaluation.a1_posture import nested_candidate_scores
from fogguard.evaluation.readiness import frozen_folds, split_runtime_truth
from fogguard.modeling.a1_context import CONTEXT_VARIANT, context_matrix
from fogguard.modeling.a1_fog import A1FogModel
from fogguard.modeling.a1_posture import (
    POSTURE_VARIANT,
    PostureFoG,
    posture_matrix,
    search_classifier,
)


def training_diagnosis(metadata, scores, threshold, people, output):
    frame = metadata.copy()
    if set(frame.subject_id.astype(int).astype(str)) != set(people):
        raise ValueError("Wrong training diagnosis people")
    decision = scores >= threshold
    truth = frame.truth_fog_label.to_numpy(int)
    frame["outcome"] = np.select([(truth == 1) & decision, (truth == 0) & decision,
                                  (truth == 1) & ~decision], ["TP", "FP", "FN"], default="TN")
    frame["label_region"] = np.select([frame.truth_fog_fraction.eq(0), frame.truth_fog_fraction.eq(1)],
                                      ["pure_negative", "pure_fog"], default="boundary_mixed")
    frame["inner_score"] = scores
    frame.to_csv(output / "training_only_diagnosis.csv", index=False)
    rows = []
    for column in ("truth_activity_context", "truth_manifestation", "label_region", "subject_id"):
        counts = frame.groupby([column, "outcome"], dropna=False).size().unstack(fill_value=0)
        for name, row in counts.iterrows():
            rows.append({"grouping": column, "group": str(name),
                         **{n: int(row.get(n, 0)) for n in ("TP", "FP", "TN", "FN")}})
    pd.DataFrame(rows).to_csv(output / "training_error_groups.csv", index=False)


def plot_results(output, results, audits, frames, case):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir()

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    fields = ["precision", "recall", "fog_f1", "auc", "average_precision"]
    x = np.arange(len(fields))
    for i, row in enumerate(results):
        bars = ax.bar(x + (i - 1) * 0.25, [row[n] for n in fields], width=0.24, label=row["variant"])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, fields)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Exploratory participant-held-out metric")
    ax.set_title("Posture/slow-motion summaries and nested training-only model selection")
    ax.legend(fontsize=8)
    save(fig, "model_comparison")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, field, title in zip(axes, ["covered_events", "false_alarm_events"],
                                 ["Raw events covered / 101", "False alarm events"], strict=True):
        bars = ax.bar([r["variant"] for r in results], [r[field] for r in results])
        ax.bar_label(bars, padding=2)
        ax.set_title(title)
        ax.set_ylim(0, max(r[field] for r in results) * 1.2)
        ax.tick_params(axis="x", labelsize=8)
    save(fig, "event_tradeoff")
    ids = [c["candidate_id"] for c in audits[0]["selection"]["candidates"]]
    matrix = np.array([[next(c["selected_threshold"]["fog_f1"] for c in a["selection"]["candidates"]
                            if c["candidate_id"] == name) for a in audits] for name in ids])
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    shown = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1)
    for row in range(len(ids)):
        for col in range(len(audits)):
            ax.text(col, row, f"{matrix[row, col]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if matrix[row, col] > 0.65 else "black")
    ax.set_yticks(range(len(ids)), ids)
    ax.set_xticks(range(len(audits)), [f"Outer {i + 1} TRAIN" for i in range(len(audits))])
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Training-inner F1 only; not outer-test model ranking")
    fig.colorbar(shown, ax=ax, label="Inner F1")
    save(fig, "training_selection")
    diagnostic = pd.read_csv(output / "training_error_groups.csv")
    diagnostic = diagnostic.loc[diagnostic.grouping.eq("truth_activity_context")]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    x = np.arange(len(diagnostic))
    ax.bar(x - 0.2, diagnostic.FP, width=0.4, label="False positives")
    ax.bar(x + 0.2, diagnostic.FN, width=0.4, label="False negatives")
    ax.set_xticks(x, diagnostic.group)
    ax.set_xlabel("Activity truth code, used for diagnosis only")
    ax.set_ylabel("Windows")
    ax.set_title("First outer TRAIN participants; fresh participant-held-out inner scores")
    ax.legend()
    save(fig, "training_error_diagnosis")
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout="constrained")
    for ax, (variant, frame) in zip(axes, frames.items(), strict=True):
        mask = np.ones(len(frame), bool)
        for name, value in zip(["subject_id", "session_id", "task_id"], case, strict=True):
            mask &= frame[name].astype(int).eq(int(value))
        rows = frame.loc[mask]
        t = rows.window_end_s.to_numpy() - rows.window_start_s.iloc[0]
        ax.plot(t, rows.fog_probability, label="Held-out score")
        ax.plot(t, rows.fog_threshold, linestyle="--", label="Training-selected threshold")
        ax.set_title(f"{variant}: {rows.candidate_id.iloc[0]}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Seconds from record start; decisions available at window end")
    fig.suptitle(f"Same predeclared record {case}; raw-event annotations in recognition reports")
    save(fig, "oof_replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("reports/readiness_20261028_verified"))
    parser.add_argument("--context-dir", type=Path, default=Path("reports/a1_context_v1_20261028"))
    parser.add_argument("--config", type=Path, default=Path("config/a1_posture_search_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = Path(__file__).resolve().parents[1], args.output_dir
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(root / "src"),
               MPLCONFIGDIR=str(root / "artifacts/matplotlib-a1-posture-cache"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    os.environ.setdefault("MPLCONFIGDIR", env["MPLCONFIGDIR"])
    cfg = json.loads(args.config.read_text("utf-8"))
    reference = pd.read_csv(args.baseline_dir / "baseline_oof.csv")
    old_report = json.loads((args.baseline_dir / "baseline_training.json").read_text("utf-8"))
    folds = frozen_folds(old_report, reference)["folds"]
    prior = args.context_dir / CONTEXT_VARIANT
    previous = pd.read_csv(prior / "oof_predictions.csv")
    metadata_path = prior / "recognition/全部窗口识别结果.csv"
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[["subject_id", "session_id", "task_id", "window_start_s", "window_end_s",
                         "truth_fog_label", "truth_fog_fraction", "truth_activity_context", "truth_manifestation"]]
    case = json.loads((args.context_dir / "protocol.json").read_text("utf-8"))["replay_record"]
    protocol = {"config": cfg, "outer_folds": folds, "raw_sha256": sha(args.raw),
                "reference_sha256": sha(args.baseline_dir / "baseline_oof.csv"),
                "config_sha256": sha(args.config), "diagnosis_metadata_sha256": sha(metadata_path),
                "replay_record": case, "source_sha256": {str(p.relative_to(root)): sha(p) for p in [
                    root / "scripts/run_a1_posture_search.py", root / "scripts/run_a1_experiment.py",
                    *sorted((root / "src/fogguard").rglob("*.py"))]}}
    write_json(output / "protocol.json", protocol)
    print("Building 278/438 fixed features, no label/participant attributes in model input ...", flush=True)
    windows = load_fogstar_windows(args.raw, sensor_positions=tuple(cfg["sensor_positions"]), max_interpolation_gap_s=0)
    validate_window_alignment(windows, reference)
    validate_window_alignment(windows, metadata)
    context = context_matrix(windows)
    posture = posture_matrix(windows, context=context)
    matrices, schemas = {CONTEXT_VARIANT: context[0], POSTURE_VARIANT: posture[0]}, {
        CONTEXT_VARIANT: context[1], POSTURE_VARIANT: posture[1]}
    labels, groups = np.asarray([w.fog_label for w in windows], int), np.asarray([w.subject_id for w in windows], str)
    candidates = {c["id"]: c for c in cfg["candidates"]}
    outputs, audits = {v: [] for v in cfg["outputs"]}, []
    for i, fold in enumerate(folds, 1):
        train, test = np.isin(groups, fold["train"]), np.isin(groups, fold["test"])
        print(f"Outer {i}/5: six training-inner configurations", flush=True)
        inner, inner_scores = nested_candidate_scores({k: v[train] for k, v in matrices.items()},
            labels[train], groups[train], cfg, forbidden=fold["test"],
            progress=lambda name: print(f"  inner: {name}", flush=True))
        audit = {"outer_train": fold["train"], "outer_test": fold["test"], "selection": inner}
        audits.append(audit)
        write_json(output / "fold_audit.json", audits)
        scores_table = metadata.loc[train].reset_index(drop=True).copy()
        for name, score in inner_scores.items():
            scores_table[f"score__{name}"] = score
        scores_table.to_csv(output / f"outer_{i}_inner_scores.csv", index=False)
        if i == 1:
            threshold = inner["candidates"][0]["selected_threshold"]["threshold"]
            training_diagnosis(metadata.loc[train], inner_scores["context_rf"], threshold, fold["train"], output)
        outer_models = {}
        for variant in cfg["outputs"]:
            name = inner["selected_candidate_id"] if variant == "inner_selected" else variant
            candidate = candidates[name]
            threshold = next(c["selected_threshold"]["threshold"] for c in inner["candidates"] if c["candidate_id"] == name)
            if name not in outer_models:
                outer_models[name] = search_classifier(candidate, cfg["random_state"]).fit(matrices[candidate["features"]][train], labels[train])
            score = outer_models[name].predict_proba(matrices[candidate["features"]][test])[:, 1]
            frame = reference.loc[test].copy()
            frame["fog_probability"], frame["fog_threshold"] = score, threshold
            frame["is_fog"], frame["candidate_id"], frame["held_out_fold"] = score >= threshold, name, i
            frame["feature_version"] = candidate["features"]
            frame["history_2s_available"], frame["history_4s_available"] = context[2][test, 0] >= 0, context[2][test, 1] >= 0
            outputs[variant].append(frame)
            frame.to_csv(output / f"outer_{i}_{variant}_scores.csv", index=False)
        print(f"  selected inside training: {inner['selected_candidate_id']}", flush=True)
    print("Selecting deployment configuration on separate full-cohort inner OOF ...", flush=True)
    deploy, deployment_scores = nested_candidate_scores(matrices, labels, groups, cfg,
        progress=lambda name: print(f"  deployment inner: {name}", flush=True))
    write_json(output / "deployment_selection.json", deploy)
    pd.DataFrame({"subject_id": groups, "truth_fog_label": labels,
                  **{f"score__{k}": v for k, v in deployment_scores.items()}}).to_csv(output / "deployment_inner_scores.csv", index=False)
    frames, results, fitted = {}, [], {}
    for variant, parts in outputs.items():
        directory = output / variant
        directory.mkdir()
        frame = pd.concat(parts).sort_index().reset_index(drop=True)
        frame["state"] = np.where(frame.is_fog, "fog", frame.direction)
        frame["available_at_s"] = frame.window_end_s
        frame["model_version"] = f"a1_posture_search_v1/{variant}"
        frame["prediction_source"] = "subject_held_out_oof"
        frame["direction_source"] = "unchanged_historical_baseline_oof"
        frame.to_csv(directory / "oof_predictions.csv", index=False)
        runtime, truth = split_runtime_truth(frame)
        runtime.to_csv(directory / "a2_runtime.csv", index=False)
        truth.to_csv(directory / "evaluation_truth.csv", index=False)
        name = deploy["selected_candidate_id"] if variant == "inner_selected" else variant
        candidate = candidates[name]
        choice = next(c for c in deploy["candidates"] if c["candidate_id"] == name)
        if name not in fitted:
            fitted[name] = search_classifier(candidate, cfg["random_state"]).fit(matrices[candidate["features"]], labels)
        final = PostureFoG(fitted[name], candidate["features"], windows[0].channel_names,
                            schemas[candidate["features"]], choice["selected_threshold"]["threshold"], candidate_id=name)
        final.save(directory / "model.joblib")
        loaded = A1FogModel.load(directory / "model.joblib")
        stream = loaded.new_stream()
        pd.testing.assert_frame_equal(loaded.predict_windows(windows[:12]),
                                       pd.concat([stream.push(w) for w in windows[:12]], ignore_index=True))
        write_json(directory / "feature_schema.json", list(final.feature_names))
        write_json(directory / "deployment_config.json", {"candidate": candidate, "threshold": final.threshold})
        thresholds = [float(frame.loc[frame.held_out_fold.eq(i), "fog_threshold"].iloc[0]) for i in range(1, 6)]
        report = copy.deepcopy(old_report)
        report["validation"].update(fold_fog_thresholds=thresholds, deployment_fog_threshold=final.threshold,
                                      method="Frozen outer people; inner candidate and threshold selection")
        report["a1_posture_search"] = {"variant": variant, "config": cfg, "deployment_candidate": name}
        write_json(directory / "training_report.json", report)
        checked = raw_audit(args.raw, directory / "oof_predictions.csv", directory / "training_report.json", directory / "recognition", root, env)
        if variant == "context_rf":
            np.testing.assert_allclose(frame.fog_probability, previous.fog_probability, atol=1e-12, rtol=0)
            np.testing.assert_allclose(frame.fog_threshold, previous.fog_threshold, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(frame.is_fog, previous.is_fog)
        row = {"variant": variant, **checked["metrics"],
               "average_precision": float(average_precision_score(labels, frame.fog_probability)),
               **checked["diagnostics"]["event_summary"]}
        results.append(row)
        frames[variant] = frame
        write_json(output / "comparison.json", results)
        print(f"{variant}: F1={row['fog_f1']:.4f}, P={row['precision']:.4f}, R={row['recall']:.4f}, "
              f"events={row['covered_events']}/101, false alarms={row['false_alarm_events']}", flush=True)
    table = pd.DataFrame(results)[["variant", "precision", "recall", "fog_f1", "auc", "average_precision", "covered_events", "false_alarm_events"]]
    table.to_csv(output / "ppt_comparison.csv", index=False)
    (output / "environment.txt").write_text(subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    plot_results(output, results, audits, frames, case)
    links = "".join(f'<p><a href="{v}/recognition/真实数据冻结识别验收.html">{v}: 原始数据审计</a></p>' for v in frames)
    images = "".join(f'<img src="figures/{p.name}" alt="{p.stem}" style="width:100%">' for p in sorted((output / "figures").glob("*.png")))
    (output / "index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>A1 姿态与模型选择</title>'
        '<body style="max-width:1150px;margin:32px auto;padding:20px;font-family:system-ui;line-height:1.7">'
        '<h1>A1：姿态/缓慢转动摘要与训练内模型选择</h1><p>同一旧队列的探索，原默认不替换。'
        '配置与阈值在训练参与者内选择，全部外层结果保留。诊断标签不进入推理。</p>'
        + table.to_html(index=False, float_format=lambda n: f"{n:.4f}") + links + images + '</body></html>', encoding="utf-8")
    write_json(output / "manifest.json", {"protocol_sha256": sha(output / "protocol.json"),
               "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

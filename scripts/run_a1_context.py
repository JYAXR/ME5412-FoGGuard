"""Fixed current-window versus causal six-second context RF experiment."""

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
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from fogguard.data.fogstar import load_fogstar_windows
from fogguard.evaluation.a1_multimodel import inner_grouped_scores
from fogguard.evaluation.readiness import frozen_folds, split_runtime_truth
from fogguard.modeling.a1_context import CONTEXT_VARIANT, ContextFoG, context_matrix
from fogguard.modeling.a1_fog import A1FogModel, a1_matrix
from fogguard.modeling.a1_multimodel import make_classifier, model_scores


def subgroup_metrics(frame, variant):
    rows = []
    for level, names in [("history", np.where(frame.history_4s_available, "full_6s", "startup")),
                          ("participant", frame.subject_id.astype(str))]:
        for name in sorted(set(names)):
            part = frame.loc[np.asarray(names) == name]
            y, decision = part.truth_fog_label, part.is_fog
            rows.append({"variant": variant, "level": level, "group": name, "windows": len(part),
                         "positive_windows": int(y.sum()),
                         "precision": float(precision_score(y, decision, zero_division=0)),
                         "recall": float(recall_score(y, decision, zero_division=0)),
                         "fog_f1": float(f1_score(y, decision, zero_division=0))})
    return rows


def plots(output, results, frames, subgroups, case_key):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = output / "figures"
    directory.mkdir()

    def save(fig, name):
        fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.3), layout="constrained")
    for start, color, label in [(-6, "#c9dced", "Past window\n36 statistics"),
                                 (-4, "#a6c7df", "Past window\n36 statistics"),
                                 (-2, "#ffd39e", "Current window\n132 features")]:
        ax.barh(0, 2, left=start, height=0.8, color=color, edgecolor="white")
        ax.text(start + 1, 0, label, ha="center", va="center", fontsize=11)
    ax.axvline(0, color="black", linestyle="--")
    ax.text(0.2, 0, "Score now\nNo future input", va="center")
    ax.set_xlim(-6.3, 1.8)
    ax.set_ylim(-0.8, 0.9)
    ax.set_yticks([])
    ax.set_xticks([-6, -4, -2, 0])
    ax.set_xlabel("Approximate seconds relative to current decision")
    ax.set_title("278 features: current + lagged summaries + changes + history flags")
    save(fig, "causal_method")
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    fields = ["precision", "recall", "fog_f1", "auc", "average_precision"]
    x = np.arange(len(fields))
    for i, row in enumerate(results):
        bars = ax.bar(x + (i - 0.5) * 0.36, [row[f] for f in fields], width=0.35,
                      label="Current 2s" if i == 0 else "Causal context up to 6s")
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    ax.set_xticks(x, fields)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Exploratory participant-held-out metric")
    ax.legend()
    ax.set_title("Same decisions and grouped threshold rule; one fixed history candidate")
    save(fig, "context_comparison")
    people = subgroups.loc[subgroups.level.eq("participant")].pivot(
        index="group", columns="variant", values="fog_f1")
    people = people.loc[sorted(people.index, key=int)]
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    delta = people[CONTEXT_VARIANT] - people.baseline_2s_v1
    ax.bar(people.index, delta, color=np.where(delta >= 0, "#348a7e", "#ce6845"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Held-out participant")
    ax.set_ylabel("Context F1 minus current-window F1")
    ax.set_title("Participant variation; a zero can include no positive windows")
    save(fig, "participant_change")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    for ax, (variant, frame) in zip(axes, frames.items(), strict=True):
        mask = np.ones(len(frame), dtype=bool)
        for key, value in zip(["subject_id", "session_id", "task_id"], case_key, strict=True):
            mask &= frame[key].astype(int).eq(int(value))
        case = frame.loc[mask]
        t = case.window_end_s.to_numpy() - case.window_start_s.iloc[0]
        ax.plot(t, case.fog_probability, label="OOF score")
        ax.plot(t, case.fog_threshold, linestyle="--", label="Training-selected threshold")
        for end, truth in zip(t, case.truth_fog_label, strict=True):
            if truth:
                ax.axvspan(end, end + 0.5, color="#e45a5a", alpha=0.15, linewidth=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title(variant)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Seconds from record start; pink = current-window label held forward 0.5s")
    fig.suptitle(f"Predeclared first FoG record {case_key}; window truth is not raw event truth")
    save(fig, "oof_replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("reports/readiness_20261028_verified"))
    parser.add_argument("--config", type=Path, default=Path("config/a1_context_v1.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir or Path("reports") / datetime.now().strftime("a1_context_%Y%m%d_%H%M%S")
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(root / "src"),
               MPLCONFIGDIR=str(root / "artifacts/matplotlib-a1-context-cache"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    os.environ.setdefault("MPLCONFIGDIR", env["MPLCONFIGDIR"])
    cfg = json.loads(args.config.read_text("utf-8"))
    if cfg["history_lag_seconds"] != [2, 4] or cfg["history_statistics"] != [
            "rms", "gait_energy", "freeze_energy"] or cfg["variants"] != [
                "baseline_2s_v1", CONTEXT_VARIANT]:
        raise ValueError("Config differs from fixed v1 context implementation")
    reference = pd.read_csv(args.baseline_dir / "baseline_oof.csv")
    old_report = json.loads((args.baseline_dir / "baseline_training.json").read_text("utf-8"))
    folds = frozen_folds(old_report, reference)["folds"]
    protocol = {"config": cfg, "outer_folds": folds, "raw_sha256": sha(args.raw),
                "reference_sha256": sha(args.baseline_dir / "baseline_oof.csv"),
                "config_sha256": sha(args.config),
                "source_sha256": {str(p.relative_to(root)): sha(p) for p in [
                    root / "scripts/run_a1_context.py", root / "scripts/run_a1_experiment.py",
                    *sorted((root / "src/fogguard").rglob("*.py"))]}}
    write_json(output / "protocol.json", protocol)
    print("Building fixed features and history links, before any fitting ...", flush=True)
    windows = load_fogstar_windows(args.raw, sensor_positions=tuple(cfg["sensor_positions"]),
                                   max_interpolation_gap_s=0)
    validate_window_alignment(windows, reference)
    base = a1_matrix(windows, "baseline_2s_v1")
    context, schema, links = context_matrix(windows, base=base)
    labels = np.asarray([w.fog_label for w in windows], dtype=int)
    groups = np.asarray([w.subject_id for w in windows], dtype=str)
    case_i = next(i for i, w in enumerate(windows) if w.fog_label and links[i, 1] >= 0)
    protocol["replay_record"] = [windows[case_i].subject_id, windows[case_i].session_id,
                                  windows[case_i].task_id]
    protocol["replay_selection"] = "First data-order FoG window with full history, selected before fitting"
    write_json(output / "protocol.json", protocol)
    trace = reference[["subject_id", "session_id", "task_id", "window_start_s", "window_end_s"]].copy()
    trace["lag_2s_row"], trace["lag_4s_row"] = links[:, 0], links[:, 1]
    trace.to_csv(output / "history_links.csv", index=False)
    results, frames, subgroups = [], {}, []
    for variant in cfg["variants"]:
        directory = output / variant
        directory.mkdir()
        values, names = base if variant == "baseline_2s_v1" else (context, schema)
        scores, thresholds = np.full(len(labels), np.nan), np.full(len(labels), np.nan)
        assigned, seen = np.zeros(len(labels), int), np.zeros(len(labels), int)
        audits, chosen = [], []
        start = time.perf_counter()
        for i, fold in enumerate(folds, 1):
            print(f"{variant}: outer {i}/5, three inner fits + outer fit", flush=True)
            train, test = (np.isin(groups, fold[n]) for n in ("train", "test"))
            if (train & test).any() or not (train | test).all():
                raise ValueError("Invalid outer fold")
            threshold, audit = inner_grouped_scores(values[train], labels[train], groups[train], "rf", cfg)
            if {s for f in audit["folds"] for s in f["train"] + f["validation"]} & set(fold["test"]):
                raise ValueError("Outer person entered inner fitting")
            classifier = make_classifier("rf", cfg).fit(values[train], labels[train])
            scores[test] = model_scores(classifier, "rf", values[test])
            thresholds[test], assigned[test] = threshold, i
            seen[test] += 1
            chosen.append(threshold)
            audits.append({"outer_train": fold["train"], "outer_test": fold["test"], "inner": audit})
            write_json(directory / "fold_audit.json", audits)
        if not np.all(seen == 1) or not np.isfinite(scores).all():
            raise ValueError("Incomplete OOF")
        prediction = reference.copy()
        prediction["fog_probability"], prediction["fog_threshold"] = scores, thresholds
        prediction["is_fog"], prediction["held_out_fold"] = scores >= thresholds, assigned
        prediction["state"] = np.where(prediction.is_fog, "fog", prediction.direction)
        prediction["available_at_s"] = prediction.window_end_s
        prediction["model_version"] = f"a1_context_v1/{variant}"
        prediction["prediction_source"] = "subject_held_out_oof"
        prediction["direction_source"] = "unchanged_historical_baseline_oof"
        prediction["history_2s_available"], prediction["history_4s_available"] = links[:, 0] >= 0, links[:, 1] >= 0
        prediction.to_csv(directory / "oof_predictions.csv", index=False)
        runtime, truth = split_runtime_truth(prediction)
        runtime.to_csv(directory / "a2_runtime.csv", index=False)
        truth.to_csv(directory / "evaluation_truth.csv", index=False)
        report = copy.deepcopy(old_report)
        report["validation"].update(method="Frozen outer people; grouped-inner F1 threshold; causal context",
                                      fold_fog_thresholds=chosen,
                                      deployment_fog_threshold=float(np.median(chosen)))
        report["a1_context"] = {"variant": variant, "config": cfg, "features": len(names)}
        write_json(directory / "training_report.json", report)
        write_json(directory / "feature_schema.json", list(names))
        print(f"{variant}: final model and independent raw audit ...", flush=True)
        final = ContextFoG(make_classifier("rf", cfg).fit(values, labels), variant,
                            windows[0].channel_names, names, float(np.median(chosen)))
        final.save(directory / "model.joblib")
        loaded = A1FogModel.load(directory / "model.joblib")
        stream = loaded.new_stream()
        streaming = pd.concat([stream.push(w) for w in windows[:12]], ignore_index=True)
        pd.testing.assert_frame_equal(loaded.predict_windows(windows[:12]), streaming)
        checked = raw_audit(args.raw, directory / "oof_predictions.csv", directory / "training_report.json",
                            directory / "recognition", root, env)
        row = {"variant": variant, "features": len(names), **checked["metrics"],
               "average_precision": float(average_precision_score(labels, scores)),
               **checked["diagnostics"]["event_summary"], "training_seconds": time.perf_counter() - start}
        results.append(row)
        frames[variant] = prediction
        subgroups.extend(subgroup_metrics(prediction, variant))
        write_json(output / "comparison.json", results)
        print(f"{variant}: F1={row['fog_f1']:.4f} P={row['precision']:.4f} R={row['recall']:.4f}", flush=True)
    subgroups = pd.DataFrame(subgroups)
    subgroups.to_csv(output / "subgroups.csv", index=False)
    pd.DataFrame(results).to_csv(output / "comparison.csv", index=False)
    versions = subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True)
    (output / "environment.txt").write_text(versions, encoding="utf-8")
    plots(output, results, frames, subgroups, protocol["replay_record"])
    table = pd.DataFrame(results)[["variant", "features", "precision", "recall", "fog_f1", "auc",
                                  "average_precision", "covered_events", "false_alarm_events"]]
    table.to_csv(output / "ppt_comparison.csv", index=False)
    images = "".join(f'<img src="figures/{p.name}" alt="{p.stem}" style="width:100%">'
                     for p in sorted((output / "figures").glob("*.png")))
    links_html = "".join(f'<p><a href="{v}/recognition/真实数据冻结识别验收.html">{v}: 原始数据核验</a></p>'
                          for v in cfg["variants"])
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>A1 因果上下文</title>'
        '<body style="max-width:1100px;margin:32px auto;padding:0 20px;font-family:system-ui;line-height:1.7">'
        '<h1>A1：当前二秒与最长六秒因果上下文</h1><p>同一已分析队列的探索。没有增加判定等待时长。'
        '单候选不自动替换默认；原始事件真值请查看独立核验页面。</p>'
        + table.to_html(index=False, float_format=lambda n: f"{n:.4f}") + links_html + images + '</body></html>',
        encoding="utf-8")
    write_json(output / "manifest.json", {"protocol_sha256": sha(output / "protocol.json"),
               "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print(table.to_string(index=False), flush=True)
    print(f"Open {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

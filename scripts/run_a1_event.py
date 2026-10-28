"""Nested event-objective selection on fixed six-second RF sensor features."""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from run_a1_experiment import raw_audit, sha, validate_window_alignment, write_json
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fogguard.data.fogstar import load_fogstar_windows
from fogguard.evaluation.a1_event import EventSupport, choose_event_policy, grouped_inner_scores
from fogguard.evaluation.readiness import frozen_folds, split_runtime_truth
from fogguard.modeling.a1_context import CONTEXT_VARIANT, ContextFoG, context_matrix
from fogguard.modeling.a1_event import EventPolicy, apply_event_policy
from fogguard.modeling.a1_multimodel import make_classifier, model_scores


def legacy_events(raw, frame, root, output):
    """Cross-check fast objectives with unchanged historical interval diagnostics."""
    spec = importlib.util.spec_from_file_location("old_event_diagnostics", root / "recognition/diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events, alarms = [], []
    for key, signal in raw.groupby(["subjectID", "sessionID", "taskID"], sort=False):
        mask = np.ones(len(frame), bool)
        for name, value in zip(["subject_id", "session_id", "task_id"], key, strict=True):
            mask &= frame[name].astype(int).eq(int(value)).to_numpy()
        selected = frame.loc[mask].copy()
        selected["is_fog"] = selected.event_fog
        rows, alarm_rows, _ = module.event_rows(signal, selected)
        ids = dict(zip(["subject_id", "session_id", "task_id"], map(int, key), strict=True))
        events.extend({**ids, **r} for r in rows)
        alarms.extend({**ids, **r} for r in alarm_rows)
    pd.DataFrame(events).to_csv(output / "event_coverage.csv", index=False)
    pd.DataFrame(alarms).to_csv(output / "event_alarms.csv", index=False)
    delays = [r["onset_delay_s"] for r in events if r["covered"] and not r["alarm_already_active_at_onset"]]
    return {"covered_raw_events": sum(r["covered"] for r in events), "truth_events": len(events),
            "false_alarm_events": sum(not r["overlaps_truth"] for r in alarms),
            "supported_events": sum(r["model_support_s"] > 1e-6 for r in events),
            "true_alarm_s": sum(r["alarm_overlap_s"] for r in events),
            "false_alarm_s": sum(r["duration_s"] for r in alarms) - sum(r["alarm_overlap_s"] for r in events),
            "new_alarm_delay_cases": len(delays),
            "median_new_alarm_delay_s": float(np.median(delays)) if delays else None}


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

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, field, title in zip(axes, ["covered_raw_events", "false_alarm_events", "false_alarm_s"],
                                 ["Raw events covered / 101", "False alarm events", "False alarm seconds"], strict=True):
        bars = ax.bar(["Window F1", "Event objective"], [r[field] for r in results], color=["#4988b9", "#d1873a"])
        ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.set_title(title)
        ax.set_ylim(0, max(r[field] for r in results) * 1.2)
    fig.suptitle("Outer-held-out event tradeoff; 110% budget applies to TRAINING only")
    save(fig, "event_tradeoff")
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    audit = audits[0]["selection"]
    for confirmation in (1, 2):
        options = [r for r in audit["options"] if r["confirmation_windows"] == confirmation]
        ax.scatter([r["false_alarm_s"] for r in options], [r["covered_raw_events"] for r in options],
                   label=f"{confirmation} consecutive", s=16, alpha=0.6)
    ax.axvline(audit["false_alarm_budget_s"], linestyle="--", color="black", label="Training FP-time budget")
    best = audit["selected"]
    ax.scatter([best["false_alarm_s"]], [best["covered_raw_events"]], s=160, marker="*", color="red", label="Selected")
    ax.set_xlabel("False alarm seconds (first outer TRAIN people only)")
    ax.set_ylabel("Raw training events with any overlap")
    ax.set_title("182 training-only rules; time-coverage constraint also required")
    ax.legend(fontsize=8)
    save(fig, "training_selection")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    for ax, (variant, frame) in zip(axes, frames.items(), strict=True):
        mask = np.ones(len(frame), bool)
        for name, value in zip(["subject_id", "session_id", "task_id"], case, strict=True):
            mask &= frame[name].astype(int).eq(int(value)).to_numpy()
        rows = frame.loc[mask]
        t = rows.window_end_s.to_numpy() - rows.window_start_s.iloc[0]
        ax.plot(t, rows.fog_probability, label="Same held-out sensor score")
        ax.plot(t, rows.fog_threshold, linestyle="--", label="Training threshold")
        ax.step(t, rows.event_fog.astype(int) * 0.95, where="post", alpha=0.5, label="Causal event decision")
        ax.set_ylim(0, 1)
        ax.set_title(f"{variant}, confirmation={rows.event_confirmation_windows.iloc[0]}")
        ax.set_ylabel("Score / decision")
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Seconds from record start; outputs available at window end")
    fig.suptitle(f"Same record selected before prior context experiment: {case}")
    save(fig, "policy_replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("reports/readiness_20261028_verified"))
    parser.add_argument("--context-dir", type=Path, default=Path("reports/a1_context_v1_20261028"))
    parser.add_argument("--config", type=Path, default=Path("config/a1_event_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-scored-run", type=Path,
                        help="Complete a new report from frozen scores after a post-fit reporting failure")
    args = parser.parse_args()
    root, output = Path(__file__).resolve().parents[1], args.output_dir
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(root / "src"),
               MPLCONFIGDIR=str(root / "artifacts/matplotlib-a1-event-cache"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    os.environ.setdefault("MPLCONFIGDIR", env["MPLCONFIGDIR"])
    cfg = json.loads(args.config.read_text("utf-8"))
    if cfg["feature_variant"] != CONTEXT_VARIANT or cfg["confirmation_choices"] != [1, 2]:
        raise ValueError("Config differs from fixed v1 implementation")
    reference = pd.read_csv(args.baseline_dir / "baseline_oof.csv")
    old_report = json.loads((args.baseline_dir / "baseline_training.json").read_text("utf-8"))
    folds = frozen_folds(old_report, reference)["folds"]
    previous = pd.read_csv(args.context_dir / CONTEXT_VARIANT / "oof_predictions.csv")
    context_protocol = json.loads((args.context_dir / "protocol.json").read_text("utf-8"))
    protocol = {"config": cfg, "outer_folds": folds, "raw_sha256": sha(args.raw),
                "previous_oof_sha256": sha(args.context_dir / CONTEXT_VARIANT / "oof_predictions.csv"),
                "config_sha256": sha(args.config), "replay_record": context_protocol["replay_record"],
                "source_sha256": {str(p.relative_to(root)): sha(p) for p in [
                    root / "scripts/run_a1_event.py", root / "scripts/run_a1_experiment.py",
                    root / "recognition/diagnostics.py", *sorted((root / "src/fogguard").rglob("*.py"))]}}
    reused = None
    if args.reuse_scored_run:
        source = args.reuse_scored_run
        old_protocol = json.loads((source / "protocol.json").read_text("utf-8"))
        if (old_protocol["config"] != cfg or old_protocol["outer_folds"] != folds
                or old_protocol["raw_sha256"] != protocol["raw_sha256"]):
            raise ValueError("Frozen scoring input/config/folds differ")
        for path, digest in old_protocol["source_sha256"].items():
            if path != str(Path("scripts/run_a1_event.py")) and sha(root / path) != digest:
                raise ValueError(f"Scoring dependency changed: {path}")
        saved = pd.read_csv(source / "window_f1/oof_predictions.csv")
        pd.testing.assert_frame_equal(saved[["subject_id", "session_id", "task_id", "window_start_s",
                                             "window_end_s", "truth_fog_label"]],
                                       reference[["subject_id", "session_id", "task_id", "window_start_s",
                                                  "window_end_s", "truth_fog_label"]])
        reused = {"audits": json.loads((source / "fold_audit.json").read_text("utf-8")), "scores": saved,
                  "deployment": json.loads((source / "deployment_selection.json").read_text("utf-8"))}
        if len(reused["audits"]) != len(folds):
            raise ValueError("Cannot reuse incomplete scoring")
        protocol["training_score_source_protocol_sha256"] = sha(source / "protocol.json")
        protocol["report_recovery"] = "Reuse fixed completed scoring; only CSV threshold equality check changed to 1e-12 tolerance"
        write_json(output / "original_scoring_protocol.json", old_protocol)
    write_json(output / "protocol.json", protocol)
    print("Building frozen six-second features and raw truth ...", flush=True)
    windows = load_fogstar_windows(args.raw, sensor_positions=tuple(cfg["sensor_positions"]), max_interpolation_gap_s=0)
    validate_window_alignment(windows, reference)
    values, names, links = context_matrix(windows)
    raw = pd.read_csv(args.raw, usecols=["subjectID", "sessionID", "taskID", "timestamp", "fog"])
    groups = np.asarray([w.subject_id for w in windows], dtype=str)
    labels = np.asarray([w.fog_label for w in windows], dtype=int)
    outputs = {v: [] for v in ("window_f1", "event_objective")}
    audits = []
    for i, fold in enumerate(folds, 1):
        print(f"Outer {i}/5: inner scoring, training-only event selection, outer fit", flush=True)
        train, test = np.isin(groups, fold["train"]), np.isin(groups, fold["test"])
        if reused is None:
            inner_scores, inner_folds = grouped_inner_scores(values[train], labels[train], groups[train], cfg,
                                                             forbidden=fold["test"])
        else:
            saved_inner = pd.read_csv(args.reuse_scored_run / f"outer_{i}_inner_scores.csv")
            inner_scores = saved_inner.fog_probability.to_numpy(float)
            inner_folds = reused["audits"][i - 1]["inner_folds"]
        score_columns = ["subject_id", "session_id", "task_id", "window_start_s", "window_end_s",
                         "truth_fog_label", "truth_fog_fraction"]
        inner_frame = reference.loc[train, score_columns].reset_index(drop=True).copy()
        inner_frame["fog_probability"] = inner_scores
        inner_raw = raw.loc[raw.subjectID.astype(int).astype(str).isin(fold["train"])]
        support = EventSupport.build(inner_raw, inner_frame, allowed=fold["train"], forbidden=fold["test"])
        if reused is None:
            selection = choose_event_policy(inner_frame, support, cfg)
        else:
            pd.testing.assert_frame_equal(saved_inner[score_columns], inner_frame[score_columns])
            selection = reused["audits"][i - 1]["selection"]
        inner_frame.to_csv(output / f"outer_{i}_inner_scores.csv", index=False)
        audits.append({"outer_train": fold["train"], "outer_test": fold["test"],
                       "inner_folds": inner_folds, "selection": selection})
        write_json(output / "fold_audit.json", audits)
        scores = (model_scores(make_classifier("rf", cfg).fit(values[train], labels[train]), "rf", values[test])
                  if reused is None else reused["scores"].loc[test, "fog_probability"].to_numpy(float))
        for variant, choice in [("window_f1", selection["control"]), ("event_objective", selection["selected"])]:
            frame = reference.loc[test].copy()
            frame["fog_probability"] = scores
            frame["held_out_fold"] = i
            frame["history_2s_available"], frame["history_4s_available"] = links[test, 0] >= 0, links[test, 1] >= 0
            outputs[variant].append(apply_event_policy(frame, EventPolicy(choice["threshold"], choice["confirmation_windows"])))
        print(f"Selected {selection['selected']['threshold']:.2f}, confirmation "
              f"{selection['selected']['confirmation_windows']}; TRAIN covered "
              f"{selection['control']['covered_raw_events']}→{selection['selected']['covered_raw_events']}", flush=True)
    print("Separate full-cohort grouped-inner deployment policy selection ...", flush=True)
    if reused is None:
        deploy_scores, deploy_folds = grouped_inner_scores(values, labels, groups, cfg)
    else:
        deploy_scores = pd.read_csv(args.reuse_scored_run / "deployment_inner_scores.csv").fog_probability.to_numpy(float)
        deploy_folds = reused["deployment"]["inner_folds"]
    deploy_frame = reference[score_columns].copy()
    deploy_frame["fog_probability"] = deploy_scores
    full_support = EventSupport.build(raw, reference, allowed=sorted(set(groups)))
    deploy = (choose_event_policy(deploy_frame, full_support, cfg) if reused is None
              else reused["deployment"]["selection"])
    write_json(output / "deployment_selection.json", {"inner_folds": deploy_folds, "selection": deploy})
    deploy_frame.to_csv(output / "deployment_inner_scores.csv", index=False)
    if reused is None:
        classifier = make_classifier("rf", cfg).fit(values, labels)
    else:
        classifier = ContextFoG.load(args.reuse_scored_run / "window_f1/model.joblib").classifier
    results, frames = [], {}
    for variant, parts in outputs.items():
        directory = output / variant
        directory.mkdir()
        frame = pd.concat(parts).sort_index().reset_index(drop=True)
        frame["available_at_s"] = frame.window_end_s
        frame["state"] = np.where(frame.is_fog, "fog", frame.direction)
        frame["model_version"] = f"a1_event_v1/{variant}"
        frame["prediction_source"] = "subject_held_out_oof"
        frame["direction_source"] = "unchanged_historical_baseline_oof"
        frame.to_csv(directory / "oof_predictions.csv", index=False)
        runtime, truth = split_runtime_truth(frame)
        runtime.to_csv(directory / "a2_runtime.csv", index=False)
        truth.to_csv(directory / "evaluation_truth.csv", index=False)
        choice = deploy["control"] if variant == "window_f1" else deploy["selected"]
        policy = {"threshold": choice["threshold"], "confirmation_windows": choice["confirmation_windows"],
                  "version": "a1_event_v1", "scope": "Full-cohort inner selection; not external validation"}
        write_json(directory / "event_policy.json", policy)
        ContextFoG(classifier, CONTEXT_VARIANT, windows[0].channel_names, names, policy["threshold"]).save(directory / "model.joblib")
        report = copy.deepcopy(old_report)
        thresholds = [(a["selection"]["control"] if variant == "window_f1" else a["selection"]["selected"])["threshold"] for a in audits]
        report["validation"].update(fold_fog_thresholds=thresholds, deployment_fog_threshold=policy["threshold"],
                                      method="Frozen outer people; training-inner raw-event policy selection")
        report["a1_event"] = {"variant": variant, "config": cfg, "event_column": "event_fog"}
        write_json(directory / "training_report.json", report)
        checked = raw_audit(args.raw, directory / "oof_predictions.csv", directory / "training_report.json",
                            directory / "recognition", root, env)
        independent = legacy_events(raw, frame, root, directory)
        event_metrics = full_support.metrics(frame.event_fog)
        for name in ["covered_raw_events", "truth_events", "supported_events", "true_alarm_s", "false_alarm_s"]:
            if abs(independent[name] - event_metrics[name]) > 1e-6:
                raise ValueError(f"Historical interval audit mismatch: {name}")
        row = {"variant": variant, **event_metrics, "false_alarm_events": independent["false_alarm_events"],
               "median_new_alarm_delay_s": independent["median_new_alarm_delay_s"],
               "new_alarm_delay_cases": independent["new_alarm_delay_cases"],
               "precision": float(precision_score(labels, frame.event_fog, zero_division=0)),
               "recall": float(recall_score(labels, frame.event_fog, zero_division=0)),
               "fog_f1": float(f1_score(labels, frame.event_fog, zero_division=0)),
               "auc_of_same_raw_scores": float(roc_auc_score(labels, frame.fog_probability)),
               "ap_of_same_raw_scores": float(average_precision_score(labels, frame.fog_probability)),
               "raw_threshold_window_metrics": checked["metrics"]}
        if variant == "window_f1":
            for col in ["is_fog", "held_out_fold"]:
                np.testing.assert_array_equal(frame[col], previous[col])
            np.testing.assert_allclose(frame.fog_threshold, previous.fog_threshold, atol=1e-12, rtol=0)
            np.testing.assert_allclose(frame.fog_probability, previous.fog_probability, atol=1e-12, rtol=0)
        results.append(row)
        frames[variant] = frame
        write_json(output / "comparison.json", results)
        print(f"{variant}: events {row['covered_raw_events']}/{row['truth_events']}, "
              f"false events {row['false_alarm_events']}, F1 {row['fog_f1']:.4f}", flush=True)
    table = pd.DataFrame(results).drop(columns="raw_threshold_window_metrics")
    table.to_csv(output / "ppt_comparison.csv", index=False)
    plot_results(output, results, audits, frames, protocol["replay_record"])
    images = "".join(f'<img src="figures/{p.name}" alt="{p.stem}" style="width:100%">' for p in sorted((output / "figures").glob("*.png")))
    (output / "index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>A1 事件目标</title>'
        '<body style="max-width:1100px;margin:32px auto;padding:20px;font-family:system-ui;line-height:1.7">'
        '<h1>A1：训练内事件目标与确认规则</h1><p>event_fog 为可选实验判定；is_fog 保留原始阈值含义。'
        '预算仅约束训练内选择，不保证测试预算。相同旧队列探索，默认模型不替换。</p>'
        + table.to_html(index=False, float_format=lambda n: f"{n:.4f}") + images + '</body></html>', encoding="utf-8")
    write_json(output / "manifest.json", {"protocol_sha256": sha(output / "protocol.json"),
               "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

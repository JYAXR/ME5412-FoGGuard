"""Subject-held-out left-ankle fallback with all original primary decisions kept."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from run_a1_event import legacy_events
from run_a1_experiment import sha, write_json
from sklearn.metrics import f1_score, precision_score, recall_score

from fogguard.data.fogstar import load_fogstar_windows
from fogguard.evaluation.a1_event import EventSupport
from fogguard.evaluation.a1_multimodel import inner_grouped_scores
from fogguard.modeling.a1_fallback import (
    AnkleFallbackFoG,
    ankle_matrix,
    route_available_predictions,
)
from fogguard.modeling.a1_fog import A1FogModel
from fogguard.modeling.a1_multimodel import make_classifier


def validate_mixed_windows(raw, frame):
    """Reconstruct each row's actual channel set and 120 raw samples."""
    records = {tuple(map(int, key)): value for key, value in raw.groupby(["subjectID", "sessionID", "taskID"])}
    for row in frame.itertuples():
        signal = records[(int(row.subject_id), int(row.session_id), int(row.task_id))]
        time = signal.timestamp.to_numpy(float)
        a = np.searchsorted(time, row.window_start_s - 1e-6)
        b = np.searchsorted(time, row.window_end_s + 1e-6, side="right")
        chunk = signal.iloc[a:b]
        positions = ("ankleL", "back") if row.sensor_route == "primary_ankle_back" else ("ankleL",)
        columns = [f"{p}_{s}_{axis}" for p in positions for s in ("acc", "gyro") for axis in "xyz"]
        if (len(chunk) != 120 or not np.isfinite(chunk[columns].to_numpy(float)).all()
                or abs(chunk.fog.mean() - row.truth_fog_fraction) > 1e-10
                or int(chunk.fog.mean() >= 0.5) != row.truth_fog_label):
            raise ValueError("Mixed-input raw window mismatch")
    return len(frame)


def plot_results(output, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / "figures"
    folder.mkdir()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, field, title in zip(axes, ["covered_raw_events", "false_alarm_events", "time_recall"],
                                 ["Raw events covered / 101", "False alarm events", "FoG time recall"], strict=True):
        bars = ax.bar(["Primary only", "Availability fallback"], [r[field] for r in metrics])
        ax.bar_label(bars, fmt="%.3f" if field == "time_recall" else "%.0f", padding=3)
        ax.set_title(title)
        ax.set_ylim(0, max(r[field] for r in metrics) * 1.2)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Same 106 raw records; primary decisions unchanged; no truth-based switching")
    fig.savefig(folder / "fallback_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(folder / "fallback_comparison.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--primary-dir", type=Path, default=Path("reports/a1_context_v1_20261028/causal_context_6s_v1"))
    parser.add_argument("--config", type=Path, default=Path("config/a1_ankle_fallback_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = Path(__file__).resolve().parents[1], args.output_dir
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(root / "artifacts/matplotlib-a1-fallback-cache"))
    cfg = json.loads(args.config.read_text("utf-8"))
    prior_protocol = json.loads((args.primary_dir.parent / "protocol.json").read_text("utf-8"))
    folds = prior_protocol["outer_folds"]
    primary = pd.read_csv(args.primary_dir / "oof_predictions.csv", float_precision="round_trip")
    protocol = {"config": cfg, "outer_folds": folds, "raw_sha256": sha(args.raw),
                "config_sha256": sha(args.config), "primary_oof_sha256": sha(args.primary_dir / "oof_predictions.csv"),
                "source_sha256": {str(p.relative_to(root)): sha(p) for p in [root / "scripts/run_a1_fallback.py",
                    root / "scripts/run_a1_event.py", root / "recognition/diagnostics.py",
                    *sorted((root / "src/fogguard").rglob("*.py"))]}}
    write_json(output / "protocol.json", protocol)
    print("Building left-ankle-only valid windows ...", flush=True)
    windows = load_fogstar_windows(args.raw, sensor_positions=("ankleL",), max_interpolation_gap_s=0)
    values, names, links = ankle_matrix(windows)
    y, groups = np.asarray([w.fog_label for w in windows], int), np.asarray([w.subject_id for w in windows], str)
    score, threshold, assignment = np.full(len(y), np.nan), np.full(len(y), np.nan), np.zeros(len(y), int)
    audits = []
    for i, fold in enumerate(folds, 1):
        print(f"Ankle fallback: outer {i}/5, training-inner threshold", flush=True)
        train, test = np.isin(groups, fold["train"]), np.isin(groups, fold["test"])
        cutoff, audit = inner_grouped_scores(values[train], y[train], groups[train], "rf", cfg)
        if {p for f in audit["folds"] for p in f["train"] + f["validation"]} & set(fold["test"]):
            raise ValueError("Fallback test participant leakage")
        model = make_classifier("rf", cfg).fit(values[train], y[train])
        score[test] = model.predict_proba(values[test])[:, 1]
        threshold[test], assignment[test] = cutoff, i
        audits.append({"outer_train": fold["train"], "outer_test": fold["test"], "inner": audit})
        write_json(output / "fold_audit.json", audits)
    if not np.isfinite(score).all() or not np.all(assignment > 0):
        raise ValueError("Incomplete fallback OOF")
    ankle = pd.DataFrame({"subject_id": [int(w.subject_id) for w in windows],
        "session_id": [int(w.session_id) for w in windows], "task_id": [int(w.task_id) for w in windows],
        "window_start_s": [w.start_s for w in windows], "window_end_s": [w.end_s for w in windows],
        "available_at_s": [w.end_s for w in windows], "fog_probability": score, "fog_threshold": threshold,
        "is_fog": score >= threshold, "truth_fog_label": y, "truth_fog_fraction": [w.fog_fraction for w in windows],
        "held_out_fold": assignment, "model_version": "a1_ankle_fallback_v1",
        "prediction_source": "subject_held_out_oof", "history_2s_available": links[:, 0] >= 0,
        "history_4s_available": links[:, 1] >= 0})
    ankle.to_csv(output / "ankle_oof_predictions.csv", index=False)
    combined = route_available_predictions(primary, ankle)
    primary = primary.assign(sensor_route="primary_ankle_back")
    keys = ["subject_id", "session_id", "task_id", "window_start_s", "window_end_s"]
    kept = combined.loc[combined.sensor_route.eq("primary_ankle_back")].sort_values(keys)
    original = primary.sort_values(keys)
    pd.testing.assert_frame_equal(kept[primary.columns].reset_index(drop=True), original.reset_index(drop=True), check_dtype=False)
    print("Separate full-cohort inner threshold and final ankle fit ...", flush=True)
    cutoff, deployment = inner_grouped_scores(values, y, groups, "rf", cfg)
    write_json(output / "deployment_selection.json", deployment)
    final = AnkleFallbackFoG(make_classifier("rf", cfg).fit(values, y), "ankle_context_v1",
                              windows[0].channel_names, names, cutoff)
    final.save(output / "ankle_model.joblib")
    A1FogModel.load(args.primary_dir / "model.joblib").save(output / "primary_model.joblib")
    raw = pd.read_csv(args.raw)
    results = []
    for name, frame in [("primary_only", primary), ("availability_fallback", combined)]:
        directory = output / name
        directory.mkdir()
        validated = validate_mixed_windows(raw, frame)
        frame.to_csv(directory / "oof_predictions.csv", index=False)
        frame.drop(columns=[c for c in frame if c.startswith("truth_")]).to_csv(directory / "a2_runtime.csv", index=False)
        frame[[*keys, *[c for c in frame if c.startswith("truth_")]]].to_csv(directory / "evaluation_truth.csv", index=False)
        support = EventSupport.build(raw, frame, allowed=sorted(set(groups)))
        checked = legacy_events(raw, frame.assign(event_fog=frame.is_fog), root, directory)
        metrics = support.metrics(frame.is_fog)
        for field in ["covered_raw_events", "truth_events", "supported_events", "true_alarm_s", "false_alarm_s"]:
            if abs(metrics[field] - checked[field]) > 1e-6:
                raise ValueError(f"Fallback interval audit mismatch: {field}")
        results.append({"variant": name, "validated_windows": validated,
            "records_with_predictions": len(frame[keys[:3]].drop_duplicates()),
            "fallback_rows": int(frame.sensor_route.eq("fallback_ankle").sum()), **metrics,
            "false_alarm_events": checked["false_alarm_events"],
            "median_new_alarm_delay_s": checked["median_new_alarm_delay_s"],
            "new_alarm_delay_cases": checked["new_alarm_delay_cases"]})
    write_json(output / "comparison.json", results)
    write_json(output / "ankle_window_metrics.json", {"windows": len(ankle), "precision": float(precision_score(y, ankle.is_fog)),
        "recall": float(recall_score(y, ankle.is_fog)), "fog_f1": float(f1_score(y, ankle.is_fog)),
        "scope": "Different window support; not a same-window comparison with twelve-channel primary"})
    table = pd.DataFrame(results)
    table.to_csv(output / "ppt_comparison.csv", index=False)
    plot_results(output, results)
    (output / "index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>A1 左踝备用</title>'
        '<body style="max-width:1150px;margin:32px auto;padding:20px;font-family:system-ui;line-height:1.7">'
        '<h1>A1：输入缺失时的左踝备用检测</h1><p>相同 106 条原始记录；主模型评分和决定保留。'
        '新增支持不等于准确识别；窗口池不同，主要比较原始事件与时间指标。</p>'
        + table.to_html(index=False, float_format=lambda n: f"{n:.4f}")
        + '<img src="figures/fallback_comparison.png" style="width:100%" alt="Fallback comparison"></body></html>', encoding="utf-8")
    write_json(output / "manifest.json", {"protocol_sha256": sha(output / "protocol.json"),
        "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

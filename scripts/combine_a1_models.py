"""Exploratory composition of frozen nested-selected primary and ankle fallback.

No fitting, threshold search or per-record outcome-based model choice occurs.
Both components used the identical participant-held-out outer partitions.
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from run_a1_event import legacy_events
from run_a1_experiment import sha, write_json
from run_a1_fallback import validate_mixed_windows

from fogguard.evaluation.a1_event import EventSupport
from fogguard.modeling.a1_fallback import route_available_predictions


def plot_joint(output, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / "figures"
    folder.mkdir()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, field, title in zip(axes, ["covered_raw_events", "false_alarm_events", "time_recall"],
                                 ["Raw events covered / 101", "False alarm events", "FoG time recall"], strict=True):
        bars = ax.bar(["Frozen context RF", "Selected + fallback"], [r[field] for r in rows])
        ax.bar_label(bars, fmt="%.3f" if field == "time_recall" else "%.0f", padding=3)
        ax.set_title(title)
        ax.set_ylim(0, max(r[field] for r in rows) * 1.2)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Same 106 records; exploratory composition of two frozen subject-held-out components")
    fig.savefig(folder / "joint_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(folder / "joint_comparison.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--primary-dir", type=Path, default=Path("reports/a1_posture_search_v1_20261029/inner_selected"))
    parser.add_argument("--fallback-dir", type=Path, default=Path("reports/a1_ankle_fallback_v1_20261029_verified"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("reports/a1_context_v1_20261028/causal_context_6s_v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = Path(__file__).resolve().parents[1], args.output_dir
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    a = json.loads((args.primary_dir.parent / "protocol.json").read_text("utf-8"))
    b = json.loads((args.fallback_dir / "protocol.json").read_text("utf-8"))
    if a["outer_folds"] != b["outer_folds"] or a["raw_sha256"] != b["raw_sha256"] or sha(args.raw) != a["raw_sha256"]:
        raise ValueError("Components use different raw data or outer people")
    protocol = {"experiment": "a1_joint_v1", "outer_folds": a["outer_folds"], "raw_sha256": a["raw_sha256"],
        "config": {"primary": "Frozen training-inner selected pipeline", "fallback": b["config"],
                   "composition": "Availability only; same routing as fallback experiment"},
        "selection_limitation": "Composition proposed after seeing component experiments on same cohort; exploratory, not fresh validation",
        "component_manifest_sha256": {"primary": sha(args.primary_dir.parent / "manifest.json"),
                                      "fallback": sha(args.fallback_dir / "manifest.json")},
        "source_sha256": {str(p.relative_to(root)): sha(p) for p in [root / "scripts/combine_a1_models.py",
            root / "scripts/run_a1_fallback.py", root / "scripts/run_a1_event.py", root / "recognition/diagnostics.py",
            *sorted((root / "src/fogguard").rglob("*.py"))]}}
    write_json(output / "protocol.json", protocol)
    for name in ("fold_audit.json", "deployment_selection.json", "ankle_oof_predictions.csv", "ankle_model.joblib"):
        shutil.copyfile(args.fallback_dir / name, output / name)
    shutil.copyfile(args.primary_dir / "model.joblib", output / "primary_model.joblib")
    primary = pd.read_csv(args.primary_dir / "oof_predictions.csv", float_precision="round_trip")
    ankle = pd.read_csv(args.fallback_dir / "ankle_oof_predictions.csv", float_precision="round_trip")
    baseline = pd.read_csv(args.baseline_dir / "oof_predictions.csv", float_precision="round_trip").assign(sensor_route="primary_ankle_back")
    combined = route_available_predictions(primary, ankle)
    raw = pd.read_csv(args.raw)
    people = sorted(raw.subjectID.astype(int).astype(str).unique())
    results = []
    for name, frame in [("primary_only", primary.assign(sensor_route="primary_ankle_back")),
                         ("availability_fallback", combined), ("frozen_context_baseline", baseline)]:
        directory = output / name
        directory.mkdir()
        validated = validate_mixed_windows(raw, frame)
        frame.to_csv(directory / "oof_predictions.csv", index=False)
        frame.drop(columns=[c for c in frame if c.startswith("truth_")]).to_csv(directory / "a2_runtime.csv", index=False)
        keys = ["subject_id", "session_id", "task_id", "window_start_s", "window_end_s"]
        frame[[*keys, *[c for c in frame if c.startswith("truth_")]]].to_csv(directory / "evaluation_truth.csv", index=False)
        support = EventSupport.build(raw, frame, allowed=people)
        checked = legacy_events(raw, frame.assign(event_fog=frame.is_fog), root, directory)
        metrics = support.metrics(frame.is_fog)
        for metric in ("covered_raw_events", "truth_events", "supported_events", "true_alarm_s", "false_alarm_s"):
            if abs(metrics[metric] - checked[metric]) > 1e-6:
                raise ValueError("Composition event integration mismatch")
        results.append({"variant": name, "validated_windows": validated,
                       "records_with_predictions": len(frame[keys[:3]].drop_duplicates()),
                       "fallback_rows": int(frame.sensor_route.eq("fallback_ankle").sum()), **metrics,
                       "false_alarm_events": checked["false_alarm_events"],
                       "median_new_alarm_delay_s": checked["median_new_alarm_delay_s"],
                       "new_alarm_delay_cases": checked["new_alarm_delay_cases"]})
    write_json(output / "comparison.json", results)
    table = pd.DataFrame(results)
    table.to_csv(output / "ppt_comparison.csv", index=False)
    plot_joint(output, [results[2], results[1]])
    (output / "index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>A1 联合候选</title>'
        '<body style="max-width:1150px;margin:32px auto;padding:20px;font-family:system-ui;line-height:1.7">'
        '<h1>A1：训练内选择主模型＋输入可用性备用</h1><p>已冻结组件的探索性组合，不重新挑参数。'
        '相同 106 记录、101 事件；本队列已反复分析，不能称为新外部验证。原默认未替换。</p>'
        + table.to_html(index=False, float_format=lambda n: f"{n:.4f}")
        + '<img src="figures/joint_comparison.png" style="width:100%" alt="Joint candidate vs frozen context"></body></html>', encoding="utf-8")
    write_json(output / "manifest.json", {"protocol_sha256": sha(output / "protocol.json"),
        "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

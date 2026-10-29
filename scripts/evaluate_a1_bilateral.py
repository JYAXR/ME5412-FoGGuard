"""Read-only matched-input evaluation; report natural left support separately."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from report_a1_author_cnn import event_summary
from run_a1_experiment import sha, write_json

from fogguard.evaluation.a1_author import annotate, grouped_diagnostics, matched_frames, metrics
from fogguard.evaluation.a1_fixed_split import role_frame
from fogguard.modeling.a1_bilateral import anchored_gyro_windows, left_view
from fogguard.modeling.a1_fog import A1FogModel


def plots(output, table):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / "figures"
    folder.mkdir()

    def save(fig, name):
        fig.savefig(folder / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(folder / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    names = ["left_matched", "bilateral_rf"]
    pair = table.loc[table.pool.eq("validation_own")].set_index("model").loc[names]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    x = np.arange(2)
    for i, field in enumerate(("precision", "recall", "fog_f1")):
        bars = ax.bar(x + (i - 1) * 0.25, pair[field], width=0.24, label=field)
        ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xticks(x, names)
    ax.set(ylim=(0, 1.08), title=f"VALIDATION | {pair.windows.iloc[0]} identical gyro windows, left33 vs bilateral66")
    ax.legend(ncol=3)
    save(fig, "matched_window_metrics")
    names = ["left_matched", "bilateral_rf", "left_natural"]
    own = table.loc[table.pool.eq("validation_own")].set_index("model").loc[names]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    for ax, field, title in zip(axes, ("false_alarm_s", "time_recall", "decision_support_fraction"),
                                 ("False alarm seconds", "FoG-time recall", "Raw-time support"), strict=True):
        bars = ax.bar(names, own[field], color=["#438da0", "#be7f39", "#779a64"])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.tick_params(axis="x", rotation=15)
        ax.set_title(title)
        ax.set_ylim(0, max(float(own[field].max()) * 1.18, 0.1))
    save(fig, "coverage_and_alarm_tradeoffs")
    common = table.loc[table.pool.eq("all_common")]
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    bars = ax.bar(common.model, common.fog_f1, color=["#438da0", "#be7f39", "#779a64", "#8e9292", "#8e9292", "#8e9292"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.tick_params(axis="x", rotation=20)
    ax.set(ylabel="FoG F1", ylim=(0, 1), title=f"{common.windows.iloc[0]} common windows | references use different inputs/selection")
    save(fig, "historical_common")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    args = parser.parse_args()
    run, root = args.run_dir, Path(__file__).resolve().parents[1]
    output = run / "evaluation"
    if output.exists():
        parser.error("Use a run without prior evaluation")
    config, split, lock, protocol = [json.loads((run / p).read_text("utf-8")) for p in
                                    ("config.json", "split.json", "MODEL_LOCK.json", "protocol.json")]
    check_lock(run, lock)
    if sha(args.raw) != protocol["raw_sha256"]:
        raise ValueError("Raw data changed")
    for path, digest in protocol["source_sha256"].items():
        if sha(root / path) != digest:
            raise ValueError(f"Source changed: {path}")
    output.mkdir()
    write_json(output / "VALIDATION_STARTED.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
        "model_lock_sha256": sha(run / "MODEL_LOCK.json")})
    raw = pd.read_csv(args.raw)
    train, validation = [role_frame(raw, split, r) for r in ("train", "validation")]
    del raw
    dual = anchored_gyro_windows(validation, require_right=True)
    left = anchored_gyro_windows(validation, require_right=False)
    models = {name: A1FogModel.load(run / name / "model.joblib") for name in config["models"]}
    frames = {"left_matched": annotate(left_view(dual), models["left_rf"].predict_windows(left_view(dual)), "validation"),
              "bilateral_rf": annotate(dual, models["bilateral_rf"].predict_windows(dual), "validation"),
              "left_natural": annotate(left, models["left_rf"].predict_windows(left), "validation")}
    reference = Path(config["reference_run"])
    check_lock(reference / "evaluation", json.loads((reference / "evaluation/EVALUATION_COMPLETED.json").read_text("utf-8")))
    for name in ("previous_author", "reference_rf_context", "reference_rf_ankle"):
        frames[name] = pd.read_csv(reference / "evaluation" / name / "predictions.csv")
    rows = []
    for name, frame in frames.items():
        dest = output / name
        dest.mkdir()
        frame.to_csv(dest / "predictions.csv", index=False)
        frame.drop(columns=[c for c in frame if c.startswith("truth_")]).to_csv(dest / "runtime.csv", index=False)
        rows.append({"model": name, "pool": "validation_own", **metrics(frame), **event_summary(validation, frame, dest, root)})
        for column in ("subject_id", "truth_activity", "task_id"):
            grouped_diagnostics(frame, column).to_csv(dest / f"by_{column}.csv", index=False)
    for name in config["models"]:
        frame = pd.read_csv(run / name / "oof_predictions.csv")
        dest = output / f"train_oof_{name}"
        dest.mkdir()
        rows.append({"model": name, "pool": "train_oof_calibration", **metrics(frame), **event_summary(train, frame, dest, root)})
        grouped_diagnostics(frame, "truth_activity").to_csv(dest / "by_truth_activity.csv", index=False)
    for name, frame in matched_frames(frames).items():
        rows.append({"model": name, "pool": "all_common", **metrics(frame)})
        frame.to_csv(output / f"common_{name}.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(output / "comparison.csv", index=False)
    own = table.loc[table.pool.eq("validation_own")].set_index("model")
    base, candidate, natural = own.loc["left_matched"], own.loc["bilateral_rf"], own.loc["left_natural"]
    conditions = {"matched_f1_higher": bool(candidate.fog_f1 > base.fog_f1),
        "false_alarm_seconds_not_higher": bool(candidate.false_alarm_s <= base.false_alarm_s + 1e-6),
        "fog_time_recall_not_lower": bool(candidate.time_recall + 1e-9 >= base.time_recall),
        "natural_left_support_not_lost": bool(candidate.decision_support_fraction + 1e-9 >= natural.decision_support_fraction)}
    gate = {"conditions": conditions, "passed": all(conditions.values()), "default_replaced": False,
            "scope": "Inspected development cohort, not new external generalization"}
    write_json(output / "development_gate.json", gate)
    os.environ.setdefault("MPLCONFIGDIR", str(root / "artifacts/matplotlib-a1-bilateral-cache"))
    plots(output, table)
    columns = ["model", "pool", "windows", "precision", "recall", "fog_f1", "covered_raw_events", "truth_events", "false_alarm_s", "time_recall", "decision_support_fraction"]
    images = "".join(f'<h2>{p.stem}</h2><img src="figures/{p.name}" style="width:100%" alt="{p.stem}">' for p in sorted((output / "figures").glob("*.png")))
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>A1 双踝输入对照</title>'
        '<body style="max-width:1250px;margin:30px auto;font-family:system-ui;line-height:1.8">'
        '<h1>A1：失败信号审查与双踝输入对照</h1><p><a href="../input_audit/index.html">先看TRAIN输入支持与同人同活动真实失败片段</a>。'
        '两模型训练窗口完全相同，仅增加右踝gyro特征；TRAIN-OOF选阈值，模型锁定后才评价VALIDATION，无新TEST成绩。</p>'
        '<p>left_matched与bilateral_rf直接比较；left_natural另示同一冻结左踝模型的自然可用范围。'
        '保留全部原始角色记录/事件，缺测未知不当正常；任意事件重叠不等于准确率或事件F1。历史参考训练/输入不同，仅作参考。</p>'
        f'<p>预定条件：{gate}。验证队列已被开发使用，未证明外部泛化，不自动替换默认。</p>'
        + table[columns].to_html(index=False, float_format=lambda v: f"{v:.4f}", na_rep="—") + images + '</body>', encoding="utf-8")
    check_lock(run, lock)
    write_json(output / "EVALUATION_COMPLETED.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_lock_sha256": sha(run / "MODEL_LOCK.json"), "validation_people": split["validation"],
        "test_windows_constructed": 0, "validation_driven_refits": 0,
        "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print(table[columns].to_string(index=False), flush=True)
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()

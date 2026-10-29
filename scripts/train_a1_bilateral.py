"""TRAIN failure-signal audit, then conditional fixed left33 / bilateral66 RF fits."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json
from sklearn.ensemble import RandomForestClassifier

from fogguard.evaluation.a1_author import annotate
from fogguard.evaluation.a1_bilateral import (
    failure_pairs,
    match_left_indices,
    support_summary,
    vector_rms,
)
from fogguard.evaluation.a1_fixed_split import choose_threshold, role_frame
from fogguard.evaluation.a1_temporal import person_indices
from fogguard.modeling.a1_bilateral import (
    BilateralFoG,
    anchored_gyro_windows,
    left_view,
    matched_features,
)
from fogguard.modeling.a1_fixed_split import prediction_frame
from fogguard.modeling.a1_fog import A1FogModel


def audit_figures(folder, dual, frame, rows, supports, cases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = folder / "figures"
    figures.mkdir()

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    labels = ["Left gyro", "Both gyros"]
    for ax, field, title in zip(axes, ("windows", "support_fraction"), ("TRAIN usable windows", "TRAIN raw-time support"), strict=True):
        bars = ax.bar(labels, [supports[k][field] for k in ("left", "dual")], color=["#3d8da4", "#bd7d3d"])
        ax.bar_label(bars, fmt="%.3f" if field == "support_fraction" else "%.0f", padding=3)
        ax.set_title(title)
        ax.set_ylim(0, max(supports[k][field] for k in ("left", "dual")) * 1.17)
    save(fig, "input_support")
    if len(rows):
        groups = [frame.truth_fog_fraction.le(1e-12) & frame.truth_activity.astype(str).eq("3"),
                  frame.truth_fog_fraction.le(1e-12) & frame.truth_activity.astype(str).isin(["6", "7"]),
                  frame.truth_fog_fraction.ge(1 - 1e-12)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        for ax, field in zip(axes, ("left_rms", "right_rms"), strict=True):
            data = [np.log10(rows.loc[mask, field].to_numpy() + 1e-6) for mask in groups]
            ax.boxplot(data, tick_labels=[f"non-FoG Stand\nn={sum(groups[0])}", f"non-FoG Turn\nn={sum(groups[1])}", f"pure FoG\nn={sum(groups[2])}"], showfliers=False)
            ax.set(title=field, ylabel="log10(demeaned vector RMS deg/s + 1e-6)")
        fig.suptitle("TRAIN descriptive distributions; overlap is not an impossibility proof")
        save(fig, "signal_distributions")
    for case in cases:
        a, b = case["negative_index"], case["positive_index"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, layout="constrained")
        for col, index in enumerate((a, b)):
            for row, (start, ankle) in enumerate(((0, "left"), (3, "right"))):
                ax = axes[row, col]
                for k, axis in enumerate("xyz"):
                    ax.plot(np.arange(120) / 60, dual[index].samples[:, start + k], lw=0.9, label=axis)
                ax.set(ylabel=f"{ankle} gyro deg/s", xlabel="Window-relative seconds")
                ax.grid(alpha=0.15)
                ax.legend(ncol=3, fontsize=7)
            record = frame.iloc[index]
            axes[0, col].set_title(f"{'Pure non-FoG FP' if col == 0 else 'Pure FoG'} | saved held-out score={record.fog_probability:.3f}")
        fig.suptitle(f"{case['group']} | same person/activity, disjoint windows; closest-left-feature pair\n"
                     f"left distance={case['left_distance']:.3f}, 66D distance={case['dual_distance']:.3f} (different dimensions, not an information-gain score)", fontsize=10)
        save(fig, case["group"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/a1_bilateral_v1.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run, root = args.run_dir, Path(__file__).resolve().parents[1]
    if run.exists() and any(run.iterdir()):
        parser.error("Use a fresh audit/training directory")
    config = json.loads(args.config.read_text("utf-8"))
    if config["models"] != ["left_rf", "bilateral_rf"]:
        raise ValueError("Unexpected input comparison")
    reference = Path(config["reference_run"])
    check_lock(reference, json.loads((reference / "MODEL_LOCK.json").read_text("utf-8")))
    split = json.loads(Path(config["split_config"]).read_text("utf-8"))["split"]
    old_protocol = json.loads((reference / "protocol.json").read_text("utf-8"))
    if sha(args.raw) != old_protocol["raw_sha256"] or split != json.loads((reference / "split.json").read_text("utf-8")):
        raise ValueError("Raw data or split changed")
    raw = pd.read_csv(args.raw)
    train = role_frame(raw, split, "train")
    del raw
    left, dual = [anchored_gyro_windows(train, require_right=flag) for flag in (False, True)]
    indices = match_left_indices(left, dual)
    previous = pd.read_csv(reference / "control/train_oof/predictions.csv")
    if len(left) != len(previous):
        raise ValueError("TRAIN left schedule differs from saved diagnostic control")
    for w, row in zip(left, previous.itertuples(), strict=True):
        if ((int(w.subject_id), int(w.session_id), int(w.task_id)) != (row.subject_id, row.session_id, row.task_id)
                or abs(w.end_s - row.window_end_s) > 1e-6 or w.fog_label != row.truth_fog_label):
            raise ValueError("Saved diagnostic predictions do not align")
    matched = previous.iloc[indices].reset_index(drop=True)
    run.mkdir(parents=True, exist_ok=True)
    audit = run / "input_audit"
    audit.mkdir()
    write_json(run / "config.json", config)
    write_json(run / "split.json", split)
    plans = json.loads((reference / "fold_plan.json").read_text("utf-8"))
    write_json(run / "fold_plan.json", plans)
    sources = [root / f"scripts/{n}.py" for n in ("train_a1_bilateral", "evaluate_a1_bilateral", "verify_a1_bilateral", "report_a1_author_cnn", "run_a1_event")]
    sources += [root / "recognition/diagnostics.py", *sorted((root / "src/fogguard").rglob("*.py"))]
    write_json(run / "protocol.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "raw_sha256": sha(args.raw),
        "config_sha256": sha(args.config), "split_sha256": sha(config["split_config"]),
        "reference_lock_sha256": sha(reference / "MODEL_LOCK.json"),
        "source_sha256": {str(p.relative_to(root)): sha(p) for p in sources},
        "data_roles_constructed": ["train"], "validation_windows_before_lock": 0, "test_windows_constructed": 0})
    previous.to_csv(audit / "left_reference_predictions.csv", index=False)
    matched.to_csv(audit / "joint_reference_predictions.csv", index=False)
    np.save(audit / "matched_left_indices.npy", indices)
    supports = {"left": support_summary(train, previous), "dual": support_summary(train, matched)}
    write_json(audit / "support.json", supports)
    people = []
    for p in split["train"]:
        people.append({"subject_id": p, "left_windows": sum(w.subject_id == str(p) for w in left),
                       "joint_windows": sum(w.subject_id == str(p) for w in dual)})
    pd.DataFrame(people).to_csv(audit / "by_subject.csv", index=False)
    left_counts = previous.groupby(["subject_id", "session_id", "task_id"]).size().to_dict()
    joint_counts = matched.groupby(["subject_id", "session_id", "task_id"]).size().to_dict()
    records = []
    for key, signal in train.groupby(["subjectID", "sessionID", "taskID"], sort=False):
        records.append({"subject_id": int(key[0]), "session_id": int(key[1]), "task_id": int(key[2]),
            "samples": len(signal), "fog_samples": int(signal.fog.sum()),
            "left_windows": int(left_counts.get(key, 0)), "joint_windows": int(joint_counts.get(key, 0)),
            "right_complete_samples": int(np.isfinite(signal[[f"ankleR_gyro_{a}" for a in "xyz"]].to_numpy()).all(axis=1).sum())})
    pd.DataFrame(records).to_csv(audit / "record_inputs.csv", index=False)
    ratio = len(dual) / len(left) if left else 0
    eligibility = {"joint_fraction": ratio, "minimum_joint_fraction": 0.8, "both_classes_each_outer_fit": False, "passed": False}
    rows, chosen, summaries, banks = pd.DataFrame(), [], [], None
    if len(dual):
        print(f"TRAIN input audit: left={len(left)}, dual={len(dual)} ({ratio:.1%}). Extract existing features once.", flush=True)
        banks = matched_features(dual)
        candidates, chosen, summaries = failure_pairs(matched, banks["bilateral_rf"][0])
        candidates.to_csv(audit / "all_candidate_pairs.csv", index=False)
        rows = pd.DataFrame({"left_rms": [vector_rms(w.samples[:, :3]) for w in dual],
                             "right_rms": [vector_rms(w.samples[:, 3:]) for w in dual]})
        rows.to_csv(audit / "signal_rms.csv", index=False)
        y, groups = np.array([w.fog_label for w in dual]), np.array([int(w.subject_id) for w in dual])
        eligibility["both_classes_each_outer_fit"] = all(set(y[np.isin(groups, p["fit_people"])]) == {0, 1} for p in plans)
        eligibility["passed"] = bool(ratio >= 0.8 and eligibility["both_classes_each_outer_fit"])
    write_json(audit / "selected_pairs.json", chosen)
    write_json(audit / "pair_groups.json", summaries)
    write_json(audit / "eligibility.json", eligibility)
    os.environ.setdefault("MPLCONFIGDIR", str(root / "artifacts/matplotlib-a1-bilateral-cache"))
    audit_figures(audit, dual, matched, rows, supports, chosen)
    images = "".join(f'<h2>{p.stem}</h2><img src="figures/{p.name}" style="width:100%" alt="{p.stem}">' for p in sorted((audit / "figures").glob("*.png")))
    (audit / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>TRAIN 信号审查</title>'
        '<body style="max-width:1200px;margin:30px auto;font-family:system-ui;line-height:1.8">'
        '<h1>TRAIN：可用输入与真实失败片段</h1><p>仅TRAIN，不是盲测；非FoG站立不等于主动停步。'
        '配对刻意选择同人同活动的最近左踝特征，不证明理论不可分或右踝必然有效。</p>'
        f'<pre>{json.dumps(supports, indent=2)}</pre><pre>{json.dumps(summaries, indent=2)}</pre>' + images + '</body>', encoding="utf-8")
    if not eligibility["passed"]:
        write_json(run / "TRAINING_NOT_RUN.json", {"reason": "Predeclared input eligibility failed", **eligibility})
        print("Input audit complete; training NOT_RUN, no redraw/imputation.", flush=True)
        return
    for name in config["models"]:
        directory = run / name
        directory.mkdir()
        values, names = banks[name]
        windows = left_view(dual) if name == "left_rf" else dual
        scores, seen = np.full(len(windows), np.nan), np.zeros(len(windows), int)
        for plan in plans:
            fit, held = [person_indices(groups, plan[p]) for p in ("fit_people", "held_out_people")]
            print(f"{name} fold{plan['fold']}: fit={len(fit)}, held={len(held)}", flush=True)
            classifier = RandomForestClassifier(**config["rf"], random_state=config["random_state"], n_jobs=-1).fit(values[fit], y[fit])
            joblib.dump(classifier, directory / f"fold_{plan['fold']}.joblib")
            if len(held):
                scores[held] = classifier.predict_proba(values[held])[:, 1]
                seen[held] += 1
        if not np.all(seen == 1) or not np.isfinite(scores).all():
            raise ValueError("Incomplete TRAIN held-out predictions")
        choice = choose_threshold(y, scores, config)
        cutoff = choice["selected"]["threshold"]
        predictions = annotate(windows, prediction_frame(windows, scores, cutoff, name), "train_oof")
        predictions["model_version"], predictions["prediction_source"] = f"a1_bilateral_v1/{name}", "participant_held_out_train_oof"
        predictions.to_csv(directory / "oof_predictions.csv", index=False)
        write_json(directory / "threshold_selection.json", choice)
        classifier = RandomForestClassifier(**config["rf"], random_state=config["random_state"], n_jobs=-1).fit(values, y)
        fitted = tuple(sorted(map(int, set(groups))))
        detector = BilateralFoG(classifier, name, windows[0].channel_names, names, cutoff, fitted, fitted)
        detector.save(directory / "model.joblib")
        restored = A1FogModel.load(directory / "model.joblib")
        pd.testing.assert_frame_equal(restored.predict_windows(windows[:10]), detector.predict_windows(windows[:10]))
        write_json(directory / "fit_metadata.json", {"fit_people": list(fitted), "assigned_train_people": split["train"],
            "fit_windows": len(windows), "features": len(names), "threshold_people": list(fitted),
            "validation_people_used": [], "test_people_used": [], "training_support": "joint-eligible left-anchored windows"})
    (run / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    write_json(run / "MODEL_LOCK.json", {"locked_utc": datetime.now(timezone.utc).isoformat(),
        "fit_people": list(fitted), "validation_windows_before_lock": 0, "test_windows_constructed": 0,
        "files": {str(p.relative_to(run)): sha(p) for p in run.rglob("*") if p.is_file()}})
    print("Input audit and models locked; independent VALIDATION evaluation may start.", flush=True)


if __name__ == "__main__":
    main()

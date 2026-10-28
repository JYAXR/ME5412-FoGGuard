"""Describe saved participant-held-out variability, not a new independent test."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = {
    "two_second_RF": "a1_multimodel_v1_20261028/rf/oof_predictions.csv",
    "SVM": "a1_multimodel_v1_20261028/svm/oof_predictions.csv",
    "HistGB": "a1_multimodel_v1_20261028/hist_gb/oof_predictions.csv",
    "small_CNN": "a1_multimodel_v1_20261028/cnn/oof_predictions.csv",
    "six_second_RF": "a1_context_v1_20261028/causal_context_6s_v1/oof_predictions.csv",
    "nested_selected": "a1_posture_search_v1_20261029/inner_selected/oof_predictions.csv",
}


def counts(frame):
    y, p = frame.truth_fog_label.to_numpy(int), frame.is_fog.to_numpy(bool)
    return np.array([((y == 1) & p).sum(), ((y == 0) & p).sum(),
                     ((y == 1) & ~p).sum(), ((y == 0) & ~p).sum()], int)


def metrics(c):
    tp, fp, fn, tn = map(int, c)
    ratio = lambda a, b: a / b if b else None
    return {"windows": tp + fp + fn + tn, "positive_windows": tp + fn,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "fog_f1": ratio(2 * tp, 2 * tp + fp + fn)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.reports / "a1_posture_search_v1_20261029/protocol.json").read_text("utf-8"))
    people = sorted({int(p) for f in protocol["outer_folds"] for p in f["test"]})
    assignments = {int(p): i for i, f in enumerate(protocol["outer_folds"], 1) for p in f["test"]}
    rng = np.random.default_rng(5412)
    draws = rng.integers(0, len(people), size=(5000, len(people)))
    rows, fold_rows, person_rows, samples, canonical, hashes = [], [], [], {}, None, {}
    for name, relative in SOURCES.items():
        source = args.reports / relative
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
        frame = pd.read_csv(source, float_precision="round_trip")
        if (len(frame) != 9636 or set(frame.subject_id) != set(people)
                or not frame.prediction_source.eq("subject_held_out_oof").all()
                or not frame.held_out_fold.eq(frame.subject_id.map(assignments)).all()):
            raise ValueError("Wrong support/provenance or held-out assignments")
        if not np.array_equal(frame.is_fog, frame.fog_probability.ge(frame.fog_threshold)):
            raise ValueError("Invalid saved threshold decisions")
        keys = frame[["subject_id", "session_id", "task_id", "window_start_s", "window_end_s", "truth_fog_label"]]
        if canonical is not None:
            pd.testing.assert_frame_equal(keys, canonical, atol=1e-6, rtol=0)
        canonical = keys
        by_person = []
        for p in people:
            c = counts(frame.loc[frame.subject_id.eq(p)])
            by_person.append(c)
            person_rows.append({"model": name, "subject_id": p, **metrics(c)})
        for fold in range(1, 6):
            fold_rows.append({"model": name, "fold": fold,
                              **metrics(counts(frame.loc[frame.held_out_fold.eq(fold)]))})
        boot = np.asarray(by_person)[draws].sum(axis=1)
        denominator = 2 * boot[:, 0] + boot[:, 1] + boot[:, 2]
        samples[name] = np.divide(2 * boot[:, 0], denominator, out=np.full(len(draws), np.nan), where=denominator > 0)
        ci = np.nanquantile(samples[name], [0.025, 0.975])
        fold_values = [r["fog_f1"] for r in fold_rows if r["model"] == name]
        rows.append({"model": name, **metrics(np.asarray(by_person).sum(axis=0)),
                     "fold_f1_min": min(fold_values), "fold_f1_max": max(fold_values),
                     "participant_bootstrap_f1_low": float(ci[0]), "participant_bootstrap_f1_high": float(ci[1])})
    difference = samples["nested_selected"] - samples["six_second_RF"]
    interval = np.nanquantile(difference, [0.025, 0.975])
    result = {"scope": "Previously inspected cohort; saved OOF variability, not fresh validation or retraining",
              "participants": people, "windows_per_model": 9636, "models": rows,
              "bootstrap": {"unit": "participant", "repetitions": 5000, "seed": 5412,
                "meaning": "Descriptive percentile intervals conditional on these saved predictions; no model-selection bias correction"},
              "paired_selected_minus_context_f1": {"point": rows[-1]["fog_f1"] - rows[-2]["fog_f1"],
                                                    "low": float(interval[0]), "high": float(interval[1])},
              "untouched_external_test_available": False,
              "deployment_models_fit_on_all_22": True, "input_sha256": hashes}
    (output / "assessment.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "model_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(person_rows).to_csv(output / "participant_metrics.csv", index=False)
    os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/matplotlib-a1-generalization-cache").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for i, row in enumerate(rows):
        ax.plot([row["participant_bootstrap_f1_low"], row["participant_bootstrap_f1_high"]], [i, i], color="#4c7fa4")
        ax.scatter([row["fog_f1"]], [i], color="#c66e36", zorder=3)
    ax.set_yticks(range(len(rows)), [r["model"] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pooled held-out F1; 95% participant-bootstrap descriptive interval")
    ax.set_title("22 people, repeatedly explored cohort; no new test or retraining")
    ax.grid(alpha=0.2, axis="x")
    fig.savefig(output / "generalization_variability.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "generalization_variability.svg", bbox_inches="tight")
    plt.close(fig)
    (output / "index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>A1 泛化证据审查</title>'
        '<body style="max-width:1150px;margin:32px auto;padding:20px;font-family:system-ui;line-height:1.7">'
        '<h1>A1：已有跨参与者测试的波动</h1><p>这些是保存 OOF 的描述性统计。'
        '没有未使用的独立测试人群；部署模型已使用全 22 人训练。区间不消除反复选方法产生的偏差。</p>'
        + pd.DataFrame(rows).to_html(index=False, float_format=lambda n: f"{n:.4f}")
        + '<img src="generalization_variability.png" style="width:100%" alt="Participant bootstrap intervals"></body></html>', encoding="utf-8")
    print(pd.DataFrame(rows)[["model", "fog_f1", "fold_f1_min", "fold_f1_max",
                              "participant_bootstrap_f1_low", "participant_bootstrap_f1_high"]].to_string(index=False))
    print(json.dumps(result["paired_selected_minus_context_f1"], indent=2))


if __name__ == "__main__":
    main()

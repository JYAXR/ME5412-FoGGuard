"""Audit saved subject-held-out FoG predictions; never retrain or alter the model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from diagnostics import ACTIVITIES, build_diagnostics
from sklearn.metrics import roc_auc_score

IDS = ["subject_id", "session_id", "task_id"]


def acceleration_magnitude(frame, position):
    """Keep missing axes missing instead of fabricating a zero magnitude."""
    channels = [f"{position}_acc_{axis}" for axis in "xyz"]
    return np.sqrt((frame[channels] ** 2).sum(axis=1, min_count=3))


def metrics(frame):
    y = frame.truth_fog_label.to_numpy(int)
    p = frame.is_fog.to_numpy(bool)
    tn, fp, fn, tp = [
        int(v.sum()) for v in ((y == 0) & ~p, (y == 0) & p, (y == 1) & ~p, (y == 1) & p)
    ]
    div = lambda a, b: a / b if b else None
    positive_f1 = div(2 * tp, 2 * tp + fp + fn)
    negative_f1 = div(2 * tn, 2 * tn + fp + fn)
    recall, specificity = div(tp, tp + fn), div(tn, tn + fp)
    return {
        "windows": len(y),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": div(tp, tp + fp),
        "recall": recall,
        "specificity": specificity,
        "fog_f1": positive_f1,
        "macro_f1": (positive_f1 + negative_f1) / 2
        if positive_f1 is not None and negative_f1 is not None
        else None,
        "balanced_accuracy": (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None,
        "accuracy": div(tp + tn, len(y)),
        "majority_baseline": div(max(tp + fn, tn + fp), len(y)),
        "auc": float(roc_auc_score(y, frame.fog_probability))
        if len(set(y)) == 2
        else None,
    }


def validate_predictions(frame, report):
    frame = frame.copy()
    for col in IDS:
        values = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(values).all() or not (values == np.floor(values)).all():
            raise ValueError("Invalid integer identifier")
        frame[col] = values.astype(int)
    if not frame.truth_fog_label.isin([0, 1]).all():
        raise ValueError("Non-binary truth")
    mapping = {True: True, False: False, "True": True, "False": False}
    frame["is_fog"] = frame.is_fog.map(mapping)
    if frame.is_fog.isna().any():
        raise ValueError("Invalid boolean prediction")
    for col in ["fog_probability", "fog_threshold", "truth_fog_fraction"]:
        a = frame[col].to_numpy(float)
        if not (np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()):
            raise ValueError(f"Invalid {col}")
    times = frame[["window_start_s", "window_end_s"]].to_numpy(float)
    if not np.isfinite(times).all() or not (times[:, 1] > times[:, 0]).all():
        raise ValueError("Invalid timing")
    if frame.duplicated(IDS + ["window_start_s"]).any():
        raise ValueError("Duplicate windows")
    for _, group in frame.groupby(IDS):
        if not (np.diff(group.window_start_s) > 0).all():
            raise ValueError("Unordered windows")
    if not (frame.is_fog == (frame.fog_probability >= frame.fog_threshold)).all():
        raise ValueError("Threshold decision mismatch")
    validation = report["validation"]
    folds = validation["fold_subjects"]
    thresholds = validation["fold_fog_thresholds"]
    if len(folds) != len(thresholds):
        raise ValueError("Fold metadata mismatch")
    seen = set()
    all_subjects = set(frame.subject_id)
    assignments = {}
    for index, (fold, threshold) in enumerate(zip(folds, thresholds), 1):
        train, test = set(map(int, fold["train"])), set(map(int, fold["test"]))
        if train & test or test & seen or train | test != all_subjects:
            raise ValueError("Subject leakage or incomplete fold")
        seen |= test
        for subject in test:
            assignments[subject] = index
            if not np.allclose(
                frame.loc[frame.subject_id == subject, "fog_threshold"],
                threshold,
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError("Fold threshold mismatch")
    if seen != all_subjects:
        raise ValueError("Missing held-out subjects")
    frame["held_out_fold"] = frame.subject_id.map(assignments)
    frame["outcome"] = np.select(
        [
            (frame.truth_fog_label == 1) & frame.is_fog,
            (frame.truth_fog_label == 0) & frame.is_fog,
            (frame.truth_fog_label == 1) & ~frame.is_fog,
        ],
        ["TP", "FP", "FN"],
        default="TN",
    )
    return frame


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_raw_windows(windows, predictions):
    """Match full-resolution window labels, fractions, times and one-to-one keys."""
    keyed = predictions.set_index(IDS + ["window_start_s"])
    lookup = {
        (int(a), int(b), int(c), round(float(t), 6)): row
        for (a, b, c, t), row in keyed.iterrows()
    }
    matched = set()
    if len(lookup) != len(predictions):
        raise ValueError("Rounded window keys collide")
    for window in windows:
        key = (
            int(window.subject_id),
            int(window.session_id),
            int(window.task_id),
            round(window.start_s, 6),
        )
        if key not in lookup or key in matched:
            raise ValueError(f"Raw window key mismatch: {key}")
        row = lookup[key]
        if (
            window.fog_label != row.truth_fog_label
            or abs(window.fog_fraction - row.truth_fog_fraction) > 1e-10
            or abs(window.end_s - row.window_end_s) > 1e-6
        ):
            raise ValueError(f"Raw truth mismatch: {key}")
        matched.add(key)
    if len(matched) != len(predictions):
        raise ValueError("Some predictions have no raw window")
    return len(matched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from fogguard.data.fogstar import iter_fogstar_windows, load_fogstar_frame

    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    if digest(args.raw, "md5") != report["dataset"]["md5"]:
        raise ValueError("Raw data differs from training provenance")
    predictions = validate_predictions(pd.read_csv(args.predictions), report)
    raw = load_fogstar_frame(args.raw)
    matched = validate_raw_windows(iter_fogstar_windows(raw), predictions)
    dt = float(np.median(np.diff(raw.timestamp)))
    if abs(dt - 1 / 60) > 1e-5:
        raise ValueError("Unexpected raw timestamp unit/sample interval")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    diagnostics, predictions = build_diagnostics(raw, predictions, metrics, output)
    predictions.to_csv(
        output / "全部窗口识别结果.csv", index=False, encoding="utf-8-sig"
    )
    subject_rows = []
    for subject, group in predictions.groupby("subject_id"):
        subject_rows.append(dict(subject_id=int(subject), **metrics(group)))
    pd.DataFrame(subject_rows).to_csv(
        output / "参与者指标.csv", index=False, encoding="utf-8-sig"
    )
    records, record_rows, cases = [], [], []
    for key, signal in raw.groupby(["subjectID", "sessionID", "taskID"], sort=True):
        a, b, c = map(int, key)
        group = predictions[
            (predictions.subject_id == a)
            & (predictions.session_id == b)
            & (predictions.task_id == c)
        ].sort_values("window_start_s")
        origin = float(signal.timestamp.iloc[0])
        # Display at 20 Hz only; metrics and label verification use all 60 Hz samples.
        view = signal.iloc[::3]
        magnitudes = [acceleration_magnitude(view, pos) for pos in ["ankleL", "back"]]
        sample_rows = np.column_stack([view.timestamp - origin, *magnitudes, view.fog])
        window_rows = [
            [
                round(r.window_start_s - origin, 5),
                round(r.window_end_s - origin, 5),
                int(r.truth_fog_label),
                int(r.is_fog),
                float(r.fog_probability),
                float(r.fog_threshold),
                r.outcome,
                float(r.truth_fog_fraction),
                r.truth_manifestation,
                ACTIVITIES[int(r.truth_activity_context)],
            ]
            for r in group.itertuples()
        ]
        name = f"P{a:02d} / S{b} / T{c}"
        records.append(
            {
                "name": name,
                "subject": a,
                "fold": int(
                    predictions.loc[predictions.subject_id == a, "held_out_fold"].iloc[
                        0
                    ]
                ),
                "origin_s": origin,
                "samples": [
                    [round(float(v), 5) if np.isfinite(v) else None for v in row]
                    for row in sample_rows
                ],
                "windows": window_rows,
                "metrics": metrics(group),
            }
        )
        record_rows.append(
            dict(subject_id=a, session_id=b, task_id=c, **metrics(group))
        )
        # All contiguous error runs, ranked later by window count, then stable record/time order.
        outcomes = group.outcome.tolist()
        starts = group.window_start_s.to_numpy()
        start = 0
        for i in range(1, len(group) + 1):
            if (
                i == len(group)
                or outcomes[i] != outcomes[start]
                or starts[i] - starts[i - 1] > 0.75
            ):
                if outcomes[start] in ["FP", "FN", "TP"]:
                    cases.append(
                        {
                            "record": len(records) - 1,
                            "window": start,
                            "outcome": outcomes[start],
                            "consecutive_windows": i - start,
                            "name": name,
                            "start_s": float(starts[start] - origin),
                        }
                    )
                start = i
    cases.sort(key=lambda x: (-x["consecutive_windows"], x["record"], x["window"]))
    chosen = [
        next(c for c in cases if c["outcome"] == kind) for kind in ["FN", "FP", "TP"]
    ]
    pd.DataFrame(cases).to_csv(
        output / "连续案例.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(record_rows).to_csv(
        output / "记录指标.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "metrics": metrics(predictions),
        "subjects": len(subject_rows),
        "records": len(records),
        "records_with_predictions": sum(bool(r["windows"]) for r in records),
        "validated_raw_windows": matched,
        "raw_sample_interval_s": dt,
        "display_missing_acceleration_samples": {
            pos: int(
                raw[[f"{pos}_acc_{axis}" for axis in "xyz"]].isna().any(axis=1).sum()
            )
            for pos in ["ankleL", "back"]
        },
        "evaluation": "Saved out-of-fold predictions; no model retraining in this audit",
        "raw_sha256": digest(args.raw),
        "predictions_sha256": digest(args.predictions),
        "training_report_sha256": digest(args.training_report),
        "fold_validation": report["validation"],
        "selected_cases": chosen,
        "diagnostics": diagnostics,
    }
    (output / "核验与指标.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload = json.dumps(
        {"summary": summary, "records": records},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")
    template = Path(__file__).with_name("review.html").read_text(encoding="utf-8")
    (output / "真实数据冻结识别验收.html").write_text(
        template.replace("__DATA__", payload), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

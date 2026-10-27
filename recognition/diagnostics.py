"""Truth-stratified and causal-decision coverage diagnostics, without model changes."""

import numpy as np
import pandas as pd

SUBTYPES = {1: "拖步 Shuffling", 2: "颤动 Trembling", 3: "运动不能 Akinesia"}
ACTIVITIES = {
    0: "未分类",
    1: "行走",
    2: "坐姿",
    3: "站立",
    4: "坐到站",
    5: "站到坐",
    6: "右转",
    7: "左转",
    -1: "混合动作",
}
TASKS = {
    1: "起立行走 TUG",
    2: "站立 1 分钟",
    3: "往返行走",
    4: "门口行走",
    5: "持水行走",
    6: "计数行走",
    7: "360° 转身",
}
DT = 1 / 60
HOLD = 0.5
EPS = 1e-6


def majority(values, *, strictly=False):
    values = np.asarray(values)
    if len(values) == 0:
        return None
    labels, counts = np.unique(values, return_counts=True)
    best = counts.max()
    if (counts == best).sum() != 1 or best < len(values) / 2:
        return None
    if strictly and best <= len(values) / 2:
        return None
    return int(labels[counts.argmax()])


def annotate_windows(raw, predictions):
    """Assign subtype from positive raw samples only; never from model decisions."""
    result = predictions.copy()
    result["truth_manifestation"] = "非冻结"
    result["truth_activity_context"] = -1
    for key, group in result.groupby(["subject_id", "session_id", "task_id"]):
        a, b, c = key
        signal = raw[(raw.subjectID == a) & (raw.sessionID == b) & (raw.taskID == c)]
        time = signal.timestamp.to_numpy(float)
        for index, row in group.iterrows():
            left = np.searchsorted(time, row.window_start_s - EPS)
            right = np.searchsorted(time, row.window_end_s + EPS, side="right")
            chunk = signal.iloc[left:right]
            if (
                len(chunk) != 120
                or abs(chunk.fog.mean() - row.truth_fog_fraction) > 1e-10
            ):
                raise ValueError("Diagnostic window alignment mismatch")
            activity = majority(chunk.activity.to_numpy())
            result.loc[index, "truth_activity_context"] = (
                activity if activity is not None else -1
            )
            if row.truth_fog_label:
                manifestations = chunk.loc[chunk.fog == 1, "fog_severity"]
                if not manifestations.isin([1, 2, 3]).all():
                    raise ValueError("Invalid manifestation on positive samples")
                subtype = majority(manifestations.to_numpy(), strictly=True)
                result.loc[index, "truth_manifestation"] = SUBTYPES.get(
                    subtype, "混合表现"
                )
    return result


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + EPS:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((float(start), float(end)))
    return merged


def intersect_intervals(a, b):
    a, b = merge_intervals(a), merge_intervals(b)
    i = j = 0
    segments = []
    while i < len(a) and j < len(b):
        start, stop = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if stop > start:
            segments.append((start, stop))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return segments


def intersection_duration(a, b):
    return sum(stop - start for start, stop in intersect_intervals(a, b))


def duration(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def raw_truth_intervals(signal):
    time = signal.timestamp.to_numpy(float)
    labels = signal.fog.to_numpy(int)
    return merge_intervals([(t, t + DT) for t, label in zip(time, labels) if label])


def decision_intervals(predictions, observation_end):
    """A decision exists only after window end and is held at most one step."""
    end = predictions.window_end_s.to_numpy(float)
    alarms, support = [], []
    for index, (t, alarm) in enumerate(zip(end, predictions.is_fog)):
        stop = min(
            t + HOLD,
            end[index + 1] if index + 1 < len(end) else observation_end,
            observation_end,
        )
        if stop <= t:
            continue
        support.append((float(t), float(stop)))
        if alarm:
            alarms.append((float(t), float(stop)))
    return merge_intervals(alarms), merge_intervals(support)


def event_rows(signal, predictions):
    """Coverage uses any positive overlap; it is not one-to-one event matching."""
    truth = raw_truth_intervals(signal)
    end = float(signal.timestamp.iloc[-1]) + DT
    alarms, support = decision_intervals(predictions, end)
    observed_intervals = merge_intervals(
        [(float(t), float(t) + DT) for t in signal.timestamp]
    )
    alarms = intersect_intervals(alarms, observed_intervals)
    support = intersect_intervals(support, observed_intervals)
    rows = []
    for start, stop in truth:
        hits = [(a, b) for a, b in alarms if min(stop, b) - max(start, a) > EPS]
        first = max(start, hits[0][0]) if hits else None
        previous = bool(hits and hits[0][0] < start - EPS)
        rows.append(
            {
                "start_s": start,
                "end_s": stop,
                "duration_s": stop - start,
                "covered": bool(hits),
                "first_alarm_overlap_s": first,
                "onset_delay_s": first - start if first is not None else None,
                "alarm_already_active_at_onset": previous,
                "model_support_s": intersection_duration([(start, stop)], support),
                "alarm_overlap_s": intersection_duration([(start, stop)], alarms),
            }
        )
    observed = duration(observed_intervals)
    truth_duration = duration(truth)
    alarm_duration = duration(alarms)
    positive_duration = intersection_duration(truth, alarms)
    alarm_rows = []
    for start, stop in alarms:
        alarm_rows.append(
            {
                "start_s": start,
                "end_s": stop,
                "duration_s": stop - start,
                "overlaps_truth": intersection_duration([(start, stop)], truth) > EPS,
            }
        )
    summary = {
        "observed_s": observed,
        "truth_fog_s": truth_duration,
        "alarm_s": alarm_duration,
        "true_alarm_s": positive_duration,
        "false_alarm_s": max(0.0, alarm_duration - positive_duration),
        "missed_fog_s": max(0.0, truth_duration - positive_duration),
        "model_support_s": duration(support),
    }
    return rows, alarm_rows, summary


def clustered_intervals(predictions, repetitions=2000, seed=5412):
    """Resample whole participants; do not treat overlapping windows as IID."""
    counts = []
    for _, group in predictions.groupby("subject_id"):
        counts.append(
            [(group.outcome == label).sum() for label in ["TP", "TN", "FP", "FN"]]
        )
    counts = np.asarray(counts)
    rng = np.random.default_rng(seed)
    samples = counts[rng.integers(0, len(counts), size=(repetitions, len(counts)))].sum(
        axis=1
    )
    tp, tn, fp, fn = samples.T.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = {
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
            "fog_f1": 2 * tp / (2 * tp + fp + fn),
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
        }
    result = {}
    for name, values in scores.items():
        finite = values[np.isfinite(values)]
        result[name] = {
            "low": float(np.quantile(finite, 0.025)) if len(finite) else None,
            "high": float(np.quantile(finite, 0.975)) if len(finite) else None,
            "valid_repetitions": len(finite),
        }
    return {
        "method": "participant cluster bootstrap percentile 95%",
        "seed": seed,
        "repetitions": repetitions,
        "metrics": result,
    }


def build_diagnostics(raw, predictions, metric_function, output):
    enriched = annotate_windows(raw, predictions)
    positive = enriched[enriched.truth_fog_label == 1]
    subtype_rows = []
    for label, group in positive.groupby("truth_manifestation", sort=True):
        tp = int(group.is_fog.sum())
        subtype_rows.append(
            {
                "manifestation": label,
                "positive_windows": len(group),
                "subjects": int(group.subject_id.nunique()),
                "TP": tp,
                "FN": len(group) - tp,
                "recall": tp / len(group),
            }
        )
    activity_rows, task_rows = [], []
    for key, group in enriched.groupby("truth_activity_context"):
        activity_rows.append(
            dict(activity=ACTIVITIES[int(key)], **metric_function(group))
        )
    for key, group in enriched.groupby("task_id"):
        task_rows.append(dict(task=TASKS[int(key)], **metric_function(group)))
    events, alarms, record_summaries = [], [], []
    record_indices = {
        key: i
        for i, (key, _) in enumerate(
            raw.groupby(["subjectID", "sessionID", "taskID"], sort=True)
        )
    }
    for key, signal in raw.groupby(["subjectID", "sessionID", "taskID"], sort=True):
        key = tuple(map(int, key))
        a, b, c = key
        group = enriched[
            (enriched.subject_id == a)
            & (enriched.session_id == b)
            & (enriched.task_id == c)
        ].sort_values("window_start_s")
        truth_rows, alarm_rows, summary = event_rows(signal, group)
        identifiers = {"subject_id": a, "session_id": b, "task_id": c}
        origin = float(signal.timestamp.iloc[0])
        for row in truth_rows:
            # Event metadata used only to navigate; no model input depends on truth.
            closest = (
                int(np.argmin(np.abs(group.window_end_s.to_numpy() - row["start_s"])))
                if len(group)
                else None
            )
            if row["model_support_s"] <= EPS:
                if not len(group):
                    support_reason = "整段无有效预测窗口"
                elif row["end_s"] <= float(group.window_end_s.iloc[0]) + EPS:
                    support_reason = "首个判定前"
                elif row["start_s"] >= float(group.window_end_s.iloc[-1]) + HOLD - EPS:
                    support_reason = "末个判定保持期后"
                else:
                    support_reason = "预测时间断档"
            elif row["model_support_s"] < row["duration_s"] - EPS:
                support_reason = "部分时段有判定"
            else:
                support_reason = "全事件有判定"
            channels = [
                f"{pos}_{kind}_{axis}"
                for pos in ["ankleL", "back"]
                for kind in ["acc", "gyro"]
                for axis in "xyz"
            ]
            event_samples = signal[
                (signal.timestamp >= row["start_s"] - EPS)
                & (signal.timestamp < row["end_s"] - EPS)
            ]
            missing_imu = int(
                (~np.isfinite(event_samples[channels].to_numpy(float)))
                .any(axis=1)
                .sum()
            )
            events.append(
                dict(
                    **identifiers,
                    **row,
                    record=record_indices.get(key),
                    closest_window=closest,
                    relative_start_s=row["start_s"] - origin,
                    support_reason=support_reason,
                    raw_event_samples=len(event_samples),
                    incomplete_imu_samples=missing_imu,
                )
            )
        alarms.extend(dict(**identifiers, **row) for row in alarm_rows)
        channels = [
            f"{pos}_{kind}_{axis}"
            for pos in ["ankleL", "back"]
            for kind in ["acc", "gyro"]
            for axis in "xyz"
        ]
        record_summaries.append(
            dict(
                **identifiers,
                **summary,
                raw_samples=len(signal),
                incomplete_imu_samples=int(
                    (~np.isfinite(signal[channels].to_numpy(float))).any(axis=1).sum()
                ),
                all_missing_input_channels="|".join(
                    col for col in channels if signal[col].isna().all()
                ),
                prediction_windows=len(group),
                truth_events=len(truth_rows),
            )
        )
    total = {
        key: sum(row[key] for row in record_summaries)
        for key in [
            "observed_s",
            "truth_fog_s",
            "alarm_s",
            "true_alarm_s",
            "false_alarm_s",
            "missed_fog_s",
            "model_support_s",
        ]
    }
    covered = sum(row["covered"] for row in events)
    missed_no_support = sum(
        not row["covered"] and row["model_support_s"] <= EPS for row in events
    )
    delayed = [
        row["onset_delay_s"]
        for row in events
        if row["covered"] and not row["alarm_already_active_at_onset"]
    ]
    ratio = lambda a, b: a / b if b else None
    summary = dict(
        **total,
        truth_events=len(events),
        covered_events=covered,
        missed_events=len(events) - covered,
        missed_events_without_model_support=missed_no_support,
        event_any_overlap_coverage=ratio(covered, len(events)),
        time_precision=ratio(total["true_alarm_s"], total["alarm_s"]),
        time_recall=ratio(total["true_alarm_s"], total["truth_fog_s"]),
        decision_support_fraction=ratio(total["model_support_s"], total["observed_s"]),
        alarm_events=len(alarms),
        false_alarm_events=sum(not r["overlaps_truth"] for r in alarms),
        covered_with_preexisting_alarm=sum(
            row["alarm_already_active_at_onset"] for row in events
        ),
        new_alarm_delay_cases=len(delayed),
        median_new_alarm_onset_delay_s=float(np.median(delayed)) if delayed else None,
        no_support_reasons={
            reason: sum(
                not r["covered"]
                and r["model_support_s"] <= EPS
                and r["support_reason"] == reason
                for r in events
            )
            for reason in [
                "整段无有效预测窗口",
                "首个判定前",
                "末个判定保持期后",
                "预测时间断档",
            ]
        },
    )
    files = {
        "分组窗口识别结果.csv": enriched,
        "冻结表现分组.csv": pd.DataFrame(subtype_rows),
        "动作分组.csv": pd.DataFrame(activity_rows),
        "任务分组.csv": pd.DataFrame(task_rows),
        "真实冻结事件覆盖.csv": pd.DataFrame(events),
        "报警片段.csv": pd.DataFrame(alarms),
        "记录时间覆盖.csv": pd.DataFrame(record_summaries),
        "无预测记录问题清单.csv": pd.DataFrame(
            [r for r in record_summaries if not r["prediction_windows"]]
        ),
    }
    for name, data in files.items():
        data.to_csv(output / name, index=False, encoding="utf-8-sig")
    missing = [row for row in events if not row["covered"]]
    missing.sort(
        key=lambda row: (-row["duration_s"], row["subject_id"], row["start_s"])
    )
    return {
        "subtypes": subtype_rows,
        "activities": activity_rows,
        "tasks": task_rows,
        "bootstrap": clustered_intervals(enriched),
        "event_summary": summary,
        "missed_events": missing[:20],
        "no_prediction_records": [
            dict(
                **row,
                record=record_indices[
                    (row["subject_id"], row["session_id"], row["task_id"])
                ],
            )
            for row in record_summaries
            if not row["prediction_windows"]
        ],
    }, enriched

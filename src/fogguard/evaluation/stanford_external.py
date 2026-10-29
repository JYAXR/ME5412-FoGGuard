"""Native128Hz interval metrics; decisions start at delayed availability, not window time."""

import numpy as np

from fogguard.data.stanford_external import native_intervals
from fogguard.evaluation.intervals import merge, overlap


def intersections(first, second):
    return merge((max(a, c), min(b, d)) for a, b in first for c, d in second if min(b, d) > max(a, c))


def difference(first, second):
    result = []
    for a, b in first:
        cursor = a
        for c, d in second:
            if d <= cursor or c >= b:
                continue
            if c > cursor:
                result.append((cursor, min(c, b)))
            cursor = max(cursor, d)
            if cursor >= b:
                break
        if cursor < b:
            result.append((cursor, b))
    return result


def record_metrics(time, labels, predictions):
    time, labels = np.asarray(time, float), np.asarray(labels, int)
    observed = native_intervals(time, np.ones(len(time), bool))
    truth = native_intervals(time, labels.astype(bool))
    starts = predictions.available_at_s.to_numpy(float)
    if (np.diff(starts) <= 0).any():
        raise ValueError("Decision availability not increasing")
    stops = np.minimum(starts + 0.5, np.r_[starts[1:], time[-1] + 1 / 128]) if len(starts) else []
    supported = intersections(merge(zip(starts, stops, strict=True)), observed)
    alarm = intersections(merge((a, b) for a, b, y in zip(starts, stops, predictions.is_fog, strict=True) if y), observed)
    true_alarm = intersections(alarm, truth)
    false_alarm = difference(alarm, truth)
    events = []
    for i, (a, b) in enumerate(truth):
        hit = intersections([(a, b)], alarm)
        missing = difference([(a, b)], alarm)
        events.append({"event_in_record": i, "start_s": a, "end_s": b, "duration_s": b - a,
            "support_s": overlap(a, b, supported), "alarm_s": sum(d - c for c, d in hit),
            "covered": bool(hit), "first_overlap_delay_s": hit[0][0] - a if hit else None,
            "alarm_already_active_at_onset": any(c < a and d > a for c, d in alarm),
            "longest_unalarmed_s": max((d - c for c, d in missing), default=0.0)})
    true_s, false_s = sum(b - a for a, b in true_alarm), sum(b - a for a, b in false_alarm)
    truth_s, observed_s = sum(b - a for a, b in truth), sum(b - a for a, b in observed)
    support_s = sum(b - a for a, b in supported)
    result = {"observed_s": observed_s, "truth_s": truth_s, "support_s": support_s,
        "truth_events": len(events), "covered_raw_events": sum(e["covered"] for e in events),
        "supported_events": sum(e["support_s"] > 1e-6 for e in events), "true_alarm_s": true_s, "false_alarm_s": false_s,
        "nonfog_alarm_pieces": len(false_alarm), "time_recall": true_s / truth_s if truth_s else 0,
        "time_precision": true_s / (true_s + false_s) if true_s + false_s else 0,
        "decision_support_fraction": support_s / observed_s if observed_s else 0}
    return result, events, {"observed": observed, "truth": truth, "supported": supported, "alarm": alarm, "false_alarm": false_alarm}


def aggregate(rows):
    fields = ["observed_s", "truth_s", "support_s", "truth_events", "covered_raw_events", "supported_events", "true_alarm_s", "false_alarm_s", "nonfog_alarm_pieces"]
    total = {k: float(sum(r[k] for r in rows)) for k in fields}
    total.update(time_recall=total["true_alarm_s"] / total["truth_s"] if total["truth_s"] else 0,
        time_precision=total["true_alarm_s"] / (total["true_alarm_s"] + total["false_alarm_s"]) if total["true_alarm_s"] + total["false_alarm_s"] else 0,
        decision_support_fraction=total["support_s"] / total["observed_s"] if total["observed_s"] else 0)
    return total

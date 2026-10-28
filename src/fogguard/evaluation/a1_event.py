"""Training-only event objective with exact causal half-second decision support."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from fogguard.modeling.a1_event import EventPolicy, apply_event_policy
from fogguard.modeling.a1_multimodel import make_classifier, model_scores

EPS = 1e-6
IDS = ["subject_id", "session_id", "task_id"]


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1] + EPS:
            result[-1] = result[-1][0], max(end, result[-1][1])
        else:
            result.append((float(start), float(end)))
    return result


def overlap(start, end, intervals):
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in intervals)


@dataclass
class EventSupport:
    duration: np.ndarray
    positive: np.ndarray
    event_overlap: np.ndarray
    truth_s: float
    raw_people: list[str]

    @classmethod
    def build(cls, raw, windows, *, allowed, forbidden=()):
        """raw must be prefiltered; reject test people rather than silently dropping."""
        allowed, forbidden = set(map(str, allowed)), set(map(str, forbidden))
        people = set(raw.subjectID.astype(int).astype(str))
        window_people = set(windows.subject_id.astype(int).astype(str))
        if allowed & forbidden or people != allowed or not window_people <= allowed:
            raise ValueError("Wrong raw people or test participant leakage")
        duration, positive = np.zeros(len(windows)), np.zeros(len(windows))
        event_rows, truth_s = [], 0.0
        for key, signal in raw.groupby(["subjectID", "sessionID", "taskID"], sort=False):
            time = signal.timestamp.to_numpy(float)
            if len(time) > 1 and not np.all(np.diff(time) > 0):
                raise ValueError("Raw timestamps must be ordered within a record")
            observed = merge(zip(time, time + 1 / 60, strict=True))
            truth = merge((t, t + 1 / 60) for t, y in zip(time, signal.fog, strict=True) if y)
            truth_s += sum(b - a for a, b in truth)
            mask = np.ones(len(windows), dtype=bool)
            for name, value in zip(IDS, key, strict=True):
                mask &= windows[name].astype(int).eq(int(value)).to_numpy()
            positions = np.flatnonzero(mask)
            ends = windows.iloc[positions].window_end_s.to_numpy(float)
            if len(ends) > 1 and not np.all(np.diff(ends) > 0):
                raise ValueError("Decision times must be ordered")
            stops = np.minimum(ends + 0.5, np.r_[ends[1:], time[-1] + 1 / 60]) if len(ends) else []
            stops = np.minimum(stops, time[-1] + 1 / 60)
            for j, start, end in zip(positions, ends, stops, strict=True):
                duration[j] = overlap(start, end, observed)
                positive[j] = overlap(start, end, truth)
            for start, end in truth:
                row = np.zeros(len(windows), dtype=bool)
                if len(ends):
                    row[positions] = np.minimum(stops, end) - np.maximum(ends, start) > EPS
                event_rows.append(row)
        matrix = np.asarray(event_rows, dtype=bool).reshape((-1, len(windows)))
        return cls(duration, positive, matrix, truth_s, sorted(people))

    def metrics(self, decision):
        decision = np.asarray(decision, dtype=bool)
        if decision.shape != self.duration.shape:
            raise ValueError("Decision/support length mismatch")
        true_s = float(self.positive @ decision)
        false_s = float(np.maximum(self.duration - self.positive, 0) @ decision)
        covered = int(np.any(self.event_overlap[:, decision], axis=1).sum())
        return {"truth_events": len(self.event_overlap), "covered_raw_events": covered,
                "supported_events": int(self.event_overlap.any(axis=1).sum()),
                "true_alarm_s": true_s, "false_alarm_s": false_s,
                "time_recall": true_s / self.truth_s if self.truth_s else 0.0,
                "time_precision": true_s / (true_s + false_s) if true_s + false_s else 0.0}


def grouped_inner_scores(values, labels, groups, config, *, forbidden=()):
    groups = np.asarray(groups).astype(str)
    if set(groups) & set(map(str, forbidden)):
        raise ValueError("Outer test people entered inner fitting")
    splitter = StratifiedGroupKFold(n_splits=config["inner_folds"], shuffle=True,
                                    random_state=config["random_state"])
    scores, seen, folds = np.full(len(labels), np.nan), np.zeros(len(labels), int), []
    for train, validation in splitter.split(values, labels, groups):
        a, b = set(groups[train]), set(groups[validation])
        if a & b or set(np.asarray(labels)[train]) != {0, 1}:
            raise ValueError("Inner leakage or missing class")
        model = make_classifier("rf", config).fit(values[train], np.asarray(labels)[train])
        scores[validation] = model_scores(model, "rf", values[validation])
        seen[validation] += 1
        folds.append({"train": sorted(a), "validation": sorted(b)})
    if not np.all(seen == 1) or not np.isfinite(scores).all():
        raise ValueError("Incomplete inner OOF")
    return scores, folds


def choose_event_policy(frame, support, config):
    grid = config["threshold_grid"]
    options = []
    y = frame.truth_fog_label.to_numpy(int)
    for threshold in np.linspace(grid["start"], grid["stop"], grid["count"]):
        for confirmation in config["confirmation_choices"]:
            policy = EventPolicy(float(threshold), int(confirmation))
            decision = apply_event_policy(frame, policy).event_fog.to_numpy(bool)
            options.append({"threshold": float(threshold), "confirmation_windows": int(confirmation),
                            "fog_f1": float(f1_score(y, decision, zero_division=0)),
                            "balanced_accuracy": float(balanced_accuracy_score(y, decision)),
                            **support.metrics(decision)})
    control = max((r for r in options if r["confirmation_windows"] == 1),
                  key=lambda r: (r["fog_f1"], r["balanced_accuracy"], r["threshold"]))
    budget = control["false_alarm_s"] * config["false_duration_budget_multiplier"]
    feasible = [r for r in options if r["false_alarm_s"] <= budget + EPS and (
        not config["require_control_time_coverage"] or r["true_alarm_s"] >= control["true_alarm_s"] - EPS)]
    selected = max(feasible, key=lambda r: (r["covered_raw_events"], r["true_alarm_s"],
                                           -r["false_alarm_s"], -r["confirmation_windows"], r["threshold"]))
    return {"selection_people": support.raw_people, "control": control, "selected": selected,
            "false_alarm_budget_s": budget, "feasible_count": len(feasible), "options": options}

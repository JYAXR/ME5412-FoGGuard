"""Descriptive pure-label failure pairs; no claim of universal identifiability."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fogguard.evaluation.a1_event import EventSupport, merge


def window_key(window):
    return (str(window.subject_id), str(window.session_id), str(window.task_id), round(window.start_s, 6), round(window.end_s, 6))


def match_left_indices(left, dual):
    lookup = {window_key(w): i for i, w in enumerate(left)}
    indices = np.array([lookup[window_key(w)] for w in dual], int)
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate matched window")
    return indices


def vector_rms(samples):
    centered = samples - samples.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))


def support_summary(raw, frame):
    observed = sum(sum(b - a for a, b in merge(zip(g.timestamp, g.timestamp + 1 / 60, strict=True)))
                   for _, g in raw.groupby(["subjectID", "sessionID", "taskID"]))
    if not len(frame):
        events = sum(len(merge((t, t + 1 / 60) for t, y in zip(g.timestamp, g.fog, strict=True) if y))
                     for _, g in raw.groupby(["subjectID", "sessionID", "taskID"]))
        return {"raw_records": raw.groupby(["subjectID", "sessionID", "taskID"]).ngroups,
                "windows": 0, "truth_events": events, "supported_events": 0,
                "observed_s": observed, "support_s": 0.0, "support_fraction": 0.0}
    support = EventSupport.build(raw, frame, allowed=raw.subjectID.unique())
    return {"raw_records": raw.groupby(["subjectID", "sessionID", "taskID"]).ngroups,
            "windows": len(frame), "truth_events": len(support.event_overlap),
            "supported_events": int(support.event_overlap.any(axis=1).sum()),
            "observed_s": observed, "support_s": float(support.duration.sum()),
            "support_fraction": float(support.duration.sum()) / observed}


def failure_pairs(frame, values):
    """Closest-left-feature pure FP/FoG pairs within person+activity, TRAIN only."""
    z = StandardScaler().fit_transform(values)
    candidates, chosen, summaries = [], [], []
    codes = [(3, "nonfog_stand_vs_fog"), (6, "right_turn_nonfog_vs_fog"), (7, "left_turn_nonfog_vs_fog")]
    for code, label in codes:
        negative = frame.truth_fog_fraction.le(1e-12) & frame.is_fog & frame.truth_activity.astype(str).eq(str(code))
        positive = frame.truth_fog_fraction.ge(1 - 1e-12) & frame.truth_activity.astype(str).eq(str(code))
        found = []
        for i in np.flatnonzero(negative):
            row = frame.iloc[i]
            same_person = frame.subject_id.astype(str).eq(str(row.subject_id)).to_numpy()
            same_record = frame.session_id.eq(row.session_id) & frame.task_id.eq(row.task_id)
            disjoint = ~same_record | (frame.window_start_s - row.window_start_s).abs().ge(6 - 1e-6)
            eligible = np.flatnonzero(positive.to_numpy() & same_person & disjoint.to_numpy())
            if not len(eligible):
                continue
            distances = np.sqrt(np.mean((z[eligible, :33] - z[i, :33])**2, axis=1))
            rank = min(range(len(eligible)), key=lambda j: (distances[j], int(eligible[j])))
            j = int(eligible[rank])
            pair = {"group": label, "activity": code, "negative_index": int(i), "positive_index": j,
                    "left_distance": float(distances[rank]),
                    "dual_distance": float(np.sqrt(np.mean((z[j] - z[i])**2))),
                    "person": int(row.subject_id)}
            candidates.append(pair)
            found.append(pair)
        if found:
            chosen.append(min(found, key=lambda p: (p["left_distance"], p["negative_index"], p["positive_index"])))
        summaries.append({"group": label, "activity": code, "pure_fp_queries": int(negative.sum()),
                          "queries_with_disjoint_same_person_fog": len(found), "pair_found": bool(found)})
    columns = ["group", "activity", "negative_index", "positive_index", "left_distance", "dual_distance", "person"]
    return pd.DataFrame(candidates, columns=columns), chosen, summaries

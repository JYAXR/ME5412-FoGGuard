"""Read-only sample-aligned disagreement audit; never fills unknown decisions."""

import numpy as np
import pandas as pd

IDS = ["subject_id", "session_id", "task_id"]
RAW_IDS = ["subjectID", "sessionID", "taskID"]
EPS = 1e-6


def segments(time, mask):
    """Half-open sample-index runs, split at false samples or raw time gaps."""
    time, mask = np.asarray(time, float), np.asarray(mask, bool)
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    cuts = np.flatnonzero((np.diff(indices) != 1) | (np.diff(time[indices]) > 1.5 / 60)) + 1
    return [(int(part[0]), int(part[-1]) + 1) for part in np.split(indices, cuts)]


def timeline(time, predictions):
    """Map sample-aligned completed-window decisions to observed raw cells only."""
    time = np.asarray(time, float)
    if not len(time) or not np.isfinite(time).all() or (np.diff(time) <= 0).any():
        raise ValueError("Raw timestamps must be finite and increasing")
    state, score = np.full(len(time), -1, int), np.full(len(time), np.nan)
    ends = predictions.window_end_s.to_numpy(float)
    if not np.isfinite(ends).all() or (np.diff(ends) <= 0).any():
        raise ValueError("Prediction times must be finite and increasing")
    if not np.allclose(predictions.available_at_s, ends, atol=EPS, rtol=0):
        raise ValueError("Prediction backdated")
    if not np.array_equal(predictions.is_fog.astype(bool), predictions.fog_probability >= predictions.fog_threshold):
        raise ValueError("Score/threshold decisions differ")
    for i, row in enumerate(predictions.itertuples()):
        a = int(np.searchsorted(time, row.window_end_s - EPS))
        if a == len(time) or abs(time[a] - row.window_end_s) > EPS:
            raise ValueError("Audit requires decisions aligned to actual raw samples")
        stop = min(row.window_end_s + 0.5, ends[i + 1] if i + 1 < len(ends) else time[-1] + 1 / 60)
        b = int(np.searchsorted(time, stop - EPS))
        state[a:b], score[a:b] = int(row.is_fog), row.fog_probability
    return state, score


def span_seconds(time, mask):
    return sum(time[b - 1] + 1 / 60 - time[a] for a, b in segments(time, mask))


def longest_seconds(time, mask):
    return max((time[b - 1] + 1 / 60 - time[a] for a, b in segments(time, mask)), default=0.0)


def record_audit(raw, base, candidate):
    events, pieces, records, payload = [], [], [], []
    for key, signal in raw.groupby(RAW_IDS, sort=False):
        signal = signal.reset_index(drop=True)
        time, truth = signal.timestamp.to_numpy(float), signal.fog.to_numpy(bool)
        frames = []
        for frame in (base, candidate):
            keep = np.ones(len(frame), bool)
            for column, value in zip(IDS, key, strict=True):
                keep &= frame[column].astype(int).eq(int(value)).to_numpy()
            frames.append(frame.loc[keep].sort_values("window_end_s").reset_index(drop=True))
        states = [timeline(time, frame)[0] for frame in frames]
        a, b = states
        if not np.array_equal(a >= 0, b >= 0):
            raise ValueError("Matched models must have identical support")
        fields = dict(zip(IDS, map(int, key), strict=True))
        masks = {"lost_fog": truth & (a == 1) & (b == 0), "recovered_fog": truth & (a == 0) & (b == 1),
                 "removed_false_alarm": ~truth & (a == 1) & (b == 0), "new_false_alarm": ~truth & (a == 0) & (b == 1)}
        row = {**fields, "samples": len(signal), "windows": len(frames[0]), "truth_events": len(segments(time, truth)),
               "observed_s": span_seconds(time, np.ones(len(time), bool)), "support_s": span_seconds(time, a >= 0),
               "truth_s": span_seconds(time, truth), "base_true_s": span_seconds(time, truth & (a == 1)),
               "candidate_true_s": span_seconds(time, truth & (b == 1)),
               "base_false_s": span_seconds(time, ~truth & (a == 1)), "candidate_false_s": span_seconds(time, ~truth & (b == 1)),
               **{name + "_s": span_seconds(time, mask) for name, mask in masks.items()}}
        records.append(row)
        for kind, mask in masks.items():
            for start, stop in segments(time, mask):
                pieces.append({**fields, "kind": kind, "start_s": time[start], "end_s": time[stop - 1] + 1 / 60,
                    "duration_s": time[stop - 1] + 1 / 60 - time[start]})
        for number, (start, stop) in enumerate(segments(time, truth)):
            t0, t1 = time[start], time[stop - 1] + 1 / 60
            select = np.zeros(len(time), bool)
            select[start:stop] = True
            duration = t1 - t0
            event = {**fields, "event_in_record": number, "start_s": t0, "end_s": t1, "duration_s": duration,
                     "support_s": span_seconds(time, select & (a >= 0)),
                     "lost_fog_s": span_seconds(time, select & masks["lost_fog"]),
                     "recovered_fog_s": span_seconds(time, select & masks["recovered_fog"])}
            for name, state in zip(("base", "candidate"), states, strict=True):
                alarm = select & (state == 1)
                overlap = span_seconds(time, alarm)
                event[name + "_alarm_s"] = overlap
                event[name + "_recall"] = overlap / duration
                event[name + "_covered"] = overlap > EPS
                event[name + "_first_overlap_delay_s"] = float(time[np.flatnonzero(alarm)[0]] - t0) if alarm.any() else None
                event[name + "_longest_unalarmed_s"] = longest_seconds(time, select & (state != 1))
            start_zone = select & (time < min(t0 + 3, t1) - EPS)
            end_zone = select & ~start_zone & (time >= max(t1 - 3, t0) - EPS)
            interior = select & ~start_zone & ~end_zone
            for name, zone in (("onset", start_zone), ("interior", interior), ("ending", end_zone)):
                event[name + "_lost_s"] = span_seconds(time, zone & masks["lost_fog"])
                event[name + "_recovered_s"] = span_seconds(time, zone & masks["recovered_fog"])
            events.append(event)
        payload.append((fields, signal, frames, states))
    return pd.DataFrame(records), pd.DataFrame(events), pd.DataFrame(pieces), payload


def select_cases(events, pieces):
    result, selected = [], set()
    if len(events):
        work = events.assign(loss=events.base_alarm_s - events.candidate_alarm_s,
                             recall_drop=events.base_recall - events.candidate_recall)
        for kind, score in (("largest_fog_time_loss", "loss"), ("largest_other_recall_drop", "recall_drop")):
            ranked = work.sort_values([score, *IDS, "start_s"], ascending=[False, True, True, True, True])
            chosen = next((r for r in ranked.to_dict("records") if tuple(r[k] for k in [*IDS, "start_s"]) not in selected and r[score] > EPS), None)
            if chosen:
                selected.add(tuple(chosen[k] for k in [*IDS, "start_s"]))
                result.append({"case": kind, **{k: chosen[k] for k in [*IDS, "start_s", "end_s"]}})
    for kind in ("removed_false_alarm", "new_false_alarm"):
        ranked = pieces.loc[pieces.kind.eq(kind)].sort_values(["duration_s", *IDS, "start_s"], ascending=[False, True, True, True, True]) if len(pieces) else pieces
        if len(ranked):
            chosen = ranked.iloc[0]
            result.append({"case": "longest_" + kind, **{k: float(chosen[k]) for k in [*IDS, "start_s", "end_s"]}})
    return result

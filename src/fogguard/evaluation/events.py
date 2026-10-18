"""Event-level FoG evidence derived from labelled replay data."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fogguard.data.dataset import validate_frame
from fogguard.data.replay import WindowReplay
from fogguard.interfaces import FoGDetector


def labelled_fog_events(frame: pd.DataFrame) -> list[dict[str, float | int]]:
    """Return contiguous sample-level FoG events with inclusive end timestamps."""
    validated = validate_frame(frame)
    if "fog_label" not in validated.columns:
        raise ValueError("FoG event evaluation requires a fog_label column")

    labels = validated["fog_label"].to_numpy(dtype=int)
    timestamps = validated["timestamp_s"].to_numpy(dtype=float)
    padded = np.pad(labels, (1, 1), constant_values=0)
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1) - 1
    return [
        {
            "event_id": index + 1,
            "start_s": float(timestamps[start]),
            "end_s": float(timestamps[stop]),
        }
        for index, (start, stop) in enumerate(zip(starts, stops, strict=True))
    ]


def _delay_summary(delays: Sequence[float]) -> dict[str, float | None]:
    if not delays:
        return {
            "mean_detection_delay_s": None,
            "median_detection_delay_s": None,
            "max_detection_delay_s": None,
        }
    return {
        "mean_detection_delay_s": float(np.mean(delays)),
        "median_detection_delay_s": float(np.median(delays)),
        "max_detection_delay_s": float(np.max(delays)),
    }


def evaluate_fog_events(
    replay: WindowReplay,
    detector: FoGDetector,
    *,
    post_event_grace_s: float | None = None,
) -> dict[str, object]:
    """Match predicted positive windows to labelled FoG events.

    Detection time is the window output time (the final sample timestamp), while
    event start comes from the sample-level label. A grace period of one window
    is used by default to retain delayed outputs caused by windowing.
    """
    if post_event_grace_s is None:
        post_event_grace_s = replay.window_seconds
    if post_event_grace_s < 0:
        raise ValueError("post_event_grace_s must not be negative")

    events = labelled_fog_events(replay.frame)
    prediction_times = [
        float(window.timestamp_s[-1])
        for window in replay.windows()
        if detector.predict(window).is_fog
    ]
    used_predictions: set[int] = set()
    matched_events: list[dict[str, object]] = []
    delays: list[float] = []

    for event in events:
        start_s = float(event["start_s"])
        end_s = float(event["end_s"])
        candidate = next(
            (
                (index, timestamp)
                for index, timestamp in enumerate(prediction_times)
                if index not in used_predictions
                and start_s <= timestamp <= end_s + post_event_grace_s
            ),
            None,
        )
        if candidate is None:
            matched_events.append({**event, "detected": False, "first_detection_s": None, "delay_s": None})
            continue
        prediction_index, first_detection_s = candidate
        used_predictions.add(prediction_index)
        delay_s = first_detection_s - start_s
        delays.append(delay_s)
        matched_events.append(
            {
                **event,
                "detected": True,
                "first_detection_s": first_detection_s,
                "delay_s": delay_s,
            }
        )

    n_events = len(events)
    n_detected = len(delays)
    positive_windows_outside_events = sum(
        not any(
            float(event["start_s"])
            <= timestamp
            <= float(event["end_s"]) + post_event_grace_s
            for event in events
        )
        for timestamp in prediction_times
    )
    return {
        "n_events": n_events,
        "n_detected_events": n_detected,
        "n_missed_events": n_events - n_detected,
        "event_recall": float(n_detected / n_events) if n_events else 0.0,
        "n_positive_windows_outside_events": positive_windows_outside_events,
        **_delay_summary(delays),
        "events": matched_events,
    }

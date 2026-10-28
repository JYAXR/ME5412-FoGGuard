"""Optional causal consecutive-score gate, separate from raw model is_fog."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventPolicy:
    threshold: float
    confirmation_windows: int = 1

    def __post_init__(self):
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("Invalid policy threshold")
        if self.confirmation_windows not in (1, 2):
            raise ValueError("Confirmation must be 1 or 2")


class EventGate:
    """One state per input stream; confirmation never labels an earlier window."""

    def __init__(self, policy):
        self.policy = policy
        self.previous = None
        self.count = 0

    def push(self, score, key, end_s):
        if not np.isfinite([score, end_s]).all() or not 0 <= score <= 1:
            raise ValueError("Invalid event score/time")
        key = tuple(str(v) for v in key)
        if self.previous is not None:
            old_key, old_end = self.previous
            if key == old_key and end_s <= old_end:
                raise ValueError("Event windows must be chronological without duplicates")
            if key != old_key or not np.isclose(end_s - old_end, 0.5, atol=1e-6, rtol=0):
                self.count = 0
        self.count = self.count + 1 if score >= self.policy.threshold else 0
        self.previous = key, end_s
        return self.count >= self.policy.confirmation_windows


def apply_event_policy(frame, policy):
    """Scores in, raw threshold decision plus separate event decision out."""
    result = frame.copy()
    gate = EventGate(policy)
    decisions = [gate.push(float(r.fog_probability), (r.subject_id, r.session_id, r.task_id),
                           float(r.window_end_s)) for r in result.itertuples()]
    result["fog_threshold"] = policy.threshold
    result["is_fog"] = result.fog_probability.ge(policy.threshold)
    result["event_fog"] = decisions
    result["event_confirmation_windows"] = policy.confirmation_windows
    result["event_policy_version"] = "a1_event_v1"
    return result

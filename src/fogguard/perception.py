"""Causal A1 + current-direction adapter, independent of truth and cue control."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from fogguard.context.fogstar import (
    FogEventPhase,
    FogStarIntegrationConfig,
    FogStarTemporalIntegrator,
    _parse_boolean,
)
from fogguard.types import TurnDirection

KEYS = ["subject_id", "session_id", "task_id"]
TIMES = ["window_start_s", "window_end_s", "available_at_s"]
COMMON = KEYS + TIMES + ["model_version", "prediction_source"]
FOG = ["fog_probability", "fog_threshold", "is_fog"]
DIRECTION = ["direction", "direction_confidence", "direction_task", "horizon_s"]


@dataclass(frozen=True)
class PerceptionConfig:
    """Source lifetime starts at input end, including any computation delay."""

    source_max_hold_s: float = 0.5
    events: FogStarIntegrationConfig = field(default_factory=lambda: FogStarIntegrationConfig(
        turn_horizon_s=0.0,
    ))

    def __post_init__(self):
        if not np.isfinite(self.source_max_hold_s) or not 0 < self.source_max_hold_s <= 0.5:
            raise ValueError("Source lifetime must be positive and at most 0.5 seconds")
        if self.events.use_yaw_agreement_fallback or self.events.suppress_stationary_fog:
            raise ValueError("Fusion accepts model results only, without motion/yaw substitution")
        if self.events.turn_horizon_s != 0:
            raise ValueError("Current-direction fusion requires zero turn horizon")


def _identifier(value):
    if pd.isna(value) or not str(value).strip():
        raise ValueError("Every source row requires non-missing record keys")
    if isinstance(value, (int, float, np.number)):
        if not np.isfinite(value):
            raise ValueError("Record keys must be finite")
        if float(value).is_integer():
            return str(int(value))
    return str(value).strip()


def _prepare(frame, kind):
    fields = FOG if kind == "fog" else DIRECTION
    missing = set(COMMON + fields) - set(frame)
    if missing:
        raise ValueError(f"{kind} source missing columns: {sorted(missing)}")
    # Whitelist runtime fields. Annotation columns never reach the state machine.
    frame = frame[COMMON + fields + (["input_valid"] if "input_valid" in frame else [])].copy()
    for name in KEYS:
        frame[name] = frame[name].map(_identifier)
    for name in TIMES:
        frame[name] = pd.to_numeric(frame[name], errors="raise").round(9)
    if not np.isfinite(frame[TIMES].to_numpy(float)).all():
        raise ValueError("Source timestamps must be finite")
    if ((frame.window_end_s <= frame.window_start_s)
            | (frame.available_at_s < frame.window_end_s)).any():
        raise ValueError("A prediction cannot be available before its complete input window")
    if "input_valid" not in frame:
        frame["input_valid"] = True
    frame["input_valid"] = frame.input_valid.map(lambda v: _parse_boolean(v, "input_valid"))
    for name in ("model_version", "prediction_source"):
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Source requires explicit {name}")
    numbers = ["fog_probability", "fog_threshold"] if kind == "fog" else ["direction_confidence"]
    for name in numbers:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        values = frame.loc[frame.input_valid, name]
        if not np.isfinite(values).all() or not values.between(0, 1).all():
            raise ValueError(f"{name} must be finite and between zero and one")
    if kind == "fog":
        frame["is_fog"] = frame.is_fog.map(lambda v: _parse_boolean(v, "is_fog"))
    else:
        if not frame.direction.isin([d.value for d in TurnDirection]).all():
            raise ValueError("Direction must be straight, left, right or unknown")
        if (not frame.direction_task.eq("current").all()
                or not pd.to_numeric(frame.horizon_s, errors="raise").eq(0).all()):
            raise ValueError("Preturn outputs cannot be used as current direction")
    groups = {}
    for key, group in frame.groupby(KEYS, sort=False):
        if ((group.available_at_s.diff().dropna() <= 0).any()
                or (group.window_end_s.diff().dropna() <= 0).any()):
            raise ValueError("Each source must have strictly increasing availability and window ends")
        groups[key] = group.to_dict("records")
    return groups


class _PerceptionEvents(FogStarTemporalIntegrator):
    """Reuse event/hysteresis rules; advance counters only on new observations.

    No cue controller is called here. Source expiry invalidates the affected
    state independently; it never supplies a negative FoG observation.
    """

    def __init__(self, config):
        super().__init__(config)
        self.reported_turn = TurnDirection.UNKNOWN

    def step(self, fog, direction, fog_new, direction_new):
        if fog is None:
            self._phase = FogEventPhase.SAFE_NONE
            self._positive_count = self._negative_count = 0
            self._active_event_id = None
            reason = "fog_unavailable"
        elif fog_new:
            entry = fog["is_fog"] and fog["fog_probability"] >= fog["fog_threshold"]
            hold = entry or (
                self._phase in {FogEventPhase.CONFIRMED, FogEventPhase.RECOVERY}
                and fog["fog_probability"] >= max(
                    0, fog["fog_threshold"] - self.config.exit_probability_margin,
                )
            )
            reason = self._advance_event(entry, hold)
        else:
            reason = "held_fog_observation"
        if direction is None:
            self.reported_turn = self._smooth_direction(TurnDirection.UNKNOWN)
        elif direction_new:
            candidate = self._map_direction(direction["direction"], direction["direction_confidence"])
            self.reported_turn = self._smooth_direction(candidate)
        turn = self.reported_turn
        return {
            "event_phase": self._phase.value, "event_id": self._active_event_id,
            "stable_fog_detected": self._phase is FogEventPhase.CONFIRMED,
            "turn": turn.value,
            "turn_confidence": (direction["direction_confidence"]
                                if direction is not None and turn is not TurnDirection.UNKNOWN else 0),
            "turn_horizon_s": 0.0, "integration_reason": reason,
        }


def fuse_perception(fog, direction, *, clock=None, config=None):
    """Fuse independent ordered streams on availability and expiry ticks.

    A result is usable iff available_at <= now < input_end + source lifetime.
    An optional clock (record keys + timestamp_s) exposes unsupported time and
    records. Without it, output covers the union of source arrivals/expiries.
    Repeated held observations do not count toward event/direction confirmation.
    """
    config = config or PerceptionConfig()
    sources = {"fog": _prepare(fog, "fog"), "direction": _prepare(direction, "direction")}
    clocks = {}
    if clock is not None:
        if not set(KEYS + ["timestamp_s"]) <= set(clock):
            raise ValueError("Clock requires record keys and timestamp_s")
        clock = clock[KEYS + ["timestamp_s"]].copy()
        for name in KEYS:
            clock[name] = clock[name].map(_identifier)
        clock["timestamp_s"] = pd.to_numeric(clock.timestamp_s, errors="raise").round(9)
        if not np.isfinite(clock.timestamp_s).all():
            raise ValueError("Clock timestamps must be finite")
        for key, group in clock.groupby(KEYS, sort=False):
            if (group.timestamp_s.diff().dropna() <= 0).any():
                raise ValueError("Clock must be strictly increasing within a record")
            clocks[key] = group.timestamp_s.tolist()
    keys = dict.fromkeys([*clocks, *sources["fog"], *sources["direction"]])
    output = []
    next_event_id = 1
    for key in keys:
        state = _PerceptionEvents(config.events)
        state._next_event_id = next_event_id
        records = {name: groups.get(key, []) for name, groups in sources.items()}
        ticks = set(clocks.get(key, []))
        for rows in records.values():
            for row in rows:
                ticks.add(row["available_at_s"])
                # An excessively delayed row is already stale on arrival.
                ticks.add(max(row["available_at_s"],
                              round(row["window_end_s"] + config.source_max_hold_s, 9)))
        indices = dict.fromkeys(records, 0)
        latest = dict.fromkeys(records)
        for now in sorted(ticks):
            item = {**dict(zip(KEYS, key, strict=True)), "available_at_s": now}
            usable, new = {}, {}
            for name, rows in records.items():
                arrived = False
                while indices[name] < len(rows) and rows[indices[name]]["available_at_s"] <= now:
                    latest[name] = rows[indices[name]]
                    indices[name] += 1
                    arrived = True
                row = latest[name]
                age = round(now - row["window_end_s"], 9) if row is not None else np.nan
                fresh = bool(row is not None and row["input_valid"]
                             and 0 <= age < config.source_max_hold_s)
                usable[name] = row if fresh else None
                new[name] = arrived and fresh
                item.update({f"{name}_input_fresh": fresh, f"{name}_observation_is_new": new[name],
                             f"{name}_input_age_s": age,
                             f"{name}_status": ("missing" if row is None else "invalid"
                                                if not row["input_valid"] else "fresh" if fresh
                                                else "stale")})
                for column in TIMES + ["model_version", "prediction_source"]:
                    item[f"{name}_{column}"] = row[column] if row is not None else None
            f, d = usable["fog"], usable["direction"]
            item.update({
                # B1's legacy clock alias is explicitly the decision time.
                "window_end_s": now,
                "fog_probability": f["fog_probability"] if f is not None else np.nan,
                "fog_threshold": f["fog_threshold"] if f is not None else np.nan,
                "is_fog": f["is_fog"] if f is not None else None,
                "fog_state": "unknown" if f is None else "fog" if f["is_fog"] else "non_fog",
                "direction": d["direction"] if d is not None else "unknown",
                "direction_confidence": d["direction_confidence"] if d is not None else 0.0,
                "input_fresh": f is not None and d is not None,
                "input_age_s": max(item["fog_input_age_s"], item["direction_input_age_s"])
                if f is not None and d is not None else np.nan,
                **state.step(f, d, new["fog"], new["direction"]),
            })
            output.append(item)
        next_event_id = state._next_event_id
    if not output:
        raise ValueError("No source results or clock ticks to fuse")
    result = pd.DataFrame(output)
    result["event_id"] = result.event_id.astype("Int64")
    return result


def summarize_perception(frame, config=None):
    """Coverage/behavior only: no accuracy is inferred without separate truth."""
    return {
        "schema": "fogguard.perception.v1", "config": asdict(config or PerceptionConfig()),
        "records": len(frame[KEYS].drop_duplicates()), "decision_rows": len(frame),
        "new_fog_observations": int(frame.fog_observation_is_new.sum()),
        "new_direction_observations": int(frame.direction_observation_is_new.sum()),
        "confirmed_events": int(frame.event_id.nunique()),
        "fog_status_rows": frame.fog_status.value_counts().to_dict(),
        "direction_status_rows": frame.direction_status.value_counts().to_dict(),
        "turn_rows": frame.turn.value_counts().to_dict(),
        "scope": "Unweighted decision-tick behavior counts; not accuracy or clinical efficacy",
    }

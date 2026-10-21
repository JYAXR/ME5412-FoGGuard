"""Proposed guidance contracts, local to cueing pending the #16 team review."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GuidanceConfig:
    """Deterministic simulation settings; these are not measured patient parameters."""

    step_length_m: float = 0.40
    step_width_m: float = 0.20
    turn_radius_m: float = 0.80
    target_count: int = 4
    first_foot: str = "left"
    cadence_bpm: float = 80.0
    volume: float = 0.60
    opacity: float = 1.0
    minimum_direction_confidence: float = 0.60
    maximum_input_age_s: float = 1.0
    maximum_window_gap_s: float = 0.75
    refractory_seconds: float = 3.0
    maximum_episode_s: float = 15.0
    enable_response_policy: bool = True
    response_timeout_s: float = 3.0
    reset_duration_s: float = 3.0
    minimum_response_reliability: float = 0.80
    minimum_forward_m: float = 0.15
    maximum_step_gap_s: float = 1.50
    stable_steps: int = 3
    enable_fading: bool = True
    fade_seconds: float = 1.0
    enable_adaptive_spacing: bool = False
    minimum_step_length_m: float = 0.32
    maximum_step_length_m: float = 0.55
    maximum_step_adjustment_m: float = 0.03

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.type in ("bool", bool):
                if not isinstance(value, bool):
                    raise ValueError(f"{item.name} must be a boolean")
            elif item.type in ("float", float, "int", int):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{item.name} must be numeric")
                if not math.isfinite(value):
                    raise ValueError(f"{item.name} must be finite")
        for name in ("target_count", "stable_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "step_length_m", "step_width_m", "turn_radius_m", "cadence_bpm",
            "maximum_window_gap_s", "maximum_episode_s", "response_timeout_s",
            "reset_duration_s", "minimum_forward_m", "maximum_step_gap_s", "fade_seconds",
            "minimum_step_length_m", "maximum_step_length_m", "maximum_step_adjustment_m",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("volume", "opacity", "minimum_direction_confidence",
                     "minimum_response_reliability"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.maximum_input_age_s < 0 or self.refractory_seconds < 0:
            raise ValueError("Freshness and refractory limits must be non-negative")
        if not self.minimum_step_length_m <= self.step_length_m <= self.maximum_step_length_m:
            raise ValueError("step_length_m must be inside the configured personal range")
        if self.first_foot not in {"left", "right"}:
            raise ValueError("first_foot must be left or right")

    @classmethod
    def load(cls, path: str | Path) -> GuidanceConfig:
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise TypeError("Guidance config must be a YAML mapping")
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown guidance settings: {sorted(unknown)}")
        return cls(**values)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GuidanceInput:
    now_s: float
    sequence: tuple[str, ...]
    event_id: str | None
    event_phase: str
    stable_fog_detected: bool
    turn: str
    turn_confidence: float
    input_fresh: bool
    input_age_s: float = 0.0
    quality_reason: str = ""


@dataclass(frozen=True)
class StepResponse:
    timestamp_s: float
    foot: str
    forward_m: float | None
    lateral_m: float | None
    response_status: str
    reliability: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("Response availability time must be finite and non-negative")
        if not math.isfinite(self.reliability) or not 0 <= self.reliability <= 1:
            raise ValueError("Response reliability must be between 0 and 1")
        if self.response_status not in {"success", "delayed", "unable", "no_response",
                                        "missing_input"}:
            raise ValueError("Unsupported response status")
        if self.foot not in {"left", "right", "none"}:
            raise ValueError("Response foot must be left, right or none")
        if self.response_status in {"success", "delayed"} and (
            self.foot == "none" or self.forward_m is None or self.lateral_m is None
            or not math.isfinite(self.forward_m) or not math.isfinite(self.lateral_m)
        ):
            raise ValueError("Completed responses require a foot and finite local offsets")


@dataclass(frozen=True)
class FootTarget:
    target_id: str
    foot: str
    x_m: float
    y_m: float
    yaw_deg: float


@dataclass(frozen=True)
class GuidanceCommand:
    schema_version: str
    timestamp_s: float
    command_id: str | None
    event_id: str | None
    action: str
    control_state: str
    reason: str
    visual_enabled: bool
    audio_enabled: bool
    visual_opacity: float
    audio_volume: float
    cadence_bpm: float
    audio_mode: str
    cue_started_at_s: float | None
    mode_started_at_s: float | None
    targets: tuple[FootTarget, ...]
    stable_steps: int
    response_source: str
    step_length_m: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

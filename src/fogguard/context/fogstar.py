"""Turn FoG-STAR window predictions into safe, auditable controller input."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from fogguard.cueing import AdaptiveCueingController
from fogguard.interfaces import CueingController
from fogguard.types import (
    ControllerContext,
    CueType,
    DetectionResult,
    MotionEstimate,
    MotionState,
    TurnDirection,
    TurnIntent,
)


class FogEventPhase(str, Enum):
    NORMAL = "normal"
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"
    RECOVERY = "recovery"
    SAFE_NONE = "safe_none"


@dataclass(frozen=True)
class FogStarIntegrationConfig:
    """Fixed engineering rules applied after the FoG-STAR model.

    Overlapping 2 s windows arrive every 0.5 s. Two consecutive entry
    candidates therefore add 0.5 s of persistence evidence; three negative
    windows provide a 1.0 s recovery observation after the first negative.
    """

    confirmation_windows: int = 2
    recovery_windows: int = 3
    direction_confirmation_windows: int = 2
    minimum_direction_confidence: float = 0.60
    minimum_fallback_confidence: float = 0.60
    use_yaw_agreement_fallback: bool = False
    suppress_stationary_fog: bool = False
    exit_probability_margin: float = 0.10
    maximum_input_age_s: float = 1.0
    maximum_window_gap_s: float = 0.75
    turn_horizon_s: float = 0.5

    def __post_init__(self) -> None:
        if self.confirmation_windows < 1 or self.recovery_windows < 1:
            raise ValueError("FoG confirmation and recovery windows must be positive")
        if self.direction_confirmation_windows < 1:
            raise ValueError("Direction confirmation windows must be positive")
        if not 0 <= self.minimum_direction_confidence <= 1:
            raise ValueError("minimum_direction_confidence must be between 0 and 1")
        if not 0 <= self.minimum_fallback_confidence <= 1:
            raise ValueError("minimum_fallback_confidence must be between 0 and 1")
        if not 0 <= self.exit_probability_margin <= 1:
            raise ValueError("exit_probability_margin must be between 0 and 1")
        if self.maximum_input_age_s < 0 or self.maximum_window_gap_s <= 0:
            raise ValueError("Freshness limits must be non-negative")


REQUIRED_PREDICTION_COLUMNS = {
    "window_start_s",
    "window_end_s",
    "fog_probability",
    "is_fog",
    "direction",
    "direction_confidence",
    "subject_id",
    "session_id",
}


def _parse_boolean(value: object, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
        return value.strip().lower() in {"true", "1"}
    raise ValueError(f"{field} must contain booleans")


class FogStarTemporalIntegrator:
    """Causal state machine between model windows and the cue controller."""

    def __init__(
        self,
        config: FogStarIntegrationConfig | None = None,
        controller: CueingController | None = None,
    ) -> None:
        self.config = config or FogStarIntegrationConfig()
        # Event confirmation owns the model threshold. The controller still
        # requires stable is_fog=True and applies its refractory period.
        self.controller = controller or AdaptiveCueingController(
            min_confidence=0.0,
            suppress_stationary_fog=self.config.suppress_stationary_fog,
        )
        self._next_event_id = 1
        self._reset_sequence()

    def _reset_sequence(self) -> None:
        self._phase = FogEventPhase.NORMAL
        self._positive_count = 0
        self._negative_count = 0
        self._active_event_id: int | None = None
        self._last_window_end_s: float | None = None
        self._last_cue_time_s: float | None = None
        self._stable_direction = TurnDirection.UNKNOWN
        self._pending_direction = TurnDirection.UNKNOWN
        self._pending_direction_count = 0

    def _map_direction(
        self,
        value: object,
        confidence: float,
        minimum_confidence: float | None = None,
    ) -> TurnDirection:
        required_confidence = (
            self.config.minimum_direction_confidence
            if minimum_confidence is None
            else minimum_confidence
        )
        if confidence < required_confidence:
            return TurnDirection.UNKNOWN
        try:
            direction = TurnDirection(str(value))
        except ValueError:
            return TurnDirection.UNKNOWN
        return direction

    def _smooth_direction(self, candidate: TurnDirection) -> TurnDirection:
        if candidate is TurnDirection.UNKNOWN:
            self._stable_direction = TurnDirection.UNKNOWN
            self._pending_direction = TurnDirection.UNKNOWN
            self._pending_direction_count = 0
            return TurnDirection.UNKNOWN

        if candidate is self._stable_direction:
            self._pending_direction = TurnDirection.UNKNOWN
            self._pending_direction_count = 0
            return self._stable_direction

        if candidate is self._pending_direction:
            self._pending_direction_count += 1
        else:
            self._pending_direction = candidate
            self._pending_direction_count = 1

        if self._pending_direction_count >= self.config.direction_confirmation_windows:
            self._stable_direction = candidate
            self._pending_direction = TurnDirection.UNKNOWN
            self._pending_direction_count = 0
            return candidate
        return TurnDirection.UNKNOWN

    def _advance_event(self, entry_positive: bool, hold_positive: bool) -> str:
        if self._phase in {FogEventPhase.NORMAL, FogEventPhase.SAFE_NONE}:
            if not entry_positive:
                self._phase = FogEventPhase.NORMAL
                return "no_fog_candidate"
            self._positive_count = 1
            if self.config.confirmation_windows == 1:
                self._confirm_event()
                return "fog_confirmed"
            self._phase = FogEventPhase.SUSPECTED
            return "fog_candidate_awaiting_confirmation"

        if self._phase is FogEventPhase.SUSPECTED:
            if not entry_positive:
                self._phase = FogEventPhase.NORMAL
                self._positive_count = 0
                return "fog_candidate_rejected"
            self._positive_count += 1
            if self._positive_count >= self.config.confirmation_windows:
                self._confirm_event()
                return "fog_confirmed"
            return "fog_candidate_awaiting_confirmation"

        if self._phase is FogEventPhase.CONFIRMED:
            if hold_positive:
                return "fog_event_continues"
            self._phase = FogEventPhase.RECOVERY
            self._negative_count = 1
            return "fog_recovery_observation"

        if hold_positive:  # RECOVERY: merge a short score dip into the same event.
            self._phase = FogEventPhase.CONFIRMED
            self._negative_count = 0
            return "fog_reconfirmed_same_event"
        self._negative_count += 1
        if self._negative_count >= self.config.recovery_windows:
            self._phase = FogEventPhase.NORMAL
            self._negative_count = 0
            self._positive_count = 0
            self._active_event_id = None
            return "fog_recovery_completed"
        return "fog_recovery_observation"

    def _confirm_event(self) -> None:
        self._phase = FogEventPhase.CONFIRMED
        self._positive_count = 0
        self._negative_count = 0
        if self._active_event_id is None:
            self._active_event_id = self._next_event_id
            self._next_event_id += 1

    def process(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Process ordered model predictions without using evaluation labels."""
        missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
        if missing:
            raise ValueError(f"FoG-STAR predictions are missing required columns: {missing}")
        if predictions.empty:
            return predictions.copy()

        result_rows: list[dict[str, object]] = []
        previous_key: tuple[str, ...] | None = None
        for source_row in predictions.to_dict(orient="records"):
            row = dict(source_row)
            start_s = float(row["window_start_s"])
            end_s = float(row["window_end_s"])
            probability = float(row["fog_probability"])
            direction_confidence = float(row["direction_confidence"])
            model_positive = _parse_boolean(row["is_fog"], "is_fog")
            if (
                not np.isfinite([start_s, end_s, probability, direction_confidence]).all()
                or end_s <= start_s
                or not 0 <= probability <= 1
                or not 0 <= direction_confidence <= 1
            ):
                raise ValueError("Prediction times, probabilities, and confidences must be valid")

            key_names = ["subject_id", "session_id"]
            if "task_id" in row:
                key_names.append("task_id")
            key = tuple(str(row[name]) for name in key_names)
            if previous_key is not None and key != previous_key:
                self._reset_sequence()
            previous_key = key

            received_at_s = float(row.get("received_at_s", end_s))
            input_age_s = received_at_s - end_s
            stale = (
                not np.isfinite(received_at_s)
                or input_age_s < -1e-9
                or input_age_s > self.config.maximum_input_age_s
            )
            gap = (
                self._last_window_end_s is not None
                and end_s - self._last_window_end_s > self.config.maximum_window_gap_s
            )
            reversed_time = self._last_window_end_s is not None and end_s <= self._last_window_end_s
            if reversed_time:
                raise ValueError("FoG-STAR prediction windows must be strictly ordered within a task")

            if stale or gap:
                self._reset_sequence()
                self._phase = FogEventPhase.SAFE_NONE
                reason = "stale_input" if stale else "input_gap"
                turn = TurnDirection.UNKNOWN
                turn_confidence = 0.0
                motion = None
                fog_suppressed = False
            else:
                motion = None
                if "motion_state" in row:
                    try:
                        motion_state = MotionState(str(row["motion_state"]))
                    except ValueError as exc:
                        raise ValueError("motion_state must be walking, stationary, or unknown") from exc
                    motion = MotionEstimate(
                        state=motion_state,
                        confidence=float(row.get("motion_confidence", 0.0)),
                        dynamic_accel_rms_mps2=float(
                            row.get("dynamic_accel_rms_mps2", float("nan"))
                        ),
                    )
                raw_direction_value = str(row["direction"])
                raw_direction = self._map_direction(raw_direction_value, direction_confidence)
                direction_source = "fogstar_model"
                if raw_direction is TurnDirection.UNKNOWN:
                    direction_source = "rejected"
                    fallback_value = str(row.get("yaw_turn", "unknown"))
                    fallback_confidence = float(row.get("yaw_turn_confidence", 0.0))
                    # FoG-STAR `other` is an explicit rejection and is never
                    # overridden. A known but low-confidence model class can
                    # be accepted only when the independent yaw baseline agrees.
                    if (
                        self.config.use_yaw_agreement_fallback
                        and raw_direction_value != "other"
                    ):
                        fallback_direction = self._map_direction(
                            fallback_value,
                            fallback_confidence,
                            self.config.minimum_fallback_confidence,
                        )
                        if fallback_direction.value == raw_direction_value:
                            raw_direction = fallback_direction
                            direction_confidence = fallback_confidence
                            direction_source = "yaw_agreement_fallback"
                turn = self._smooth_direction(raw_direction)
                turn_confidence = direction_confidence if turn is not TurnDirection.UNKNOWN else 0.0
                threshold = float(row.get("fog_threshold", 0.5))
                if not 0 <= threshold <= 1:
                    raise ValueError("fog_threshold must be between 0 and 1")
                exit_threshold = max(0.0, threshold - self.config.exit_probability_margin)
                fog_suppressed = (
                    self.config.suppress_stationary_fog
                    and motion is not None
                    and motion.state is MotionState.STATIONARY
                    and model_positive
                )
                if (
                    self.config.suppress_stationary_fog
                    and motion is not None
                    and motion.state is MotionState.STATIONARY
                ):
                    turn = TurnDirection.UNKNOWN
                    turn_confidence = 0.0
                    direction_source = "stationary_rejection"
                entry_positive = model_positive and probability >= threshold and not fog_suppressed
                hold_positive = entry_positive or (
                    self._phase in {FogEventPhase.CONFIRMED, FogEventPhase.RECOVERY}
                    and probability >= exit_threshold
                    and not fog_suppressed
                )
                reason = self._advance_event(entry_positive, hold_positive)
                if fog_suppressed:
                    reason = "stationary_state_suppresses_fog"

            self._last_window_end_s = end_s
            stable_fog = self._phase is FogEventPhase.CONFIRMED
            seconds_since_last_cue = (
                None if self._last_cue_time_s is None else end_s - self._last_cue_time_s
            )
            cue = self.controller.decide(
                ControllerContext(
                    now_s=end_s,
                    fog=DetectionResult(
                        probability=probability,
                        is_fog=stable_fog,
                        model_name="fogstar-temporal-integration",
                        evidence={"raw_is_fog": float(model_positive)},
                    ),
                    turn=TurnIntent(
                        direction=turn,
                        confidence=turn_confidence,
                        horizon_s=self.config.turn_horizon_s,
                    ),
                    seconds_since_last_cue=seconds_since_last_cue,
                    motion=motion,
                )
            )
            if cue.cue_type is not CueType.NONE:
                self._last_cue_time_s = end_s

            row.update(
                {
                    "input_fresh": not stale and not gap,
                    "input_age_s": input_age_s,
                    "event_phase": self._phase.value,
                    "event_id": self._active_event_id,
                    "stable_fog_detected": stable_fog,
                    "fog_suppressed": fog_suppressed,
                    "effective_model_fog": model_positive and not fog_suppressed,
                    "turn": turn.value,
                    "turn_confidence": turn_confidence,
                    "turn_source": direction_source if not (stale or gap) else "unavailable",
                    "turn_horizon_s": self.config.turn_horizon_s,
                    "integration_reason": reason,
                    "seconds_since_last_cue": (
                        seconds_since_last_cue if seconds_since_last_cue is not None else np.nan
                    ),
                    "cue": cue.cue_type.value,
                    "cue_intensity": cue.intensity,
                    "cue_duration_s": cue.duration_s,
                    "cue_reason": cue.reason,
                }
            )
            result_rows.append(row)

        result = pd.DataFrame(result_rows)
        if "event_id" in result:
            result["event_id"] = result["event_id"].astype("Int64")
        return result

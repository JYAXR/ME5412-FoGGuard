"""Causal event-driven visual/audio guidance; consumes Robot 2's stable events."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace

from fogguard.cueing.contracts import (
    GuidanceCommand,
    GuidanceConfig,
    GuidanceInput,
    StepResponse,
)
from fogguard.cueing.targets import make_targets


class GuidanceController:
    """One chronological stream. Event inference remains owned by Robot 2.

    `start` starts one synchronized episode, `hold` never restarts playback,
    `update` changes an existing episode, and `stop` cancels both modalities.
    Responses use timestamp_s as availability time and local per-step offsets.
    """

    def __init__(
        self, config: GuidanceConfig | None = None, *, responses_configured: bool = False,
    ) -> None:
        self.config = config or GuidanceConfig()
        self.responses_configured = responses_configured
        self._sequence: tuple[str, ...] | None = None
        self._last_time: float | None = None
        self._last_start: float | None = None
        self._last_phase = "normal"
        self._counter = 0
        self._blocked_events: set[str] = set()
        self._requires_rearm = False
        self._rearm_count = 0
        self._response_usable = responses_configured and self.config.enable_response_policy
        self._last_response_seen: float | None = None
        self._clear_episode()

    def _clear_episode(self) -> None:
        self._command_id: str | None = None
        self._event_id: str | None = None
        self._started: float | None = None
        self._state = "IDLE"
        self._targets = ()
        self._target_index = 0
        self._step_length = self.config.step_length_m
        self._steps = 0
        self._last_step: float | None = None
        self._last_foot: str | None = None
        self._recent_lengths: list[float] = []
        self._reset_started: float | None = None
        self._reset_used = False
        self._fade_started: float | None = None
        self._previous_output: GuidanceCommand | None = None

    def _clear_recovery_evidence(self) -> None:
        self._steps = 0
        self._recent_lengths.clear()
        self._last_step = self._last_foot = self._fade_started = None

    def _stop(self, now: float, reason: str, state: str = "IDLE") -> GuidanceCommand:
        result = GuidanceCommand(
            schema_version="guidance-proposal-v1", timestamp_s=now,
            command_id=self._command_id, event_id=self._event_id,
            action="stop" if self._command_id else "none", control_state=state, reason=reason,
            visual_enabled=False, audio_enabled=False, visual_opacity=0.0, audio_volume=0.0,
            cadence_bpm=self.config.cadence_bpm, audio_mode="off",
            cue_started_at_s=self._started, mode_started_at_s=None, targets=(),
            stable_steps=self._steps, response_source=self._response_source(),
            step_length_m=self._step_length,
        )
        self._clear_episode()
        return result

    def _response_source(self) -> str:
        if not self.responses_configured:
            return "not_configured"
        if not self.config.enable_response_policy:
            return "simulated_disabled"
        return "simulated" if self._response_usable else "unavailable_fixed_fallback"

    def _fault(self, now: float, reason: str) -> GuidanceCommand:
        self._requires_rearm = True
        self._rearm_count = 0
        self._last_phase = "safe_none"
        return self._stop(now, reason, "SAFE_NONE")

    def _start(self, item: GuidanceInput) -> None:
        self._clear_episode()
        self._counter += 1
        key = hashlib.sha256(json.dumps(item.sequence).encode()).hexdigest()[:12]
        self._command_id = f"guidance-{key}-{self._counter}"
        self._event_id = item.event_id
        self._started = self._last_start = item.now_s
        self._state = "GUIDING"
        self._targets = make_targets(
            self.config, command_id=self._command_id, first_index=0,
            first_foot=self.config.first_foot, direction=item.turn,
            step_length_m=self._step_length,
        )

    def _accept_responses(self, item: GuidanceInput, responses: tuple[StepResponse, ...]) -> str:
        reason = ""
        if not self.config.enable_response_policy:
            return reason
        for response in responses:
            if self._started is None or response.timestamp_s < self._started:
                continue
            if item.now_s - response.timestamp_s > self.config.maximum_step_gap_s:
                reason = "stale_response_ignored"
                continue
            if response.response_status == "missing_input" or (
                response.reliability < self.config.minimum_response_reliability
            ):
                self._response_usable = False
                self._clear_recovery_evidence()
                self._state = "GUIDING"
                reason = "response_unavailable_fixed_fallback"
                continue
            self._response_usable = True
            if response.response_status in {"no_response", "unable"}:
                self._clear_recovery_evidence()
                if not self._reset_used:
                    self._state = "RESET_CUE"
                    self._reset_used = True
                    self._reset_started = item.now_s
                    reason = f"simulated_{response.response_status}_countdown"
                else:
                    if self._state in {"FADING", "RECOVERING"}:
                        self._state = "GUIDING"
                    reason = f"simulated_{response.response_status}_fixed_guidance"
                continue
            if response.response_status not in {"success", "delayed"}:
                continue
            if (
                not self._targets or response.foot != self._targets[0].foot
                or response.forward_m is None
                or response.forward_m < self.config.minimum_forward_m
            ):
                self._clear_recovery_evidence()
                if self._state in {"FADING", "RECOVERING"}:
                    self._state = "GUIDING"
                reason = "response_not_expected_forward_step"
                continue
            consecutive = (
                self._last_step is not None and self._last_foot != response.foot
                and 0 < response.timestamp_s - self._last_step <= self.config.maximum_step_gap_s
            )
            if not consecutive:
                self._steps = 0
                self._recent_lengths.clear()
            self._steps += 1
            self._last_step = response.timestamp_s
            self._last_foot = response.foot
            self._recent_lengths.append(response.forward_m)
            self._recent_lengths = self._recent_lengths[-self.config.stable_steps:]
            if self.config.enable_adaptive_spacing and self._steps >= self.config.stable_steps:
                desired = sum(self._recent_lengths) / len(self._recent_lengths)
                desired = min(self.config.maximum_step_length_m,
                              max(self.config.minimum_step_length_m, desired))
                delta = min(self.config.maximum_step_adjustment_m,
                            max(-self.config.maximum_step_adjustment_m, desired - self._step_length))
                self._step_length += delta
            landed = self._targets[0]
            yaw = math.radians(landed.yaw_deg)
            lateral = (1 if landed.foot == "left" else -1) * self.config.step_width_m / 2
            anchor = (landed.x_m + lateral * math.sin(yaw),
                      landed.y_m - lateral * math.cos(yaw), yaw)
            self._target_index += 1
            self._targets = make_targets(
                self.config, command_id=self._command_id, first_index=self._target_index,
                first_foot="right" if response.foot == "left" else "left",
                direction=item.turn, step_length_m=self._step_length, anchor=anchor,
            )
            # Re-anchor on the completed configured target, never claim measured global pose.
            if self._state == "RESET_CUE":
                self._state = "GUIDING"
            reason = "simulated_forward_step_accepted"
        return reason

    def step(
        self, item: GuidanceInput, responses: tuple[StepResponse, ...] = (),
    ) -> GuidanceCommand:
        now = item.now_s
        if not math.isfinite(now) or now < 0:
            return self._fault(self._last_time or 0.0, "invalid_timestamp")
        # Validate the entire incoming batch before changing the controller. A
        # rejected batch must not partially advance targets, health or clocks.
        if responses and not self.responses_configured:
            raise ValueError("Step responses require an explicitly configured simulated source")
        last_seen = self._last_response_seen if self._sequence == item.sequence else None
        for response in responses:
            if response.timestamp_s > now:
                raise ValueError("Future step responses must never be consumed")
            if last_seen is not None and response.timestamp_s <= last_seen:
                raise ValueError("Step responses must be consumed once, in time order")
            last_seen = response.timestamp_s
        if self._sequence != item.sequence:
            previous = self._command_id is not None
            stopped = self._stop(now, "sequence_changed") if previous else None
            self._sequence = item.sequence
            self._last_time = self._last_start = None
            self._last_response_seen = None
            self._blocked_events.clear()
            self._requires_rearm = False
            self._rearm_count = 0
            self._last_phase = "normal"
            self._response_usable = (
                self.responses_configured and self.config.enable_response_policy
            )
            if stopped is not None:
                self._last_time = now
                return stopped
        if self._last_time is not None and now <= self._last_time:
            return self._fault(self._last_time, "time_not_strictly_increasing")
        gap = self._last_time is not None and (
            now - self._last_time > self.config.maximum_window_gap_s
        )
        self._last_time = now
        self._last_response_seen = last_seen
        for response in responses:
            if self.config.enable_response_policy:
                # Source health survives episode stops, including reports that
                # arrive before the next confirmed event begins.
                self._response_usable = (
                    response.response_status != "missing_input"
                    and response.reliability >= self.config.minimum_response_reliability
                )
        if gap:
            return self._fault(now, "input_gap")
        if (
            not item.input_fresh or not math.isfinite(item.input_age_s)
            or not 0 <= item.input_age_s <= self.config.maximum_input_age_s
        ):
            return self._fault(now, item.quality_reason or "stale_input")
        if item.quality_reason:
            return self._fault(now, item.quality_reason)
        if (
            item.turn not in {"straight", "left", "right"}
            or not math.isfinite(item.turn_confidence)
            or not self.config.minimum_direction_confidence <= item.turn_confidence <= 1
        ):
            return self._fault(now, "direction_unavailable")
        if item.event_phase not in {"normal", "suspected", "confirmed", "recovery"}:
            return self._fault(now, "event_unavailable")
        confirmed = item.event_phase == "confirmed"
        if item.stable_fog_detected != confirmed or (
            item.event_phase in {"confirmed", "recovery"} and item.event_id is None
        ):
            return self._fault(now, "inconsistent_event_contract")
        previous_phase = self._last_phase
        self._last_phase = item.event_phase
        if self._requires_rearm:
            if item.event_phase in {"normal", "suspected"}:
                self._requires_rearm = False
                self._rearm_count = 0
                return self._stop(now, "waiting_upstream_confirmation")
            self._rearm_count = self._rearm_count + 1 if confirmed else 0
            if self._rearm_count < 2:
                return self._stop(now, "waiting_fresh_reconfirmation", "SAFE_NONE")
            self._requires_rearm = False
        if self._command_id is not None and confirmed and item.event_id != self._event_id:
            return self._stop(now, "event_changed")
        if self._command_id is not None and item.event_phase == "recovery" and (
            item.event_id != self._event_id
        ):
            return self._fault(now, "inconsistent_recovery_event")
        starting = False
        if self._command_id is None:
            if not confirmed:
                return self._stop(now, "no_confirmed_event")
            if item.event_id in self._blocked_events:
                return self._stop(now, "event_finished_waiting_new_event", "IDLE")
            if self._last_start is not None and now - self._last_start < (
                self.config.refractory_seconds
            ):
                return self._stop(now, "start_refractory")
            self._start(item)
            starting = True
        if now - self._started >= self.config.maximum_episode_s:
            self._blocked_events.add(self._event_id)
            return self._stop(now, "episode_timeout", "FAILED")

        reason = "confirmed_event_started" if starting else "same_event_maintained"
        if confirmed and previous_phase != "confirmed" and self._state in {
            "RECOVERING", "FADING",
        }:
            self._state = "GUIDING"
            self._steps = 0
            self._recent_lengths.clear()
            self._fade_started = None
            reason = "fog_reconfirmed_restore_intensity"
        response_reason = self._accept_responses(item, responses)
        if reason != "fog_reconfirmed_restore_intensity" or (
            response_reason == "response_unavailable_fixed_fallback"
        ):
            reason = response_reason or reason
        recent_step = self._response_usable and self._last_step is not None and (
            now - self._last_step <= self.config.maximum_step_gap_s
        )
        if item.event_phase in {"normal", "suspected"} and not (
            recent_step and self._steps > 0
        ):
            return self._stop(now, "detection_recovery_completed")
        if not confirmed:
            if recent_step and self._steps > 0:
                if self._state not in {"RECOVERING", "FADING"}:
                    self._state = "RECOVERING"
                    reason = "simulated_step_recovery"
                if self.config.enable_fading and self._steps >= self.config.stable_steps:
                    if self._fade_started is None:
                        self._fade_started = now
                        self._state = "FADING"
                        reason = "stable_simulated_steps_begin_fade"
                    if now - self._fade_started >= self.config.fade_seconds:
                        return self._stop(now, "simulated_recovery_fade_completed")
            elif self._state in {"RECOVERING", "FADING"}:
                return self._stop(now, "recovery_response_timeout")
            else:
                self._state = "EXIT_OBSERVE"
                reason = response_reason or "detection_recovery_observation"
        elif self._state == "EXIT_OBSERVE":
            self._state = "GUIDING"
            reason = "same_event_reconfirmed"

        if confirmed and self._response_usable and not self._reset_used:
            last_progress = self._last_step if self._last_step is not None else self._started
            if now - last_progress >= self.config.response_timeout_s:
                self._state = "RESET_CUE"
                self._reset_used = True
                self._reset_started = now
                reason = "simulated_response_timeout_countdown"
        if self._state == "RESET_CUE" and (
            now - self._reset_started >= self.config.reset_duration_s
        ):
            self._state = "GUIDING"
            reason = "countdown_completed_fixed_guidance"
        return self._emit(now, reason, starting)

    def finish(self) -> GuidanceCommand | None:
        """Explicitly cancel an unfinished command at replay end (a censored episode)."""
        if self._command_id is None:
            return None
        return self._stop(self._last_time or 0.0, "replay_ended")

    def _emit(self, now: float, reason: str, starting: bool) -> GuidanceCommand:
        factor = 1.0 if self._fade_started is None else max(
            0.0, 1 - (now - self._fade_started) / self.config.fade_seconds,
        )
        mode = "countdown_3_2_1_go" if self._state == "RESET_CUE" else "metronome"
        result = GuidanceCommand(
            schema_version="guidance-proposal-v1", timestamp_s=now,
            command_id=self._command_id, event_id=self._event_id, action="start",
            control_state=self._state, reason=reason,
            visual_enabled=True, audio_enabled=True,
            visual_opacity=self.config.opacity * factor, audio_volume=self.config.volume * factor,
            cadence_bpm=self.config.cadence_bpm, audio_mode=mode,
            cue_started_at_s=self._started,
            mode_started_at_s=self._reset_started if mode != "metronome" else self._started,
            targets=self._targets, stable_steps=self._steps,
            response_source=self._response_source(), step_length_m=self._step_length,
        )
        if not starting:
            old = self._previous_output
            changed = old is None or any(
                getattr(result, name) != getattr(old, name) for name in (
                    "control_state", "targets", "visual_opacity", "audio_volume", "audio_mode",
                )
            )
            result = replace(result, action="update" if changed else "hold")
        self._previous_output = result
        return result

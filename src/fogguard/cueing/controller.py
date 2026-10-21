"""Deterministic and auditable adaptive cueing policy."""

from dataclasses import dataclass

from fogguard.types import ControllerContext, CueCommand, CueType, MotionState, TurnDirection


@dataclass
class AdaptiveCueingController:
    min_confidence: float = 0.60
    refractory_seconds: float = 3.0
    duration_seconds: float = 2.0

    def decide(self, context: ControllerContext) -> CueCommand:
        if (
            context.motion is not None
            and context.motion.state is MotionState.STATIONARY
            and context.fog.is_fog
        ):
            return CueCommand(CueType.NONE, 0.0, 0.0, "Stationary state suppresses FoG cue")

        in_refractory = (
            context.seconds_since_last_cue is not None
            and context.seconds_since_last_cue < self.refractory_seconds
        )
        if (
            in_refractory
            or not context.fog.is_fog
            or context.fog.probability < self.min_confidence
        ):
            return CueCommand(CueType.NONE, 0.0, 0.0, "No confident FoG event or refractory")

        if context.turn.direction is TurnDirection.UNKNOWN:
            return CueCommand(CueType.NONE, 0.0, 0.0, "Turning context unavailable")

        if context.turn.direction in {TurnDirection.LEFT, TurnDirection.RIGHT}:
            return CueCommand(CueType.HAPTIC, 0.8, self.duration_seconds, "FoG during turn")
        return CueCommand(CueType.AUDITORY, 0.6, self.duration_seconds, "FoG during straight gait")

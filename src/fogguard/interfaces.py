"""Protocols that allow each sub-team to work against stable boundaries."""

from collections.abc import Iterable
from typing import Protocol

from fogguard.types import (
    ControllerContext,
    CueCommand,
    DetectionResult,
    MotionEstimate,
    SensorWindow,
    TurnIntent,
)


class DatasetReplay(Protocol):
    def windows(self) -> Iterable[SensorWindow]: ...


class FoGDetector(Protocol):
    def predict(self, window: SensorWindow) -> DetectionResult: ...


class TurnIntentRecognizer(Protocol):
    def predict(self, window: SensorWindow) -> TurnIntent: ...


class MotionStateRecognizer(Protocol):
    def predict(self, window: SensorWindow) -> MotionEstimate: ...


class CueingController(Protocol):
    def decide(self, context: ControllerContext) -> CueCommand: ...

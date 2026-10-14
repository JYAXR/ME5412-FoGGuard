"""Protocols that allow each sub-team to work against stable boundaries."""

from typing import Iterable, Protocol

from fogguard.types import (
    ControllerContext,
    CueCommand,
    DetectionResult,
    SensorWindow,
    TurnIntent,
)


class DatasetReplay(Protocol):
    def windows(self) -> Iterable[SensorWindow]: ...


class FoGDetector(Protocol):
    def predict(self, window: SensorWindow) -> DetectionResult: ...


class TurnIntentRecognizer(Protocol):
    def predict(self, window: SensorWindow) -> TurnIntent: ...


class CueingController(Protocol):
    def decide(self, context: ControllerContext) -> CueCommand: ...

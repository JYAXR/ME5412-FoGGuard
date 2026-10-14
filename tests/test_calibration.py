import pytest

from fogguard.data import WindowReplay
from fogguard.detection import FreezeIndexDetector
from fogguard.evaluation import select_fog_threshold, sweep_fog_thresholds
from scripts.generate_demo_data import build_scenario


def test_threshold_selection_uses_declared_tie_breaking_rule() -> None:
    reports = sweep_fog_thresholds(
        [
            WindowReplay(build_scenario("normal")),
            WindowReplay(build_scenario("straight_fog")),
        ],
        FreezeIndexDetector(),
        [0.15, 0.16, 0.21, 0.22],
    )

    selected = select_fog_threshold(reports)

    assert selected["threshold"] == pytest.approx(0.21)
    assert selected["false_positives"] == 0
    assert selected["false_negatives"] == 0
    assert selected["f1"] == pytest.approx(1.0)
    assert FreezeIndexDetector().threshold == pytest.approx(selected["threshold"])


def test_threshold_selection_rejects_unsatisfied_false_positive_constraint() -> None:
    reports = sweep_fog_thresholds(
        [WindowReplay(build_scenario("normal"))],
        FreezeIndexDetector(),
        [0.0],
    )

    with pytest.raises(ValueError, match="No threshold satisfies"):
        select_fog_threshold(reports)

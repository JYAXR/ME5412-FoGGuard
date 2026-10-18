import pytest

from fogguard.data import WindowReplay
from fogguard.detection import FreezeIndexDetector
from fogguard.evaluation import evaluate_fog_detector
from scripts.generate_demo_data import build_scenario


def test_evaluate_fog_detector_reports_window_counts_and_metrics() -> None:
    report = evaluate_fog_detector(
        WindowReplay(build_scenario("straight_fog")),
        FreezeIndexDetector(threshold=0.35),
    )

    assert report["n_windows"] == 57
    assert report["n_true_fog"] == 11
    assert report["n_predicted_fog"] == 7
    assert report["precision"] == pytest.approx(1.0)
    assert report["recall"] == pytest.approx(7 / 11)
    assert report["f1"] == pytest.approx(7 / 9)


def test_evaluate_fog_detector_requires_window_labels() -> None:
    frame = build_scenario("normal").drop(columns="fog_label")

    with pytest.raises(ValueError, match="requires every window"):
        evaluate_fog_detector(WindowReplay(frame), FreezeIndexDetector())


def test_evaluation_reports_specificity_and_balanced_accuracy() -> None:
    report = evaluate_fog_detector(
        WindowReplay(build_scenario("normal")),
        FreezeIndexDetector(threshold=0.35),
    )

    assert report["specificity"] == pytest.approx(1.0)
    assert report["balanced_accuracy"] == pytest.approx(0.5)

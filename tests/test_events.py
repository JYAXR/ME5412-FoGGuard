import pytest

from fogguard.data import WindowReplay
from fogguard.detection import FreezeIndexDetector
from fogguard.evaluation import evaluate_fog_events, labelled_fog_events
from scripts.generate_demo_data import build_scenario


def test_labelled_fog_events_use_sample_level_boundaries() -> None:
    events = labelled_fog_events(build_scenario("fog_recovery"))

    assert events == [{"event_id": 1, "start_s": 8.0, "end_s": pytest.approx(12.98)}]


def test_event_evaluation_reports_detection_delay() -> None:
    report = evaluate_fog_events(
        WindowReplay(build_scenario("straight_fog")),
        FreezeIndexDetector(),
    )

    assert report["n_events"] == 1
    assert report["n_detected_events"] == 1
    assert report["n_missed_events"] == 0
    assert report["event_recall"] == 1.0
    assert report["mean_detection_delay_s"] is not None
    assert report["events"][0]["first_detection_s"] >= 12.0


def test_event_evaluation_keeps_missed_events() -> None:
    report = evaluate_fog_events(
        WindowReplay(build_scenario("straight_fog")),
        FreezeIndexDetector(threshold=1.0),
    )

    assert report["n_events"] == 1
    assert report["n_missed_events"] == 1
    assert report["events"][0]["detected"] is False
    assert report["events"][0]["delay_s"] is None

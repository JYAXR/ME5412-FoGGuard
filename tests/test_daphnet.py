from pathlib import Path

import numpy as np
import pytest

from fogguard.data import DAPHNET_SAMPLE_RATE_HZ, WindowReplay, load_daphnet_segments
from fogguard.turning import YawRateTurnRecognizer


def _write_recording(path: Path, annotations: list[int]) -> None:
    rows = []
    for index, annotation in enumerate(annotations):
        timestamp_ms = round(index * 1000 / DAPHNET_SAMPLE_RATE_HZ)
        rows.append(
            " ".join(
                str(value)
                for value in (
                    timestamp_ms,
                    100,
                    200,
                    300,
                    400,
                    500,
                    600,
                    700,
                    800,
                    900,
                    annotation,
                )
            )
        )
    path.write_text("\n".join(rows), encoding="utf-8")


def test_daphnet_loader_maps_units_labels_and_protocol_segments(tmp_path: Path) -> None:
    path = tmp_path / "S01R01.txt"
    _write_recording(path, [0, 1, 2, 0, 1, 1])

    segments = load_daphnet_segments(path)

    assert len(segments) == 2
    assert segments[0]["fog_label"].tolist() == [0, 1]
    assert segments[1]["fog_label"].tolist() == [0, 0]
    assert segments[0]["subject_id"].unique().tolist() == ["S01"]
    assert segments[0]["gyro_available"].unique().tolist() == ["false"]
    assert segments[0]["gx"].eq(0.0).all()
    assert segments[0].iloc[0]["ax"] == pytest.approx(0.980665)
    assert segments[0].iloc[0]["ay"] == pytest.approx(2.941995)
    assert segments[0].iloc[0]["az"] == pytest.approx(1.96133)


def test_daphnet_provenance_reaches_windows_and_blocks_turning(tmp_path: Path) -> None:
    path = tmp_path / "S03R02.txt"
    _write_recording(path, [1] * 128)
    frame = load_daphnet_segments(path, sensor_position="trunk")[0]
    window = next(WindowReplay(frame, sample_rate_hz=DAPHNET_SAMPLE_RATE_HZ).windows())

    assert window.metadata["subject_id"] == "S03"
    assert window.metadata["recording_id"] == "S03R02"
    assert window.metadata["sensor_position"] == "trunk"
    assert window.metadata["source_dataset"] == "daphnet"
    assert window.metadata["source_license"] == "CC-BY-4.0"
    assert window.metadata["gyro_available"] == "false"
    assert np.count_nonzero(window.gyro_dps) == 0
    with pytest.raises(ValueError, match="requires measured gyroscope"):
        YawRateTurnRecognizer().predict(window)


def test_daphnet_loader_rejects_unknown_annotations(tmp_path: Path) -> None:
    path = tmp_path / "S02R01.txt"
    _write_recording(path, [1, 3])

    with pytest.raises(ValueError, match="annotation must contain only"):
        load_daphnet_segments(path)

from pathlib import Path

import pandas as pd
import pytest

from fogguard.data import load_step_responses, validate_step_responses
from scripts.generate_simulated_responses import (
    build_simulated_responses,
    write_simulated_responses,
)


def test_simulated_responses_cover_required_outcomes() -> None:
    frame = build_simulated_responses()

    assert set(frame["response_status"]) == {
        "success",
        "delayed",
        "unable",
        "no_response",
        "missing_input",
    }
    assert set(frame["source"]) == {"simulated"}


def test_written_simulated_responses_load(tmp_path: Path) -> None:
    path = tmp_path / "responses.csv"
    write_simulated_responses(path)

    loaded = load_step_responses(path)

    assert len(loaded) == 7
    assert loaded["reliability"].between(0.0, 1.0).all()


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("source", "measured", "source=simulated"),
        ("foot", "unknown", "foot must be"),
        ("reliability", 1.5, "reliability must be"),
    ],
)
def test_simulated_response_validation_rejects_invalid_values(
    column: str,
    value: object,
    message: str,
) -> None:
    frame = build_simulated_responses()
    frame.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        validate_step_responses(frame)


def test_completed_response_requires_landing_coordinates() -> None:
    frame = build_simulated_responses()
    frame.loc[0, "forward_m"] = pd.NA

    with pytest.raises(ValueError, match="finite landing coordinates"):
        validate_step_responses(frame)

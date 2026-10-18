"""Generate deterministic, explicitly simulated landing and response inputs."""

from pathlib import Path

import numpy as np
import pandas as pd

from fogguard.data.responses import validate_step_responses

DEFAULT_OUTPUT = Path("data/demo/simulated_step_responses.csv")


def build_simulated_responses() -> pd.DataFrame:
    """Build success, delay, failure, no-response, and missing-input cases."""
    records = [
        ("response_success", 12.5, "left", 0.32, 0.10, "success", 0.0, 1.0),
        ("response_success", 13.2, "right", 0.34, -0.10, "success", 0.0, 1.0),
        ("response_delayed", 14.0, "left", 0.28, 0.09, "delayed", 1.5, 0.9),
        ("response_delayed", 14.8, "right", 0.30, -0.09, "success", 0.0, 0.9),
        ("response_unable", 12.5, "none", np.nan, np.nan, "unable", np.nan, 1.0),
        ("response_none", 12.5, "none", np.nan, np.nan, "no_response", np.nan, 1.0),
        ("response_missing", 12.5, "none", np.nan, np.nan, "missing_input", np.nan, 0.0),
    ]
    frame = pd.DataFrame(
        records,
        columns=(
            "scenario_id",
            "timestamp_s",
            "foot",
            "forward_m",
            "lateral_m",
            "response_status",
            "delay_s",
            "reliability",
        ),
    )
    frame["source"] = "simulated"
    return validate_step_responses(frame)


def write_simulated_responses(destination: Path = DEFAULT_OUTPUT) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = build_simulated_responses()
    frame.to_csv(destination, index=False)
    print(f"Wrote {destination} ({len(frame)} simulated response rows)")


def main() -> None:
    write_simulated_responses()


if __name__ == "__main__":
    main()

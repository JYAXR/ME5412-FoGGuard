"""Run a trained FoG-STAR baseline on new same-layout multi-IMU data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fogguard.data import iter_fogstar_windows, load_fogstar_frame
from fogguard.modeling import FogStarMultiTaskModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/fogstar_multitask_baseline.joblib"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/new_gait_predictions.csv"))
    args = parser.parse_args()

    model = FogStarMultiTaskModel.load(args.model)
    sensor_positions = tuple(dict.fromkeys(name.split("_", 1)[0] for name in model.channel_names))
    frame = load_fogstar_frame(
        args.input,
        sensor_positions=sensor_positions,
        require_labels=False,
    )
    rows = []
    for window in iter_fogstar_windows(frame, sensor_positions=sensor_positions):
        rows.append(
            {
                "window_start_s": window.start_s,
                "window_end_s": window.end_s,
                "subject_id": window.subject_id,
                "session_id": window.session_id,
                **model.predict_window(window),
            }
        )
    if not rows:
        raise ValueError("No complete valid windows were found in the input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()

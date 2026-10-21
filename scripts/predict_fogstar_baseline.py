"""Run a trained FoG-STAR baseline on new same-layout multi-IMU data."""

from __future__ import annotations

import argparse
from pathlib import Path

from fogguard.data import load_fogstar_frame
from fogguard.fogstar_pipeline import predict_fogstar_frame
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
    result = predict_fogstar_frame(frame, model, sensor_positions=sensor_positions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()

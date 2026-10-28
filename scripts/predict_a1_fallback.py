"""Predict with frozen primary plus left-ankle availability fallback on new input."""

import argparse
from pathlib import Path

import pandas as pd

from fogguard.data.fogstar import iter_fogstar_windows
from fogguard.modeling.a1_fallback import route_available_predictions
from fogguard.modeling.a1_fog import A1FogModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    sensors = pd.read_csv(args.input).drop(columns=["fog", "activity", "fog_severity"], errors="ignore")
    predictions = {}
    for name in ("primary", "ankle"):
        model = A1FogModel.load(args.model_dir / f"{name}_model.joblib")
        if not set(model.channel_names) <= set(sensors.columns):
            predictions[name] = None
            continue
        positions = tuple(dict.fromkeys(n.split("_", 1)[0] for n in model.channel_names))
        windows = list(iter_fogstar_windows(sensors, sensor_positions=positions, max_interpolation_gap_s=0))
        predictions[name] = model.predict_windows(windows) if windows else None
    if all(v is None for v in predictions.values()):
        raise ValueError("No complete valid sensor windows; missing inputs are unknown")
    template = next(v for v in predictions.values() if v is not None)
    primary = predictions["primary"] if predictions["primary"] is not None else template.iloc[:0]
    ankle = predictions["ankle"] if predictions["ankle"] is not None else template.iloc[:0]
    result = route_available_predictions(primary, ankle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} sensor-only rows; fallback={int(result.sensor_route.eq('fallback_ankle').sum())}")


if __name__ == "__main__":
    main()

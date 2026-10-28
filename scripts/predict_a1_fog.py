"""Predict FoG on new-layout-compatible data; does not estimate direction or steps."""

import argparse
from pathlib import Path

from fogguard.data.fogstar import load_fogstar_frame
from fogguard.modeling.a1_fog import A1FogModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    model = A1FogModel.load(args.model)
    positions = tuple(dict.fromkeys(name.split("_", 1)[0] for name in model.channel_names))
    frame = load_fogstar_frame(args.input, sensor_positions=positions, require_labels=False)
    predictions = model.predict_frame(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Wrote {len(predictions)} FoG-only rows to {args.output}")


if __name__ == "__main__":
    main()

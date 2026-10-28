"""Apply one of the trained A1 multi-model arms to compatible sensor data."""

import argparse
from pathlib import Path

from fogguard.data.fogstar import load_fogstar_frame
from fogguard.modeling.a1_multimodel import MultiModelFoG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    model = MultiModelFoG.load(args.model)
    positions = tuple(dict.fromkeys(n.split("_", 1)[0] for n in model.channel_names))
    frame = load_fogstar_frame(args.input, sensor_positions=positions, require_labels=False)
    prediction = model.predict_frame(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(args.output, index=False)
    print(f"Wrote {len(prediction)} FoG-only rows")

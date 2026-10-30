"""Run a trusted A2 model bundle on unlabelled FoG-STAR-format IMU records."""

import argparse
from pathlib import Path

import pandas as pd

from fogguard.turning.learned import RAW_KEYS, DirectionModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a fresh filename")
    model = DirectionModel.load(args.model)
    frame = pd.read_csv(args.input, usecols=["timestamp", *RAW_KEYS, *model.channels])
    result = model.predict_frame(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} windows to {args.output}")


if __name__ == "__main__":
    main()

"""Use a frozen bilateral coordination FoG detector on sensor-only CSV data."""

import argparse
from pathlib import Path

import pandas as pd

from fogguard.modeling.a1_coordination import CoordinationFoG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    result = CoordinationFoG.load(args.model).predict_frame(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} FoG-only decisions; missing required gyro is unknown")


if __name__ == "__main__":
    main()

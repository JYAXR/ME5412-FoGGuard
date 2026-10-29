"""Predict preturn triggers from unlabelled IMU with a trusted binary model."""

import argparse
from pathlib import Path

import pandas as pd

from fogguard.data.fogstar import sensor_columns
from fogguard.turning.learned import RAW_KEYS
from fogguard.turning.preturn_binary import FEATURE_POSITIONS, BinaryPreturnModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output filename")
    model = BinaryPreturnModel.load(args.model)
    channels = sensor_columns(FEATURE_POSITIONS[model.feature_set])
    frame = pd.read_csv(args.input, usecols=["timestamp", *RAW_KEYS, *channels])
    output = model.predict_frame(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output)} windows")


if __name__ == "__main__":
    main()

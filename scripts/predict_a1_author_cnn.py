"""Run a saved gyro FoG model without activity/ground-truth or accelerometers."""

import argparse
from pathlib import Path

import pandas as pd

from fogguard.modeling.a1_author_cnn import GyroFoG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    model = GyroFoG.load(args.model)
    prediction = model.predict_frame(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(args.output, index=False)
    print(f"Saved {len(prediction)} FoG-only decisions; no direction or step-success inference")


if __name__ == "__main__":
    main()

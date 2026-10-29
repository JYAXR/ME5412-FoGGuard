"""FoG-only causal CNN/GRU CSV replay; no activity or clinical labels required."""

import argparse
from pathlib import Path

import pandas as pd

from fogguard.modeling.a1_temporal_cnn import TemporalFoG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    frame = TemporalFoG.load(args.model).predict_frame(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} causal FoG-only rows; context resets on record/gap")


if __name__ == "__main__":
    main()

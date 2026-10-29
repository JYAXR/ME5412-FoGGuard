"""One sensor-only inference entry for the three retained FoG methods."""

import argparse
import json
from pathlib import Path

import pandas as pd

from fogguard.selected_fog import METHODS, check_lock, load_selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="bilateral_rf")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    check_lock(args.bundle, json.loads((args.bundle / "MANIFEST.json").read_text("utf-8")))
    model = load_selected(args.bundle / args.method / "model.joblib", args.method)
    frame = model.predict_frame(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Saved {len(frame)} FoG-only decisions; unsupported time stays unknown")


if __name__ == "__main__":
    main()

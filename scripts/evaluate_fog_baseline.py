"""Evaluate the frequency-domain FoG baseline on generated demo scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fogguard.data import WindowReplay, load_csv
from fogguard.detection import FreezeIndexDetector
from fogguard.evaluation import evaluate_fog_detector

SCENARIO_FILES = ("normal_gait.csv", "straight_fog.csv", "turning_fog.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/demo"),
        help="Directory containing the generated scenario CSV files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="FoG probability threshold used by FreezeIndexDetector",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    detector = FreezeIndexDetector(threshold=args.threshold)
    scenario_results: dict[str, dict[str, float | int]] = {}
    for filename in SCENARIO_FILES:
        path = args.data_dir / filename
        frame = load_csv(path)
        scenario_results[path.stem] = evaluate_fog_detector(
            WindowReplay(frame), detector
        )

    report = {
        "model": "freeze-index-baseline",
        "threshold": args.threshold,
        "scenarios": scenario_results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

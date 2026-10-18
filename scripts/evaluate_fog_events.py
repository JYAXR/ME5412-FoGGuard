"""Evaluate window and event FoG evidence on all generated P0 scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fogguard.data import WindowReplay, load_csv
from fogguard.detection import FreezeIndexDetector
from fogguard.evaluation import evaluate_fog_detector, evaluate_fog_events

SCENARIO_FILES = {
    "normal": "normal_gait.csv",
    "straight_fog": "straight_fog.csv",
    "turning_fog": "turning_fog.csv",
    "right_turning_fog": "right_turning_fog.csv",
    "normal_right_turn": "normal_right_turn.csv",
    "voluntary_stop": "voluntary_stop.csv",
    "fog_recovery": "fog_recovery.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/demo"))
    parser.add_argument(
        "--threshold",
        type=float,
        default=FreezeIndexDetector().threshold,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = FreezeIndexDetector(threshold=args.threshold)
    results: dict[str, object] = {}
    for scenario, filename in SCENARIO_FILES.items():
        replay = WindowReplay(load_csv(args.data_dir / filename))
        results[scenario] = {
            "window_metrics": evaluate_fog_detector(replay, detector),
            "event_metrics": evaluate_fog_events(replay, detector),
        }
    print(
        json.dumps(
            {
                "model": "freeze-index-baseline",
                "threshold": args.threshold,
                "data_scope": "deterministic synthetic P0 scenarios",
                "scenarios": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

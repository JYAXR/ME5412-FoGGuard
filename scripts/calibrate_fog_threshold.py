"""Select a FoG threshold on synthetic development scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fogguard.data import WindowReplay, load_csv
from fogguard.detection import FreezeIndexDetector
from fogguard.evaluation import (
    evaluate_fog_detector,
    select_fog_threshold,
    sweep_fog_thresholds,
)

DEVELOPMENT_FILES = ("normal_gait.csv", "straight_fog.csv")
HOLDOUT_FILE = "turning_fog.csv"
THRESHOLDS = tuple(index / 100 for index in range(101))
SELECTION_RULE = (
    "Require zero false positives on development scenarios; then maximize F1, "
    "recall, and threshold in that order."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/demo"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/fog_threshold_calibration.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    development_replays = [
        WindowReplay(load_csv(args.data_dir / filename))
        for filename in DEVELOPMENT_FILES
    ]
    reports = sweep_fog_thresholds(
        development_replays,
        FreezeIndexDetector(),
        THRESHOLDS,
    )
    selected = select_fog_threshold(reports, max_false_positives=0)
    selected_threshold = float(selected["threshold"])
    holdout_metrics = evaluate_fog_detector(
        WindowReplay(load_csv(args.data_dir / HOLDOUT_FILE)),
        FreezeIndexDetector(threshold=selected_threshold),
    )
    report = {
        "model": "freeze-index-baseline",
        "data_scope": "deterministic synthetic demo scenarios",
        "development_scenarios": [Path(name).stem for name in DEVELOPMENT_FILES],
        "holdout_scenario": Path(HOLDOUT_FILE).stem,
        "threshold_grid": {"start": 0.0, "stop": 1.0, "step": 0.01},
        "selection_rule": SELECTION_RULE,
        "selected_threshold": selected_threshold,
        "development_metrics": selected,
        "holdout_metrics": holdout_metrics,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

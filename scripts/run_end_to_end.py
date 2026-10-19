"""Run the P0 scenarios through the shared end-to-end replay pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fogguard.cueing import AdaptiveCueingController
from fogguard.detection import FreezeIndexDetector
from fogguard.pipeline import replay_csv
from fogguard.turning import YawRateTurnRecognizer

DEFAULT_SCENARIOS = {
    "normal": Path("data/demo/normal_gait.csv"),
    "straight_fog": Path("data/demo/straight_fog.csv"),
    "turning_fog": Path("data/demo/turning_fog.csv"),
    "right_turning_fog": Path("data/demo/right_turning_fog.csv"),
    "normal_right_turn": Path("data/demo/normal_right_turn.csv"),
    "voluntary_stop": Path("data/demo/voluntary_stop.csv"),
    "fog_recovery": Path("data/demo/fog_recovery.csv"),
}


def run_scenarios(
    scenarios: dict[str, Path],
    *,
    fog_threshold: float = 0.21,
    turn_threshold_dps: float = 20.0,
    turn_uncertainty_margin: float = 0.10,
    cue_threshold: float | None = None,
) -> pd.DataFrame:
    """Return one machine-readable table for all requested scenarios."""
    cue_threshold = fog_threshold if cue_threshold is None else cue_threshold
    results = []
    for scenario_id, path in scenarios.items():
        result = replay_csv(
            path,
            detector=FreezeIndexDetector(threshold=fog_threshold),
            turner=YawRateTurnRecognizer(
                threshold_dps=turn_threshold_dps,
                uncertainty_margin=turn_uncertainty_margin,
            ),
            controller=AdaptiveCueingController(min_confidence=cue_threshold),
        )
        result["scenario_id"] = scenario_id
        results.append(result)
    return pd.concat(results, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        help="Canonical replay CSV; repeat to process multiple files. Defaults to all P0 scenarios.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/robot2_end_to_end.csv"),
    )
    parser.add_argument("--fog-threshold", type=float, default=0.21)
    parser.add_argument("--turn-threshold-dps", type=float, default=20.0)
    parser.add_argument("--turn-uncertainty-margin", type=float, default=0.10)
    parser.add_argument(
        "--cue-threshold",
        type=float,
        help="Controller FoG threshold. Defaults to --fog-threshold for one consistent trigger.",
    )
    args = parser.parse_args()

    scenarios = {path.stem: path for path in args.input} if args.input else DEFAULT_SCENARIOS
    result = run_scenarios(
        scenarios,
        fog_threshold=args.fog_threshold,
        turn_threshold_dps=args.turn_threshold_dps,
        turn_uncertainty_margin=args.turn_uncertainty_margin,
        cue_threshold=args.cue_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    summary = result.groupby("scenario_id", sort=False).agg(
        windows=("window_end_s", "size"),
        fog_windows=("fog_detected", "sum"),
        cues=("cue", lambda values: int((values != "none").sum())),
    )
    print(summary.to_string())
    print(f"\nWrote {args.output} ({len(result)} windows)")


if __name__ == "__main__":
    main()

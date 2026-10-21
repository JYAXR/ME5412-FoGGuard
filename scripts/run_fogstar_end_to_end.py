"""Run FoG-STAR predictions through Robot 2 temporal context integration."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fogguard.context import FogStarIntegrationConfig
from fogguard.fogstar_pipeline import integrate_fogstar_predictions, replay_fogstar_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input",
        type=Path,
        help="FoG-STAR-layout sensor CSV; requires --model.",
    )
    source.add_argument(
        "--predictions",
        type=Path,
        help="Precomputed raw prediction CSV, preferably subject-held-out predictions.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/fogstar_multitask_baseline.joblib"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/fogstar_robot2_end_to_end.csv"),
    )
    parser.add_argument("--confirmation-windows", type=int, default=2)
    parser.add_argument("--recovery-windows", type=int, default=3)
    parser.add_argument("--direction-confidence", type=float, default=0.60)
    parser.add_argument("--maximum-input-age-s", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FogStarIntegrationConfig(
        confirmation_windows=args.confirmation_windows,
        recovery_windows=args.recovery_windows,
        minimum_direction_confidence=args.direction_confidence,
        maximum_input_age_s=args.maximum_input_age_s,
    )
    if args.predictions:
        result = integrate_fogstar_predictions(pd.read_csv(args.predictions), config=config)
    else:
        input_path = args.input or Path("data/raw/fogstar/sensor_data.csv")
        result = replay_fogstar_csv(input_path, model_path=args.model, config=config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = {
        "windows": len(result),
        "raw_fog_windows": int(result["is_fog"].sum()),
        "confirmed_fog_windows": int(result["stable_fog_detected"].sum()),
        "confirmed_events": int(result["event_id"].dropna().nunique()),
        "cues": int((result["cue"] != "none").sum()),
        "unknown_turn_windows": int((result["turn"] == "unknown").sum()),
        "unsafe_input_windows": int((~result["input_fresh"]).sum()),
    }
    print(pd.Series(summary).to_string())
    print(f"\nWrote {args.output} ({len(result)} windows)")


if __name__ == "__main__":
    main()

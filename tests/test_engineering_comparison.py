"""Check experiment denominators, time accounting and fair single-factor contrasts."""

import importlib.util
import itertools
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from fogguard.complete import strategy_configs
from fogguard.cueing.march import MarchGuidanceController
from fogguard.cueing.replay import replay_guidance

_spec = importlib.util.spec_from_file_location(
    "engineering_comparison",
    Path(__file__).resolve().parents[1] / "scripts/run_engineering_comparison.py",
)
experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment)


def test_grid_unique_and_single_factor():
    cases = list(experiment.make_cases())
    assert len(cases) == 112
    assert len({name for name, *_ in cases}) == 112
    signatures = [tuple(sorted(spec.items())) for _, spec, *_ in cases]
    assert len(set(signatures)) == 112
    full = asdict(strategy_configs()["march"])
    for name, field in [("countdown", "enable_march_fallback"),
                        ("no_fade", "enable_fading"),
                        ("no_spacing", "enable_adaptive_spacing")]:
        other = asdict(strategy_configs()[name])
        assert [key for key in full if full[key] != other[key]] == [field]


def test_known_intervals_not_row_counts():
    _, spec, _, events, responses = next(experiment.make_cases())
    commands = replay_guidance(events, responses=responses, config=strategy_configs()["march"],
                               controller_type=MarchGuidanceController)
    measured = experiment.measure(events, commands, spec)
    # Independently integrate consecutive command timestamps, including disabled stop rows.
    records = commands.to_dict("records")
    expected = sum((b["window_end_s"] - a["window_end_s"])
                   for a, b in itertools.pairwise(records) if a["guidance_visual_enabled"])
    assert measured["active_s"] == pytest.approx(expected)
    assert measured["start_count"] == measured["stop_count"]
    assert measured["unclosed_commands"] == 0


def test_normal_missing_metrics_are_not_zero_efficacy():
    case = next(c for c in experiment.make_cases() if c[1]["family"] == "normal")
    _, spec, _, events, responses = case
    commands = replay_guidance(events, responses=responses, config=strategy_configs()["march"],
                               controller_type=MarchGuidanceController)
    measured = experiment.measure(events, commands, spec)
    assert measured["truth_to_start_s"] is None
    assert measured["spacing_error_m"] is None
    assert measured["truth_events"] == measured["active_s"] == 0


def test_pair_alignment_and_absent_values():
    frame = pd.DataFrame([
        {"case": case, "family": "forward", "strategy": strategy,
             "active_s": value, "after_final_truth_end_active_s": value,
             "truth_to_start_s": None, "spacing_error_m": None,
             "unmatched_starts": 0, "missed_truth_events": 0}
        for case, strategy, value in [("a", "march", 3), ("b", "march", 5),
                                      ("b", "no_fade", 9), ("a", "no_fade", 7)]
    ])
    pairs = experiment.paired_results(frame)
    row = pairs.loc[pairs.alternative.eq("no_fade") & pairs.metric.eq("active_s")].iloc[0]
    assert row.pairs == 2
    assert row.mean_paired_difference == -4
    assert row.negative == 2
    assert "spacing_error_m" not in pairs.metric.tolist()

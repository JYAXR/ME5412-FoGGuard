"""Independent checks from saved CSVs; does not import the experiment or controller."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    out = parser.parse_args().output_dir
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((out / "runs.csv").open(encoding="utf-8")))
    assert len(rows) == manifest["runs"] == 560
    assert manifest["cases"] == 112
    assert len({r["case"] for r in rows}) == 112
    checked = 0


    def equivalent(left, right):
        if left == right:
            return True
        try:
            return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
        except ValueError:
            return False


    for row in rows:
        commands = list(csv.DictReader((out / "timelines" / (
            row["case"] + "_" + row["strategy"] + ".csv")).open(encoding="utf-8")))
        events = list(csv.DictReader((out / "inputs" / (
            row["case"] + "_events.csv")).open(encoding="utf-8")))
        windows = [r for r in commands if r["guidance_terminal"] == "False"]
        assert len(windows) == len(events)
        for event, command in zip(events, windows):
            assert all(equivalent(command[key], value) for key, value in event.items())
        positive = [float(r["window_end_s"]) for r in events if int(r["truth_fog_label"]) == 1]
        truth_end = positive[-1] + 0.5 if positive else float("inf")
        active = tail = nonfog = 0.0
        for current, following in itertools.pairwise(windows):
            if current["guidance_visual_enabled"] == "True":
                t = float(current["window_end_s"])
                duration = min(0.5, float(following["window_end_s"]) - t)
                active += duration
                if t >= truth_end:
                    tail += duration
                if int(current["truth_fog_label"]) == 0:
                    nonfog += duration
        assert abs(active - float(row["active_s"])) < 1e-9
        assert abs(tail - float(row["after_final_truth_end_active_s"])) < 1e-9
        assert abs(nonfog - float(row["active_non_fog_s"])) < 1e-9
        starts = [r for r in commands if r["guidance_action"] == "start"]
        stops = [r for r in commands if r["guidance_action"] == "stop"]
        assert len(starts) == len(stops) == int(row["start_count"])
        for start in starts:
            stop = [r for r in stops if r["guidance_command_id"] == start["guidance_command_id"]]
            assert len(stop) == 1
            assert 0 <= float(stop[0]["window_end_s"]) - float(start["window_end_s"]) <= 15
        checked += 1
    for name, digest in manifest["csv_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    critical = ["audit_failures", "invalid_input_active_rows", "unclosed_commands",
                "spacing_out_of_bounds_rows", "turn_errors", "censored_stops"]
    assert all(float(r[k]) == 0 for r in rows for k in critical)
    slow = [r for r in rows if r["family"] == "march" and r["strategy"] == "march"
            and float(r["interval"]) == 1]
    fast = [r for r in rows if r["family"] == "march" and r["strategy"] == "march"
            and float(r["interval"]) == 0.5]
    assert sum(int(r["march_mode_timeouts"]) for r in slow) == 4
    assert sum(int(r["march_completions"]) for r in fast) == 4
    result = {"status": "PASS", "independently_checked_runs": checked,
                  "csv_hashes_verified": len(manifest["csv_sha256"]),
                  "input_command_alignment": "all input columns / all 560 runs match; numeric CSV "
                                          "representations compared with tolerance 1e-12 (e.g. 1 vs 1.0)",
                  "duration_check": "saved timestamp integration, no production metric imports",
                  "maximum_episode_s": 15, "slow_march_timeouts": 4, "fast_march_completions": 4,
                  "critical_failure_counts": {k: sum(int(r[k]) for r in rows) for k in critical}}
    (out / "独立复核.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""English test-report figures from saved results; no fitting or score selection."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fogguard.selected_fog import sha, write_json

KEYS = ["subject_id", "session_id", "task_id"]
LABELS = {"ankle_history_rf": "Ankle-history RF", "bilateral_rf": "Bilateral RF", "ankle_cnn": "Ankle CNN"}


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(out / f"{name}.svg", facecolor="white")
    plt.close(fig)


def draw_matrix(ax, matrix, rows, columns, title):
    matrix = np.asarray(matrix)
    ax.imshow(matrix, cmap="Blues", aspect="auto")
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, str(count), ha="center", va="center",
                color="white" if count > matrix.max() / 2 else "#243549")
    ax.set(xticks=range(len(columns)), xticklabels=columns, yticks=range(len(rows)),
           yticklabels=rows, xlabel="Predicted", ylabel="Window target", title=title)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a fresh output directory")
    results = read(args.run / "results.json")
    verification = read(args.run / "INDEPENDENT_VERIFICATION.json")
    if verification["status"] != "PASS":
        raise ValueError("Independently verify the saved results before plotting")
    args.output_dir.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "fogguard-heldout-v1"})
    fog = results["a1_common_windows"]
    raw = results["a2_current_direction"]["primary_known_targets"]
    fused = results["fusion"]["direction_on_a2_arrival_grid"]["primary_known_targets"]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(3)
    for offset, key, label, color in [(-0.25, "precision", "Precision", "#396b9e"),
                                      (0, "recall", "Recall", "#41a18b"),
                                      (0.25, "fog_f1", "FoG F1", "#d69843")]:
        values = [r[key] for r in fog]
        bars = ax.bar(x + offset, values, 0.24, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=10, padding=3)
    ax.set(xticks=x, xticklabels=[LABELS[r["method"]] for r in fog], ylim=(0, 1),
           ylabel="Score", title="FoG detection on locked, participant-held-out TEST")
    ax.legend(loc="upper right", ncol=3)
    ax.grid(axis="y", alpha=0.15)
    fig.text(0.5, 0.02, f"4 participants | {results['records']} records | "
             f"{fog[0]['windows']:,} matched windows | {fog[0]['positive_windows']} FoG windows. "
             "Existing thresholds; no TEST tuning.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, args.output_dir, "01_fog_test")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, row in zip(axes, fog, strict=True):
        draw_matrix(ax, row["confusion_matrix"], ["Non-FoG", "FoG"], ["Non-FoG", "FoG"],
                    LABELS[row["method"]])
    fig.suptitle("FoG TEST confusion counts: identical supported windows", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, args.output_dir, "02_fog_confusion")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1.1, 1.1, 1.3]})
    for ax, report, title in [(axes[0], raw, "Raw current-direction model"),
                              (axes[1], fused, "After fixed gating + smoothing")]:
        draw_matrix(ax, report["confusion_matrix"][:3], ["Straight", "Left", "Right"],
                    ["Straight", "Left", "Right", "Unknown"], title)
        ax.tick_params(axis="x", labelrotation=30)
    xs = np.arange(3)
    raw_values = [raw[k] for k in ["accuracy", "macro_f1", "prediction_coverage"]]
    fused_values = [fused[k] for k in ["accuracy", "macro_f1", "prediction_coverage"]]
    axes[2].bar(xs - 0.18, raw_values, 0.36, label="Raw", color="#396b9e")
    axes[2].bar(xs + 0.18, fused_values, 0.36, label="Fused", color="#d69843")
    axes[2].set(xticks=xs, xticklabels=["Accuracy", "Macro-F1", "Coverage"], ylim=(0, 1),
                title="Same 676 known-target windows")
    axes[2].tick_params(axis="x", labelrotation=20)
    axes[2].legend()
    fig.suptitle("Direction TEST: report model quality and fusion rejection separately", weight="bold")
    fig.text(0.5, 0.02, "Unknown predictions count as errors on known targets. "
             "Invalid sensor windows are excluded and counted separately. "
             "Current direction is not future intention.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    save(fig, args.output_dir, "03_direction_test")
    perception = pd.read_csv(args.run / "perception_runtime.csv")
    first = next(iter(perception.groupby(KEYS, sort=True)))[1]
    identity = first[KEYS].iloc[0]
    start = float(first.available_at_s.min())
    first = first.loc[first.available_at_s <= start + 30].copy()
    first["time_s"] = first.available_at_s - start
    def clip(frame):
        mask = np.ones(len(frame), dtype=bool)
        for name in KEYS:
            mask &= frame[name].eq(identity[name]).to_numpy()
        return frame.loc[mask & frame.available_at_s.between(start, start + 30)].copy()
    primary = results["primary_fusion_fog_method"]
    fog_case = clip(pd.read_csv(args.run / primary / "evaluation.csv"))
    direction_case = clip(pd.read_csv(args.run / "direction_evaluation.csv"))
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    times = fog_case.available_at_s - start
    axes[0].step(times, fog_case.truth_fog_label, where="post", color="#b94d54", alpha=0.7,
                 label="Trailing-window FoG target")
    axes[0].step(times, fog_case.fog_probability, where="post", color="#396b9e", label="Bilateral RF score")
    axes[0].axhline(fog[1]["threshold"], linestyle="--", color="#7a8390", label="Locked threshold")
    axes[0].set(ylabel="Score / target", ylim=(-0.05, 1.1))
    axes[0].legend(loc="upper right", fontsize=9, ncol=3)
    phase = {"safe_none": 0, "normal": 1, "suspected": 2, "confirmed": 3, "recovery": 4}
    axes[1].step(first.time_s, first.event_phase.map(phase), where="post", color="#347a69")
    axes[1].set(ylabel="FoG event", yticks=range(5), yticklabels=list(phase), ylim=(-0.3, 4.3))
    turns = {"unknown": 0, "straight": 1, "left": 2, "right": 3}
    target = direction_case.target.map(turns).where(direction_case.input_valid)
    axes[2].step(direction_case.available_at_s - start, target, where="post", linestyle="--",
                 color="#7a8390", label="Window direction target")
    axes[2].step(first.time_s, first.direction.map(turns), where="post", color="#9ab4d0", label="Raw model")
    axes[2].step(first.time_s, first.turn.map(turns), where="post", color="#224e84", label="Fused direction")
    axes[2].set(ylabel="Direction", yticks=range(4), yticklabels=list(turns), ylim=(-0.3, 3.3),
                xlabel="Decision time from record start (s)")
    axes[2].legend(loc="upper right", fontsize=9, ncol=3)
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.set_xlim(0, min(30, float(first.time_s.max())) + 0.01)
    fig.suptitle("Locked TEST replay: first record, first 30 seconds", weight="bold")
    fig.text(0.5, 0.015, "Fixed display rule; no selection by prediction quality. "
             "Targets describe completed windows; recovery is a score state, not a measured step.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, args.output_dir, "04_test_timeline")
    table = pd.read_csv(args.run / "fog_comparison.csv")
    table.to_csv(args.output_dir / "fog_test_metrics.csv", index=False)
    pd.DataFrame([{"stage": stage, **{k: v for k, v in report.items()
                                      if not isinstance(v, (list, dict))}}
                  for stage, report in [("raw_current_direction", raw), ("fused_direction", fused)]])\
        .to_csv(args.output_dir / "direction_test_metrics.csv", index=False)
    markup = table[["method", "accuracy", "precision", "recall", "fog_f1"]].to_html(
        index=False, float_format=lambda v: f"{v:.3f}")
    body = '<!doctype html><html lang="en"><meta charset="utf-8"><title>FoGGuard TEST report</title>'
    body += '<style>body{max-width:1250px;margin:35px auto;font:17px/1.55 sans-serif;color:#213547}'
    body += 'img{width:100%;margin:20px 0}table{border-collapse:collapse;width:100%}'
    body += 'td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}</style>'
    body += '<h1>FoGGuard A-group: locked held-out TEST</h1><p>4 participants, 16 records. '
    body += 'Existing weights and thresholds; no fitting or parameter search on TEST.</p>' + markup
    body += f'<p>Raw direction: accuracy <b>{raw["accuracy"]:.2%}</b>, '
    body += f'macro-F1 {raw["macro_f1"]:.3f}, coverage {raw["prediction_coverage"]:.2%}. '
    body += f'Fused direction: accuracy <b>{fused["accuracy"]:.2%}</b>, '
    body += f'macro-F1 {fused["macro_f1"]:.3f}, coverage {fused["prediction_coverage"]:.2%}. '
    body += 'Direction scores use 676 valid known-target windows. '
    body += '398 invalid sensor windows and 531 unknown-target windows have separate denominators.</p>'
    body += '<p>FoG false alarms remain substantial. Fixed direction fusion sacrifices coverage; '
    body += 'it does not improve overall direction accuracy. Four held-out participants do not '
    body += 'establish clinical performance or external-dataset generalization.</p>'
    for name in ["01_fog_test", "02_fog_confusion", "03_direction_test", "04_test_timeline"]:
        body += f'<img src="{html.escape(name)}.png" alt="{html.escape(name)}">'
    body += '<p>Unknown input remains unknown. No motion-response or clinical outcome is evaluated.</p></html>'
    (args.output_dir / "index.html").write_text(body, encoding="utf-8")
    write_json(args.output_dir / "MANIFEST.json", {
        "schema": "fogguard.a_group.test_figures.v1", "source_sha256": sha(Path(__file__)),
        "test_results_sha256": sha(args.run / "results.json"),
        "independent_verification_sha256": sha(args.run / "INDEPENDENT_VERIFICATION.json"),
        "selection": "First TEST record by sorted identity; first 30 seconds from record start",
        "files": {p.name: sha(p) for p in args.output_dir.iterdir() if p.is_file()},
    })
    print(f"Saved English TEST figures and tables to {args.output_dir}")


if __name__ == "__main__":
    main()

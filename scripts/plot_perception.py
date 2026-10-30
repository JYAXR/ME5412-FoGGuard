"""English presentation figures from saved perception decisions; no truth needed."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fogguard.perception import KEYS
from fogguard.selected_fog import sha, write_json


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(directory / f"{name}.svg", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a fresh output directory")
    frame = pd.read_csv(args.runtime)
    if frame.empty:
        raise ValueError("Presentation requires runtime decisions")
    args.output_dir.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "fogguard-perception-v1"})
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set(xlim=(0, 12), ylim=(0, 5))
    ax.axis("off")
    boxes = [(1.4, 2.5, "60 Hz IMU\nComplete 2 s windows"),
             (4.2, 3.5, "A1: FoG\nBilateral RF"),
             (4.2, 1.5, "A2: current direction\nGravity + rotation RF"),
             (7.3, 2.5, "Causal source adapter\nRecord + availability\nLifetime <= 0.5 s"),
             (10.4, 2.5, "Perception events\nConfirmation + recovery\nUnknown / safe_none")]
    for x, y, text in boxes:
        ax.text(x, y, text, ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.7", "fc": "#eef4fa", "ec": "#3f6c90"})
    for start, end in [((2.5, 2.8), (3.1, 3.4)), ((2.5, 2.2), (3.1, 1.6)),
                       ((5.3, 3.4), (6.1, 2.8)), ((5.3, 1.6), (6.1, 2.2)),
                       ((8.5, 2.5), (9.1, 2.5))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#3f6c90"})
    ax.text(6, 4.6, "A1 + A2: from independent predictions to a causal timeline",
            ha="center", fontsize=16, weight="bold")
    ax.text(6, 0.45, "Only new observations advance confirmation. Expiry withdraws state.\n"
            "B1 consumes saved events; feedback and cue decisions remain downstream.", ha="center")
    fig.tight_layout()
    save(fig, args.output_dir, "pipeline")

    # Fixed display rule, independent of annotations/accuracy: first record with
    # both a confirmed model event and a known smoothed turn; otherwise first.
    groups = list(frame.groupby(KEYS, sort=True))
    selected = next((g for _, g in groups if g.event_phase.eq("confirmed").any()
                     and g.turn.isin(["left", "right"]).any()), groups[0][1])
    center = selected.loc[selected.event_phase.eq("confirmed"), "available_at_s"]
    center = float(center.iloc[0]) if len(center) else float(selected.available_at_s.iloc[0])
    origin = max(float(selected.available_at_s.min()), center - 5)
    clip = selected.loc[selected.available_at_s.between(origin, origin + 30)].copy()
    clip["time_s"] = (clip.available_at_s - origin).round(9)
    times = clip.time_s.to_numpy()
    phase_names = ["safe_none", "normal", "suspected", "confirmed", "recovery"]
    turn_names = ["unknown", "straight", "left", "right"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1, 1.2]})
    axes[0].step(times, clip.fog_probability, where="post", color="#b6404a", label="A1 FoG score")
    axes[0].step(times, clip.fog_threshold, where="post", color="#6b7280", linestyle="--",
                 label="A1 threshold")
    axes[0].set(ylabel="Model score", ylim=(-0.04, 1.04))
    axes[0].legend(loc="upper right", ncol=2, fontsize=9)
    axes[1].step(times, clip.event_phase.map(dict(zip(phase_names, range(5), strict=True))),
                 where="post", color="#347a69")
    axes[1].set(ylabel="FoG event", yticks=range(5), yticklabels=phase_names, ylim=(-0.3, 4.3))
    turn_map = dict(zip(turn_names, range(4), strict=True))
    axes[2].step(times, clip.direction.map(turn_map), where="post", color="#98a7ba",
                 linewidth=1.5, label="A2 raw current direction")
    axes[2].step(times, clip.turn.map(turn_map), where="post", color="#2a5c99",
                 linewidth=2, label="Smoothed direction")
    axes[2].set(ylabel="Direction", yticks=range(4), yticklabels=turn_names, ylim=(-0.3, 3.3),
                xlabel="Decision time relative to displayed clip start (s)")
    axes[2].legend(loc="upper right", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.set_xlim(float(times[0]), float(times[-1]) + 0.001)
    fig.suptitle("Real-IMU development replay: scores, event confirmation and direction",
                 fontsize=15, weight="bold")
    fig.text(0.5, 0.012, "Illustrative record; unknown includes rejection, expiry and missing input. "
             "No labels, motion feedback or clinical outcomes shown.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, args.output_dir, "real_timeline")
    clip[["time_s", "fog_probability", "fog_threshold", "event_phase", "direction", "turn",
          "fog_status", "direction_status"]].to_csv(args.output_dir / "timeline_data.csv", index=False)
    (args.output_dir / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>FoGGuard perception</title>'
        '<style>body{max-width:1200px;margin:32px auto;font:17px sans-serif;color:#213547}'
        'img{width:100%;margin:20px 0}p{line-height:1.6}</style>'
        '<h1>FoGGuard: A1 + A2 perception</h1>'
        '<p>Independent FoG and current-direction models feed a causal event timeline. '
        'These figures show the implementation and one real-IMU development replay. '
        'They do not measure independent test accuracy or clinical recovery.</p>'
        '<img src="pipeline.png" alt="Causal perception pipeline">'
        '<img src="real_timeline.png" alt="Real development replay">'
        '<p><a href="timeline_data.csv">Derived figure data</a> · '
        '<a href="MANIFEST.json">Figure provenance</a></p></html>', encoding="utf-8")
    # Preserve the graph's bounded provenance without publishing record identity.
    write_json(args.output_dir / "MANIFEST.json", {
        "schema": "fogguard.perception.presentation.v1", "runtime_sha256": sha(args.runtime),
        "source_sha256": sha(Path(__file__)), "truth_used": False,
        "selection": "First record by sorted keys with confirmed event and smoothed left/right; "
                     "fallback first record; 30 s clip starting 5 s before first confirmation",
        "scope": "Illustrative development predictions, not a measured recovery outcome",
        "files": {p.name: sha(p) for p in args.output_dir.iterdir() if p.is_file()},
    })
    print(f"Saved English figures to {args.output_dir}")


if __name__ == "__main__":
    main()

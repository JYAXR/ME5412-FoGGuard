"""Export reviewed bilateral figures without modifying locked experiment evidence."""

import argparse
import html
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json

from fogguard.evaluation.a1_fixed_split import role_frame
from fogguard.modeling.a1_bilateral import anchored_gyro_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLCONFIGDIR", str(root / "artifacts/matplotlib-a1-bilateral-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run, output = args.run_dir, args.output_dir
    if output.exists():
        parser.error("Use a fresh presentation directory")
    lock = json.loads((run / "MODEL_LOCK.json").read_text("utf-8"))
    check_lock(run, lock)
    complete = json.loads((run / "evaluation/EVALUATION_COMPLETED.json").read_text("utf-8"))
    for name, digest in complete["files"].items():
        if sha(run / "evaluation" / name) != digest:
            raise ValueError(f"Evaluation evidence changed: {name}")
    if json.loads((run / "verification.json").read_text("utf-8"))["status"] != "PASS":
        raise ValueError("Verify the experiment first")
    protocol = json.loads((run / "protocol.json").read_text("utf-8"))
    if sha(args.raw) != protocol["raw_sha256"]:
        raise ValueError("Raw data changed")
    split = json.loads((run / "split.json").read_text("utf-8"))
    # Only TRAIN windows are constructed for waveform export; no fitting or scoring.
    train = role_frame(pd.read_csv(args.raw), split, "train")
    windows = anchored_gyro_windows(train, require_right=True)
    frame = pd.read_csv(run / "input_audit/joint_reference_predictions.csv")
    cases = json.loads((run / "input_audit/selected_pairs.json").read_text("utf-8"))
    output.mkdir(parents=True)
    titles = {"nonfog_stand_vs_fog": "Stand", "right_turn_nonfog_vs_fog": "Right turn",
              "left_turn_nonfog_vs_fog": "Left turn"}
    for case in cases:
        # Fixed margins and wrapped titles repair clipping in the original export.
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.78, hspace=0.39, wspace=0.25)
        for col, index in enumerate((case["negative_index"], case["positive_index"])):
            record = frame.iloc[index]
            for row, (start, ankle) in enumerate(((0, "Left"), (3, "Right"))):
                ax = axes[row, col]
                for k, axis in enumerate("xyz"):
                    ax.plot(np.arange(120) / 60, windows[index].samples[:, start + k], lw=0.9, label=axis)
                ax.set_ylabel(f"{ankle} gyro (deg/s)")
                ax.grid(alpha=0.15)
                ax.legend(ncol=3, fontsize=8)
                if row == 1:
                    ax.set_xlabel("Window-relative time (s)")
            label = "Pure non-FoG, prior OOF false positive" if col == 0 else "Pure FoG"
            axes[0, col].set_title(f"{label}; prior OOF score {record.fog_probability:.3f}\n"
                                   f"P{int(record.subject_id)} / S{int(record.session_id)} / T{int(record.task_id)}, "
                                   f"start {record.window_start_s:.3f}s", fontsize=10)
        fig.suptitle(f"TRAIN | {titles[case['group']]} | same person/activity, disjoint windows\n"
                     f"Deliberately closest left-feature pair: left distance {case['left_distance']:.3f}, "
                     f"66D distance {case['dual_distance']:.3f}\n"
                     "Different dimensions: distances are descriptive, not an information-gain score", fontsize=11, y=0.97)
        for suffix in ("png", "svg"):
            fig.savefig(output / f"{case['group']}.{suffix}", dpi=180)
        plt.close(fig)
    for source in (run / "input_audit/figures", run / "evaluation/figures"):
        for file in source.iterdir():
            if file.suffix in (".png", ".svg") and file.stem not in titles:
                shutil.copyfile(file, output / file.name)
    table = pd.read_csv(run / "evaluation/comparison.csv")
    shown = table.loc[table.pool.eq("validation_own"),
                      ["model", "windows", "precision", "recall", "fog_f1", "false_alarm_s", "time_recall"]]
    intro = ("双踝输入研发对照：固定TRAIN人员折、相同有效窗口、相同随机森林。"
             "VALIDATION为已开发队列，不是新盲测；历史模型输入/选择流程不同。"
             "TRAIN波形是刻意挑选的最近左踝特征配对，不能作为代表性样本或无法区分的证明。"
             "未评分新模型TEST，未替换默认，不代表转向意图、迈步成功或临床效果。")
    images = "".join(f'<h2>{html.escape(p.stem)}</h2><img src="{p.name}" alt="{html.escape(p.stem)}">'
                     for p in sorted(output.glob("*.png")))
    (output / "index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>A1 双踝对照展示</title><style>body{max-width:1200px;margin:32px auto;'
        'font:16px/1.6 sans-serif;padding:0 20px}img{max-width:100%}td,th{padding:6px}</style>'
        f'<h1>A1 双踝输入对照</h1><p>{intro}</p>' + shown.to_html(index=False, float_format=lambda x: f"{x:.4f}")
        + images + '</html>', encoding="utf-8")
    write_json(output / "provenance.json", {"model_lock_sha256": sha(run / "MODEL_LOCK.json"),
        "evaluation_completion_sha256": sha(run / "evaluation/EVALUATION_COMPLETED.json"),
        "raw_sha256": sha(args.raw), "report_source_sha256": sha(Path(__file__)),
        "kind": "Layout-only waveform re-export plus five unchanged figures; no refit/rescore",
        "test_windows_constructed": 0,
        "files": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print(f"Saved {len(list(output.glob('*.png')))} PNG/SVG pairs and an HTML index")


if __name__ == "__main__":
    main()

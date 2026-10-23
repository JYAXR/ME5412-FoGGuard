"""Build an offline review from saved results, reusing Automation 1's renderer.

No model inference or controller rule is implemented in this presentation layer.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fogguard.cueing.review import write_guidance_review
from fogguard.evaluation.automation2 import SEQUENCE


def read_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={name: str for name in SEQUENCE})


def make_figure(frame: pd.DataFrame, title: str) -> go.Figure:
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                           subplot_titles=("模型评分", "真值与事件确认", "提示输出"))
    time = frame.guidance_timestamp_s
    figure.add_trace(go.Scatter(x=time, y=frame.fog_probability, name="FoG 评分",
                               mode="lines+markers", line_color="#246a9b"), row=1, col=1)
    for field, label, color in (("truth_fog_label", "窗口真值", "#222222"),
                                 ("is_fog", "原始判定", "#d38a28"),
                                 ("stable_fog_detected", "持续确认", "#27796c")):
        figure.add_trace(go.Scatter(x=time, y=frame[field].astype(int), name=label,
                                   mode="lines", line={"shape": "hv", "color": color}),
                         row=2, col=1)
    figure.add_trace(go.Scatter(x=time, y=frame.guidance_visual_opacity, name="视觉透明度",
                               mode="lines", line={"shape": "hv", "color": "#2869a4"}),
                     row=3, col=1)
    figure.add_trace(go.Scatter(x=time, y=frame.guidance_audio_volume, name="声音音量",
                               mode="lines", line={"shape": "hv", "color": "#b56b37"}),
                     row=3, col=1)
    starts = frame.loc[frame.guidance_action.eq("start")]
    figure.add_trace(go.Scatter(x=starts.guidance_timestamp_s,
                               y=starts.guidance_visual_opacity, name="新提示启动",
                               mode="markers", marker={"size": 12, "symbol": "diamond"}),
                     row=3, col=1)
    figure.update_layout(title=title, height=650, template="plotly_white",
                         margin={"l": 55, "r": 30, "t": 90, "b": 55},
                         legend={"orientation": "h", "y": -0.15})
    figure.update_xaxes(title_text="输出时间 / 秒", row=3, col=1)
    return figure


def build_review(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    acceptance = json.loads((directory / "acceptance.json").read_text(encoding="utf-8"))
    output = directory / "review"
    output.mkdir(parents=True, exist_ok=True)
    cases = {}
    metadata = {}
    folders = sorted((directory / "timelines" / name for name in manifest["case_names"]),
                     key=lambda p: (p.name != "sensor_straight_fog", p.name))
    for folder in folders:
        if folder.name == "real_oof" or folder.name.endswith("suppression_on"):
            continue
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        for strategy in metrics:
            name = f"{folder.name} / {strategy}"
            frame = read_frame(folder / f"{strategy}.csv")
            cases[name] = json.loads(frame.to_json(orient="records"))
            metadata[name] = {
                **metrics[strategy]["lifecycle"],
                "source": "合成 IMU 回放" if folder.name.startswith("sensor_")
                else "构造候选＋明确模拟响应；只验证控制行为",
                "fixed_reference": metrics["fixed"]["lifecycle"],
            }
    write_guidance_review(output / "control_playback.html", cases, {"cases": metadata})
    packets = []

    def packet(name, frame, title, explanation):
        if frame.empty:
            raise ValueError(f"Empty evidence packet: {name}")
        csv_name = f"{name}.csv"
        figure_name = f"{name}.html"
        frame.to_csv(output / csv_name, index=False, encoding="utf-8-sig")
        figure = make_figure(frame, title)
        figure.write_html(output / figure_name, include_plotlyjs=True, full_html=True)
        fields = ["window_end_s", "truth_fog_label", "is_fog", "event_phase", "turn",
                  "guidance_action", "guidance_control_state", "guidance_reason"]
        table = frame[fields].to_html(index=False, border=0, escape=True)
        packets.append({"name": name, "title": title, "explanation": explanation,
                        "rows": len(frame), "csv": csv_name, "figure": figure_name,
                        "first_output_s": float(frame.window_end_s.iloc[0]),
                        "last_output_s": float(frame.window_end_s.iloc[-1]),
                        "table_html": table})

    for name, title, text in (
        ("voluntary_stop", "主动停步误提示", ("默认规则仍把部分主动停步窗口判断为冻结，并启动提示。"
         "这是保留的失败案例；另有静止抑制对照，未改变默认策略。")),
        ("fog_recovery", "冻结恢复边界", ("原始判定在恢复边界有额外阳性窗口；"
         "持续确认与恢复观察的输出需分别检查。")),
    ):
        frame = read_frame(directory / f"timelines/sensor_{name}/response.csv")
        packet(name, frame, title, text)
    if "real_oof" in manifest["case_names"]:
        folder = directory / "timelines/real_oof"
        frame = read_frame(folder / "response.csv")
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        missed = next((event for event in metrics["response"]["metrics"]["events"]
                       if not event["detected"]), None)
        if missed is not None:
            mask = frame.window_end_s.between(missed["start_s"] - 2, missed["end_s"] + 3)
            for field in SEQUENCE:
                mask &= frame[field].eq(str(missed[field]))
            packet("real_missed_event", frame.loc[mask], "真实数据：未匹配的确认事件",
                   "按严格口径选出的第一个确认起点未匹配事件。提前确认也会导致起点不匹配，"
                   "不代表事件期间完全没有识别或提示。请将确认起点、提示启动与实际覆盖分别查看。")
        uncovered = next((event for event in metrics["response"]["metrics"]["events"]
                          if event["guidance_active_truth_windows"] == 0), None)
        if uncovered is not None:
            mask = frame.window_end_s.between(uncovered["start_s"] - 2, uncovered["end_s"] + 3)
            for field in SEQUENCE:
                mask &= frame[field].eq(str(uncovered[field]))
            packet("real_no_active_guidance", frame.loc[mask], "真实数据：真值冻结窗口没有活动提示",
                   "该真值事件的阳性窗口均无活动视觉提示。与起点未匹配分开检查；前后文保留方向"
                   "拒识、确认状态及控制原因。不把安全撤销自动改成继续提示。")
        starts = frame.loc[frame.guidance_action.eq("start") & frame.truth_fog_label.eq(0)]
        if not starts.empty:
            chosen = starts.iloc[0]
            mask = frame.window_end_s.between(chosen.window_end_s - 2, chosen.window_end_s + 3)
            for field in SEQUENCE:
                mask &= frame[field].eq(str(chosen[field]))
            packet("real_negative_window_start", frame.loc[mask], "真实数据：负例窗口启动提示",
                   "按文件顺序选择第一个启动窗口真值为负的提示。负例窗口启动和不匹配事件的启动"
                   "定义不同；前后文应一并查看，不将其等同于全部临床误提示。")
    rows = pd.DataFrame(acceptance["checks"])
    acceptance_table = rows[["check", "status", "meaning"]].to_html(index=False, border=0, escape=True)
    sections = []
    for item in packets:
        sections.append(f'<section><h2>{html.escape(item["title"])}</h2>'
                        f'<p>{html.escape(item["explanation"])}</p>'
                        f'<p><a href="{item["figure"]}">打开交互时间图</a> · '
                        f'<a href="{item["csv"]}">查看完整片段 CSV</a></p>'
                        f'<details><summary>逐窗证据（{item["rows"]} 行）</summary>'
                        f'<div class="table">{item["table_html"]}</div></details></section>')
    known = acceptance["counts"]["KNOWN_LIMITATION"]
    status = "工程回归通过" if acceptance["regression_gate_passed"] else "存在工程回归失败"
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FoGGuard 评估证据</title>
<style>body{{font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif;color:#223347;background:#f4f6f8;
margin:0}}main{{max-width:1080px;margin:30px auto;padding:32px;background:white}}h1{{font-size:30px}}
h2{{font-size:22px}}section{{border-top:1px solid #dbe2e9;margin-top:28px;padding-top:16px}}
a{{color:#1b6387}}.status{{padding:18px;background:#eaf2f7}}table{{border-collapse:collapse;
width:100%;font-size:13px}}td,th{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}
.table{{overflow:auto}}summary{{cursor:pointer}}@media(max-width:700px){{main{{margin:0;padding:18px}}}}
</style><main><h1>FoGGuard · 自动化 2 评估证据</h1>
<p class="status"><b>{status}</b>；通过 {acceptance['counts']['PASS']} 项，失败
{acceptance['counts']['FAIL']} 项，另有 {known} 项已知局限。回归通过不等于识别错误已经解决。</p>
<p>读取保存的控制结果，不另写模型或控制规则。全部页面可离线打开；图表和脚印用于工程说明。</p>
<p><a href="control_playback.html">打开脚印、声音参数与时间线回放</a>（复用自动化 1 原有展示器）</p>
<p>真实数据状态：{html.escape(manifest['real_data_status'])}。</p>
<section><h2>自动验收结果</h2><div class="table">{acceptance_table}</div></section>
{''.join(sections)}<section><h2>仍需团队完成</h2><p>原地交替踏步尚未实现；共享控制接口与正式 UI 接线
仍需组内对齐。这里的审阅页不替代工业设计最终人物动画与录屏。</p></section></main></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")
    packet_records = [{k: v for k, v in item.items() if k != "table_html"} for item in packets]
    result = {"playback_case_strategies": len(cases), "failure_packets": packet_records,
              "acceptance": acceptance["counts"],
              "renderer": "unchanged fogguard.cueing.review.write_guidance_review"}
    (output / "review_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("reports/automation2"))
    args = parser.parse_args()
    result = build_review(args.input_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Review: {args.input_dir.resolve() / 'review/index.html'}")


if __name__ == "__main__":
    main()

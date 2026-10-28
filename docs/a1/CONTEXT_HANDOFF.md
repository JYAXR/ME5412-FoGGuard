# A1 因果上下文实验交给 A2

方法见 [固定协议](CONTEXT_PROTOCOL_V1.md)。本轮为 FoG-only 传感器历史候选，不训练转向意图，也不改变共享接口、默认模型或控制器。

[实测结果](CONTEXT_RESULTS_V1.md)：F1 0.5994→0.6188、误报警事件 337→235，同时冻结召回下降、真实事件有重叠覆盖 77→70/101。候选仅保留用于研究，未替换原默认。279 项测试及独立历史/分折/60 文件哈希核验通过，候选 CLI 和 9,636 行事件管线接口通过。

## 本地产物

输出目录 `reports/a1_context_v1_20261028/` 的 `baseline_2s_v1/` 与 `causal_context_6s_v1/` 各提供：

- `a2_runtime.csv`：外层受试者留出预测，不含 truth_*。
- `evaluation_truth.csv`：独立保存的评估真值，不进入运行链路。
- `model.joblib`：全量拟合模型，只用于新输入；训练集回放不能作为测试成绩。
- `fold_audit.json`、`training_report.json`、`feature_schema.json`：分折、阈值和模型定义。
- `recognition/`：全部原始记录覆盖、活动/参与者分层及失败案例。

根目录包含 `history_links.csv`、预先记录方法/输入/源码哈希的 `protocol.json`、指标与分组 CSV、四张 PNG/SVG 展示图、`index.html`、`manifest.json` 及完成后的 `verification.json`。原始数据、模型和批量输出不进源码 Git。

## 新输入与历史状态

```powershell
python -X utf8 scripts/predict_a1_fog.py --input path/to/new_sensor_data.csv --model reports/a1_context_v1_20261028/causal_context_6s_v1/model.joblib --output reports/a1_context_new_scores.csv
```

新数据须为原左踝/腰背 12 通道、60 Hz，轴向与单位一致。整段 CSV 推理使用 `predict_frame`；标签会在窗口构造前去掉。输出为 FoG-only，不带活动真值和方向。A2 将方向预测按记录键和窗口时间对齐；`available_at_s` 等于窗口结束，不能回填到窗口起点。

逐窗推理应为每个输入流分别创建状态：

```python
from pathlib import Path
from fogguard.modeling.a1_fog import A1FogModel

model = A1FogModel.load(Path("reports/a1_context_v1_20261028/causal_context_6s_v1/model.joblib"))
stream = model.new_stream()
for completed_window in completed_windows:
    scores = stream.push(completed_window)
```

`completed_window` 是按时间提供的有效 `FogStarWindow`。每个流保存最多九个完成窗口，包含四秒前的二秒摘要，即最长六秒传感器历史。参与者/session/task 变化或 0.5 秒步进断档会清空历史；同记录倒序或重复输入报错。不要反复对单个窗口调用 `predict_windows([w])`，那会每次重启历史，也不要将不同实时输入源共用一个 stream。

历史不完整时不等待六秒，仍在原二秒窗口结束输出。`history_2s_available`、`history_4s_available` 表示两项历史是否可用；它们不是数据质量、冻结或恢复真值。历史不足也不能将其当作无预测或直接否决报警。

## 现有事件管线

```powershell
python -X utf8 scripts/run_fogstar_end_to_end.py --predictions reports/a1_context_v1_20261028/causal_context_6s_v1/a2_runtime.csv --output reports/a1_context_events.csv
```

候选 runtime 仍带明确标记的旧方向 OOF，`direction_source=unchanged_historical_baseline_oof`。该例仅用于验证兼容；A2 应替换为自己的方向输出。本轮评分阈值来自训练内三折，事件确认/退出迟滞仍需单独验证，不能把窗口指标直接当最终提示链路指标。

低 FoG 分数不是已经迈步的独立证据。此数据不能验证真实足部落点、迈步失败后的踏步响应或提示疗效。是否采用本候选以本轮完整结果和误报/召回取舍为准，原默认不自动替换。

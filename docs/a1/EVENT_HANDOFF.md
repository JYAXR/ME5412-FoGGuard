# A1 事件目标实验交给 A2

方法见 [固定协议](EVENT_PROTOCOL_V1.md)。本轮保留六秒上下文 RF，新增一个可选连续确认门；原默认模型、共享类型和原事件集成器均未替换。

[实测结论](EVENT_RESULTS_V1.md)：事件覆盖 70→71/101，但误报警时长增加 14.5%、event_fog 的窗口 F1 0.6188→0.6061，不采用新规则。288 项测试、65 个哈希及原始区间核验通过；传感器推理接口 30 行通过。最终产物由固定评分恢复报告，详见结果页的复现说明。

## 本地产物与字段

目录 `reports/a1_event_v1_20261029_verified/` 下，`window_f1/` 为原六秒上下文控制，`event_objective/` 为训练内事件目标选择的候选。各含：

- `a2_runtime.csv`：按人留出的运行输入，不含 truth_*；带历史方向 OOF，必须保留其来源标记。
- `evaluation_truth.csv`：隔离保存的真值。
- `oof_predictions.csv`：评估输入；性能仅使用未见该人的外层结果。
- `event_coverage.csv`、`event_alarms.csv`：event_fog 的全量原始事件/报警。
- `recognition/`：is_fog 原始阈值的独立审计。这里的事件数字与连续确认后数字可能不同，不能混用。
- `model.joblib`、`event_policy.json`：完整部署模型与对应阈值/确认数，须一起传递。

运行字段 `is_fog` 一直是原始 score>=fog_threshold；`event_fog` 是本轮确认后输出，不能将二者混为同一个指标。`event_confirmation_windows` 记录确认数；`available_at_s` 等于窗口结束，没有回填确认前的时刻。A2 的转向预测按记录键和窗口时间对齐，不读取活动真值。

根目录的 `fold_audit.json` 包含五折训练内全部 182 套规则及选择依据；`outer_*_inner_scores.csv` 保存未见对应参与者的内层评分。`deployment_selection.json` 为全队列另外一次内层选择，不是独立测试成绩。实验图片、CSV 和模型为本地产物，未加入源码 Git。

## 对新传感器 CSV 推理

```powershell
python -X utf8 scripts/predict_a1_events.py --input path/to/new_sensor_data.csv --model reports/a1_event_v1_20261029_verified/event_objective/model.joblib --policy reports/a1_event_v1_20261029_verified/event_objective/event_policy.json --output reports/a1_event_new_scores.csv
```

仍要求原左踝/腰背 12 通道、60 Hz，单位、轴向一致；标签在窗口构造前删除。该脚本输出 FoG-only 原始评分及 event_fog，没有运行时活动或冻结真值。

## 逐窗流式调用

```python
import json
from pathlib import Path
from fogguard.modeling.a1_fog import A1FogModel
from fogguard.modeling.a1_event import EventGate, EventPolicy

folder = Path("reports/a1_event_v1_20261029_verified/event_objective")
model = A1FogModel.load(folder / "model.joblib")
policy = json.loads((folder / "event_policy.json").read_text("utf-8"))
sensor_stream = model.new_stream()
event_gate = EventGate(EventPolicy(policy["threshold"], policy["confirmation_windows"]))
for completed_window in completed_windows:
    row = sensor_stream.push(completed_window).iloc[0]
    event_fog = event_gate.push(row.fog_probability,
                               (row.subject_id, row.session_id, row.task_id), row.window_end_s)
```

每个独立输入源各保留 sensor_stream 与 event_gate；新任务、断档会重置。原始无效传感器不能强行补零构造有效窗口；缺测必须由下游继续保留为未知状态。仅将流换成逐个单窗无状态调用会丢失上下文。

## 接入边界

现有 run_fogstar_end_to_end.py 读取原始 is_fog，并另外施加默认二次确认、退出迟滞和恢复规则。它不直接消费 event_fog。直接运行它能验证原始评分接口，但不能把其输出称为本轮事件规则的结果。A2 若采用本实验规则，应消费 event_fog 作为已确认检测，再单独处理方向及提示生命周期，并验证组合后输出，避免重复确认。

本轮训练内 110% 误报警时长预算不能当作测试或部署保证。最终是否采用须看完整事件/时间取舍；低 FoG 分数不证明已经迈步，没有真实踏步响应或提示疗效验证。

# 标准回放数据格式

每一行表示一个已经同步的 IMU 采样点。

| 字段 | 单位 | 是否必需 | 含义 |
|---|---|---:|---|
| `timestamp_s` | s | 是 | 从回放开始计算的单调时间 |
| `ax`, `ay`, `az` | m/s² | 是 | 三轴加速度 |
| `gx`, `gy`, `gz` | deg/s | 是 | 三轴角速度 |
| `fog_label` | 0/1 | 评估时 | FoG 真值标签 |
| `turn_label` | 文本 | 评估时 | `left`、`right`、`straight` 或 `unknown` |
| `scenario_id` | 文本 | 可选 | 场景分组标识 |

仓库中只允许保存合成数据或经过批准并去除身份信息的数据。不得提交参与者原始数据。每个数据集的说明文件中应记录传感器放置位置和标签生成假设。

加载和回放时执行以下最低质量检查：

- 必需传感器列和时间戳必须为有限数值，不接受 NaN 或 Inf。
- `timestamp_s` 必须严格递增，不能重复或倒退。
- 回放声明的采样率须与时间戳中位采样间隔一致，默认允许 5% 相对误差。
- `fog_label` 只能为 0/1；`turn_label` 只能为 `left`、`right`、`straight` 或 `unknown`。
- `scenario_id` 存在时不得为空。

## 窗口级 FoG 标签

`WindowReplay` 将采样点标签聚合到每个回放窗口，并通过 `SensorWindow.metadata` 提供：

- `fog_fraction`：窗口内 `fog_label=1` 的采样点比例。
- `fog_label`：当 `fog_fraction >= 0.5` 时为 `"1"`，否则为 `"0"`。

默认的 0.5 比例可以通过 `WindowReplay.fog_positive_fraction` 调整。元数据值保持为字符串，以遵守共享的 `SensorWindow` 接口；评估代码读取时应显式转换为 `float` 或 `int`。没有 `fog_label` 列的回放数据仍可用于演示，但窗口元数据不会包含上述字段。

## 窗口级转向标签

存在 `turn_label` 时，`WindowReplay` 将窗口内占比最高的方向写入元数据：

- `turn_fraction`：占比最高方向的样本比例。
- `turn_label`：当最高占比达到默认 0.5 且没有并列时，使用该方向；否则为 `unknown`。

存在 `scenario_id` 时，同一窗口内值一致则原样保留；混合来源窗口标为 `mixed`。这些标签只用于评估，不得作为自动识别或提示决策输入。

## FoG 事件证据

事件评估将样本级连续 `fog_label=1` 区间视为一个 FoG 事件。事件开始和结束来自样本时间戳；检测时间使用检测窗口最后一个样本的时间，即该窗口输出可用的时间。默认允许一个窗口长度的事件后匹配宽限，以保留由窗口造成的延迟输出。

事件报告包含事件开始、结束、首次检测时间、检测延迟和漏检状态。无事件场景中的阳性窗口单独记为事件外阳性窗口。

## 模拟落脚与响应输入

`data/demo/simulated_step_responses.csv` 与 IMU 数据分开保存，仅用于 P1 策略仿真：

| 字段 | 含义 |
|---|---|
| `scenario_id` | 模拟响应场景 |
| `timestamp_s` | 模拟事件时间 |
| `foot` | `left`、`right` 或 `none` |
| `forward_m`, `lateral_m` | 成功/延迟落脚时的示意坐标 |
| `response_status` | `success`、`delayed`、`unable`、`no_response` 或 `missing_input` |
| `delay_s` | 成功响应的模拟延迟 |
| `reliability` | 0–1 的输入可靠性配置，不代表临床概率 |
| `source` | 必须为 `simulated` |

成功或延迟落脚必须提供有限坐标和非负延迟。无响应、无法完成及缺失输入不得伪装成真实落脚，也不能用于证明提示产生治疗效果。

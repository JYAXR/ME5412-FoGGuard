# 标准回放数据格式

每一行表示一个已经同步的 IMU 采样点。

| 字段 | 单位 | 是否必需 | 含义 |
|---|---|---:|---|
| `timestamp_s` | s | 是 | 从回放开始计算的单调时间 |
| `ax`, `ay`, `az` | m/s² | 是 | 三轴加速度 |
| `gx`, `gy`, `gz` | deg/s | 是 | 三轴角速度 |
| `fog_label` | 0/1 | 评估时 | FoG 真值标签 |
| `turn_label` | 文本 | 评估时 | `left`、`right` 或 `straight` |
| `scenario_id` | 文本 | 可选 | 场景分组标识 |

仓库中只允许保存合成数据或经过批准并去除身份信息的数据。不得提交参与者原始数据。每个数据集的说明文件中应记录传感器放置位置和标签生成假设。

## 窗口级 FoG 标签

`WindowReplay` 将采样点标签聚合到每个回放窗口，并通过 `SensorWindow.metadata` 提供：

- `fog_fraction`：窗口内 `fog_label=1` 的采样点比例。
- `fog_label`：当 `fog_fraction >= 0.5` 时为 `"1"`，否则为 `"0"`。

默认的 0.5 比例可以通过 `WindowReplay.fog_positive_fraction` 调整。元数据值保持为字符串，以遵守共享的 `SensorWindow` 接口；评估代码读取时应显式转换为 `float` 或 `int`。没有 `fog_label` 列的回放数据仍可用于演示，但窗口元数据不会包含上述字段。

# 数据格式与来源

## FoG-STAR 原始信号

正式感知任务采用 60 Hz 同步 IMU，保存于 `data/raw/fogstar/sensor_data.csv`。下载入口为 `scripts/download_fogstar.py`，人员划分及数据哈希见 `config/fog_selected.json`。

| 列 | 单位/含义 |
|---|---|
| `subjectID / sessionID / taskID` | 记录身份；窗口不能跨记录 |
| `timestamp` | 秒，记录内严格递增 |
| `<position>_acc_x/y/z` | 加速度，g |
| `<position>_gyro_x/y/z` | 角速度，deg/s |
| `fog` | 训练/评价用二值冻结标签 |
| `activity` | 训练/评价用活动标签 |

位置包括 `ankleL / ankleR / back / wrist`。标签不进入运行控制；活动 1、6、7 分别映射直行、右转、左转，其他活动和不明确边界保留拒识语义。

## A1 输入

| 方法 | 所需列（另需 timestamp） |
|---|---|
| `ankle_history_rf` | 左踝三轴 acc 与三轴 gyro |
| `bilateral_rf` | 左踝和右踝三轴 gyro |
| `ankle_cnn` | 左踝三轴 gyro |

采样率 60 Hz，窗口 120 点、步长 30 点，输入轴向须与训练一致。窗口末端为最后一个采样点，即起点加 `119/60` 秒；完整两秒采样窗不等于首尾时间差恰好两秒。冻结样本占比至少 50% 才为窗口阳性标签。

A1 推理不插值缺测。双踝模型在左踝连续段固定网格上切窗，右踝缺失跳过对应窗口；历史特征在断档或记录切换时重置。缺失时段保留未知。

## 通用工程回放

通用 `SensorWindow`/演示 CSV 使用 `timestamp_s`、`ax/ay/az`（m/s²）和 `gx/gy/gz`（deg/s）。原始 g 转 m/s² 时必须显式乘以适配器采用的标准重力常数，禁止将两种格式直接拼接。

可选 `fog_label / turn_label / scenario_id` 用于评价或案例标识。真值输出单独保存，不能作为模型或控制器的输入特征。

## 反馈与模拟响应

B1 反馈目标字段和来源见 [接口约定](INTERFACE_CONTRACT.md)。真实 IMU 可以生成带来源的动作代理；公开数据不提供独立逐步向前位移真值。

工程模拟响应独立保存，主要字段为 `scenario_id / timestamp_s / foot / forward_m / lateral_m / response_status / delay_s / reliability / source`。当前加载器要求 `source=simulated`，成功/延迟响应要有脚别、有限坐标与非负延迟；缺失输入、无响应和无法完成不能伪装为成功落脚。

## 保存与共享

Git 只存代码、配置、合成接口样例与汇总指标。原始数据、模型、逐窗运行输出放在忽略目录。B2 的最终结果包分别记录真实信号、模型输出、运动代理、构造候选和模拟响应来源。

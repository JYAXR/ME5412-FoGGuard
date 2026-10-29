# 模块接口约定

本文区分已运行的模块字段与待接入的组合契约。A1 推理、事件聚合和模拟控制各有可运行接口；将新方向模型与 IMU 反馈接入完整系统，分别由 A2、B1 实现，B2 验收。共享 `types.py`、`interfaces.py` 的变更须经接收方评审。

## 公共身份与时间

每条记录使用 `subject_id / session_id / task_id`。原始 FoG-STAR 对应 `subjectID / sessionID / taskID`。时间单位为秒；换记录重置状态。`window_start_s / window_end_s` 表示输入范围，`available_at_s` 表示结果何时可被下游使用。

来源结果最多保持 0.5 秒，缺失或过期设为未知。提前转向模型的预测目标时间由 `horizon_s` 表达，结果可用时刻仍是实际输入末端或更晚，不得向过去回填。

## A1：已实现的冻结输出

| 字段 | 含义 |
|---|---|
| `fog_probability` | 未经临床概率校准的模型分数 |
| `fog_threshold / is_fog` | 模型阈值和二元候选 |
| `window_start_s / window_end_s / available_at_s` | 输入范围和可用时刻 |
| 三个记录键 | 结果所属记录 |
| `model_version / prediction_source` | 模型与运行来源 |

历史特征 RF 另有历史可用标志。无完整窗口时不产生有效预测，不能填一个假阴性。输出不含方向或迈步成功结论。

## A2：方向输出和事件适配

目标输出包含记录键、输入时间、可用时刻、`direction / direction_confidence / direction_task`。方向为 `straight / left / right / unknown`；当前动作分类和提前方向预测分开标记，提前任务另有 `horizon_s`。

现有 `FogStarTemporalIntegrator` 接收的最小列为记录键、`window_start_s / window_end_s / fog_probability / fog_threshold / is_fog / direction / direction_confidence`。`other` 映射为未知。该模块目前依赖 `window_end_s`，因此 A2 在交给它之前必须完成来源对齐、0.5 秒有效期检查，并拒绝尚不可用的结果；延迟产生的预测不得伪装成准时结果。

事件输出提供 `event_id / event_phase / stable_fog_detected / turn / turn_confidence / input_fresh / integration_reason`。当前事件层的恢复状态不代表真实迈步完成。

## B1：运动反馈目标字段

| 字段 | 含义 |
|---|---|
| `feedback_state` | `step_evidence / no_step_evidence / unknown` |
| `feedback_valid` | 当前观察质量与时效是否符合判据 |
| `feedback_source` | `measured / derived_proxy / simulated / unavailable` |
| `feedback_kind` | `forward / in_place / movement_only / unknown` |
| `feedback_available_at_s` | 反馈可用时刻 |
| `feedback_reason` | 判据、无效原因或证据说明 |

以上为接入目标，当前未实现统一 IMU 反馈估计器。`derived_proxy` 只承诺对应的运动证据；没有位置来源时 `feedback_kind` 使用 `movement_only` 或 `unknown`，不能写成 `forward`。没有足够有效观察时输出 `unknown`，不能计为没迈步。

当前 `StepResponse` 需要明确脚别和有限位置才能表达成功响应，CSV 加载器接受 `source=simulated`。B1 必须实现显式适配并验证新来源，不能给 IMU 代理虚构坐标或更改来源。

## B1：提示命令

现有模块内 `GuidanceCommand` 包含命令/事件 ID、时间、`action / control_state / reason`、脚印 `targets`、声音节拍与音量、视觉透明度及响应来源。保存 CSV 使用 `guidance_` 前缀。

`start / hold / update / stop` 表达生命周期。缺失输入撤销相关输出，切换和结束均记录原因；展示端从保存的命令读取这些字段，不重复做控制决策。

## B2：结果与展示包

建议交付 `perception_runtime.csv`、`feedback_runtime.csv`、`commands.csv`、独立的 `evaluation_truth.csv`、`metrics.json`、配置与哈希清单。文件名称是最终包约定，B2 负责组合导出器，当前单模块命令不会自动产生全部文件。

运行输入不得读取 `truth_*`。真实检测成绩、运动代理评价和模拟策略行为分表；没有对应真值的指标标记为不可评价。

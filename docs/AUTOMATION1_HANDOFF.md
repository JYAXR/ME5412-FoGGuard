# 自动化 1：脚印＋声音控制交接

更新：2026-10-21。实现分支 `codex/automation1-guidance`，依赖机器人 2 PR #23 的事件输出，基线提交 `af838ceb74a9098ea4ab360bedd2553de883492f`。

本文描述提交团队评审的实现；共享契约及正式管线接线待 #16 讨论。新增契约是提示模块内的 `guidance-proposal-v1` 提案，未修改 `types.py` / `interfaces.py` 的已共享类型，也未替换原 Streamlit 或原声音/触觉基线。依据任务 #4、#16、#17。

## 一条命令运行

按 README 建立 Python 环境后，在仓库根目录执行：

```bash
python scripts/run_guidance_demo.py --scenario all
```

不需要下载原始数据、训练模型或联网。生成 21 个场景的 `reports/guidance_<场景>.csv/.json`、固定对照 CSV、单因素对照 CSV、`guidance_summary.json` 和自包含 `guidance_review.html`。HTML 可直接在本地浏览器打开，包含场景选择、时间线、脚印、参数和按按钮试听的节拍；不实现另一套控制规则。

其他运行方式：

```bash
# 消费机器人 2 实际提供的 20 窗口人工候选样例，使用其真实集成器。
python scripts/run_guidance_demo.py \
  --predictions docs/examples/robot2_predictions.csv \
  --output-dir reports/handoff_review

# 消费已经集成的事件时间线。
python scripts/run_guidance_demo.py --events reports/robot2_handoff_demo.csv

# 单独展示一次备用、恢复和渐弱。
python scripts/run_guidance_demo.py --scenario persistent_no_response
python scripts/run_guidance_demo.py --scenario successful_recovery
python scripts/run_guidance_demo.py --scenario refreeze_during_fade
python scripts/run_guidance_demo.py --scenario recovery_interrupted

# 自适应间距独立开关，默认关闭；和固定间距在同一构造输入上对照。
python scripts/run_guidance_demo.py --scenario adaptive_spacing --adaptive-spacing

# 固定策略关闭响应驱动状态、备用、渐弱和自适应间距。
python scripts/run_guidance_demo.py --scenario all --fixed-only
python -m pytest -q
python -m ruff check .
```

每次运行会覆盖同一输出目录内的同名报告；需要保留对照实验时使用不同的 `--output-dir`。报告目录和模型/原始数据均不进入提交。

## 已交付能力

- 消费机器人 2 的 `event_phase/event_id/stable_fog_detected`、已接受方向和新鲜度；不读取 `truth_*` 作决策，不重复实现事件检测。
- 固定步幅和节拍的视觉＋声音同步输出；直行、左转、右转采用不同局部目标布局。
- start / update / hold / stop 命令生命周期；同一持续事件不逐窗重启播放。短暂回落在同一事件内保持控制。
- 默认静止不否决已确认 FoG，沿用 #23 的范围调整；主动停步误报作为观察案例保留。
- 方向未知、置信度不足、过期、异常时间等立即停止双模态；恢复后经两个有效 confirmed 输出重新启用，或回到正常/疑似后等待上游重新确认。
- P0 根据上游恢复观察结束退出。P1 另用可靠且已经发生的模拟落脚支持恢复/渐弱，绝不把评分回落解释为实测迈步。
- 无响应只启用一次 `3-2-1-go` 倒计时备用；倒计时结束回到固定节律。持续未完成在最大提示时长处失败停止，同一事件不会立即重启。
- 对连续可靠交替向前响应计数，满足条件后渐弱音量和透明度；重新 confirmed 恢复完整强度。
- 渐弱时遇到无法迈步、无响应、错误脚别或不足的向前位移，撤销稳定步数与渐弱依据；重新积累有效交替步数后才可再次渐弱。
- 可选的下一步间距调整：连续有效落脚达到稳定步数后启用，限制单次变化和个人配置范围，只在落脚时更新后续目标。
- 回放结束和会话切换显式停止未完成命令；停止原因区分恢复、异常、超时和输入结束。

原地踏步、真实定位/音频硬件、患者响应模型和最终人物动画不在此实现中。最终动画由工业设计读取控制时间线；共享在线类型接线由机器人 2 在 #16 对齐后进行。

## 控制状态和优先级

| 状态 | 条件与输出 |
|---|---|
| IDLE | 正常/疑似，无救援提示；等待 confirmed |
| GUIDING | confirmed 启动固定脚印＋节律；相同事件维持同一 command_id |
| EXIT_OBSERVE | 上游 recovery、无可用迈步证据；保持既有提示，观察结束即停止 |
| RESET_CUE | 明确模拟无响应/无法完成，或已配置且可用的响应源超时；一次倒计时备用 |
| RECOVERING | 模型已回落，同时有近期可靠交替向前响应；保持目标 |
| FADING | 达到稳定模拟步数，逐步降低透明度/音量；节拍和步幅不随渐弱改变 |
| SAFE_NONE | 核心输入异常或方向不可用；停止后等待重新满足条件 |
| FAILED | 最大提示时长达到；停止并锁定本事件，等新的事件 ID |

安全门控 → 会话/事件边界 → 时长上限 → 响应处理 → 恢复/备用决策。停止永远不受不应期阻挡。不应期只限制新 start，不能中断同一 episode 内的节拍，也不阻止 hold/update。

P1 若收到可靠响应，可在上游 normal 后有限延续恢复/渐弱；若缺少近期可靠步伐，就依照 P0 退出。最大提示时长覆盖所有活动阶段。完成渐弱后，如果同一上游事件重新 confirmed，允许按启动不应期重新提示；超时失败事件则保持锁定。

## 输入与响应语义

事件必需字段：`subject_id, session_id, window_end_s, event_id, event_phase, stable_fog_detected, turn, turn_confidence, input_fresh`。可选 `task_id/scenario_id` 缺省为 `default`，可选 `input_age_s` 缺省为 0；如果提供必须有效。事件行按受试者/会话/任务/场景连续排列，禁止交错会话。

`--predictions` 先调用机器人 2 的 `integrate_fogstar_predictions`。`--events` 读取该输出。新策略拥有自身的实际启动历史，原 `cue/seconds_since_last_cue` 仅保留为基线记录，不影响新策略。

模拟响应复用机器人 1 的加载校验器，必须 `source=simulated`。按 `scenario_id` 显式配对，一份响应场景只允许对应一个受试者/会话/任务，禁止跨受试者重用同一落脚记录。未匹配的响应场景报错。

本提案固定以下语义，需在 #16 与团队确认：

- `timestamp_s` 是模拟响应**可用时间**，只能在该时刻及之后消费一次；`delay_s` 是描述信息，不再次加到时间戳。未来响应无法改变当前输出。
- `forward_m/lateral_m` 视为该次落脚相对于配置步态局部参考的示意偏移，用 `forward_m` 做向前完成和可选间距判断；不把它当作全局轨迹，也不从相邻行求全局速度。
- 未配置响应源：固定策略，无响应相关自适应与备用关闭。已配置的离线响应源：假设自场景开始可用，直到显式 missing_input 或低可靠性记录中断；其静默才可作为响应超时。
- 响应缺失/不可靠：关闭响应驱动功能、保持固定提示，记录降级；核心识别数据无效：停止所有提示。两种故障不混同。
- 仅接受符合期望脚别、向前偏移达到配置下限且可靠的完成响应；非预期脚、碎步、重复/过旧记录不会伪造连续稳定步数。
- 直接调用控制器时，响应批次必须严格递增、不可重复，也不能包含未来记录；整批通过验证后才消费。去重游标跨同一会话内的提示启停保留，避免同一响应在新 episode 中重用。
- 不将几条预设成功记录解释为提示改善患者步态。原始响应表仅两条连续成功记录，不能单独完成稳定步数/渐弱验收，因此新增明确标注的构造用例。

## 输出契约提案

Python 入口为 `fogguard.cueing.replay.replay_guidance(events, config=..., responses=...)`。类定义在 `cueing/contracts.py`，当前是模块内提案；在 #16 讨论后再决定共享位置和正式编排接线。

输出保留输入列，并增加 `guidance_` 前缀字段：

| 字段组 | 含义 |
|---|---|
| schema_version | `guidance-proposal-v1`，表示尚待组内冻结 |
| timestamp_s / command_id / event_id | 输出可用时间、活动提示唯一 ID、所依据事件 |
| action | start 启动一次、hold 不重播、update 更新同一提示、stop 双模态撤销、none 无命令 |
| control_state / reason | 控制阶段与明确原因 |
| visual_enabled / visual_opacity | 视觉启用与清晰度；停止时 opacity=0 且 targets 为空 |
| audio_enabled / audio_volume / cadence_bpm / audio_mode | 声音参数；模式为 metronome 或 countdown_3_2_1_go，停止为 off |
| cue_started_at_s / mode_started_at_s | episode 和声音模式的共同时间基准，hold 不重置 |
| targets | JSON 数组：target_id、foot、x_m、y_m、yaw_deg |
| stable_steps / response_source / step_length_m | 模拟响应审计与有效步幅 |
| terminal | 回放结束或会话切换补出的停止记录，不是新的模型窗口 |

坐标原点是每次提示启动时的**配置示意原点**，x 向前、y 向左、yaw 以度为单位且左转为正。首步脚别、正常步幅、左右间距和圆弧半径来自配置。方向改变但没有落脚时保持当前整组目标；确认落脚后才采用新方向生成下一组目标。目标 ID 不随 hold 改变。

CSV 的 `targets` 是 JSON 字符串；逐窗 JSON 中同样保留这一格式，展示端解析一次即可。最后一条 terminal stop 可以和最后一个输入窗口具有相同时间，按文件顺序处理；它表示输入结束的截尾，不可统计为成功恢复。

## 配置与比较

`config/guidance.yaml` 由脚本实际加载。报告保存生效配置、输入/响应内容 SHA-256、基线提交号、控制代码 SHA-256 与运行库版本；未提交代码以内容哈希识别，不能仅靠基线提交号复现。所有参数值是演示工程参数。

- 首选 0.40 m 步幅、0.20 m 左右间距、80 BPM、音量 0.60；输入方向置信度门限 0.60，沿用新鲜度 1.0 s 和窗口间隔 0.75 s。
- 新启动不应期 3 s；episode 最大 15 s；可用响应源无进展 3 s 后启动一次 3 s 倒计时备用。
- 3 个连续交替可靠向前步后可渐弱，最大步间隔 1.5 s，渐弱 1 s。
- 可选自适应步幅限制为 0.32–0.55 m，单次最大 0.03 m。默认关闭；不会在目标尚未完成时移动脚印。

报告为每个场景运行同样事件和响应输入下的固定对照。`start_commands` 按 start 计数，hold 行数和节拍数均不计作新启动；`active_duration_s` 按同 command_id 的 start/stop 匹配计算。确认后到首次 start 的延迟与未提示事件单列。不提供无标签的“零误报”或临床疗效指标。实际检测准确率、误报/分钟等最终表由自动化 2 结合独立真值产生。

固定对照关闭整个响应策略，适合查看组合行为差异。为分别检查因素，启用渐弱时另存 `_no_fade.csv`（只关闭 `enable_fading`），启用自适应间距时另存 `_no_spacing_adjustment.csv`（只关闭 `enable_adaptive_spacing`）。两者与主策略消费完全相同的事件及响应；摘要的 `single_factor_references` 保存唯一改变的配置项、完整生效配置、文件名和行为计数，不能把组合差异归因于单一因素。

## 下游最小调用示例

```python
import pandas as pd
from fogguard.cueing.contracts import GuidanceConfig
from fogguard.cueing.replay import replay_guidance
from fogguard.data.responses import load_step_responses

events = pd.read_csv("reports/robot2_handoff_demo.csv")
commands = replay_guidance(events, config=GuidanceConfig())
# 仅在 scenario_id 与事件明确匹配时提供模拟响应：
# commands = replay_guidance(events, responses=load_step_responses("matched_responses.csv"))
commands.to_csv("reports/guidance.csv", index=False)
```

展示端按文件顺序消费 `guidance_action`：start 创建该 ID 的视觉/声音实例，hold 继续既有实例且不重置播放时钟，update 更新同 ID 参数，stop 撤销双模态。保留 terminal stop；相同时间戳按原文件顺序处理。声音模式切换读取 `mode_started_at_s`，基础节拍读取 `cue_started_at_s`；不要用旧基线 `cue` 再启动第二个播放器。

## 需要下游审阅的事项

1. 机器人 2：#16 控制契约、上下游 recovery 的语义、实际启动历史唯一归属及正式管线接线。提交时依赖 #23，避免覆盖他人的事件层。
2. 工业设计：坐标系、首脚配置、目标 ID、start/hold/update/stop，以及声音模式起点。试听按钮只预览四拍，不冒充最终实时播放器。
3. 自动化 2：固定对照、截尾停止、无提示事件、状态转换及失败锁定规则；构造候选、模拟响应和真实数据实验分开报告。

## 验证记录

2026-10-21 本地验证结果见下方最终记录；覆盖原基线 77 项与本模块新增测试。场景 CLI、机器人 2 的 20 窗口交接样例及可选自适应间距入口分别运行。

最终记录：**145 passed**（原基线 77 项＋新增 68 项），Ruff 与 `git diff --check` 通过；21 个场景全部完成且未关闭命令总数为 0。机器人 2 的 20 窗口交接样例输出 2 次新控制启动、2 次停止；可选间距调整和两类单因素对照导出通过。

新增行为测试覆盖重复触发、恢复/渐弱、重现冻结、坏时间、方向不可靠、源缺失、不应期、终止、因果性、真值不变性和步幅边界。审阅页已在浏览器检查左转目标、渐弱参数和倒计时模式，无控制台错误；它从生成结果绘制，不重新推断状态。

本次关联 #4/#16/#17，不自动关闭尚需组内对齐的任务；契约仍待团队冻结。PR 以机器人 2 分支为基线，便于只审查自动化 1 的改动；#23 合并后由维护者将目标分支改为最新 `main`，并在最终合并前完成组员互审。

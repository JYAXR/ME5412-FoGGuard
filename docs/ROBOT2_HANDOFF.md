# 机器人 2 情境识别与端到端集成交接

本文记录 Issue #3 的 P0 实现、输出契约、复现方式和已知边界。转向标签只用于事后评估，不进入识别或提示决策。

**下游现在可开工：** FoG-STAR 情境/事件集成已推送至 `feat/robot2-fogstar-integration`。请先阅读 [机器人 2 下游开工入口](ROBOT2_QUICKSTART.md)，其中提供无需模型的样例、复现命令、各角色交付与控制扩展依赖；合并状态以 GitHub PR 为准。当前已通过 77 项测试及 Ruff，机器人 2 转入控制与展示联调。

## 当前范围：FoG 步态提示优先

按项目负责人本轮调整，暂缓区分主观停步和 FoG，把工作重点放在 FoG 事件后的步态提示及恢复引导。默认 `suppress_stationary_fog=False`：继续记录传感器静止状态，但静止本身不再否决 FoG 事件、修改方向或阻止提示。模型持续确认、方向拒识、输入新鲜度和提示不应期仍然生效。

当前验收重点：持续 FoG 候选能进入提示流程；直走和左右转输出可供脚印与声音策略消费；检测输出回落后进入恢复观察并退出；短暂回落保留同一事件；输入过期或方向不可用时明确记录原因。机器人 2 提供情境、事件和控制接口，脚印＋声音、备用动作、渐弱与落脚响应的完整策略仍按团队分工由自动化 1 和工业设计接续完成。

`voluntary_stop` 保留为误报观察场景，其“零提示”暂不作为本轮完成条件。旧静止抑制可通过 `--suppress-stationary-fog` 显式启用作对照。当前代码识别的是传感器静止状态，不能仅凭 IMU 证明主观意图；本轮也不以疗效作为软件验收结论。

## FoG-STAR 新数据重新接入与验收（2026-10-21）

机器人 2 已按机器人 1 最新 FoG-STAR v3.0 交接重新实现端到端链路。默认链路依次执行：字段校验 → 方向拒识/平滑与运动状态记录 → FoG 持续确认与迟滞 → 恢复观察/事件合并 → `ControllerContext` → 控制器。真值字段只在最后的离线评估中读取。

### 固定的工程规则

- 连续 `2` 个模型阳性窗口才从 `suspected` 进入 `confirmed`；单窗阳性不提示。
- 退出阈值比每折模型阈值低 `0.10`，避免阈值附近反复切换。
- 首个阴性窗口进入 `recovery`，连续 `3` 个恢复窗口后退出；期间再次阳性并回同一个事件 ID。
- 相邻输出间隔超过 `0.75 s`、输入年龄超过 `1.0 s` 或时间倒序时，不沿用旧事件；输出 `safe_none` 或拒绝输入。
- FoG-STAR `direction=other` 始终映射为 `TurnDirection.UNKNOWN`。方向置信度低于 `0.60` 时同样拒识；方向还需连续两窗一致才成为最终 `turn`。
- `YawRateTurnRecognizer` 的输出以 `yaw_turn/yaw_turn_confidence` 保留作对照。它在本数据轴向和默认阈值下覆盖率高但左右转区分很差，因此默认不参与仲裁；只有显式设置 `use_yaw_agreement_fallback=True` 才启用“一致才回退”。
- 控制器现在必须同时看到 `DetectionResult.is_fog=True` 和足够评分，避免高分但尚未确认的单窗直接触发。

FoG-STAR 时间线的关键输出如下；输入中的 `state` 仍只是模型级显示字段：

| 字段 | 含义 |
|---|---|
| `fog_probability`, `fog_threshold`, `is_fog` | 原始评分、该留出折阈值和原始模型判定 |
| `direction`, `direction_confidence` | 原始 FoG-STAR 方向及置信度 |
| `yaw_turn`, `yaw_turn_confidence` | 保留作对照的角速度基线，不默认替换模型方向 |
| `motion_state`, `motion_confidence`, `dynamic_accel_rms_mps2` | 独立静止代理及物理证据 |
| `fog_suppressed`, `effective_model_fog` | 可选静止抑制及有效候选；默认不启用静止抑制 |
| `event_phase`, `event_id`, `stable_fog_detected` | 因果时序状态、合并事件和最终控制输入 |
| `turn`, `turn_confidence`, `turn_source` | 拒识和平滑后的控制情境及来源 |
| `input_fresh`, `input_age_s`, `integration_reason` | 数据质量与每次状态变化原因 |
| `cue`, `cue_reason`, `seconds_since_last_cue` | 控制器输出、不提示原因和真实提示历史 |
| `truth_*` | 仅供离线验收的标签，时序集成器不读取这些字段作决策 |

### 静止代理及边界

FoG-STAR 链路保留左踝动态加速度 RMS 静止判断，工程阈值为 `0.30 m/s²`。默认仅记录它；只有显式开启旧静止抑制策略时，静止才会令 `fog_suppressed=true` 并阻止该窗口进入事件或提示。

这里识别的仍然只是传感器的“静止状态”，不是患者的主观意图。即使数据标签为 `Stand`，也不能仅凭 IMU 证明患者是主动停步；同样，真正的运动不能排除 FoG。该代理在本次 FoG-STAR 留出预测中覆盖 `55.4%` 的站立窗口，在行走窗口上的静止率为 `0.19%`，因此不能被描述为已经可靠解决“主动停步与 FoG”的临床区分问题。

### 无受试者泄漏的结果

训练脚本现在额外保存五折、按受试者隔离的 `reports/fogstar_oof_predictions.csv`。机器人 2 的指标来自这些留出预测，而不是最终模型对训练数据的回放。默认参数结果为：

| 指标 | 结果 |
|---|---:|
| 真值窗口事件 | 81 |
| 已确认事件 | 69 |
| 事件召回率 | 0.852 |
| 中位确认延迟 | 0.50 s |
| 平均确认延迟 | 0.754 s |
| 误报事件 | 119 |
| 误报事件/分钟 | 1.440 |
| 最终方向覆盖率 | 0.606 |
| 已接受方向 macro-F1 | 0.957 |

默认关闭静止抑制后，提示总数为 201 次；7,739 个非 FoG 窗口上仍有 69 次提示。1,283 个非 FoG `Stand` 窗口中，原始模型报阳性 381 个，静止规则抑制 0 个，产生 1 次提示。静止抑制开关影响事件结果，但这批回放中的实际提示总数没有变化，因为方向拒识与不应期仍起作用。真实直行 FoG 的方向经常被拒识，因此也存在无提示的事件；这些失败继续记录，不能只引用事件召回率。

### 复现

```bash
python scripts/download_fogstar.py
python scripts/train_fogstar_baseline.py
python scripts/evaluate_fogstar_integration.py
python scripts/run_fogstar_end_to_end.py --predictions reports/fogstar_oof_predictions.csv
pytest -q
ruff check .
```

生成但不纳入 Git 的文件包括模型、机器人 1 训练报告、留出预测、机器人 2 逐窗时间线和集成指标 JSON。

## 历史交接记录

以下记录保留旧实现与旧参数的验收证据。其中“静止即抑制”和 `voluntary_stop` 零提示结论只适用于显式开启旧策略的对照运行；当前默认行为与范围以上文为准。

## 来自机器人 1 的新模型输入（2026-10-21）

机器人 1 已新增 FoG-STAR 多任务基线，能够逐窗口输出 FoG 概率/判定和 `straight`、`left`、`right`、`other` 方向候选。该输出是机器人 2 情境层的新输入选项，不自动替代本文件中的 `YawRateTurnRecognizer`。

机器人 2 接入时应：

- 将 `other` 映射为 `TurnDirection.UNKNOWN`，保留拒识语义。
- 保留原始 `fog_probability` 和 `direction_confidence`，不要只传最终 `state`。
- 明确选择“对照、回退、仲裁或替换”策略，并在报告中记录。
- 完成时间平滑、事件级评价、`ControllerContext` 映射和提示安全门控后，才允许进入端到端链路。

完整模型说明和字段契约见 `docs/FOGSTAR_MODEL.md` 与 `docs/ROBOT1_HANDOFF.md`。

> [!IMPORTANT]
> 本次工作新增了静止状态识别：系统根据 IMU 加速度活动量输出
> `motion_state=stationary`，并在静止状态与原始 FoG 阳性冲突时抑制救援提示。
> 系统识别的是传感器呈现的“静止状态”，不能仅凭 IMU 证明患者的主观意图
> 确实是主动停步。因此，本文将静止状态作为合成 `voluntary_stop` 场景的可测代理，
> 不把它描述为已经完成真实患者主动意图识别。

## 已交付内容

- `YawRateTurnRecognizer` 输出直行、左转、右转或未知，并提供 `[0, 1]` 置信度及 `0.5 s` 时间范围。
- 无陀螺仪、非有限角速度或距离阈值过近时输出 `unknown`，不把占位零值解释为直行。
- `AccelerationMotionRecognizer` 以动态加速度 RMS 区分行走、静止和未知；静止是主动停步的可测代理，不宣称能够推断人的主观意图。
- 静止与原始 FoG 阳性冲突时保留原始检测记录，设置抑制标记，并阻止控制器发出冻结救援提示。
- `fogguard.pipeline` 提供 UI、评估脚本和命令行共用的唯一回放链路。
- 管线实际传入上次提示时间，使现有控制器的不应期生效。
- `scripts/run_end_to_end.py` 一次处理七个 P0 场景，生成机器可读 CSV。
- Streamlit 页面改为调用相同管线，不再维护独立的识别与控制循环。

## 转向规则

默认阈值为 `20 deg/s`，取每个窗口 z 轴角速度的平均值：

| 输入 | 输出 |
|---|---|
| 陀螺仪不可用或角速度包含非有限数值 | `unknown`，置信度 0 |
| 平均角速度绝对值低于阈值 | `straight` |
| 平均角速度大于等于正阈值 | `left` |
| 平均角速度小于等于负阈值 | `right` |
| 与阈值的归一化距离小于 `0.10` | `unknown`，避免强行分类边界窗口 |

置信度表示当前结果到判定边界的归一化距离，不是经过校准的正确概率。`horizon_s=0.5` 对应本项目窗口步长，只表示结果覆盖范围，不代表已验证的提前转向预测。

## 主动停步规则

默认取三轴加速度去除各轴均值后的合成 RMS：

| 输入 | 运动状态与控制行为 |
|---|---|
| RMS 低于 `0.15 m/s²` | `stationary`；若原始 FoG 为阳性，设置 `fog_suppressed=true` 且不提示 |
| RMS 高于阈值 | `walking`；FoG 和转向结果继续进入提示控制器 |
| 与阈值的归一化距离小于 `0.10` | `unknown`；不声称可靠区分静止与行走 |

该规则只在本仓库确定性合成场景上完成 P0 验证。它识别的是传感器静止状态，不足以在真实患者数据中证明停步是主动行为。

## 一条命令复现

完成 README 中的环境安装后运行：

```bash
python scripts/run_end_to_end.py
```

默认输出为 `reports/robot2_end_to_end.csv`。也可以指定一个或多个输入：

```bash
python scripts/run_end_to_end.py \
  --input data/demo/turning_fog.csv \
  --input data/demo/right_turning_fog.csv \
  --output reports/selected_replay.csv
```

## 输出契约

| 字段 | 含义 |
|---|---|
| `window_start_s`, `window_end_s` | 窗口时间范围 |
| `fog_probability`, `fog_detected` | FoG 模型评分与阈值结果；评分未经概率校准 |
| `motion_state`, `motion_confidence`, `dynamic_accel_rms_mps2` | 独立运动状态及其活动量证据 |
| `fog_suppressed`, `effective_fog_detected` | 静止冲突是否被抑制，以及进入提示决策的有效 FoG 结果 |
| `turn`, `turn_confidence`, `turn_horizon_s` | 情境识别输出 |
| `gyro_available` | 是否有实测陀螺仪数据 |
| `cue`, `cue_intensity`, `cue_duration_s`, `cue_reason` | 控制器输出与原因 |
| `seconds_since_last_cue` | 管线传给控制器的提示历史 |
| `truth_fog_label`, `truth_turn_label` | 仅用于评估的窗口真值 |
| `fog_matches_truth`, `effective_fog_matches_truth`, `turn_matches_truth` | 原始 FoG、有效 FoG 和转向的事后对比，不参与决策 |
| `scenario_id`, `source_file`, `source_dataset` | 场景和数据来源 |

## P0 场景结果

使用仓库中的确定性合成数据、FoG 阈值 `0.21`、转向阈值 `20 deg/s` 运行：

| 场景 | 转向结果 | FoG 冲突 | 提示行为 |
|---|---|---|---|
| `normal` | 57/57 直行匹配 | 无 | 不提示 |
| `straight_fog` | 57/57 直行匹配 | 无 | 事件内提示并受不应期限制 |
| `turning_fog` | 57/57 匹配，含 2 个真值与输出均为未知的边界窗口 | 无 | 左转 FoG 期间提示 |
| `right_turning_fog` | 57/57 匹配，含 2 个真值与输出均为未知的边界窗口 | 无 | 右转 FoG 期间提示 |
| `normal_right_turn` | 57/57 匹配，含 2 个真值与输出均为未知的边界窗口 | 无 | 不提示 |
| `voluntary_stop` | 57/57 直行匹配；7 个完整静止窗口被识别 | 原始 FoG 的 7 个误报全部记录并抑制 | 有效 FoG 0 个，提示 0 次 |
| `fog_recovery` | 57/57 直行匹配 | 恢复边界多 1 个阳性窗口 | 提示后退出；完整恢复状态机仍由自动化 1 完成 |

边界窗口的窗口真值本身因直行/转向样本数量相同而为未知，识别输出也按低置信度安全降级为未知；模型没有读取标签。主动停步仍保留已有 FoG 检测基线的 7 个原始误报，但运动状态门控将其全部标记为冲突并抑制，因此不再发出错误提示。原始结果和有效结果同时输出，自动化 2 可以检查冲突没有被隐藏。

## 下游交接

- **自动化 1：** 使用 `motion_state`、`fog_suppressed`、`turn`、`turn_confidence` 和 `cue_reason`；静止冲突和 `unknown` 转向必须保持无提示或明确安全降级。
- **工业设计：** 直接读取端到端 CSV 或调用 `replay_csv`，同时展示模型结果、真值和原因，不能把真值当作实时识别结果。
- **自动化 2：** 使用 `*_matches_truth`、`fog_suppressed`、提示次数和时间字段做回归；主动停步的原始误报与恢复边界阳性必须保留，主动停步的有效 FoG 和提示必须为 0。

## 尚未包含

- 真实转向数据验证和提前转向预测。
- 真实患者数据上的主动意图识别；当前实现只识别低活动静止状态。
- 脚印＋声音完整状态机、恢复观察和 P1 踏步备用。
- 真实硬件、足部定位或临床验证。

# 机器人 1 数据与 FoG 交接说明

更新日期：2026-10-19。项目剩余两周，因此机器人 1 的范围收敛为：立即交付稳定的 P0 数据管线、公开真实数据基准和失败证据；随后只提供最小、来源透明的 P1 模拟响应输入，并协助最终评估。新的参与者招募或数据采集不在两周范围内，但公开真实数据验证属于必做项。

## 已交付基础

- 严格校验必需传感器列、有限数值、时间戳、标签和采样率。
- 保留 2 秒窗口、0.5 秒步长和窗口内至少 50% FoG 样本的标签规则。
- 在窗口元数据中提供 `fog_label`、`fog_fraction`、`turn_label`、`turn_fraction` 和 `scenario_id`。
- 保留 Freeze Index 与阈值 0.21 作为工程基线，不重新包装成临床概率。
- 增加事件级 FoG 开始、结束、首次检测、延迟和漏检记录。
- 增加与 IMU 数据分开的模拟落脚/响应 CSV，覆盖成功、延迟、无法完成、无响应和缺失输入。
- 增加 UCI Daphnet 官方数据下载、校验、格式映射和按参与者留一验证，不把同一参与者拆到训练与测试两侧。
- 真实数据没有陀螺仪时显式标记 `gyro_available=false`，阻止转向模块把占位零值误认为直走。

## P0 场景矩阵

| 场景 | FoG | 转向 | 主要用途 |
|---|---|---|---|
| `normal` | 无 | 直走 | 正常负例 |
| `straight_fog` | 有 | 直走 | 直行冻结检测 |
| `turning_fog` | 有 | 左转 | 左转冻结与提示方向 |
| `right_turning_fog` | 有 | 右转 | 右转冻结与提示方向 |
| `normal_right_turn` | 无 | 右转 | 检查正常转弯不触发冻结救援 |
| `voluntary_stop` | 无 | 直走 | 主动停步误报检查 |
| `fog_recovery` | 有后恢复 | 直走 | 事件退出和恢复观察 |

上表文件均为 30 秒、50 Hz、1500 个采样点的确定性合成信号，只验证工程行为。FoG 跨参与者表现另用 Daphnet 真实患者数据检查，见下节。

## 真实数据证据

使用 Daphnet 踝部加速度、2 秒窗口、0.5 秒步长和 LOSO 验证后，十折合并结果为：precision 0.185、recall 0.634、specificity 0.700、F1 0.287、balanced accuracy 0.667。237 个 FoG 事件中检出 219 个，已检出事件的平均首次检测延迟为 0.779 秒；与此同时，非 FoG 阳性重叠窗口达到 30.95 个/分钟。

这说明当前 Freeze Index 能覆盖多数事件，但误报太多，尚不能直接驱动真实提示。合成阈值 `0.21` 仍只是演示参数；真实数据各训练折所选阈值中位数 `0.73` 也不能直接成为生产默认值。完整参与者级结果、数据来源和限制见 [Daphnet 验证记录](DAPHNET_VALIDATION.md)。

## 当前基线证据

阈值 0.21 下：

- 正常、正常右转场景没有 FoG 阳性窗口。
- 直行 FoG、左转 FoG、右转 FoG 的窗口分类在当前合成数据上均无误报或漏报；首次检测约在事件开始后 0.98 秒可用。
- `voluntary_stop` 出现 7 个误报窗口，是当前基线最重要的失败案例。
- `fog_recovery` 在窗口级多报 1 个阳性，该窗口位于恢复边界附近，提示退出需要观察期和明确时序规则。

这些结果说明现有检测器可作为可解释基线，但不能单独区分主动停步与 FoG。后续不得删除失败案例或通过标签泄漏规避它。

## 复现命令

```bash
python scripts/generate_demo_data.py
python scripts/generate_simulated_responses.py
python scripts/evaluate_fog_baseline.py
python scripts/evaluate_fog_events.py
python scripts/download_daphnet.py
python scripts/evaluate_daphnet.py
pytest -q
ruff check .
```

## 交给下游的内容

### 机器人 2

- 使用窗口元数据中的 `turn_label` 作为评估真值，不作为模型输入。
- 验证 `gz>0` 对应左转、`gz<0` 对应右转，并覆盖 `unknown`。
- 把 `voluntary_stop` 和 `fog_recovery` 接入端到端流程。

### 自动化 1

- 使用 `voluntary_stop` 验证控制器不会仅凭单一分数无限重复救援提示。
- 使用 `fog_recovery` 检查恢复观察、退出和再次触发时序。
- P1 读取模拟响应文件；缺少或低可靠性输入时关闭相应自适应功能。
- 根据 Daphnet 的高误报结果，提示触发必须包含持续性确认、事件合并、不应期和恢复观察；不能把每个阳性窗口当成一次新提示。

### 自动化 2

- 窗口指标继续调用 `evaluate_fog_detector`。
- 事件指标调用 `evaluate_fog_events`，并单列漏检与事件外阳性窗口。
- 把主动停步误报作为代表性失败案例，不只报告合成 FoG 场景的满分结果。
- 最终报告必须同时列出 Daphnet LOSO 的参与者级差异，并把非 FoG 阳性窗口率与实际提示次数分开统计。

### 工业设计

- 页面必须区分模型评分、真实标签、配置参数和模拟响应。
- `probability` 字段在页面上显示为“FoG 评分（未校准）”。
- 明确标注模拟落脚不是传感器自动识别结果。

## 本次没有代替其他角色完成的内容

- 没有修改共享 `ControllerContext` 或 `CueCommand` 契约。
- 没有实现脚印＋声音状态机、恢复退出或不应期集成。
- 没有实现真实足部定位、真实提示响应模型或临床验证；Daphnet 只验证离线 FoG 检测。
- 没有因新增场景直接改变默认阈值；如需重标定，应提交独立证据。

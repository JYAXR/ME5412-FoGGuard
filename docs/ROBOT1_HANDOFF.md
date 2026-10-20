# 机器人 1 数据与 FoG 交接说明

更新日期：2026-10-21。项目剩余两周，因此机器人 1 的范围收敛为：立即交付稳定的 P0 数据管线、公开真实数据基准和失败证据；随后只提供最小、来源透明的 P1 模拟响应输入，并协助最终评估。新的参与者招募或数据采集不在两周范围内，但公开真实数据验证属于必做项。

## 本轮新增交付：FoG-STAR 联合识别模型

机器人 1 当前工作的结束点是：把可复现的数据处理、训练、按受试者验证、模型产物和机器可读推理结果交给机器人 2。机器人 1 提供的方向分类是模型侧候选证据，不代表接管机器人 2 的情境识别、时序状态机或端到端系统集成职责。

### 机器人 1 已完成

- 使用 FoG-STAR 公开数据，下载脚本会核验官方 MD5。
- 以受试者、会话、任务为边界切窗，窗口不会跨越这些边界或长缺失段。
- 输入为左脚踝和下背部共 12 路 IMU；同时保留原始姿态信息和动态频带特征。
- 训练两个输出头：FoG 概率/判定，以及 `straight`、`left`、`right`、`other` 情境。
- 用受试者分组的 5 折验证，避免同一受试者同时出现在训练和验证中。
- 产出可加载模型、JSON 评估报告和面向新 CSV 的批量推理脚本。

关键交付文件：

- `src/fogguard/data/fogstar.py`
- `src/fogguard/modeling/fogstar_baseline.py`
- `scripts/download_fogstar.py`
- `scripts/train_fogstar_baseline.py`
- `scripts/predict_fogstar_baseline.py`
- `artifacts/fogstar_multitask_baseline.joblib`（本地生成，不纳入 Git）
- `reports/fogstar_multitask_baseline.json`（本地生成，不纳入 Git）
- `docs/FOGSTAR_MODEL.md`

本次交接所用本地产物校验值：

- 模型 SHA256：`17F71663DD6B637019FC4526EB14A6B73016D5A121BC5149D7C5EF37CD89FDBC`
- 报告 SHA256：`0B9CCE86A5AA8A0820DFF1D25317042CBD07D363CB1AC35F3FEF55405B761B64`

重新训练会产生新的校验值；机器人 2 若不直接接收上述模型文件，应使用相同数据版本和训练命令重新生成。

复现命令：

```bash
python scripts/download_fogstar.py
python scripts/train_fogstar_baseline.py
python scripts/predict_fogstar_baseline.py path/to/new_gait.csv
```

当前完整数据基线结果：FoG macro-F1 为 `0.749`，四状态 macro-F1 为 `0.762`。这组结果用于工程基线和后续比较，不应被描述为临床验证结论。

### 交给机器人 2 的输出契约

推理 CSV 每行对应一个窗口，机器人 2 可以直接消费以下字段：

| 字段 | 含义 | 机器人 2 的使用约束 |
|---|---|---|
| `window_start_s`, `window_end_s` | 窗口时间范围 | 用于排序、合并事件和对齐其他传感器 |
| `fog_probability` | 原始 FoG 概率 | 必须保留，不能只保留最终状态 |
| `is_fog` | 按训练期校准阈值得到的 FoG 判定 | 作为冻结候选事件输入 |
| `direction` | `straight/left/right/other` | `other` 必须映射为未知或拒识，不能强制改成三类之一 |
| `direction_confidence` | 当前方向预测置信度 | 可用于拒识和多模块仲裁 |
| `state` | FoG 优先的四状态结果 | 仅是模型级结果，不能直接触发提示 |
| `subject_id`, `session_id` | 数据来源标识 | 用于审计、分组和避免跨会话合并 |

### 机器人 2 的接入任务

1. 将 `direction` 映射到现有 `TurnDirection`，并把 `other` 映射为 `UNKNOWN`。
2. 保留现有 `YawRateTurnRecognizer` 作为可比较的基线或回退路径，明确记录最终采用、仲裁或替换策略。
3. 增加时间平滑、迟滞、最短持续时间、事件合并、恢复期和输入新鲜度检查。
4. 将稳定后的结果接入 `ControllerContext`、控制器、回放流程和 UI；模型单窗输出不能直接驱动提示。
5. 同时记录原始概率、模型判定、情境输出和最终控制决策，保证可追踪。
6. 在真实提示逻辑启用前补充事件级灵敏度、误报率、检测延迟和跨受试者测试。

### 明确不属于机器人 1 本轮范围

- 控制器和提示状态机的修改。
- UI、ROS 或实时采集链路的端到端接入。
- 机器人 2 现有情境识别模块的无记录替换。
- 把窗口级研究结果表述为临床安全结论。

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

# ME5412 FoGGuard

> **2026-10-28 A1 困难负样本实验完成：** [结果与图表](docs/a1/HARD_NEGATIVE_RESULTS_V1.md) · [固定方法](docs/a1/HARD_NEGATIVE_PROTOCOL_V1.md) · [A2 接口说明](docs/a1/HARD_NEGATIVE_HANDOFF.md)。训练内高分纯负样本加倍权重后，F1 从 0.5994 降至 0.5792，误报警事件增多；该候选不采用，原默认模型保持不变。

> **2026-10-28 A1 多模型比较完成：** [RF/SVM/HistGB/CNN 实测](docs/a1/MULTIMODEL_RESULTS_V1.md) · [方法](docs/a1/MULTIMODEL_PROTOCOL_V1.md) · [A2 交接](docs/a1/MULTIMODEL_HANDOFF.md)。本轮固定配置中 RF 的 F1 最高，深度模型未显示升级依据，原默认模型保持不变。

> **2026-10-28 A1 首轮实验：** [方法](docs/a1/EXPERIMENT_V1.md) · [实测结论](docs/a1/RESULTS_V1.md) · [交给 A2](docs/a1/A2_HANDOFF.md)。近期特征的改善幅度小且存在事件覆盖取舍，保留原默认模型，新候选作为对照交付。

> **2026-10-28 算法改进前置核查：** [A1 开工入口](docs/algorithm_readiness/START_HERE.md) · [本轮实测结果](docs/algorithm_readiness/RESULTS_2026-10-28.md)。包含原基线重训、全量数据支持审计、转弯前标签可行性、接口样例与 PPT 图表；新迈步反馈与意图模型尚未实现。

> **自动化 2 新交付（2026-10-23，待组内评审）：** [从这里开始](AUTOMATION2_START_HERE.md) · [AI 接手说明](docs/automation2/AI_HANDOFF.md) · [机器可读状态](docs/automation2/status.json)。
> 包含策略比较、可选踏步仿真、真实数据识别与缺测分析。依赖 PR #24；不表示已合并 main，也不表示模型已达标。


自动化 1 的脚印＋声音仿真控制实现已提交本分支，待组内评审：

```bash
python scripts/run_guidance_demo.py --scenario all
```

无需下载模型，生成控制时间线、固定策略对照和离线 `reports/guidance_review.html`。
新增控制接口仍为模块内提案；用法、字段及恢复/失败规则见
[自动化 1 交接](docs/AUTOMATION1_HANDOFF.md)。

机器人 2 的 FoG-STAR 情境与事件链路已交付至 `feat/robot2-fogstar-integration`。后续组员请从 [下游开工入口](docs/ROBOT2_QUICKSTART.md) 获取最小样例、接口与任务安排；无需下载原始数据即可开始控制、评估和展示开发。是否已合并进 `main` 以 PR 状态为准。

FoGGuard 是一个 ME5412 小组项目：研究人在步态冻结、重新启动和恢复行走时，系统怎样用投影脚印与声音节奏提供合适的提示。FoG 是“冻结步态”的英文缩写，指想继续走却难以迈步的现象。

本期做数据回放和提示策略仿真，面向项目组与课程评审展示。工程场景使用合成数据，FoG 检测另外使用公开、去标识的 UCI Daphnet 真实患者数据做参与者级验证；项目不自行招募或采集患者数据，也不据此证明治疗效果。

## 我们希望完成什么

| 行走过程 | 目标提示行为 |
|---|---|
| 冻结、难以迈步 | 给出预设可完成的脚印目标和声音节奏 |
| 仍无法启动 | 可切换左右交替踏步提示，有响应后再引导向前；并非每次都必须踏步 |
| 恢复行走 | 根据最近几步的完成情况调整下一步脚印，保持在个人预设范围内 |
| 转弯 | 脚印随转弯方向排列，同时考虑朝向和左右间距 |
| 连续稳定行走 | 按设定规则逐渐减弱脚印清晰度和声音音量；保留持续提示选项 |
| 再次冻结 | 恢复已配置的提示，不自动加快节拍或要求更大步 |

正在迈向的脚印保持不动，确认落脚后再更新下一步。踏步备用、间距调整和渐弱都是待验证设计，不能把它们写成已证实有效的治疗方法。

## 已经有了什么，还需要做什么

以下为本分支的实现状态（2026-10-21）；进入 `main` 的状态以 PR 合并为准。

| 部分 | 已有基础 | 接下来要做 |
|---|---|---|
| 数据与检测 | 合成步态场景、Daphnet 导入、冻结评分及参与者留一验证 | 根据真实数据结果明确失败边界，交给控制与评估负责人处理事件合并和误报 |
| 行走情境 | 直走和左右转弯的简单判断 | 检查正常转弯、右转及无法确定情境的情况 |
| 提示控制 | 保留声音/触觉基线；新增固定脚印＋声音、生命周期、倒计时备用、模拟恢复/渐弱及可选间距调整 | 在 #16 评审组合控制契约，由机器人 2 接入共享管线，工业设计消费保存输出 |
| 展示与验证 | Streamlit 曲线、结果表和基础测试 | 展示提示变化与原因，完成策略比较和完整流程检查 |

本分支已经实现模拟落脚驱动的下一步间距调整和稳定后渐弱；备用路径采用一次倒计时。可选原地踏步尚未实现。真实步长、足部位置和稳定性不能直接从现有输出中获得；完整行为与验证记录见[自动化 1 交接](docs/AUTOMATION1_HANDOFF.md)。

## 分两步实现完整仿真

**完整仿真目标包含踏步备用、下一步间距调整、稳定后渐弱及冻结后恢复。先打通基础流程，再接入这些交互。** 本文用 P0、P1 表示开发顺序；功能是否完成以验收记录为准，不能把阶段一完成写成全部完成。

- **阶段一（P0）：基础流程。** 继续使用现有数据检测冻结与转向，展示固定配置的脚印和声音状态，说明触发、退出和异常处理的原因，并完成检测与提示时序评估。
- **阶段二（P1）：完整交互。** 用明确标注的模拟落脚与响应，接入踏步备用、下一步间距调整、稳定后渐弱与冻结后恢复；分别检查成功、无响应、延迟响应及缺失输入。没有所需输入时关闭相应自动调整。
- **后续方向（Future）。** 新的人体数据采集、真实足部定位、投影硬件和人体研究不属于本期任务；本期必须完成公开真实数据的离线验证。起步前或转弯前的提前提示、提示改变后续步态的响应模型仍为可选探索。

两阶段分别验收，先保留可运行的基础版本。具体实现与工作量由原负责人衔接；若进度需要缩减范围，应同步团队并明确未完成项，不能默认为完整目标已经实现。

## 仿真最后怎么展示

**最终仿真形式待团队确定。** 现阶段明确的是：观众能看懂当前行走状态、脚印与声音怎样变化、为什么变化，并能查看过程记录和评估结果。

是否使用二维或三维人物、是否允许手动操控、是否随机生成场景，以及声音实际播放的方式，均待后续选择；这些讨论中的形式暂不作为新增验收要求。现有 Streamlit 页面是基础，不等于最终表现形式已定。

## 数据怎么用，结果怎么判断

合成数据记录加速度、角速度和冻结/转向标签，用于检查完整工程流程；Daphnet 提供真实患者加速度与 FoG 标签，用于检查检测基线能否跨参与者泛化。数据格式见 [DATA_SCHEMA](docs/DATA_SCHEMA.md)，真实数据实验见 [Daphnet 验证记录](docs/DAPHNET_VALIDATION.md)。

模拟落脚和响应用于检查“提示后系统怎样调整”，需要另外提供并标明来源。它们不能冒充自动识别结果，也不能预设“提示后一定恢复”再将其当作疗效证据。

项目围绕三个问题给出结果：

1. 冻结检测在哪些情况判断正确，哪些情况误报、漏报或延迟？
2. 直走和左右转弯能否区分，无法确定时怎样处理？
3. 与固定提示相比，自适应方案在触发、方向、间距、切换和退出上有什么差异？

现有回放数据不会因提示而改变；仿真主要验证系统行为。文献支持分别考虑起步、行走和转弯，但未证明本项目组合策略有效，依据与验收细节见[系统架构与需求](docs/ARCHITECTURE.md)。

## 课程交付与分工

课程说明允许概念、设计与仿真，不要求实体原型。正式汇报 **10–15 分钟＋5 分钟问答**，**2026 年 11 月 3 日提交幻灯片作为报告**；软件演示只是汇报的一部分。报告应讲清问题、已有方案、拟议设计、仿真结果和局限。

保留原五人分工：数据与检测、情境与集成、提示控制、评估、体验与表达。具体交付与衔接见[分工与路线图](docs/ROLES_AND_ROADMAP.md)。机器人 1 已补充公开真实数据导入与验证代码；其他角色按交接结果继续完成控制、集成、评估和展示。

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
streamlit run src/fogguard/ui/app.py
```

使用 `python scripts/generate_demo_data.py` 生成小型演示数据；使用 `python scripts/generate_simulated_responses.py` 生成明确标注的 P1 模拟响应；使用 `pytest` 和 `ruff check .` 进行检查。

FoG、直行和左右转向的联合识别改用 FoG-STAR 多任务基线：

```bash
python scripts/download_fogstar.py
python scripts/train_fogstar_baseline.py
python scripts/predict_fogstar_baseline.py path/to/new_sensor_data.csv
```

数据选择、标签、跨受试者验证及新数据格式见 [FoG-STAR 多任务识别基线](docs/FOGSTAR_MODEL.md)。

这部分是机器人 1 的“数据与模型”交付：模型提供 FoG 与方向候选结果；机器人 2 负责把结果与现有情境识别器对照或仲裁，加入时间逻辑和安全门控后再接入 `ControllerContext`、控制器和 UI。详细字段契约见 [机器人 1 数据与 FoG 交接说明](docs/ROBOT1_HANDOFF.md)。

机器人 2 已完成 FoG-STAR 时序接入。当前项目重点是 FoG 步态提示和恢复引导，暂缓主观停步意图区分；静止状态继续记录，默认不据此抑制 FoG 提示。以下命令使用按受试者隔离的五折预测验收持续确认、迟滞、恢复观察、方向拒识和提示门控：

```bash
python scripts/train_fogstar_baseline.py
python scripts/evaluate_fogstar_integration.py
```

逐窗结果写入 `reports/fogstar_robot2_end_to_end.csv`，事件指标写入 `reports/fogstar_robot2_integration.json`。当前真实数据结果仍有明显误报和方向拒识，因此仅作为课程工程基线，不表示临床可用。规则、字段、数值和限制见[机器人 2 情境与集成交接](docs/ROBOT2_HANDOFF.md)。

机器人 2 的 P0 端到端回放可用一条命令运行：

```bash
python scripts/run_end_to_end.py
```

命令依次处理正常、直行 FoG、左右转 FoG、正常右转、停步和恢复场景，将逐窗口结果写入 `reports/robot2_end_to_end.csv`。输出包含原始 FoG、有效 FoG、评估真值、运动状态、方向/置信度/时间范围和提示原因。默认保留静止期间的 FoG 候选；如需复现历史静止抑制实验，可加 `--suppress-stationary-fog`。无陀螺仪或阈值边界不可靠时方向仍输出 `unknown`。实现、字段和交接说明见[机器人 2 情境与集成交接](docs/ROBOT2_HANDOFF.md)。

生成数据后，可以重复运行 FoG 基线评估：

```bash
python scripts/evaluate_fog_baseline.py
python scripts/calibrate_fog_threshold.py
python scripts/evaluate_fog_events.py
```

下载并验证公开 Daphnet 数据：

```bash
python scripts/download_daphnet.py
python scripts/evaluate_daphnet.py
```

脚本从 UCI 官方地址下载并校验文件哈希，原始数据和逐折 JSON 报告均保持在 Git 忽略目录。当前踝部数据 LOSO 结果为：35,545 个窗口，precision 0.185、recall 0.634、specificity 0.700、F1 0.287；237 个事件检出 219 个，但非 FoG 阳性重叠窗口达到 30.95 个/分钟。结论是当前基线事件覆盖较高但误报过多，不能直接驱动真实提示；完整口径和参与者级结果见 [Daphnet 验证记录](docs/DAPHNET_VALIDATION.md)。

合成评估命令会按场景输出窗口数量、真实/预测 FoG 窗口数，以及 accuracy、precision、recall 和 F1。标定命令会用预先声明的规则选择阈值，并将本地报告写入 `reports/fog_threshold_calibration.json`。合成场景得到的 `0.21` 只用于检查工程管线，不代表临床性能，也不会因 Daphnet 各折阈值中位数为 `0.73` 就直接替换。标定记录中的 F1＝1.00 只表示同一生成器数据上的窗口分类没有误报或漏报，详见[阈值标定记录](docs/FOG_THRESHOLD_CALIBRATION.md)。

机器人 1 的扩展场景、事件级证据、已知失败案例及交接方式见[数据与 FoG 交接说明](docs/ROBOT1_HANDOFF.md)。

最终可视化仿真视频的系统分层、论文约束恢复策略、分镜和机器人 1 交接后的四人计划见[仿真视频与剩余工作计划](docs/SIMULATION_VIDEO_PLAN.md)。

## 仓库结构与负责人

| 路径 | 主要角色 | 用途 |
|---|---|---|
| `src/fogguard/data/` | 机器人 1 | CSV 加载、校验和定长窗口回放 |
| `src/fogguard/detection/` | 机器人 1 | FoG 评分与阈值基线 |
| `src/fogguard/turning/` | 机器人 2 | 当前转向状态基线；接口沿用 TurnIntent 命名 |
| `src/fogguard/cueing/` | 自动化 1 | 提示规则基线；目标行为见需求文档 |
| `src/fogguard/ui/` | 工业设计 | Streamlit 演示与交互流程 |
| `src/fogguard/evaluation/` | 自动化 2 | 指标、实验与可复现性 |
| `src/fogguard/types.py` | 全组 | 共享且不可变的数据契约 |
| `src/fogguard/interfaces.py` | 全组 | 模块协议；只能经团队讨论后修改 |
| `docs/` | 全组 | 架构、数据格式、分工和协作流程 |

## 模块接口约定

所有模块通过 `fogguard.types` 中的数据类交换信息：

- `SensorWindow`：单个 IMU 窗口；加速度采用 m/s²，角速度采用 deg/s，时间采用秒。
- `DetectionResult`：FoG 评分、布尔判断及可选判据；现有字段名为 `probability`，尚未经概率校准。
- `TurnIntent`：方向、置信度及时间范围字段；当前实现依据窗口角速度，不具备已验证的提前预测能力。
- `ControllerContext`：当前预测、时间信息和不应期状态。
- `CueCommand`：提示类型、强度、持续时间和可读原因。

实现必须满足 `fogguard.interfaces` 中的协议。各功能分支可以替换基线模型，但不应迫使下游代码同步重写。任何接口修改都必须先建立 Issue、征得相关负责人同意，并更新 `docs/ARCHITECTURE.md`。

## 团队协作方式

1. 先认领一个 GitHub Issue，再创建简短分支，例如 `feat/fog-baseline`。
2. Pull Request 保持小而清晰，并用 `Closes #<编号>` 关联 Issue。
3. 合并到 `main` 前至少由一位组员审查。
4. 严禁提交可识别参与者身份的数据；本地原始数据放在已忽略的 `data/raw/`。
5. 每天结束前更新 Issue 清单并记录阻塞事项。

详细内容见[分工与两周路线图](docs/ROLES_AND_ROADMAP.md)、[协作规则](docs/COLLABORATION.md)和[系统架构](docs/ARCHITECTURE.md)。

## 安全与项目边界

本仓库仅用于课程项目和工程评估，不用于诊断、预防或治疗疾病。需求要求输入缺失、过期或相关判断不可靠时返回 `CueType.NONE`；当前实现尚需按架构文档补齐并验收。任何涉及人体参与者的数据采集都必须在仓库之外先取得教师同意及适用的伦理审批。

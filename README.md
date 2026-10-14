# ME5412 FoGGuard

FoGGuard 是一个为期三周的 ME5412 小组项目：回放可穿戴设备的步态数据，检测冻结步态（Freezing of Gait，FoG），识别转向意图，并选择自适应提示方式。最终成果是一个可复现、可量化评估的离线 Streamlit 演示系统，不是医疗器械，也不用于实时临床决策。

## 三周最小可行成果（MVP）

演示系统采用一条稳定的数据流程：

```text
CSV / synthetic replay
        ↓ SensorWindow
FoG detector ────────┐
                     ├─ ControllerContext → adaptive cue → Streamlit timeline
turn-intent model ───┘
        ↓
离线指标与场景报告
```

完成标准：新成员能用一组命令启动项目；系统可以回放至少三种预设场景；每位成员能解释模块边界；最终报告包含 FoG 精确率、召回率、F1、转向意图准确率和提示延迟。

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
streamlit run src/fogguard/ui/app.py
```

使用 `python scripts/generate_demo_data.py` 生成小型演示数据；使用 `pytest` 和 `ruff check .` 进行检查。

生成数据后，可以重复运行 FoG 基线评估：

```bash
python scripts/evaluate_fog_baseline.py
python scripts/calibrate_fog_threshold.py
```

评估命令会按场景输出窗口数量、真实/预测 FoG 窗口数，以及 accuracy、precision、recall 和 F1。标定命令会用预先声明的规则选择阈值，并将本地报告写入 `reports/fog_threshold_calibration.json`。当前合成场景标定得到 `0.21`；它只用于检查工程管线，不代表临床性能。

## 仓库结构与负责人

| 路径 | 主要角色 | 用途 |
|---|---|---|
| `src/fogguard/data/` | 机器人 1 | CSV 加载、校验和定长窗口回放 |
| `src/fogguard/detection/` | 机器人 1 | FoG 评分与阈值基线 |
| `src/fogguard/turning/` | 机器人 2 | 转向意图识别 |
| `src/fogguard/cueing/` | 自动化 1 | 自适应提示状态机 |
| `src/fogguard/ui/` | 工业设计 | Streamlit 演示与交互流程 |
| `src/fogguard/evaluation/` | 自动化 2 | 指标、实验与可复现性 |
| `src/fogguard/types.py` | 全组 | 共享且不可变的数据契约 |
| `src/fogguard/interfaces.py` | 全组 | 模块协议；只能经团队讨论后修改 |
| `docs/` | 全组 | 架构、数据格式、分工和协作流程 |

## 模块接口约定

所有模块通过 `fogguard.types` 中的数据类交换信息：

- `SensorWindow`：按时间排列、采用 SI 单位的单个 IMU 窗口。
- `DetectionResult`：FoG 概率、布尔判断及可选判据。
- `TurnIntent`：左转、右转、直行类别及置信度。
- `ControllerContext`：当前预测、时间信息和不应期状态。
- `CueCommand`：提示类型、强度、持续时间和可读原因。

实现必须满足 `fogguard.interfaces` 中的协议。各功能分支可以替换基线模型，但不应迫使下游代码同步重写。任何接口修改都必须先建立 Issue、征得相关负责人同意，并更新 `docs/ARCHITECTURE.md`。

## 团队协作方式

1. 先认领一个 GitHub Issue，再创建简短分支，例如 `feat/fog-baseline`。
2. Pull Request 保持小而清晰，并用 `Closes #<编号>` 关联 Issue。
3. 合并到 `main` 前至少由一位组员审查。
4. 严禁提交可识别参与者身份的数据；本地原始数据放在已忽略的 `data/raw/`。
5. 每天结束前更新 Issue 清单并记录阻塞事项。

详细内容见[分工与三周路线图](docs/ROLES_AND_ROADMAP.md)、[协作规则](docs/COLLABORATION.md)和[系统架构](docs/ARCHITECTURE.md)。

## 安全与项目边界

本仓库仅用于课程项目和工程评估，不用于诊断、预防或治疗疾病。输入缺失、过期或置信度不足时，提示控制器必须默认返回 `CueType.NONE`。任何涉及人体参与者的数据采集都必须在仓库之外先取得教师同意及适用的伦理审批。

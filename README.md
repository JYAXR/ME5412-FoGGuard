# FoGGuard

FoGGuard 是 ME5412 课程项目，研究如何利用可穿戴 IMU 识别冻结步态与转向，并根据提示后的运动反馈，控制脚印和声音提示。

系统目标是：检测到持续冻结后，给出与方向一致的迈步提示；在有效观察中持续没有迈步响应时，切换原地交替踱步；获得可靠响应后重新引导向前，满足退出条件后停止提示。

## 项目入口

| 内容 | 文档 |
|---|---|
| 项目目标、系统设定与交付范围 | [项目说明](docs/PROJECT_OVERVIEW.md) |
| A1 / A2 / B1 / B2 与工业设计的职责 | [分工与交付顺序](docs/ROLES_AND_ROADMAP.md) |
| 模块关系、状态和验收要求 | [系统架构](docs/ARCHITECTURE.md) |
| 当前完成状态 | [项目状态](docs/PROJECT_STATUS.json) |
| A 组统一图表与数据展示（英文） | **[打开统一页面](docs/a_group_showcase/index.html)** · [图表及数据索引](docs/a_group_showcase/FIGURE_CATALOG.csv) |
| A 组测试来源与算法复现 | [固定测试报告](docs/a_group/README.md) · [感知接入](docs/a2/fusion/README.md) |
| 环境、模型推理与工程演示 | [快速开始](docs/QUICKSTART.md) |
| 字段、单位和时序约定 | [接口约定](docs/INTERFACE_CONTRACT.md) · [数据格式](docs/DATA_SCHEMA.md) |

## 系统与分工

```mermaid
flowchart LR
  IMU[同步IMU数据] --> A1[A1 冻结识别]
  IMU --> A2[A2 转向识别与感知融合]
  A1 --> A2
  IMU --> B1[B1 迈步反馈与提示控制]
  A2 --> B1
  B1 --> OUT[保存事件与提示时间线]
  A2 --> B2[B2 评估与联调]
  B1 --> B2
  OUT --> DESIGN[工业设计 PPT与最终可视化]
```

A 组负责感知，B 组负责反馈、决策和验证。最终可视化读取保存的模块输出，模型和控制规则由算法层执行。

## A1：已完成冻结识别交付

**[Unified English charts and data](docs/a_group_showcase/index.html)**：流程与方法、数据范围、冻结、方向、融合影响和联合时间线集中展示；每图有简短说明及 PNG/SVG/CSV。正式测试为主线，开发验证集中在页面附录，用于解释方案选择。

提供三种可运行的冻结 / 非冻结模型、统一推理入口、训练配置、效果对比和交接包。下表使用冻结权重和阈值后的独立 TEST：4 人、16 条记录、1605 个共同窗口，其中 253 个冻结窗口。

| 方法 | 输入与表征 | Precision | Recall | FoG F1 |
|---|---|---:|---:|---:|
| 左踝历史特征 RF | 左踝加速度和角速度，140 维时序特征 | 0.346 | 0.806 | 0.485 |
| 双踝 RF | 左右踝角速度，66 维特征 | 0.413 | 0.838 | 0.554 |
| 左踝 CNN | 左踝三轴角速度，两秒波形 | 0.535 | 0.581 | 0.557 |

模型由运行参数显式选择，融合参考仍为测试前确定的 `bilateral_rf`。冻结误报与漏检仍较明显。当前方向模型在 676 个有效且目标明确的测试窗口上准确率为 81.07%、macro-F1 为 0.844；固定门控和平滑后的准确率为 51.78%、覆盖率为 54.73%，需单独报告。[固定测试报告](docs/a_group/README.md) 给出基线、完整分母和限制。模型实现见 [A1 方法](docs/a1/SELECTED_METHODS.md)。

## 后续交付

| 成员 | 当前基础 | 要完成的交付 |
|---|---|---|
| A1 | 三种冻结模型和评估已交付 | 维护模型包与接口，支持 A2 接入 |
| A2 | [方向与融合已完成固定独立测试](docs/a_group/README.md)；提前预测未达到可用水平 | [覆盖率及误报适用性复核、交付审查](docs/tasks/A2.md) |
| B1 | 有脚印声音控制、模拟响应与踱步状态机 | [真实信号运动代理、响应判据、提示切换与控制验证](docs/tasks/B1.md) |
| B2 | 有指标审计、工程场景和策略对照工具 | [四模块联调、统一验收、对比报告和展示数据包](docs/tasks/B2.md) |
| 工业设计 | 有保存时间线的回放页面 | [PPT、最终可视化与演示视频](docs/tasks/DESIGN.md) |

## 最小运行

Python 3.12，在仓库根目录：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -X utf8 scripts/run_complete_demo.py --output-dir reports/project_demo_NEW
```

打开生成的 `index.html` 查看脚印、声音状态和控制过程。该命令使用构造候选与模拟响应，验证工程行为。真实 A1 模型的运行方法见 [A1 接入](docs/A1_A2_START_HERE.md)。

## 证据与范围

FoG-STAR 用于真实冻结识别与转向任务；运动响应代理、预设模拟响应和真实位置反馈分别记录来源。公开数据没有独立逐步前进位移真值，运动代理只能支持相应的动作判断。工程演示展示切换过程，不能作为患者恢复效果。

原始数据、模型权重及生成目录不提交 Git。共享类型的变更按 [协作规则](docs/COLLABORATION.md) 评审；各模块的新适配器完成接线前，现有演示不代表四个正式算法已经全部集成。

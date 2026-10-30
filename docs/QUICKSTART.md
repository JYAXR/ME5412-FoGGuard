# 快速开始

## 获取代码和环境

A 组感知、测试和统一图表已合并到 `main`。新同学从 `main` 获取当前基线：

```bash
git clone --branch main --single-branch https://github.com/JYAXR/ME5412-FoGGuard.git FoGGuard
cd FoGGuard
python -m venv .venv
```

激活虚拟环境后，使用 Python 3.12：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

CNN 另外安装 `requirements-a1-deep.txt`，数据可行性静态图工具另外使用 `requirements-readiness.txt`。模型包内的 `TESTED_VERSIONS.json` 记录权重运行的依赖版本。

## 先看已交付图表

直接打开 [统一英文展示](a_group_showcase/index.html)，不需要模型权重或原始数据。7 张 PNG/SVG、9 张 CSV 和简短说明可供工业设计制作 PPT。正式成绩来源为 [固定独立测试](a_group/README.md)，开发验证只在明确标注的附录中展示。

## A1 模型推理

将三模型包解压，使模型位于 `artifacts/fog-selected/<method>/model.joblib`：

```bash
python -X utf8 scripts/predict_selected_fog.py --bundle artifacts/fog-selected --method bilateral_rf --input artifacts/fog-selected/synthetic_input.csv --output reports/fog_interface_NEW.csv
```

`--method` 可选 `ankle_history_rf / bilateral_rf / ankle_cnn`。合成样例用于验证接口；正式效果见 [A 组测试](a_group/README.md)。A1 输出只有冻结结果；当前方向和融合已由 A2 交付，真实反馈由 B1 后续提供。

## 下载并使用预训练模型

**[模型下载与运行](MODEL_DOWNLOADS.md)** 提供 GitHub Release 中的四个正式权重：A1 三个冻结模型和 A2 当前方向模型。下载 ZIP、校验并解压即可运行，无需训练；`git clone` 本身不下载这些附件。包内合成样例不需要原始数据，真实回放另需 FoG-STAR 数据集。

## 工程演示

真实 A1+A2 感知的训练重建、联合回放和英文图表入口见 [感知交接](a2/fusion/README.md)。它需要单独提供 A1 模型包和原始数据；下面的工程演示使用构造输入。

```bash
python -X utf8 scripts/run_complete_demo.py --output-dir reports/project_demo_NEW
```

打开 `reports/project_demo_NEW/index.html`。页面读取保存的命令，提供脚印、声音状态和策略对照。输入为构造候选与模拟响应，不需要下载患者数据或训练模型。生成目录使用新的名称。

事件层的最小接口样例：

```bash
python -X utf8 scripts/run_fogstar_end_to_end.py --predictions docs/examples/robot2_predictions.csv --output reports/perception_interface_NEW.csv
```

该样例演示事件确认、方向拒识和输入时效；不是 A1 三模型的系统联调成绩。

## 开发检查

完整检查包含 A2 评价测试，先安装其绘图依赖：

```bash
python -m pip install -r requirements-a2.txt
```

```bash
python -X utf8 scripts/run_automation2_delivery_checks.py
```

该命令包括 `tests` 与 `recognition`、Ruff、工程演示和策略比较。没有提供原始数据、OOF 和训练报告时，真实数据审计标为 `NOT_RUN`。

## 模型维护与可选训练入口

```bash
python scripts/download_fogstar.py
python -X utf8 scripts/train_selected_fog.py --method all --output reports/selected_train_NEW
python -X utf8 scripts/evaluate_selected_fog.py --bundle reports/selected_train_NEW --output reports/selected_evaluation_NEW
```

这些训练/开发验证入口按 `config/fog_selected.json` 使用 TRAIN/开发验证，不评分 TEST。现有固定 TEST 已通过专门入口完成，参见 [锁定评价说明](a_group/README.md)。下游开发直接使用已交付权重和感知表；本轮任务不要求重训 A 组或重新选择测试阈值。

任务入口：[A1 维护](tasks/A1.md) · [A2 维护](tasks/A2.md) · [B1 反馈/控制](tasks/B1.md) · [B2 接入/验证](tasks/B2.md) · [工业设计](tasks/DESIGN.md)。依赖和执行顺序见 [规划](ROLES_AND_ROADMAP.md)。

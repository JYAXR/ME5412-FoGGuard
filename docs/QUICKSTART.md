# 快速开始

## 获取代码和环境

当前 A1+A2 感知交接贡献分支为 `feat/a2-perception-fusion`，请克隆到新目录；合并以接收方审查为准：

```bash
git clone --branch feat/a2-perception-fusion --single-branch https://github.com/JYAXR/ME5412-FoGGuard.git FoGGuard
cd FoGGuard
python -m venv .venv
```

激活虚拟环境后，使用 Python 3.12：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

CNN 另外安装 `requirements-a1-deep.txt`，数据可行性静态图工具另外使用 `requirements-readiness.txt`。模型包内的 `TESTED_VERSIONS.json` 记录权重运行的依赖版本。

## A1 模型推理

将三模型包解压，使模型位于 `artifacts/fog-selected/<method>/model.joblib`：

```bash
python -X utf8 scripts/predict_selected_fog.py --bundle artifacts/fog-selected --method bilateral_rf --input artifacts/fog-selected/synthetic_input.csv --output reports/fog_interface_NEW.csv
```

`--method` 可选 `ankle_history_rf / bilateral_rf / ankle_cnn`。合成样例用于验证接口；真实效果见 [A1 结果](a1/SELECTED_METHODS.md)。输出只有冻结结果，方向与反馈由对应模块提供。

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

## 真实数据与训练

```bash
python scripts/download_fogstar.py
python -X utf8 scripts/train_selected_fog.py --method all --output reports/selected_train_NEW
python -X utf8 scripts/evaluate_selected_fog.py --bundle reports/selected_train_NEW --output reports/selected_evaluation_NEW
```

训练与评价按 `config/fog_selected.json` 使用 TRAIN/开发验证，当前命令不评分 TEST。A2 可直接使用 A1 交接包开展工作，无需重新训练冻结模型。

任务入口：[A2](tasks/A2.md) · [B1](tasks/B1.md) · [B2](tasks/B2.md) · [工业设计](tasks/DESIGN.md)。

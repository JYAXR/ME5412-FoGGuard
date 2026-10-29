# 真实数据识别核验

## A1 三模型评价

冻结方法和同窗成绩见 [A1 方法说明](../a1/SELECTED_METHODS.md)，运行入口为 `scripts/evaluate_selected_fog.py`。当前成绩范围为开发验证 4 人、20 记录、2014 窗口，不能写为独立最终测试。

## 通用 OOF 审计工具

`recognition/evaluate.py` 接收原始数据、按参与者留出的 OOF 预测及匹配训练报告，核查记录键、标签、窗口范围和来源，并生成逐人/逐记录、原始事件和时间覆盖结果。

```bash
python -X utf8 recognition/evaluate.py --raw data/raw/fogstar/sensor_data.csv --predictions path/to/fogstar_oof_predictions.csv --training-report path/to/fogstar_training.json --output reports/recognition_NEW
```

该工具的输入契约属于通用组合预测，不能直接将 A1 的 FoG-only 文件作为其完整输入。B2 根据正式模块的评价格式编写组合适配，独立保存真值。

## 统一评价语义

原始记录应包括没有预测的记录。无预测时段保留未知；事件触及、冻结时间覆盖和提示触发分别报告。真值分组只解释二分类表现，不代表模型识别冻结亚型。

预测从实际可用时刻开始计入有效时间。模型对训练数据的最终预测不能作为留出成绩。没有相应真值时报告不可评价。联合验收的任务和展示材料见 [B2 任务书](../tasks/B2.md)。

# A1 困难负样本实验交接

本轮候选未通过，原默认模型保持不变。普通 RF 与加权 RF 的完整成绩见 [结果](HARD_NEGATIVE_RESULTS_V1.md)。本轮新产物用于对照及接口验证，不作为推荐升级模型。

## A2 读取什么

本地目录 `reports/a1_hard_negative_v1_20261028/` 中的 `baseline/` 与 `hard_negative/` 各含：`a2_runtime.csv`（无 truth_*）、`evaluation_truth.csv`（隔离保存的真值）、`oof_predictions.csv`（评估）、`model.joblib`（全量拟合，不能用训练回放当留出性能）、分折及训练说明。不要把两组分数平均或选择逐段表现较好的模型。

runtime 沿用旧方向 OOF，`direction_source=unchanged_historical_baseline_oof`。A2 应将自己的方向预测按参与者、记录和窗口时间对齐；旧方向列不表示已完成转向意图识别。独立模型推理只需传感器，不需活动真值或样本权重。

```powershell
python -X utf8 scripts/predict_a1_multimodel.py --input path/to/new_sensor_data.csv --model reports/a1_hard_negative_v1_20261028/baseline/model.joblib --output reports/a1_fog_scores.csv
python -X utf8 scripts/run_fogstar_end_to_end.py --predictions reports/a1_hard_negative_v1_20261028/baseline/a2_runtime.csv --output reports/a1_rf_events.csv
```

此处 baseline 是本轮 F1 选阈值的对照 RF，不是自动替换历史默认模型。历史默认交接及其他候选接口见 [原 A2 交接](A2_HANDOFF.md) 和 [多模型交接](MULTIMODEL_HANDOFF.md)。新输入须与原左踝/腰背 12 通道、60 Hz、单位和轴向一致。评分输出保留 `fog_threshold` 与 `available_at_s`；不要将窗口起点当作可用时刻。

原事件层的退出迟滞差为 0.10，不同评分分布下不保证等效。本轮成绩均为各自训练内阈值后的原始报警；最终确认、迟滞、退出和方向融合还需由 A2 在训练参与者内部确定，再单独评价集成链路。

低 FoG 分数不能单独证明已经迈步或恢复。真实迈步完成、原地踏步响应和提示效果仍缺乏对应真值，本轮不提供这些判断的有效性证据。

## 已验证

候选推理 30 行、候选 runtime 集成 9,636 行通过接口检查；三层人员隔离、权重、65 个哈希核验通过；完整测试 269 项通过。图片、模型、OOF 等本地产物未推送到 Git，传递时应保留 manifest 及数据来源说明。源码为本地贡献，未自动推送或合并。

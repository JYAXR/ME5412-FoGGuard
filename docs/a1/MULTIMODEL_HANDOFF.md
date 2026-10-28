# A1 多模型结果交给 A2

本轮为 RF、SVM、HistGB 与小型 CNN 的同输入对照。方法见 MULTIMODEL_PROTOCOL_V1.md，结论见 MULTIMODEL_RESULTS_V1.md。原默认模型保持不变；不要仅凭模型名称或是否深度学习选择部署方案。

## 本轮本地产物

输出目录：reports/a1_multimodel_v1_20261028/。rf/、svm/、hist_gb/、cnn/ 各目录含：

- a2_runtime.csv：外层受试者留出结果，不含 truth_*。
- evaluation_truth.csv：单独的评估标签。
- oof_predictions.csv：完整留出结果，供原始数据独立核验。
- model.joblib：全量拟合的 FoG-only 模型，用于新输入，不用其训练回放当留出成绩。
- training_report.json、fold_audit.json：五个外层测试人员、三折内层人员、91 个阈值成绩和选择结果。
- recognition/：原始记录覆盖、逐窗验证、逐人/活动/任务结果、真实错误案例交互页。
- CNN 另有 network.json，记录结构、参数数及全量训练损失；全量损失不是测试性能。

根目录有 comparison.json/csv、ppt_comparison.csv、index.html、figures/、environment.txt、protocol.json 与 manifest.json。图像同时提供 PNG/SVG，可由工业设计同学编排，无需重新计算算法。

## 接现有事件链路

每个 a2_runtime.csv 保留了未改动的旧方向 OOF，direction_source 标注 unchanged_historical_baseline_oof。可用原 run_fogstar_end_to_end.py 接入，验证 FoG 分数/阈值和事件确认链路；这不等于本轮训练了方向意图模型。

```powershell
python -X utf8 scripts/run_fogstar_end_to_end.py --predictions reports/a1_multimodel_v1_20261028/hist_gb/a2_runtime.csv --output reports/a1_hist_gb_events.csv
```

上例的 hist_gb 是接口示例，不是预先指定胜出模型。其他方案替换目录名即可。

## 对新输入预测

```powershell
python -X utf8 scripts/predict_a1_multimodel.py --input path/to/new_sensor_data.csv --model reports/a1_multimodel_v1_20261028/hist_gb/model.joblib --output reports/a1_new_model_scores.csv
```

输入与训练一致：左踝/腰背 12 通道，60 Hz，加速度 g、角速度 deg/s、轴向一致。模型内部完成对应特征或波形处理。SVM 标准化和 CNN 通道统计均从模型读取，不在新数据上重新拟合。

独立推理只输出记录键、窗口起止/可用时刻、fog_probability、fog_threshold、is_fog、模型版本与来源。A2 自己的方向结果按记录键/窗口时间对齐后再进入事件控制层。CSV 中的 fog_probability 是沿用字段名的评分；SVM 是 sigmoid(margin)，不是校准概率，不能跨模型直接比较分数大小。

原事件层的退出迟滞为分数差 0.10；跨模型的分数尺度/校准不同，数值相同不保证含义相同。本轮算法性能使用各自训练内阈值后的原始报警口径。接入示例只验证兼容性；A2 如重新设定迟滞和确认规则，应在训练参与者内部确定，再单独评价最终提示链路。

CNN 模型需要额外安装 CPU 版 PyTorch，具体版本见本轮 environment.txt。RF/SVM/HistGB 不需要导入 CNN 模块，原运行环境不被强制增加 torch 依赖。

## 后续职责与限制

A1 交付冻结检测，不判断真实步伐是否迈出；低 FoG 分数不是恢复行走的独立证据。A2 继续方向与转向前任务，不把活动标签直接当运行时方向。

新比较统一以训练内 FoG F1 选阈值，与前一轮 balanced accuracy 优先不同。历史默认模型、上一轮近期特征候选与本轮算法之间不可忽略阈值规则直接下结论。本轮四个模型之间才是同规则比较。

数据中无预测支持的记录不删除。事件有重叠覆盖不等于一对一事件 F1；单种子 CNN 未胜出也不能否定其他深度模型。

源码与说明在本地贡献分支，未自动推送或合并。原始数据、临床表、训练模型和整批输出不加入源码 Git；如需传递模型/OOF/图片，使用经核查的版本化文件包并保留哈希。

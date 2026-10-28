# A1 固定划分重训模型交接

方法见 [固定协议](FIXED_SPLIT_PROTOCOL_V1.md)，测试及失败边界见 [结果](FIXED_SPLIT_RESULTS_V1.md)。本轮八个模型全部重新训练，拟合者固定为 TRAIN 14 人、阈值仅来自 VALIDATION 4 人；TEST 7/11/21/22 没有进入本轮拟合、归一化或选择。旧队列已被研究，仍非全新外部验证。

## 文件

`reports/a1_fixed_split_v1_20261029/` 下的 rf_2s、svm_2s、histgb_2s、cnn_2s、extra_2s、rf_context、extra_posture、rf_ankle 各有 model.joblib、fit_metadata.json、validation_predictions.csv、validation_selection.json。不要与此前全 22 人部署文件混用。

TRAINING_LOCK.json 在测试前生成；validation_winner.json 指定 rf_context 为主模型。该选择不会因为 CNN 的测试 F1 略高就改变。所有模型继续保持只拟合 TRAIN，不在测试后再用全队列覆盖。

test_results/index.html 展示原始测试及压力对比。clean/模型名/ 下有 predictions.csv（含评估真值）、runtime.csv（不含 truth_*）、participant_metrics.csv、metrics.json、event_coverage.csv、event_alarms.csv。其他六个压力场景有相同结构；原始 TEST 记录和事件分母保持一致，缺测窗口数可以改变。

## 新输入推理

单模型仍使用原 FoG-only CLI，例如：

```powershell
python -X utf8 scripts/predict_a1_fog.py --input path/to/new_sensor_data.csv --model reports/a1_fixed_split_v1_20261029/rf_context/model.joblib --output reports/a1_fixed_context_scores.csv
```

验证选定主模型＋左踝备用已复制到 test_results/joint_models/，与锁定的训练文件哈希一致：

```powershell
python -X utf8 scripts/predict_a1_fallback.py --input path/to/new_sensor_data.csv --model-dir reports/a1_fixed_split_v1_20261029/test_results/joint_models --output reports/a1_fixed_joint_scores.csv
```

输入仍为 60Hz，左踝及腰背原轴向，加速度 g、角速度 deg/s。主模型缺新鲜输入时才启用左踝，双方缺测仍未知。推理只输出 FoG，不提供新方向、意图、真实落脚或恢复证据；A2 需对齐自己的方向预测。不要把原默认集成器额外确认后的成绩冒充原始检测成绩。

模型对象保留 fit_participants、threshold_participants；每个持续输入源单独创建 new_stream() 以保留历史。模型未在新输入上重新标准化。SVM 的 fog_probability 为 sigmoid(margin)，不是校准概率，不能跨模型直接按分数大小比较。

## 复现顺序

```powershell
python -X utf8 scripts/train_a1_fixed_split.py --run-dir reports/a1_fixed_split_reproduction
python -X utf8 scripts/evaluate_a1_fixed_split.py --run-dir reports/a1_fixed_split_reproduction
python -X utf8 scripts/verify_a1_fixed_split.py --run-dir reports/a1_fixed_split_reproduction
```

新训练目录必须为空。测试脚本要求已存在锁、模型/特征源码哈希一致，拒绝覆盖已有测试输出，并且不含 fit/fit_transform/partial_fit 调用。测试评分不能反过来决定重新分人或选择更好阈值。

本地 311 项测试和独立文件/原始窗口/区间核验通过，性能仍未达标。报告所列测试分母为 4 人、16 记录、23 个冻结事件，不能与以前 22 人、101 事件直接对比。原默认、共享类型和控制器未改；源码尚未推送/合并，模型、原始数据和全部运行输出不加入源码 Git。

# A1 失败信号审查与双踝输入交接

方法见 [固定协议](BILATERAL_PROTOCOL_V1.md)，实际输入支持和比较见 [结果](BILATERAL_RESULTS_V1.md)。本轮先审查TRAIN，再按预定可用性条件做相同窗口左33/双66特征RF对照；没有新增协同手工特征、深度网络、loss改动或后处理。无新候选TEST成绩，VALIDATION仍是已被开发的队列。

## 文件及模型

`reports/a1_bilateral_v1_20261029/input_audit/` 保存TRAIN左右可用窗口、全部70条记录/逐人输入表、时间/事件支持、旧TRAIN-OOF误报与同人同活动纯FoG配对候选、固定最近配对、真实双踝波形与分布图。配对是刻意最相近的现有左踝特征，不是随机代表样本，不证明生理不可区分或右踝必然有效；非FoG站立不等于主动停步。

先看`input_audit/eligibility.json`。若预定门槛失败，只有审查与`TRAINING_NOT_RUN.json`，不得把它称为模型通过。若已拟合，则`left_rf/model.joblib`与`bilateral_rf/model.joblib`均在**相同双踝有效TRAIN窗口**从零拟合；每折与最终实际拟合人员见metadata，不会把没有有效窗口的人写成已拟合。

阈值只来自TRAIN按人留出的OOF，模型锁定后才单独评价VALIDATION。`evaluation/index.html`和`development_gate.json`区分匹配左踝、双踝、同一冻结左踝模型的自然全左踝窗口，不能只挑额外传感器可用片段而隐藏未知时段。历史CNN/RF仅参考，训练选择或输入不同。

## 推理

无需PyTorch。Python 3.12及原numpy/pandas/scipy/scikit-learn依赖即可；仓库根目录：

```powershell
python -X utf8 scripts/predict_a1_bilateral.py --input path/to/sensors.csv --model reports/a1_bilateral_v1_20261029/bilateral_rf/model.joblib --output reports/a1_bilateral_new_input.csv
```

输入timestamp为秒，60Hz规则采样；gyro单位deg/s，字段为ankleL_gyro_x/y/z与ankleR_gyro_x/y/z。左模型只需左三轴。轴向、安装位置与训练一致，不需要accel、腰背、activity、fog。多记录提供subjectID/sessionID/taskID避免串接。

时间网格永远由左踝连续有效段的120点窗口、30点步进决定。双踝要求同一120行右三轴全部有限；缺失就无该窗口预测，不补零、不插值，也不移动到右踝恢复点重新切窗。新记录/左踝断档才重新起网格。无完整有效窗口则报错，不能理解为判正常。必须由上层保留无预测时段为未知。

Python：`BilateralFoG.load(path).predict_frame(frame)`，或向`predict_windows`提交当前完成的2秒`FogStarWindow`；本模型无跨窗口状态，已完成窗口可逐个推理。输入schema必须精确为左三轴或左三轴后右三轴；输出FoG-only，available_at_s为窗口结束，不回填窗口起点，不提供方向、提前意图、真实落脚或迈步成功。

`left_rf`自然推理可以在右踝缺失时继续，但该模型训练使用匹配双踝窗口；本轮没有把它与双踝拼成自动备用路由。不得自行将未知填正常、以真值活动筛预测，或把不同支持下的分数直接排名后替换默认。

## 复现

```powershell
python -X utf8 scripts/train_a1_bilateral.py --run-dir reports/a1_bilateral_reproduction
python -X utf8 scripts/evaluate_a1_bilateral.py --run-dir reports/a1_bilateral_reproduction
python -X utf8 scripts/verify_a1_bilateral.py --run-dir reports/a1_bilateral_reproduction
python -X utf8 scripts/report_a1_bilateral.py --run-dir reports/a1_bilateral_reproduction --output-dir reports/a1_bilateral_presentation_reproduction
python -X utf8 scripts/run_automation2_delivery_checks.py
```

需要同版公开数据与前轮context_learning锁定目录，继承人员折、旧OOF和历史参考。使用空目录；输入门槛失败时不运行后续评价、核验与展示导出。评价不拟合或改阈值，锁/源码/输入哈希须一致。独立核验重建左锚定网格、右踝有效性、原始真值、前33特征、配对规则/距离、外层留出评分及实际拟合人员、TRAIN阈值和锁定时间；不声称独立重训。

展示用 `reports/a1_bilateral_presentation_v1_20261029/index.html`，八对PNG/SVG可放入PPT。独立report脚本修复三张波形图原导出的长标题/边距裁切，其余五图原样复制；锁定实验目录和模型结果保持原样，展示来源哈希单独保存。不是最终PPT或3D交付。

默认模型、A2方向、共享类型及控制器保持原样；本地源码尚未推送或合并。原始数据、模型与生成目录不入Git，真实信号审查不能充当临床疗效或迈步响应验证。

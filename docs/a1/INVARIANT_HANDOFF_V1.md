# A1 旋转不变表征实验交接 v1

见 [固定协议](INVARIANT_PROTOCOL_V1.md)、[真实结果与取舍](INVARIANT_RESULTS_V1.md)、[invariant_status.json](invariant_status.json)。10维整体替换未通过开发门槛，不替换原默认检测器。保留模型供复核，不能以可加载代替验收。

## 输入及输出

模块：`fogguard.modeling.a1_invariant.InvariantFoG`，两变体`axis_control`/`invariant_rf`。候选只读timestamp秒及ankleL_gyro_x/y/z、ankleR_gyro_x/y/z，规则60Hz、deg/s、原左右顺序。无需加速度、活动、fog、fog_severity。旋转不变只涵盖每踝固定正交轴旋转；不涵盖单位/采样率变化、左右互换、幅度或不同安装位置。

多记录应带subjectID/sessionID/taskID。仅沿左踝连续段生成120点/30点步进窗口；任意右踝缺失则跳过该窗，不补零、不插值、不因右踝恢复重开网格。无有效窗口报错，上层保留未知。窗口内滤波使用已完成窗口，不看其末端以后的数据。

```powershell
python -X utf8 scripts/predict_a1_invariant.py --model reports/a1_invariant_v1_20261029/invariant_rf/model.joblib --input path/to/sensors.csv --output reports/a1_invariant_new_input.csv
```

Python调用：`InvariantFoG.load(path).predict_frame(frame)`，或向`predict_windows(windows)`提交已完成的`FogStarWindow`序列。返回FoG概率、TRAIN阈值（候选0.37，对照0.23）、二元判断、记录键、窗口起止和available_at_s；本地FoG-STAR可用时刻为窗口末端。外部128Hz适配的1/6秒延迟由适配层单独加，不包含在模型中。最多0.5秒保持由评估/上层处理，缺测不自动变成正常。

没有方向、主观意图、步长、足部位置、迈步成功、提示响应或治疗效果输出。A2继续消费既有契约；本模块是可选实验，不改共享types/interfaces。

## 保存内容与复现

`reports/a1_invariant_v1_20261029/`含两种表征各三个TRAIN折RF、最终model.joblib、训练特征矩阵、实际人员/配置/阈值网格/OOF表、源及输入哈希和MODEL_LOCK。`evaluation/validation/`包含20条原记录审计与同窗评分；`evaluation/external/`包含全60记录/7人及211个原事件。外部对照重用既有锁定66维分数，候选只作一次研发复查。

使用项目Python 3.12环境及现有numpy/pandas/scipy/scikit-learn，不需PyTorch。先准备同版FoG-STAR raw、固定双踝TRAIN/VALIDATION参照、Stanford锁定输入/旧评分，以及上一轮独立验证的不变摘要；入口分别见[BILATERAL_HANDOFF_V1.md](BILATERAL_HANDOFF_V1.md)、[STANFORD_EXTERNAL_HANDOFF_V1.md](STANFORD_EXTERNAL_HANDOFF_V1.md)、[EXTERNAL_FAILURE_DIAGNOSTIC_RESULTS_V1.md](EXTERNAL_FAILURE_DIAGNOSTIC_RESULTS_V1.md)。脚本默认这些历史版本路径，复现到其他位置需调整配置中的reference_run/external_reference及核验/报告的--diagnostic-reference。

输出目录必须是新的，训练锁定后不得改模型代码再评分：

```powershell
python -X utf8 scripts/train_a1_invariant.py --run-dir reports/a1_invariant_NEW
python -X utf8 scripts/evaluate_a1_invariant.py --run-dir reports/a1_invariant_NEW
python -X utf8 scripts/verify_a1_invariant.py --run-dir reports/a1_invariant_NEW
python -X utf8 scripts/report_a1_invariant.py --run-dir reports/a1_invariant_NEW --output-dir reports/a1_invariant_presentation_NEW
python -X utf8 scripts/run_automation2_delivery_checks.py
```

核验会复用旧摘要锁与独立核验记录，并逐点重算新VALIDATION公式；不会将旧摘要称作本轮独立重训。全量项目检查若需旧真实数据复核，还应传--raw、--oof、--training-report，否则该项NOT_RUN。生成目录和模型不入Git，保留日期状态及聚合证据。

本轮默认未替换，没有新候选TEST评价。VALIDATION和Stanford均已被研发使用，后续改善不能叫新独立测试。结果只说明当前10维压缩不合适；下一轮增量摘要实验需重新固定协议，而非改写当前结果。

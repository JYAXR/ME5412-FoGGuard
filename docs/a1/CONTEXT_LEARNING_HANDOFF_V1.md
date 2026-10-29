# A1 参与者均衡与活动辅助交接

固定假设见 [协议](CONTEXT_LEARNING_PROTOCOL_V1.md)，实际取舍见 [结果](CONTEXT_LEARNING_RESULTS_V1.md)。本轮三组同参数、同输入、同TRAIN嵌套人员折，分别只改训练损失的一项；不是把两种改动混在一起。VALIDATION是已被研究的开发队列，没有新候选TEST成绩。

## 模型与来源

`reports/a1_context_learning_v1_20261029/` 下的 `control`、`person_balanced`、`activity_aux` 各有 `model.joblib`：最终在全部TRAIN14从零训练；epoch为三个TRAIN内层checkpoint最佳轮数的中位数，阈值来自TRAIN按人留出的OOF。模型不包含VALIDATION拟合或阈值选择；各目录 `fit_metadata.json`、`person_weights.csv`、训练曲线、内外层模型/人员/权重表可核查来源。

三组共享CNN-GRU32与Linear32->7辅助头、14536参数。对照均匀FoG BCE，人员组用N/(P*n_person)乘训练FoG BCE，活动组均匀BCE加0.2×活动CE。后两者分开，checkpoint始终是未加权FoG BCE＋相同kernel L2。0/混合活动忽略，站立允许同时为FoG。

活动头在对照/人员组辅助loss权重为0，不能解释为训练好了的活动识别器；活动组也没有作为A2方向/意图模型独立验证。本轮接口只提供二元FoG，不暴露辅助活动预测，不用真值活动或辅助预测否决FoG。

## 推理与流式接入

沿用现有因果推理CLI，Python 3.12、原依赖及可选 `requirements-a1-deep.txt`：

```powershell
python -X utf8 scripts/predict_a1_temporal_cnn.py --input path/to/sensors.csv --model reports/a1_context_learning_v1_20261029/activity_aux/model.joblib --output reports/a1_activity_aux_new_input.csv
```

CSV要求timestamp（秒）、ankleL_gyro_x/y/z（deg/s）、60Hz且轴向/安装位置与训练一致。多记录提供subjectID/sessionID/taskID，不需要activity、fog、加速度或腰背。当前完整120点窗口每30点滑动，独立减均值；没有完整窗口就报错，不以未知代替正常。

Python：`ContextFoG.load(model_path).predict_frame(frame)`；已完成窗口流：

```python
from fogguard.modeling.a1_context_learning import ContextFoG

model = ContextFoG.load(model_path)
stream = model.new_stream()
for window in completed_gyro_windows:
    decision = stream.push(window)
```

每个独立输入源单独建stream，最多缓存9个完成窗口，取前4秒/前2秒/当前片段。记录/缺测/步长断档重置；历史不足只用真实可用片段，不多等6秒。输出`context_windows`为1/2/3，`available_at_s`为当前窗口结束。流/批分数float32容差1e-6。原始采样缓冲与切窗需调用方另行实现、验证，不能把任意CSV分块独立predict_frame当整段。

推理不重新计算个人均衡权重；它们只用于拟合loss，不以新用户ID作为特征。当前FoG分数不代表提前冻结、主观转向意图、迈步成功或患者提示疗效；A2仍需独立方向输出及组合验证。

## 复现与审计

```powershell
python -X utf8 scripts/train_a1_context_learning.py --run-dir reports/a1_context_learning_reproduction
python -X utf8 scripts/evaluate_a1_context_learning.py --run-dir reports/a1_context_learning_reproduction
python -X utf8 scripts/verify_a1_context_learning.py --run-dir reports/a1_context_learning_reproduction
python -X utf8 scripts/run_automation2_delivery_checks.py
```

需要同版公开原始数据及上一轮时序实验锁定目录，继承人员折与历史参考。新目录必须为空；报告不覆盖旧评价。MODEL_LOCK早于VALIDATION开始且源码/输入哈希一致；只读评价脚本不含拟合。OOF成绩参与阈值校准，不作独立测试。Windows pytest临时目录不可写时，可把TEMP/TMP设到工作区内artifacts/tmp新子目录。

`evaluation/index.html`、PNG/SVG、共同窗口/逐人/动作/任务/事件表与预定条件 `development_gates.json` 供复查。独立核验重建原始活动辅助目标、实际拟合人员权重、检查点FoG-only损失与外层评分，检查全部角色记录及未知时段；不声称另行独立重训。原默认、共享类型和控制器不变；模型/原始数据/生成目录不入Git，源码仅本地贡献，尚未推送或合并。

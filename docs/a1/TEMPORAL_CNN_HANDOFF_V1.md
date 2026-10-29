# A1 因果 CNN＋GRU 交接

固定方法见 [协议](TEMPORAL_CNN_PROTOCOL_V1.md)，实际取舍见 [结果](TEMPORAL_CNN_RESULTS_V1.md)。本轮只比较两种架构；模型、轮数和阈值只来自TRAIN14，随后评价已被研究的VALIDATION4，没有新候选TEST成绩。

## 模型及推理

本地 `reports/a1_temporal_cnn_v1_20261029/` 下，`cnn_control/model.joblib` 和 `cnn_gru/model.joblib` 都是全部TRAIN14随机初始化后拟合的最终模型，不是某个折或内层checkpoint权重。metadata中`threshold_participants`也为TRAIN14：本轮阈值来自TRAIN内部OOF，与前轮VALIDATION选阈值不同。

从仓库根目录安装原依赖及可选 `requirements-a1-deep.txt`，Python 3.12：

```powershell
python -X utf8 scripts/predict_a1_temporal_cnn.py --input path/to/sensors.csv --model reports/a1_temporal_cnn_v1_20261029/cnn_gru/model.joblib --output reports/a1_temporal_new_input.csv
```

输入：60Hz，timestamp单位秒，左踝gyro_x/y/z为deg/s，字段名`ankleL_gyro_x/y/z`，轴向/安装位置与原数据一致。不同记录须提供subjectID/sessionID/taskID；标签与加速度/腰背不是必需输入。每个已完成120点窗口独立去均值，步进30点，不插值。只提供不足一个完整窗口则报错，不输出“正常”。

输出保持FoG-only记录键、window_start_s/end_s、available_at_s、fog_probability、fog_threshold、is_fog；额外`context_windows`说明实际使用的1/2/3个真实片段。第一条判断仍从当前2秒窗口结束提供；不会为等满6秒而延迟启动。分数不是提前冻结预测，也不提供方向、意图、迈步成功或提示疗效。

## 按窗口流式接入

```python
from fogguard.modeling.a1_temporal_cnn import TemporalFoG

model = TemporalFoG.load(model_path)
stream = model.new_stream()  # each independent sensor stream needs its own buffer
for window in completed_gyro_windows:
    decision = stream.push(window)
```

window为`FogStarWindow`，只含3个gyro通道、120点/60Hz，按每0.5秒递增。缓冲保留最多9个完成窗口，从中取当前、之前2秒和4秒的片段；缺历史用显式长度表示，不伪造原始输入。换记录、缺测或步长断档后清空；同记录逆序/重复窗口报错。流式与批量分数允许1e-6 float32误差。

`predict_frame`是完整CSV回放接口，每次调用重新建立历史；任意分块不能直接独立调用并宣称与整段一致。在线采样缓冲/切窗由调用方实现且需单独验证。A2只能消费模型输出、对齐自己的方向判定，不能把activity真值作为运行开关；低FoG分数仍不等于有效迈步。

## 重现与核查

```powershell
python -X utf8 scripts/train_a1_temporal_cnn.py --run-dir reports/a1_temporal_reproduction
python -X utf8 scripts/evaluate_a1_temporal_cnn.py --run-dir reports/a1_temporal_reproduction
python -X utf8 scripts/verify_a1_temporal_cnn.py --run-dir reports/a1_temporal_reproduction
python -X utf8 scripts/run_automation2_delivery_checks.py
```

需要同版原始FoG-STAR，以及前轮作者CNN实验的锁定目录（用于继承TRAIN人员折和历史参考）。每个新实验使用空目录，评价不覆盖旧结果；模型锁、源码与输入哈希须一致。独立评价脚本不含拟合调用；检查锁定时间早于VALIDATION开始。

每个模型的fold_0/1/2保存inner checkpoint和从零outer refit模型/曲线/人员记录；`train_oof`是按人留出的评分，但其标签也用于选阈值，所以表中OOF F1仅研发校准指标。最终epoch取三个内层最佳轮数的中位数，不用VALIDATION决定轮数或阈值。

`evaluation/index.html`、PNG/SVG、共同窗口和事件/逐人/活动统计、`development_gate.json`及`verification.json`供审查。新模型的事件分母为各角色全部原始记录，包括无预测时段；人工构造的单元测试/推理冒烟不作为真实泛化成绩。原模型、共享类型和控制器保持原样；本地模型/数据/生成目录不入Git，源码尚未推送或合并。

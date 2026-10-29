# A1到A2：三种冻结模型的交接

A1只保留三种方法，详细实现与同窗对比见[a1/SELECTED_METHODS.md](a1/SELECTED_METHODS.md)。A2无需重跑研究过程。

| --method | 输入（60Hz） | 用途与取舍 |
|---|---|---|
| ankle_history_rf | 左踝acc xyz（g）和gyro xyz（deg/s） | 140维历史RF，同窗F1最高，误报较多 |
| bilateral_rf | 左右踝gyro xyz（deg/s） | 66维RF，仅需gyro，可先用于接通A2 |
| ankle_cnn | 左踝gyro xyz（deg/s） | 8353参数CNN，误报少、漏检更多，另需PyTorch |

## 运行

Python3.12环境安装requirements.txt及项目（`python -m pip install -e .`）。CNN额外需要PyTorch，模型包记录实际测试版本。

将三模型附件解压到`artifacts/fog-selected/`，在仓库根目录运行：

```powershell
python -X utf8 scripts/predict_selected_fog.py --bundle artifacts/fog-selected --method bilateral_rf --input artifacts/fog-selected/synthetic_input.csv --output reports/fog_smoke.csv
```

更换--method选择另两种模型。合成样例仅检查接口，不能证明实际识别效果。新数据必须与训练的单位、左右及轴向约定一致，多记录加subjectID/sessionID/taskID，timestamp为秒。缺测不补零、不插值，完整120点窗末端才输出，每30点推进；无完整窗口报错。

## A2与A1的关系

A1提供是否冻结，A2独立学习当前方向/转向，并验证其转向意图目标。A2不需要重新训练A1的FoG模型，可把自己的方向结果与以下输出对齐：

- subject_id/session_id/task_id：记录键。
- window_start_s/window_end_s：输入范围。
- available_at_s：实际可用时刻，不可回填到起点。
- fog_probability/fog_threshold/is_fog：未校准FoG分数、阈值和二元判断。

融合时按同一记录和实际可用时刻取结果，各自最多保持0.5秒，过期或无输入标unknown，不能无界沿用旧预测。方向与FoG的输入窗长度可不同，但必须记录各自可用时刻，不得借用未来的方向结果。

最终提示还需结合迈步反馈，才能判断向前提示后是否迈出脚、是否切换原地踱步。低FoG分数不能替代迈步成功检测，当前转向动作标签也不能自动当提前意图标签。共享types/interfaces与控制器本轮不改。

## A1维护入口

```powershell
python -X utf8 scripts/train_selected_fog.py --method all --output reports/selected_train_NEW
python -X utf8 scripts/evaluate_selected_fog.py --bundle reports/selected_train_NEW --output reports/selected_review_NEW
python -X utf8 scripts/bundle_selected_fog.py --models reports/selected_train_NEW --output reports/selected_bundle_NEW
```

输出目录必须是新的。模型、原始信号、整批生成结果不入Git，三模型ZIP单独共享或作为Release附件。源码当前只保留这三种A1候选及公共依赖，原项目方向/控制和兼容基线保留；旧研究实现可在Git历史648bbd8找回。当前默认集成器未自动换模型。

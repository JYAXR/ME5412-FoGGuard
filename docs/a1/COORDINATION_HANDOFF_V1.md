# A1 左右协调摘要交接 v1

方法见 [固定协议](COORDINATION_PROTOCOL_V1.md)，实际取舍见 [结果](COORDINATION_RESULTS_V1.md)，状态见 [coordination_status.json](coordination_status.json)。本轮66维双踝RF对照与78维候选输入相同，仅增加12项短窗运动协调代理量；不是新增转向头、迈步识别或已验证的生理相位。

## 保存内容

`reports/a1_coordination_v1_20261029/`含两组`bilateral_control`/`coordination_rf`的三个按人外层RF、最终`model.joblib`、OOF表、TRAIN阈值表、实际拟合人员、特征表、源/数据/配置哈希和MODEL_LOCK。`evaluation/`是锁定后VALIDATION结果、原始事件与时间、逐人/活动诊断、历史共同窗口及开发门槛。没有本轮新模型TEST成绩。

66维对照重新拟合，TRAIN-OOF及验证分数须与前轮固定配置双踝RF一致。两组前66维逐值相同，完整窗口一致；既有14人TRAIN折不变，阈值仅用TRAIN-OOF。OOF成绩参与校准，不是独立测试；四人VALIDATION已被开发使用，不是外部泛化证据。

## 推理

无需PyTorch，原numpy/pandas/scipy/scikit-learn依赖即可：

```powershell
python -X utf8 scripts/predict_a1_coordination.py --model reports/a1_coordination_v1_20261029/coordination_rf/model.joblib --input path/to/sensors.csv --output reports/a1_coordination_new_input.csv
```

Python接口：`CoordinationFoG.load(path).predict_frame(frame)`；可向`predict_windows`逐个提交已完成的`FogStarWindow`，没有历史状态。输入timestamp秒、规则60Hz、左右gyro六轴deg/s，字段为ankleL_gyro_x/y/z、ankleR_gyro_x/y/z，左右轴序/安装与训练一致。多记录应提供subjectID/sessionID/taskID，不需accel、activity、fog或fog_severity。

左踝连续段120点窗口、30点步进为唯一网格。当前120行任一右踝缺失就跳过，不重启右踝网格、不补零/插值；新记录/左踝断档才重置。无完整窗口则报错，上层必须保留未知时段，不能填正常。缺失输入规则与双踝66维对照相同；本轮不包含单踝自动备用路由。

输出只有FoG概率、TRAIN阈值、二元判断、记录键及窗口末端available_at_s；不回填到窗口起点，不提供方向、主观意图、步长、落脚或迈步成功。窗口内双向滤波仅使用该已完成窗口，不使用窗口结束后样本。包络相关延迟不是提前预测或实测双脚相位。

## 复现及展示

依赖同版公开raw及前轮锁定双踝目录。使用新目录：

```powershell
python -X utf8 scripts/train_a1_coordination.py --run-dir reports/a1_coordination_reproduction
python -X utf8 scripts/evaluate_a1_coordination.py --run-dir reports/a1_coordination_reproduction
python -X utf8 scripts/verify_a1_coordination.py --run-dir reports/a1_coordination_reproduction
python -X utf8 scripts/report_a1_coordination.py --run-dir reports/a1_coordination_reproduction --output-dir reports/a1_coordination_presentation_reproduction
python -X utf8 scripts/run_automation2_delivery_checks.py
```

核验独立公式、原始窗口网格/真值、同窗口66维前缀、六折重载评分、实际人员/OOF阈值、锁定时间、无真值推理及原始事件/时间；不声称独立重训。报告脚本只读锁定表，固定沿用旧TRAIN的三组困难配对，不重新挑选展示案例；TRAIN标准化仅用于特征色图，不进入RF。

本地展示入口为`reports/a1_coordination_presentation_v1_20261029/index.html`，含六对PNG/SVG和旧固定TRAIN配对的12项特征表。它是算法证据，不是最终PPT/3D展示。配对剖面不能当代表性性能证明，非FoG站立也不是主观主动停止标签。

原默认模型、共享类型及控制器不变，A2需独立验证方向及组合链路；低FoG分数不代表成功迈步。本地代码尚未推送/评审或合并。原始数据、模型和生成目录不入Git。

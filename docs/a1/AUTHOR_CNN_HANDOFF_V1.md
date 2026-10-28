# A1 作者 CNN 架构复现：接口与复查

训练方法与差异见 [预先固定协议](AUTHOR_CNN_PROTOCOL_V1.md)，结果见 [本轮结果](AUTHOR_CNN_RESULTS_V1.md)。本轮用于研发，只有 TRAIN14 内的 RF 诊断 OOF 与 VALIDATION4 的模型对照，没有新的 TEST 成绩。未替换原默认或共享控制器。

## 本地产物

`reports/a1_author_cnn_v1_20261029/` 保存：

- `author_cnn/model.joblib`：公开 CNN 架构，独立随机初始化、TRAIN14 拟合、VALIDATION4 选 checkpoint/阈值；不是作者 model.h5。
- `rf_gyro_2s/model.joblib`：相同三轴 gyro 窗口的33特征 RF 对照，TRAIN14 拟合、VALIDATION4 选阈值。
- 两模型目录的 `predictions.csv` 是验证集研发结果，含真值；`runtime.csv` 不含 truth_*；`fit_metadata.json` 与 `threshold_selection.json` 保留拟合/选择来源。
- `reference_rf_context` / `reference_rf_ankle` 为上一轮锁定验证评分，未重训或改阈值；输入支持不同。
- `train_oof_rf_context`：TRAIN14 内三折按人留出诊断，历史验证阈值0.16；是当前RF的错误分析，不是CNN的OOF。
- `MODEL_LOCK.json`、`protocol.json`、配置、环境、全部产物哈希；`report/index.html`、PNG/SVG、分组/事件/共同窗口表；`verification.json` 为独立文件/窗口/评分审计，不声称独立重训。

## 独立推理

从仓库根目录，安装原依赖以及可选 `requirements-a1-deep.txt`。使用 Python 3.12：

```powershell
python -X utf8 scripts/predict_a1_author_cnn.py --input path/to/sensors.csv --model reports/a1_author_cnn_v1_20261029/author_cnn/model.joblib --output reports/a1_gyro_new_input.csv
```

CSV 只需 `timestamp`（秒）、`ankleL_gyro_x/y/z`（deg/s）；60Hz、与训练一致的轴向/安装位置。若有多个记录，请提供 subjectID/sessionID/taskID，避免跨记录切窗。无需 accel、腰背、activity、fog、fog_severity。缺 gyro 或断档时不填零；无完整窗口则明确报错，不产生正常判定。

Python 接口为 `GyroFoG.load(...).predict_frame(frame)`。CNN 每个已完成窗口独立去均值，输出在窗口结束可用；不是提前预警，也不输出方向、意图、实际落脚或迈步成功。A2 可按记录键、窗口和 available_at_s 对齐自己的方向评分；不能用活动真值选择提示。

本模型为完整CSV离线回放接口，不提供跨调用的原始样本缓冲；分块/在线适配需保留120点窗口与30点步进，并单独验证，不可把每个任意分块当独立记录后宣称与整段一致。

## 复现

原始数据版本与上一轮固定划分一致；需要已有固定划分训练锁及验证参考文件。新目录必须不存在或为空，后续命令不覆盖历史报告：

```powershell
python -X utf8 scripts/run_a1_author_cnn.py --run-dir reports/a1_author_cnn_reproduction
python -X utf8 scripts/report_a1_author_cnn.py --run-dir reports/a1_author_cnn_reproduction
python -X utf8 scripts/verify_a1_author_cnn.py --run-dir reports/a1_author_cnn_reproduction
python -X utf8 scripts/run_automation2_delivery_checks.py
```

新划分、不同版本原始数据或被改动的参考文件会被拒绝。Windows 若 pytest 系统临时目录不可写，可把 TEMP/TMP 指向工作区内新建的 artifacts/tmp 子目录再运行；不需要改模型或测试断言。

模型和原始数据不进 Git；源码/协议/汇总结论为本地贡献，未推送或合并。尚未完成跨新队列证明、CNN跨人OOF、转向辅助学习或临床验证。

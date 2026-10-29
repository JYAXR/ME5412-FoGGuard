# A1 模型接入

A1 已完成冻结识别算法交付，提供三种方法、训练配置、模型包、同窗评价和运行入口。A2 接入后完成方向与感知融合；B1 完成反馈及控制，B2 完成联合评价。

## 模型选择

| --method | 输入（60 Hz） | 取舍 |
|---|---|---|
| `ankle_history_rf` | 左踝 acc xyz（g）和 gyro xyz（deg/s） | 140 维历史特征，F1 0.691、召回较高 |
| `bilateral_rf` | 左右踝 gyro xyz（deg/s） | 66 维特征，F1 0.686，A2 接入参考 |
| `ankle_cnn` | 左踝 gyro xyz（deg/s） | 8353 参数 CNN，F1 0.677，误报少、漏检更多 |

完整效果与训练政策见 [方法说明](a1/SELECTED_METHODS.md)。模型包包含权重、配置、依赖版本、文件哈希和合成接口样例。

## 运行

安装依赖及项目，将模型解压到 `artifacts/fog-selected/`：

```bash
python -X utf8 scripts/predict_selected_fog.py --bundle artifacts/fog-selected --method bilateral_rf --input artifacts/fog-selected/synthetic_input.csv --output reports/fog_interface_NEW.csv
```

CNN 另外安装 `requirements-a1-deep.txt`。示例只用于接口检查；真实数据采用相同单位、轴向和采样率，多记录提供身份键。缺测不补零、不插值；没有完整窗口时不产生有效预测。

## A2 接入

A1 输出记录键、`window_start_s / window_end_s / available_at_s`、`fog_probability / fog_threshold / is_fog`、模型版本和来源。窗口结束后结果才可用，最多保持 0.5 秒；不回填到窗口起点。

A2 按同记录与实际可用时刻融合方向，加入确认、拒识、恢复观察和时效门控。A1 输出不含方向或迈步成功，不能直接作为完整控制输入。任务步骤见仓库内 `docs/tasks/A2.md`，字段见 `docs/INTERFACE_CONTRACT.md`。

## 下游关系

B1 接收感知事件并观察运动反馈；低 FoG 分数不能替代迈步成功判断。B2 分别评价感知、反馈和策略，再导出展示包。工业设计从保存输出制作最终画面。

## 维护入口

```bash
python -X utf8 scripts/train_selected_fog.py --method all --output reports/selected_train_NEW
python -X utf8 scripts/evaluate_selected_fog.py --bundle reports/selected_train_NEW --output reports/selected_evaluation_NEW
python -X utf8 scripts/bundle_selected_fog.py --models reports/selected_train_NEW --output reports/selected_bundle_NEW
```

模型、原始数据和生成目录不提交 Git；权重通过交接包单独传递。来源与完成状态按仓库内 `docs/PROJECT_STATUS.json` 记录。

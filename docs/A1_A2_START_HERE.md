# A1 模型接入

A1/A2 已完成并合并感知交付：三种冻结方法、当前方向、因果融合、固定独立测试及统一图表。此页说明 A1 模型入口；下游直接使用 [感知输出](a2/fusion/README.md)，B1 继续反馈/控制，B2 继续联合验证。

## 模型选择

| --method | 输入（60 Hz） | 取舍 |
|---|---|---|
| `ankle_history_rf` | 左踝 acc xyz（g）和 gyro xyz（deg/s） | 140 维历史特征，TEST F1 0.485、Recall 0.806 |
| `bilateral_rf` | 左右踝 gyro xyz（deg/s） | 66 维特征，TEST F1 0.554、Recall 0.838；固定融合参考 |
| `ankle_cnn` | 左踝 gyro xyz（deg/s） | 8353 参数 CNN，TEST F1 0.557、Recall 0.581 |

上表为同一组 1,605 个 TEST 窗口的成绩，[正式报告](a_group/README.md) 给出分母和误报漏检。训练方法与开发验证选型说明见 [方法文档](a1/SELECTED_METHODS.md)。模型包包含权重、配置、依赖版本、文件哈希和合成接口样例。

## 运行

安装依赖及项目，将模型解压到 `artifacts/fog-selected/`：

```bash
python -X utf8 scripts/predict_selected_fog.py --bundle artifacts/fog-selected --method bilateral_rf --input artifacts/fog-selected/synthetic_input.csv --output reports/fog_interface_NEW.csv
```

CNN 另外安装 `requirements-a1-deep.txt`。示例只用于接口检查；真实数据采用相同单位、轴向和采样率，多记录提供身份键。缺测不补零、不插值；没有完整窗口时不产生有效预测。

## 已交付的 A1+A2 感知接口

A1 输出记录键、`window_start_s / window_end_s / available_at_s`、`fog_probability / fog_threshold / is_fog`、模型版本和来源。窗口结束后结果才可用，最多保持 0.5 秒；不回填到窗口起点。

A2 已实现同记录与实际可用时刻融合，含确认、拒识、恢复观察和时效门控。A1 输出不含方向或迈步成功，不能直接作为完整控制输入。后续 A1/A2 仅提供维护支持，字段见 [接口约定](INTERFACE_CONTRACT.md)，接收复核见 [B2 任务](tasks/B2.md)。

已实现的 [A1+A2 感知入口](a2/fusion/README.md) 可直接融合两个运行表，或从原始传感器运行两模型后导出 `perception_runtime.csv`；方法、正式测试、融合与联合时间线统一展示于 [英文图表页](a_group_showcase/index.html)。

## 下游关系

B1 接收感知事件并观察运动反馈；低 FoG 分数不能替代迈步成功判断。B2 分别评价感知、反馈和策略，再导出展示包。工业设计从保存输出制作最终画面。

## 维护入口

```bash
python -X utf8 scripts/train_selected_fog.py --method all --output reports/selected_train_NEW
python -X utf8 scripts/evaluate_selected_fog.py --bundle reports/selected_train_NEW --output reports/selected_evaluation_NEW
python -X utf8 scripts/bundle_selected_fog.py --models reports/selected_train_NEW --output reports/selected_bundle_NEW
```

模型、原始数据和生成目录不提交 Git；权重通过交接包单独传递。来源与完成状态按仓库内 `docs/PROJECT_STATUS.json` 记录。

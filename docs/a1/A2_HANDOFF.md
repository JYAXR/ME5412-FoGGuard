# A1 → A2 交接入口

> 后续多模型比较已完成，最新对照与对应推理入口见 [多模型交接](MULTIMODEL_HANDOFF.md)。下文保留首轮近期特征实验的具体文件契约。

本轮实验方法见 EXPERIMENT_V1.md，实测结论见 RESULTS_V1.md。不要仅凭新文件名选择模型；先看公平基线和候选之间的结果与代价。模型不会自动替换原项目默认。

## 本地交付目录

本轮输出目录为 reports/a1_recent_trend_v1_20261028/。每个方案子目录 baseline_2s_v1/、recent_trend_v1/ 含：

- model.joblib：该方案全量拟合的 FoG-only 模型。只用于后续新输入，不用其训练数据回放成绩替代 OOF。
- a2_runtime.csv：外层参与者留出预测，不含 truth_*。
- evaluation_truth.csv：独立标签附件，按记录和窗口键对齐。
- oof_predictions.csv：供原独立核验器使用的完整留出结果，包含评估真值。
- training_report.json、feature_schema.json、fold_audit.json：数据版本、折、阈值和特征契约。
- recognition/：全部 106 条原始记录的独立核验、真实错误案例、事件覆盖和交互页。

根目录 comparison.csv/json 包含三组窗口和事件对照；subject_comparison.csv 包含逐人结果及阳性样本数；figures/ 是同源 PNG/SVG；protocol.json 在本轮预测前保存规则和代码哈希。

## 直接接现有集成器

a2_runtime.csv 保留旧方向基线的 direction/direction_confidence 等字段，因此可以直接接入现有 run_fogstar_end_to_end.py。方向来源明确标为 unchanged_historical_baseline_oof，不是 A1 新训练了方向/意图模型。

```powershell
python -X utf8 scripts/run_fogstar_end_to_end.py --predictions reports/a1_recent_trend_v1_20261028/recent_trend_v1/a2_runtime.csv --output reports/a1_recent_trend_v1_20261028/recent_trend_v1/a2_events.csv
```

该命令生成事件/提示候选，不验证真实患者响应，不是新的真实迈步反馈闭环。

## 对新输入推理

```powershell
python -X utf8 scripts/predict_a1_fog.py --input path/to/new_sensor_data.csv --model reports/a1_recent_trend_v1_20261028/recent_trend_v1/model.joblib --output reports/a1_new_predictions.csv
```

输入至少有 timestamp 和训练时完全相同的左踝/腰背 12 个通道，60 Hz，加速度 g、角速度 deg/s、轴向一致。记录键可提供，缺省使用既有加载器默认。缺测不补零，不能构成完整窗口时没有预测，整段没有可用窗口会明确报错。

输出列：

| 字段 | 含义 |
|---|---|
| subject_id/session_id/task_id | 记录键 |
| window_start_s/window_end_s | 输入窗口范围 |
| available_at_s | 本方案等于窗口最后采样点时间，不能提前回填 |
| fog_probability | 未校准模型分数，沿用共享字段名 |
| fog_threshold/is_fog | 部署阈值与当前窗口候选 |
| feature_version/model_version | 特征和模型格式版本 |
| prediction_source | full_fit_new_input；明确区别于 OOF |

独立推理输出不含 direction、state、truth_* 或反馈字段。A2 将自己的方向结果按记录键、窗口时间和可用时间对齐，然后调用现有事件适配层。若采用不同时间频率，必须明确新鲜度和保持时长，不能无界沿用旧方向。

## A2 开发接续

1. 先用已有 a2_runtime.csv 验证文件读取和时间逻辑。
2. 开发方向/转弯前任务时沿用相同外层参与者划分；不要把 A1 已分析过的数据称为新独立外部测试。
3. 将活动真值与运行时输入分开；转弯前候选仍需负样本，203/202 个正片段只是可行性证据。
4. 分别报告 FoG、方向及联合提示覆盖，不只报告高置信度方向子集。
5. 保留 unknown、缺测与安全退化；真实迈步反馈另行验证，不能把 A1 低 FoG 分数解释为已迈步恢复。

共享 types.py/interfaces.py 和原控制器未改。本轮只提供新适配与冻结识别实验，正式接口替换由团队互审。

## 共享与复现

源码和文档可随贡献分支交接。原始患者数据、临床表、部署模型及整批运行目录不加入源码 Git。模型/OOF/图表如需共享，作为经核对的版本化交付文件传递，并附 manifest.json；原数据由接收方自行取得同版公开文件。

本轮当前为本地交付，未自动推送、合并或更改旧 Issue。A2 可以在收到相应分支与产物后按上述步骤接续。

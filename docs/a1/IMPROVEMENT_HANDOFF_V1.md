# A1 继续改进：两个独立候选的接口

日期：2026-10-29。本轮包含姿态/缓慢转动特征搜索和输入可用性备用，组件分别评价后又实际评价冻结组合；不能拼接各自最好的数字。原默认不自动替换。实测与局限见 [本轮结果](IMPROVEMENT_RESULTS_V1.md)。

## 当前联合研究候选

目录 `reports/a1_joint_v1_20261029_verified/` 提供所选主模型＋同一左踝备用。相对旧六秒 RF，原始事件覆盖 70→75/101，时间 Precision 0.5357→0.5788、Recall 0.5985→0.6441，误报警事件 235→232。组合是在观察组件结果后提出，仍是旧队列探索，未替换默认。

```powershell
python -X utf8 scripts/predict_a1_fallback.py --input path/to/new_sensor_data.csv --model-dir reports/a1_joint_v1_20261029_verified --output reports/a1_joint_scores.csv
```

模型目录有 primary_model.joblib、ankle_model.joblib，接口与下文备用相同；availability_fallback/a2_runtime.csv 为联合留出输出。primary_only/ 为所选主组件，frozen_context_baseline/ 为旧六秒 RF；不能把两者混称同一主模型。组合源自相同外层人员分折，不重新拟合或选阈值。

## 姿态特征及训练内选择器

目录 `reports/a1_posture_search_v1_20261029/` 下的 `context_rf/`、`posture_rf/`、`inner_selected/` 分别为旧六秒 RF、新特征固定 RF、训练内选择流程。每组提供 model.joblib、部署配置、无 truth_* 的 a2_runtime.csv、分离真值、完整 OOF 与原始识别审计。训练内选择器每折可以选不同配置，不能把某一外层配置的分数叫作部署模型测试成绩。

```powershell
python -X utf8 scripts/predict_a1_fog.py --input path/to/new_sensor_data.csv --model reports/a1_posture_search_v1_20261029/inner_selected/model.joblib --output reports/a1_selected_scores.csv
```

独立推理只需要传感器。它输出 FoG 分数和阈值，不训练方向、意图或迈步完成。逐窗持续调用使用 model.new_stream()，与六秒上下文契约相同；不要每个窗口重新初始化。

## 左踝备用

最终目录 `reports/a1_ankle_fallback_v1_20261029_verified/` 提供 primary_model.joblib 与 ankle_model.joblib。前者是冻结的旧六秒 RF；后者为另行训练的左踝 RF。配套 OOF、人员分折、部署阈值和路由审计保留在目录中。

```powershell
python -X utf8 scripts/predict_a1_fallback.py --input path/to/new_sensor_data.csv --model-dir reports/a1_ankle_fallback_v1_20261029_verified --output reports/a1_fallback_scores.csv
```

输入包含 timestamp、左踝加速度/角速度六通道；腰背六通道可存在或缺失。有效主窗口用主模型，没有新鲜主模型判定时才补左踝；两者都不能组成完整窗口则没有预测，不能当作正常。没有用零填补腰背，没有根据冻结/活动真值或评分高低切模型。CSV 采样率、单位和轴向须保持原数据契约。

输出 sensor_route 为 primary_ankle_back 或 fallback_ankle，保留各自模型版本与阈值。新的传感器推理不提供方向；A2 必须自己按记录/时刻对齐方向，备用时方向未知也不能套用过期方向。实验 a2_runtime.csv 中原有方向等旧字段仅在主模型行有历史来源，备用行的空值表示未提供，不能宣称是新的方向预测。

路由是完成窗口评分后的因果合并：主判定最长保持半秒；被忽略的左踝到达不能刷新主模型有效期。同一采样时刻以原审计的一微秒数值精度匹配，保留原始时间字段；同刻主模型优先。不要以未来主预测撤销过去备用判定。实时实现需为两个传感器模型分别保留历史，并按实际可用时刻执行此规则，接入后还要单独评估调度延迟。

## 评价与展示

姿态搜索使用相同 9,636 个窗口，能比较窗口 F1。备用模型扩充了有效窗口集合，其主要公平对照是相同 106 条原始记录、101 个冻结事件和逐点时间指标；不能将不同窗口池的 F1 写成同窗口提升。

两个目录分别含 index.html、PNG/SVG 图及 ppt_comparison.csv；训练内诊断、六配置比较、原始错误案例和备用覆盖都有保存数据。供汇报同学直接编排，不包含三维最终展示或患者提示响应。

standalone FoG 输出仍需 A2 的方向与事件层融合。原默认集成器会另外施加确认、退出迟滞与恢复规则，最终提示成绩需要重新评价，不能直接沿用这里的原始报警数字。低 FoG 分数仍不等于真实迈步成功。

源码与说明在本地贡献分支，未自动推送/合并；模型、原始数据和批量输出未加入源码 Git。共享模型与 CSV 时保留各自 manifest、配置、来源及对应说明。

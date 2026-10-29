# A1 实验索引与保留决定

2026-10-29。源码及日期证据保留供复现，日常接续使用[A2入口](../A1_A2_START_HERE.md)。只比较同一行的固定对照，不能把不同人群/窗口/阈值的数值排成总榜。

| 尝试 | 评价范围 | 本行主要结果 | 判断及来源 |
|---|---|---|---|
| 近期/趋势特征 | 旧22人OOF探索 | F1 0.5817到0.5879；事件79到78/101 | 小幅改善，有取舍。[结果](RESULTS_V1.md) |
| RF/SVM/HistGB/小CNN | 同9636窗，旧22人OOF | F1 0.5994/0.5533/0.5812/0.5479 | 本轮固定配置RF较高。[结果](MULTIMODEL_RESULTS_V1.md) |
| 困难纯负加权 | 同流程旧22人OOF | F1 0.5994到0.5792，误报增 | 不采用。[结果](HARD_NEGATIVE_RESULTS_V1.md) |
| 六秒历史摘要 | 同9636窗，旧22人OOF | F1 0.5994到0.6188；误报337到235段，事件77到70 | 有研究价值及漏检代价。[结果](CONTEXT_RESULTS_V1.md) |
| 事件目标及确认 | 旧106记录/101事件 | 触及70到71事件，误报时间增14.5% | 不采用。[结果](EVENT_RESULTS_V1.md) |
| 姿态特征/模型选择 | 旧22人OOF探索 | 选择流程F1 0.6188到0.6317，召回降 | 未全面更好。[结果](IMPROVEMENT_RESULTS_V1.md) |
| 缺测备用/联合组件 | 同106记录，窗口扩展 | 联合触及70到75/101，时间P/R升 | 支持有价值，仍为探索。[结果](IMPROVEMENT_RESULTS_V1.md) |
| 固定14/4/4重训八模型 | 内部隔离TEST，旧队列 | 六秒RF F1 0.5113，备用组合0.5307 | 误报严重，保留锁定结果。[结果](FIXED_SPLIT_RESULTS_V1.md) |
| 作者CNN架构 | TRAIN/VAL研发 | 同gyro F1 0.6388到0.6772，事件10到8/10 | 少误报、更多漏检，非原创架构。[结果](AUTHOR_CNN_RESULTS_V1.md) |
| CNN加GRU | 同流程，VAL2014窗 | F1 0.5471到0.6571，误报时间增17.7% | 未通过升级条件。[结果](TEMPORAL_CNN_RESULTS_V1.md) |
| 人员均衡/活动辅助 | 同流程，VAL2014窗 | F1 对照0.6476、均衡0.6717、辅助0.6438 | 均衡误报增，辅助无F1收益。[结果](CONTEXT_LEARNING_RESULTS_V1.md) |
| 匹配左踝到双踝 | 同TRAIN窗口/VAL2014窗 | F1 0.6234到0.6860，误报时间减15.8% | 开发门槛通过，保留基线。[结果](BILATERAL_RESULTS_V1.md) |
| 追加12项协调摘要 | 同双踝流程，VAL2014窗 | F1 0.6860到0.6801，时间Recall降 | 不采用。[结果](COORDINATION_RESULTS_V1.md) |
| 10维不变摘要替换 | 同双踝流程及外部研发 | VAL F1 0.6860到0.6234，外部0.5657到0.5154 | 不采用当前压缩方案。[结果](INVARIANT_RESULTS_V1.md) |

诊断及验证单列，不算新模型改善：[任务可行性](../algorithm_readiness/RESULTS_2026-10-28.md)、[泛化审查](GENERALIZATION_ASSESSMENT_V1.md)、[分歧回放](DISAGREEMENT_RESULTS_V1.md)、[Stanford迁移](STANFORD_EXTERNAL_RESULTS_V1.md)、[外部失败](EXTERNAL_FAILURE_DIAGNOSTIC_RESULTS_V1.md)。

## 代码保留范围

- A2日常使用：`predict_a1_bilateral.py`、`modeling/a1_bilateral.py`及公共依赖。`export_a1_handoff.py`打包附件。
- 汇报与证据：本索引、`PROJECT_BRIEFING.md`、结果页和核验器、`build_a1_briefing.mjs`。
- 研究复现：其余训练/评价脚本及模型类完整保留，不作为A2默认操作入口。避免删除历史模型引用的模块。
- 不入源码Git：原始数据、模型权重和整批实验输出。附件仅冻结模型及合成接口样例，单独共享。

历史文档的“下一轮”是当时建议，不是当前仍需执行的任务清单。当前默认模型及共享控制器不变。

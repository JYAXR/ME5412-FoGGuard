# 汇总数据字段说明

- headline_metrics：主指标及其样本范围；value 是 0–1 比例。
- method_comparison：同一明确方向范围的方法对比；accuracy / macro_f1 是 0–1 比例。
- per_class：分别提供 known_direction_targets（1130 个窗口）和 full_stream（1921 个窗口）的 P/R/F1；support 是计数。
- per_subject：四名开发验证参与者的窗口数、准确率、macro-F1 和覆盖率。
- confusion_main / confusion_full：行是真值，列是预测，值为窗口计数。
- preturn_feasibility：二分类可行性结果，召回率、精确率和准确率为比例，事件数为整数，误报率单位为次/分钟。
- preturn_threshold_diagnostic：固定模型在开发验证上的诊断点，未用于重新挑选发布阈值。

主三类与完整四类评价范围不同，不应混合计算。未包含患者原始数据或逐窗输出。

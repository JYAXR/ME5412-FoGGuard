# 评估工具入口

B2 的正式任务见 [评估与系统联调](tasks/B2.md)。工程工具包括 `run_automation2_delivery_checks.py`、`run_engineering_comparison.py` 与 `verify_engineering_comparison.py`，指标工具位于 `evaluation/` 和 `recognition/`。

项目检查涵盖代码测试、工程场景、命令生命周期及策略对照。构造输入验证实现，真实模型成绩使用独立真值和对应的留出/开发范围。A1 三模型评价入口为 `evaluate_selected_fog.py`。

请按 [快速开始](QUICKSTART.md) 运行，并将联合结果按 [接口约定](INTERFACE_CONTRACT.md) 导出。项目完成状态见 [项目状态](PROJECT_STATUS.json)。

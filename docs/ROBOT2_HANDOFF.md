# 方向与事件模块技术入口

A2 的完整交付要求见 [任务书](tasks/A2.md)。当前可使用的基础为：

- `turning/` 的方向规则与未知输出。
- `context/fogstar.py` 的两窗冻结确认、三窗恢复观察、迟滞、事件合并和方向平滑。
- `fogstar_pipeline.py` 的事件与控制适配。
- `docs/examples/robot2_predictions.csv` 的构造候选接口样例。

这些模块提供开发基础。A2 需要交付自己的方向评价、转向前任务验证，以及 A1 三模型输出的显式适配器。来源有效期最多 0.5 秒，详见 [架构](ARCHITECTURE.md) 与 [接口](INTERFACE_CONTRACT.md)。

运行样例见 [快速开始](QUICKSTART.md)。恢复观察表示模型输出回落；B1 另行处理运动响应证据。

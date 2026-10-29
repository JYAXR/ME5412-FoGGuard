# 提示控制模块技术入口

B1 的正式任务见 [运动反馈与提示控制](tasks/B1.md)。工程基础包括 `cueing/guidance.py`、`cueing/march.py`、`cueing/contracts.py` 与 `data/responses.py`。

当前模块能消费感知事件和明确标为模拟的响应，生成脚印、声音、渐弱、原地交替及启停命令。B1 需要完成真实 IMU 运动代理、反馈来源适配、有效无响应计时和最终参数一致性验证。

字段和来源约束见 [接口约定](INTERFACE_CONTRACT.md)，运行见 [快速开始](QUICKSTART.md)。共享类型与模块内提示契约分别管理，接收方评审后再进行共享接口变更。

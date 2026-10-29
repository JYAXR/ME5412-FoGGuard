# 数据可行性与接口核查工具

项目方案和任务以 [项目说明](../PROJECT_OVERVIEW.md)、[分工](../ROLES_AND_ROADMAP.md) 为准。A1 已完成冻结算法交付，见 [A1 方法](../a1/SELECTED_METHODS.md)。

## 工具用途

`evaluation/readiness.py` 提供记录通道支持、转向前正候选、人员划分与运行/真值分离检查。`audit_algorithm_readiness.py` 和 `run_algorithm_readiness.py` 可生成数据核查资料；它们服务数据可行性及兼容接口审计，不是 A1 三模型的正式效果入口。

安装 `requirements-readiness.txt` 后可运行核查脚本。组合脚本会训练或读取通用兼容基线并生成其审计，输入和模型范围必须写入输出，不能将其数字替代三模型成绩。

## 下游使用

A2 可使用转向前候选检查输入完整性，但仍须构造负例并完成人员隔离评价。B1 可检查所需通道；没有独立步/位移真值时反馈只能声明运动代理。B2 核对来源、哈希和有效时间。

`contract-v1.json` 描述模块适配目标。详细字段见 [接口约定](../INTERFACE_CONTRACT.md)，新适配器仍由对应负责人实现和接收方验收。

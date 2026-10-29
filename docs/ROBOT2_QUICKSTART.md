# 感知时间线接入

环境与样例命令统一在 [快速开始](QUICKSTART.md)。A2 接入 A1 后将方向、事件和输入质量交给 B1/B2，具体步骤见 [A2 任务书](tasks/A2.md)。

提示控制读取 `event_id / event_phase / stable_fog_detected / turn / turn_confidence / input_fresh`，不能绕过事件确认直接使用原始 FoG 阳性触发。所有来源按实际可用时刻对齐，缺失或过期保留未知。

工业设计读取 B2 保存的时间线，不在展示层复制事件和提示规则。职责及输出见 [分工](ROLES_AND_ROADMAP.md) 和 [接口约定](INTERFACE_CONTRACT.md)。

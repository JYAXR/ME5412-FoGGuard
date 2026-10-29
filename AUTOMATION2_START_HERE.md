# 评估与联调入口

FoGGuard 的项目入口为 [README](README.md)，角色与当前任务见 [分工](docs/ROLES_AND_ROADMAP.md)，完成情况见 [项目状态](docs/PROJECT_STATUS.json)。

B2 负责感知、反馈、控制的联合验证，先阅读 [B2 任务书](docs/tasks/B2.md)、[接口约定](docs/INTERFACE_CONTRACT.md) 和 [快速开始](docs/QUICKSTART.md)。

```bash
python -X utf8 scripts/run_automation2_delivery_checks.py
```

工程演示使用构造候选和模拟响应。真实数据审计需同时提供原始数据、OOF 和匹配训练报告；未提供时记录 `NOT_RUN`。A1 三模型的开发验证成绩由专用评价入口提供，不能用通用兼容模型的结果替代。

代码接手要求见 [AGENTS.md](AGENTS.md)，提交与模块互审见 [协作规则](docs/COLLABORATION.md)。

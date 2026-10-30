# 新版代码与复现

本分支已将新增文件放到仓库相应的 src/、scripts/、config/ 与 tests/ 路径。模型、原始传感器数据和本机环境不在仓库中。保留队友现有文件；提交与合并按项目协作规则审阅。

代码基线：b04ea8618e155d62891ca2970fc7c028ce3b1f06。新版运行及源文件摘要位于 02_Results/01_Updated_Pipeline/RESULTS.json，上传文件清单位于同目录 UPLOAD_MANIFEST.json。

## 主要代码

- src/fogguard/motion/proxy_feedback.py：运动迹象代理、时间与质量门控。
- src/fogguard/cueing/proxy_replay.py：新模型感知输出与运动代理的提示适配，含反馈基线和渐弱开关。
- scripts/run_updated_cue_comparison.py：本轮 240 次提示比较。
- scripts/verify_updated_cue_comparison.py：独立复核已保存输出。
- scripts/evaluate_sensor_variants.py 与 evaluation/sensor_variants.py：72 份随机扰动信号生成和冻结模型测试。
- scripts/evaluate_b2_randomized_models.py：24 段开发验证数据及异常条件测试。
- config/、tests/：对应固定参数和检查。

为复现本机 356 项测试所覆盖的源码状态，快照也保留了早期的逐段分析与候选波峰辅助代码。它们没有接入本轮报告的提示流程，不应作为已验证的真实迈步识别功能。原有团队控制器由项目仓库提供。

## 环境

使用 Python 3.12，安装原项目与 A 组依赖，优先匹配团队模型包标注的依赖版本。原项目的 requirements-a2.txt 和 requirements-a1-deep.txt 包含所需依赖范围。将项目安装到环境，或把 src 加入 PYTHONPATH。

准备 FoG-STAR sensor_data.csv，以及团队正式模型包（保留 fog-selected 和 a2-current 目录结构）。运行脚本会按团队 TEST_LOCK 核对权重。不要根据公开的 TEST 成绩调参再称为新独立测试。

## 从原始输入重现

在原项目根目录执行。尖括号替换成实际路径，输出必须用新的目录名称，避免覆盖已有证据。

```text
python scripts/run_b2_randomized_checks.py --raw <sensor_data.csv> --bundle <package>/fog-selected --direction-model <package>/a2-current/current/model.joblib --output-dir <plan-run>
python scripts/evaluate_b2_randomized_models.py --plan <plan-run>/sample_plan --package <package> --output-dir <model-run>
python scripts/run_b1_b2_joint_review.py --plan <plan-run>/sample_plan --model-run <model-run> --package <package> --output-dir <prior-joint>
python scripts/run_updated_cue_comparison.py --plan <plan-run>/sample_plan --model-run <model-run> --package <package> --prior-joint <prior-joint> --output <updated-run>
python scripts/verify_updated_cue_comparison.py --run <updated-run> --prior-joint <prior-joint> --output <new-verification.json>
```

随机生成数据补充测试：

```text
python scripts/evaluate_sensor_variants.py --plan <plan-run>/sample_plan --package <package> --clean-run <model-run> --output <generated-run>
```

完整原有交付检查：

```text
python scripts/run_automation2_delivery_checks.py --output-dir <new-check-run>
```

省略通用 OOF 审计输入时，那一项为 NOT_RUN，不是 PASS。运行得到的完整真实数据派生文件应保留在本地实验目录，图表和交接包只使用明确范围的汇总或构造样例。

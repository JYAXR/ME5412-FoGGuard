# 自动化 2：从这里检查我们的交付

本分支补齐评估、策略对照、失败案例和一个可选踏步仿真扩展。**识别效果仍有不足，代码可运行不等于模型达标或全组项目完成。**

依赖原仓库 [PR #24](https://github.com/JYAXR/ME5412-FoGGuard/pull/24)，其上还有 [PR #23](https://github.com/JYAXR/ME5412-FoGGuard/pull/23)。本贡献通过独立分支提交评审，未替团队合并到 main，未关闭原任务。

**原仓库评审入口：[PR #25](https://github.com/JYAXR/ME5412-FoGGuard/pull/25)。**
首个源码提交的 Windows / Ubuntu 离线 CI 均已通过，见 [Actions 记录](https://github.com/NuyoahQQ/ME5412-FoGGuard/actions/runs/39898636276)；后续提交状态以各自 Checks 为准。

## 新增：五种提示方案的比较结论

**[直接看中文结论和对照表](docs/automation2/evidence/2026-10-23-engineering-v1/RESULTS.md)**，无需逐个操作网页。112 个参数化案例 × 5 方案，共 560 次运行，与之前的 110 次演示分开保存。

- 持续向前响应案例：冻结标注结束后继续提示均值，开启渐弱 1.0 秒、关闭渐弱 7.5 秒；固定提示也是 1.0 秒。
- 相对裁剪后预设步长的间距误差：自适应 2.46 cm、固定间距 6.67 cm。
- 踏步提示总时长 9.25 秒、倒计时 9.00 秒；未显示时长优势。慢节奏还暴露四步与三秒窗口不匹配。
- 四个故意注入冻结误判的主动停步案例，五方案都错误提示；缺点如实保留。

[下载本轮全部输入和时间线](https://github.com/NuyoahQQ/ME5412-FoGGuard/releases/tag/automation2-engineering-comparison-2026-10-23)。本轮本地 245 项测试及独立复核通过；CI 以本次提交 Checks 为准。以上是工程比较，不能宣称患者疗效或完整方案全面更好。

## 不运行代码，先检查结果

[下载本次版本附件](https://github.com/NuyoahQQ/ME5412-FoGGuard/releases/tag/automation2-handoff-2026-10-23)：

| 文件 | 怎么检查 |
|---|---|
| `FoGGuard-recognition-review-v2.zip` | 解压，打开 `results/真实数据冻结识别验收.html`；先点“查看事件 1”，查看缺失输入造成的无预测记录，再检查有预测但漏报的案例 |
| `FoGGuard-integrated-demo-v3.zip` | 打开包内 `验收演示.html`；切换固定、倒计时、原地交替、关闭渐弱、关闭间距调整，检查目标、声音和停止原因 |
| `FoGGuard-demo.mp4` | 带节拍的演示视频；仅展示保存的仿真输出，不是人体疗效视频 |
| `FoGGuard-report-draft.pptx` | 可编辑课程汇报初稿，需第五位同学完善成员信息、视觉呈现及最终格式 |

这些是带版本的交付快照，源码包可能早于当前分支后续文档提交。最新源码和 AI 入口以本分支为准。附件大小与 SHA256 见 [release-assets.json](docs/automation2/release-assets.json)。Windows 浏览器即可检查，Ubuntu 不是必要条件。

## 我们具体做了什么

| 部分 | 新增内容 | 代码与说明 |
|---|---|---|
| 原链路评估 | 合成传感器、事件、提示的端到端比较；固定/响应/单因素对照；生命周期审计与主动停步失败案例 | [初版评估交接](docs/AUTOMATION2_EVALUATION.md) |
| 仿真补充 | 可选原地左右交替备用、向前恢复、间距调整、渐弱与再冻结；21 个内置场景＋1 个机器人 2 交接样例，共 22×5 次运行 | [完整演示交接](docs/COMPLETE_DEMO_HANDOFF.md)、`src/fogguard/cueing/march.py` |
| 真实识别 | 核验 9,636 个 OOF 窗口；按参与者、动作、任务、冻结表现分组；全量事件与时间覆盖、缺测记录和失败案例 | [识别评估](docs/automation2/RECOGNITION.md)、[recognition/](recognition/) |
| 接手与复现 | 一键检查、版本化证据、输入/资产哈希、AI 状态与后续上传步骤 | [AI 接手说明](docs/automation2/AI_HANDOFF.md)、[机器可读状态](docs/automation2/status.json) |

## 真实数据结论：不要把不同指标混为一谈

- 22 人、106 段原始记录；95 段有预测，11 段因所需输入通道缺失而没有有效窗口。
- 有效窗口：冻结精确率 **56.02%**、召回率 **65.74%**、冻结 F1 **0.605**、两类宏平均 F1 **0.749**。
- 全量逐点真值：101 个冻结事件中 76 个与冻结判定有重叠；冻结时间覆盖率只有 **56.03%**，报警时间精确率 **54.22%**。
- 25 个完全未覆盖事件中，18 个没有任何模型判定支持，7 个有判定支持但没有冻结报警。没有预测不等于判断正常。
- “事件有重叠”允许一个报警跨越多个事件，不是一对一事件 F1。概率从窗口结束才可用；本评估不是提前预测或硬件延迟实验。
- 模型只识别冻结/非冻结。按真值统计的拖步、颤动、运动不能召回率分别为 50.0%、73.3%、64.7%，不是模型在预测亚型，也不自行定义“动态冻结”。

所有表格可直接在 GitHub 阅读：[日期证据目录](docs/automation2/evidence/2026-10-23/)。其中有每人/每段/每类指标、11 段无预测问题清单、101 个真实事件和页面截图。原始信号、全量窗口表和模型未塞进源码库。

## 运行与检查

本贡献尚未进入默认 main。首次拿代码请明确选择贡献分支，并使用一个不存在的新目录，避免覆盖已有项目：

```bash
git clone --branch feat/automation2-review-and-recognition --single-branch https://github.com/NuyoahQQ/ME5412-FoGGuard.git ME5412-FoGGuard-automation2
cd ME5412-FoGGuard-automation2
```

然后在该仓库根目录，Python 3.12：

```bash
python -m pip install -r requirements.txt -c docs/automation2/tested-constraints.txt
python -m pip install -e .
python -X utf8 scripts/run_automation2_delivery_checks.py
```

约束文件只新增本次实测依赖版本，不改团队原 requirements。该检查命令运行累计测试、Ruff、初版策略评估和 22×5 完整演示。输出到新的 `reports/automation2-delivery/<运行时间>/`，不覆盖已有实验；模型与原始数据未提供时，真实数据复现明确显示 `NOT_RUN`。

单独查看完整演示：

```bash
python -X utf8 scripts/run_complete_demo.py --predictions docs/examples/robot2_predictions.csv --output-dir reports/complete-review
```

打开 `reports/complete-review/index.html`。已有目录可能被该底层脚本覆盖，保留旧结果时换新目录。

若要复核真实数据，最省时的方式是使用识别附件内 `inputs/` 的保存 OOF 和训练报告，另准备同版原始数据：

```bash
python scripts/download_fogstar.py
python -X utf8 recognition/evaluate.py --raw data/raw/fogstar/sensor_data.csv --predictions path/to/inputs/fogstar_oof_predictions.csv --training-report path/to/inputs/fogstar_training.json --output reports/recognition-new-run
```

也可沿用原 `scripts/train_fogstar_baseline.py` 重新训练并生成 OOF，需额外时间和内存。不能用全量训练模型重新预测训练集来替代 OOF。真实数据评估没有证明患者疗效，当前参数也曾用于工程探索。

## 队友审查重点与剩余工作

- 第一位：确认 11 段缺失输入的覆盖边界、站立误报与参与者差异；如设计备用模型，需要重新做参与者隔离评估。
- 第二位：审查可选控制器接线、事件边界和输入缺失状态。共享在线契约仍待 #16 对齐。
- 第三位：核对踏步扩展与原倒计时的边界。默认原控制器没有被替换；模拟响应不能充当疗效证据。
- 第五位：可直接消费保存的时间线、曲线与 CSV，完善最终界面、人物展示和汇报材料。
- 第四位：已提供可重复评估与失败证据；主动停步误报等问题仍保留为 `KNOWN_LIMITATION`，不会为了验收全绿而隐藏。

未删除原文件。已有代码的兼容改动仅为 `replay.py` 的可选控制器工厂、`pyproject.toml` 的新模板包数据；README 增加入口、`.gitignore` 增加新运行目录。共享类型、原模型、原控制器与原测试保留。此贡献不自动合并或关闭 #5/#8。

## 让 AI 接手

将下面一段交给 AI 即可：

> 请先读 AGENTS.md、AUTOMATION2_START_HERE.md、docs/automation2/AI_HANDOFF.md 和 status.json，再检查当前 Git 分支、未提交改动及 PR #23/#24 状态。区分原模型、可选踏步仿真、真实 OOF 识别和全量事件覆盖。保留队友的内容，先说明目标、证据来源和已知问题，再按本次任务修改并运行一键检查。后续 GitHub 更新按 CONTINUING_AND_UPLOAD.md 追加提交，不重写历史证据。

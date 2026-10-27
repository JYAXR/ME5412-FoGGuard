# 机器人 2 已交付：下游开工入口

更新日期：2026-10-21。状态：机器人 2 的 FoG-STAR 情境与事件集成已推送，可供下游分支开发；合并状态以 GitHub PR 为准。

## 取用版本与范围

接入分支：`feat/robot2-fogstar-integration`。功能提交：`96c7b31`；步态提示范围调整：`39d42ea`。本文件随后的文档提交提供最新开工安排。

当前重点是 FoG 步态提示和恢复引导。暂缓主观停步意图区分，`suppress_stationary_fog=False` 为默认设置。静止仍记录，但不单独否决事件或提示；停步误报仍报告，不再要求该场景必须零提示。

以下能力已经交付，无需下游重新实现：

- FoG-STAR 模型适配、`other → unknown`、低置信方向拒识与两窗方向确认。
- 两窗 FoG 确认、概率迟滞、三窗恢复观察、同事件合并和受试者/会话/任务边界重置。
- 时间缺口和输入新鲜度检查、真实提示历史及不应期传递。
- CLI、Streamlit、评估共用的回放入口和机器可读时间线。
- 受试者隔离的五折 OOF 预测导出与事件级基线评估。

注意：事件层 `recovery` 只表示检测输出回落，未确认患者实际迈步。现有 `cue` 仍是声音/触觉的工程基线，脚印＋声音组合、备用动作和渐弱尚待控制与展示负责人实现。

## 先跑最小样例：无需下载数据或训练模型

在自己的干净工作目录获取此分支，再从它创建自己的功能分支；不要直接修改公共接入分支或 `main`：

```bash
git fetch origin
git switch -c feat/my-cueing-work origin/feat/robot2-fogstar-integration
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/run_fogstar_end_to_end.py --predictions docs/examples/robot2_predictions.csv --output reports/robot2_handoff_demo.csv
```

示例是人工构造的模型候选，用于验证接口和控制时序；它不是患者数据，也不用于性能指标。输出应包含 `normal/suspected/confirmed/recovery/safe_none`、直走/左右转/未知、提示及原因。末行人为延迟 2 秒，应无提示并记录 `stale_input`。

此样例已实际验证：20 个窗口、2 个确认事件、3 次基线提示、1 个过期输入。3.483333 秒的静止 FoG 候选可确认并产生声音提示，4.483333 秒重新确认仍属于同一事件；6.983333 秒和 9.983333 秒分别产生左右转的基线触觉提示。

各角色使用不同的 `feat/...` 分支名。等机器人 2 PR 合并后，后续新分支改从最新 `origin/main` 创建；已有分支按团队流程同步。

## 给自动化 1：提示控制

第一步读取 `reports/robot2_handoff_demo.csv`，从 `stable_fog_detected`、`event_phase`、`event_id`、`turn`、`turn_confidence` 和 `input_fresh` 接入自己的策略。原始 `is_fog` 或 `state` 不能绕过持续确认直接启动提示。

先实现确认事件中的固定脚印＋声音，以及恢复观察后的退出，再接入明确标注的模拟落脚响应。没有模拟响应时，使用固定策略，不假定患者已经成功迈步。模拟响应可运行 `python scripts/generate_simulated_responses.py` 生成，格式见 `src/fogguard/data/responses.py`。

控制接口可以通过 `integrate_fogstar_predictions(frame, controller=your_controller)` 注入实现现有 `decide(ControllerContext) -> CueCommand` 协议的控制器。当前 `ControllerContext` 尚无事件阶段字段，`CueCommand` 只能表示单一提示类型，无法完整表达脚印位置与双模态同步。因此本轮先消费统一 CSV 做控制原型，再在 Issue #16 提出控制扩展契约，由机器人 2 配合接线；不能把新控制字段写成已冻结接口。

交付：控制实现、配置、按时间戳对齐的控制输出、进入/维持/退出原因，以及成功/无响应/缺失输入的用例。任务入口：[#4](https://github.com/JYAXR/ME5412-FoGGuard/issues/4)、[#17](https://github.com/JYAXR/ME5412-FoGGuard/issues/17)。

## 给自动化 2：评估与回归

先确认最小样例中单窗候选不提示、短暂回落仍是同一事件、方向未知及过期输入不提示。静止不应单独抑制已确认 FoG；旧静止抑制仅作为可选对照。

全量真实数据评估使用以下命令，避免用最终模型对训练数据的预测评估泛化能力：

```bash
python scripts/download_fogstar.py
python scripts/train_fogstar_baseline.py
python scripts/evaluate_fogstar_integration.py
```

生成 `reports/fogstar_oof_predictions.csv`、`reports/fogstar_robot2_end_to_end.csv` 和 `reports/fogstar_robot2_integration.json`。原始数据、模型和报告在 Git 忽略目录，各人本地生成；最小样例不依赖这些文件。

当前基线：9,636 窗口，81 个真值窗口事件，69 个命中，119 个误报事件，1.440 次/分钟；中位确认延迟 0.50 秒；201 次基线提示，其中 69 次位于非 FoG 窗口。延迟从首个真值阳性窗口的输出时刻算起，不能当作样本级发病到提示延迟；方向 macro-F1 0.957 仅针对已接受方向，其覆盖率为 0.606。默认规则的参数曾在这批数据上探索过，结果属于工程基线，不是独立最终测试。

交付：新控制策略与固定基线在相同输入上的提示次数、首次提示延迟、误提示、退出与恢复结果、失败清单和复现命令。任务入口：[#5](https://github.com/JYAXR/ME5412-FoGGuard/issues/5)、[#8](https://github.com/JYAXR/ME5412-FoGGuard/issues/8)。

## 给工业设计：显示与视频

立即读取最小样例生成的时间线制作 HUD：`fog_probability`、`turn/turn_confidence`、`event_phase`、`cue/cue_reason`。以 `window_end_s` 作为输出可用时刻，保持受试者/会话/任务隔离。无 `truth_*` 时也应正常显示；有真值时另加评估图层，不用真值驱动动画或控制。

现阶段可画示意脚印和声音状态，明确采用固定演示参数；自动化 1 交付控制输出后，用其位置、侧别和声音参数替换占位呈现，展示层不另写模型和提示规则。

交付：同一时间线驱动的界面，正常、左右转、FoG、恢复与异常片段，以及离线录屏。任务入口：[#6](https://github.com/JYAXR/ME5412-FoGGuard/issues/6)。

## 给机器人 2：接下来的联调

维护本接入契约及兼容性；审查自动化 1 的控制扩展提案，与工业设计对齐时间轴，把模拟响应接入控制层，并由自动化 2 复核。当前不承诺脚印＋声音、稳定步数、备用动作或渐弱已经完成。

已验证代码基线：`pytest` 77 项通过，Ruff 通过。下游可以开始分支开发；进入 `main` 仍需组员评审。详细字段及历史实验见 [ROBOT2_HANDOFF.md](ROBOT2_HANDOFF.md)，总体分工见 [ROLES_AND_ROADMAP.md](ROLES_AND_ROADMAP.md)。

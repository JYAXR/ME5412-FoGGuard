# 自动化 2：初版评估与交接

> 本文描述初版评估层。当前累计交付范围、踏步扩展、真实数据 v2 与上传入口以 [从这里开始](../AUTOMATION2_START_HERE.md) 和 [status.json](automation2/status.json) 为准。

本交付基于 PR #24 的提交 `58135167a3be0aaa4c93e9c138d732efda9dd38a`，
该提交已包含 PR #23 的事件集成。本交付只新增文件，不修改原有算法、参数、共享类型、
测试、README 或 UI；没有替组员合并 PR，也没有将 `guidance-proposal-v1` 宣布为正式接口。

## Windows 与 Ubuntu

两者均可；已在 Windows / Python 3.12 实际运行。无需 ROS、WSL、Ubuntu 或真实硬件。
在仓库根目录建立独立环境：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe scripts/run_automation2_checks.py
```

Ubuntu 将 `.venv\Scripts\python.exe` 替换为 `.venv/bin/python`。依赖安装完成后，
默认评估不联网，不需要模型文件。不要把原始数据和本地模型加入 Git。

检查入口在子进程中启用 UTF-8，并依次运行全部 pytest、Ruff 和评估，保存日志与退出码。
它还生成离线审阅页；为避免数学库在线程较多的电脑上启动时内存不足，子进程的
OpenBLAS/OMP/MKL 线程数限制为 1。这只改变本次检查的资源使用，不修改算法或系统设置。
原有 HTML 测试使用未指定编码的 `read_text()`，中文 Windows 默认 GBK 会解码失败；
此处通过运行环境解决，未修改该测试或 HTML。单独运行测试时使用
`python -X utf8 -m pytest -q`。首次未启用 UTF-8 的失败日志也保留在交付证据中。

## 一条命令评估什么

`python scripts/run_automation2_evaluation.py --output-dir reports/automation2`

- 七个已提交的 IMU 场景，分别运行默认规则与可选静止抑制对照。
- 用现有 `replay_csv` 生成 FreezeIndex/方向候选，经明确标记的适配器传给现有
  `integrate_fogstar_predictions`，然后调用现有 `replay_guidance`。
  这是合成数据上的 FreezeIndex 链路，不能称为 FoG-STAR 模型识别结果。
- 原有 21 个构造候选/模拟响应场景，检查状态转换与策略差异。
- 机器人 2 的 20 窗口交接样例，保留为接口检查；不凭人工候选报告模型准确率。
- 独立审计 start/hold/update/stop、双模态开关、episode 时钟、目标数值与结束撤销。
- 若发现生命周期异常，输出失败明细并以非零退出码结束；不自动修改控制策略。
- 新增独立功能验收门槛：负例不提示、核心事件启停、左右转目标、安全撤销、渐弱、
  再冻结恢复强度、一次备用与时长上限、响应源缺失降级、可选间距边界。
  任一必要证据缺失或规则失败均以非零退出码结束。
- 主动停步默认误提示单列 `KNOWN_LIMITATION`，不伪装为 PASS。
  工程回归通过不意味着已解决模型识别问题或完成全组验收。

## 真实数据复现

沿用原有下载/训练脚本，不调整阈值或超参数：

```bash
python scripts/download_fogstar.py
python scripts/train_fogstar_baseline.py
python scripts/run_automation2_evaluation.py --oof reports/fogstar_oof_predictions.csv --training-report reports/fogstar_multitask_baseline.json
python scripts/download_daphnet.py
python scripts/evaluate_daphnet.py
```

完整评估需要下载公开数据并训练，耗时和内存取决于电脑。OOF 文件必须来自受试者隔离
预测，不能用最终模型对训练数据再预测。脚本检查配套报告中的训练/测试受试者不重叠，
并保存预测和报告哈希；这项结构检查本身不能证明外部文件从未被修改。本次交付实际
重新运行了原训练脚本，得到五折 OOF 预测。

FoG-STAR 没有真实提示响应，因此该部分不开启模拟自适应。固定与 response 策略
在相同输入下相同是预期行为，不是遗漏。Daphnet 只验证检测，缺少转向和提示响应，
不强接双模态控制。

## 对照与指标口径

| strategy | 与默认 response 的差异 |
|---|---|
| fixed | 关闭响应策略、渐弱和间距调整，是组合对照 |
| response | 加载原 `config/guidance.yaml`，间距调整默认关闭 |
| no_fade | 只关闭渐弱，其他配置相同 |
| spacing | 只打开间距调整，其他配置相同 |

每组使用相同输入和响应；有效配置写入各组 `metrics.json`。

- `raw_*` 为原始窗口识别 TP/TN/FP/FN，`stable_*` 为连续确认后的窗口结果。
- 无真阳性时召回率为空；没有提示不是“100% 正确”。
- 事件真值由连续阳性**输出窗口**构成；不与样本级起病时刻混用。
- 新增严格起点匹配：确认/提示启动必须落在真值开始至结束后 2 秒之间，
  按真值事件顺序一对一匹配；若宽限期重叠，先匹配较早事件。
  提前出现并延续至真值事件的误报不会自动变成命中。
- 同时保存原有重叠匹配指标 `real_upstream_overlap_metrics.json`，不覆盖原统计口径。
  两种定义可能得到不同的命中数，不能相互替换或混报。
- `start_commands` 按 start 计数，hold、节拍次数、旧 `cue` 脉冲不能算成新启动。
- `starts_on_non_fog_windows` 是启动所在窗口为负例；
  `unmatched_guidance_starts` 是不匹配真值事件的启动，两者定义不同。
- `active_non_fog_windows` 包括合理恢复观察尾段，不可全部称为错误提示。
- 逐事件记录另列 `guidance_start_matched`、`guidance_start_delay_s` 和
  `guidance_active_truth_windows`。确认起点未匹配、没有新启动、冻结期间完全没有活动提示
  是三种不同情况，不能都叫作漏检或漏提示。
- `terminal_stops` 是回放/会话结束截尾，不能算作成功恢复。
- `fade_completed_stops` 只说明给定模拟响应下控制器完成渐弱，不代表患者疗效。

## 产物

- `comparison.csv`：逐场景、逐策略对比表。
- `timelines/<case>/events.csv`：实际事件输入。
- 同目录下的策略 CSV、模拟响应 CSV、`metrics.json` 和识别错误窗口表。
- `real_subjects_fixed.csv` / `real_subjects_response.csv`：真实数据参与者级结果。
- `audit_failures.json`：独立命令审计失败清单。
- `manifest.json`：输入/代码哈希、基线提交、版本、配置和解释范围。
- `acceptance.csv/.json`：可重复执行的功能验收，区分 PASS、FAIL 和 KNOWN_LIMITATION。
- `review/index.html`：离线验收总览和失败案例入口。
- `review/control_playback.html`：复用原自动化 1 展示器，读取已保存结果，
  可切换固定/响应驱动/单因素对照；不重新判断模型或控制状态。
- `review/*event*.html` 等交互图与对应 CSV：真实漏检、负例窗口启动、主动停步和恢复边界证据。

单独重新生成审阅页：

```bash
python scripts/build_automation2_review.py --input-dir reports/automation2
```

真实数据未在本次评估运行时，不把旧目录中遗留的真实结果混入当前审阅。
当前运行的场景列表由 `manifest.json` 的 `case_names` 明确记录。

## 已知未完成与交接

- 初版原控制器的备用为一次倒计时。累计交付另新增可选踏步仿真扩展，见 [完整演示交接](COMPLETE_DEMO_HANDOFF.md)；不改变原控制器默认行为。
- 新共享类型、正式在线接线及 UI 集成仍需在 #16 由相关负责人确认。
  本评估入口已经能串起离线流程，但不取代全组接口冻结。
- 默认静止不否决已确认 FoG；主动停步误报是保留的失败案例。
  可选抑制对照证明行为差异，不构成默认策略已经更改。
- 最终人物动画、界面设计和录屏交给工业设计，读取现有控制时间线。
- 真实数据的控制改善和临床疗效未被验证；当前参数曾用于工程探索，
  即使受试者隔离，仍不能称为全新独立最终测试。

# 后续修改与上传

## 代码继续加到同一分支

贡献仓库：`NuyoahQQ/ME5412-FoGGuard`。分支：`feat/automation2-review-and-recognition`。
原仓库：`JYAXR/ME5412-FoGGuard`。此次 PR 基于 #24 的 `codex/automation1-guidance`，
不是把尚未合并的 #23/#24 当作 main 的现状。

建议本地 remotes：origin 指向贡献 fork，upstream 指向原仓库。开始前检查实际状态：

```bash
git status --short --branch
git remote -v
git fetch origin
git fetch upstream
git log --oneline --decorate -8
```

保留现有未提交内容，确认所在分支和上次提交。不要使用 `git reset --hard`、
`git clean`、强推，或通过拷贝整个旧目录覆盖新分支。远端有新提交时先读取差异；
不要为使上传成功丢弃它们。

修改后从仓库根目录运行一键检查，用新运行目录保存结果：

```bash
python -X utf8 scripts/run_automation2_delivery_checks.py
git diff --check
git diff --stat
```

检查只包含当前任务的文件，再显式逐个暂存并追加提交，例如：

```bash
git add docs/automation2/status.json
git add path/to/the/changed/source.py
git diff --cached --name-status
git diff --cached
git commit -m "docs: update Automation 2 handoff and verification"
git push origin feat/automation2-review-and-recognition
```

示例路径需替换成实际改动；不要在数据工作流后直接 `git add .`。同一分支的新提交会更新
同一 PR，通常不需要再建一个重复 PR。不要自动合并、关闭原任务或改队友 PR。

## 新结果单独留版本

- `docs/automation2/evidence/2026-10-23/` 是此次历史快照；不改旧结果让它看起来对应新代码。
- 新实验用新日期/序号目录，记录代码提交、参数、输入哈希、指标定义和失败案例。
- 更新 `status.json` 的最新快照入口与解释范围。没有重新运行真实数据时明确标为未运行。
- 大型自包含 HTML、视频、PPT、OOF 输入和 ZIP 作为带版本的 Release assets。
  新版建新 tag/Release，例如 `automation2-handoff-2026-10-24`，不替换旧附件。
- 保持 `release-assets.json` 的文件名、大小、SHA256、版本链接一致，避免上传本机其他文件。
  不上传临床人口学表、个人备份、凭据、原始数据或部署模型。

## 上游依赖合并之后

先查看 #23/#24 的实际合并状态和代码差异，再决定是否将本 PR 的 base 改为 main。
只有确认 main 已包含所依赖的实现后才改目标分支；随后重新看 PR diff、运行检查并更新说明。
如需解决冲突，保留双方意图和变更证据，请相关模块负责人审查共享接口，不用旧版文件覆盖冲突。

## 给下一次 AI 的起始指令

> 在这个分支继续。先读 AGENTS.md、AUTOMATION2_START_HERE.md、docs/automation2/AI_HANDOFF.md、
> status.json 和本文件，检查未提交改动、origin/upstream 与现有 PR 状态。
> 本次目标是：〔填具体修改〕。保留队友内容，只追加必要修改，不强推、不自动合并。
> 运行相应检查，分别说明本次实测与沿用的历史结果；新证据另存快照，再追加提交到同一 PR。

# 工程演示与保存输出

工程演示用于检查事件、脚印声音、响应分支、踱步和退出。它采用构造候选与模拟响应，不能代表正式四模块的真实数据识别成绩。

```bash
python -X utf8 scripts/run_complete_demo.py --output-dir reports/project_demo_NEW
```

打开输出目录的 `index.html`，按场景查看提示过程。页面读取保存命令，声音由用户主动开启；暂停、切换和结束均应撤销声音。

如传入 `--predictions`，数据应先满足事件模块输入契约；A1 的 FoG-only 输出需要 A2 提供方向适配。`--responses` 使用当前加载器支持的 `source=simulated` 格式。IMU 代理接入由 B1 完成。

显示坐标为配置示意位置，原地交替不能计为向前位移。策略与反馈验证见 [B1](tasks/B1.md)、[B2](tasks/B2.md)，最终展示见 [演示方案](SIMULATION_VIDEO_PLAN.md)。

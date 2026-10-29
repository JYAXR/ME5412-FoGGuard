# A2 从这里接续

2026-10-29。日常开发只需本页、一个推理脚本和一个模型附件。研究源码保留供复现，暂不继续追加训练。原默认模型、共享类型和控制器保持原样。

## 保留的研究基线

**66维双踝gyro随机森林**，类为`BilateralFoG`，variant=`bilateral_rf`，TRAIN按人OOF阈值0.23。它相对匹配左踝输入通过开发门槛，不代表所有历史模型中最好。

| 范围 | Precision | Recall | FoG F1 | 原始事件任意重叠 |
|---|---:|---:|---:|---:|
| FoG-STAR开发验证，4人/2014窗 | 0.572 | 0.857 | 0.686 | 10/10 |
| Stanford首次迁移，7人/10033窗 | 0.640 | 0.507 | 0.566 | 131/211 |

验证误报153秒/约17.8分钟观测；Stanford冻结时间Recall只有48.7%，单位/采样已适配，轴系未逐试验统一。两来源都已查看，后续复查不能称新盲测。该双踝模型无新的四人TEST成绩。10/10不等于100%准确率。

## 三步运行

1. 取得当前贡献分支源码，在Python 3.12环境安装`requirements.txt`，再执行`python -m pip install -e .`。
2. 将同版本模型附件解压到`artifacts/a1-a2-handoff/`。模型不放进源码Git，附件带SHA256和测试依赖版本。
3. 在仓库根目录运行：

```powershell
python -X utf8 scripts/predict_a1_bilateral.py --model artifacts/a1-a2-handoff/model.joblib --input artifacts/a1-a2-handoff/synthetic_input.csv --output reports/a1-a2-smoke.csv
```

示例是构造的接口测试信号，含右踝断档；输出不用于证明效果。换成自己的传感器CSV即可推理。无完整窗口会报错，调用方保留未知。

## 输入、输出和融合

输入：timestamp秒，ankleL_gyro_x/y/z及ankleR_gyro_x/y/z，60Hz、deg/s、左右及安装轴与训练约定一致。多记录加subjectID/sessionID/taskID，不需fog、activity或其他真值。

输出：记录键、window_start_s/window_end_s、available_at_s、fog_probability、fog_threshold、is_fog及版本/来源。概率是模型分数，未校准为临床风险。

- 完整120点窗口结束才输出，步进30点。available_at_s不可回填起点，最多保持0.5秒，超时未知。
- 右踝缺失跳过原窗口，不补零、不插值、不重新开网格。
- A2按记录键、窗口和实际可用时刻对齐自己的方向结果，并独立评价融合。这里没有direction字段，不能直接当完整方向融合输入。
- 低FoG分数不证明迈步成功。转向意图、迈步反馈及原地踱步切换需要各自证据。

Python接口为`fogguard.modeling.a1_bilateral.BilateralFoG.load(path).predict_frame(frame)`。

## 维护与共享

日常代码：`scripts/predict_a1_bilateral.py`、`src/fogguard/modeling/a1_bilateral.py`及公共特征/窗口依赖。保留历史模型类，避免删除后已保存模型无法加载。

A1维护者可生成附件，A2无需重训：

```powershell
python -X utf8 scripts/export_a1_handoff.py --output reports/a1_handoff_NEW
```

附件仅含冻结模型、合成接口样例、依赖版本及说明，不含原始患者信号。同步GitHub使用贡献分支，模型附件通过Release或单独文件共享，不把权重/数据加入Git，不合并main。

[汇报摘要](a1/PROJECT_BRIEFING.md) · [实验索引](a1/EXPERIMENT_INDEX.md) · [方法及复现](a1/BILATERAL_HANDOFF_V1.md)

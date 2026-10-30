# A2 当前方向模型：可直接下载使用

本目录发布的是已经训练好的正式 A2 当前方向模型，不需要重新训练。
权重文件：[model.joblib](model.joblib)，大小 50,713,064 字节。

SHA-256：`a612c8d8eeed2ab76addecbdd67848d57a18e0f32bef3f8512ed72737a14f8bd`。
GitHub 文件页选择 **Download raw file** 即可下载；本文件通过普通 Git 保存，不需要 Git LFS。

## 文件

- `model.joblib`：12 维重力投影特征 + 300 棵树随机森林，当前方向识别。
- `training_config.json`：原训练配置；最终选择的特征和阈值另见 `MANIFEST.json`。
- `requirements.txt`：已验证的推理依赖版本，建议 Python 3.12（验证版本 3.12.13）。
- `synthetic_input.csv`：全为构造信号的接口样例，不含患者原始数据。
- `synthetic_expected.csv`：样例的预期接口输出，只用于检查加载和接线。
- `MANIFEST.json`：来源、模型参数、运行版本、文件哈希与验证范围。

代码哈希对应 `code_commit` 中的 Git blob 字节，不受 Windows 工作区换行转换影响。

模型使用自定义 Python 类 `fogguard.turning.learned.DirectionModel`。
加载时需要同时安装本仓库源码；不要只下载 joblib 后在缺少代码的环境中加载。

## 最小运行

先获取 `codex/a2-direction-preturn` 分支代码，在仓库根目录使用 Python 3.12：

```bash
python -m pip install -r models/a2/current/requirements.txt
python -m pip install -e .
python -X utf8 scripts/predict_a2_turning.py --model models/a2/current/model.joblib --input models/a2/current/synthetic_input.csv --output reports/a2_model_smoke_NEW.csv
```

直接在 Python 中加载：

```python
from pathlib import Path
import pandas as pd
from fogguard.turning.learned import DirectionModel

model = DirectionModel.load(Path("models/a2/current/model.joblib"))
signals = pd.read_csv("models/a2/current/synthetic_input.csv")
predictions = model.predict_frame(signals)
```

只加载可信来源的 joblib 文件。

## 真实输入和边界

输入是 60 Hz 的 `timestamp`（秒）、`subjectID/sessionID/taskID`，以及左踝 `ankleL`
和背部 `back` 的 `acc_x/y/z`（g）、`gyro_x/y/z`（deg/s）。
不需要 `activity` 或 `fog` 标签；两秒窗口结束后才可产生输出，每 0.5 秒更新。
缺测或不连续窗口输出 unknown，记录内错序时间拒绝运行。字段及示例见合成 CSV。
真实来源适配仍需按实际完成推理的时刻处理，并落实最多 0.5 秒的来源有效期。

开发验证主准确率 **66.99%**、三类 macro-F1 **71.97%**：14 人训练、4 人开发验证，
1130 个有效且方向明确的窗口，拒识计错；TEST 未评分。
完整四类评价、局限和图表见 [A2 结果报告](../../../docs/a2/RESULTS.md)。

本目录不是 A1 三冻结模型包，也不包含尚未达到可用水平的提前转向预测权重。

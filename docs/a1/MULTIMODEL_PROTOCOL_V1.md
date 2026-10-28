# A1 多模型比较 v1：预先固定的实验方法

本轮比较 RF、RBF-SVM、直方图梯度提升树、一个小型 1D-CNN。各模型只有一组固定超参数，不用外层测试结果选超参数或训练轮数。目的为检查模型家族是否比单纯增加特征更值得继续研究，不预先保证深度学习胜出。

## 共同输入与划分

使用现有 FoG-STAR v3 左踝、腰背 12 个通道；120 点窗口、30 点步进、窗口 FoG 占比至少 50% 为阳性。禁用缺失插值，不改变共享数据类。逐窗对齐前置基线的 9,636 个样本，所有 106 条原始记录仍纳入事件覆盖审计。

RF、SVM、梯度提升使用共同的 132 个原冻结特征；CNN 使用同一窗口同一通道的 0.5–20 Hz 完整窗口滤波波形，形状 N×12×120。比较中没有为 CNN 增加通道或额外时长。

五折外层参与者列表与前置基线完全相同。每个外层训练集内再做三折参与者验证，各模型内层人员划分相同。外层测试参与者不会进入训练、归一化、阈值选择或轮数选择。

## 模型固定配置

| 模型 | 配置 |
|---|---|
| RF | 300 棵树、叶最小 2、balanced_subsample，seed=5412 |
| SVM | StandardScaler + RBF-SVC，C=1、gamma=scale、balanced；只读 decision_function |
| HistGB | 200 次提升、lr=0.05、15 个叶、叶最小 20、L2=1、balanced，关闭内部随机样本早停 |
| CNN | Conv(12→16,k9)→ReLU→Pool2→Conv(16→32,k5)→ReLU→Pool2→Conv(32→32,k3)→ReLU→全局均值池化→Dropout0.3→Linear1，共 7,473 参数 |

CNN 从零训练，无预训练权重；固定 20 轮、batch128、Adam lr=0.001、weight_decay=0.0001。BCEWithLogitsLoss 的正类权重为该训练集负样本数/正样本数。使用 CPU、固定随机种子和确定性算法。没有内部随机行划分或早停。

SVM 标准化只在每次训练数据上拟合；CNN 每通道均值与标准差同样只在该训练数据上拟合。测试时仅读取这些已存统计，不重新拟合。

SVM 的 margin 经 sigmoid 映射到 0–1 以复用评分字段，保持排序，但不宣称这是概率校准。其他模型的分数也不作为校准后的临床概率。

## 本轮阈值目标

本轮统一在内层参与者留出分数上，从 0.05–0.95 的 91 个阈值中优先最大化 **FoG F1**，其次 balanced accuracy，最后较高阈值。

这与上一轮优先 balanced accuracy 的协议不同。为保证模型家族比较公平，本轮重新运行 RF。不能把本轮与上一轮之间的全部差异归因于分类器；本轮四种模型之间使用相同阈值目标。

全量拟合模型的部署阈值取五个外层训练阈值的中位数。评价仍使用各折自己的训练期阈值，不以部署阈值重算外层成绩。

## 结果与展示

记录窗口 P/R/F1/AUC/AP、混淆矩阵、逐人和活动分层、全部真实事件覆盖与漏检、误报警事件、时间覆盖、新报警条件延迟。PR 曲线来自外层 OOF；不同模型的“分数数值”不能直接解释为同一校准概率。

图包包括模型对照、OOF PR 曲线、CNN 网络结构、五个外层训练的损失曲线。损失曲线是训练损失，不冒充留出准确率。每个模型保留原独立核验器的信号与错误案例交互页。

所有方案均保存，不自动替换原默认模型。单次单配置实验不等于某模型家族的最优能力；CNN 只有一个种子和一个结构，失败也不能说明深度学习普遍无效。

## 运行与文件

安装原依赖、requirements-readiness.txt 和可选的 requirements-a1-deep.txt，CPU 版 torch 可按官方安装说明从 CPU 源安装。当前已实测环境会保存在输出目录 environment.txt。

```powershell
python -X utf8 scripts/run_a1_multimodel.py --baseline-dir reports/readiness_20261028_verified
```

输出为全新的 reports/a1_models_时间戳/。目录中的 rf/、svm/、hist_gb/、cnn/ 分别包含模型、OOF、独立真值、A2 运行时数据、训练报告、折审计和 recognition/。CNN 另有 network.json。根目录保存协议/源码哈希、comparison.json/csv、ppt_comparison.csv、PNG/SVG、HTML 总览与输出哈希。

独立推理使用 scripts/predict_a1_multimodel.py。模型输出 FoG-only，不补造方向、转向意图、步事件或提示响应；A2 接收 OOF CSV 时，方向仍是明确标注的旧基线方向候选。

## 来源与限制

- [PyTorch 官方 CPU 安装说明](https://pytorch.org/get-started/locally/)。
- [scikit-learn SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)：本轮不使用随机样本层面的概率校准。
- [scikit-learn HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)：本轮关闭默认可能启用的样本层面早停。
- [FoG 研究作者公开 CNN](https://github.com/Lu1g1n0/FoG-detection-using-a-single-ankle-mounted-sensor)：支持采用 CNN 作为比较方向。我们的 12 通道小网络不是该单踝三通道网络的复现，未加载其权重，也不引用其性能作为本项目结果。

本项目参与者/数据已在先前研究中被分析，并非新的独立外部验证。窗口重叠不能增加独立参与者数量。没有提示干预或迈步真值，模型识别结果不证明提示疗效。

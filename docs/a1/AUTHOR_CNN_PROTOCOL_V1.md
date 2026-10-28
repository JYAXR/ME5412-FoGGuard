# A1 作者 CNN 架构复现与误报诊断：预先固定协议

日期：2026-10-29。仅推进误报诊断及一个文献架构基线，不增加 GRU、增强、转向辅助头、阈值预算或默认模型替换。训练前提交本配置及协议，运行保存源码/输入哈希。

## 数据隔离与比较

继承 `config/a1_fixed_split_v1.json` 的 TRAIN14 / VALIDATION4 / TEST4。本轮只构造 TRAIN 和 VALIDATION 的窗口；TEST 已经看过历史结果，但本轮不再提取或评分。最终泛化结论仍需独立真实数据。本轮验证结果用于研发，不是盲测。

1. 当前六秒 RF 的误报诊断：TRAIN14 内三折 GroupKFold，按参与者完全隔离拟合，使用历史验证选定阈值 0.16，不用本折真值选阈值。保存每折 fit/held-out 人员、OOF 评分及动作/参与者/任务统计。诊断沿用既有验证阈值，是研发材料而非新独立测评。
2. 文献 CNN 输入：仅左踝三轴 gyro，60Hz、120 点/2 秒、30 点步进。逐窗口减均值，不按整个人或整条记录归一化；不滤波、不插值。不跨记录、时间断档和所需 gyro 缺测；加速度缺失不影响 gyro 模型。
3. 同输入 RF 对照：相同 gyro 窗口，33 个既有二秒统计特征，固定历史 RF 参数300树/min_leaf2/balanced_subsample。CNN 与此 RF 在相同窗口直接比较。旧 RF_context / RF_ankle 仅作不同输入支持的参考，另列匹配窗口结果，不能直接按全池 F1 排名。
4. TRAIN14 从头拟合；VALIDATION4 选择 CNN checkpoint 和阈值。CNN 固定最多120轮、batch64、AdamW lr0.008/decay0.0004、无类别加权 BCE、kernel L2 0.001。按验证总损失 early stop patience10/min_delta0.001 并恢复权重；阈值继承91点 F1/BA/较高阈值目标。仅一个种子5412，不按结果换种子、重试或扩大搜索。

## 参考与复现边界

参考 [作者仓库](https://github.com/Lu1g1n0/FoG-detection-using-a-single-ankle-mounted-sensor) 的 `model.json` 公开结构，独立用 PyTorch 实现：四层 causal Conv1D，通道16/16/24/24、kernel9/7/5/5；pool4 在第二层后，pool3 在第三层后；pool 后 dropout0.4；全局平均池化、Dense48 ReLU/dropout0.4、单输出 sigmoid。参数预计8353。kernel L2 只作用于四个卷积与 Dense48；偏置和最终输出层不加该正则。以 model.json 的 GlorotNormal 为准，最后层 GlorotUniform。

论文：Borzì et al., *Freezing of gait detection: The effect of sensor type, position, activities, datasets, and machine learning model*, Journal of Parkinson's Disease, 2025, DOI 10.1177/1877718X241302766。

不下载或载入作者 model.h5，避免同队列已有权重进入隔离实验。公开训练脚本随机拆重叠窗口、传三份输入、先载权重、声明但未使用 EarlyStopping；本轮不照搬这些行为。项目采用 >=50% 阳性，包括一半一半；作者 mode 在平票时取0。保留项目规则以便内部比较。PyTorch/TF优化器及初始化实现不能保证数值等价，AdamW衰减语义也有差异。因此本轮称**公开架构复现、训练流程适配**，不称论文分数严格复现。

## 产物与核验

新建独立 reports/a1_author_cnn_*，保留历史输出。保存训练OOF、角色/拟合记录、训练曲线、验证预测及阈值、同窗口与全部原始验证记录的事件/时间/未知支持、PNG/SVG与HTML入口。推理模型不消费真值，保存后重载比对。动作只用于诊断分组，不能以真值活动过滤运行输出。图中所有分数从窗口结束后才可用。

独立核验角色无交叉、OOF人员隔离、gyro原始窗口/真值、保存评分重算阈值/指标、记录和事件分母、源码/模型/预测哈希、推理真值独立。运行完整项目检查。本轮无TEST报告；不做临床疗效、提前预警、迈步成功或模型已达标的声明。

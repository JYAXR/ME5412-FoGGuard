# A1 外部验证准备：Stanford schema 核对 v1

2026-10-29。**找到具备双踝gyro的独立研究数据来源，并核对了真实工作簿表头、时间间隔和文件哈希；尚未运行外部模型成绩。**本页是适配准备，不是泛化验证PASS。没有读取目标标签值、调参或根据目标成绩挑模型。

## 为什么选择这个来源

Stanford作者仓库提供去标识化7人数据和处理代码；README说明原始XLSX按试验保存，gyro单位rad/s、时间秒、标签0/1。[作者仓库及格式说明](https://github.com/stanfordnmbl/imu-fog-detection/blob/main/README.md)。原研究报告双踝等IMU、128Hz采集，且包含多次到访；不能将不同visit当独立人员。[原论文](https://link.springer.com/article/10.1186/s12984-022-00992-x)。

作者仓库当前树`e95687842801ca3565463486a50d562fff44d182`中，`data/raw/imus6_subjects7/`含60个工作簿，`imus11_subjects4/`含20个；原论文说明11传感器来自7人中的4人子集。独立评价先只纳入前一目录，不能拼接两个目录并冒充更多独立人/试验。正式下载前还须锁定完整文件清单和逐文件哈希，而非只锁定动态main名称。

DAPHNET官方通道为踝/大腿/躯干加速度，没有当前模型所需gyro，不能复制/改名成左右gyro或补零后称外部测试。[UCI原始说明](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait)。它适合另行训练加速度模型的研究，不是当前66维gyro冻结模型的直接评价输入。

## 本轮实际检查

`scripts/inspect_stanford_schema.py`固定下载配置表及按目录排序的首条试验，先校验Git blob SHA1，再检查表头和B列前128个时间间隔。没有运行作者脚本、模型推理或读取freeze_label数据值；原始文件只保存在被Git忽略的目录。

| 文件 | 字节数 | 固定Git blob SHA1 |
|---|---:|---|
| `data/imus6_subjects7_configs.xlsx` | 10252 | `ef95834cd4ab7a54401936b76604b63659ead65e` |
| `data/raw/imus6_subjects7/pt1_visit_24_tbc_walklr_0_trial_2.xlsx` | 6106211 | `a3f5dfa9862c4ab35815305c4346c84accac9291` |

原始试验SHA256：`db5558e3a836acae441aa8d1d53dde7b9362851824e1fbff997e79de4273cd52`；配置表SHA256：`a7c1d09b15ccceb42fbb813b29ec3eb742d002a6b2169379e4f641768406b45d`。保存结果见本地`data/raw/stanford_schema_v1_20261029/schema.json`，修订后检查器再次本地重读验证一致，见`reports/a1_external_readiness_v1_20261029.json`。

样本包含`subject_ID`、`time`、`imu_ankle_l_gx/gy/gz`、`imu_ankle_r_gx/gy/gz`和`freeze_label`表头；前128个时间间隔中位数约0.007813秒，与论文128Hz及时间精度相符，**不是60Hz**。表头还含head/thigh/wrist字段，即使文件位于6-IMU目录，目录名也不能代替逐文件有效性检查。

## 下一轮固定适配事项

| 项目 | 当前模型 | Stanford | 评价前必须完成 |
|---|---|---|---|
| gyro单位 | deg/s | rad/s | 明确乘180/pi，保留来源与变换哈希 |
| 采样 | 规则60Hz | 128Hz来源 | 固定抗混叠128→60方案及时间对齐；离线重采样若使用未来样本，显式计入可用时刻/延迟，不冒充实时因果 |
| 通道名 | ankleL/ankleR_gyro_xyz | imu_ankle_l/r_gxyz | 明确左右对应，不能用foot代替ankle |
| 轴系与安装 | FoG-STAR原训练轴系 | APDM安装与朝向 | 根据原文/资料固定轴旋转/符号映射，不能按目标分数试朝向；未核实就标记未对齐域迁移 |
| 记录键 | subject/session/task | subject_ID及文件名中的visit/trial | 明确同人多visit、逐文件边界，禁止拼窗/重置成新受试者 |
| 缺失 | 任一侧当前窗口不完整→未知 | 尚未全量审计 | 每文件确认所需六轴、空值、时钟断档，不填零、不丢无支持事件 |
| 标签 | 原始0/1、窗口占比>=0.5 | README定义0/1 | 先固定原始事件边界与窗口投影，目标标签只进评价支路 |

主候选预先保留当前66维双踝模型，TRAIN阈值0.23，不使用目标数据再选择。冻结文件为`reports/a1_coordination_v1_20261029/bilateral_control/model.joblib`，SHA256 `44defb95104067b4e62070cb4a09db4a1be75e53192d7a0da55fa0a1ade76817`。78维协调模型未通过既定条件，不因未来外部成绩而回头改变主候选的身份；若作探索参考须另列。

完成适配、完整输入审计及评价协议后，才可一次性报告外部原始事件/时间/窗口指标、全部人员记录分母和未知支持。若需要目标队列校准/微调，须事先分出适配人员与封存人员，分别报告零样本迁移和适配后的成绩；不能训练后再把同人称测试。

当前状态：`SCHEMA_CHECKED_EXTERNAL_SCORES_NOT_RUN`。轴系映射、全量文件质量审计和采样/标签适配器仍待实现。原论文成绩不等于我们的模型成绩；仿真步态也不能替代这项验证。外部数据同样不保证包含转向主观意图、真实落脚成功或提示响应标签。

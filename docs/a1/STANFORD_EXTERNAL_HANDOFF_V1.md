# A1 Stanford 零样本外部迁移交接 v1

方法见 [评分前固定协议](STANFORD_EXTERNAL_PROTOCOL_V1.md)，真实结果见 [评价结果](STANFORD_EXTERNAL_RESULTS_V1.md)，状态见 [stanford_external_status.json](stanford_external_status.json)。本轮仅应用既有66维双踝FoG-STAR RF，阈值0.23；无外部训练、归一化拟合、校准、朝向试分或模型挑选。单位/采样已适配，研究间解剖轴系**未核实统一**，不得省略该限制。

## 文件与角色

- `config/stanford_files_v1.json`：作者固定commit、全部60条七人目录工作簿、大小和Git blob SHA1；不使用另一20条四人子集目录。
- `config/a1_stanford_external_v1.json`：固定主模型哈希、单位、轴序、因果滤波/延迟及评价规则。
- `src/fogguard/data/stanford_external.py`：按列读取XLSX、128Hz时钟规范化、原始断档、因果128→60Hz、评价标签投影。sensor模式不解码freeze_label数据值。
- `src/fogguard/evaluation/stanford_external.py`：原生128Hz事件和时间积分，按实际available_at起效、最多保持0.5秒；无预测时段未知。
- `reports/a1_stanford_external_v1r1_20261029/`：全部传感器输入、逐记录审计、INPUT_LOCK、独立评价及核验。`reports/a1_stanford_external_presentation_v1_20261029/index.html`含可读结果和PNG/SVG。最初`v1`准备因累计时钟取整检查停止，保留失败证据；未读标签/评分。

源数据、NPZ、模型、生成目录不入Git。文件名包含原始visit/trial，可追溯每条记录；输出subjectID=1000+原pt，sessionID为同人visit字符串排序序号，taskID为清单文件序号。visit不是新人，记录之间不得串窗；数字命名空间不等于证明匿名队列绝无人员重合。

## 复现

仓库根目录，Python3.12及原numpy/pandas/scipy/sklearn依赖，无需PyTorch或Excel读取附加包。首次下载约393MB，逐文件哈希锁定；可以断点重用已完整验证的工作簿。

```powershell
python -X utf8 scripts/download_stanford_v1.py --output-dir data/raw/stanford_external_v1_20261029
python -X utf8 scripts/prepare_stanford_external.py --run-dir reports/a1_stanford_external_reproduction
python -X utf8 scripts/evaluate_stanford_external.py --run-dir reports/a1_stanford_external_reproduction
python -X utf8 scripts/verify_stanford_external.py --run-dir reports/a1_stanford_external_reproduction
python -X utf8 scripts/report_stanford_external.py --run-dir reports/a1_stanford_external_reproduction --output-dir reports/a1_stanford_external_presentation_reproduction
python -X utf8 scripts/run_automation2_delivery_checks.py
```

后四步使用新目录；不覆盖评分来偷偷调参。模型来自前轮coordination v1的`bilateral_control/model.joblib`，SHA256须与配置相同。prepare只读传感器/时间/ID并写锁，之后独立评价进程才读目标标签。评价逐文件先sensor-only评分，再读取原始标签，不能把带标签窗口送入模型。

## 适配后的输入与时序

原字段`imu_ankle_l/r_gx/gy/gz`按明确左右对应映射到`ankleL/R_gyro_x/y/z`，rad/s乘180/pi。保留xyz和符号，不用foot通道替代，不按目标成绩旋转。NaN在因果滤波中传播，完整双踝窗口才预测；不填缺失、不删原始事件。

r1对每步原时间差按1/128秒检查2微秒量化容差，并从每段t0重建已知128Hz采样时钟；逐记录保存max_clock_reconstruction_s。原文件45将时间步长固定取整为0.007812秒，累计误差不能与单步精度混淆。该规则在任何外部标签解码/评分前修订并提交，原生事件时间同样基于重建时钟。

FIR为641抽头，up15/down32，完整启动后才保留输出，无未来尾补。NPZ中的logical_time为滤波信号所代表的时刻；预测实际可用时刻=逻辑窗口末端+1/6秒。原始128Hz真值用于事件与时间指标，弱窗口标签只是逻辑时间网格上的前一原生采样格投影，不能把窗口F1当“当前时刻/提前检测”性能。

`resample_gyro()`是离线批处理实现，滤波数学上因果，追加未来样本不改变既有输出；**尚未提供在线分块滤波状态机或实测硬件延迟**。在线接入需保存FIR历史及有理采样相位，不能把每个新块独立重启并引用本轮时序。A2若试接必须使用真实available_at，不把逻辑窗口结束当已可用。

运行时CSV删除truth_*，仅FoG-only记录键、逻辑窗口范围、延迟后的可用时刻、固定阈值/概率/判断等。原始任务多为转向诱发过程，不提供本模型的主观意图、足部位置、真实落脚成功、提示响应或临床疗效验证。

## 指标与复查

`evaluation/by_record.csv`保留60条记录，`by_person.csv`保留7人，`events.csv`保留全部原始事件。`summary.json`明确轴系未统一、冻结模型和原始时间/窗口指标；`runtime.csv`与truth-bearing表分离。无有效窗口的记录仍保留原始时间/事件，不将未知当正常。

事件任意报警重叠允许一条报警跨多个事件，不能当一对一事件F1或准确率。`nonfog_alarm_pieces`是把报警减去真值事件后的连续非FoG区间数，与历史“整段报警完全不碰FoG”的定义不同，不能直接跨报告比较。当前评价保留复采样延迟，未测算法运行耗时或端到端提示延迟。

展示固定选择清单首条试验与原始FoG时长最多的一条，不按最佳成绩挑案例。原项目默认、共享类型、A2/控制器保持不变；该贡献目前仅本地保存，未推送或合并。

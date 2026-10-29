// Editable evidence deck. Invoked inside an Artifact Session, or with a Presentation facade.
export function buildBriefing(p, applyChartFont) {
  p.slides.keep([]);
  const F='Microsoft YaHei', C={bg:'#F5F6F3',ink:'#132C36',muted:'#52666B',teal:'#337F84',rust:'#B87345',red:'#AC4E3E'};
  const sources=[];
  function txt(s,text,x,y,w,h,size=27,color=C.ink,bold=false){
    const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
    a.text=text;a.text.style={typeface:F,fontSize:size,color,bold,wrap:'word',insets:0};return a;
  }
  function slide(title,sub,notes,refs=[]){
    const s=p.slides.add({width:1280,height:720});s.background.fill=C.bg;
    txt(s,title,65,48,1150,65,43,C.ink,true);
    if(sub)txt(s,sub,67,124,1140,58,24,C.muted);
    txt(s,String(p.slides.items.length).padStart(2,'0'),1170,665,55,28,16,C.muted);
    s.speakerNotes.text=notes+'\n\n依据：\n'+refs.map(r=>'docs/a1/'+r).join('\n');
    sources.push({slide:p.slides.items.length,title,notes,refs});return s;
  }
  function table(s,values,x,y,width,height,widths,size=24){
    const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height,columnWidths:widths,values});
    for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
      const cell=t.getCell(r,c);cell.fill=r===0?C.ink:(r%2?'#FFFFFF':C.bg);
      cell.text.style={typeface:F,fontSize:size,color:r===0?'#FFFFFF':C.ink,bold:r===0};
    }
    return t;
  }
  function chart(s,categories,series,x,y,w,h,max=1){
    const ch=s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},categories,series,
      barOptions:{direction:'column',grouping:'clustered',gapWidth:90},hasLegend:series.length>1,
      legend:{position:'bottom',textStyle:{fontSize:21,fill:C.ink}},
      xAxis:{textStyle:{fontSize:21,fill:C.ink},line:{fill:C.muted,width:1}},
      yAxis:{min:0,max,numberFormatCode:'0.0',textStyle:{fontSize:18,fill:C.muted},majorGridlines:{fill:'#D9E1DF',width:1}},
      dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000',textStyle:{fontSize:20,fill:C.ink}},
      chartFill:'none',plotAreaFill:'none'});
    applyChartFont(ch,{fontFamily:F,availableFonts:[F]});return ch;
  }
  let s=slide('FoGGuard 冻结步态识别','A1 算法阶段总结与 A2 交接',
    '建议汇报8至10分钟，附录用于答疑。开场结论：已有局部、可复现的改善，但跨人和跨数据集的稳定识别仍未解决。本次收敛已有工作，停止为汇报继续堆训练。', ['A1_DELIVERY_SUMMARY.md']);
  txt(s,'已有明确改善\n泛化仍有明显不足',72,255,1100,190,61,C.ink,true);
  txt(s,'真实数据对照、失败分析与可复现接口',75,515,1100,50,29,C.teal);
  txt(s,'ME5412 课程项目    2026年10月29日',75,623,1050,35,23,C.muted);

  s=slide('数据与验证范围','按参与者隔离，重叠窗口不等于独立样本',
    'FoG-STAR为22人106条原始记录，329027个60Hz样本。固定TRAIN14、VALIDATION4、TEST4。TEST曾完成一次锁定评价，后续新双踝候选没有在该TEST上再评分。Stanford为另一来源7人60记录，128Hz原生数据。初次零样本评价后，后续诊断和不变特征实验已使用该来源，因此后续结果为研发复查。各行窗口池不同，不建立跨行模型排名。', ['FIXED_SPLIT_RESULTS_V1.md','BILATERAL_RESULTS_V1.md','STANFORD_EXTERNAL_RESULTS_V1.md']);
  table(s,[['数据范围','人员 / 记录','用途与边界'],['FoG-STAR TRAIN','14人 / 70条','拟合模型，训练内按人选阈值'],['FoG-STAR VALIDATION','4人 / 20条','反复用于开发，不能称新盲测'],['FoG-STAR TEST','4人 / 16条','既有一次锁定测试，新双踝未再评分'],['Stanford 外部来源','7人 / 60条','初次迁移评价后，已参与研发']],67,208,1145,340,[340,240,565],24);
  txt(s,'窗口：2秒输入，每0.5秒更新。标签：窗口至少一半采样点为冻结。',70,594,1130,65,26,C.teal);

  s=slide('当前保留的双踝研究基线','从传感器输入到可交接的 FoG 判定',
    '双踝gyro共6轴，单位deg/s，60Hz。左踝连续段锚定120点窗口，每30点推进。右踝任一轴缺失跳过当前窗口，不插值不补零。完成窗内四阶0.5至20Hz滤波，逐轴11项统计及频带特征，共66维。300棵随机森林，叶最小2点，balanced_subsample，seed5412。阈值0.23来自TRAIN按人OOF。概率不是经过校准的临床风险。决策从窗口末端起可用，最多保持0.5秒。', ['BILATERAL_RESULTS_V1.md','BILATERAL_HANDOFF_V1.md']);
  table(s,[['环节','实现'],['输入','左右踝6轴角速度，60Hz，deg/s'],['表征','完成的2秒窗口，66项统计与频带特征'],['训练','随机森林300棵，按人留出选择阈值0.23'],['输出','FoG分数、阈值、二元判定、记录键、可用时刻']],67,205,1145,345,[225,920],26);
  txt(s,'缺测保留未知。低 FoG 分数不能解释为已迈步成功。',70,596,1130,65,28,C.rust,true);

  s=slide('双踝输入带来了可复现的局部改善','同一2014个验证窗口、相同训练及阈值选择流程',
    '左33维与双66维都在相同6383个双踝完整TRAIN窗口拟合。验证相同2014窗口、470阳性、4人20记录10事件。F1绝对提升0.0625，误报时间181.95降到153.28秒，下降15.76%。四项开发门槛通过。但TRAIN双踝时间支持由自然左踝92.41%降到88.67%，增加传感器有缺测代价。历史左踝acc+gyro/上下文RF在2014窗F1为0.6912，双踝并非所有模型中最高。', ['BILATERAL_RESULTS_V1.md']);
  chart(s,['Precision','Recall','FoG F1'],[{name:'匹配左踝33维',values:[.5116,.7979,.6234],fill:C.rust},{name:'双踝66维',values:[.5716,.8574,.6860],fill:C.teal}],65,210,800,355,1.05);
  txt(s,'误报时长',905,240,300,40,25,C.muted);
  txt(s,'181.95s\n153.28s',904,298,305,145,44,C.teal,true);
  txt(s,'减少约16%',905,466,300,45,29,C.ink,true);
  txt(s,'开发集改善成立，尚未证明稳定泛化，也未替换原项目默认模型。',70,602,1125,60,25,C.muted);

  s=slide('后续尝试的主要取舍','每项都在各自固定对照内评价，完整实验表见附录',
    'RF、SVM、HistGB、小CNN的首轮固定配置比较没有显示其他家族胜过RF。六秒上下文与CNN-GRU出现局部改善，但分别有事件覆盖或误报时长代价。人员均衡F1增加但误报时间增14.9%。协调摘要与10维整体替换都未通过既定开发门槛。不同训练流程、窗口池和阈值规则的数字不能混排。', ['MULTIMODEL_RESULTS_V1.md','CONTEXT_RESULTS_V1.md','TEMPORAL_CNN_RESULTS_V1.md','CONTEXT_LEARNING_RESULTS_V1.md','COORDINATION_RESULTS_V1.md','INVARIANT_RESULTS_V1.md']);
  table(s,[['尝试方向','观察到的结果','当前判断'],['模型家族替换','RF在首轮固定配置中F1最高','未发现普遍更优家族'],['更长上下文 / GRU','部分F1或事件检出改善','同时存在漏检或误报代价'],['困难负样本 / 辅助目标','加权下降，活动辅助未改善F1','当前候选不采用'],['追加协调摘要','少误报，冻结时间Recall下降','未通过开发门槛'],['10维不变表征替换','两来源F1下降，误报时间减少','压缩方案不采用']],67,205,1145,370,[285,560,300],23);
  txt(s,'结论限于已经尝试的固定配置，不代表这些方法永远无效。',70,610,1130,50,25,C.muted);

  s=slide('当前双踝基线的实际检测效果','FoG-STAR验证集，2014个窗口，其中470个冻结窗口',
    '展示TP403、FN67、FP302、TN1242。Precision57.16%，Recall85.74%，F168.60%。Accuracy为81.68%，但全判非冻结也有76.66%，所以不能只报Accuracy。原始10个事件均有任意报警重叠，但时间Recall84.51%，仍有153.28秒误报。事件重叠不等于100%准确率，也不是一对一事件F1。', ['BILATERAL_RESULTS_V1.md']);
  table(s,[['真实 / 预测','预测冻结','预测非冻结'],['真实冻结','403','67'],['真实非冻结','302','1242']],70,230,660,250,[260,200,200],28);
  txt(s,'57.2%',805,226,340,70,57,C.teal,true);txt(s,'报警窗口中真实冻结的比例',805,300,365,55,25,C.muted);
  txt(s,'85.7%',805,383,340,70,57,C.teal,true);txt(s,'冻结窗口被检出的比例',805,457,365,50,25,C.muted);
  txt(s,'F1 0.686    原始事件重叠10/10    误报153秒 / 约17.8分钟观测',70,562,1135,45,28,C.ink,true);
  txt(s,'Accuracy为81.7%，单看该数会掩盖类别不平衡与错误报警。',70,620,1135,40,23,C.muted);

  s=slide('泛化表现仍明显不足','内部锁定测试与外部迁移属于不同模型和窗口范围',
    '内部TEST是先前锁定的六秒左踝加腰背RF及其左踝备用组合，不是最新双踝RF的测试。主模型1207窗F1.5113，组合1605窗F1.5307且事件20/23，Precision只有.3850。外部Stanford是冻结双踝66维RF，10033窗，P.6401 R.5068 F1.5657，事件131/211，时间Recall.4874，误报409.81秒/5140.91秒。单位与因果采样已适配，解剖轴系未逐试验统一。两行不能当同条件排行榜。', ['FIXED_SPLIT_RESULTS_V1.md','STANFORD_EXTERNAL_RESULTS_V1.md']);
  table(s,[['评价范围','模型及样本范围','P / R / F1'],['内部TEST，4人','既有六秒RF加左踝备用\n1605窗，16记录，23事件','0.385 / 0.854 / 0.531'],['Stanford首次迁移','冻结双踝66维RF\n10033窗，60记录，211事件','0.640 / 0.507 / 0.566']],67,210,1145,265,[270,525,350],25);
  txt(s,'Stanford：131/211事件有重叠，实际冻结时间只覆盖48.7%。',70,525,1135,60,29,C.rust,true);
  txt(s,'双踝模型无新内部TEST。Stanford仅适配单位与采样，轴系未逐试验统一。\nStanford后续复查已属于研发。',70,600,1135,62,23,C.muted);

  s=slide('外部参与者之间的差异很大','同一冻结双踝模型，同一阈值0.23',
    'P1至P7窗口F1分别.5605 .6091 .5371 .7764 .3712 .0632 .2606。P6只触及1/19个事件，P7为12/48。P5贡献408/758误报窗口。逐人F1等权均值.4540，低于汇总.5657。不能挑P4当代表，也不能用P6的高Precision掩盖极低Recall。', ['STANFORD_EXTERNAL_RESULTS_V1.md']);
  chart(s,['P1','P2','P3','P4','P5','P6','P7'],[{name:'外部窗口F1',values:[.5605,.6091,.5371,.7764,.3712,.0632,.2606],fill:C.teal}],65,205,1145,335,1.0);
  txt(s,'P6：1/19事件有重叠     P7：12/48     逐人F1均值：0.454',70,565,1140,50,29,C.rust,true);
  txt(s,'总体平均分不能保证每位参与者都能获得可靠检测。',70,626,1135,40,24,C.muted);

  s=slide('失败诊断缩小了问题范围','仍未找到可以单独解释全部错误的原因',
    '80个完全漏报事件中78个已有自身占至少60/120点的窗口及时可用，不能只解释为事件太短。纯FoG时P6中位score.0674，P7.0982，低于.23。名义轴映射仅是图示假设，不是逐试验校准。旋转不变摘要仍有分布差异，但不证明分类错误因果。10维替换实验源验证F1.6234，外部研发F1.5154，说明压缩方案当前无效，不能据此否定所有不变表征。', ['EXTERNAL_FAILURE_DIAGNOSTIC_RESULTS_V1.md','INVARIANT_RESULTS_V1.md']);
  txt(s,'78 / 80',75,220,515,110,76,C.rust,true);
  txt(s,'完全漏检事件已有及时的\n事件自身多数窗口',80,355,510,100,31,C.ink,true);
  txt(s,'仅靠缩窗或转轴\n缺乏充分解释',735,231,470,110,38,C.ink,true);
  txt(s,'10维压缩替换后的F1\n验证 0.623\n外部研发复查 0.515',740,390,440,145,30,C.muted);
  txt(s,'这是事后诊断。及时窗口由真值构造，不能当可部署模型或性能上限。',70,606,1135,55,24,C.muted);

  s=slide('A1收敛交接，A2可继续独立开发','冻结当前基线与证据，项目展示以真实回放为主',
    '已完成FoG检测基线、多种对照、真实输入与事件审计、概率回放和模块级推理。当前推荐A2使用保留的双踝研究基线作为可选输入并保留原系统默认，不等待全面更优模型。按记录键、窗口范围、可用时刻对齐方向输出。2秒窗末端输出最多.5秒新鲜度。A2的转向识别不等于主观转向意图已验证，迈步成功与冻结后原地踱步策略还需独立数据/仿真边界，不能写成A1已完成。当前贡献是可复现对照及失败边界，没有SOTA或临床创新结论。', ['BILATERAL_HANDOFF_V1.md','A1_DELIVERY_SUMMARY.md']);
  table(s,[['交接部分','当前状态'],['FoG研究基线与推理','保留66维双踝RF，提供模型包与运行入口'],['评估与展示','保留实验汇总、失败证据和同步真实数据回放'],['A2方向及融合','按记录键、窗口及可用时刻对齐，独立评价'],['迈步成功与原地踱步','仍需反馈定义与证据，仿真响应单独标注']],67,208,1145,330,[320,825],25);
  txt(s,'当前可以汇报可复现的研究原型与边界，尚不能宣称可靠临床识别。',70,588,1135,65,29,C.teal,true);

  s=slide('附录：早期模型与训练目标实验','旧22人队列上的探索性按人留出结果',
    '各行只能比较本行固定对照，不能把不同阈值目标混成时间趋势。首轮近期趋势按balanced accuracy优先选择阈值，后续多模型按F1优先。旧队列反复查看，所以这些不是多次独立泛化验证。完整来源逐行对应项目实验索引。', ['RESULTS_V1.md','MULTIMODEL_RESULTS_V1.md','HARD_NEGATIVE_RESULTS_V1.md','CONTEXT_RESULTS_V1.md','EVENT_RESULTS_V1.md','IMPROVEMENT_RESULTS_V1.md']);
  table(s,[['尝试','本行F1对照','主要结论'],['近期 / 趋势特征','0.5817 / 0.5879','小幅改善，事件覆盖79降至78/101'],['RF / SVM / HistGB / CNN','0.5994 / 0.5533 / 0.5812 / 0.5479','本轮固定配置RF较高'],['困难纯负样本加权','0.5994 / 0.5792','窗口指标下降，误报增多'],['六秒历史摘要','0.5994 / 0.6188','误报337降至235段，事件77降至70'],['事件目标与确认','0.6188 / 0.6061','多触及1个事件，误报时间增加14.5%'],['训练内模型选择 / 备用','主模型0.6188 / 0.6317','联合触及75/101，仍为旧队列探索']],65,208,1150,395,[295,430,425],21);
  txt(s,'备用组合扩展输入支持，窗口集合会变化，联合事件结果单独评价。',70,626,1130,42,23,C.muted);

  s=slide('附录：固定划分后的研发实验','均保留失败结果，不能仅挑F1或事件覆盖最高者',
    '作者CNN复现不是原创架构，也没有严格复现论文数值。GRU与人员均衡相较各自重新训练的对照，不能与不同流程历史模型直接作单因素比较。双踝相较匹配左踝通过本轮开发门槛。协调与10维不变替换均未通过。最新双踝等候选没有再次评分原四人TEST。', ['FIXED_SPLIT_RESULTS_V1.md','AUTHOR_CNN_RESULTS_V1.md','TEMPORAL_CNN_RESULTS_V1.md','CONTEXT_LEARNING_RESULTS_V1.md','BILATERAL_RESULTS_V1.md','COORDINATION_RESULTS_V1.md','INVARIANT_RESULTS_V1.md']);
  table(s,[['尝试','本行F1与评价范围','结论或代价'],['固定14/4/4重训八模型','主模型测试0.5113','备用测试0.5307，误报仍多'],['公开作者CNN架构','gyro RF 0.6388 / CNN 0.6772','少误报，事件10降至8/10'],['同流程CNN加GRU','0.5471 / 0.6571','误报时间增加17.7%'],['人员均衡 / 活动辅助','对照0.6476 / 0.6717 / 0.6438','均衡误报增，辅助未改善F1'],['匹配左踝 / 双踝','0.6234 / 0.6860','本轮开发门槛通过，有缺测代价'],['66维追加12项协调摘要','0.6860 / 0.6801','少误报，冻结时间漏检增加'],['66维替换为10维不变摘要','0.6860 / 0.6234','外部研发复查同样下降']],65,205,1150,425,[330,420,400],21);
  return sources;
}

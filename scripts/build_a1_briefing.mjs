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
  let s=slide('FoGGuard 冻结步态识别','三种方法的实现、效果与 A2 交接',
    '本次汇报只保留三种具有不同输入需求和误报漏检取舍的方案。最高F1为0.6912，仍有明显误报和泛化问题。方法选择基于已经查看的开发结果。', ['SELECTED_METHODS.md']);
  txt(s,'单踝历史 RF\n双踝 RF\n单踝 CNN',74,245,1100,250,55,C.ink,true);
  txt(s,'相同2014个验证窗口，比较可复现的已训练方案',75,548,1120,48,28,C.teal);
  txt(s,'ME5412    2026年10月29日',75,625,1050,35,23,C.muted);

  s=slide('数据与评价范围','FoG-STAR，按参与者划分，冻结 / 非冻结二分类',
    '原数据22人106条记录。TRAIN14人，VALIDATION4人，TEST4人。此次比较只复查4位验证者的20条完整原始记录，三模型使用同一2014个窗口，其中470个阳性，10个原始事件。2秒窗每0.5秒更新，窗口至少半数采样点为冻结才为阳性。共同判定支持93.72%，缺失时段保留未知。原TEST此前曾用于旧方案评价，不能称从未接触的新测试。本次没有构造TEST窗口或重新训练。', ['SELECTED_METHODS.md']);
  table(s,[['范围','数量','本次用途'],['训练参与者','14人','已有模型拟合'],['开发验证参与者','4人 / 20条记录','同一2014窗，其中470窗冻结'],['原测试参与者','4人','此次三个候选未重新评分'],['检测输入与更新','2秒 / 0.5秒','完成窗口末端给出结果']],67,205,1145,330,[320,270,555],25);
  txt(s,'验证集已用于研发和方案选择，不能将本页结果称为新盲测。',70,586,1135,75,28,C.rust,true);

  s=slide('方法1：左踝历史随机森林','一个左踝 IMU，结合当前波形与过去4秒信息',
    '左踝acc xyz单位g，gyro xyz单位deg/s，60Hz。120点完成窗内0.5至20Hz四阶滤波，每轴11项特征共66维。加入前2秒、前4秒的RMS、步态及冻结频带能量、差值和历史可用标志共140维，断档和换记录重置。RF300棵，叶最小2，balanced_subsample，seed5412。TRAIN14拟合，VALIDATION4选择F1阈值0.18。F1最高仅限这三方案和此开发窗口，误报也最多。', ['SELECTED_METHODS.md']);
  table(s,[['步骤','实现'],['传感器','左踝加速度xyz和角速度xyz，60Hz'],['特征','当前66维，加入历史后共140维'],['训练与阈值','随机森林300棵，验证集阈值0.18'],['效果','F1 0.691，冻结窗口Recall 89.8%']],67,205,1145,330,[250,895],27);
  txt(s,'适合优先减少漏检的研究对照，误报时间166.9秒。',70,592,1135,60,28,C.teal,true);

  s=slide('方法2：双踝角速度随机森林','左右踝互补表征，输入只需角速度',
    '左右gyro共6轴，单位deg/s，60Hz。同样2秒窗和0.5至20Hz滤波，每轴11项特征共66维，不使用更早历史。左踝连续段固定切窗网格，右踝缺失跳过原窗口，不补零不重新对齐。RF300棵，TRAIN内部按参与者三折OOF选择阈值0.23，再拟合全部TRAIN14。F1与方法1接近，输入和阈值流程不同。它可用于A2先接通接口。', ['SELECTED_METHODS.md']);
  table(s,[['步骤','实现'],['传感器','左右踝角速度xyz，60Hz'],['特征','两秒窗口，6轴合计66维'],['训练与阈值','RF300棵，TRAIN按人OOF阈值0.23'],['效果','F1 0.686，冻结窗口Recall 85.7%']],67,205,1145,330,[250,895],27);
  txt(s,'可先接入A2，误报153.3秒。双踝缺失时保留未知。',70,592,1135,60,28,C.teal,true);

  s=slide('方法3：左踝角速度 CNN','直接学习120点波形，共8353个参数',
    '只用左踝gyro xyz，每窗去均值。四层causal卷积通道16/16/24/24，核9/7/5/5，池化4和3，dropout0.4，全局平均和Dense48，8353参数。AdamW、BCE从零训练，最多120轮，验证早停实际17轮恢复第7轮，阈值0.50。结构参考公开作者实现，不声称原创CNN，也未严格复现论文分数。causal卷积不等于冻结前预警。来源 https://github.com/Lu1g1n0/FoG-detection-using-a-single-ankle-mounted-sensor ; DOI 10.1177/1877718X241302766', ['SELECTED_METHODS.md']);
  table(s,[['步骤','实现'],['传感器与预处理','左踝角速度xyz，两秒窗独立去均值'],['网络','4层一维卷积、池化、全局平均及Dense48'],['训练与阈值','AdamW，开发验证早停，阈值0.50'],['效果','F1 0.677，Precision 67.2%，Recall 68.3%']],67,205,1145,330,[285,860],26);
  txt(s,'误报时间降至82.2秒，代价是漏掉更多冻结。',70,580,1135,50,28,C.teal,true);
  txt(s,'公开作者CNN的架构适配，来源见本页讲稿备注。',70,638,1135,32,22,C.muted);

  s=slide('相同窗口上的指标对比','同一组2014窗与真值，输入和阈值选择流程各有不同',
    'Precision表示报警窗口中真冻结比例，Recall表示真冻结窗口的检出比例，F1综合两者。历史RF .561917/.897872/.691237，双踝RF .571631/.857447/.685957，CNN .671548/.682979/.677215。不能把这些小差异当统计显著或单纯算法家族贡献。AP分别.703601/.629415/.731436。以上是已用于开发和选择的验证结果，没有新盲测。', ['SELECTED_METHODS.md']);
  chart(s,['Precision','Recall','FoG F1'],[{name:'左踝历史 RF',values:[.561917,.897872,.691237],fill:C.teal},{name:'双踝 RF',values:[.571631,.857447,.685957],fill:C.rust},{name:'左踝 CNN',values:[.671548,.682979,.677215],fill:'#6371A1'}],65,205,1145,355,1.05);
  txt(s,'历史RF的F1最高，CNN报警更准，但冻结窗口检出更少。',70,594,1135,60,28,C.ink,true);

  s=slide('误报与冻结时间覆盖','同一约17.84分钟观测区间，判定时间支持93.72%',
    '每个窗口结果从末端起最多保持0.5秒，按实际时间积分去重。三方法误报166.883、153.283、82.233秒，冻结时间Recall88.461%、84.507%、66.800%，原始事件有报警重叠10/10、10/10、8/10。任意重叠即算触及，不能称100%准确。双踝RF的Stanford初次迁移F1为.5657，冻结时间Recall48.74%，当时适配单位采样但轴系未逐试验统一。没有另两种模型的外部结果，因此不能称任何一种已经证明稳定泛化。', ['SELECTED_METHODS.md']);
  table(s,[['指标','左踝历史 RF','双踝 RF','左踝 CNN'],['误报时间','166.9秒','153.3秒','82.2秒'],['冻结时间Recall','88.5%','84.5%','66.8%'],['原始事件有重叠','10 / 10','10 / 10','8 / 10']],67,205,1145,310,[320,275,275,275],26);
  txt(s,'事件有重叠只表示曾触及该次冻结，仍可能漏掉其大部分时长。',70,548,1135,65,25,C.muted);
  txt(s,'双踝RF既有外部F1为0.566，稳定泛化仍未解决。',70,618,1135,45,27,C.rust,true);

  s=slide('A1与A2的衔接','A1负责冻结检测，A2负责方向 / 转向识别及融合',
    'A2直接依赖A1的输出接口，但不需重训A1。按同记录键、窗口范围、实际可用时刻与方向输出对齐，各自最多保持0.5秒，过期未知，不能未来回填。A2先使用双踝RF跑通，硬件仅单踝时换历史RF或CNN。转向动作标签不自动证明提前意图。向前提示后未迈步再转原地踱步还需要独立的步伐响应反馈，低FoG分数不是迈步成功。共享types/interfaces和原控制默认不改。', ['SELECTED_METHODS.md','../A1_A2_START_HERE.md']);
  table(s,[['模块','职责与接口'],['A1冻结识别','分数、阈值、冻结判定、记录键及可用时刻'],['A2方向与转向','独立训练，与A1按实际可用时刻对齐'],['提示后的步伐反馈','判断脚是否迈出，再决定是否原地踱步'],['交接内容','三模型包、统一推理入口、训练与同窗评价源码']],67,205,1145,335,[310,835],25);
  txt(s,'当前交付支持模块联调。转向意图与步伐反馈仍需各自验证。',70,590,1135,65,28,C.teal,true);
  return sources;
}

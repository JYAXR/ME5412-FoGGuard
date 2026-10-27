// Video is rendered from saved command rows. No detection or control rules live here.
const recordButton=document.createElement('button');recordButton.textContent='导出完整演示视频';recordButton.id='recordVideo';recordButton.style.marginTop='15px';document.querySelector('aside').append(recordButton);
recordButton.onclick=async()=>{
  pause();recordButton.disabled=true;recordButton.textContent='正在导出，请保留此页面…';
  if(!window.MediaRecorder){recordButton.textContent='当前浏览器不支持视频导出';return}
  const canvas=document.createElement('canvas');canvas.width=1280;canvas.height=720;
  const ctx=canvas.getContext('2d');const stream=canvas.captureStream(15);
  const recordingAudio=new(window.AudioContext||window.webkitAudioContext)();recordingAudio.resume().catch(()=>{});
  const audioDestination=recordingAudio.createMediaStreamDestination();
  audioDestination.stream.getAudioTracks().forEach(track=>stream.addTrack(track));let videoBeat='';
  const chunks=[];const recorder=new MediaRecorder(stream,{mimeType:MediaRecorder.isTypeSupported('video/webm;codecs=vp9')?'video/webm;codecs=vp9':'video/webm',videoBitsPerSecond:2600000});
  recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};
  recorder.onstop=()=>{const blob=new Blob(chunks,{type:'video/webm'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='FoGGuard_完整仿真演示.webm';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),10000);stream.getTracks().forEach(track=>track.stop());recordingAudio.close();recordButton.disabled=false;recordButton.textContent='导出完整演示视频';};
  const sequence=[['normal',2,12,10],['left_fog',3,13,10],['right_fog',3,13,10],['march_recovery',3,10,7],['march_recovery',10,15,5],['refreeze_during_fade',5,14,9],['march_unknown_direction',4,11,7],['persistent_no_response',4,21,17],['voluntary_stop_observation',3,10,7]];
  const duration=sequence.reduce((sum,item)=>sum+item[3],0);const origin=performance.now();let interval;
  function text(s,x,y,size=22,color='#18334c'){ctx.fillStyle=color;ctx.font=`${size}px "Microsoft YaHei",sans-serif`;ctx.fillText(s,x,y)}
  function wrap(s,x,y,width){let line='';for(const char of s){if(ctx.measureText(line+char).width>width){ctx.fillText(line,x,y);line='';y+=31}line+=char}ctx.fillText(line,x,y)}
  function draw(){let elapsed=Math.min(duration,(performance.now()-origin)/1000),offset=elapsed,segment=sequence.at(-1);for(const item of sequence){segment=item;if(offset<item[3])break;offset-=item[3]}
    const selected=DATA.playback.find(x=>x.case===segment[0]&&x.strategy==='march');const t=segment[1]+Math.min(1,offset/segment[3])*(segment[2]-segment[1]);let i=0;while(i+1<selected.rows.length&&selected.rows[i+1].guidance_timestamp_s<=t)i++;const row=selected.rows[i];
    if(row.guidance_audio_enabled){const mode=row.guidance_audio_mode;const start=mode==='metronome'?row.guidance_cue_started_at_s:row.guidance_mode_started_at_s;const interval=mode==='countdown_3_2_1_go'?1:60/row.guidance_cadence_bpm;const beat=Math.floor((t-start)/interval);const key=sequence.indexOf(segment)+'/'+row.guidance_command_id+'/'+mode+'/'+start+'/'+beat;
      if(key!==videoBeat&&beat>=0&&(mode!=='countdown_3_2_1_go'||beat<=3)){videoBeat=key;const tone=recordingAudio.createOscillator(),gain=recordingAudio.createGain();tone.frequency.value=mode==='march_alternating'?(beat%2?660:440):880;gain.gain.setValueAtTime(row.guidance_audio_volume*0.12,recordingAudio.currentTime);gain.gain.exponentialRampToValueAtTime(0.0001,recordingAudio.currentTime+.06);tone.connect(gain).connect(audioDestination);tone.start();tone.stop(recordingAudio.currentTime+.07)}}
    ctx.fillStyle='#eef4f8';ctx.fillRect(0,0,1280,720);text('FoGGuard',45,55,35);text('冻结步态提示仿真',255,54,22);text('构造候选与预设模拟响应 · 工程行为演示',45,91,17,'#607587');text(labels[segment[0]]||segment[0],45,140,28);
    ctx.fillStyle='#ffffff';ctx.fillRect(40,168,730,500);ctx.strokeStyle='#dbe6ee';ctx.lineWidth=1;for(let x=60;x<750;x+=38){ctx.beginPath();ctx.moveTo(x,180);ctx.lineTo(x,655);ctx.stroke()}for(let y=185;y<660;y+=38){ctx.beginPath();ctx.moveTo(50,y);ctx.lineTo(760,y);ctx.stroke()}
    let anchor=[0,0];for(let j=1;j<=i;j++){if(selected.rows[j].guidance_reason==='simulated_forward_step_accepted'){const landed=targets(selected.rows[j-1])[0];if(landed)anchor=[landed.x_m,landed.y_m]}}
    const feet=targets(row);const range=Math.max(1.5,...feet.map(v=>Math.abs(v.x_m-anchor[0])+.4));const scale=Math.min(190,390/range);
    feet.forEach((target,n)=>{ctx.save();ctx.translate(400-(target.y_m-anchor[1])*scale,590-(target.x_m-anchor[0])*scale);ctx.rotate(-target.yaw_deg*Math.PI/180);ctx.globalAlpha=row.guidance_visual_opacity*(n?0.5:1);ctx.fillStyle='#127e89';ctx.beginPath();ctx.ellipse(0,0,10,24,0,0,Math.PI*2);ctx.fill();text(target.foot==='left'?'左':'右',16,5,17);ctx.restore()});
    ctx.fillStyle='#18334c';ctx.beginPath();ctx.arc(400,637,15,0,Math.PI*2);ctx.fill();text('示意人物',435,646,15,'#607587');text(t.toFixed(1)+' s',675,200,21);
    text(stateNames[row.guidance_control_state]||row.guidance_control_state,810,207,28,'#127e89');text('FoG 评分  '+Number(row.fog_probability??0).toFixed(2),810,259,23);text('方向  '+({straight:'直行',left:'左转',right:'右转',unknown:'未知'}[row.turn]||row.turn),810,303,23);text('步幅  '+Number(row.guidance_step_length_m).toFixed(2)+' m',810,347,23);text('节拍  '+row.guidance_cadence_bpm+' BPM',810,391,23);text('透明度 / 音量  '+Math.round(row.guidance_visual_opacity*100)+'% / '+Math.round(row.guidance_audio_volume*100)+'%',810,435,21);text('向前恢复步数  '+row.guidance_stable_steps,810,479,21);
    ctx.font='20px "Microsoft YaHei",sans-serif';ctx.fillStyle='#18334c';wrap(reasonNames[row.guidance_reason]||row.guidance_reason,810,536,410);text('模拟踏步不计作向前恢复',810,646,16,'#607587');text('课程工程原型；仍有误报、漏提示及方向拒识，未验证患者疗效',45,700,17,'#607587');
    recordButton.textContent='视频导出 '+elapsed.toFixed(0)+' / '+duration+' 秒';
    if(elapsed>=duration){clearInterval(interval);recorder.stop()}
  }
  recorder.start(1000);draw();interval=setInterval(draw,1000/15);
};

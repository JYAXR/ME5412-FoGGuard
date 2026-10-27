"""Offline review artifact; renders saved decisions and never infers control state."""

import json
from pathlib import Path


def write_guidance_review(path: Path, cases: dict, report: dict) -> None:
    payload = json.dumps({"cases": cases, "report": report}, ensure_ascii=False,
                         allow_nan=False).replace("<", "\\u003c")
    path.write_text(_TEMPLATE.replace("__GUIDANCE_DATA__", payload), encoding="utf-8")


_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>FoGGuard · 自动化 1 本地审阅</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#193047;font:15px/1.6 system-ui,sans-serif}
main{max-width:1180px;margin:30px auto;padding:0 24px}h1{font-size:30px;margin:6px 0}h2{font-size:18px}
.eyebrow{color:#207d80;font-weight:700;letter-spacing:2px}p{margin:6px 0 18px}.muted{color:#61748a}
.toolbar,.card{background:white;border:1px solid #dbe3ed;border-radius:15px;padding:20px}
.toolbar{display:flex;align-items:center;gap:18px;flex-wrap:wrap;margin:22px 0}
select,button{font:inherit;padding:8px 12px;border:1px solid #c4d2e1;border-radius:8px;background:white}
button{cursor:pointer;color:#146b70}input{accent-color:#197d82;width:100%}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.badge{display:inline-block;border-radius:8px;padding:4px 10px;background:#e3f1f0;color:#146b70;font-weight:700}
.facts{display:grid;grid-template-columns:1fr 1fr;gap:12px}.fact{padding:12px;background:#f6f8fb;border-radius:8px}
.fact strong{display:block;font-size:21px}.reason{padding:14px;background:#eef5fb;border-radius:8px;overflow-wrap:anywhere}
svg{width:100%;display:block;background:#f7fafc;border-radius:8px}.timeline{display:flex;height:20px;gap:2px;margin:12px 0}
.timeline span{flex:1;cursor:pointer;border-radius:2px}.summary{font-size:14px;border-top:1px solid #dce5ed;margin-top:18px;padding-top:12px}
pre{white-space:pre-wrap;font:12px/1.6 monospace;max-height:130px;overflow:auto}label{font-weight:600}
@media(max-width:760px){.grid{grid-template-columns:1fr}main{padding:0 12px}.toolbar{gap:10px}}
</style><main>
<div class="eyebrow">FOGGUARD / CONTROL REVIEW</div><h1>脚印＋声音提示 · 自动化 1</h1>
<p class="muted">离线审阅：读取同一控制输出。人工候选与模拟响应只用于验证工程行为；局部脚印不是实测定位。</p>
<div class="toolbar"><label>场景 <select id="scenario"></select></label>
<button id="play">播放时间线</button><button id="sound">试听当前节拍</button>
<span id="clock" class="badge"></span><span id="provenance" class="muted"></span></div>
<div class="grid"><section class="card"><h2>配置脚印目标 · x 向前 / y 向左</h2>
<svg id="targets" viewBox="0 0 500 360" style="height:360px"></svg>
<p class="muted">蓝色：左脚；橙色：右脚。当前目标只在接受有效模拟落脚后更新。</p><pre id="targetids"></pre></section>
<section class="card"><div id="state" class="badge"></div><h2 id="action"></h2>
<div class="facts"><div class="fact">上游事件阶段<strong id="phase"></strong></div>
<div class="fact">事件 ID<strong id="event"></strong></div><div class="fact">声音节拍<strong id="bpm"></strong></div>
<div class="fact">音量 / 视觉透明度<strong id="strength"></strong></div>
<div class="fact">声音模式<strong id="mode" style="font-size:16px"></strong></div>
<div class="fact">连续模拟步数<strong id="steps"></strong></div></div>
<h2>控制原因</h2><div id="reason" class="reason"></div><p id="response" class="muted"></p>
<pre id="command"></pre></section></div>
<section class="card" style="margin-top:18px"><label for="position">窗口与命令时间线 <span id="index"></span></label>
<input type="range" min="0" value="0" id="position"><div id="timeline" class="timeline"></div>
<svg id="scores" viewBox="0 0 900 95" style="height:95px"></svg>
<p class="muted">色带表示控制状态；曲线是输入模型评分。start 启动一次，hold 维持，update 更新，stop 撤销。</p>
<div id="summary" class="summary"></div></section>
<p class="muted" style="margin-top:20px">契约提案 guidance-proposal-v1 · 供组内审阅与接口对齐。实际播放/最终视频由展示端消费命令。</p>
</main><script>
const data=__GUIDANCE_DATA__;
const $=id=>document.getElementById(id); let timer=null,rows=[];
const colors={IDLE:'#cad5e1',GUIDING:'#237eac',EXIT_OBSERVE:'#a2b6d1',RESET_CUE:'#dd9a37',
RECOVERING:'#238d85',FADING:'#93c6b2',SAFE_NONE:'#ba6b76',FAILED:'#913e49'};
for(const name of Object.keys(data.cases)){const o=document.createElement('option');o.value=o.textContent=name;$('scenario').append(o)}
if(data.cases.straight_fog)$('scenario').value='straight_fog';
function load(){clearInterval(timer);timer=null;$('play').textContent='播放时间线';rows=data.cases[$('scenario').value];
$('position').max=rows.length-1;$('position').value=Math.max(0,rows.findIndex(r=>r.guidance_action==='start'));$('timeline').replaceChildren();
rows.forEach((r,i)=>{const s=document.createElement('span');s.style.background=colors[r.guidance_control_state]||'#aaa';
s.title=`${r.guidance_timestamp_s}s ${r.guidance_control_state}`;s.onclick=()=>{$('position').value=i;draw()};$('timeline').append(s)});
const points=rows.filter(r=>!r.guidance_terminal).map((r,i)=>`${10+i*880/Math.max(1,rows.length-2)},${85-75*(r.fog_probability||0)}`).join(' ');
$('scores').innerHTML=`<line x1="10" x2="890" y1="85" y2="85" stroke="#c7d4e0"/><polyline points="${points}" fill="none" stroke="#237eac" stroke-width="2"/>`;
const s=data.report.cases[$('scenario').value];$('provenance').textContent=s.source;
$('summary').textContent=`本策略：${s.start_commands} 次启动，${s.stop_commands} 次停止，覆盖 ${s.active_duration_s}s，未关闭 ${s.unclosed_commands}。固定对照：${s.fixed_reference.start_commands} 次启动、覆盖 ${s.fixed_reference.active_duration_s}s。`;
draw()}
function draw(){const i=Number($('position').value),r=rows[i];$('clock').textContent=`t = ${r.guidance_timestamp_s.toFixed(2)} s`;
$('index').textContent=`${i+1} / ${rows.length}`;$('state').textContent=r.guidance_control_state;
$('action').textContent=`${r.guidance_action.toUpperCase()} · ${r.turn}`;$('phase').textContent=r.event_phase;
$('event').textContent=r.guidance_event_id??'—';$('bpm').textContent=r.guidance_audio_enabled?`${r.guidance_cadence_bpm} BPM`:'关闭';
$('strength').textContent=`${r.guidance_audio_volume.toFixed(2)} / ${r.guidance_visual_opacity.toFixed(2)}`;
$('mode').textContent=r.guidance_audio_mode;$('steps').textContent=r.guidance_stable_steps;
$('reason').textContent=r.guidance_reason;$('response').textContent=`响应来源：${r.guidance_response_source}`;
$('command').textContent=`command: ${r.guidance_command_id??'—'}
started: ${r.guidance_cue_started_at_s??'—'}s · target step length: ${r.guidance_step_length_m.toFixed(2)}m`;
const targets=typeof r.guidance_targets==='string'?JSON.parse(r.guidance_targets):r.guidance_targets;
$('sound').disabled=!r.guidance_audio_enabled;
$('targetids').textContent=targets.map(t=>`${t.target_id} (${t.x_m.toFixed(2)}, ${t.y_m.toFixed(2)})`).join('\n');
const xs=[0,...targets.map(t=>-t.y_m)],ys=[0,...targets.map(t=>-t.x_m)];
const left=Math.min(...xs)-.3,top=Math.min(...ys)-.3,w=Math.max(.8,Math.max(...xs)-left+.3),h=Math.max(1.2,Math.max(...ys)-top+.3);
$('targets').setAttribute('viewBox',`${left} ${top} ${w} ${h}`);
let svg=`<circle cx="0" cy="0" r=".05" fill="#193047"/><text x=".09" y=".07" font-size=".09" fill="#61748a">配置原点</text>`;
if(targets.length){svg+=`<polyline points="0,0 ${targets.map(t=>`${-t.y_m},${-t.x_m}`).join(' ')}" fill="none" stroke="#bed1df" stroke-width=".018" stroke-dasharray=".04 .04"/>`;
svg+=targets.map((t,j)=>`<g opacity="${r.guidance_visual_opacity*(j===0?1:.5)}" transform="translate(${-t.y_m} ${-t.x_m}) rotate(${-t.yaw_deg})"><ellipse rx=".065" ry=".12" fill="${t.foot==='left'?'#237eac':'#d99337'}"/><text y=".035" text-anchor="middle" font-size=".085" fill="white">${t.foot==='left'?'L':'R'}</text></g>`).join('')}
else svg+=`<text x="-.19" y="-.45" font-size=".12" fill="#61748a">当前无救援目标</text>`;
$('targets').innerHTML=svg}
$('scenario').onchange=load;$('position').oninput=draw;
$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放时间线';return}
if(Number($('position').value)>=rows.length-1)$('position').value=0;$('play').textContent='暂停';
timer=setInterval(()=>{if(Number($('position').value)>=rows.length-1){clearInterval(timer);timer=null;$('play').textContent='播放时间线';return}
$('position').value=Number($('position').value)+1;draw()},500)};
$('sound').onclick=()=>{const r=rows[Number($('position').value)];if(!r.guidance_audio_enabled)return;
const ctx=new(window.AudioContext||window.webkitAudioContext)();
const spacing=r.guidance_audio_mode==='countdown_3_2_1_go'?1:60/r.guidance_cadence_bpm;
for(let i=0;i<4;i++){const o=ctx.createOscillator(),g=ctx.createGain(),t=ctx.currentTime+i*spacing;
o.frequency.value=i===3&&spacing===1?880:660;g.gain.setValueAtTime(0,t);g.gain.linearRampToValueAtTime(.15*r.guidance_audio_volume,t+.005);
g.gain.exponentialRampToValueAtTime(.001,t+.09);o.connect(g).connect(ctx.destination);o.start(t);o.stop(t+.1)}
setTimeout(()=>ctx.close(),(spacing*3+.2)*1000)};load();
</script></html>'''

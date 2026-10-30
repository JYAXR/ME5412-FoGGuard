"""Independent audit of saved updated cue comparisons; no study metrics imported."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--prior-joint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
run=args.run;repo=Path(__file__).resolve().parents[1]
if args.output.exists():raise ValueError('Preserve prior verification: choose a fresh output file')
args.output.parent.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((run/'MANIFEST.json').read_text(encoding='utf8'))['files_sha256']
for p,h in manifest.items():assert sha(run/p)==h,p
result=json.loads((run/'RESULTS.json').read_text(encoding='utf8'))
for p,h in result['source_sha256'].items():assert sha(repo/p)==h,p
stats=pd.read_csv(run/'runs.csv');summary=pd.read_csv(run/'summary.csv');pairs=pd.read_csv(run/'paired_comparisons.csv')
checked=0;positive_counter_checks=0;legacy_files=0;all_values=[]
for origin in ['real','constructed']:
 for folder in sorted((run/origin).iterdir()):
  if not folder.is_dir():continue
  p=pd.read_csv(folder/'perception_runtime.csv');fb=pd.read_csv(folder/'feedback_runtime.csv')
  assert not any(c.startswith('truth_') or c in ('fog','activity') for c in p)
  for strategy in result['strategies']:
   d=pd.read_csv(folder/f'commands_{strategy}.csv');a=d[~d.guidance_terminal].copy();times=a.guidance_timestamp_s.to_numpy(float)
   dt=np.diff(times,append=times[-1]);active=a.guidance_visual_enabled.to_numpy(bool)
   assert (dt>=0).all() and a.guidance_timestamp_s.is_unique
   assert len(a)==len(p) and np.allclose(times,p.available_at_s,rtol=0,atol=1e-8)
   for col in p.columns:
    if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(p[col]):
     np.testing.assert_allclose(a[col],p[col],rtol=0,atol=1e-8,equal_nan=True)
    else: assert a[col].fillna('<missing>').astype(str).equals(p[col].fillna('<missing>').astype(str)),col
   assert not (a.guidance_visual_enabled & (~a.input_fresh | a.turn.eq('unknown'))).any()
   if strategy!='fixed':assert not (a.guidance_visual_enabled & ~a.proxy_feedback_valid_now).any()
   assert (a.guidance_visual_enabled==a.guidance_audio_enabled).all()
   assert a.guidance_stable_steps.eq(0).all() and a.guidance_step_length_m.eq(.4).all()
   if strategy=='no_fade':assert a.loc[a.guidance_visual_enabled,'guidance_visual_opacity'].eq(1).all()
   for val in a.guidance_targets:
    targets=json.loads(val)
    for t in targets:
     assert t['foot'] in ('left','right') and all(np.isfinite(t[k]) for k in ('x_m','y_m','yaw_deg'))
   starts=d[d.guidance_action.eq('start')];stops=d[d.guidance_action.eq('stop')]
   assert set(starts.guidance_command_id)==set(stops.guidance_command_id)
   assert not starts.guidance_command_id.duplicated().any() and not stops.guidance_command_id.duplicated().any()
   for row in starts.itertuples(index=False):
    stop=stops[stops.guidance_command_id.eq(row.guidance_command_id)].iloc[0]
    assert -1e-7<=stop.guidance_timestamp_s-row.guidance_timestamp_s<=15+1e-7
   last=0.
   for row in a.itertuples(index=False):
    value=float(row.proxy_no_motion_valid_s)
    if value>last+1e-7:
     positive_counter_checks+=1
     assert row.proxy_observation_is_new and row.proxy_feedback_valid_now and row.proxy_feedback_state=='no_movement_evidence'
     f=fb[(fb.feedback_available_at_s-row.guidance_timestamp_s).abs()<1e-6]
     assert len(f)==1 and f.feedback_window_start_s.iloc[0]>=row.guidance_cue_started_at_s-1e-7
     assert value-last<=29/60+1e-7
    last=value
   recorded=stats[(stats.case==folder.name)&(stats.strategy==strategy)]
   assert len(recorded)==1
   r=recorded.iloc[0]
   values={'starts':len(starts),'stops':len(stops),'active_s':float(np.sum(dt*active)),
           'visual_intensity_s':float(np.sum(dt*active*a.guidance_visual_opacity)),
           'audio_volume_s':float(np.sum(dt*a.guidance_audio_enabled*a.guidance_audio_volume)),
           'fade_entries':int(a.guidance_reason.eq('perception_recovery_with_motion_proxy_begin_fade').sum()),
           'fallback_entries':int(a.guidance_reason.isin(['valid_no_motion_enter_march','valid_no_motion_enter_countdown']).sum()),
           'fallback_timeout_returns':int(a.guidance_reason.eq('fallback_timeout_return_forward_unverified').sum())}
   for k,v in values.items():assert abs(float(r[k])-v)<1e-7,(folder.name,strategy,k)
   all_values.append({'origin':r.origin,'strategy':strategy,**values})
   if origin=='real' and strategy in ['fixed','countdown','march']:
    old=pd.read_csv(args.prior_joint/f'real/{folder.name}/commands_{strategy}.csv')
    assert len(d)==len(old)
    for col in old.columns:
     if pd.api.types.is_numeric_dtype(old[col]) and pd.api.types.is_numeric_dtype(d[col]):
      np.testing.assert_allclose(d[col],old[col],rtol=0,atol=1e-8,equal_nan=True)
     else:assert old[col].fillna('<missing>').astype(str).equals(d[col].fillna('<missing>').astype(str)),col
    legacy_files+=1
   checked+=len(d)
  # Single-factor comparisons: exactly the same perception/feedback in the same case.
  march=pd.read_csv(folder/'commands_march.csv');countdown=pd.read_csv(folder/'commands_countdown.csv')
  assert np.array_equal(march.guidance_visual_enabled,countdown.guidance_visual_enabled)
  assert np.allclose(march.guidance_visual_opacity,countdown.guidance_visual_opacity)
  feedback=pd.read_csv(folder/'commands_feedback.csv')
  assert not feedback.guidance_control_state.isin(['MARCH_FALLBACK','RESET_CUE']).any()
branch_checks=0
for turn in ('straight','left','right'):
 for strategy in result['strategies']:
  for profile in ('low_motion','normal','uncertain_motion','recovery_early','recovery_late','refreeze'):
   d=pd.read_csv(run/'constructed'/f'{profile}_{turn}'/f'commands_{strategy}.csv')
   if profile=='normal':assert not d.guidance_visual_enabled.any()
   elif profile=='uncertain_motion':
    if strategy!='fixed':assert not d.guidance_visual_enabled.any()
   elif profile=='low_motion':
    count=int(d.guidance_reason.isin(['valid_no_motion_enter_march','valid_no_motion_enter_countdown']).sum())
    assert count==(0 if strategy in ('fixed','feedback') else 1)
   elif profile.startswith('recovery'):
    fading=d.guidance_control_state.eq('FADING_PROXY').any()
    assert fading==(strategy not in ('fixed','no_fade'))
   elif profile=='refreeze' and strategy not in ('fixed','no_fade'):
    assert d.guidance_reason.eq('perception_reconfirmed_restore_intensity').any()
   branch_checks+=1
recount=pd.DataFrame(all_values).groupby(['origin','strategy']).sum(numeric_only=True)
for row in summary.itertuples(index=False):
 for k in ['starts','stops','active_s','visual_intensity_s','audio_volume_s','fade_entries','fallback_entries','fallback_timeout_returns']:
  assert abs(getattr(row,k)-recount.loc[(row.origin,row.strategy),k])<1e-6
assert stats.measured_spacing_error_m.isna().all()
for row in pairs.itertuples(index=False):
 a=stats[(stats.origin==row.origin)&(stats.strategy==row.reference)].set_index('case')[row.metric]
 b=stats[(stats.origin==row.origin)&(stats.strategy==row.alternative)].set_index('case')[row.metric].reindex(a.index)
 assert len(a)==row.pairs and abs(float((a-b).mean())-row.mean_difference)<1e-8
report={'status':'PASS','manifest_files_verified':len(manifest),'source_files_verified':len(result['source_sha256']),
        'constructed_branch_checks':branch_checks,'case_strategy_runs':len(stats),'command_rows_checked':checked,'valid_counter_increases_checked':positive_counter_checks,
        'unchanged_prior_real_timelines_verified':legacy_files,'pair_rows_recomputed':len(pairs),
        'spacing_accuracy_unavailable':True,'scope':'Engineering replay checks; no patient response or treatment validation',
        'verification_script_sha256':sha(Path(__file__))}
args.output.write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report))

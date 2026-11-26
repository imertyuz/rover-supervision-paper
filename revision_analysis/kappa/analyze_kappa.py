from pathlib import Path
import json,pickle,collections,math,bisect,csv,statistics,ast
import numpy as np
B=Path(__file__).parent;BASE=B.parent
# Historical selector used unchanged for exact manuscript reproduction.
m={};tree=ast.parse((BASE.parent/'revision_work/analysis/analysis_scripts/model_validation.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Import,ast.ImportFrom)) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MISSION_S' for t in n.targets))],type_ignores=[]),'historical_model','exec'),m)
runs=pickle.load(open(B/'loaded_data.pkl','rb')); summaries=[];one_rows=[];frame_rows=[];gap_rows=[];pred_total=collections.Counter();hist_total=collections.Counter();hist_pool=collections.Counter();longs=[];allgaps=[];keptgaps=[];allfusiondt=[]
Tf=.1

def positive(s):return isinstance(s,(int,float)) and math.isfinite(s) and s>0

def merge(intervals):
 out=[]
 for a,b in sorted(intervals):
  if b<=a:continue
  if out and a<=out[-1][1]+1e-7:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return out

def predict(intervals,origin):
 # Exact uniform-phase expectation for the ideal timer over disjoint eligible intervals.
 spans=[(a-origin,b-origin) for a,b in intervals];cuts={0.,Tf}
 for a,b in spans:cuts.add(a%Tf);cuts.add(b%Tf)
 cuts=sorted(cuts);prob=collections.Counter()
 for lo,hi in zip(cuts,cuts[1:]):
  if hi-lo<1e-10:continue
  phi=(lo+hi)/2
  k=sum(math.ceil((b-phi)/Tf)-math.ceil((a-phi)/Tf) for a,b in spans)
  prob[k]+=(hi-lo)/Tf
 assert abs(sum(prob.values())-1)<1e-6
 return dict(prob)

for R in runs:
 T=R['test'];bag=R['bag'];state=R['state'];w=R['window'];ts=[s['receipt'] for s in state];state3=[(s['receipt'],s['mode'],s['armed']) for s in state];ok=m['armed_auto_selector'](state3,T)
 t0=next(t for t,mode,armed in state3 if mode=='AUTO');end=t0+m['MISSION_S'][T]
 def dynamic_ok(t):
  idx=bisect.bisect_right(ts,t)-1
  return t0<=t<=w['WP5_event_s'] and idx>=0 and state[idx]['armed'] and state[idx]['mode']=='AUTO' and t-ts[idx]<=2
 gates=[]
 for j,z in enumerate(state):
  if z['armed'] and z['mode']=='AUTO':
   a=max(t0,z['receipt']);b=min(end,z['receipt']+2,state[j+1]['receipt'] if j+1<len(state) else end)
   if b>a:gates.append((a,b))
 gates=merge(gates)
 cams=bag['/camera/perception/status'];fullbr=bag['/safety_bridge/status'];fusion=bag['/fusion/status_v4'];br=[(t,p) for t,p in fullbr if ok(t) and positive(p.get('camera_source_stamp_s'))]
 counts=collections.Counter(round(p['camera_source_stamp_s'],3) for t,p in br);hist=collections.Counter(counts.values());hist_total.update(hist)
 frames={}
 for t,p in cams:
  if ok(t) and p.get('valid') and positive(p.get('yolo_sensor_stamp_s')):frames.setdefault(round(p['yolo_sensor_stamp_s'],3),(t,p))
 S=sorted(frames);gaps=[b-a for a,b in zip(S,S[1:])];allgaps+=gaps;keptgaps +=[g for g in gaps if g<2]
 for a,b in zip(S,S[1:]):
  if b-a>=2:
   row={'test':T,'sigma_before':a,'sigma_after':b,'gap_s':b-a,'kind':'gap_between_selected_valid_camera_sources'};longs.append(row);gap_rows.append(row)
 source_steps=[p['camera_source_stamp_s'] for t,p in br];back=sum(b<a-1e-6 for a,b in zip(source_steps,source_steps[1:]))
 # Independent camera availability intervals from recorder arrivals; invalid output clears current source.
 timeline=[]
 for t,p in cams:
  key=round(p['yolo_sensor_stamp_s'],3) if p.get('valid') and positive(p.get('yolo_sensor_stamp_s')) else None
  if not timeline or key!=timeline[-1][1]:timeline.append([t,key,p.get('yolo_sensor_stamp_s')])
 available=collections.defaultdict(list)
 for j,(a,key,raw) in enumerate(timeline):
  if key is None:continue
  b=min(timeline[j+1][0] if j+1<len(timeline) else end,raw+2.)
  if b>a:available[key].append((a,b))
 eligible={}
 for key,ivals in available.items():
  clipped=[]
  for a,b in ivals:
   for c,d in gates:
    if min(b,d)>max(a,c):clipped.append((max(a,c),min(b,d)))
  clipped=merge(clipped)
  if clipped:eligible[key]=clipped
 pool=set(eligible)|set(counts);pred=collections.Counter();unmapped=set(counts)-set(eligible)
 # No measured kappa or Bridge-derived dwell length is used to construct the prediction.
 for key in sorted(pool):
  observed=counts.get(key,0);hist_pool[observed]+=1
  if key not in eligible:continue
  probs=predict(eligible[key],t0);pred.update(probs);pred_total.update(probs)
  length=sum(b-a for a,b in eligible[key]);all_len=sum(b-a for a,b in available[key]);clip=all_len-length>1e-5
  probs_obs=probs.get(observed,0)
  frame_rows.append({'test':T,'source_stamp':key,'observed_kappa':observed,'eligible_camera_availability_s':length,'availability_parts':len(eligible[key]),'gate_or_window_clipped':clip,'ideal_expected_kappa':sum(k*v for k,v in probs.items()),'observed_kappa_possible_under_ideal':probs_obs>1e-6})
 for key,k in counts.items():
  if k!=1:continue
  selected=[(t,p) for t,p in br if round(p['camera_source_stamp_s'],3)==key];t,p=selected[0]
  allmatches=[(tt,pp) for tt,pp in fullbr if positive(pp.get('camera_source_stamp_s')) and round(pp['camera_source_stamp_s'],3)==key]
  ivals=eligible.get(key,[]);L=sum(b-a for a,b in ivals);av=available.get(key,[]);avL=sum(b-a for a,b in av)
  clip=bool(ivals) and avL-L>1e-5
  near=[pp['stamp'] for tt,pp in fusion if t-.5<=pp['stamp']<=t+.5];near.sort();maxgap=max(np.diff(near),default=0)
  poss=predict(ivals,t0).get(1,0)>1e-6 if ivals else False
  category='selection_boundary' if clip else ('short_availability_ideal_compatible' if poss else 'outside_ideal_prediction_or_arrival_proxy')
  one_rows.append({'test':T,'source_stamp':key,'bridge_bag_time':t,'bridge_publication':p.get('stamp'),'seconds_from_first_AUTO':p.get('stamp',t)-t0,'camera_availability_s':avL,'eligible_availability_s':L,'all_recording_bridge_repetitions':len(allmatches),'gate_or_window_clipped':clip,'ideal_kappa1_possible':poss,'nearby_max_fusion_interval_s':float(maxgap),'category':category})
 fts=sorted(p['stamp'] for t,p in fusion if t0<=p['stamp']<=end);dt=np.diff(fts);allfusiondt+=list(dt)
 # Same float values in deployed payload, grouping at 1 us vs 1 ms.
 k6=collections.Counter(round(p['camera_source_stamp_s'],6) for t,p in br)
 camraw=[p['yolo_sensor_stamp_s'] for t,p in cams if ok(t) and p.get('valid') and positive(p.get('yolo_sensor_stamp_s'))]
 byround=collections.defaultdict(set)
 for raw in camraw:byround[round(raw,3)].add(round(raw,6))
 collisions=sum(len(v)>1 for v in byround.values())
 summ={'test':T,'n_bridge':len(br),'n_bridge_sources':len(counts),'hist':dict(sorted(hist.items())),'n_selected_valid_camera_sources':len(frames),'all_source_gaps':len(gaps),'excluded_ge2_gaps':sum(g>=2 for g in gaps),'mean_all_source_gaps_s':float(np.mean(gaps)),'mean_lt2_source_gaps_s':float(np.mean([g for g in gaps if g<2])),'excluded_gap_sum_s':sum(g for g in gaps if g>=2),'mean_kappa':sum(counts.values())/len(counts),'backwards_steps':back,'bridge_group_hist_at_1us_same':collections.Counter(k6.values())==hist,'camera_rounding_collision_bins':collisions,'camera_availability_pool':len(eligible),'unmapped_observed_sources':len(unmapped),'observed_kappa0_on_camera_pool':sum(k not in counts for k in eligible),'predicted_hist':dict(sorted(pred.items())),'observed_on_same_prediction_pool':dict(sorted(collections.Counter(counts.get(k,0) for k in eligible).items())),'boundary_ones':sum(x['test']==T and x['category']=='selection_boundary' for x in one_rows),'dynamic_endpoint_bridge_count':sum(dynamic_ok(t) and positive(p.get('camera_source_stamp_s')) for t,p in fullbr),'legacy_mission_end_s':end,'wp5_event_end_s':w['WP5_event_s'],'fusion_dt':{'n':len(dt),'min':float(min(dt)),'median':float(np.median(dt)),'p99':float(np.percentile(dt,99)),'max':float(max(dt))}}
 summaries.append(summ);print(json.dumps(summ),flush=True)

def write_csv(name,rows):
 if not rows:return
 with (B/name).open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
write_csv('kappa1_cases.csv',one_rows);write_csv('camera_availability_frames.csv',frame_rows);write_csv('excluded_source_gaps.csv',gap_rows)
result={'per_test':summaries,'observed_histogram':dict(sorted(hist_total.items())),'n_bridge':sum(r['n_bridge'] for r in summaries),'n_bridge_sources':sum(r['n_bridge_sources'] for r in summaries),'n_kappa1':len(one_rows),'kappa1_categories':dict(collections.Counter(x['category'] for x in one_rows)),'source_gaps':{'all_n':len(allgaps),'lt2_n':len(keptgaps),'excluded_n':len(longs),'excluded_sum_s':sum(x['gap_s'] for x in longs),'mean_all_s':float(np.mean(allgaps)),'mean_lt2_s':float(np.mean(keptgaps))},'prediction_histogram':dict(sorted(pred_total.items())),'observed_histogram_with_camera_pool_zeros':dict(sorted(hist_pool.items())),'fusion_interval_s':{'n':len(allfusiondt),'min':float(min(allfusiondt)),'median':float(np.median(allfusiondt)),'p99':float(np.percentile(allfusiondt,99)),'max':float(max(allfusiondt))}}
(B/'kappa_results.json').write_text(json.dumps(result,indent=2));print('POOLED',json.dumps(result))
assert result['observed_histogram']=={1:29,2:2135,3:2559,4:164,5:8}

from pathlib import Path
import json,pickle,csv,collections,bisect
import numpy as np
B=Path(__file__).parent;r=json.loads((B/'kappa_results.json').read_text());runs=pickle.load(open(B/'loaded_data.pkl','rb'));gaps=list(csv.DictReader((B/'excluded_source_gaps.csv').open()));out=[]
for row in gaps:
 R=next(x for x in runs if x['test']==row['test']);a=float(row['sigma_before']);b=float(row['sigma_after']);cam=R['bag']['/camera/perception/status'];state=R['state'];st=[x['receipt'] for x in state]
 raw=[(t,p) for t,p in cam if isinstance(p.get('yolo_sensor_stamp_s'),(int,float)) and a<=p['yolo_sensor_stamp_s']<=b]
 source_sorted=sorted({p['yolo_sensor_stamp_s'] for t,p in raw if p.get('valid')})
 excluded=[]
 for t,p in raw:
  idx=bisect.bisect_right(st,t)-1
  if idx>=0 and not(state[idx]['armed'] and state[idx]['mode']=='AUTO'):excluded.append((t,p))
 modes=collections.Counter(state[max(0,bisect.bisect_right(st,t)-1)]['mode'] for t,p in raw)
 out.append({**row,'camera_publications_between_capture_endpoints':len(raw),'valid_distinct_sources_between':len(source_sorted),'max_adjacent_valid_capture_gap_s':max(np.diff(source_sorted),default=0),'outputs_not_armed_AUTO':len(excluded),'modes_at_camera_arrivals':dict(modes)})
r['excluded_gap_audit']=out
N=sum(r['prediction_histogram'].values());obs=r['observed_histogram_with_camera_pool_zeros'];pred=r['prediction_histogram'];r['histogram_total_variation']=sum(abs(obs.get(k,0)-pred.get(k,0)) for k in set(obs)|set(pred))/(2*N)
rows=list(csv.DictReader((B/'camera_availability_frames.csv').open()));bad=[x for x in rows if x['observed_kappa_possible_under_ideal']=='False'];r['outside_ideal_frame_bounds']=len(bad);r['prediction_pool_n']=int(N)
with (B/'outside_ideal_bounds.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(bad)
(B/'kappa_results.json').write_text(json.dumps(r,indent=2))
print('GAPS',json.dumps(out,indent=2));print('dynamic count',[(x['test'],x['n_bridge'],x['dynamic_endpoint_bridge_count'],x['wp5_event_end_s']-x['legacy_mission_end_s']) for x in r['per_test']]);print('Precision checks',[(x['test'],x['bridge_group_hist_at_1us_same'],x['camera_rounding_collision_bins']) for x in r['per_test']])

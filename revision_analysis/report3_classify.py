from pathlib import Path
import json,collections,csv,hashlib
base=Path(__file__).parent
traces=json.loads((base/'report3_fusion_traces.json').read_text()); rows=[];summary=[]
def stamp(p,sensor):return p.get('sensors',{}).get(sensor,{}).get('raw_stamp_s')
def reasonkeys(p):return [str(r).split(':')[0] for r in p.get('reasons',[])]
for run in dict.fromkeys(x['run'] for x in traces):
 ts=[x for x in traces if x['run']==run];counts=collections.Counter();trans=collections.Counter();causes=collections.Counter()
 for x in ts:
  a,b=x['recorded'],x['replayed'];diff=[k for k in ['state','hazard_type','reasons','primary_reason','fresh'] if a.get(k)!=b.get(k)]
  if not diff:continue
  counts['any']+=1
  state=a['state']!=b['state']
  if state:trans[a['state']+'->'+b['state']]+=1
  dc=stamp(a,'camera')!=stamp(b,'camera');dl=stamp(a,'lidar')!=stamp(b,'lidar');di=stamp(a,'imu')!=stamp(b,'imu')
  if state and ('CRITICAL' in (a['state'],b['state'])):cat='critical_state_difference'
  elif state and ('person' in (a.get('hazard_type'),b.get('hazard_type'))):cat='person_caution_clear_difference'
  elif state and any('tilt_caution' in r for r in reasonkeys(a)+reasonkeys(b)):cat='tilt_caution_clear_difference'
  elif state:cat='clear_caution_geometry_boundary'
  elif reasonkeys(a)==reasonkeys(b) and a.get('hazard_type')==b.get('hazard_type'):cat='same_reason_categories_numeric_difference'
  elif a.get('hazard_type')!=b.get('hazard_type'):cat='same_state_hazard_selection_difference'
  else:cat='same_state_reason_category_difference'
  causes[cat]+=1
  rows.append({'run':run,'stamp_s':x['stamp_s'],'fields':','.join(diff),'recorded_state':a['state'],'replayed_state':b['state'],'category':cat,'camera_stamp_diff':dc,'lidar_stamp_diff':dl,'imu_stamp_diff':di,'recorded_camera_stamp':stamp(a,'camera'),'replayed_camera_stamp':stamp(b,'camera'),'recorded_lidar_stamp':stamp(a,'lidar'),'replayed_lidar_stamp':stamp(b,'lidar'),'recorded_reasons':json.dumps(a.get('reasons')),'replayed_reasons':json.dumps(b.get('reasons'))})
 summary.append({'run':run,'n':len(ts),'any':counts['any'],'state_transitions':dict(trans),'categories':dict(causes),'state_camera_stamp_differences':sum(r['camera_stamp_diff'] for r in rows if r['run']==run and r['recorded_state']!=r['replayed_state']),'state_lidar_stamp_differences':sum(r['lidar_stamp_diff'] for r in rows if r['run']==run and r['recorded_state']!=r['replayed_state'])})
with (base/'report3_all_mismatches.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(base/'report3_classification.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

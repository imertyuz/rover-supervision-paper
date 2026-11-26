"""Open-loop, virtual-clock Fusion/Bridge rate-phase experiment.
Frozen source functions unchanged; alternative two-message counter is explicitly labelled.
"""
from pathlib import Path
import json,sqlite3,struct,collections,math,sys,csv,hashlib
import numpy as np
BASE=Path(__file__).resolve().parents[1]; ROOT=BASE.parent/'revision_stage1_check'
fns={'__file__':str(BASE/'fusion_conformance.py')}
exec((BASE/'fusion_conformance.py').read_text().split('results=[]')[0],fns)
clock=fns['clock'];FM=fns['mod'];String=fns['String'];NS=fns['NS']
bns={'__file__':str(BASE/'bridge_conformance.py'),'shared_clock':clock}
s=(BASE/'bridge_conformance.py').read_text().split('results=[]')[0].replace('clock=NS(now=0.0)','clock=shared_clock')
exec(s,bns); BM=bns['mod']; Bridge=BM.SemanticSafetyBridgeV2
class MessageBridge(Bridge):
 def _handle_semantic_critical(self,**kwargs):
  # Counterfactual only: each eligible delivery is counted, including repeated sources.
  self.semantic_last_camera_stamp=None
  return super()._handle_semantic_critical(**kwargs)

windows={x['run']:x for x in json.loads((BASE/'stage1_audit.json').read_text())}
PERIODS=[.05,.1,.2,.5]; PHASES=[0,.25,.5,.75]
all_results=[];detail=[]

def source(p):return p.get('sensors',{}).get('camera',{}).get('raw_stamp_s')
def valid_stamp(x):return isinstance(x,(int,float)) and math.isfinite(x) and x>0

for run in sorted((ROOT/'data/logs/MACHINES_PAPER/FIELDKIT_V1').glob('MISSION*'))[:4]:
 w=windows[run.name];start=w['observer_AUTO_start_s'];end=w['WP5_event_s'];dest=next(ROOT.rglob('*_'+run.name+'_ANALYSIS_unpacked'))
 callbacks=[];recorded=[];camera_available=[];fcus=[]
 for line in (run/'observer.jsonl').open():
  z=json.loads(line)
  if z['topic']=='/mavros/state':fcus.append((z['receipt_wall_s'],z['payload']))
 for db in sorted(dest.rglob('*.db3')):
  con=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
  for name,t,b in con.execute("select t.name,m.timestamp,m.data from messages m join topics t on t.id=m.topic_id where t.name in ('/camera/perception/status','/lidar/sectors_status','/lidar/sectors','/mavros/imu/data','/fusion/status_v4')"):
   t=t/1e9
   if name=='/mavros/imu/data':callbacks.append((t,'_imu_cb',fns['imu'](b)))
   elif name=='/lidar/sectors':callbacks.append((t,'_lidar_cb',NS()))
   else:
    p=fns['unpack_json'](b)
    if name=='/fusion/status_v4':recorded.append((t,p))
    else:
     callbacks.append((t,'_camera_cb' if name=='/camera/perception/status' else '_lidar_status_cb',String(json.dumps(p))))
     if name=='/camera/perception/status' and valid_stamp(p.get('yolo_sensor_stamp_s')):
      camera_available.append((t,round(p['yolo_sensor_stamp_s'],3)))
  con.close()
 callbacks.sort(key=lambda z:z[0]);recorded.sort();fcus.sort();camera_available.sort()
 t0=min(p['stamp'] for _,p in recorded); initial=min(t0,callbacks[0][0],fcus[0][0]);target_sources={s for t,s in camera_available if start<=t<=end}
 # Recorded Fusion is available only on bag arrival in the separate held-output control.
 baseevents=[(t,0,method,p) for t,method,p in callbacks if t<=end]+[(t,0,'fcu',p) for t,p in fcus if t<=end]
 holdevents=[(t,0,'recorded_fusion',p) for t,p in recorded if t<=end]+[(t,0,'fcu',p) for t,p in fcus if t<=end]
 for variant in ['frozen_chain','held_recorded_fusion']:
  for period in PERIODS:
   for phase in PHASES:
    clock.now=initial;fusion=FM.FusionNodeV4() if variant=='frozen_chain' else None
    bridges=[Bridge(),MessageBridge()]; mode='UNKNOWN';armed=False;held=None
    ticks=t0+phase*period+np.arange(int(math.ceil((end-t0)/period))+1,dtype=float)*period
    ev=(baseevents if fusion else holdevents)+[(float(t),1,'tick',None) for t in ticks if t<=end]
    ev.sort(key=lambda z:(z[0],z[1]))
    active=None;runs=[];coverage=set();evals=0;freshness=collections.Counter();requests=[[],[]];states=collections.Counter()
    def close_run():
     nonlocal_dummy=None
     if active is not None:runs.append(active)
    for t,kind,method,p in ev:
     clock.now=t
     if method=='fcu':
      mode=p['mode'];armed=p['armed']
      for node in bridges:node._state_cb(NS(mode=mode,armed=armed))
      continue
     if method=='recorded_fusion':held=p;continue
     if method!='tick':getattr(fusion,method)(p);continue
     if fusion:
      fusion._loop();out=fusion.published[-1];fusion.published.clear()
     else:
      if held is None:continue
      out=held
     outputs=[]
     for node in bridges:
      node._fusion_cb(String(json.dumps(out)));outputs.append(node.published[-1]);node.published.clear()
     if t<start:continue
     evals+=1;states[out['state']]+=1
     stamp=source(out);fresh=out.get('sensors',{}).get('camera',{}).get('fresh',False)
     if valid_stamp(stamp):coverage.add(stamp)
     for i,o in enumerate(outputs):
      if o['action'].startswith('HOLD_REQUESTED'):requests[i].append({'t':t,'action':o['action']})
     eligible=(mode=='AUTO' and out.get('state')=='CRITICAL' and out.get('hazard_type')=='person' and fresh and valid_stamp(stamp))
     if eligible:
      if active is None:active={'first_t':t,'last_t':t,'samples':0,'sources':[],'source_trigger_t':None,'message_trigger_t':None}
      active['last_t']=t;active['samples']+=1
      if not active['sources'] or abs(stamp-active['sources'][-1])>1e-6:active['sources'].append(stamp)
      if active['samples']==2:active['message_trigger_t']=t
      if len(active['sources'])==2 and active['source_trigger_t'] is None:active['source_trigger_t']=t
     elif active is not None:runs.append(active);active=None
     if mode=='AUTO':freshness['auto']+=1;freshness['invalid_AUTO']+=out['state']=='INVALID'
    if active is not None:runs.append(active)
    for r in runs:
     for policy in ['source','message']:
      tt=r[policy+'_trigger_t'];r[policy+'_delay_from_first_eval_s']=None if tt is None else tt-r['first_t']
    result={'run':run.name,'variant':variant,'period_s':period,'phase_fraction':phase,'anchor_s':t0,'evaluations':evals,'person_runs':len(runs),'Rm_person':sum(r['message_trigger_t'] is not None for r in runs),'Rs_person':sum(r['source_trigger_t'] is not None for r in runs),'available_camera_sources':len(target_sources),'sampled_available_camera_sources':len(coverage & target_sources),'unseen_available_camera_sources':len(target_sources-coverage),'auto_evaluations':freshness['auto'],'invalid_AUTO_evaluations':freshness['invalid_AUTO'],'source_bridge_requests':len(requests[0]),'message_bridge_requests':len(requests[1]),'source_person_requests':sum('CONFIRMED_PERSON' in x['action'] for x in requests[0]),'message_person_requests':sum('CONFIRMED_PERSON' in x['action'] for x in requests[1])}
    all_results.append(result);detail.append({**result,'person_episodes':runs,'source_requests':requests[0],'message_requests':requests[1],'states':dict(states)})
  print(run.name,variant,'done',flush=True)
OUT=Path(__file__).parent
(OUT/'rate_phase_results.json').write_text(json.dumps(detail,indent=2))
with (OUT/'rate_phase_summary.csv').open('w') as f:
 wr=csv.DictWriter(f,fieldnames=all_results[0]);wr.writeheader();wr.writerows(all_results)
aggregates=[]
for variant in ['frozen_chain','held_recorded_fusion']:
 for period in PERIODS:
  byphase=[]
  for phase in PHASES:
   rr=[r for r in all_results if r['variant']==variant and r['period_s']==period and r['phase_fraction']==phase]
   byphase.append({'phase_fraction':phase,**{k:sum(r[k] for r in rr) for k in ['evaluations','person_runs','Rm_person','Rs_person','available_camera_sources','unseen_available_camera_sources','source_bridge_requests','message_bridge_requests','source_person_requests','message_person_requests']}})
  aggregates.append({'variant':variant,'period_s':period,'by_phase':byphase})
(OUT/'rate_phase_aggregate.json').write_text(json.dumps(aggregates,indent=2))
print(json.dumps(aggregates,indent=2))

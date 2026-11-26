"""Recorded-Fusion policy sensitivity. No live ROS, no altered vehicle feedback.
M variants invoke the frozen Bridge with only semantic_confirm_frames changed.
2-of-3 is an explicit analytic alternative, not deployed or safety validated.
"""
from pathlib import Path
import pickle,json,bisect,csv,math,collections
B=Path(__file__).resolve().parent; BASE=B.parent
ns={'__file__':str(BASE/'bridge_conformance.py')}
exec((BASE/'bridge_conformance.py').read_text().split('results=[]')[0],ns)
clock=ns['clock']; NS=ns['NS']; String=ns['String']; Bridge=ns['mod'].SemanticSafetyBridgeV2
runs=pickle.load(open(BASE/'kappa/loaded_data.pkl','rb'))
rows=[];trace=[];event_id=0
def two_of_three(history,source,positive,window):
 if not positive:return False
 h=[x for x in history[-3:] if 0<=source-x['source']<=window+1e-6]
 return len(h)>=2 and sum(x['positive'] for x in h)>=2

# Behavioral checks: intermittent 1-0-1 may confirm; old evidence may not.
sample=[{'source':0.,'positive':True},{'source':.2,'positive':False},{'source':.4,'positive':True}]
assert two_of_three(sample,.4,True,.5)
assert not two_of_three(sample,.4,True,.3)
assert not two_of_three(sample,.4,False,.5)
assert not two_of_three(sample[:1],0.,True,.5)
for R in runs:
 state=R['state']; start=R['window']['observer_AUTO_start_s'];end=R['window']['WP5_event_s']
 nodes={m:Bridge() for m in (2,3,4)}
 for m,node in nodes.items():node.semantic_confirm_frames=m
 events=[(s['receipt'],0,s) for s in state]+[(t,1,p) for t,p in R['bag']['/fusion/status_v4']]
 mode='UNKNOWN'; history=[];last_source=None; outputs=[]
 for t,kind,p in sorted(events,key=lambda x:(x[0],x[1])):
  clock.now=t
  if kind==0:
   if p['mode']=='AUTO' and mode!='AUTO':history=[];last_source=None
   mode=p['mode']
   for node in nodes.values():node._state_cb(NS(mode=mode,armed=p['armed']))
   continue
  camera=p.get('sensors',{}).get('camera',{});s=camera.get('raw_stamp_s');fresh=bool(camera.get('fresh'))
  valid=isinstance(s,(int,float)) and math.isfinite(s) and s>0
  positive=p.get('state')=='CRITICAL' and p.get('hazard_type')=='person' and fresh and valid
  # Only first fresh Fusion presentation of each distinct source contributes
  # a binary observation. Current input must also be positive at confirmation.
  if not fresh or not valid or p.get('state')=='INVALID':history=[]
  elif last_source is None or abs(s-last_source)>1e-6:
   history.append({'source':s,'positive':positive,'first_receipt':t});history=history[-3:];last_source=s
  native={}
  for m,node in nodes.items():
   node._fusion_cb(String(json.dumps(p)));o=node.published[-1];node.published.clear()
   native[m]={'count':o['semantic_frame_count'],'action':o['action']}
  alternatives={}
  for W in (.5,1.,1.5):
   alternatives[W]=two_of_three(history,s,positive,W) if valid else False
  outputs.append({'t':t,'source':s,'positive':positive,'mode':mode,'M':native,'two_of_three':alternatives,'history':[dict(x) for x in history]})
 for e in R['window'].get('episodes',[]):
  event_id+=1;eid='P'+str(event_id);first=e['sources'][0]['bridge_source_s']
  raw=e['sources'][0]['camera_candidates'][0]['source_s'];eb=e['first_bridge_s']
  # Find recorded-Fusion episode by source and nearest first Bridge snapshot.
  candidates=[i for i,x in enumerate(outputs) if x['positive'] and abs(x['source']-first)<1e-6]
  i=min(candidates,key=lambda i:abs(outputs[i]['t']-eb))
  while i>0 and outputs[i-1]['positive']:i-=1
  j=i
  while j<len(outputs) and outputs[j]['positive']:j+=1
  episode=outputs[i:j] # extends beyond recorded HOLD, explicitly labelled
  for label in ['M2','M3','M4','2of3_W0.5','2of3_W1.0','2of3_W1.5']:
   if label.startswith('M'):
    m=int(label[1:]);hits=[x for x in episode if x['M'][m]['count']>=m]
   else:hits=[x for x in episode if x['two_of_three'][float(label.split('W')[1])]]
   hit=hits[0] if hits else None
   row={'event':eid,'test':R['test'],'policy':label,'first_source_raw_s':raw,'first_fusion_receipt_s':episode[0]['t'],'confirmation_receipt_s':hit['t'] if hit else None,'delay_from_source_s':hit['t']-raw if hit else None,'delay_from_first_fusion_s':hit['t']-episode[0]['t'] if hit else None,'mode_at_confirmation':hit['mode'] if hit else None,'confirmed_on_recorded_trace':bool(hit),'confirmed_in_recorded_AUTO':bool(hit and hit['mode']=='AUTO'),'episode_distinct_sources':len(set(x['source'] for x in episode))}
   rows.append(row)
  trace.append({'event':eid,'test':R['test'],'first_raw_source_s':raw,'episode':episode})
 # Search all mission AUTO intervals for alternative rising edges, including
 # any bridging of a single negative frame not represented by P1-P5.
 for W in (.5,1.,1.5):
  prior=False
  for x in outputs:
   value=x['two_of_three'][W]
   if start<=x['t']<=end and x['mode']=='AUTO' and value and not prior:
    trace.append({'alternative_auto_rising_edge':True,'test':R['test'],'window_s':W,'t':x['t'],'source':x['source'],'history':x['history']})
   prior=value
with (B/'confirmation_comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(B/'confirmation_trace.json').write_text(json.dumps(trace,indent=2))
summary={'method':'Recorded Fusion; original timestamps and fixed recorded FCU feedback; post-HOLD confirmations are trace-only counterfactuals. 2of3 uses up to three latest distinct first-presentation labels, current input positive, source-time span W; not object tracking.','rows':rows,'alternative_auto_rising_edges':[x for x in trace if x.get('alternative_auto_rising_edge')]}
(B/'confirmation_results.json').write_text(json.dumps(summary,indent=2))
for r in rows:print(r['event'],r['policy'],r['delay_from_source_s'],r['mode_at_confirmation'])
print('Additional AUTO edges:',summary['alternative_auto_rising_edges'])
assert len(rows)==30
assert all(not x['confirmed_on_recorded_trace'] for x in rows if x['event'] in ('P1','P2','P3','P4'))
assert all(x['confirmed_on_recorded_trace'] for x in rows if x['event']=='P5')

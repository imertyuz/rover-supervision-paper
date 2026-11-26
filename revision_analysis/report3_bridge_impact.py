from pathlib import Path
import json,collections
# Import only the original replay's ROS/service adapter setup, no original replay loop.
base=Path(__file__).parent;ns={'__file__':str(base/'bridge_conformance.py')}
exec((base/'bridge_conformance.py').read_text().split('results=[]')[0],ns)
clock=ns['clock'];Node=ns['mod'].SemanticSafetyBridgeV2;String=ns['String'];NS=ns['NS'];root=ns['ROOT']
tr=json.loads((base/'report3_fusion_traces.json').read_text());results=[]
for run in dict.fromkeys(x['run'] for x in tr):
 rows=[x for x in tr if x['run']==run];obsrun=root/'data/logs/MACHINES_PAPER/FIELDKIT_V1'/run
 fcus=[json.loads(l) for l in (obsrun/'observer.jsonl').open() if '"/mavros/state"' in l]
 nodes=[Node(),Node()];ev=[(x['receipt_wall_s'],0,x['payload']) for x in fcus]+[(x['stamp_s'],1,x) for x in rows]
 diffs=[];requests=[[],[]];counts=[collections.Counter(),collections.Counter()];episodes=[[],[]];current=[[],[]]
 for t,k,p in sorted(ev,key=lambda z:(z[0],z[1])):
  clock.now=t
  if k==0:
   for node in nodes:node._state_cb(NS(mode=p['mode'],armed=p['armed']))
   continue
  outputs=[]
  for i,field in enumerate(['recorded','replayed']):
   nodes[i]._fusion_cb(String(json.dumps(p[field])));o=nodes[i].published[-1];outputs.append(o)
   counts[i][o['action']]+=1
   if o['action'].startswith('HOLD_REQUESTED'):requests[i].append({'t':t,'action':o['action']})
   if o.get('fusion_state')=='CRITICAL' and o.get('hazard_type')=='person' and o.get('fcu_mode')=='AUTO':current[i].append({'t':t,'count':o.get('semantic_frame_count'),'action':o.get('action')})
   elif current[i]:episodes[i].append(current[i]);current[i]=[]
  bad={k:[outputs[0].get(k),outputs[1].get(k)] for k in ['action','semantic_frame_count','hold_requested'] if outputs[0].get(k)!=outputs[1].get(k)}
  if bad:diffs.append({'t':t,'differences':bad})
 for i in range(2):
  if current[i]:episodes[i].append(current[i])
 results.append({'run':run,'n':len(rows),'different_decisions':len(diffs),'differences':diffs,'requests_from_recorded_fusion':requests[0],'requests_from_replayed_fusion':requests[1],'person_episodes':[[{'n':len(e),'counters':[x['count'] for x in e],'first':e[0]['t'],'last':e[-1]['t']} for e in ee] for ee in episodes]})
(base/'report3_bridge_impact.json').write_text(json.dumps(results,indent=2))
for r in results:print(r['run'],r['different_decisions'],'requests',r['requests_from_recorded_fusion'],r['requests_from_replayed_fusion'],'episodes',r['person_episodes'])

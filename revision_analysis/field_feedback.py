from pathlib import Path
import json,sqlite3,struct,bisect
ROOT=Path(__file__).resolve().parents[1]/'revision_stage1_check';out=[]
def js(b):
 e='<' if b[1]&1 else '>';n=struct.unpack_from(e+'I',b,4)[0];return json.loads(b[8:8+n-1])
def state(b):
 e='<' if b[1]&1 else '>';n=struct.unpack_from(e+'I',b,12)[0];p=16+n; flags=b[p:p+4];p+=4;p=4+((p-4+3)//4)*4;n=struct.unpack_from(e+'I',b,p)[0];return {'connected':bool(flags[0]),'armed':bool(flags[1]),'mode':b[p+4:p+4+n-1].decode()}
for i,run in enumerate(sorted((ROOT/'data/logs/MACHINES_PAPER/FIELDKIT_V1').glob('MISSION*'))[:4],1):
 dest=next(ROOT.rglob('*_'+run.name+'_ANALYSIS_unpacked'));topics={k:[] for k in ['/safety_bridge/status','/mavros/state','/camera/perception/status']}
 for db in sorted(dest.rglob('*.db3')):
  c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
  for name,t,b in c.execute("select t.name,m.timestamp,m.data from messages m join topics t on t.id=m.topic_id where t.name in ('/safety_bridge/status','/mavros/state','/camera/perception/status')"):
   topics[name].append((t/1e9,state(b) if name=='/mavros/state' else js(b)))
  c.close()
 for a in topics.values():a.sort()
 last=''
 for t,p in topics['/safety_bridge/status']:
  action=p.get('action','')
  if action.startswith('HOLD_REQUESTED') and action!=last and p.get('fcu_mode')=='AUTO':
   holds=[(u,q) for u,q in topics['/mavros/state'] if t<=u<t+2 and q['mode']=='HOLD']; z={'test':i,'action':action,'request_receipt_s':t,'request_publication_s':p['stamp'],'first_HOLD_receipt_s':holds[0][0] if holds else None,'receipt_interval_ms':(holds[0][0]-t)*1000 if holds else None}
   if p.get('hazard_type')=='person':
    cams=[(u,q) for u,q in topics['/camera/perception/status'] if t-2<u<t+2]
    persons=[{'receipt':u,'source':q.get('sensor_stamp_s'),'range':o.get('distance_m'),'risk':o.get('risk')} for u,q in cams for o in q.get('objects',[]) if o['label']=='person' and isinstance(o.get('distance_m'),(int,float))];z['person_samples']=persons
   out.append(z)
  last=action
print(json.dumps(out,indent=2));Path(__file__).with_name('field_feedback.json').write_text(json.dumps(out,indent=2))

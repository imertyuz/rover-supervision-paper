"""Frozen bridge callback replay with ROS/service doubles; no live ROS traffic.
FCU updates use observer receipt time; Fusion uses bag receipt time.
Service is assumed available/accepted. Fixed recorded FCU feedback: not closed loop.
"""
from pathlib import Path
from types import SimpleNamespace as NS, ModuleType
import importlib.util,sys,json,sqlite3,struct,bisect
ROOT=Path(__file__).resolve().parents[1]/'revision_stage1_check'
clock=NS(now=0.0)
class String:
    def __init__(self,data=''):self.data=data
class Logger:
    def __getattr__(self,name):return lambda *a,**k:None
class Future:
    def add_done_callback(self,cb):cb(self)
    def result(self):return NS(mode_sent=True)
class Client:
    def wait_for_service(self,**kw):return True
    def call_async(self,req):return Future()
class Node:
    def __init__(self,*a):self.published=[]
    def create_subscription(self,*a):pass
    def create_publisher(self,*a):return NS(publish=lambda m:self.published.append(json.loads(m.data)))
    def create_client(self,*a):return Client()
    def get_logger(self):return Logger()
for name,attrs in {'rclpy':{},'rclpy.node':{'Node':Node},'std_msgs':{},'std_msgs.msg':{'String':String},'mavros_msgs':{},'mavros_msgs.msg':{'State':NS},'mavros_msgs.srv':{'SetMode':NS(Request=NS)}}.items():
    mod=ModuleType(name);mod.__dict__.update(attrs);sys.modules[name]=mod
source=ROOT/'rover_project/workspace_src/fusion_node/fusion_node/semantic_safety_bridge_v2.py'
spec=importlib.util.spec_from_file_location('frozen_bridge',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.time=NS(time=lambda:clock.now,monotonic=lambda:clock.now)
results=[]
for i,run in enumerate(sorted((ROOT/'data/logs/MACHINES_PAPER/FIELDKIT_V1').glob('MISSION*')),1):
    events=[];obs=[]
    for l in (run/'observer.jsonl').open():
        d=json.loads(l)
        if d['topic']=='/mavros/state':events.append((d['receipt_wall_s'],0,d['payload']))
    dest=next(ROOT.rglob('*_'+run.name+'_ANALYSIS_unpacked'))
    for db in dest.rglob('*.db3'):
        c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
        for name,t,data in c.execute("select t.name,m.timestamp,m.data from messages m join topics t on m.topic_id=t.id where t.name in ('/fusion/status_v4','/safety_bridge/status')"):
            en='<' if data[1]&1 else '>';n=struct.unpack_from(en+'I',data,4)[0];p=json.loads(data[8:8+n-1])
            if name=='/fusion/status_v4':events.append((t/1e9,1,p))
            else:obs.append(p)
        c.close()
    obs.sort(key=lambda p:p['stamp']);ts=[p['stamp'] for p in obs]
    node=mod.SemanticSafetyBridgeV2();compared=0;unpaired=0;diff=[];used=set()
    for t,kind,p in sorted(events,key=lambda e:(e[0],e[1])):
        clock.now=t
        if kind==0:node._state_cb(NS(mode=p['mode'],armed=p['armed']));continue
        node._fusion_cb(String(json.dumps(p)))
        actual=node.published[-1]
        # Pair only within 50 ms of bag arrival, checking the incoming Fusion signature.
        pos=bisect.bisect_left(ts,t);candidates=range(max(0,pos-3),min(len(obs),pos+4))
        candidates=[j for j in candidates if j not in used and abs(ts[j]-t)<0.05 and obs[j]['fusion_state']==p.get('state') and obs[j]['hazard_type']==p.get('hazard_type') and obs[j].get('reasons')==p.get('reasons')]
        if not candidates:unpaired+=1;continue
        j=min(candidates,key=lambda j:abs(ts[j]-t));used.add(j);compared+=1
        fields=['action','semantic_frame_count','fcu_mode','armed','hold_requested','camera_source_stamp_s']
        differences={k:[obs[j].get(k),actual.get(k)] for k in fields if obs[j].get(k)!=actual.get(k)}
        if differences:diff.append({'fusion_bag_receipt_s':t,'recorded_bridge_stamp_s':ts[j],'differences':differences})
    results.append({'run':run.name,'fusion_inputs':sum(k==1 for _,k,_ in events),'paired_snapshots':compared,'unpaired_inputs':unpaired,'different_snapshots':len(diff),'mismatch_examples':diff[:15]})
Path(__file__).with_name('bridge_conformance.json').write_text(json.dumps(results,indent=2))
for r in results:print(r['run'],{k:v for k,v in r.items() if k not in ['run','mismatch_examples']},r['mismatch_examples'][:2])

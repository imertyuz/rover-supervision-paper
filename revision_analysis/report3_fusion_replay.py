"""Frozen Fusion callback replay: bag receipt clocks, recorded evaluation stamps.
This checks trace conformance under approximate callback arrival times, not independent validation.
"""
from pathlib import Path
from types import SimpleNamespace as NS,ModuleType
import importlib.util,sys,json,sqlite3,struct,math,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'revision_stage1_check'
clock=NS(now=0.0)
class String:
    def __init__(self,data=''):self.data=data
class Logger:
    def __getattr__(self,n):return lambda *a,**k:None
class Node:
    def __init__(self,*a):self.published=[]
    def create_subscription(self,*a):pass
    def create_publisher(self,*a):return NS(publish=lambda m:self.published.append(json.loads(m.data)))
    def create_timer(self,*a):pass
    def get_logger(self):return Logger()
for name,attrs in {'rclpy':{},'rclpy.node':{'Node':Node},'rclpy.qos':{'qos_profile_sensor_data':None},'std_msgs':{},'std_msgs.msg':{'String':String,'Float32MultiArray':NS},'sensor_msgs':{},'sensor_msgs.msg':{'Imu':NS}}.items():
    mod=ModuleType(name);mod.__dict__.update(attrs);sys.modules[name]=mod
source=ROOT/'rover_project/workspace_src/fusion_node/fusion_node/fusion_v4.py'
spec=importlib.util.spec_from_file_location('frozen_fusion',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.time=NS(time=lambda:clock.now)

def unpack_json(b):
    en='<' if b[1]&1 else '>';n=struct.unpack_from(en+'I',b,4)[0];return json.loads(b[8:8+n-1])
def imu(b):
    en='<' if b[1]&1 else '>';sec,nano=struct.unpack_from(en+'iI',b,4);n=struct.unpack_from(en+'I',b,12)[0];p=16+n;p=4+((p-4+7)//8)*8
    x,y,z,w=struct.unpack_from(en+'dddd',b,p)
    if abs(x*x+y*y+z*z+w*w-1)>0.01:raise ValueError('Unexpected quaternion norm')
    return NS(header=NS(stamp=NS(sec=sec,nanosec=nano)),orientation=NS(x=x,y=y,z=z,w=w))
results=[]; traces=[]
windows={x['run']:x for x in json.loads(Path(__file__).with_name('stage1_audit.json').read_text())}
for run in sorted((ROOT/'data/logs/MACHINES_PAPER/FIELDKIT_V1').glob('MISSION*')):
    dest=next(ROOT.rglob('*_'+run.name+'_ANALYSIS_unpacked'));events=[]
    for db in dest.rglob('*.db3'):
        c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
        for name,t,b in c.execute("select t.name,m.timestamp,m.data from messages m join topics t on m.topic_id=t.id where t.name in ('/camera/perception/status','/lidar/sectors_status','/lidar/sectors','/mavros/imu/data','/fusion/status_v4')"):
            if name=='/mavros/imu/data':events.append((t/1e9,0,'_imu_cb',imu(b)))
            elif name=='/lidar/sectors':events.append((t/1e9,0,'_lidar_cb',NS()))
            else:
                p=unpack_json(b)
                if name=='/fusion/status_v4':events.append((p['stamp'],1,'_loop',p))
                else:events.append((t/1e9,0,'_camera_cb' if name=='/camera/perception/status' else '_lidar_status_cb',String(json.dumps(p))))
        c.close()
    clock.now=min(e[0] for e in events);node=mod.FusionNodeV4();total=0;diff=collections.Counter();examples=[]
    w=windows[run.name];start=w['observer_AUTO_start_s'];end=w['WP5_event_s']
    for t,kind,method,p in sorted(events,key=lambda e:(e[0],e[1])):
        clock.now=t
        if not kind:getattr(node,method)(p);continue
        node._loop();actual=node.published[-1]
        if start is None or end is None or not start<=t<=end:continue
        total+=1
        traces.append({'run':run.name,'stamp_s':t,'recorded':p,'replayed':actual})
        fields=['state','hazard_type','reasons','primary_reason','fresh']
        bad={k:[p.get(k),actual.get(k)] for k in fields if p.get(k)!=actual.get(k)}
        for k in bad:diff[k]+=1
        if bad:examples.append({'stamp_s':t,'differences':bad})
    results.append({'run':run.name,'mission_snapshots':total,'field_mismatches':dict(diff),'examples':examples})
    print(run.name,total,dict(diff),flush=True)
Path(__file__).with_name('report3_fusion_summary.json').write_text(json.dumps(results,indent=2))
Path(__file__).with_name('report3_fusion_traces.json').write_text(json.dumps(traces))

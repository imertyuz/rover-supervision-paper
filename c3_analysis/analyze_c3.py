from pathlib import Path
from types import SimpleNamespace as NS
import struct,sqlite3,json,math,collections
import numpy as np
ROOT=Path(__file__).resolve().parent/'input'
class CDR:
 def __init__(self,b):self.b=b;self.p=4;self.en='<' if b[1]&1 else '>'
 def unpack(self,fmt,align):
  self.p=4+((self.p-4+align-1)//align)*align
  v=struct.unpack_from(self.en+fmt,self.b,self.p);self.p+=struct.calcsize(fmt);return v[0] if len(v)==1 else v
 def string(self):
  n=self.unpack('I',4);v=self.b[self.p:self.p+n-1].decode();self.p+=n;return v
 def header(self):
  sec=self.unpack('i',4);nano=self.unpack('I',4);frame=self.string();return sec+nano*1e-9,frame

def read(db):
 out=collections.defaultdict(list);c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
 for name,typ,t,b in c.execute("select t.name,t.type,m.timestamp,m.data from messages m join topics t on t.id=m.topic_id where t.name in ('/test/event','/mavros/state','/mavros/local_position/velocity_local','/mavros/local_position/pose','/safety_bridge/status') order by m.timestamp"):
  r=CDR(b);receipt=t/1e9
  if typ=='std_msgs/msg/String':
   text=r.string()
   try:p=json.loads(text)
   except ValueError:p=text
   out[name].append({'receipt':receipt,'payload':p});continue
  stamp,frame=r.header()
  if typ=='mavros_msgs/msg/State':
   flags=r.unpack('BBBB',1);mode=r.string();out[name].append({'receipt':receipt,'stamp':stamp,'armed':bool(flags[1]),'mode':mode,'frame':frame})
  elif typ=='geometry_msgs/msg/TwistStamped':
   v=r.unpack('dddddd',8);out[name].append({'receipt':receipt,'stamp':stamp,'frame':frame,'vx':v[0],'vy':v[1],'vz':v[2]})
  elif typ=='geometry_msgs/msg/PoseStamped':
   x,y,z,qx,qy,qz,qw=r.unpack('ddddddd',8)
   if abs(qx*qx+qy*qy+qz*qz+qw*qw-1)>.02:raise ValueError('pose quaternion norm')
   yaw=math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz));out[name].append({'receipt':receipt,'stamp':stamp,'x':x,'y':y,'yaw':yaw,'frame':frame})
 c.close();return out

def integrate(t,v,start,end):
 sel=(t>start)&(t<end);tt=np.r_[start,t[sel],end];vv=np.maximum(np.interp(tt,t,v),0)
 return float(np.trapezoid(vv,tt)),float(np.max(np.diff(tt)))
report=[];inventory=[]
for root in sorted(ROOT.glob('*/field_logs/C3_STOP*')):
 db=next(root.rglob('*.db3'));d=read(db)
 events=d['/test/event'];triggers=[]
 for e in events:
  if isinstance(e['payload'],str) and e['payload'].startswith('STOP_TRIGGER'):
   if not triggers or e['receipt']-triggers[-1]['receipt']>0.1:triggers.append(e)
 states=d['/mavros/state'];transitions=[]
 for a,b in zip(states,states[1:]):
  if a['mode']!=b['mode']:transitions.append((b['receipt'],a['mode'],b['mode']))
 inventory.append({'run':root.name,'triggers':len(triggers),'state_transitions':transitions,'velocity_samples':len(d['/mavros/local_position/velocity_local']),'events':events})
 print('\nRUN',root.name,'triggers',len(triggers),'transitions',transitions)
 vel=d['/mavros/local_position/velocity_local'];pose=d['/mavros/local_position/pose']
 t=np.array([x['receipt'] for x in vel]);ts=np.array([x['stamp'] for x in vel]);vx=np.array([x['vx'] for x in vel]);vy=np.array([x['vy'] for x in vel])
 for n,e in enumerate(triggers,1):
  trigger=e['receipt'];holds=[x for x in states if x['mode']=='HOLD' and trigger<=x['receipt']<=trigger+2]
  if not holds:print('NO HOLD');continue
  hold=holds[0];h=hold['receipt'];before=[p for p in pose if p['receipt']<=trigger]
  if not before:print('NO POSE');continue
  yaw=before[-1]['yaw'];v=vx*math.cos(yaw)+vy*math.sin(yaw)
  # Define an analysis endpoint independently of the archived results: first <=0.10 sustained for >=0.5 s.
  candidates=np.flatnonzero((t>=h)&(t<h+5)&(np.abs(v)<.1));end=None
  for k in candidates:
   last=np.searchsorted(t,t[k]+.5)
   if last<len(t) and t[last]-t[k]>=.5 and np.max(np.abs(v[k:last+1]))<.1:end=t[k];break
  if end is None:print('NO SUSTAINED STOP');continue
  row={'run':root.name,'within_run_trial':n,'trigger_receipt_s':trigger,'hold_receipt_s':h,'hold_source_s':hold['stamp'],'trigger_to_hold_ms':(h-trigger)*1e3,'yaw_rad':yaw,'trigger_projected_speed_mps':float(np.interp(trigger,t,v)),'hold_projected_speed_mps':float(np.interp(h,t,v)),'stop_receipt_s':end,'hold_to_stop_s':end-h,'trigger_to_stop_distance_m':integrate(t,v,trigger,end)[0],'hold_to_stop_distance_m':integrate(t,v,h,end)[0],'max_integration_gap_s':integrate(t,v,h,end)[1], 'velocity_source_age_range_s':[float(np.min(t-ts)),float(np.max(t-ts))]}
  # Diagnostic reproduction of historical forward-only first sign crossing, using either origin.
  for originlabel,origin in [('trigger',trigger),('hold',h)]:
   idx=np.flatnonzero((t>=origin)&(t<origin+5)&(v<=0))
   if len(idx):
    k=int(idx[0]);cross=t[k-1]+(t[k]-t[k-1])*v[k-1]/(v[k-1]-v[k]) if k and v[k-1]>0 else t[k]
    row[originlabel+'_to_first_nonpositive_m']=integrate(t,v,origin,cross)[0]
    row[originlabel+'_to_first_nonpositive_s']=float(cross-origin)
   sub=(t>=origin)&(t<=origin+2)
   row[originlabel+'_2s_positive_integral_m']=integrate(t,v,origin,origin+2)[0]
  report.append(row);print(json.dumps(row))
Path(__file__).with_name('c3_recomputed.json').write_text(json.dumps(report,indent=2))
Path(__file__).with_name('c3_inventory.json').write_text(json.dumps(inventory,indent=2))

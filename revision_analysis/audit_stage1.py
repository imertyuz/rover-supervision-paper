"""Read-only ROS 2 SQLite/observer audit. No ROS nodes or vehicle services."""
from pathlib import Path
import sqlite3,struct,json,collections,bisect
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'revision_stage1_check'
BASE=ROOT/'data/logs/MACHINES_PAPER/FIELDKIT_V1'

def bag_json(dest):
    out=collections.defaultdict(list)
    for db in dest.rglob('*.db3'):
        con=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
        for topic,t,data in con.execute("select t.name,m.timestamp,m.data from messages m join topics t on m.topic_id=t.id where t.type in ('std_msgs/msg/String','std_msgs/String')"):
            endian='<' if data[1]&1 else '>'
            length=struct.unpack_from(endian+'I',data,4)[0]
            try: p=json.loads(data[8:8+length-1].decode())
            except (ValueError,UnicodeDecodeError):continue
            out[topic].append((t/1e9,p))
        con.close()
    for v in out.values():v.sort(key=lambda x:x[0])
    return out

def pct(vals):
    return dict(zip(['n','median','p95','p99','max'],[len(vals),*map(float,np.percentile(vals,[50,95,99,100]))])) if vals else {'n':0}

report=[]
for i,run in enumerate(sorted(BASE.glob('MISSION*')),1):
    name=run.name
    dest=next(ROOT.rglob('*_'+name+'_ANALYSIS_unpacked'))
    bag=bag_json(dest)
    obs=collections.defaultdict(list)
    for l in (run/'observer.jsonl').open():
        d=json.loads(l);obs[d['topic']].append(d)
    events=[json.loads(l) for l in (run/'run_evidence/events.jsonl').open()]
    autos=[d['receipt_wall_s'] for d in obs['/mavros/state'] if d['payload'].get('armed') and d['payload'].get('mode')=='AUTO']
    complete=[d['wall_s'] for d in events if d['event']=='WAYPOINT_REACHED_5']
    start=min(autos) if autos else None;end=min(complete) if complete else None
    # Observer-receipt window is diagnostic, not the publication's DataFlash clock alignment.
    window=lambda t:start is not None and end is not None and start<=t<=end
    result={'run':name,'label':f'T{i}' if i<=4 else 'extra_disarmed_recording','observer_AUTO_start_s':start,'WP5_event_s':end,'observer_duration_s':end-start if start and end else None,'states':sorted(set((d['payload'].get('armed'),d['payload'].get('mode')) for d in obs['/mavros/state'])),'missing':{},'episodes':[]}
    for topic in ['/camera/perception/status','/fusion/status_v4','/safety_bridge/status','/lidar/navigation_status']:
        # Match exact published stamp or NAV sequence, never rounded.
        field='sequence' if topic=='/lidar/navigation_status' else 'stamp'
        ids={p.get(field) for t,p in bag[topic]}
        missing=[d for d in obs[topic] if d['payload'].get(field) not in ids]
        result['missing'][topic]={'all_observer':len(missing),'mission_observer_receipt':sum(window(d['receipt_wall_s']) for d in missing),'min_receipt_s':min((d['receipt_wall_s'] for d in missing),default=None),'max_receipt_s':max((d['receipt_wall_s'] for d in missing),default=None), 'measure_phase_missing':sum(d.get('phase')=='MEASURE' for d in missing), 'measure_first_missing_s':min((d['receipt_wall_s'] for d in missing if d.get('phase')=='MEASURE'),default=None)}
    bridge=[(t,p) for t,p in bag['/safety_bridge/status'] if window(p.get('stamp',t))]
    # Consecutive eligible person-CRITICAL bridge snapshots, broken by every other snapshot.
    episodes=[];current=[]
    for t,p in bridge:
        eligible=p.get('armed') and p.get('fcu_mode')=='AUTO' and p.get('fusion_state')=='CRITICAL' and p.get('hazard_type')=='person' and p.get('camera_fresh') and (p.get('camera_source_stamp_s') or 0)>0
        if eligible:current.append((t,p))
        elif current:episodes.append(current);current=[]
    if current:episodes.append(current)
    cameras=bag['/camera/perception/status']
    for ep in episodes:
        stamps=[]
        for t,p in ep:
            s=p['camera_source_stamp_s']
            if not stamps or abs(s-stamps[-1])>1e-6:stamps.append(s)
        matches=[]
        for s in stamps:
            candidates=[(t,p) for t,p in cameras if p.get('yolo_sensor_stamp_s') is not None and abs(p['yolo_sensor_stamp_s']-s)<=0.000501]
            unique={p['yolo_sensor_stamp_s']:(t,p) for t,p in candidates}
            matches.append({'bridge_source_s':s,'camera_candidates':[{'source_s':ss,'publication_s':p.get('stamp'),'person_tracks':p.get('person_tracks'),'valid':p.get('valid')} for ss,(t,p) in unique.items()]})
        fusion_pairs=[(t,p) for t,p in bag['/fusion/status_v4'] if p.get('state')=='CRITICAL' and p.get('hazard_type')=='person' and ep[0][1]['stamp']-0.05<=p.get('stamp',t)<=ep[-1][1]['stamp']+0.01]
        req=[p['stamp'] for t,p in ep if str(p.get('action','')).startswith('HOLD_REQUESTED')]
        result['episodes'].append({'first_bridge_s':ep[0][1]['stamp'],'last_bridge_s':ep[-1][1]['stamp'],'snapshots':len(ep),'source_transitions':len(stamps),'counters':[p.get('semantic_frame_count') for t,p in ep],'actions':[p.get('action') for t,p in ep],'request_s':req,'sources':matches,'person_critical_fusion_snapshots':len(fusion_pairs),'first_source_to_request_s':(req[0]-matches[0]['camera_candidates'][0]['source_s']) if req and matches[0]['camera_candidates'] else None})
    armedauto=[p for t,p in bridge if p.get('armed') and p.get('fcu_mode')=='AUTO']
    source_groups=[]
    for p in armedauto:
        s=p.get('camera_source_stamp_s')
        if not source_groups or s!=source_groups[-1][0]:source_groups.append([s,1])
        else:source_groups[-1][1]+=1
    # This broad cohort is diagnostic: later match paper's precise freshness/eligibility cohort.
    camera=[p for t,p in cameras if window(p.get('stamp',t))]
    result['camera_publish_age_s']=pct([p['capture_age_at_publish_s'] for p in camera if p.get('capture_age_at_publish_s') is not None])
    result['ambiguity_publications']=sum(bool(p.get('semantic_conflict_active')) for p in camera)
    result['ambiguity_distances_m']=[c.get('person_distance_m') for p in camera for c in p.get('semantic_conflicts',[])]
    report.append(result)
output=Path(__file__).parent/'stage1_audit.json';output.write_text(json.dumps(report,indent=2))
for r in report:
    print(r['label'],r['run'],'duration',r['observer_duration_s'],'missing',r['missing'],'episodes',[(e['snapshots'],e['source_transitions'],e['counters'],e['actions']) for e in r['episodes']],'ambiguity',r['ambiguity_publications'],'camera_age',r['camera_publish_age_s'])

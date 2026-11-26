from pathlib import Path
import json,sqlite3,struct,collections,bisect,ast
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parent/'revision_stage1_check'
ns={'__file__':str(BASE.parent/'c3_analysis/analyze_c3.py')};exec((BASE.parent/'c3_analysis/analyze_c3.py').read_text().split('report=[];inventory=[]')[0],ns)
model=BASE.parent/'revision_work/analysis/analysis_scripts/model_validation.py';tree=ast.parse(model.read_text());defs=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Import,ast.ImportFrom)) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MISSION_S' for t in n.targets))];m={};exec(compile(ast.Module(body=defs,type_ignores=[]),str(model),'exec'),m)
windows=json.loads((BASE/'stage1_audit.json').read_text()); runs=[]
for i,w in enumerate(windows[:4],1):
 dest=next(ROOT.rglob('*_'+w['run']+'_ANALYSIS_unpacked'));bag=collections.defaultdict(list);state=[]
 for db in sorted(dest.rglob('*.db3')):
  state+=ns['read'](db)['/mavros/state']
  c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
  for name,t,b in c.execute("select t.name,m.timestamp,m.data from messages m join topics t on t.id=m.topic_id where t.name in ('/camera/perception/status','/fusion/status_v4','/safety_bridge/status')"):
   en='<' if b[1]&1 else '>';n=struct.unpack_from(en+'I',b,4)[0];bag[name].append((t/1e9,json.loads(b[8:8+n-1])))
  c.close()
 for v in bag.values():v.sort(key=lambda z:z[0])
 state.sort(key=lambda z:z['receipt']);ok=m['armed_auto_selector']([(z['receipt'],z['mode'],z['armed']) for z in state],f'T{i}')
 bb=[(t,p) for t,p in bag['/safety_bridge/status'] if ok(t) and p.get('camera_source_stamp_s')];cams=[(t,p) for t,p in bag['/camera/perception/status'] if ok(t)]
 hist=collections.Counter(collections.Counter(round(p['camera_source_stamp_s'],3) for t,p in bb).values())
 print(i,'nbridge',len(bb),'cam',len(cams),'hist',hist)
 runs.append({'test':f'T{i}','run':w['run'],'bag':dict(bag),'state':state,'window':w,'hist':dict(hist)})
import pickle
pickle.dump(runs,open(Path(__file__).with_name('loaded_data.pkl'),'wb'))

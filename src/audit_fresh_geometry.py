"""Fixed-ID representation and parameter geometry audit for fresh checkpoints."""
import json, statistics
from pathlib import Path
import torch
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from run_stage1_r22 import paths as r22_paths, out as r22_out, read_rows

ROOT=Path('.'); OUT=r22_out(ROOT)/'fresh_lineage/statistics'; OUT.mkdir(parents=True,exist_ok=True)
ids=[]
for i,r in enumerate(read_rows(r22_paths(ROOT)['candidate_pools'])):
 if i>=24: break
 ids += [str(r['query_id'])] + [str(x) for x in r['positive_target_ids'][:2]]
ids=list(dict.fromkeys(ids))
store=FeatureStore.from_path(r22_paths(ROOT)['features'],cache_size=256)
def summarize(ck):
 m=load_student(ck,torch.device('cpu')).eval(); x=torch.stack([store.embedding_features(i).embedding for i in ids]); out={}
 for typ,role in [('table','query'),('table','target')]:
  z=m.project(x,typ,role=role); norms=z.norm(dim=1); cov=z-z.mean(0); s=torch.linalg.svdvals(cov); p=(s*s)/(s*s).sum(); er=float(1/(p*p).sum())
  out[f'{typ}_{role}']={'norm_p10':float(torch.quantile(norms,.1)),'norm_median':float(torch.quantile(norms,.5)),'norm_p90':float(torch.quantile(norms,.9)),'norm_max':float(norms.max()),'mean_norm':float(z.mean(0).norm()),'effective_rank':er}
 out['parameters']={k:{'norm':float(v.float().norm()),'mean':float(v.float().mean()),'std':float(v.float().std())} for k,v in m.state_dict().items() if v.ndim>=2}
 return out
paths={'B13':r22_paths(ROOT)['b13']}
for arm in ('F1-A-SUP','F1-A-T0','F1-A-T1','S1','S1-aug'):
 for s in (13,29): paths[f'{arm}_seed{s}']=r22_out(ROOT)/f'fresh_lineage/{arm}/seed{s}/edge/checkpoints/step_001318.pt'
result={k:summarize(v) for k,v in paths.items() if Path(v).exists()}
(OUT/'geometry_audit.json').write_text(json.dumps({'status':'complete_fixed_calibration_ids','ids':ids,'models':result},indent=2))
print(json.dumps({'models':list(result),'calibration_ids':len(ids)}))

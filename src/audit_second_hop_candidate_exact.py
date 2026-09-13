"""Exact-score the ANN second-hop candidate sets retained in detailed paths."""
import argparse, gzip, json, statistics
from pathlib import Path
import torch
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from run_stage1_r22 import paths as r22_paths, out as r22_out

ROOT=Path(__file__).resolve().parents[1]
@torch.inference_mode()
def run(arm,seed,device_name):
 device=torch.device(device_name); store=FeatureStore.from_path(r22_paths(ROOT)['features'],cache_size=50000)
 p=r22_out(ROOT)/f'fresh_lineage/{arm}/seed{seed}/full_lake/step_001318/detailed_paths.jsonl.gz'; pairs={}
 with gzip.open(p,'rt') as f:
  for l in f:
   r=json.loads(l)
   for target in r['evidence']:
    tid=str(target['target_id'])
    for path in target.get('paths',[]):
     if path.get('kind')=='evidence': pairs[(str(path['evidence_id']),tid,str(path.get('evidence_type','text')))] = float(path.get('evidence_target_score',0.0))
 ck=r22_out(ROOT)/f'fresh_lineage/{arm}/seed{seed}/edge/checkpoints/step_001318.pt'; model=load_student(ck,device).eval(); keys=list(pairs); ann=[]; exact=[]
 for st in range(0,len(keys),2048):
  part=keys[st:st+2048]; src=torch.stack([store.embedding_features(k[0]).embedding for k in part]).to(device); dst=torch.stack([store.embedding_features(k[1]).embedding for k in part]).to(device)
  vals=[]
  for typ in ('text','image'):
   idx=[i for i,k in enumerate(part) if k[2]==typ]
   if idx: vals.append((idx,model.raw_score_embeddings(src[idx],typ,dst[idx],'table').detach().cpu().tolist()))
  exact_map={i:v for idx,vs in vals for i,v in zip(idx,vs)}
  ann.extend(pairs[k] for k in part); exact.extend(exact_map[i] for i in range(len(part)))
 # Pearson and ranking agreement over candidate sets (global score order proxy).
 mean_a=statistics.fmean(ann); mean_e=statistics.fmean(exact); cov=sum((a-mean_a)*(e-mean_e) for a,e in zip(ann,exact)); va=sum((a-mean_a)**2 for a in ann); ve=sum((e-mean_e)**2 for e in exact); corr=cov/(va*ve)**0.5 if va and ve else 0.0
 result={'status':'complete_candidate_set_exact','arm':arm,'seed':seed,'pairs':len(keys),'ann_exact_pearson':corr,'ann_score_mean':mean_a,'exact_score_mean':mean_e,'note':'Exact scores are computed on retained ANN second-hop candidates, not a full-corpus exact top-k search.'}
 out=r22_out(ROOT)/'fresh_lineage/statistics'; out.mkdir(parents=True,exist_ok=True); (out/f'second_hop_candidate_exact_{arm}_seed{seed}.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--arm',required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--device',default='cuda:0'); a=ap.parse_args(); run(a.arm,a.seed,a.device)

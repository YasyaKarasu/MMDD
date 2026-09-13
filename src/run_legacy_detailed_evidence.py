"""Retain first/second-hop paths for archived R21 checkpoints."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import StudentANNIndices, retrieve_zero_one_hop_detailed_many
from run_stage1_r21 import paths as r21_paths, out as r21_out, read_rows, write_rows

ROOT=Path(__file__).resolve().parents[1]
@torch.inference_mode()
def run(arm,seed,device_name,step=1318,checkpoint=None,index_arm=None):
 device=torch.device(device_name); ps=r21_paths(ROOT); store=FeatureStore.from_path(ps['features'],cache_size=100000)
 ck=Path(checkpoint) if checkpoint else r21_out(ROOT)/arm/f'seed{seed}'/'checkpoints'/f'step_{step:06d}.pt'; idx=r21_out(ROOT)/'indexes'/(index_arm or arm)/f'seed{seed}'/f'step_{step:06d}'
 model=load_student(ck,device).eval(); indices=StudentANNIndices(model,store,idx,device=device,checkpoint_sha256=checkpoint_fingerprint(ck),corpus_sha256=checkpoint_fingerprint(ps['corpus']))
 pools=list(read_rows(ps['candidate_pools'])); detailed=retrieve_zero_one_hop_detailed_many([str(r['query_id']) for r in pools],indices,k=100,direct_k=100,evidence_k=20,targets_per_evidence=20,query_batch_size=16)
 out=[]
 for pool,ret in zip(pools,detailed):
  compact=lambda xs:[{'target_id':str(v['target_id']),'score':float(v.get('score') or 0.0),'paths':v.get('paths',[])} for v in xs]
  direct,evidence,fused=ret['direct'],ret['evidence'],ret['fused']; out.append({'query_id':str(pool['query_id']),'query_kind':pool['query_kind'],'positive_target_ids':pool['positive_target_ids'],'direct':compact(direct),'evidence':compact(evidence),'direct_ann':[str(v['target_id']) for v in direct],'evidence_ann':[str(v['target_id']) for v in evidence],'U':list(dict.fromkeys([str(v['target_id']) for v in direct]+[str(v['target_id']) for v in evidence])),'u_exact_ranking':[str(v['target_id']) for v in fused]})
 dest=r21_out(ROOT)/'detailed_paths'/arm/f'seed{seed}_step{step:06d}'/'detailed_paths.jsonl.gz'; write_rows(dest,out); write_json(dest.with_name('summary.json'),{'status':'complete','arm':arm,'seed':seed,'queries':len(out),'checkpoint_sha256':checkpoint_fingerprint(ck)})
 print(json.dumps({'status':'complete','arm':arm,'seed':seed,'queries':len(out)}))
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--arm',required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--checkpoint'); ap.add_argument('--index-arm'); a=ap.parse_args(); run(a.arm,a.seed,a.device,checkpoint=a.checkpoint,index_arm=a.index_arm)

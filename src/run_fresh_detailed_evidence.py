"""Re-run fresh full-lake retrieval while retaining first/second-hop paths."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
import torch
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import StudentANNIndices, build_indices, load_corpus_ids, retrieve_zero_one_hop_detailed_many
from run_stage1_r21 import paths as r21_paths, out as r21_out, read_rows, write_rows
from run_stage1_r22 import paths as r22_paths, out as r22_out

ROOT=Path(__file__).resolve().parents[1]
def run(arm,seed,step,device_name,checkpoint=None,index_dir=None,output_dir=None):
 device=torch.device(device_name); ps=r21_paths(ROOT); store=FeatureStore.from_path(ps['features'],cache_size=100000)
 ck=Path(checkpoint) if checkpoint else r22_out(ROOT)/'fresh_lineage'/arm/f'seed{seed}'/'edge'/'checkpoints'/f'step_{step:06d}.pt'; model=load_student(ck,device).eval()
 ids_by_type=load_corpus_ids(ps['corpus'],store); idxdir=Path(index_dir) if index_dir else r22_out(ROOT)/'fresh_lineage'/arm/f'seed{seed}'/'indexes'/f'step_{step:06d}';
 if not (idxdir/'manifest.json').is_file() or not (idxdir/'image.hnsw').is_file() or not (idxdir/'text.hnsw').is_file(): build_indices(model,store,ids_by_type,idxdir,device=device,checkpoint_sha256=checkpoint_fingerprint(ck),corpus_sha256=checkpoint_fingerprint(ps['corpus']),batch_size=4096)
 indices=StudentANNIndices(model,store,idxdir,device=device,checkpoint_sha256=checkpoint_fingerprint(ck),corpus_sha256=checkpoint_fingerprint(ps['corpus']))
 pools=list(read_rows(r22_paths(ROOT)['candidate_pools'])); qids=[str(r['query_id']) for r in pools]
 detailed=retrieve_zero_one_hop_detailed_many(qids,indices,k=100,direct_k=100,evidence_k=20,targets_per_evidence=20,query_batch_size=16)
 out=[]
 for pool,ret in zip(pools,detailed):
  pos=set(map(str,pool['positive_target_ids'])); direct=ret['direct']; evidence=ret['evidence']; fused=ret['fused']; union=list(dict.fromkeys([str(x['target_id']) for x in direct]+[str(x['target_id']) for x in evidence])); ranking=[str(x['target_id']) for x in fused]
  compact=lambda x:[{'target_id':str(v['target_id']),'score':float(v.get('score') if v.get('score') is not None else (v.get('direct_score') or 0.0)),'paths':v.get('paths',[])} for v in x]
  out.append({'query_id':str(pool['query_id']),'query_kind':pool['query_kind'],'positive_target_ids':sorted(pos),'direct':compact(direct),'evidence':compact(evidence),'direct_ann':[str(v['target_id']) for v in direct],'evidence_ann':[str(v['target_id']) for v in evidence],'U':union,'u_exact_ranking':ranking})
 dest=(Path(output_dir)/'detailed_paths.jsonl.gz') if output_dir else r22_out(ROOT)/'fresh_lineage'/arm/f'seed{seed}'/'full_lake'/f'step_{step:06d}'/'detailed_paths.jsonl.gz'; write_rows(dest,out)
 write_json(dest.with_name('detailed_paths_summary.json'),{'status':'complete','arm':arm,'seed':seed,'queries':len(out),'checkpoint_sha256':checkpoint_fingerprint(ck),'note':'Retains first-hop evidence IDs and second-hop target path scores.'})
 return {'status':'complete','arm':arm,'seed':seed,'queries':len(out)}
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--arm',required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--step',type=int,default=1318); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--checkpoint'); ap.add_argument('--index-dir'); ap.add_argument('--output-dir'); a=ap.parse_args(); print(json.dumps(run(a.arm,a.seed,a.step,a.device,a.checkpoint,a.index_dir,a.output_dir)))

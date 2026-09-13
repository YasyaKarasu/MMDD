"""Full-corpus exact second-hop audit for retained S1/S1-aug evidence nodes."""
from __future__ import annotations
import argparse, collections, gzip, json, statistics
from pathlib import Path
import torch
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from run_stage1_r21 import paths as r21_paths, out as r21_out, read_rows, write_rows
from run_stage1_r22 import out as r22_out

ROOT=Path(__file__).resolve().parents[1]
@torch.inference_mode()
def run(arm,seed,device_name,detailed_path=None,checkpoint_path=None,output_tag=None):
 device=torch.device(device_name); store=FeatureStore.from_path(r21_paths(ROOT)['features'],cache_size=100000)
 detailed=Path(detailed_path) if detailed_path else r22_out(ROOT)/f'fresh_lineage/{arm}/seed{seed}/full_lake/step_001318/detailed_paths.jsonl.gz'; evidence=collections.defaultdict(set)
 with gzip.open(detailed,'rt') as f:
  for l in f:
   r=json.loads(l)
   for t in r['evidence']:
    for p in t.get('paths',[]):
     if p.get('kind')=='evidence': evidence[(str(p['evidence_id']),str(p.get('evidence_type','text')))].add(str(t['target_id']))
 ps=r21_paths(ROOT); ck=Path(checkpoint_path) if checkpoint_path else r22_out(ROOT)/f'fresh_lineage/{arm}/seed{seed}/edge/checkpoints/step_001318.pt'; model=load_student(ck,device).eval(); table_ids=json.loads((r21_out(ROOT)/'indexes'/'Qwen-Raw'/'table_ids.json').read_text())
 target_emb=torch.stack([store.embedding_features(str(x)).embedding for x in table_ids]).to(device); target_vec=model.project(target_emb,'table',role='target'); results=[]
 for typ in ('text','image'):
  keys=[k for k in evidence if k[1]==typ]
  for st in range(0,len(keys),64):
   part=keys[st:st+64]; src=torch.stack([store.embedding_features(k[0]).embedding for k in part]).to(device); src_vec=model.project(src,typ,role='evidence'); rel=model.relations[model.relation_key(typ,'table')]; scores=(src_vec@rel)@target_vec.T; vals,idx=scores.topk(k=20,dim=1)
   for k,vv,ii in zip(part,vals.cpu(),idx.cpu()):
    exact=[str(table_ids[int(i)]) for i in ii]; ann=evidence[k]; results.append({'evidence_id':k[0],'evidence_type':k[1],'exact_top20':exact,'ann_candidate_union':sorted(ann),'candidate_overlap':len(set(exact)&ann)/20,'candidate_recall':len(set(exact)&ann)/len(ann) if ann else 0.0})
 out=r22_out(ROOT)/'fresh_lineage/statistics'; out.mkdir(parents=True,exist_ok=True); tag=output_tag or arm; path=out/f'second_hop_full_exact_{tag}_seed{seed}.jsonl.gz'; write_rows(path,results)
 summary={'status':'complete_full_corpus_exact_second_hop','arm':arm,'seed':seed,'evidence_nodes':len(results),'mean_exact_top20_overlap_with_ann_candidates':statistics.fmean(r['candidate_overlap'] for r in results),'mean_ann_candidate_recall_of_exact_top20':statistics.fmean(r['candidate_recall'] for r in results),'rankings':str(path.resolve()),'note':'For each retained evidence node, exact top-20 scored against all corpus tables.'}
 (out/f'second_hop_full_exact_{tag}_seed{seed}.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary))
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--arm',required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--detailed'); ap.add_argument('--checkpoint'); ap.add_argument('--output-tag'); a=ap.parse_args(); run(a.arm,a.seed,a.device,a.detailed,a.checkpoint,a.output_tag)

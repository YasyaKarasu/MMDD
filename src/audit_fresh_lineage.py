"""Read-only teacher quality and retrieval concentration audit."""
import argparse, collections, gzip, json, math, statistics
from pathlib import Path

def teacher(root, stage, seed):
 p=root/f"fresh_lineage/{stage}/seed{seed}/teacher_soft_scores.jsonl.gz"; m={}
 with gzip.open(p,'rt') as f:
  for l in f:
   r=json.loads(l); m[r['query_id']]=r
 vals={k:[] for k in ('raw','r10','r20','r50')}; n=0
 with (root/'manifests/full_natural.jsonl').open() as f:
  for l in f:
   e=json.loads(l)
   if e.get('split')!='train' or e.get('source_type')!='table' or e.get('destination_type')!='table': continue
   r=m.get(e['query_id']); pos=set(map(str,e.get('positive_ids',[])))
   if not r or not pos: continue
   cand=list(map(str,r['candidate_ids'])); rank=[cand[i] for i in sorted(range(len(cand)),key=lambda i:(-r['scores'][i],cand[i]))]
   vals['raw'].append(len(pos&set(cand))/len(pos)); vals['r10'].append(len(pos&set(rank[:10]))/len(pos)); vals['r20'].append(len(pos&set(rank[:20]))/len(pos)); vals['r50'].append(len(pos&set(rank[:50]))/len(pos)); n+=1
 return {'queries':n,**{k:statistics.fmean(v) for k,v in vals.items()}}

def concentration(root, arm, seed):
 p=root/f'fresh_lineage/{arm}/seed{seed}/full_lake/step_001318/rankings.jsonl.gz'; c=collections.Counter(); n=0
 with gzip.open(p,'rt') as f:
  for l in f: r=json.loads(l); c.update(r['U']); n+=1
 t=sum(c.values()); probs=[v/t for v in c.values()]
 return {'queries':n,'unique_targets':len(c),'hhi':sum(x*x for x in probs),'entropy':-sum(x*math.log(x) for x in probs),'universal_target_count':sum(v==n for v in c.values())}

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('work/stage1_optimization_r22_20260911')); a=ap.parse_args(); out={'teacher':{},'concentration':{}}
 for st in ('T0','T1-A','T1-B'):
  for s in (13,29): out['teacher'][f'{st}_seed{s}']=teacher(a.root,st,s)
 for arm in ('F1-A-SUP','F1-A-T0','F1-A-T1','S1','S1-aug'):
  for s in (13,29): out['concentration'][f'{arm}_seed{s}']=concentration(a.root,arm,s)
 (a.root/'fresh_lineage/statistics').mkdir(parents=True,exist_ok=True); (a.root/'fresh_lineage/statistics/quality_hubness_audit.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

import gzip,json,math,statistics
from pathlib import Path
from collections import Counter,defaultdict
R=Path('/home/oycy/MMDD/work/stage1_optimization_r26_20260914'); out=R/'statistics/lse_lme_posthoc';out.mkdir(parents=True,exist_ok=True)
models=['B13','R25-B13-FULL/seed13/step178','R25-B13-FULL/seed29/step178','R26-O-NATIVE/seed13/step178','R26-O-NATIVE/seed29/step178','R26-O-SUP/seed13/step178','R26-O-SUP/seed29/step178','R26-O-LSE-QTKD/seed13/step178','R26-O-LSE-QTKD/seed29/step178','R26-O-UQTKD/seed13/step178','R26-O-UQTKD/seed29/step178']
def lse(v,t=1):
 m=max(v);return m+t*math.log(sum(math.exp((x-m)/t) for x in v))
def rank_e(rows,mode):
 vals=[]
 for r in rows:
  ps=[float(p['path_score']) for p in r.get('paths',[]) if p.get('kind')=='evidence']
  if ps:
   s=lse(ps)-(math.log(len(ps)) if mode=='LME' else 0)
  else:s=float('-inf')
  vals.append((s,str(r['target_id']),len(ps)))
 return [x[1] for x in sorted(vals,key=lambda x:(-x[0],x[1]))],vals
def rrf(d,e):
 s=defaultdict(float)
 for i,t in enumerate(d,1):s[t]+=1/(60+i)
 for i,t in enumerate(e,1):s[t]+=1/(60+i)
 return [t for t,_ in sorted(s.items(),key=lambda x:(-x[1],x[0]))]
def metric(rank,pos,k):return len(set(rank[:k])&set(pos))/len(pos) if pos else 0
def pearson(a,b):
 if len(a)<2:return None
 ma=sum(a)/len(a);mb=sum(b)/len(b); den=(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))**.5
 return (sum((x-ma)*(y-mb) for x,y in zip(a,b))/den) if den else None
summary=[]; allrows=[]
for model in models:
 p=R/'rankings'/model/'rankings.jsonl.gz';
 if not p.exists(): continue
 rows=[]
 with gzip.open(p,'rt') as f:
  for line in f: rows.append(json.loads(line))
 stats={m:{k:[] for k in ('R10','R20','R50')} for m in ('LSE','LME','Direct','Fused_LSE','Fused_LME')}; counts=[]; deltas=[]; wins=loss=tie=0; strict={x:[0,0] for x in ('ann','exact')}
 for row in rows:
  pos=set(map(str,row['positive_target_ids'])); d=[str(x['target_id']) for x in row.get('D100_ANN',[])]; dx=[str(x) for x in row.get('D100_EXACT',[])]; er=row.get('E_pre_retention',[])
  rl,vl=rank_e(er,'LSE'); rm,vm=rank_e(er,'LME'); fl=rrf(d,rl);fm=rrf(d,rm)
  n=sum(x[2] for x in vl); counts.extend([x[2] for x in vl]); deltas.append((statistics.mean([x[0] for x in vl]) if vl else 0, min([rl.index(x) for x in pos if x in rl] or [10000]), min([rm.index(x) for x in pos if x in rm] or [10000])))
  for name,rk in [('LSE',rl),('LME',rm),('Direct',d),('Fused_LSE',fl),('Fused_LME',fm)]:
   for k,key in [(10,'R10'),(20,'R20'),(50,'R50')]:stats[name][key].append(metric(rk,pos,k))
  for label,dd in [('ann',d),('exact',dx)]:
   strict[label][0]+=sum(1 for t in pos if t not in set(dd) and t in set(rl)); strict[label][1]+=len(pos)
  a=next(iter(pos),None); il=min([rl.index(x) for x in pos if x in rl] or [10000]); im=min([rm.index(x) for x in pos if x in rm] or [10000])
  if im<il:wins+=1
  elif im>il:loss+=1
  else:tie+=1
  allrows.append({'model':model,'query_id':row['query_id'],'path_count':n,'LSE_rank':il+1 if il<10000 else None,'LME_rank':im+1 if im<10000 else None,'rank_delta_LME_minus_LSE':(im-il if il<10000 and im<10000 else None)})
 for name in stats:
  for k in stats[name]:stats[name][k]=sum(stats[name][k])/len(rows)
 corr=pearson(counts,[x[2]-x[1] for x in deltas])
 summary.append({'model':model,'queries':len(rows),'path_count_mean':sum(counts)/len(counts),'path_count_median':statistics.median(counts),'path_count_dist':dict(sorted(Counter(counts).items())),'metrics':stats,'positive_first_rank_W_L_T':{'LME_better':wins,'LSE_better':loss,'tie':tie},'path_count_vs_positive_rank_delta_pearson':corr,'strict_evidence_only_positive_pairs':strict})
json.dump({'models':summary,'method':'same frozen E_pre_retention paths; LME=LSE-log(n); RRF k=60; strict EO defined as qrel target in E ranking and outside D100 ANN/exact'},open(out/'summary.json','w'),indent=2)
with gzip.open(out/'per_query.jsonl.gz','wt') as f:
 for x in allrows:f.write(json.dumps(x)+'\n')
print(json.dumps({'models':len(summary),'queries':sum(x['queries'] for x in summary),'out':str(out)}))

import os
from pathlib import Path
import json, collections, hashlib
import numpy as np
ROOT=Path(os.environ.get('MMDD_AUDIT_ROOT', '/mnt/data/audit_review_20260918/mmdd_s1_audit_20260918'))
OUT=Path(__file__).parent
load=lambda p:json.loads((ROOT/p).read_text())
rows=lambda p:[json.loads(s) for s in (ROOT/p).read_text().splitlines() if s.strip()]
pop={r['query_id']:r for r in rows('data/raw_gt/dev.population.jsonl')}
raw={r['query_id']:r for r in rows('data/raw_retrieval/dev/query_rankings.jsonl')}
orders=load('data/teacher_health/fixed_pool_orderings.json')
qs=sorted(pop);methods=list(orders[qs[0]])
G={q:set(pop[q]['positive_target_ids']) for q in qs}
I={q:set(pop[q]['implicit_target_ids']) for q in qs}
D={q:set(raw[q]['target_ids'][:100]) for q in qs}
C={q:set(orders[q]['admission']) for q in qs}
for q in qs:
 for m in methods: assert len(orders[q][m])==100 and len(set(orders[q][m]))==100 and set(orders[q][m])==C[q]
metrics={}
for m in methods:
 metrics[m]={}
 for label,field in [('overall','positive_target_ids'),('implicit','implicit_target_ids'),('explicit','direct_target_ids')]:
  sub=[q for q in qs if pop[q][field]]
  metrics[m][label]={str(k):float(np.mean([len(set(orders[q][m][:k])&set(pop[q][field]))/len(pop[q][field]) for q in sub])) for k in [10,20,50,100]}
ref=load('data/teacher_health/fixed_pool_metrics.json')['orderings']
maxdiff=max(abs(metrics[m][lab][k]-ref[m][lab][k]) for m in methods for lab in metrics[m] for k in ['10','20','50'])
epochs={}
for ep in range(1,7):
 ranks={r['query_id']:r['Top50'] for r in rows(f'data/teacher/dev_epoch_{ep:02}.rankings.jsonl')}
 assert set(ranks)==set(qs)
 assert all(len(v)==50 and len(set(v))==50 and set(v)<=D[q] for q,v in ranks.items())
 epochs[ep]={str(k):float(np.mean([len(set(ranks[q][:k])&G[q])/len(G[q]) for q in qs])) for k in [10,20,50]}
 cov={'D100':D,'C100':C}
coverage={n:{'recall':float(np.mean([len(c[q]&G[q])/len(G[q]) for q in qs])), 'oracle_R10':float(np.mean([min(10,len(c[q]&G[q]))/len(G[q]) for q in qs])), 'zero_positive_queries':sum(not(c[q]&G[q]) for q in qs),'positive_pairs':sum(len(c[q]&G[q]) for q in qs)} for n,c in cov.items()}
composition={}
for m in methods:
 ctr=collections.Counter(t for q in qs for t in orders[q][m][:10]); stats={}
 for k in [10,20,50,100]:
  stats[str(k)]={'E_only_share':float(np.mean([len(set(orders[q][m][:k])-D[q])/k for q in qs])), 'D_member_positive_hits':sum(len(set(orders[q][m][:k])&D[q]&G[q]) for q in qs),'E_only_positive_hits':sum(len((set(orders[q][m][:k])-D[q])&G[q]) for q in qs)}
 stats['top_hubs']=[{'target':t,'top10_queries':v,'gt_queries':sum(t in G[q] for q in qs)} for t,v in ctr.most_common(12)]
 composition[m]=stats
filtered={m:{str(k):float(np.mean([len(set([t for t in orders[q][m] if t in D[q]][:k])&G[q])/len(G[q]) for q in qs])) for k in [10,20,50]} for m in methods}
# Collapse by paired source-group bootstrap with multiplicities and original query macro.
groups=collections.defaultdict(list)
for j,q in enumerate(qs):groups[pop[q]['source_table_id']].append(j)
gidx=list(groups.values());glen=np.array([len(x) for x in gidx]);boot={}
for a,b in [('J-natural-on-C','RAW-QT-on-C'),('J-natural-on-C','J-empty-on-C'),('P-QT-on-C','RAW-QT-on-C')]:
 dif=np.array([(len(set(orders[q][a][:10])&G[q])-len(set(orders[q][b][:10])&G[q]))/len(G[q]) for q in qs]); sm=np.array([dif[ix].sum() for ix in gidx]); rng=np.random.default_rng(13); vals=[]
 for it in range(100):
  ix=rng.integers(len(gidx),size=(100,len(gidx)));vals.extend((sm[ix].sum(1)/glen[ix].sum(1)).tolist())
 boot[a+' minus '+b]={'diff':float(dif.mean()),'ci95':np.quantile(vals,[.025,.975]).tolist(),'groups':len(groups),'positive_queries':int((dif>0).sum()),'negative_queries':int((dif<0).sum())}
# Support visibility (population witnesses already canonical in this package).
support={'all_implicit_pairs':sum(len(I[q]) for q in qs),'all_implicit_visible_pairs':0,'in_C_visible':0,'in_C_unobserved':0,'witness_visible_T_outside_C':0,'by_method':{m:{'visible_hits10':0,'unobserved_hits10':0} for m in methods}}
for q in qs:
 B=set(raw[q]['text_ids'][:10]+raw[q]['image_ids'][:10])
 for t in I[q]:
  vis=bool(set(pop[q]['witnesses'].get(t,[]))&B)
  support['all_implicit_visible_pairs']+=vis
  support['witness_visible_T_outside_C']+=vis and t not in C[q]
  if t not in C[q]:continue
  support['in_C_visible' if vis else 'in_C_unobserved']+=1
  for m in methods:support['by_method'][m]['visible_hits10' if vis else 'unobserved_hits10']+=t in orders[q][m][:10]
# Real Teacher logits on fixed D/E/C panel. Only labelled positive sets, masks respected.
packets=rows('data/teacher_health/packet_health.csv.jsonl');pstats={}
for p in ['D','E','C']:
 valid=[]
 for r in packets:
  if r['packet']!=p or 'teacher_logits' not in r:continue
  q=r['query_id'];log=r['teacher_logits']
  w=pop[q]['witnesses'];wq=set(e for es in w.values() for e in es)
  if p=='D':P=G[q];excluded=set()
  elif p=='E':P=wq;excluded=set()
  else:P=set(pop[q]['direct_target_ids'])|{t for t in I[q] if r['anchor'] in w.get(t,[])};excluded=G[q]-P
  cand=set(log)-excluded;ranked=sorted(cand,key=lambda t:(-log[t],t));P=P&cand;N=cand-P
  assert len(P)==r['positives']
  PA=np.mean([sum((log[t]>log[n])+.5*(log[t]==log[n]) for n in N)/len(N) for t in P])
  x={'q':q,'pairs':PA,'candidate_count':len(cand),'mean_rank':float(np.mean([ranked.index(t)+1 for t in P])),'positive_top_score':max(log[t] for t in P),'top_score':max(log.values()),'mask':len(excluded&set(log))}
  for k in [1,10,20,50]: x['recall'+str(k)]=len(P&set(ranked[:k]))/len(P)
  x['teacher_vs_stored_pair_error']=abs(PA-r['pair_accuracy_teacher'])
  valid.append(x)
 pstats[p]={'queries':len(valid),'mean_candidates':float(np.mean([x['candidate_count'] for x in valid])),'pair_accuracy':float(np.mean([x['pairs'] for x in valid])), 'mean_positive_rank':float(np.mean([x['mean_rank'] for x in valid])), 'positive_rank_median':float(np.median([x['mean_rank'] for x in valid])),'max_pair_error':max(x['teacher_vs_stored_pair_error'] for x in valid), **{f'R@{k}':float(np.mean([x['recall'+str(k)] for x in valid])) for k in [1,10,20,50]}}
 (OUT/f'packet_{p}_recomputed.jsonl').write_text('\n'.join(json.dumps(x) for x in valid)+'\n')
result={'queries':len(qs),'source_groups':len(groups),'max_metric_difference':maxdiff,'same_C_all_views':True,'coverage':coverage,'fixed_metrics':metrics,'epoch_D100_metrics':epochs,'composition':composition,'same_filtered_C_intersect_D100':filtered,'paired_cluster_bootstrap_10000':boot,'support':support,'packet_recomputed':pstats}
(OUT/'RECOMPUTED.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in result.items() if k not in ['composition','epoch_D100_metrics']},indent=2,ensure_ascii=False))
print('\nCOMPOSITION',json.dumps({m:{'at10':composition[m]['10'],'hubs':composition[m]['top_hubs'][:3]} for m in methods},indent=2))

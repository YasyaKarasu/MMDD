"""Independent JSONL qrels + rankings evaluator. No imports from training code.

Reads the exported ``eval_labels/<split>/{queries,qrels}.jsonl`` and a
``rankings.*.jsonl.gz`` file and recomputes query-macro target recall from raw IDs.
"""
from __future__ import annotations
import argparse,csv,gzip,json,math
from collections import defaultdict
from pathlib import Path

def rows(path:Path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8-sig') as f:
        for n,line in enumerate(f,1):
            if not line.strip():continue
            try:yield json.loads(line)
            except json.JSONDecodeError as e:raise ValueError(f'{path}:{n}: {e}') from e

def evaluate(queries:Path,qrels:Path,rankings:Path,out:Path,allow_subset:bool=False):
    meta={}
    for row in rows(queries):
        q=str(row['query_id'])
        if q in meta:raise ValueError(f'duplicate query {q}')
        if not row.get('source_group'):raise ValueError(f'missing source_group {q}')
        meta[q]=row
    gold=defaultdict(set)
    for row in rows(qrels):
        q=str(row['query_id'])
        if q not in meta:raise ValueError(f'qrels query absent from metadata: {q}')
        if float(row['rel'])>0:gold[q].add(str(row['target_id']))
    data=[];seen=set(); zero=[]
    for row in rows(rankings):
        q=str(row['query_id']);ids=list(map(str,row['target_ids']))
        if q not in meta or q in seen:raise ValueError(f'unknown/duplicate ranking query {q}')
        seen.add(q)
        if len(ids)!=len(set(ids)):raise ValueError(f'duplicate ranked ID: {q}')
        if 'scores' in row:
            scores=row['scores']
            if len(scores)!=len(ids) or not all(math.isfinite(float(x)) for x in scores):raise ValueError(f'invalid scores {q}')
            expected=sorted(zip(ids,scores),key=lambda x:(-x[1],x[0].encode()))
            if [i for i,_ in expected]!=ids:raise ValueError(f'not sorted by score/tie rule {q}')
        g=gold[q]
        if not g:zero.append(q);continue
        rec={'query_id':q,'source_group':str(meta[q]['source_group']),
             'query_kind':meta[q].get('query_kind','unknown'),'gold_count':len(g),'pool_size':len(ids)}
        for k in [10,20,30,40,50]:
            h=len(set(ids[:k])&g);rec[f'hits{k}']=h;rec[f'R{k}']=h/len(g)
        n=len(set(ids)&g);rec['gold_in_pool']=n;rec['CR_pool']=n/len(g)
        rec['QueryHitRate_pool']=int(n>0);rec['Oracle10']=min(10,n)/len(g);data.append(rec)
    expected={q for q in meta if gold[q]}
    missing=expected-seen
    if missing and not allow_subset:raise ValueError(f'{len(missing)} positive queries have no ranking; explicit --allow-subset required for registered probe')
    if not data:raise ValueError('no evaluable ranked queries')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'per_query.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    fields=['R10','R20','R30','R40','R50','CR_pool','QueryHitRate_pool','Oracle10']
    result={'source':'independent_raw_qrels_and_rankings','n_metadata_queries':len(meta),
            'n_scored_queries':len(data),'subset':bool(missing),'missing_queries':sorted(missing),
            'zero_gold_exclusions':sorted(q for q in meta if not gold[q]),'segments':{}}
    for kind in ['overall','implicit','explicit','mixed','unknown']:
        part=data if kind=='overall' else [r for r in data if r['query_kind']==kind]
        result['segments'][kind]={'n':len(part),**{f:sum(r[f] for r in part)/len(part) if part else None for f in fields}}
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for a in ['queries','qrels','rankings','out']:p.add_argument('--'+a,type=Path,required=True)
    p.add_argument('--allow-subset',action='store_true');args=p.parse_args()
    print(json.dumps(evaluate(args.queries,args.qrels,args.rankings,args.out,args.allow_subset),ensure_ascii=False,indent=2))

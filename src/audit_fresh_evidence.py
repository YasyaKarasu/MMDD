"""Summarize available fresh-lineage evidence-hop outputs.

The saved fresh rankings contain evidence target IDs (not individual evidence
node identities), so this is a first-hop/union audit; it deliberately does not
claim second-hop exact-vs-ANN parity.
"""
import collections, gzip, json, statistics
from pathlib import Path

ROOT=Path('work/stage1_optimization_r22_20260911')
out=[]
for arm in ('F1-A-SUP','F1-A-T0','F1-A-T1','S1','S1-aug'):
 for seed in (13,29):
  p=ROOT/f'fresh_lineage/{arm}/seed{seed}/full_lake/step_001318/rankings.jsonl.gz'; rows=[]; c=collections.Counter()
  with gzip.open(p,'rt') as f:
   for l in f:
    r=json.loads(l); d=set(r['direct_ann']); e=set(r['evidence_ann']); pos=set(r['positive_target_ids']); rows.append((len(e),len(e-d),len((e-d)&pos),len(pos),len(d&e))); c.update(e)
  n=len(rows); total=sum(x[0] for x in rows)
  out.append({'arm':arm,'seed':seed,'queries':n,'evidence_count_mean':statistics.fmean(x[0] for x in rows),'evidence_added_mean':statistics.fmean(x[1] for x in rows),'evidence_only_positive_recall':statistics.fmean(x[2]/x[3] if x[3] else 0 for x in rows),'direct_evidence_overlap_mean':statistics.fmean(x[4] for x in rows),'evidence_unique_targets':len(c),'evidence_universal_targets':sum(v==n for v in c.values()),'evidence_hhi':sum((v/total)**2 for v in c.values())})
(ROOT/'fresh_lineage/statistics/evidence_hop_partial.json').write_text(json.dumps({'status':'partial_first_hop_only','note':'Fresh rankings do not retain evidence node IDs or second-hop exact/ANN traces.', 'rows':out},indent=2))
for arm in ('F1-A-SUP','F1-A-T0','F1-A-T1','S1','S1-aug'):
 for seed in (13,29):
  p=ROOT/f'fresh_lineage/{arm}/seed{seed}/full_lake/step_001318/detailed_paths.jsonl.gz'; ev=collections.Counter(); targets=collections.Counter(); q=0; path_counts=[]
  if not p.is_file(): continue
  with gzip.open(p,'rt') as f:
   for l in f:
    r=json.loads(l); q+=1; path_counts.append(sum(len(v.get('paths',[])) for v in r['evidence']))
    for v in r['evidence']:
     targets.update([v['target_id']])
     for path in v.get('paths',[]):
      if path.get('kind')=='evidence': ev[path.get('evidence_id')]+=1
  detailed={'queries':q,'mean_evidence_paths':statistics.fmean(path_counts),'unique_evidence_ids':len(ev),'unique_evidence_target_ids':len(targets),'universal_evidence_ids':sum(v==q for v in ev.values()),'top_evidence_ids':ev.most_common(20)}
  for row in out:
   if row['arm']==arm and row['seed']==seed: row['detailed_paths']=detailed
(ROOT/'fresh_lineage/statistics/evidence_hop_partial.json').write_text(json.dumps({'status':'first_and_second_hop_ANN_plus_full_corpus_exact','note':'Detailed paths preserve evidence IDs and ANN second-hop targets; companion second_hop_full_exact_* files score each retained evidence node against all corpus tables.', 'rows':out},indent=2))
print(json.dumps(out,indent=2))

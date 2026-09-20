"""Small synthetic tests of the uploaded production code, not an experiment rerun."""
import os
from pathlib import Path
import sys,json, tempfile
import torch
ROOT=Path(os.environ.get('MMDD_AUDIT_ROOT', '/mnt/data/audit_review_20260918/mmdd_s1_audit_20260918'))
sys.path.insert(0,str(ROOT/'src'))
from mmdd_stage1_clean import train,sampling
OUT=Path(__file__).parent
torch.set_num_threads(1)
corpora={'target':['tpos']+[f't{i:02}' for i in range(40)],'evidence_text':[f'et{i:02}' for i in range(20)],'evidence_image':[f'ei{i:02}' for i in range(20)]}
corpora['evidence']=corpora['evidence_text']+corpora['evidence_image']
q='query_probe'
row={'query_id':q,'split':'train','positive_target_ids':['tpos'],'direct_target_ids':[], 'implicit_target_ids':['tpos'],'witnesses':{'tpos':['et00']}}
builder=train.PacketBuilder(gt={'train':{'population':[row]}},corpora=corpora)
ranks={'D':train.RankTable({q:[(f't{i:02}',1-i*.01) for i in range(20)]}),'E_text':train.RankTable({q:[(x,1-i*.01) for i,x in enumerate(corpora['evidence_text'])]}),'E_image':train.RankTable({q:[(x,1-i*.01) for i,x in enumerate(corpora['evidence_image'])]})}
qrank={'text_ids':corpora['evidence_text'][:10],'image_ids':corpora['evidence_image'][:10]}
base=dict(split='train',row=row,phase='teacher',sampling_arm='T',anchor_rank={'et00':[(f't{i:02}',.8-i*.01) for i in range(20)]}, rank_tables=ranks,per_modality=10,include_bundle=True)
observations=[]
for ep in [3,4,5,6]:
 built=builder.build(**base,epoch=ep,q_rank=None)
 b=built['packets']['B'];after=train._expand_packets(train._compact_packets(built))['packets']['B']
 n=len(b['candidates']);x=torch.zeros(n,requires_grad=True)
 orig=train.teacher_rank_term(x,b);actual=train.teacher_rank_term(x,after)
 grad_pos=None
 if actual is not None:
  actual.backward();grad_pos=float(x.grad[b['candidates'].index('tpos')])
 correct=builder.build(**base,epoch=ep,q_rank=qrank)['packets']['B']
 observations.append({'epoch':ep,'view':b['view'],'actual_context':b['context'], 'expected_context_with_raw_qrank_count':len(correct['context']),'before_override':b.get('positive_ids','MISSING'),'after_override':after.get('positive_ids','MISSING'),'before_loss':None if orig is None else float(orig.detach()),'after_loss':None if actual is None else float(actual.detach()), 'positive_gradient_actual':grad_pos})
# Demonstrate that the two ranked streams lose the intended interleaving.
inter=train.merge_modality_ranks([('text0',.1),('text1',.09)],[('image0',.9),('image1',.89)],4)
make=sampling.MakeList(phase='teacher',sampling_arm='T',corpus_universe={})
re_sorted=make.hard_pool(inter)
# Execute the exact refresh method against spy objects.
class B:
 def population(self,split):return [row]
class Batch:
 def __init__(self):self.calls=[]
 def score(self,teacher,queries,candidates,contexts,mode,chunk):
  self.calls.append({'queries':queries,'candidates':candidates,'contexts':contexts,'mode':mode})
  return [torch.arange(len(v),dtype=torch.float32) for v in candidates]
t=train.TeacherTrainer.__new__(train.TeacherTrainer)
t.teacher=torch.nn.Linear(1,1);t.builder=B();t.rank_tables=ranks.copy();t.anchor_rank={'et00':[('special_E_target',1.0)]};t.batch=Batch();t.chunk=8
with tempfile.TemporaryDirectory() as temp:
 t.output_dir=Path(temp);refresh=t.refresh_hard_ranks();payload=(t.output_dir/'mining_epoch2'/'meta.json').read_text()
refresh_data={'result':refresh,'calls':t.batch.calls,'C_anchor_rank_after':t.anchor_rank,'D_pool_before':[x for x,_ in ranks['D'].get(q)],'D_pool_seen':t.batch.calls[0]['candidates'][0]}
result={'scope':'Synthetic probes of uploaded source, no real weights trained.', 'B_packet':observations,'rank_interleave':{'input_interleaved':[x for x,s in inter],'after_hard_pool':re_sorted},'hard_refresh':refresh_data}
(OUT/'SOURCE_PROBES.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(json.dumps(result,indent=2,ensure_ascii=False))

"""Probe formula and gradient routing using small synthetic tensors only."""
import os
from pathlib import Path
import sys,json
import torch,numpy as np
ROOT=Path(os.environ.get('MMDD_AUDIT_ROOT', '/mnt/data/audit_review_20260918/mmdd_s1_audit_20260918'))
sys.path.insert(0,str(ROOT/'src'))
from mmdd_stage1_clean import train,models,reference,evaluate
from mmdd_stage1_clean.timing import Timing
torch.set_num_threads(1);torch.manual_seed(13)
d0,d=16,8
s=reference.CompactStudent(input_dim=d0,d=d,rank=2)
with torch.no_grad():
 s.direct.weight.copy_(torch.diag(torch.arange(1,d+1,dtype=torch.float32)))
 s.evidence.weight.copy_(-torch.eye(d));s.base_e.weight.copy_(2*torch.eye(d))
ids=['q','t1','t2','e1','e2']; rng=np.random.default_rng(13)
bank=models.ObjectBank(ids,np.array([0,0,0,1,1]),np.tile(np.arange(9),(5,1)),rng.normal(size=(5,d0)).astype('float32'),rng.normal(size=(5,8,d0)).astype('float16'),np.ones((5,9),dtype='uint8'))
t=train.StudentTrainer.__new__(train.StudentTrainer);t.student=s;t.bank=bank;t.device='cpu';t.object_ids=ids;t.timing=Timing()
class Builder:corpora={'target':['t1','t2'],'evidence_text':['e1','e2'],'evidence_image':[]}
t.builder=Builder()
packets={'D':{'mode':'P','context':[],'candidates':['t1','t2']},'E':{'mode':'P','context':[],'candidates':['e1','e2']},'C':{'mode':'J','context':['e1'],'candidates':['t1','t2']}}
res={}
for label,p in packets.items():
 s.zero_grad(set_to_none=True);actual=t.score_packet('q',p)
 cache,valid,mod,kind=bank.keys(ids,'cpu');v=s.encode(cache,valid,mod,kind)
 q={'D':s.query_direct(v[0]),'E':s.query_evidence(v[0]),'C':s.query_next(v[0],v[3])}[label]
 key=v[[ids.index(x) for x in p['candidates']]]
 expected=s.logits(q,key)
 actual.sum().backward()
 res[label]={'actual_logits':actual.detach().tolist(),'specified_logits':expected.detach().tolist(),'max_error':float((actual-expected).abs().max().detach()),'evidence_weight_grad_is_none':s.evidence.weight.grad is None,'direct_weight_grad_is_none':s.direct.weight.grad is None,'base_e_weight_grad_is_none':s.base_e.weight.grad is None}
keys=t.key_index()
cache,valid,mod,kind=bank.keys(['t1','t2'],'cpu');expected_T=s.encode(cache,valid,mod,kind).detach()
cache,valid,mod,kind=bank.keys(['e1','e2'],'cpu');expected_E=s.encode(cache,valid,mod,kind).detach()
res['mining_keys']={'D_max_error_to_nuT':float((torch.tensor(keys['D_keys'])-expected_T).abs().max()),'E_max_error_to_nuE':float((torch.tensor(keys['E_text_keys'])-expected_E).abs().max()),'note':'C uses 2I in this probe; normalization hides its extra map. Actual source applies base_e to every candidate.'}
Path(__file__).with_name('STUDENT_PROBES.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))

"""Synthetic reference tests, not real MMDD integration or training receipts."""
import json
import copy
from pathlib import Path
import pytest
import torch
from contracts import *

torch.set_num_threads(1)


def tensors():
    g=torch.Generator().manual_seed(11)
    q=torch.randn(7,generator=g,dtype=torch.float64,requires_grad=True)
    keys=torch.randn(19,7,generator=g,dtype=torch.float64,requires_grad=True)
    p=torch.tensor([i in {2,10,18} for i in range(19)])
    a=torch.tensor([i not in {0,7,8,14} for i in range(19)])
    return q,keys,p,a

@pytest.mark.parametrize('chunk',[1,2,4,7,19,27])
def test_full_denominator_and_gradients(chunk):
    q,k,p,a=tensors()
    full=rank_loss(k@q,p,a)
    streamed=streamed_full_loss(q,k,p,a,chunk)
    assert torch.allclose(full,streamed,atol=1e-10,rtol=1e-10)
    grad=torch.autograd.grad(full,(q,k),retain_graph=True)
    other=torch.autograd.grad(streamed,(q,k))
    for x,y in zip(grad,other): assert torch.allclose(x,y,atol=1e-10,rtol=1e-10)


def test_chunk_ce_average_is_not_full_loss():
    s=torch.tensor([0.,4.,1.,0.],dtype=torch.float64)
    p=torch.tensor([True,False,True,False])
    wrong=(rank_loss(s[:2],p[:2])+rank_loss(s[2:],p[2:]))/2
    assert abs((rank_loss(s,p)-wrong).item())>0.1


def test_positive_ignore_and_all_targets():
    p,i,n=pin_sets('abcde','a','ab','ac')
    assert (p,i,n)==({'a'},{'b','c'},{'d','e'})
    assert p|i|n==set('abcde') and not p&i and not n&i


def test_outside_positive_rejected():
    with pytest.raises(ValueError): pin_sets('ab','c','c','')


def test_empty_positive_not_fallback():
    packet={'positive_ids':[], 'query_gold':['a'], 'ignore_ids':['b']}
    roundtrip=json.loads(json.dumps(packet))
    assert 'positive_ids' in roundtrip and roundtrip['positive_ids']==[]
    p,a=masks(['a','b'],set(roundtrip['positive_ids']),set(roundtrip['ignore_ids']))
    assert rank_loss(torch.zeros(2),p,a) is None


def test_no_competitor_is_inactive():
    assert rank_loss(torch.tensor([1.]),torch.tensor([True])) is None


def test_ignore_zero_gradient():
    s=torch.tensor([0.,200.,1.],dtype=torch.float64,requires_grad=True)
    v=rank_loss(s,torch.tensor([True,False,False]),torch.tensor([True,False,True]))
    v.backward(); assert s.grad[1]==0


def test_each_known_positive_has_nonpositive_score_gradient():
    s=torch.tensor([0.,1.,2.],dtype=torch.float64,requires_grad=True)
    rank_loss(s,torch.tensor([True,True,False])).backward()
    assert (s.grad[:2]<0).all() and s.grad[2]>0


def test_kd_direction_and_teacher_detach():
    ss=torch.tensor([0.1,1.,-2.],dtype=torch.float64,requires_grad=True)
    tt=torch.tensor([2.,-1.,0.],dtype=torch.float64,requires_grad=True)
    a=torch.tensor([True,True,False])
    loss=kd_loss(ss,tt,a)
    p=torch.softmax(tt.detach()[a]/2,0)
    expected=4*(p*(p.log()-torch.log_softmax(ss[a]/2,0))).sum()
    assert torch.allclose(loss,expected)
    loss.backward(); assert tt.grad is None and ss.grad[2]==0


def test_kd_zero_is_sup_gradient():
    x=torch.tensor([0.5,-0.3,2.],requires_grad=True)
    p=torch.tensor([True,False,False]); a=torch.ones(3,dtype=torch.bool)
    sup=rank_loss(x,p,a)
    total=sup+0.0*kd_loss(x,torch.tensor([2.,0.,1.]),a)
    assert torch.equal(torch.autograd.grad(sup,x,retain_graph=True)[0],torch.autograd.grad(total,x)[0])


def test_kd_same_logits_zero():
    z=torch.tensor([2.,-1.,0.])
    assert abs(kd_loss(z,z,torch.ones(3,dtype=torch.bool)).item())<1e-7

@pytest.mark.parametrize('magnitude',[0.,0.01,1.,50.,10000.])
def test_trust_region(magnitude):
    torch.manual_seed(3)
    base=torch.randn(11,7,dtype=torch.float64)
    delta=torch.randn(11,7,dtype=torch.float64)*magnitude
    v=clipped_query(base,delta)
    assert ((v-base).norm(dim=-1) <= 0.5*base.norm(dim=-1)+1e-10).all()
    if magnitude==0: assert torch.equal(v,base)


def test_zero_base_does_not_nan():
    b=torch.zeros(5,dtype=torch.float64)
    d=torch.randn(5,dtype=torch.float64,requires_grad=True)
    v=clipped_query(b,d); v.sum().backward()
    assert torch.equal(v,b) and torch.isfinite(d.grad).all()


def test_zero_residual_has_finite_gradient():
    b=torch.ones(5,dtype=torch.float64)
    d=torch.zeros(5,dtype=torch.float64,requires_grad=True)
    clipped_query(b,d).sum().backward()
    assert torch.equal(d.grad,torch.ones_like(d))


def test_final_normalization_does_not_change_ip_ranking():
    torch.manual_seed(4); q=torch.randn(7); k=torch.randn(18,7)
    assert torch.equal(torch.argsort(k@q),torch.argsort(k@(q/q.norm())))


def test_adapter_zero_and_first_gradient():
    torch.manual_seed(2); net=ConditionalAdapter(7,8).double()
    q,e,b=[torch.randn(3,7,dtype=torch.float64) for _ in range(3)]
    v=net(q,e,b)
    assert torch.equal(v,b)
    v.sum().backward()
    assert net.output.weight.grad.norm()>0
    assert net.hidden.weight.grad.norm()==0  # normal consequence of zero W2 at step zero


def test_eonly_does_not_read_q_after_nonzero_training():
    torch.manual_seed(2); net=ConditionalAdapter(7,8).double()
    with torch.no_grad(): net.output.weight.normal_(0,0.1)
    q,e,b=[torch.randn(3,7,dtype=torch.float64) for _ in range(3)]
    assert torch.equal(net(q,e,b,read_q=False),net(q+17,e,b,read_q=False))
    assert not torch.equal(net(q,e,b),net(q+17,e,b))


def test_asymmetric_R_static_target_identity():
    z=torch.tensor([[1.,2.,4.]],dtype=torch.float64)
    dest=torch.tensor([[1.,0.,2.],[0.,1.,3.]],dtype=torch.float64)
    p=torch.tensor([[1.,2.,0.],[0.,1.,1.]],dtype=torch.float64)
    r=torch.tensor([[1.,3.],[-2.,2.]],dtype=torch.float64)
    u=projected(z,p); keys=projected(dest,p)
    score=bilinear_query(u,r)@keys.T
    expected=torch.stack([u[0]@r@t for t in keys])
    assert torch.equal(score[0],expected)
    assert not torch.allclose(score,(u@r.T)@keys.T)


def test_full_lake_hub_missed_by_small_list_has_gradient():
    s=torch.tensor([1.,0.,10.],dtype=torch.float64,requires_grad=True)
    p=torch.tensor([True,False,False])
    small=rank_loss(s[:2],p[:2]); full=rank_loss(s,p)
    gs=torch.autograd.grad(small,s,retain_graph=True)[0]
    gf=torch.autograd.grad(full,s)[0]
    assert gs[2]==0 and gf[2]>.99 and full>small


def test_zero_path_identity():
    s=torch.tensor(2.,requires_grad=True)
    assert torch.equal(target_path_lse(s,[]),s)


def test_path_lse_gradient_reaches_evidence():
    s=torch.tensor(0.,requires_grad=True); e=torch.tensor(1.,requires_grad=True)
    target_path_lse(s,[e]).backward(); assert s.grad>0 and e.grad>0


def test_duplicate_triplets_no_extra_vote():
    p=[('q','e','t'),('q','e','t'),('q','f','t')]
    assert len(unique_paths(p))==2


def test_augmentation_shared_not_positive_only():
    g={'t1':['e1'],'t2':[]}
    h=augment_shared('q',['t1','t2'],g,'ew')
    assert 'ew' in h['t1'] and 'ew' in h['t2'] and g['t2']==[]


def test_augmentation_no_duplicate_evidence():
    assert augment_shared('q',['t'],{'t':['e']},'e')=={'t':['e']}


def test_equal_rrf_not_direct_only():
    assert equal_rrf(['a','b'],['c','b'])==['b','a','c']
    assert equal_rrf([],['c','a'])==['c','a']


def test_equal_rrf_no_phantom_vote():
    assert equal_rrf(['a'],['b'],budget=1)==['a']
    with pytest.raises(ValueError): equal_rrf(['a','a'],[])


def test_stable_ties():
    assert stable_rank(['b','a','c'],[1.,1.,0.])==['a','b','c']


def test_macro_recall_not_hit():
    g={'q1':{'a','b'},'q2':{'c'}}
    assert macro_recall(g,{'q1':['a']},1)==.25


def test_failed_query_stays_in_denominator():
    assert macro_recall({'a':{'t'},'b':{'u'}},{'a':['t']},10)==.5


def test_wlt_ties_count_population():
    g={'q1':{'a'},'q2':{'b'},'q3':{'c'}}
    result=wlt(g,{'q1':['a']},{'q2':['b']},1)
    assert result==(1,1,1) and sum(result)==3


def test_probe_pair_to_query_macro():
    assert pair_query_macro([('q1',1.),('q1',1.),('q2',0.)])==.5


def test_strict_excludes_both_ann_and_exact():
    assert strict_eo(set('abc'),set('abc'),{'a'},{'b'})=={'c'}


def test_all_text_tokens_contribute_bins():
    x=torch.arange(130,dtype=torch.float32).reshape(65,2).requires_grad_()
    y=fixed_bins(x); assert y.shape==(64,2)
    y.sum().backward(); assert (x.grad>0).all()


def test_small_content_not_forced_to_64_slots():
    x=torch.randn(9,4)
    assert torch.equal(fixed_bins(x),x)


def test_empty_content_not_zero_filled():
    with pytest.raises(ValueError): fixed_bins(torch.empty(0,4))


def test_local_rng_is_not_global_state():
    random.seed(1); state=random.getstate()
    a=local_rng('q1').sample(range(1000),20)
    assert state==random.getstate()
    assert a==local_rng('q1').sample(range(1000),20)
    assert a!=local_rng('q2').sample(range(1000),20)


def test_tail_microbatch_gradient():
    x=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    samples=torch.tensor([1.,4.,5.,2.,7.],dtype=torch.float64)
    full=((x*samples-1)**2).mean()
    g=torch.autograd.grad(full,x)[0]
    x2=x.detach().clone().requires_grad_()
    for batch in samples.split(2): (((x2*batch-1)**2).sum()/len(samples)).backward()
    assert torch.allclose(g,x2.grad,atol=1e-10)


def lineage():
    return {'data':Artifact('data','dataset',None),'qwen':Artifact('qwen','public_backbone',None),
            'z':Artifact('z','pure_backbone_cache',None,('data','qwen'),True),
            'pca':Artifact('pca','PCA','new',('z',)),
            'Te':Artifact('Te','task_checkpoint','new',('z','data')),
            'Tp':Artifact('Tp','task_checkpoint','new',('Te',)),
            'S1':Artifact('S1','task_checkpoint','new',('pca','Tp')),
            'S2':Artifact('S2','task_checkpoint','new',('S1','Tp'))}


def test_current_run_stage_parent_is_legal(): validate_lineage(lineage(),'S2','new')

@pytest.mark.parametrize('kind',['task_checkpoint','optimizer','training_list','PCA','teacher_logits'])
def test_old_task_artifact_rejected(kind):
    a=lineage(); a['old']=Artifact('old',kind,'historical')
    a['S1']=Artifact('S1','task_checkpoint','new',('old',))
    with pytest.raises(ValueError,match='external'): validate_lineage(a,'S2','new')


def test_learned_cache_cannot_masquerade_as_pure():
    a=lineage(); a['z']=Artifact('z','pure_backbone_cache',None,('Te',),True)
    with pytest.raises(ValueError,match='learned'): validate_lineage(a,'S2','new')


def test_unverified_pure_cache_rejected():
    a=lineage(); a['z']=Artifact('z','pure_backbone_cache',None,('data','qwen'),False)
    with pytest.raises(ValueError,match='unverified'): validate_lineage(a,'S2','new')


def test_cycle_rejected():
    a=lineage(); a['Te']=Artifact('Te','task_checkpoint','new',('Tp',))
    with pytest.raises(ValueError,match='cycle'): validate_lineage(a,'S2','new')


def good_metrics():
    return dict(main_R10=.4,raw_same_T_R10=.38,main_implicit_R10=.35,raw_same_T_implicit_R10=.34,
        same_pool_QT_R10=.401,trained_QT_same_pool_R10=.401,swap_implicit_R10=.34,QE_ET_R10=.34,EONLY_ET_R10=.33,SUP_R10=.401,
        strict_pairs=25,QE_strict_retained=12,EONLY_strict_retained=11)


def test_repeat_gate_all_conditions(): assert repeat_gate(good_metrics())==(True,[])


def test_no_repeat_when_E_ignored():
    m=good_metrics();m['swap_implicit_R10']=m['main_implicit_R10']
    assert 'evidence_content' in repeat_gate(m)[1]


def test_no_repeat_when_only_QT_strong():
    m=good_metrics();m['same_pool_QT_R10']=.45
    assert 'path_guard' in repeat_gate(m)[1]


def test_no_repeat_if_no_strict_population():
    m=good_metrics();m['strict_pairs']=10
    assert 'strict_population' in repeat_gate(m)[1]


def test_protocol_root_no_old_init():
    p=json.loads((Path(__file__).parents[1]/'protocol.json').read_text())
    assert p['scope']=='complete_fresh_stage1'
    assert p['data']['train_scope']=='all_original_train'
    assert p['lineage']['external_task_checkpoint'] is False
    assert p['lineage']['external_training_list'] is False
    assert p['training']['max_stage_jobs_total']==26
    assert p['teacher']['all_modules_fresh'] and p['teacher']['all_modules_trainable_during_training']
    assert p['student']['C2_trainable']==['conditional_adapter']
    assert p['student']['C2_target_KD_weight']>0
    assert p['encoder']['query_max_rows'] is None and not p['encoder']['query_merge_rows']

from teacher_structure import FreshPathTeacher, ObjectInput

def tiny_teacher():
    torch.manual_seed(5)
    return FreshPathTeacher(input_dim=6,width=8,heads=2,layers=3,ffn=16,text_slots=2,image_slots=3).double().eval()

def obj(kind,n):
    return ObjectInput(kind,torch.randn(6,dtype=torch.float64),torch.randn(n,6,dtype=torch.float64))

def test_teacher_random_compressors_trainable():
    t=tiny_teacher(); q= obj('table',14); dest=obj('table',5); e=obj('text',7); im=obj('image',9)
    loss=t(q,dest,e)+t(q,dest,im)+t(q,e)
    loss.backward()
    assert all(p.requires_grad for p in t.parameters())
    for k in ('table','text','image'):
        assert t.adapters[k].weight.grad is not None and t.adapters[k].weight.grad.norm()>0
    assert t.poolers['text'].queries.grad.norm()>0 and t.poolers['image'].queries.grad.norm()>0
    assert t.globals['table'][0].weight.grad.norm()>0


def test_teacher_keeps_all_query_row_tokens():
    t=tiny_teacher(); q=obj('table',21)
    compressed,_=t.compress(q)
    assert len(compressed)==21 # schema + 20 rows; neither 7 nor 12 cap


def test_teacher_layers_not_identical_after_initialization():
    t=tiny_teacher()
    assert not torch.equal(t.relation.layers[0].linear1.weight,t.relation.layers[1].linear1.weight)


def test_pair_and_triplet_use_single_output_head():
    t=tiny_teacher(); calls=[]
    h=t.scoring_head.register_forward_hook(lambda *args:calls.append(1))
    q,d,e=obj('table',3),obj('table',4),obj('text',5)
    t(q,d); t(q,d,e); h.remove()
    assert len(calls)==2 and not hasattr(t,'qt_head') and not hasattr(t,'path_head')


def test_qet_forward_reads_evidence_content():
    t=tiny_teacher(); q,d,e=obj('table',3),obj('table',4),obj('text',5)
    other=ObjectInput('text',e.z+1,e.content*3+2)
    assert not torch.allclose(t(q,d,e),t(q,d,other),atol=1e-9,rtol=1e-9)


def test_frozen_teacher_has_no_grad():
    t=tiny_teacher().requires_grad_(False)
    y=t(obj('table',3),obj('table',4),obj('image',6))
    assert not y.requires_grad


def test_c2_target_path_kd_reaches_only_adapter():
    torch.manual_seed(9)
    adapter=ConditionalAdapter(4,5).double()
    q,e,base=[torch.randn(4,dtype=torch.float64) for _ in range(3)]
    keys=torch.randn(3,4,dtype=torch.float64)
    frozen_keys=keys.clone(); fixed_direct=torch.tensor([.1,.3,.4],dtype=torch.float64)
    v=adapter(q,e,base)
    next_scores=keys@v
    path_scores=torch.stack([target_path_lse(fixed_direct[i],[next_scores[i]+.2]) for i in range(3)])
    teacher_scores=torch.tensor([2.,-1.,.2],dtype=torch.float64,requires_grad=True)
    p=torch.tensor([True,False,False]); a=torch.ones(3,dtype=torch.bool)
    total=rank_loss(path_scores,p,a)+.5*kd_loss(path_scores,teacher_scores,a)
    total.backward()
    assert adapter.output.weight.grad.norm()>0 and teacher_scores.grad is None
    assert keys.grad is None and torch.equal(keys,frozen_keys)

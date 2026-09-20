from pathlib import Path
from copy import deepcopy
import json
import pytest
import torch
from contracts import rank_loss, repeat_gate
from control_contracts import (QTOnlyStudent,full_qt_loss,native_et_score,strict_direct_pool,
    validate_qt_label_payload,score_cache_key,validate_dag,ready_stages,MemoryProfile,can_colocate)
from teacher_structure import FreshPathTeacher,ObjectInput

ROOT=Path(__file__).parents[1]

def dag(): return json.loads((ROOT/'EXECUTION_DAG.json').read_text())['stages']

def make_qt():
    torch.manual_seed(7)
    m=QTOnlyStudent(torch.randn(3,5,dtype=torch.float64))
    with torch.no_grad(): m.R_QT.copy_(torch.randn(3,3,dtype=torch.float64))
    return m

@pytest.mark.parametrize('chunk',[1,2,3,8])
def test_qt_full_denominator_and_both_sides_gradient(chunk):
    a=make_qt();b=deepcopy(a)
    q=torch.randn(5,dtype=torch.float64);ts=torch.randn(8,5,dtype=torch.float64)
    pos=torch.tensor([True,False,False,False,False,False,False,True]);allowed=torch.ones(8,dtype=torch.bool)
    allowed[2]=False
    loss=rank_loss(a(q,ts),pos,allowed);loss.backward()
    chunked=full_qt_loss(b,q,ts,pos,allowed,chunk);chunked.backward()
    assert torch.allclose(loss,chunked,atol=1e-9,rtol=1e-9)
    for p,r in zip(a.parameters(),b.parameters()):
        assert torch.allclose(p.grad,r.grad,atol=1e-9,rtol=1e-9)

def test_qt_target_projection_must_not_be_detached():
    a=make_qt();b=deepcopy(a)
    q=torch.randn(5,dtype=torch.float64);ts=torch.randn(4,5,dtype=torch.float64)
    pos=torch.tensor([True,False,False,False]);allowed=torch.ones(4,dtype=torch.bool)
    full_qt_loss(a,q,ts,pos,allowed,2).backward()
    rank_loss(b.query(q)@b.keys(ts).detach().T,pos,allowed).backward()
    assert not torch.allclose(a.P_table.grad,b.P_table.grad,atol=1e-8)

def test_qt_updates_require_fresh_keys():
    m=make_qt();z=torch.randn(5,5,dtype=torch.float64)
    old=m.keys(z).detach().clone()
    with torch.no_grad():m.P_table.add_(.2)
    assert not torch.equal(old,m.keys(z))

def test_only_table_and_qt_parameters():
    assert set(dict(make_qt().named_parameters()))=={'P_table','R_QT'}

def test_qt_init_is_copy_not_alias():
    a=make_qt();b=deepcopy(a)
    with torch.no_grad():a.P_table.add_(1)
    assert not torch.equal(a.P_table,b.P_table)

def test_native_formula_is_plain_bilinear_without_query():
    torch.manual_seed(4)
    ze=torch.randn(5,dtype=torch.float64);zt=torch.randn(6,5,dtype=torch.float64)
    pe=torch.randn(3,5,dtype=torch.float64);pt=torch.randn(3,5,dtype=torch.float64);r=torch.randn(3,3,dtype=torch.float64)
    score=native_et_score(ze,zt,pe,pt,r)
    ref=torch.stack([(pe@ze)@r@(pt@t) for t in zt])
    assert torch.allclose(score,ref,atol=1e-10)
    for _q in (torch.zeros(5),torch.ones(5)):
        assert torch.equal(score,native_et_score(ze,zt,pe,pt,r))

def test_strict_direct_only_pool():assert strict_direct_pool(['t2','t1'])==['t2','t1']

@pytest.mark.parametrize('e',[[],['e1'],{}])
def test_strict_direct_rejects_even_empty_evidence_argument(e):
    with pytest.raises(ValueError):strict_direct_pool(['t'],evidence_rankings=e)

def test_qt_labels_accept_implicit_G_positive():
    validate_qt_label_payload({'query_id':'implicit_q','positive_target_ids':['t'],'legal_target_ids':['t','n']})

@pytest.mark.parametrize('field',['W','D','Epos','evidence','path_logits'])
def test_qt_labels_reject_evidence_and_unapproved_annotations(field):
    x={'query_id':'q','positive_target_ids':['t'],'legal_target_ids':['t','n'],field:{}}
    with pytest.raises(ValueError):validate_qt_label_payload(x)

def test_teacher_branch_cache_keys_cannot_alias():
    a=score_cache_key('run','T_QT','abc','q',None,['t','n'],'mask','target')
    b=score_cache_key('run','T_PATH','abc','q',None,['t','n'],'mask','target')
    assert a!=b

def test_teacher_cache_order_and_masks_matter():
    a=score_cache_key('run','T_QT','abc','q',None,['t','n'],'mask','target')
    assert a!=score_cache_key('run','T_QT','abc','q',None,['n','t'],'mask','target')
    assert a!=score_cache_key('run','T_QT','abc','q',None,['t','n'],'other_mask','target')

def test_two_teacher_branches_share_parent_values_not_parameter_storage():
    torch.manual_seed(11)
    edge=FreshPathTeacher(input_dim=4,width=8,heads=2,layers=1,ffn=16,text_slots=2,image_slots=2).double()
    path=deepcopy(edge);qt=deepcopy(edge)
    assert all(torch.equal(a,b) for a,b in zip(path.parameters(),qt.parameters()))
    with torch.no_grad():next(path.parameters()).add_(1)
    assert not torch.equal(next(path.parameters()),next(qt.parameters()))
    assert torch.equal(next(edge.parameters()),next(qt.parameters()))

def test_qt_forward_has_no_bridge_token_gradient():
    torch.manual_seed(12)
    t=FreshPathTeacher(input_dim=4,width=8,heads=2,layers=1,ffn=16,text_slots=2,image_slots=2).double().eval()
    q=ObjectInput('table',torch.randn(4,dtype=torch.float64),torch.randn(3,4,dtype=torch.float64))
    dest=ObjectInput('table',torch.randn(4,dtype=torch.float64),torch.randn(3,4,dtype=torch.float64))
    t(q,dest).backward()
    assert t.roles.weight.grad[2].abs().sum()==0
    assert t.adapters['image'].weight.grad is None

def test_dag_has_13_current_run_stages():
    d=dag();assert len(d)==13 and len(validate_dag(d))==13
    ids={x['stage'] for x in d}
    assert {'T_QT','S_QT_SUP_C1','S_QT_KD_C2','S_KD_NATIVE_C2'}<=ids

def test_both_teacher_branches_ready_after_edge_data_committed():
    ready=ready_stages(dag(),{'T_EDGE'},{'P1','edge_refresh','common_target_graph','PCA'})
    assert 'T_QT' in ready and 'T_PATH' in ready
    assert 'S_KD_C1' not in ready

def test_no_read_of_unsealed_data():
    ready=ready_stages(dag(),{'T_EDGE'},{'P1'})
    assert 'T_QT' not in ready and 'T_PATH' not in ready

def test_qt_sup_does_not_wait_for_multimodal_teacher():
    ready=ready_stages(dag(),set(),{'P1','PCA','raw_QT_only_lists'})
    assert 'S_QT_SUP_C1' in ready and 'S_QT_KD_C1' not in ready

def test_kd_does_not_use_unfinished_teacher():
    ready=ready_stages(dag(),{'T_EDGE'},{'P1','PCA','edge_refresh'},running={'T_PATH'})
    assert 'S_KD_C1' not in ready

def test_native_waits_for_its_readonly_qt_teacher():
    ready=ready_stages(dag(),{'T_EDGE','T_PATH','S_KD_C1'},{'P1','common_target_graph','conditional_lists'})
    assert 'S_KD_NATIVE_C2' not in ready and 'S_KD_QE_C2' in ready

def test_no_same_stage_duplicate_launch():
    ready=ready_stages(dag(),{'T_EDGE'},{'P1','edge_refresh','common_target_graph'},running={'T_PATH'})
    assert 'T_PATH' not in ready and 'T_QT' in ready

def test_dag_cycle_rejected():
    d=deepcopy(dag());d[0]['weight_parents']=['T_PATH']
    with pytest.raises(ValueError,match='cyclic'):validate_dag(d)

def can(profiles,**kwargs):
    values=dict(device_total_gib=24,external_used_gib=0,parity_passed=True,
                serial_fixed_work_seconds=20,concurrent_fixed_work_seconds=15)
    values.update(kwargs)
    return can_colocate(profiles,**values)

def test_safe_parallel_memory_and_speedup():assert can([MemoryProfile(5),MemoryProfile(7)])

def test_framework_reserved_and_external_usage_count():
    assert not can([MemoryProfile(8,12),MemoryProfile(7)],external_used_gib=2)

def test_same_gpu_max_two():assert not can([MemoryProfile(1)]*3)

def test_no_parallel_on_failed_parity():assert not can([MemoryProfile(2)]*2,parity_passed=False)

def test_no_parallel_if_it_slows_down():assert not can([MemoryProfile(2)]*2,concurrent_fixed_work_seconds=22)

def test_memory_headroom_rule():assert not can([MemoryProfile(9),MemoryProfile(10)])

@pytest.mark.parametrize('bad',[float('nan'),-1,float('inf')])
def test_invalid_memory_profile_rejected(bad):
    with pytest.raises(ValueError):MemoryProfile(bad).reserve()

def test_independent_trained_qt_is_required_by_gate():
    from test_contracts import good_metrics
    m=good_metrics();m['trained_QT_same_pool_R10']=.5
    ok,failed=repeat_gate(m)
    assert not ok and 'trained_qt_guard' in failed

def test_configuration_matches_all_controls_and_only_dual4090():
    p=json.loads((ROOT/'protocol.json').read_text())
    assert p['version']=='2.1'
    assert p['training']['max_stage_jobs_per_seed']==13
    assert p['training']['max_stage_jobs_total']==26
    assert len(p['training']['student_endpoints'])==6
    assert p['resources']['allowed_gpu_models']==['RTX 4090']
    assert p['resources']['gpu_count']==2
    assert p['teacher']['qt_control']['required'] and not p['teacher']['qt_control']['uses_QET_forward']
    assert not p['controls']['native']['adapter']
    assert p['controls']['direct_only']['C2_target_keys_trainable']

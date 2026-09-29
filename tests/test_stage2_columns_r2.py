import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from mmdd_stage2.column_data import natural_evidence
from mmdd_stage2.column_r2_cache import job, job_key, phase_a_variants
from mmdd_stage2.column_r2_metrics import paired_bootstrap, report_metrics
from mmdd_stage2.column_r2_phase_a import normalized_aggregates
from mmdd_stage2.column_metrics import prediction
from mmdd_stage2.column_r2_models import EvidenceCorrection, complete_scores, mix_condition, shortlist


def test_dev_early_stopping_warmup_patience_and_small_cumulative_gains():
    from mmdd_stage2.column_r2_training import DevPlateauStopper
    stop = DevPlateauStopper(patience=2, min_epochs=3, min_delta=.01)
    assert not stop.update(1, .8)
    assert not stop.update(2, .79)
    assert stop.bad_checks == 0
    assert not stop.update(3, .805)
    assert stop.bad_checks == 1
    # Improvements accumulate relative to the last meaningful best, not the
    # preceding noisy minibatch/epoch. This resets patience at epoch four.
    assert not stop.update(4, .811)
    assert stop.bad_checks == 0
    assert not stop.update(5, .810)
    assert stop.update(6, .810)
    disabled = DevPlateauStopper(patience=0, min_epochs=1)
    assert not any(disabled.update(epoch, .8) for epoch in range(1, 25))


def test_phase_a_normalizes_each_pass_before_aggregation():
    scores = torch.tensor([[2., 1., 0.], [-1., 0., 3.]])
    base = normalized_aggregates(scores)
    shifted = normalized_aggregates(scores + torch.tensor([[100.], [-400.]]))
    for a, b in zip(base, shifted):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(base[1].exp(), scores.softmax(-1).mean(0))


def test_variants_and_cache_identity_do_not_read_gold():
    item = {'dataset': 'd', 'query_id': 'q', 'target_id': 't', 'evidence_ids': {'O-R': ['a', 'b', 'c']}}
    assert phase_a_variants(item) == phase_a_variants({**item, 'gold_column_indices': [99]})
    assert job_key(job(item, [])) == job_key(job({**item, 'gold': [1]}, []))
    assert phase_a_variants(item)['loo_1'] == ['a', 'c']


def test_absent_natural_target_is_empty_without_oracle_fill():
    assert natural_evidence({'results': [{'target_id': 'other', 'paths': []}]}, 'gold') == []


def test_joint_metrics_include_prior_misses_and_net_correction():
    population = [{'dataset': 'd', 'query_id': str(i), 'target_id': 't', 'source_table_id': str(i),
                   'candidate_column_indices': [0,1,2,3], 'gold_column_indices': [g],
                   'natural_evidence_empty': False, 'modality': 'text', 'evidence_count': 1}
                  for i,g in enumerate([0,1,3])]
    prior = [prediction(p, [4.,3.,2.,1.]) for p in population]
    after = [prediction(p, [3.,4.,2.,1.]) for p in population]
    metrics, _ = report_metrics(population, after, prior)
    full = metrics['subsets']['full']
    assert full['pairs'] == 3
    assert full['corrected_cases'] == full['damaged_cases'] == 1
    assert full['net_correction_query_macro'] == 0
    assert full['PriorAdmissionAt3'] == pytest.approx(2/3)


def test_bootstrap_averages_targets_then_two_seeds_within_query():
    rows = [{'dataset': 'd', 'query_id': q, 'target_id': str(i), 'source_table_id': q, 'score': value}
            for i,(q,value) in enumerate([('a',1.),('a',1.),('b',0.)])]
    zero = [{**r, 'score': 0.} for r in rows]
    second = [{**r, 'score': 0.} for r in rows]
    result = paired_bootstrap([rows,second], [zero,zero], 'score')
    assert result['difference'] == .25
    assert result['queries'] == result['source_groups'] == 2
    assert result['replicates'] == 10000


def test_mix_schedule_is_fixed_and_near_half_without_evidence_or_label_inputs():
    values = [mix_condition(13, 1, str(i), 't') for i in range(1000)]
    assert values == [mix_condition(13, 1, str(i), 't') for i in range(1000)]
    assert 450 < values.count('O-R') < 550
    assert values != [mix_condition(13, 2, str(i), 't') for i in range(1000)]


@pytest.mark.parametrize('separate', [False, True])
def test_empty_evidence_is_exactly_prior(separate):
    model = EvidenceCorrection(4, separate).eval()
    prior = torch.tensor([.1, .3, -.2])
    result, attention = model(prior, torch.randn(3,8), torch.empty(0,3,8), torch.empty(0,dtype=torch.long))
    assert torch.equal(result, prior)
    assert attention is None


def test_separate_attention_is_column_conditioned_permutation_invariant():
    torch.manual_seed(3)
    model = EvidenceCorrection(4, True).eval()
    base, evidence, mods = torch.randn(3,8), torch.randn(4,3,8), torch.tensor([0,1,0,1])
    prior = torch.randn(3)
    result, attention = model(prior, base, evidence, mods)
    perm = torch.tensor([3,0,2,1])
    reordered, _ = model(prior, base, evidence[perm], mods[perm])
    torch.testing.assert_close(result, reordered)
    assert attention.shape == (4,3)
    assert not torch.equal(attention[:,0], attention[:,1])


def test_fixed_shortlist_never_inserts_outside_gold_and_preserves_tail():
    prior = torch.tensor([2.,5.,3.,4.,1.])
    columns = [0,1,2,3,4]
    selected = shortlist(prior, columns)
    assert selected == [1,3,2]
    scores = complete_scores(prior, torch.tensor([-10.,-12.,-11.]), selected, columns)
    record = {'dataset':'d','query_id':'q','target_id':'t','candidate_column_indices':columns}
    assert prediction(record, scores.tolist())['ranking'] == [1,2,3,0,4]


def test_multigold_column_loss_uses_total_accepted_mass():
    from mmdd_stage2.column_training import column_loss
    scores = torch.tensor([1.,2.,3.])
    assert column_loss(scores, [0,1,2], [0,2]) == pytest.approx(float(-scores.softmax(0)[[0,2]].sum().log()))


def test_prior_record_consumes_frozen_logits_without_running_prior():
    from types import SimpleNamespace
    from mmdd_stage2.column_metrics import pair_key
    from mmdd_stage2.column_r2_training import prior_record
    item = {'dataset':'d','query_id':'q','target_id':'t','gold_column_indices':[99]}
    record = {'input_hash':'fixed','candidate_column_indices':[3,1,2,0]}
    saved = {**record, 'prior_logits':[4.,3.,2.,1.], 'shortlist_columns':[3,1,2]}
    data = SimpleNamespace(get=lambda *args:record, prior_inputs_frozen=True,
                           fixed_prior={(pair_key(item),0):saved})
    def forbidden(*args):
        raise AssertionError('Prior should not be recomputed')
    _, logits, positions = prior_record(data, forbidden, item, 0)
    assert positions == [0,1,2]
    assert logits.tolist() == saved['prior_logits']
    with pytest.raises(ValueError,match='Missing frozen shortlist'):
        prior_record(data, forbidden, item, 1)


def test_support_witness_requires_complete_matching_review():
    from mmdd_stage2.column_r2_support import supported_witness
    population = {'gold_source_column_indices':[7], 'source_to_local':{'7':2}}
    recovery = {'recovered_attribute':{'column_index':7,'column_name':'color','value':'red'}}
    assert supported_witness(recovery, population) is None
    review = {'review_complete':True,'verdict':'supported','attribute_name':'color','claimed_value':'red'}
    recovery['auto_check'] = {'reviews':[review]}
    assert supported_witness(recovery, population) == 2
    for changed in ({'review_complete':False},{'verdict':'unknown'},{'claimed_value':'blue'},{'attribute_name':'size'}):
        recovery['auto_check']['reviews'] = [{**review,**changed}]
        assert supported_witness(recovery,population) is None


def test_correction_backward_never_updates_frozen_prior_and_checkpoint_roundtrips(tmp_path):
    from mmdd_stage2.column_r2_training import load_head
    from mmdd_stage2.verifier import CandidateColumnScorer
    prior = CandidateColumnScorer(4,head_type='mlp').eval().requires_grad_(False)
    h0 = torch.randn(3,8)
    base = prior(h0[:,:4],h0[:,4:])
    model = EvidenceCorrection(4,True)
    original = {k:v.clone() for k,v in prior.state_dict().items()}
    model(base,h0,torch.randn(2,3,8),torch.tensor([0,1]))[0].sum().backward()
    assert all(p.grad is None for p in prior.parameters())
    assert all(torch.equal(v,original[k]) for k,v in prior.state_dict().items())
    path = tmp_path/'selected.pt'
    torch.save({'metadata':{'arm':'PVR_SEPARATE','hidden_dim':4},'state_dict':model.state_dict()},path)
    restored, _ = load_head(path)
    for k,v in restored.state_dict().items():
        torch.testing.assert_close(v,model.state_dict()[k])


def test_cost_reconstructs_original_processor_image_pixels():
    from mmdd_stage2.column_r2_cost import processed_pixels
    pixels, _ = processed_pixels({'image_dimensions':{'img':{'reader':[512,512]}}})
    assert pixels == 262144
    assert processed_pixels({'image_pixels':123})[0] == 123


def test_flat_control_and_mix_run_matched_complete_schedules(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from mmdd_stage2 import column_r2_training as training
    from mmdd_stage2.column_data import write_json, write_jsonl
    from mmdd_stage2.column_r2_audit import read_json
    from mmdd_stage2.column_training import parameter_hash
    from mmdd_stage2.verifier import CandidateColumnScorer
    r1, output = tmp_path/'r1',tmp_path/'r2'
    r1.mkdir()
    inputs = [{'dataset':'d','query_id':f'q{i}','target_id':'t',
               'evidence_ids':{'O-O':['e'],'O-R':[] if i==0 else ['e']}} for i in range(2)]
    pop = [{**p, 'candidate_column_indices':[0,1,2,3], 'gold_column_indices':[i],
            'source_table_id':p['query_id']} for i,p in enumerate(inputs)]
    for split in ('train','dev'):
        write_jsonl(r1/f'COLUMN_POPULATION.{split}.jsonl',pop)
    write_json(r1/'READER_IDENTITY.json',{})
    write_json(output/'NATURAL_TRAIN/MANIFEST.json',{})
    torch.manual_seed(13)
    initial = parameter_hash(CandidateColumnScorer(4,head_type='mlp'))
    write_json(r1/'RUN_RECEIPTS/C2/13.json',{'initial_parameter_hash':initial})
    torch.manual_seed(77)
    opened, closed = torch.randn(4,4),torch.randn(4,4)
    class FakeData:
        def __init__(self,*args):
            self.objects = {'evidence':{'e':{'asset_type':'text'}}}
            self.loaded = {}
            self.index = SimpleNamespace(entries={})
        def get(self,item,ids,view):
            return {**item,'candidate_column_indices':[0,1,2,3],
                    'open_states':opened + .1*len(ids), 'close_states':closed + .01*view}
    monkeypatch.setattr(training,'TrainingData',FakeData)
    monkeypatch.setattr(training,'inputs_for',lambda *args:inputs)
    control = training.train_arm(r1,output,'OO_CONTROL',13)
    mixed = training.train_arm(r1,output,'FLAT_MIX',13)
    scalar_output = tmp_path/'scalar'
    scalar = training.train_arm(r1,output,'FLAT_MIX',13,execution='scalar',training_output=scalar_output)
    assert control['optimizer_steps'] == mixed['optimizer_steps'] == 20
    assert control['initial_parameter_sha256'] == mixed['initial_parameter_sha256'] == initial
    a = read_json(output/'OO_CONTROL/checkpoints/13/history.json')
    b = read_json(output/'FLAT_MIX/checkpoints/13/history.json')
    assert len(a) == len(b) == 20
    assert [h['visits_sha256'] for h in a] == [h['visits_sha256'] for h in b]
    assert all(h['base_samples_seen']==h['unique_base_visits']==2 for h in a+b)
    assert all(sum(h['conditions'].values())==2 for h in b)
    reference = read_json(scalar_output/'FLAT_MIX/checkpoints/13/history.json')
    assert scalar['head_execution'] == 'scalar' and mixed['head_execution'] == 'batched'
    assert scalar['optimizer_steps'] == mixed['optimizer_steps']
    for actual, expected in zip(b, reference):
        for key in ('base_samples_seen','unique_base_visits','visits_sha256',
                    'condition_and_evidence_schedule_sha256','admitted_samples','optimizer_steps'):
            assert actual[key] == expected[key]
        assert actual['train_loss'] == pytest.approx(expected['train_loss'], abs=2e-5)
    assert scalar['selected_epoch'] == mixed['selected_epoch']
    assert not (scalar_output/'NATURAL_TRAIN').exists()
    with pytest.raises(ValueError,match='immutable'):
        training.train_arm(r1,output,'OO_CONTROL',13)
    original_report = training.report_metrics
    def constant_selection_metrics(*args):
        report, rows = original_report(*args)
        report['query_macro'].update(MRR=.5, **{'ColHit@1':.5})
        return report, rows
    monkeypatch.setattr(training,'report_metrics',constant_selection_metrics)
    stopped = training.train_arm(r1,output,'PRIOR',13,early_stopping_patience=2,
                                 min_epochs=3,log_every_pairs=1)
    assert stopped['actual_epochs'] == stopped['optimizer_steps'] == 4
    assert stopped['stopped_early'] and stopped['stop_reason'] == 'dev_mrr_plateau'
    assert stopped['selected_epoch'] == 1
    history = read_json(output/'PRIOR/checkpoints/13/history.json')
    assert history[-1]['early_stop_triggered']
    assert all(row['base_samples_seen'] == 2 for row in history)
    assert all(row['loss_progress'][-1]['pairs_seen'] == 2 for row in history)
    selected = torch.load(output/'PRIOR/checkpoints/13/selected.pt',weights_only=True)
    assert selected['metadata']['selected_epoch'] == 1
    assert selected['metadata']['parameter_sha256'] == history[0]['parameter_sha256']


def test_final_report_is_generated_from_completed_artifacts(tmp_path):
    from mmdd_stage2.column_data import write_json
    from mmdd_stage2.column_r2_cost import cost_summary
    from mmdd_stage2.column_r2_evaluation import ARMS
    from mmdd_stage2.column_r2_report import build_report
    p = {'dataset':'d','query_id':'q','target_id':'t','source_table_id':'s',
         'candidate_column_indices':[0,1,2,3], 'gold_column_indices':[0],
         'natural_evidence_empty':False,'modality':'text','evidence_count':1}
    preds = [prediction(p,[4.,3.,2.,1.])]
    metrics, _ = report_metrics([p],preds,preds)
    results = {'test':{arm:{s:metrics for s in ('13','29')} for arm in ('R1_C2',)+ARMS}}
    interval = {'difference':0.,'ci95':[-.01,.01]}
    comparisons = ('FLAT_MIX-OO_CONTROL','FLAT_MIX-R1_C2','PVR_BUNDLE-FLAT_MIX',
                   'PVR_SEPARATE-FLAT_MIX','PVR_SEPARATE-PVR_BUNDLE',
                   'PVR_BUNDLE-PRIOR','PVR_SEPARATE-PRIOR')
    bootstrap = {'test':{c:{sub:{field:interval for field in ('ColHit@1','MRR')}
                               for sub in ('full','non_empty','M>3')} for c in comparisons}}
    phase_a = {s:{m:metrics for m in ('bundle','Mean','LME','oracle_single_evidence_upper_bound')} for s in ('13','29')}
    phase_a['paired_bootstrap'] = {m:{f:interval for f in ('ColHit@1','MRR')} for m in ('Mean','LME')}
    cost = cost_summary([{'reader_forwards':1,'total_tokens':100,'image_pixels':0,
                         'reader_latency_seconds':.1,'peak_vram_bytes':100,'cache_bytes':100}])
    costs = {'cold_per_qt':{'test':{'O-R':{a:{'full':cost} for a in
        ('PRIOR','FLAT','PVR_BUNDLE','PVR_SEPARATE','BUNDLE_EVIDENCE_ONLY','SEPARATE_EVIDENCE_ONLY')}}}}
    fixtures = {'FORMAL_TEST.json':{}, 'MODEL_SELECTION_LOCK.json':{'checkpoints':dict.fromkeys(ARMS,{})},
                'ANALYSIS/subgroup_metrics.json':results,'ANALYSIS/paired_bootstrap.json':bootstrap,
                'ANALYSIS/flip_analysis.json':{'cases':[]}, 'PHASE_A/RESULTS.json':phase_a,
                'SUPPORT_AUDIT/MANIFEST.json':{'pairs_with_explicit_witness':0,'execute':False},
                'COST/reader_costs.json':costs,'COST/cache_costs.json':{}}
    for name,value in fixtures.items():
        write_json(tmp_path/name,value)
    build_report(tmp_path/'unused',tmp_path)
    report = (tmp_path/'FINAL_REPORT.zh-CN.md').read_text()
    assert 'actually executed' in report and 'actually evaluated' in report
    assert 'PVR-Bundle' in report and 'PVR-Separate' in report
    assert '0.000 pp [-1.000, +1.000]' in report
    assert (tmp_path/'EXECUTION_MANIFEST.json').is_file()

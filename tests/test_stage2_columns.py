from pathlib import Path
import copy
import json

import pytest
import torch

from mmdd_stage2.data import CANDIDATE_OPEN, CANDIDATE_CLOSE, serialize_table, permute_table_columns, local_column_index
from mmdd_stage2.verifier import CandidateColumnScorer
from mmdd_stage2.checkpoints import save_candidate_scorer, load_candidate_scorer
from mmdd_stage2.column_metrics import prediction, evaluate, column_order
from mmdd_stage2.column_data import visible_table, natural_evidence, digest
from mmdd_stage2.column_cache import load_features, condition_input
from mmdd_stage2.column_training import column_loss


def table():
    return {'table_id': 'T', 'join_col': 7, 'columns': [
        {'column_index': 7, 'source_column_index': 30, 'column_name': 'Name'},
        {'column_index': 2, 'source_column_index': 10, 'column_name': 'Name'},
        {'column_index': 9, 'source_column_index': 20, 'column_name': CANDIDATE_OPEN}],
        'rows': [{'cells': [{'column_index': 7, 'text': 'A'}, {'column_index': 2, 'text': 'B'},
                           {'column_index': 9, 'text': CANDIDATE_CLOSE}]}]}


def pop(q='q', t='T', gold=None):
    return {'dataset': 'lake', 'query_id': q, 'target_id': t,
            'candidate_column_indices': [7, 2, 9], 'gold_column_indices': gold or [9]}


def test_tail_visibility_and_literal_escaping():
    original = table()
    changed = copy.deepcopy(original)
    changed['rows'][0]['cells'][0]['text'] = 'Z'
    for layout in ['header_markers_v0', 'tail_candidates_v1']:
        a, b = [serialize_table(t, mark_candidates=True, reader_layout_version=layout) for t in [original, changed]]
        assert a.count(CANDIDATE_OPEN) == a.count(CANDIDATE_CLOSE) == 3
        pos = a.rfind(CANDIDATE_CLOSE)
        assert (a[:pos] == b[:pos]) == (layout == 'header_markers_v0')
        if layout == 'tail_candidates_v1':
            assert a.index(CANDIDATE_OPEN) > a.index('Row:')


def test_non_contiguous_mapping_and_permutation_cells():
    t = table()
    assert local_column_index(t, 30) == 7
    for seed in range(8):
        shuffled = permute_table_columns(t, seed=seed)
        assert shuffled['rows'] == t['rows']
        values = {7: 'A', 2: 'B', 9: '&lt;|object_ref_end|>'}
        assert serialize_table(shuffled).split('Row: ')[1] == ' | '.join(values[c['column_index']] for c in shuffled['columns'])


def test_prediction_tie_rank_hit_consistent_and_label_free():
    p = pop()
    result = prediction(p, [0., 0., 0.])
    assert result['ranking'] == [2, 7, 9]
    assert 'gold_column_indices' not in result
    metrics, rows = evaluate([p], [result])
    assert rows[0]['rank'] == 3
    assert metrics['query_macro']['ColHit@1'] == 0
    assert metrics['query_macro']['MRR'] == pytest.approx(1/3)
    assert result['column_hypotheses'][0]['probability'] == pytest.approx(1/3)


def test_missing_and_failure_keep_denominator_with_query_macro():
    population = [pop('q', 'a'), pop('q', 'b'), pop('r', 'c')]
    outputs = [prediction(population[0], [0., 0., 1.]), {**pop('q', 'b'), 'status': 'OOM'}]
    m, _ = evaluate(population, outputs)
    assert m['pairs'] == 3 and m['queries'] == 2
    assert m['query_macro']['ColHit@5'] == .25
    assert m['pair_micro']['ColHit@5'] == pytest.approx(1/3)
    assert m['query_macro']['Random@5'] == 1
    assert m['failure_reasons'] == {'OOM': 1, 'missing_output': 1}


def test_multi_positive_rank_monotonic_and_chance():
    p = pop(gold=[7, 9])
    m, rows = evaluate([p], [prediction(p, [0., 1., 0.])])
    hits = [m['query_macro'][f'ColHit@{k}'] for k in [1, 2, 3, 5]]
    assert hits == [0., 1., 1., 1.]
    assert rows[0]['rank'] == 2
    assert m['query_macro']['Random@1'] == pytest.approx(2/3)
    assert m['query_macro']['Random@2'] == 1


def test_gold_cannot_change_predictions():
    assert prediction(pop(gold=[7]), [1., 2., 3.]) == prediction(pop(gold=[9]), [1., 2., 3.])


def test_metric_rejects_missing_candidates_and_duplicate_predictions():
    p = pop()
    result = prediction(p, [1., 2., 3.])
    with pytest.raises(ValueError):
        evaluate([p], [result, result])
    result['candidate_column_indices'] = [7, 9]
    with pytest.raises(ValueError):
        evaluate([p], [result])
    with pytest.raises(ValueError):
        column_order([float('nan')], [2])


def test_allowlist_removes_supervision_and_preserves_duplicate_names():
    t = table()
    t['hidden_attributes'] = [{'gold': True}]
    t['reason'] = 'GT'
    visible = visible_table(t)
    assert set(visible) == {'table_id', 'columns', 'rows'}
    assert all(set(c) == {'column_index', 'column_name'} for c in visible['columns'])
    assert len(visible['columns']) == 3


def test_natural_empty_keeps_target():
    assert natural_evidence({'results': [{'target_id': 'T', 'paths': [{'kind': 'direct'}]}]}, 'T') == []
    item = {'query_id': 'Q', 'target_id': 'T', 'evidence_ids': {'O-O': ['e'], 'O-R': []}}
    objects = {'queries': {'Q': visible_table(table())}, 'targets': {'T': visible_table(table())},
               'evidence': {'e': {'asset_id': 'e', 'asset_type': 'text', 'content': 'text'}}}
    for condition in ['O-R', 'No-E']:
        _, target, evidence = condition_input(item, objects, condition=condition, view=0)
        assert target['table_id'] == 'T' and len(target['columns']) == 3
        assert evidence == []
    assert digest(condition_input(item, objects, condition='No-E', view=0)) != digest(condition_input(item, objects, condition='O-O', view=0))


@pytest.mark.parametrize('head_type', ['linear', 'mlp'])
def test_checkpoint_reload_and_layout_rejection(tmp_path, head_type):
    head = CandidateColumnScorer(4, head_type=head_type).eval()
    assert head.input_dim == 8 and head.hidden_dim == 4
    path = tmp_path / 'head.pt'
    save_candidate_scorer(path, head, metadata={'reader_layout_version': 'tail_candidates_v1'})
    loaded = load_candidate_scorer(path, torch.device('cpu'), expected_reader_layout='tail_candidates_v1').eval()
    x, y = torch.randn(3, 4), torch.randn(3, 4)
    assert torch.equal(head(x, y), loaded(x, y))
    with pytest.raises(ValueError, match='layout'):
        load_candidate_scorer(path, torch.device('cpu'), expected_reader_layout='header_markers_v0')


def test_legacy_cache_rejected_for_new_training(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'format_version': 1, 'complete': True}))
    with pytest.raises(ValueError, match='Legacy'):
        load_features(path)


def test_set_mass_loss_and_gradient():
    logits = torch.tensor([0., 1., 2.], requires_grad=True)
    loss = column_loss(logits, [7, 2, 9], [7, 9])
    assert loss.item() == pytest.approx(-torch.log(torch.softmax(logits, 0)[[0, 2]].sum()).item())
    loss.backward()
    assert logits.grad[1] > 0 and logits.grad[0] < 0 and logits.grad[2] < 0


def test_cached_training_full_epochs_updates_reload_and_no_backbone(tmp_path):
    from mmdd_stage2.column_data import write_json, write_jsonl, file_hash
    from mmdd_stage2.column_training import train_head
    from mmdd_stage2.column_cache import load_features
    population = []
    entries = []
    for i in range(32):
        p = {'dataset': 'toy', 'query_id': f'q{i//2}', 'target_id': f't{i}',
             'candidate_column_indices': [7, 2], 'gold_column_indices': [7], 'split': 'train'}
        population.append(p)
        record = {k: p[k] for k in ('dataset', 'query_id', 'target_id', 'candidate_column_indices')}
        record.update(status='ok', input_hash=f'input{i}', reader_layout_version='tail_candidates_v1',
                      open_states=torch.tensor([[2., 1.], [-2., -1.]]), close_states=torch.tensor([[1., 2.], [-1., -2.]]))
        path = tmp_path/f'feature{i}.pt'
        torch.save(record, path)
        entries.append({**{k: v for k, v in record.items() if k not in ('open_states', 'close_states')},
                        'path': str(path), 'sha256': file_hash(path)})
    write_jsonl(tmp_path/'COLUMN_POPULATION.train.jsonl', population)
    write_jsonl(tmp_path/'COLUMN_POPULATION.dev.jsonl', [{**p, 'split': 'dev'} for p in population])
    write_json(tmp_path/'INPUT_MANIFEST.json', {'synthetic': True})
    paths = []
    for view, split in [(0, 'train'), (1, 'train'), (0, 'dev')]:
        path = tmp_path/f'{split}{view}.json'
        write_json(path, {'complete': True, 'contract': {'reader_layout_version': 'tail_candidates_v1',
                   'split': split, 'condition': 'O-O', 'view': view, 'input_manifest_sha256': file_hash(tmp_path/'INPUT_MANIFEST.json')},
                   'hidden_dim': 2, 'backbone_frozen': True, 'parameter_versions_unchanged': True, 'records': entries})
        paths.append(path)
    tiny = train_head(tmp_path, 'C2', 13, paths[:1], paths[2], tiny=True)
    assert tiny['passed'] and tiny['head_updated'] and tiny['reload_equal']
    receipt = train_head(tmp_path, 'C2', 13, paths[:2], paths[2], epochs=20)
    assert receipt['actual_epochs'] == 20 and receipt['optimizer_steps'] == 20
    history = json.loads(Path(receipt['history_path']).read_text())
    assert all(e['base_visits'] == e['unique_base_visits'] == 32 for e in history)
    assert [e['view'] for e in history] == [0, 1] * 10
    assert all(e['grad_norm_mean'] > 0 for e in history)
    with pytest.raises(ValueError, match='layout'):
        load_features(paths[0], expected_layout='header_markers_v0')


def test_cache_identical_visible_pairs_keep_identity_and_manifest_is_immutable(tmp_path, monkeypatch):
    from mmdd_stage2.column_cache import build_features, load_features
    from mmdd_stage2.column_data import write_json, write_jsonl, file_hash
    class Backend:
        model = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        device = torch.device('cpu')
        hidden_dim = 2
        last_reader_token_count = 12
        last_reader_image_policy = 'test'
        calls = 0
        def reader_states(self, query, target, evidence):
            self.calls += 1
            assert set(query) == {'table_id', 'columns', 'rows'}
            assert set(target) == {'table_id', 'columns', 'rows'}
            assert evidence == []
            return torch.ones(3, 2), torch.ones(3, 2)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *_: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *_: 0)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *_: 'synthetic')
    items = [{'dataset': 'toy', 'query_id': q, 'target_id': 'T', 'candidate_column_indices': [7, 2, 9],
              'evidence_ids': {'O-O': [], 'O-R': []}, 'natural_retrieval_available': True}
             for q in ['q1', 'q2']]
    objects = {'queries': {q: {**visible_table(table()), 'table_id': q} for q in ['q1', 'q2']},
               'targets': {'T': visible_table(table())}, 'evidence': {}}
    write_jsonl(tmp_path/'COLUMN_INPUTS.train.jsonl', items)
    write_jsonl(tmp_path/'COLUMN_POPULATION.train.jsonl', items)
    write_jsonl(tmp_path/'OBJECTS.jsonl.gz', [objects])
    write_json(tmp_path/'INPUT_MANIFEST.json', {'locked': True,
         'files': {p.name: file_hash(p) for p in tmp_path.glob('COLUMN_*.jsonl')},
         'objects_sha256': file_hash(tmp_path/'OBJECTS.jsonl.gz')})
    backend = Backend()
    kwargs = dict(layout='tail_candidates_v1', split='train', condition='O-O', view=0,
                  device='cpu', backend=backend, identity={'synthetic': True})
    path = build_features(tmp_path, tmp_path/'fake_model', **kwargs)
    original_hash = file_hash(path)
    _, records = load_features(path)
    assert [r['query_id'] for r in records] == ['q1', 'q2']
    assert len({r['input_hash'] for r in records}) == 2
    assert build_features(tmp_path, tmp_path/'fake_model', **kwargs) == path
    assert file_hash(path) == original_hash and backend.calls == 2

"""Sequential S2-COL-R1 execution: tiny check before full cache and training."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from typing import Any
import torch
from .column_cache import build_features, reader_identity
from .column_data import write_json, read_jsonl, write_jsonl, file_hash
from .column_training import train_head, evaluate_head
from .qwen import QwenStage2Backend


def visibility_probe(backend: QwenStage2Backend, output: Path) -> dict[str, Any]:
    query = {'table_id': 'Q', 'columns': [{'column_index': 8, 'column_name': 'Title'}],
             'rows': [{'cells': [{'column_index': 8, 'text': 'Atlas'}]}]}
    target = {'table_id': 'T', 'columns': [{'column_index': 7, 'column_name': 'Year'}, {'column_index': 2, 'column_name': 'Code'}],
              'rows': [{'cells': [{'column_index': 7, 'text': '2001'}, {'column_index': 2, 'text': '1001'}]}]}
    changed = copy.deepcopy(target)
    changed['rows'][0]['cells'][0]['text'] = '2002'
    changed['rows'][0]['cells'][1]['text'] = '1002'
    torch.manual_seed(13)
    weight = torch.randn(backend.hidden_dim * 2)
    results = {}
    for layout in ('header_markers_v0', 'tail_candidates_v1'):
        backend.reader_layout_version = layout
        states = []
        tokens = []
        for t in (target, target, changed):
            opened, closed = backend.reader_states(query, t, [])
            states.append(torch.cat((opened, closed), dim=-1))
            tokens.append(backend.last_reader_token_count)
        a, repeated, b = states
        results[layout] = {'repeat_max_abs': float((a - repeated).abs().max()),
                           'value_change_max_abs': float((a - b).abs().max()),
                           'repeat_logit_max_abs': float(((a - repeated) @ weight).abs().max()),
                           'value_change_logit_max_abs': float(((a - b) @ weight).abs().max()),
                           'token_counts': tokens, 'same_length': len(set(tokens)) == 1}
    results['passed'] = (all(r['same_length'] for r in results.values())
                         and results['tail_candidates_v1']['value_change_max_abs'] > results['tail_candidates_v1']['repeat_max_abs']
                         and results['header_markers_v0']['value_change_max_abs'] <= results['header_markers_v0']['repeat_max_abs'])
    write_json(output/'VISIBILITY_PROBE.json', results)
    return results


def run_experiment(output: Path, model_dir: Path, *, device: str = 'cuda:0', tiny_only: bool = False) -> None:
    identity = reader_identity(model_dir, image_pixels=262144)
    identity_path = output/'READER_IDENTITY.json'
    same_reader = identity_path.is_file() and json.loads(identity_path.read_text()) == identity
    write_json(output/'READER_IDENTITY.json', identity)
    backend = QwenStage2Backend(model_dir, device=device, reader_anonymize_evidence=True, reader_image_max_pixels=262144)
    previous_probe = output/'VISIBILITY_PROBE.json'
    probe = json.loads(previous_probe.read_text()) if same_reader and previous_probe.is_file() else visibility_probe(backend, output)
    if not probe['passed']:
        raise ValueError('Real-backbone visibility probe failed; inspect receipt before proceeding')
    def cache(layout: str, split: str, condition: str = 'O-O', view: int = 0, limit: int | None = None) -> Path:
        return build_features(output, model_dir, layout=layout, split=split, condition=condition, view=view,
                              device=device, limit=limit, backend=backend, identity=identity)
    tiny = {layout: cache(layout, 'train', limit=32) for layout in ('header_markers_v0', 'tail_candidates_v1')}
    write_json(output/'TINY_CACHE_PATHS.json', {k: str(v) for k, v in tiny.items()})
    for arm in ('C0', 'C1', 'C2'):
        path = tiny['header_markers_v0' if arm == 'C0' else 'tail_candidates_v1']
        for seed in (13, 29):
            receipt_path = output/'RUN_RECEIPTS'/arm/f'tiny_{seed}.json'
            result = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
            if not result or not result.get('passed') or result['cache_hashes'] != [file_hash(path)]:
                result = train_head(output, arm, seed, [path], path, tiny=True)
            if not result['passed']:
                raise ValueError(f'{arm}/{seed} tiny-overfit failed; formal training stopped')
    if tiny_only:
        return
    # Current historical scorer replay retains asset IDs, original column order,
    # and its original image policy. It is a reference, not a controlled arm.
    from .checkpoints import load_candidate_scorer
    from .column_training import predict
    from .column_metrics import evaluate
    historical_checkpoint = model_dir.parent.parent/'work/stage1_optimization_r25_final_20260914/stage2/r25_b13_column_scorer.pt'
    p0_records = []
    if historical_checkpoint.is_file():
        head = load_candidate_scorer(historical_checkpoint, torch.device('cpu'), expected_model_dir=model_dir)
        from .column_cache import load_features
        for split in ('dev', 'test'):
            for condition in ('O-O', 'O-R'):
                path = build_features(output, model_dir, layout='header_markers_v0', split=split, condition=condition,
                                      view=0, device=device, backend=backend, identity=identity, historical_input=True)
                _, features = load_features(path, expected_layout='header_markers_v0')
                predictions = predict(head, features, file_hash(historical_checkpoint))
                metrics, _ = evaluate(read_jsonl(output/f'COLUMN_POPULATION.{split}.jsonl'), predictions)
                name = f'{split}.{condition}.original'
                write_jsonl(output/'PREDICTIONS/P0-R25/reference'/f'{name}.jsonl.gz', predictions)
                write_json(output/'METRICS/P0-R25/reference'/f'{name}.json', metrics)
                p0_records.append({'split': split, 'condition': condition, 'cache_manifest': str(path)})
        write_json(output/'RUN_RECEIPTS/P0-R25/reference.json', {'planned': True, 'implemented': True, 'executed': True,
            'evaluated': True, 'checkpoint_sha256': file_hash(historical_checkpoint), 'records': p0_records,
            'scope': 'Unchanged historical scorer and reader format on current frozen oracle targets; not reproduction of original training/evaluation population'})
    else:
        write_json(output/'RUN_RECEIPTS/P0-R25/reference.json', {'planned': True, 'implemented': True,
            'executed': False, 'evaluated': False, 'reason': f'Missing historical scorer: {historical_checkpoint}'})
    caches = {}
    for layout in ('header_markers_v0', 'tail_candidates_v1'):
        training = [cache(layout, 'train', view=v) for v in (0, 1)]
        dev = cache(layout, 'dev')
        caches[layout] = {'train': training, 'dev': dev}
        for arm in (('C0',) if layout == 'header_markers_v0' else ('C1', 'C2')):
            for seed in (13, 29):
                train_head(output, arm, seed, training, dev)
    def recipe_score(arm: str) -> tuple[float, ...]:
        scores = []
        for seed in (13, 29):
            receipt = json.loads((output/'RUN_RECEIPTS'/arm/f'{seed}.json').read_text())
            history = json.loads(Path(receipt['history_path']).read_text())
            scores.append(history[receipt['selected_epoch'] - 1]['dev_query_macro'])
        return tuple(sum(s[m] for s in scores) / len(scores) for m in ('MRR', 'ColHit@3', 'ColHit@1'))
    diagnostic_arm = max(('C1', 'C2'), key=recipe_score)
    write_json(output/'DIAGNOSTIC_SELECTION.json', {'arm': diagnostic_arm, 'rule': 'mean across seeds of selected dev query-macro MRR, Hit3, Hit1; C1 wins ties'})
    # Test is evaluated only after all per-arm dev selections are immutable.
    for layout in ('header_markers_v0', 'tail_candidates_v1'):
        arms = ('C0',) if layout == 'header_markers_v0' else ('C1', 'C2')
        for split in ('dev', 'test'):
            for condition in ('O-O', 'O-R'):
                path = caches[layout]['dev'] if split == 'dev' and condition == 'O-O' else cache(layout, split, condition)
                for arm in arms:
                    for seed in (13, 29):
                        evaluate_head(output, output/'CHECKPOINTS'/arm/str(seed)/'selected.pt', path, arm, seed)
    # Diagnostics on fixed dev only; no test-based tuning.
    for condition, view in [('No-E', 0), ('Shuffled-E', 0), ('ValueShuffle', 0), ('O-O', 2)]:
        path = cache('tail_candidates_v1', 'dev', condition, view)
        for seed in (13, 29):
            evaluate_head(output, output/'CHECKPOINTS'/diagnostic_arm/str(seed)/'selected.pt', path, diagnostic_arm, seed)
    from .column_diagnostics import position_baseline, c50_compatibility, paired_differences
    position_baseline(output)
    c50_compatibility(output)
    paired_differences(output)
    from .column_reporting import report
    report(output)

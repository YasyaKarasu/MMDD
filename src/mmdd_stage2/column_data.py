"""Freeze column population and keep supervision outside reader inputs."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mmdd_dataset.utils import get_cell
from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from .data import local_column_index, serialize_table
from .oracle import _selected_artifact_records, _resolve_source_backed_targets, select_oracle_evidence


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == '.gz' else open
    temporary = path.with_name(path.name + '.tmp')
    with opener(temporary, 'wt', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def visible_table(table: dict[str, Any]) -> dict[str, Any]:
    """Allowlist only visible cells; no source IDs, roles, or hidden attributes."""
    columns = [{'column_index': int(c['column_index']), 'column_name': c.get('column_name', '')} for c in table['columns']]
    return {'table_id': table['table_id'], 'columns': columns,
            'rows': [{'cells': [{'column_index': c['column_index'], 'text': get_cell(r, c['column_index']).get('text', '')}
                                for c in columns]} for r in table['rows'][:12]]}


def natural_evidence(record: dict[str, Any] | None, target: str) -> list[str]:
    if record is None:
        return []
    result = next((x for x in record.get('results', []) if x['target_id'] == target), {})
    return list(dict.fromkeys(str(p['evidence_id']) for p in result.get('paths', [])
                             if p.get('kind') == 'evidence' and p.get('evidence_id')))[:4]


def _read_candidate_scope(path: Path) -> tuple[dict[tuple[str, str], set[str]], dict[tuple[str, str, str], list[int]]]:
    """Read a frozen target or candidate-column scope used to build a C30 population.

    The scope is deliberately a separate input contract.  JSON mappings keyed by
    ``dataset|query_id|target_id`` and JSON/JSONL records with the three identity
    fields are accepted.  A Stage-1 ``{"query": {"C30": [...]}}`` export is
    interpreted as a target-table scope.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix in {'.jsonl', '.gz'}:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and 'records' in payload:
            rows = payload['records']
        else:
            rows = payload
        if isinstance(rows, dict):
            rows = [{key: value} for key, value in rows.items()]
    target_result: dict[tuple[str, str], set[str]] = {}
    column_result: dict[tuple[str, str, str], list[int]] = {}
    for row in rows:
        # Compact Stage-1 exports often use {query_id: {"C30": [target_ids]}}.
        if not isinstance(row, dict):
            raise ValueError('candidate scope records must be objects')
        if len(row) == 1 and isinstance(next(iter(row.values())), dict):
            query_id, payload = next(iter(row.items()))
            targets = payload.get('C30', payload.get('c30', payload.get('target_ids')))
            if targets is not None:
                target_result.setdefault(('', str(query_id)), set()).update(map(str, targets))
                continue
        if len(row) == 1 and isinstance(next(iter(row.values())), list):
            compact_key, columns = next(iter(row.items()))
            parts = str(compact_key).split('|')
            if len(parts) != 3:
                raise ValueError(f'invalid candidate scope key: {compact_key!r}')
            identity = tuple(parts)
            values = [int(column) for column in columns]
            if not values or len(set(values)) != len(values):
                raise ValueError(f'candidate scope columns must be unique and non-empty: {compact_key!r}')
            column_result[identity] = values
            continue
        key = row.get('scope_key') or row.get('key')
        if key and not {'dataset', 'query_id', 'target_id'} <= row.keys():
            parts = str(key).split('|')
            if len(parts) != 3:
                raise ValueError(f'invalid candidate scope key: {key!r}')
            dataset, query_id, target_id = parts
        else:
            try:
                dataset, query_id, target_id = (str(row[k]) for k in ('dataset', 'query_id', 'target_id'))
            except KeyError as exc:
                raise ValueError('candidate scope record lacks dataset/query_id/target_id') from exc
        target_ids = row.get('candidate_target_ids', row.get('target_ids'))
        if target_ids is not None:
            target_result.setdefault((dataset, query_id), set()).update(map(str, target_ids))
            continue
        columns = row.get('candidate_column_indices', row.get('columns'))
        if columns is None:
            raise ValueError(f'candidate scope record lacks candidate columns: {row!r}')
        values = [int(column) for column in columns]
        if not values or len(set(values)) != len(values):
            raise ValueError(f'candidate scope columns must be unique and non-empty: {row!r}')
        identity = (dataset, query_id, target_id)
        if identity in column_result and column_result[identity] != values:
            raise ValueError(f'conflicting candidate scope records: {identity}')
        column_result[identity] = values
    if not target_result and not column_result:
        raise ValueError(f'candidate scope is empty: {path}')
    return target_result, column_result


load_candidate_scope = _read_candidate_scope


def audit_data(roots: list[Path], output: Path, retrieval_paths: list[Path],
               candidate_scope_file: Path | None = None) -> dict[str, Any]:
    target_scope, column_scope = _read_candidate_scope(candidate_scope_file) if candidate_scope_file else ({}, {})
    retrieval = {}
    for path in retrieval_paths:
        for r in read_jsonl(path):
            q = r['query_id']
            if q in retrieval and digest(r) != digest(retrieval[q]):
                raise ValueError(f'Conflicting retrieval records: {q}')
            retrieval[q] = r
    populations, inputs, issues, root_audits = [], [], [], []
    objects = {'queries': {}, 'targets': {}, 'evidence': {}}
    split_groups = {k: defaultdict(set) for k in ('source', 'chain', 'query_content', 'entity')}
    for root in roots:
        if not (root / 'qrels.jsonl').is_file():
            root_audits.append({'root': str(root), 'status': 'missing_qrels_and_canonical_artifacts'})
            continue
        lake = 'entitables' if 'entitables' in root.name else 'wdc' if 'wdc' in root.name else root.name
        qrels_all = list(iter_dataset_artifact(root, 'qrels'))
        qrels = [r for r in qrels_all if r.get('reason') == 'model_recoverable_join_column']
        pairs = defaultdict(list)
        unique_records = set()
        exact_duplicates = 0
        for r in qrels:
            fingerprint = digest(r)
            if fingerprint in unique_records:
                exact_duplicates += 1
                continue
            unique_records.add(fingerprint)
            pairs[(r['query_table_id'], r['target_table_id'])].append(r)
        queries = _selected_artifact_records(root, 'query_tables', {q for q, _ in pairs})
        targets = _selected_artifact_records(root, 'data_lake_tables', {t for _, t in pairs})
        _resolve_source_backed_targets(root, targets)
        recoveries = defaultdict(list)
        for r in iter_dataset_artifact(root, 'evidence_recoveries'):
            key = (r['query_table_id'], r['target_table_id'])
            if key in pairs:
                recoveries[key].append(r)
        asset_ids = {r['evidence']['asset_id'] for rs in recoveries.values() for r in rs}
        asset_ids.update(e for q, t in pairs for e in natural_evidence(retrieval.get(q), t))
        evidence = _selected_artifact_records(root, 'bridge_assets', asset_ids)
        for eid, e in evidence.items():
            if e.get('asset_type') == 'image':
                local = Path(e.get('local_path', ''))
                relative = root / e.get('relative_path', '')
                e['local_path'] = str(local if local.is_file() else relative)
            objects['evidence'][eid] = {k: e[k] for k in ('asset_id', 'asset_type', 'content', 'local_path') if k in e}
        for (qid, tid), records in sorted(pairs.items()):
            base = {'dataset': lake, 'query_id': qid, 'target_id': tid}
            try:
                query, target = queries[qid], targets[tid]
                split_set = {r['split'] for r in records}
                if len(split_set) != 1 or not split_set <= {'train', 'dev', 'test'}:
                    raise ValueError('conflicting_split')
                split = next(iter(split_set))
                if query.get('split') != split:
                    raise ValueError('object_split_mismatch')
                if query.get('query_kind', 'implicit') != 'implicit':
                    raise ValueError('reason_query_kind_mismatch')
                columns = [int(c['column_index']) for c in target['columns']]
                sources = [int(c.get('source_column_index', c['column_index'])) for c in target['columns']]
                if len(set(columns)) != len(columns) or len(set(sources)) != len(sources):
                    raise ValueError('ambiguous_column_mapping')
                gold_sources = sorted({int(r['join_attribute']['source_column_index']) for r in records})
                gold = sorted({local_column_index(target, s) for s in gold_sources})
                q_sources = {int(c.get('source_column_index', c['column_index'])) for c in query['columns']}
                if q_sources.intersection(gold_sources):
                    raise ValueError('bridge_attribute_visible_in_query')
            except (KeyError, ValueError) as error:
                issues.append({**base, 'reason': str(error)})
                continue
            if candidate_scope_file is not None:
                scope_key = (lake, qid, tid)
                query_targets = target_scope.get((lake, qid), target_scope.get(('', qid)))
                if target_scope and query_targets is None:
                    raise ValueError(f'candidate target scope missing query: {(lake, qid)}')
                if query_targets is not None and tid not in query_targets:
                    continue
                if column_scope:
                    if scope_key not in column_scope:
                        raise ValueError(f'candidate scope missing pair: {scope_key}')
                    selected_columns = column_scope[scope_key]
                    unknown = set(selected_columns) - set(columns)
                    if unknown:
                        raise ValueError(f'candidate scope has unknown columns for {scope_key}: {sorted(unknown)}')
                    if not set(gold).issubset(selected_columns):
                        # A qrel outside C30 is not a supervised pair in the C30 run.
                        continue
                    source_by_local = dict(zip(columns, sources))
                    columns = selected_columns
                    sources = [source_by_local[c] for c in columns]
                    gold = sorted(set(gold).intersection(columns))
            for qrel in records:
                split_groups['source'][(lake, qrel['source_table_id'])].add(split)
                split_groups['chain'][(lake, qrel.get('chain_id', ''))].add(split)
            split_groups['query_content'][(lake, digest(serialize_table(query)))].add(split)
            for row in query['rows']:
                for cell in row['cells']:
                    entity = cell.get('wiki_title') or (cell.get('text') if cell.get('column_name') == 'entity_url' else None)
                    if entity:
                        split_groups['entity'][(lake, entity)].add(split)
            positive_ids = sorted({r['evidence']['asset_id'] for r in recoveries[(qid, tid)]})
            selected = list(select_oracle_evidence([e for e in positive_ids if e in evidence], evidence, top_k=4))
            natural = natural_evidence(retrieval.get(qid), tid)
            # Upstream recovery checks do not establish support after reader truncation.
            meta = {**base, 'split': split, 'candidate_column_indices': columns,
                    'gold_column_indices': gold, 'gold_source_column_indices': gold_sources,
                    'positive_evidence_ids': positive_ids,
                    'source_to_local': dict(zip(map(str, sources), columns)),
                    'source_table_id': records[0]['source_table_id'],
                    'support_status': 'support_unknown',
                    'upstream_supported_recoveries': sum(bool(r.get('auto_check', {}).get('supported_attributes')) for r in recoveries[(qid, tid)]),
                    'modality': '+'.join(sorted({evidence[e]['asset_type'] for e in selected})) or 'none',
                    'natural_evidence_empty': not natural,
                    'natural_retrieval_available': qid in retrieval}
            populations.append(meta)
            inputs.append({**base, 'split': split, 'candidate_column_indices': columns,
                           'evidence_ids': {'O-O': selected, 'O-R': natural},
                           'natural_retrieval_available': qid in retrieval})
            objects['queries'][qid] = visible_table(query)
            objects['targets'][tid] = visible_table(target)
        root_audits.append({'root': str(root), 'status': 'audited', 'qrels_sha256': file_hash(root / 'qrels.jsonl'),
                            'implicit_qrels': len(qrels), 'exact_duplicates': exact_duplicates,
                            'explicit_qrels': len(qrels_all) - len(qrels),
                            'explicit_evaluation': 'N/A: missing-attribute task excludes direct visible-column labels',
                            'query_kind_contract': 'implicit inferred from recoverable reason and hidden source-column absence when query_kind is absent'})
    overlaps = {kind: [{'identity': list(k), 'splits': sorted(v)} for k, v in groups.items() if len(v) > 1]
                for kind, groups in split_groups.items()}
    duplicate_content = {tuple(r['identity']) for r in overlaps['query_content']}
    excluded_keys = {(r['dataset'], r['query_id']) for r in populations
                     if (r['dataset'], digest(serialize_table(objects['queries'][r['query_id']]))) in duplicate_content}
    duplicate_exclusions = [r for r in populations if (r['dataset'], r['query_id']) in excluded_keys]
    populations = [r for r in populations if (r['dataset'], r['query_id']) not in excluded_keys]
    inputs = [r for r in inputs if (r['dataset'], r['query_id']) not in excluded_keys]
    audit = {'protocol': 'canonical train/dev/test; no R12 split remapping', 'roots': root_audits,
             'candidate_scope': ({'path': str(candidate_scope_file.resolve()),
                                  'sha256': file_hash(candidate_scope_file),
                                  'pairs': len(target_scope) + len(column_scope), 'applied': True} if candidate_scope_file is not None
                                 else {'applied': False}),
             'excluded_before_population_lock': issues, 'cross_split_overlaps': overlaps,
             'duplicate_content_exclusions': duplicate_exclusions,
             'duplicate_policy': 'exclude all exact visible query-content groups spanning splits before lock; retain published splits',
             'entity_overlap_policy': 'report shared entity URLs/titles; source-group protocol, not entity-disjoint generalization',
             'split_counts': dict(Counter(r['split'] for r in populations)),
             'support': 'Upstream supported recoveries are not a post-truncation support audit; all retained as support_unknown',
             'population_locked': not issues and not overlaps['source'] and not overlaps['chain']}
    audit['missing_evidence_objects'] = sorted({eid for item in inputs for ids in item['evidence_ids'].values()
                                               for eid in ids if eid not in objects['evidence']})
    audit['missing_image_files'] = [eid for eid, e in objects['evidence'].items()
                                    if e.get('asset_type') == 'image' and not Path(e.get('local_path', '')).is_file()]
    audit['natural_retrieval_coverage'] = {split: {
        'available_pairs': sum(r['natural_retrieval_available'] for r in populations if r['split'] == split),
        'total_pairs': sum(r['split'] == split for r in populations),
        'empty_evidence_given_available_retrieval': sum(r['natural_retrieval_available'] and r['natural_evidence_empty']
                                                      for r in populations if r['split'] == split)}
        for split in ('train', 'dev', 'test')}
    write_json(output / 'DATA_AUDIT.json', audit)
    for split in ('train', 'dev', 'test'):
        write_jsonl(output / f'COLUMN_POPULATION.{split}.jsonl', [r for r in populations if r['split'] == split])
        write_jsonl(output / f'COLUMN_INPUTS.{split}.jsonl', [r for r in inputs if r['split'] == split])
    write_jsonl(output / 'OBJECTS.jsonl.gz', [objects])
    manifest = {'protocol': audit['protocol'], 'locked': audit['population_locked'], 'roots': root_audits,
                'files': {p.name: file_hash(p) for p in sorted(output.glob('COLUMN_*.jsonl'))},
                'objects_sha256': file_hash(output / 'OBJECTS.jsonl.gz'),
                'retrieval': [{'path': str(p), 'sha256': file_hash(p)} for p in retrieval_paths]}
    write_json(output / 'INPUT_MANIFEST.json', manifest)
    return audit

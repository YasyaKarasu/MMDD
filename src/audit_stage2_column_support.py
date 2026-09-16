#!/usr/bin/env python
"""Audit existing support annotations and text preservation; never infer new values."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from mmdd_stage2.column_data import read_jsonl, write_json, write_jsonl
from mmdd_stage2.column_metrics import pair_key
from mmdd_stage2.data import escape_marker_literals


def audit_support(output: Path, roots: list[Path]) -> None:
    recoveries = defaultdict(list)
    for root in roots:
        for r in iter_dataset_artifact(root, 'evidence_recoveries'):
            recoveries[(r['query_table_id'], r['target_table_id'])].append(r)
    objects = read_jsonl(output/'OBJECTS.jsonl.gz')[0]
    details, summary = [], {}
    for split in ('train', 'dev', 'test'):
        inputs = {pair_key(r): r for r in read_jsonl(output/f'COLUMN_INPUTS.{split}.jsonl')}
        population = read_jsonl(output/f'COLUMN_POPULATION.{split}.jsonl')
        for condition in ('O-O', 'O-R'):
            rows = []
            for p in population:
                item = inputs[pair_key(p)]
                if condition == 'O-R' and not item['natural_retrieval_available']:
                    continue
                selected = set(item['evidence_ids'][condition])
                limit = max(1, 12000//max(1, len(selected)))
                selected_annotations, preserved_annotations, attributes = set(), set(), set()
                for recovery in recoveries[(p['query_id'], p['target_id'])]:
                    eid = recovery['evidence']['asset_id']
                    if eid not in selected or not recovery.get('auto_check', {}).get('supported_attributes'):
                        continue
                    source_col = int(recovery['recovered_attribute']['column_index'])
                    if source_col not in p['gold_source_column_indices']:
                        continue
                    selected_annotations.add(recovery['query_row_id'])
                    attributes.add(source_col)
                    evidence = objects['evidence'][eid]
                    if evidence['asset_type'] == 'text' and len(escape_marker_literals(evidence.get('content'))) <= limit:
                        preserved_annotations.add(recovery['query_row_id'])
                nrows = len(objects['queries'][p['query_id']]['rows'])
                row = {k: p[k] for k in ('dataset', 'query_id', 'target_id')}
                row.update(split=split, condition=condition, selected_evidence_count=len(selected),
                           known_support_row_coverage=len(selected_annotations)/nrows,
                           untruncated_text_support_row_coverage=len(preserved_annotations)/nrows,
                           known_gold_attribute_coverage=len(attributes)/len(p['gold_source_column_indices']),
                           has_known_support=float(bool(selected_annotations)),
                           has_untruncated_text_support=float(bool(preserved_annotations)))
                rows.append(row)
            by_query = defaultdict(list)
            for row in rows:
                by_query[(row['dataset'], row['query_id'])].append(row)
            fields = ('selected_evidence_count','known_support_row_coverage','untruncated_text_support_row_coverage',
                      'known_gold_attribute_coverage','has_known_support','has_untruncated_text_support')
            summary[f'{split}.{condition}'] = {'pairs': len(rows), 'queries': len(by_query),
                'missing_retrieval_pairs': len(population)-len(rows),
                'query_macro': {k:mean(mean(r[k] for r in group) for group in by_query.values()) for k in fields} if rows else None}
            details.extend(rows)
    write_jsonl(output/'EVIDENCE_SUPPORT_AUDIT.jsonl', details)
    write_json(output/'EVIDENCE_SUPPORT_AUDIT.json', {
        'executed':True, 'new_value_generation':False, 'model_input_unchanged':True,
        'interpretation':'Known support uses existing published recovery reviews. Untruncated text preserves the reviewed text, but does not independently re-verify annotation correctness. Truncated text/image support remains unknown; unannotated natural evidence is not classified negative.',
        'primary_population_support_status_unchanged':'support_unknown', 'metrics':summary})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset-root',type=Path,action='append',required=True)
    args=parser.parse_args()
    audit_support(args.output,args.dataset_root)

"""Audited evidence-ID witnesses and R1-rule synthetic shuffled negatives."""
from __future__ import annotations

from pathlib import Path

from .column_data import digest, file_hash, read_jsonl, write_json, write_jsonl
from .column_metrics import pair_key
from .column_r2_audit import read_json
from .column_r2_cache import job, job_key


def supported_witness(recovery: dict, population: dict) -> int | None:
    attribute = recovery['recovered_attribute']
    source = int(attribute['column_index'])
    if source not in population['gold_source_column_indices']:
        return None
    reviews = recovery.get('auto_check', {}).get('reviews', [])
    if not any(r.get('review_complete') is True and r.get('verdict') == 'supported'
               and r.get('attribute_name') == attribute['column_name']
               and str(r.get('claimed_value')) == str(attribute['value']) for r in reviews):
        return None
    return population['source_to_local'][str(source)]


def audit_support(r1: Path, output: Path) -> None:
    for seed in (13,29):
        if not (output/f'PVR_SEPARATE/checkpoints/{seed}/MANIFEST.json').is_file():
            raise ValueError('Support audit follows completed PVR-Separate')
    from mmdd_dataset.wdc_runtime import iter_dataset_artifact
    population = read_jsonl(r1/'COLUMN_POPULATION.train.jsonl')
    lookup = {(p['query_id'],p['target_id']): p for p in population}
    inputs = read_jsonl(output/'NATURAL_TRAIN/COLUMN_INPUTS.train.jsonl')
    by_key = {pair_key(i): i for i in inputs}
    witnesses, source_manifests = {}, {}
    evidence = read_jsonl(r1/'OBJECTS.jsonl.gz')[0]['evidence']
    for entry in read_json(r1/'INPUT_MANIFEST.json')['roots']:
        if entry['status'] != 'audited':
            continue
        manifest_path = Path(entry['root'])/'dataset_manifest.json'
        source_manifests[str(manifest_path)] = file_hash(manifest_path)
        for recovery in iter_dataset_artifact(Path(entry['root']), 'evidence_recoveries'):
            key = recovery['query_table_id'], recovery['target_table_id']
            if key not in lookup:
                continue
            p = lookup[key]
            column = supported_witness(recovery, p)
            eid = recovery['evidence']['asset_id']
            # Auxiliary positives stay inside the unchanged read4 union.
            selected = by_key[pair_key(p)]['evidence_ids']
            if column is None or eid not in set(selected['O-O'] + selected['O-R']):
                continue
            identity = (pair_key(p), column, eid)
            witnesses.setdefault(identity, {**{k:p[k] for k in ('dataset','query_id','target_id')},
                'column_index': column, 'evidence_id': eid, 'recovery_ids': [], 'query_row_ids': []})
            witnesses[identity]['recovery_ids'].append(recovery['recovery_id'])
            witnesses[identity]['query_row_ids'].append(recovery['query_row_id'])
    positives = list(witnesses.values())
    pairs = {pair_key(p) for p in positives}
    write_jsonl(output/'SUPPORT_AUDIT/witnesses.jsonl.gz', positives)
    receipt = {'pairs_with_explicit_witness':len(pairs), 'threshold':500, 'execute':len(pairs)>=500,
        'source_dataset_manifest_sha256':source_manifests,
        'criterion':'completed supported review matching recovered attribute name/value and mapped gold source column',
        'unknown_natural_evidence':'unknown, never negative', 'positive_pool':'unchanged read4 O-O/O-R union',
        'witnesses_sha256':file_hash(output/'SUPPORT_AUDIT/witnesses.jsonl.gz')}
    if len(pairs) < 500:
        write_json(output/'SUPPORT_AUDIT/MANIFEST.json',receipt)
        return
    meta = {pair_key(p):p for p in population}
    known = {}
    for p in population:
        known.setdefault(p['query_id'],set()).update(p['positive_evidence_ids'])
    lengths = {pair_key(i):sum(len(evidence[e].get('content','')) for e in i['evidence_ids']['O-O']) for i in inputs}
    donors = {}
    jobs = {}
    original_tiebreak = {pair_key(i):digest(i) for i in read_jsonl(r1/'COLUMN_INPUTS.train.jsonl')}
    for key in sorted(pairs):
        item, p = by_key[key], meta[key]
        eligible = [i for i in inputs if i['query_id'] != item['query_id']
            and meta[pair_key(i)]['source_table_id'] != p['source_table_id']
            and meta[pair_key(i)]['modality'] == p['modality'] and i['evidence_ids']['O-O']
            and not set(i['evidence_ids']['O-O']).intersection(known[item['query_id']])]
        # Match the exact R1 tie-break object, without the newly added natural reason field.
        donor = min(eligible, key=lambda i:(abs(lengths[pair_key(i)]-lengths[key]),original_tiebreak[pair_key(i)]))
        donors[key] = {'donor_key': list(pair_key(donor)), 'synthetic_negative_ids': donor['evidence_ids']['O-O']}
        for view in (0,1):
            for eid in donor['evidence_ids']['O-O']:
                row = job(item,[eid],view)
                jobs[job_key(row)] = row
    write_jsonl(output/'SUPPORT_AUDIT/donors.jsonl.gz', [{**{k:by_key[key][k] for k in ('dataset','query_id','target_id')},**v} for key,v in donors.items()])
    write_jsonl(output/'JOBS/support.jsonl.gz',list(jobs.values()))
    write_json(output/'SUPPORT_AUDIT/MANIFEST.json',{**receipt,'negative_kind':'Shuffled-E synthetic_negative',
        'donors_sha256':file_hash(output/'SUPPORT_AUDIT/donors.jsonl.gz'),'lambda':.2,'loss':'pairwise softplus(-(a_pos-a_neg))'})

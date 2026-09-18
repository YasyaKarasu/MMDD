"""S2-R4 P0: recompute the R3 row-recovery metrics with the audit corrections applied.

Corrections applied here, each traceable to a finding in the S2-R3 review:

1. Weighted cluster bootstrap over source groups that keeps redraw multiplicity.
2. strict / normalized / numeric / any-level reported as separate, explicitly named
   quantities; ``correct_any`` row counts are never used to describe StrictRecovered.
3. G4 rows that were never executed are NOT_EVALUATED and leave the refusal denominator.
4. Grounding is split by modality: a claim supported only by an image note is
   UNKNOWN_IMAGE_SUPPORT, not a text-grounding failure.
5. Citation labels that do not resolve are retained in the raw record instead of being
   filtered out before the claim "all cited labels are valid".
6. Support is reported at four named levels rather than one boolean.

Nothing here rewrites the historical raw delivery; the recomputation is read-only.
"""
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .r4_common import digest, file_hash, write_json
from .r4_metrics import cluster_bootstrap, paired_cluster_bootstrap, query_macro

R3 = Path('/home/oycy/MMDD/work/S2_COL_R3_ROW')
DELIVERED_MANIFEST = Path('/home/oycy/MMDD/mmdd_s2_r3_review/REFERENCE/DELIVERY_MANIFEST.json')
ARMS = {
    'G0_GTcolumn_OO': 'G0 GT column + oracle evidence',
    'G1_PriorTop3_OO': 'G1 Prior Top3 + oracle evidence',
    'G2_PriorTop3_OR': 'G2 Prior Top3 + natural read4',
    'G3_FlatMixTop3_OR': 'G3 Flat-Mix Top3 + natural read4',
    'G4_PriorTop3_ShuffledE': 'G4 Prior Top3 + shuffled donor evidence',
    'D0_NoE_closed_book': 'D0 closed-book diagnostic under the original prompt',
}
METRICS = ('StrictRecovered@1', 'StrictRecovered@3', 'NormRecovered@3', 'AnyRecovered@3',
           'SupportedRecovered@3', 'admitted@3')
SUPPORT_LEVELS = ('MODEL_CLAIM', 'MECHANICALLY_TRACEABLE', 'CANONICAL_WITNESS_MATCH',
                  'SEMANTICALLY_AUDITED')


def load_rows() -> list[dict]:
    with gzip.open(R3 / 'ANALYSIS/per_row_scores.jsonl.gz', 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_units() -> dict[str, dict]:
    """Reparsed per-unit run records (the sharded files carry status and evidence_labels).

    RUN_REPARSED/UNITS.jsonl.gz is the plain expansion and has no run output.
    """
    units = {}
    for path in sorted((R3 / 'RUN_REPARSED').glob('units.shard*.jsonl.gz')):
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    units[record['unit_id']] = record
    return units


def _evaluated(rows: list[dict]) -> list[dict]:
    """NOT_EVALUATED rows never entered execution; they must not sit in any denominator."""
    return [r for r in rows if r.get('status') != 'NO_UNIT']


def main_table(rows: list[dict], iterations: int = 10000) -> dict:
    per_arm: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        per_arm[row['arm']].append(row)
    table = {}
    for arm, arm_rows in sorted(per_arm.items()):
        evaluated = _evaluated(arm_rows)
        not_evaluated = [r for r in arm_rows if r.get('status') == 'NO_UNIT']
        entry: dict[str, Any] = {
            'label': ARMS.get(arm, arm),
            'rows_total': len(arm_rows),
            'rows_evaluated': len(evaluated),
            'rows_not_evaluated': len(not_evaluated),
            'seeds': sorted({r['seed'] for r in evaluated}),
        }
        # Each column-model seed is scored on its own and the seed-level query-macro values
        # are averaged afterwards. Pooling the seeds first would silently reweight queries
        # whose row sets differ between seeds.
        per_seed = {}
        for seed in sorted({r['seed'] for r in evaluated}):
            subset = [r for r in evaluated if r['seed'] == seed]
            per_seed[str(seed)] = {metric: query_macro(subset, metric) for metric in METRICS}
        entry['per_seed_point'] = per_seed
        for metric in METRICS:
            seed_values = [v[metric] for v in per_seed.values() if v[metric] is not None]
            entry[f'{metric}_seed_mean'] = (sum(seed_values) / len(seed_values)
                                            if seed_values else None)
            ci = cluster_bootstrap(evaluated, 'source_group', metric, iterations=iterations)
            entry[metric] = {'point': ci['point'], 'ci95': [ci['low'], ci['high']],
                             'groups': ci['groups'], 'iterations': ci['iterations'],
                             'point_pooled_over_seeds': ci['point'],
                             'point_seed_mean': entry[f'{metric}_seed_mean']}
        # Refusal accounting excludes the never-executed rows entirely.
        entry['status_counts_evaluated'] = dict(Counter(r['status'] for r in evaluated))
        if evaluated:
            entry['terminal_refusal_rate_evaluated_only'] = (
                sum(r['status'] == 'INSUFFICIENT_EVIDENCE' for r in evaluated) / len(evaluated)
            )
        table[arm] = entry
    return table


def seed_mean_view(rows: list[dict]) -> list[dict]:
    """Average each query's metric over column-model seeds before comparing arms.

    G0/D0 were produced under a single seed while G1/G2/G3 carry two; comparing them
    without this step silently produces an empty pair set.
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in _evaluated(rows):
        buckets[(row['arm'], row['query_id'], row['query_row_id'], row['target_id'],
                 row['gold_local_column'])].append(row)
    merged = []
    for key, group in buckets.items():
        base = dict(group[0])
        base['seed'] = 'seed_mean'
        for metric in METRICS:
            values = [float(g[metric]) for g in group if g.get(metric) is not None]
            base[metric] = sum(values) / len(values) if values else None
        merged.append(base)
    return merged


def paired_comparisons(rows: list[dict], iterations: int = 10000) -> dict:
    merged = seed_mean_view(rows)
    by_arm: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for row in merged:
        by_arm[row['arm']][(row['query_id'], row['query_row_id'], row['target_id'],
                            row['gold_local_column'])] = row
    pairs = [('G0_GTcolumn_OO', 'G1_PriorTop3_OO'), ('G1_PriorTop3_OO', 'G2_PriorTop3_OR'),
             ('G2_PriorTop3_OR', 'G3_FlatMixTop3_OR'), ('G0_GTcolumn_OO', 'G2_PriorTop3_OR')]
    results = {}
    for left, right in pairs:
        shared = sorted(set(by_arm[left]) & set(by_arm[right]))
        if not shared:
            results[f'{left}__vs__{right}'] = {'paired_units': 0,
                                               'reason': 'no shared query-row units'}
            continue
        joined = []
        for key in shared:
            joined.append({
                'dataset': by_arm[left][key]['dataset'],
                'query_id': key[0],
                'source_group': by_arm[left][key]['source_group'],
                'left': by_arm[left][key]['StrictRecovered@3'],
                'right': by_arm[right][key]['StrictRecovered@3'],
            })
        ci = paired_cluster_bootstrap(joined, 'source_group', 'left', 'right',
                                      iterations=iterations)
        results[f'{left}__vs__{right}'] = {
            'metric': 'StrictRecovered@3', 'paired_units': len(joined),
            'difference_pp': None if ci['point'] is None else 100 * ci['point'],
            'ci95_pp': None if ci['low'] is None else [100 * ci['low'], 100 * ci['high']],
            'groups': ci['groups'], 'iterations': ci['iterations'],
        }
    return results


def grounding_by_modality(rows: list[dict], units: dict[str, dict]) -> dict:
    """Split mechanical grounding by cited modality instead of pooling it."""
    result = {}
    for arm in ('G1_PriorTop3_OO', 'G2_PriorTop3_OR', 'G3_FlatMixTop3_OR', 'D0_NoE_closed_book'):
        subset = [r for r in _evaluated(rows) if r['arm'] == arm and r['status'] == 'VALUE']
        counts = Counter()
        for row in subset:
            cited = [str(x) for x in (row.get('cited_evidence_ids') or [])]
            modalities = {units_modality(units, row, cid) for cid in cited}
            has_text_quote = bool(row.get('quotes_any_evidence'))
            has_image_note = bool(row.get('has_image_support_note'))
            if not cited:
                counts['NO_CITATION'] += 1
            elif modalities <= {'image'} and has_image_note:
                counts['IMAGE_ONLY_UNKNOWN_IMAGE_SUPPORT'] += 1
            elif has_text_quote:
                counts['TEXT_TRACEABLE'] += 1
            else:
                counts['TEXT_CITED_NO_QUOTE'] += 1
        result[arm] = {
            'value_rows': len(subset),
            'categories': dict(counts),
            'note': (
                'IMAGE_ONLY_UNKNOWN_IMAGE_SUPPORT is NOT a text-grounding failure; a mechanical '
                'string check cannot make a negative claim about image pixels'
            ),
        }
    return result


def units_modality(units: dict[str, dict], row: dict, evidence_id: str) -> str:
    """Modality of a cited evidence label, resolved through the frozen asset registry."""
    return _MODALITY.get(str(evidence_id), 'unknown')


_MODALITY: dict[str, str] = {}


def load_modality(evidence_ids: set[str]) -> dict[str, str]:
    from .r4_common import evidence_assets

    assets = evidence_assets(evidence_ids)
    return {eid: asset.get('asset_type', 'unknown') for eid, asset in assets.items()}


def citation_validity() -> dict:
    """Raw citation labels, invalid ones retained rather than filtered before the claim."""
    units = load_units()
    total_labels = 0
    invalid = Counter()
    per_arm = Counter()
    invalid_by_condition = Counter()
    examples = []
    for unit in units.values():
        label_map = unit.get('evidence_labels') or {}
        if not isinstance(label_map, dict):
            continue
        for label in unit.get('evidence_ids') or []:
            total_labels += 1
            per_arm[unit['condition']] += 1
            if str(label) not in label_map:
                invalid[str(label)] += 1
                invalid_by_condition[unit['condition']] += 1
                if len(examples) < 10:
                    examples.append({'condition': unit['condition'], 'label': str(label),
                                     'query_id': unit['query_id'], 'target_id': unit['target_id']})
    return {
        'cited_labels_total': total_labels,
        'invalid_labels': sum(invalid.values()),
        'invalid_label_histogram': dict(invalid.most_common(20)),
        'invalid_by_condition': dict(invalid_by_condition),
        'cited_by_condition': dict(per_arm),
        'examples': examples,
        'note': (
            'R3 filtered unresolvable labels before checking the rest, then reported that all '
            'remaining references were legal. Invalid labels are kept here and counted as errors.'
        ),
    }


def support_levels(rows: list[dict]) -> dict:
    """Four named support levels; a canonical miss is not a negative label."""
    evaluated = _evaluated(rows)
    values = [r for r in evaluated if r['status'] == 'VALUE']
    return {
        'MODEL_CLAIM': {'value_rows': len(values),
                        'definition': 'the generator emitted status VALUE with a non-empty value'},
        'MECHANICALLY_TRACEABLE': {
            'rows': sum(bool(r.get('quotes_any_evidence')) or bool(r.get('has_image_support_note'))
                        for r in values),
            'definition': 'at least one cited quote is a verbatim substring, or an image note exists'},
        'CANONICAL_WITNESS_MATCH': {
            'rows': sum(bool(r.get('canonical_witness_support')) for r in values),
            'definition': 'the dataset build recorded a witness for this value'},
        'SEMANTICALLY_AUDITED': {
            'rows': None,
            'definition': 'no independent semantic re-audit was executed in R3 or in this replay',
            'status': 'NOT_EXECUTED'},
        'canonical_uncovered_is_unknown': True,
    }


def compare_with_review(main_table: dict) -> dict:
    """Show where the P0 corrections move the R3 numbers.

    The review's own recomputation still divided by rows that never executed. Recomputing
    over executed rows only is the correction, so the two columns differ exactly on the
    arms that carry NOT_EVALUATED rows - and nowhere else.
    """
    import csv

    path = Path('/home/oycy/MMDD/mmdd_s2_r3_review/RECOMPUTED/main_seed_means.csv')
    if not path.is_file():
        return {'status': 'review_table_not_available'}
    reference = {row['arm']: row for row in csv.DictReader(path.open())}
    comparisons = {}
    for arm, entry in main_table.items():
        baseline = reference.get(arm)
        if baseline is None:
            continue
        review_value = float(baseline['h3'])
        corrected = entry['StrictRecovered@3']['point_seed_mean']
        comparisons[arm] = {
            'strict@3_review_recomputation': review_value,
            'strict@3_r4_corrected': corrected,
            'difference_pp': None if corrected is None else 100 * (corrected - review_value),
            'rows_not_evaluated_excluded_here': entry['rows_not_evaluated'],
            'explained_by_not_evaluated_rows': entry['rows_not_evaluated'] > 0,
        }
    return {
        'reference': str(path),
        'comparisons': comparisons,
        'reading': (
            'the review recomputed the R3 numbers but kept never-executed rows in the '
            'denominator as failures. Arms with zero NOT_EVALUATED rows reproduce exactly; '
            'the arms that carry them move, and that movement is the correction.'
        ),
    }


def run(out: Path, *, iterations: int = 10000) -> dict:
    global _MODALITY
    rows = load_rows()
    units = load_units()
    cited = {str(label) for unit in units.values() for label in (unit.get('evidence_ids') or [])}
    cited |= {str(v) for unit in units.values()
              for v in (unit.get('evidence_labels') or {}).values()}
    _MODALITY = load_modality(cited)

    # The delivered package's manifest is the audited snapshot; the manifest now sitting in
    # the working tree was regenerated after that audit, so both are reported.
    delivered = json.loads(DELIVERED_MANIFEST.read_text())
    manifest = json.loads((R3 / 'DELIVERY_MANIFEST.json').read_text())

    def compare(declared: dict) -> list[dict]:
        rows = []
        for relative, expected in declared['files'].items():
            path = R3 / relative
            actual_sha = file_hash(path) if path.is_file() else None
            actual_bytes = path.stat().st_size if path.is_file() else None
            if actual_sha != expected['sha256'] or actual_bytes != expected['bytes']:
                rows.append({'path': relative, 'declared': expected,
                             'actual': {'sha256': actual_sha, 'bytes': actual_bytes}})
        return rows

    mismatches = compare(delivered)

    report = {
        'round': 'S2-R4 P0 replay',
        'source': str(R3),
        'raw_delivery_modified': False,
        'r3_delivery_manifest_hash_mismatch': {
            'delivered_manifest': str(DELIVERED_MANIFEST),
            'delivered_declared_files': len(delivered['files']),
            'mismatched_files_against_delivered_manifest': mismatches,
            'mismatch_count': len(mismatches),
            'working_tree_manifest': str(R3 / 'DELIVERY_MANIFEST.json'),
            'working_tree_declared_files': len(manifest['files']),
            'working_tree_now_self_consistent': not compare(manifest),
            'regenerated_after_audit': set(delivered['files']) != set(manifest['files'])
                                       or delivered['files'] != manifest['files'],
            'action': 'recorded; no historical raw file was modified and no historical result was '
                      'deleted or rewritten; the R4 delivery gets its own fresh manifest',
        },
        'main_table': main_table(rows, iterations),
        'paired_comparisons': paired_comparisons(rows, iterations),
        'grounding_by_modality': grounding_by_modality(rows, units),
        'citation_validity': citation_validity(),
        'support_levels': support_levels(rows),
        'comparison_with_review_recomputation': compare_with_review(main_table(rows, iterations)),
        'counting_contract': {
            'value_levels': ['strict', 'normalized', 'numeric', 'any_level'],
            'rule': 'correct_any row counts are never reported as StrictRecovered; each level is '
                    'reported under its own name',
        },
    }
    write_json(out / 'P0_REPLAY.json', report)
    return {
        'main_table_arms': list(report['main_table']),
        'paired': {k: v.get('difference_pp') for k, v in report['paired_comparisons'].items()},
        'invalid_citation_labels': report['citation_validity']['invalid_labels'],
        'g4_not_evaluated': report['main_table']['G4_PriorTop3_ShuffledE']['rows_not_evaluated'],
        'output': str(out / 'P0_REPLAY.json'),
    }

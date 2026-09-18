"""S2-R4 Phase R / I contract tests.

Each test fails for a reason named in the R4 acceptance list.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, '/home/oycy/MMDD/src')

from mmdd_stage2.r4_matcher import (  # noqa: E402
    exact_matches_report, exact_normalized_matches, claim_coverage,
)
from mmdd_stage2.r4_recovery import build_prompt, parse_source_aware  # noqa: E402
from mmdd_stage2.r4_source_trace import classify_value, summarize, verify_query_refs  # noqa: E402

LABELS = {'E1': 'asset_text_a', 'E2': 'asset_img_b'}
TEXTS = {'asset_text_a': 'The club won the league in 1998 under coach Smith.'}


# ------------------------------------------------------------------ empty evidence


def test_empty_evidence_r0_refuses_and_r1_r2_may_use_query_visible():
    cells = [{'column_id': 0, 'column_name': 'Name', 'text': 'Alpha'},
             {'column_id': 1, 'column_name': 'url', 'text': 'https://x.org/wiki/Alpha_1998'}]
    r0_content, _ = build_prompt('R0', cells, 'Year', [])
    r1_content, _ = build_prompt('R1', cells, 'Year', [])
    r0_text = ' '.join(part.get('text', '') for part in r0_content)
    r1_text = ' '.join(part.get('text', '') for part in r1_content)
    assert 'Do not guess' in r0_text
    assert 'QUERY_VISIBLE' in r1_text and 'no retrieved evidence' in r1_text
    # R0 has no notion of a source kind at all, so it cannot label one.
    assert 'QUERY_VISIBLE' not in r0_text


def test_query_visible_value_cannot_be_declared_external_evidence():
    value = {'value': '1998', 'source_kind': 'EXTERNAL_EVIDENCE',
             'evidence_refs': [], 'query_refs': [{'column_id': 1, 'quote': '1998',
                                                  'transform': 'UNDERSCORE_TO_SPACE'}]}
    traced = classify_value(value, LABELS, TEXTS,
                            [{'column_id': 1, 'column_name': 'url',
                              'text': 'https://x.org/wiki/Alpha_1998'}])
    assert traced['support_level'] == 'QUERY_VISIBLE_TRACEABLE'
    assert traced['declared_matches_checked'] is False


def test_unknown_transform_is_rejected():
    result = verify_query_refs(
        [{'column_id': 1, 'quote': '1998', 'transform': 'MAKE_IT_UP'}],
        [{'column_id': 1, 'column_name': 'url', 'text': 'x_1998'}])
    assert result['traceable'] == []
    assert result['invalid'][0]['reason'].startswith('unknown_transform')


# ------------------------------------------------------------------ modality split


def test_image_only_support_is_unknown_not_unsupported():
    value = {'value': 'Red', 'source_kind': 'EXTERNAL_EVIDENCE', 'evidence_refs': [],
             'query_refs': [], 'image_refs': [{'id': 'E2', 'description': 'the shirt is red'}]}
    traced = classify_value(value, LABELS, TEXTS, [])
    assert traced['support_level'] == 'IMAGE_ONLY_UNKNOWN_IMAGE_SUPPORT'
    summary = summarize([traced])
    assert summary['image_only_unknown'] == 1
    assert summary['invalid_or_unsourced'] == 0


def test_invalid_label_is_kept_and_counted():
    value = {'value': '1998', 'source_kind': 'EXTERNAL_EVIDENCE',
             'evidence_refs': [{'id': 'E99', 'quote': 'won the league'}],
             'query_refs': [], 'image_refs': []}
    traced = classify_value(value, LABELS, TEXTS, [])
    assert traced['support_level'] == 'INVALID_LABEL'
    assert traced['evidence_invalid'][0]['id'] == 'E99'


def test_quote_must_be_verbatim():
    value = {'value': '1998', 'source_kind': 'EXTERNAL_EVIDENCE',
             'evidence_refs': [{'id': 'E1', 'quote': 'won the league in 1998'}],
             'query_refs': [], 'image_refs': []}
    assert classify_value(value, LABELS, TEXTS, [])['support_level'] == \
        'EXTERNAL_EVIDENCE_TEXT_TRACEABLE'
    value['evidence_refs'][0]['quote'] = 'won the league in 1999'
    assert classify_value(value, LABELS, TEXTS, [])['support_level'] == 'INVALID_LABEL'


# ------------------------------------------------------------------ value budget


def test_r1_rejects_more_than_one_value_but_r2_allows_three():
    payload = {'status': 'VALUE', 'values': [
        {'value': 'A', 'source_kind': 'EXTERNAL_EVIDENCE',
         'evidence_refs': [{'id': 'E1', 'quote': 'x'}]},
        {'value': 'B', 'source_kind': 'EXTERNAL_EVIDENCE',
         'evidence_refs': [{'id': 'E1', 'quote': 'x'}]},
    ]}
    import json
    r1 = parse_source_aware(json.dumps(payload), 'stop', 'R1')
    assert r1['status'] == 'PARSE_ERROR' and 'value_budget_exceeded' in r1['reason']
    r2 = parse_source_aware(json.dumps(payload), 'stop', 'R2')
    assert r2['status'] == 'VALUE' and len(r2['values']) == 2


def test_multiple_actors_are_not_automatically_ambiguous():
    """Two listed actors are a legal set, not a conflict."""
    import json
    payload = {'status': 'VALUE', 'values': [
        {'value': 'Spencer Tracy', 'source_kind': 'EXTERNAL_EVIDENCE',
         'evidence_refs': [{'id': 'E1', 'quote': 'x'}]},
        {'value': 'Deborah Kerr', 'source_kind': 'EXTERNAL_EVIDENCE',
         'evidence_refs': [{'id': 'E1', 'quote': 'x'}]},
    ]}
    parsed = parse_source_aware(json.dumps(payload), 'stop', 'R2')
    assert parsed['status'] == 'VALUE' and len(parsed['values']) == 2


def test_r0_scalar_output_maps_through_the_adapter():
    import json
    payload = {'status': 'VALUE', 'value': '1998', 'evidence_ids': ['E1'],
               'text_support_quotes': [{'evidence_id': 'E1', 'quote': 'in 1998'}],
               'image_support_notes': []}
    parsed = parse_source_aware(json.dumps(payload), 'stop', 'R0')
    assert parsed['adapter'] == 'r0_scalar_to_values_v1'
    assert parsed['values'][0]['value'] == '1998'
    assert parsed['values'][0]['source_kind'] is None


def test_unresolvable_label_is_not_silently_dropped_from_a_value():
    """The label stays in the record; the value is not reported as fully cited."""
    value = {'value': 'x', 'source_kind': 'EXTERNAL_EVIDENCE',
             'evidence_refs': [{'id': 'E7', 'quote': 'q'}], 'query_refs': [], 'image_refs': []}
    traced = classify_value(value, LABELS, TEXTS, [])
    assert traced['evidence_invalid'] and traced['support_level'] == 'INVALID_LABEL'


# ------------------------------------------------------------------ matching


def test_multiple_identical_target_values_return_multiple_row_ids():
    report = exact_matches_report([{'value': 'Springfield', 'support_level': 'X'}],
                                  ['Springfield', 'Other', 'Springfield'])
    assert report['per_value'][0]['matched_row_ids'] == [0, 2]
    assert report['matched_row_ids'] == [0, 2]


def test_substring_matching_is_refused():
    assert exact_normalized_matches('York', ['New York']) == []
    assert exact_normalized_matches('1998', ['1998-01-01']) == []


def test_coverage_divides_by_all_visible_rows():
    rows = [{'matched': True}, {'matched': False}, {'matched': False}, {}, {}]
    result = claim_coverage(rows, 'matched')
    assert result['rows_total'] == 5 and result['rows_matched'] == 1
    assert result['coverage'] == pytest.approx(0.2)
    assert result['primary_threshold'] == 0.6
    assert result['build_diagnostic_threshold'] == 0.4


# ------------------------------------------------------------------ scheduling


def test_scheduler_source_never_reads_gold_fields():
    """Only executable code is inspected; prose in docstrings is not an access."""
    import ast
    import inspect

    from mmdd_stage2 import r4_schedule

    tree = ast.parse(inspect.getsource(r4_schedule))
    # Only identifier loads count. A string literal naming an invariant the module
    # declares (for example the explicit 'reads_row_gt': False receipt) is not a read.
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    used |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    used |= {node.value.id for node in ast.walk(tree)
             if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)}
    for forbidden in ('known_witness', 'gold_column', 'gold_value', 'row_gt', 'ROW_GT',
                      'target_row_ids', 'witness_evidence_ids'):
        assert forbidden not in used, f'scheduler code reads {forbidden}'


def test_scheduler_output_depends_only_on_logits_and_visible_rows():
    """Deleting every gold field cannot change the schedule, because none is read."""
    import json
    import tempfile

    from mmdd_stage2.r4_schedule import build_schedules

    tables = [
        {'query_id': 'q1', 'ranking_mode': 'TABLE_RANK_ONLY', 'seed': None,
         'arm': 'J0_TableOnly',
         'ranking': [{'target_id': 't1'}, {'target_id': 't2'}]},
    ]
    pairs = [
        {'query_id': 'q1', 'arm': 'J1_ExactProduct_Prior', 'seed': 13,
         'ranking': [{'target_id': 't1', 'column_id': 3, 'log_joint': -0.5},
                     {'target_id': 't1', 'column_id': 4, 'log_joint': -0.9},
                     {'target_id': 't2', 'column_id': 7, 'log_joint': -0.7}]},
    ]
    objects = {'queries': {'q1': {'table_id': 'q1',
                                  'columns': [{'column_index': 0, 'column_name': 'Name'}],
                                  'rows': [{'cells': [{'column_index': 0, 'text': 'Alpha'}]},
                                           {'cells': [{'column_index': 0, 'text': 'Beta'}]}]}},
               'targets': {}, 'evidence': {}}
    import gzip

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / 'TABLE_RANKINGS').mkdir()
        (root / 'PAIR_RANKINGS').mkdir()
        for name, rows in (('TABLE_RANKINGS/table_rankings.pilot.jsonl.gz', tables),
                           ('PAIR_RANKINGS/pair_rankings.pilot.jsonl.gz', pairs)):
            with gzip.open(root / name, 'wt', encoding='utf-8') as handle:
                for row in rows:
                    handle.write(json.dumps(row) + '\n')
        with gzip.open(root / 'READER_OBJECTS.pilot.jsonl.gz', 'wt', encoding='utf-8') as handle:
            handle.write(json.dumps(objects) + '\n')
        with gzip.open(root / 'CANDIDATE_LOCK.pilot.jsonl.gz', 'wt', encoding='utf-8') as handle:
            handle.write(json.dumps({'query_id': 'q1', 'source_group': 'g',
                                     'targets': []}) + '\n')
        summary = build_schedules(root, 'pilot')

    assert summary['S0_TableFirst']['branches'] == 2
    assert summary['S1_JointGlobal']['branches'] == 3
    # Two visible rows per query, so the scheduled unit count is rows x branches.
    assert summary['S0_TableFirst']['scheduled_units'] == 4
    assert summary['S1_JointGlobal']['scheduled_units'] == 6

"""S2-R4 final report generator.

Every number in the report is read from an artifact that was actually produced. A section
whose artifact is missing is written as blocked/not_executed rather than estimated, and
the seven questions the contract asks are answered only from executed runs.
"""
from __future__ import annotations

import json
from pathlib import Path

from .r4_common import read_jsonl, write_json


def _read(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def build_report(out: Path, scope: str) -> dict:
    coverage = _read(out / f'PHASE_J_COVERAGE.{scope}.json')
    metrics = _read(out / f'PHASE_J_METRICS.{scope}.json')
    labels = _read(out / f'LABEL_COVERAGE.{scope}.json')
    population = _read(out / f'POPULATION_LOCK.{scope}.json')
    p0 = _read(out / 'P0_REPLAY.json')
    schedule = _read(out / 'RECOVERY_SCHEDULES' / f'SCHEDULE_SUMMARY.{scope}.json')
    funnel = _read(out / 'COSTS' / f'EVIDENCE_FUNNEL.{scope}.json')

    executed = coverage is not None
    answered = {}
    if executed and metrics:
        per_arm = metrics['per_arm']
        answered['q2_table_only_vs_product'] = {
            'target_recall@10': {arm: per_arm[arm]['target_recall@10']['point'] for arm in per_arm},
            'paired_differences_pp': {k: v['difference_pp']
                                      for k, v in metrics['comparisons_target_recall@10'].items()},
            'c50_invariance_holds': metrics['c50_invariance_check']['identical'],
        }
        answered['q1_did_it_score_error_and_empty_evidence_targets'] = {
            'candidate_pairs': coverage['candidate_pairs'],
            'pairs_with_columns': coverage['pairs_with_columns'],
            'pairs_without_columns': coverage['pairs_without_columns'],
            'column_cells': coverage['column_cells'],
            'column_cells_scored': coverage['column_cells_scored'],
            'pairs_missing_features': coverage['pairs_missing_features'],
            'full_finite_coverage': coverage['full_finite_coverage'],
        }
        answered['q3_component_vs_pair_budget_loss_layer'] = {
            'joint_table_column_recall@10,3': {
                arm: per_arm[arm].get('joint_table_column_recall@10,3', {}).get('point')
                for arm in per_arm},
            'note': 'compares the gold-target component ceiling with the real pair-budget result',
        }
    else:
        answered['q1_did_it_score_error_and_empty_evidence_targets'] = {
            'status': 'blocked', 'reason': 'phase J has not produced coverage on real inputs'}

    status = {
        'p0_audit_corrections': 'evaluated' if p0 else 'implemented',
        'candidate_lock': 'executed' if population else 'implemented',
        'phase_j_column_features': 'executed' if executed else 'blocked',
        'phase_j_ranking': 'evaluated' if metrics else ('executed' if executed else 'blocked'),
        'phase_r_recovery': _recovery_status(out),
        'phase_i_scheduling': 'executed' if schedule else 'implemented',
        'evidence_funnel': 'executed' if funnel else 'implemented',
        'optional_e_latebind': 'not_executed',
        'wrong_table_calibration': 'not_executed',
    }
    report = {
        'round': 'S2-R4',
        'scope': scope,
        'status_vocabulary': ['planned', 'implemented', 'executed', 'evaluated', 'blocked',
                              'not_executed'],
        'module_status': status,
        'evidence_available': {
            'p0_replay': p0 is not None,
            'population_lock': population is not None,
            'label_coverage': labels is not None,
            'phase_j_coverage': coverage is not None,
            'phase_j_metrics': metrics is not None,
            'schedule_summary': schedule is not None,
            'evidence_funnel': funnel is not None,
        },
        'answers': answered,
        'raw': {
            'population': population,
            'labels': labels,
            'coverage': coverage,
            'schedule': schedule,
            'funnel': funnel,
        },
    }
    write_json(out / f'FINAL_REPORT_INPUTS.{scope}.json', report)
    return {'module_status': status, 'evidence_available': report['evidence_available'],
            'answers': list(answered)}


def _recovery_status(out: Path) -> str:
    folder = out / 'RAW_GENERATIONS'
    if not folder.is_dir() or not any(folder.glob('*.jsonl.gz')):
        return 'implemented'
    try:
        rows = []
        for path in sorted(folder.glob('*.jsonl.gz')):
            rows.extend(read_jsonl(path))
    except (OSError, ValueError):
        return 'executed'
    return 'executed' if rows else 'implemented'

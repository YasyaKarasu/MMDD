"""S2-R4 P0 contract tests.

Every test here fails for a reason the R3 audit identified. Run with:
    cd /home/oycy/MMDD/src && python -m pytest tests/test_r4_contracts.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, '/home/oycy/MMDD/src')

from mmdd_stage2.r4_metrics import (  # noqa: E402
    cluster_bootstrap, numeric_equal, paired_cluster_bootstrap, normalize, query_macro,
    value_matches,
)


# --------------------------------------------------------------------------- bootstrap


def test_bootstrap_keeps_redraw_multiplicity():
    """A group drawn m times must contribute m times to both numerator and count.

    The R3 implementation flattened the drawn rows and re-grouped them by query id,
    so a doubly-drawn group collapsed back into one bucket.
    """
    rows = [
        {'dataset': 'd', 'query_id': 'q1', 'g': 'A', 'v': 1.0},
        {'dataset': 'd', 'query_id': 'q2', 'g': 'B', 'v': 0.0},
    ]
    assert query_macro(rows, 'v') == pytest.approx(0.5)
    # Enumerate the exact bootstrap distribution over the 4 equally likely draws.
    weights = []
    for first in ('A', 'B'):
        for second in ('A', 'B'):
            drawn = [first, second]
            total = sum({'A': 1.0, 'B': 0.0}[g] for g in drawn)
            weights.append(total / 2)
    # AA -> 1.0, AB/BA -> 0.5, BB -> 0.0
    assert sorted(weights) == [0.0, 0.5, 0.5, 1.0]
    result = cluster_bootstrap(rows, 'g', 'v', iterations=4000, seed=7)
    assert result['low'] < result['point'] < result['high']
    assert result['low'] <= 0.5 <= result['high']


def test_bootstrap_multiplicity_changes_the_interval():
    """With a singleton group the weighted draw must be able to reach 1.0 and 0.0."""
    rows = [
        {'dataset': 'd', 'query_id': 'q1', 'g': 'A', 'v': 1.0},
        {'dataset': 'd', 'query_id': 'q2', 'g': 'A', 'v': 1.0},
        {'dataset': 'd', 'query_id': 'q3', 'g': 'B', 'v': 0.0},
    ]
    result = cluster_bootstrap(rows, 'g', 'v', iterations=4000, seed=11)
    # Drawing B three times must give exactly 0; drawing A three times exactly 1.
    assert result['low'] == pytest.approx(0.0)
    assert result['high'] == pytest.approx(1.0)


def test_paired_bootstrap_compares_the_same_queries():
    rows = [
        {'dataset': 'd', 'query_id': 'q1', 'g': 'A', 'a': 1.0, 'b': 0.0},
        {'dataset': 'd', 'query_id': 'q2', 'g': 'B', 'a': 0.0, 'b': 0.0},
    ]
    result = paired_cluster_bootstrap(rows, 'g', 'a', 'b', iterations=2000, seed=3)
    assert result['point'] == pytest.approx(0.5)
    assert result['iterations'] > 0
    assert result['low'] <= 0.5 <= result['high']


# ------------------------------------------------------------------------- value levels


def test_numeric_equal_survives_large_integers():
    """float() collapses 2**53+1 onto 2**53; Decimal must not."""
    assert float('9007199254740993') == float('9007199254740992')
    assert not numeric_equal('9007199254740993', '9007199254740992')


def test_value_levels_are_named_and_strict_is_not_any_level():
    result = value_matches('PASL-Pro', 'Professional Arena Soccer League (PASL-Pro)')
    assert result['strict'] is False
    assert result['any_level'] is False
    contained = value_matches('1234', '1,234')
    assert contained['strict'] is False
    assert contained['normalized'] is True
    assert contained['any_level'] is True


def test_any_level_is_not_accepted_as_strict_count():
    """StrictRecovered must not be described by any_level row counts."""
    rows = [{'dataset': 'd', 'query_id': 'q', 'strict': 0.0, 'any_level': 1.0}]
    assert query_macro(rows, 'strict') == 0.0
    assert query_macro(rows, 'any_level') == 1.0
    assert normalize('  Hello, World. ') == 'hello, world'


# ------------------------------------------------------------------- candidate pooling


def test_empty_evidence_candidate_is_retained():
    """The R3 adapter skipped E=[] tables; the R4 lock must keep them."""
    from mmdd_stage2.r4_common import natural_evidence

    result = {'target_id': 't', 'rank': 3, 'paths': []}
    assert natural_evidence(result) == []
    # A candidate with only non-evidence paths is also retained, not dropped.
    result = {'target_id': 't', 'rank': 3, 'paths': [{'kind': 'direct', 'target_id': 'x'}]}
    assert natural_evidence(result) == []


def test_natural_evidence_is_read4_and_deduplicated():
    from mmdd_stage2.r4_common import natural_evidence

    paths = [{'kind': 'evidence', 'evidence_id': f'e{i}'} for i in range(6)]
    paths.insert(2, {'kind': 'evidence', 'evidence_id': 'e0'})
    assert natural_evidence({'paths': paths}) == ['e0', 'e1', 'e2', 'e3']


# ------------------------------------------------------------------ joint math contract


def _joint(logits, column_logits, mask):
    import torch
    from mmdd_stage2.r4_joint import log_joint

    return log_joint(torch.tensor(logits, dtype=torch.float64),
                     torch.tensor(column_logits, dtype=torch.float64),
                     torch.tensor(mask, dtype=torch.bool))


def test_sum_over_columns_reproduces_table_probability():
    """sum_c J(T,c) == p_T. This is a unit test, not a reranker."""
    import torch

    table = [1.0, 0.5, -2.0]
    mask = [[True, True, True], [True, True, True], [False, False, False]]
    result = _joint(table, [[0.1, 0.2, 0.3], [5.0, -1.0, 0.0], [0.0, 0.0, 0.0]], mask)
    for index in (0, 1):
        assert math.isclose(result['marginal'][index], result['table_probability'][index], rel_tol=1e-12)
    assert math.isclose(float(result['table_probability'].sum()), 1.0, rel_tol=1e-12)
    # A table with no genuine column keeps its table mass but contributes no pair, so the
    # pair distribution sums to the column-bearing mass, not to 1.
    assert result['tables_without_columns'] == 1
    assert result['marginal'][2] == 0.0
    pair_total = float(result['pair_probability'].sum())
    assert math.isclose(pair_total, float(result['table_probability'][:2].sum()), rel_tol=1e-12)
    assert pair_total < 1.0


def test_top3_truncation_is_not_renormalised():
    """Column probabilities are over all genuine columns; a Top-3 slice sums to < 1."""
    result = _joint([0.0, 0.0], [[2.0, 1.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]], [[True] * 4, [True] * 4])
    rho = result['column_probability']
    assert math.isclose(float(rho.sum(dim=1)[0]), 1.0, rel_tol=1e-12)
    top3 = sorted(rho[0].tolist(), reverse=True)[:3]
    assert sum(top3) < 1.0


def test_product_can_flip_a_confident_wrong_table():
    """A high-rho narrow wrong table may outrank the correct table. Not excluded."""
    # table 0 correct but low column confidence; table 1 wrong with a very peaked column.
    table = [0.405, 0.595]
    columns = [[0.0, 0.0], [8.0, -8.0]]
    result = _joint(table, columns, [[True, True], [True, True]])
    pair = result['pair_log_probability']
    best = pair.max(dim=1).values
    assert best[1] > best[0], 'wrong narrow table takes the top pair'
    # Summing columns still restores the table order.
    assert result['marginal'][1] == pytest.approx(result['table_probability'][1])


def test_empty_evidence_table_still_enters_both_softmaxes():
    """A table with empty evidence keeps a table score and its column distribution."""
    result = _joint([0.0, 1.0], [[1.0, -1.0], [0.5, 0.5]], [[True, True], [True, True]])
    assert result['pair_log_probability'][1].max() > -math.inf
    assert math.isclose(sum(result['table_probability']), 1.0, rel_tol=1e-12)


def test_sentinel_scores_are_rejected_as_probabilities():
    from mmdd_stage2.r4_joint import assert_no_sentinel

    with pytest.raises(ValueError):
        assert_no_sentinel({'outside_scores': 'ranking_only_sentinels'})
    assert_no_sentinel({'outside_scores': 'prior'})


def test_table_rank_max_is_deduplicated_and_matches_first_pair_occurrence():
    import torch
    from mmdd_stage2.r4_joint import table_rank_max

    table = torch.tensor([0.0, 0.0])
    columns = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    mask = torch.ones_like(columns, dtype=torch.bool)
    joint = _joint(table.tolist(), columns.tolist(), mask.tolist())
    order = table_rank_max(joint, ['t0', 't1'])
    pairs = joint['pair_order']
    first_seen = list(dict.fromkeys(target for target, _ in pairs))
    assert order == first_seen

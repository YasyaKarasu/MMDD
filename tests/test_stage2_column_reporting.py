import pytest

from finalize_stage2_columns import paired_summary
from mmdd_stage2.column_metrics import evaluate, prediction


def test_paired_report_matches_queries_by_identity_and_keeps_failures():
    population = [
        {'dataset': 'lake', 'query_id': query, 'target_id': target, 'source_table_id': 'shared_source',
         'candidate_column_indices': [8, 2], 'gold_column_indices': [8]}
        for query, target in [('q', 'a'), ('q', 'b'), ('r', 'c')]]
    _, high = evaluate(population, [prediction(population[0], [1., 0.]),
                                    prediction(population[1], [1., 0.])])
    _, low = evaluate(population, [])
    result = paired_summary(high, list(reversed(low)))
    assert result['queries'] == 2
    assert result['source_groups'] == 1
    assert result['delta']['ColHit@1'] == .5  # q=1, r=0; not pair-micro 2/3.
    assert result['delta']['MRR'] == .5
    assert result['CI95']['MRR'] == [.5, .5]
    with pytest.raises(ValueError, match='populations differ'):
        paired_summary(high, low[:1])

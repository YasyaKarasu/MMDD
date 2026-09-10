from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_stage1_r15_candidates import (
    candidate_sets,
    exact_order_and_ranks,
    export_verification,
    output_directory,
    random_admission,
    read_gzip,
    witness_details,
    write_gzip,
)
from audit_stage1_r15_original_cases import expected_values, keyed, normalize


def test_exact_rank_uses_target_id_order_for_equal_scores():
    order, ranks = exact_order_and_ranks(np.array([0.5, 1.0, 0.5, -0.1]))
    assert order.tolist() == [1, 0, 2, 3]
    assert ranks.tolist() == [2, 1, 3, 4]


def test_candidate_sets_preserve_direct_and_evidence_overlap():
    record = {"paths_by_target": {
        "a": [{"kind": "direct", "path_score": 0.2}],
        "b": [{"kind": "direct", "path_score": 0.7}, {"kind": "evidence"}],
        "c": [{"kind": "evidence"}],
    }}
    direct, evidence, union = candidate_sets(record)
    assert direct == ["b", "a"]
    assert evidence == {"b", "c"}
    assert union == {"a", "b", "c"}


def test_random_admission_preserves_direct_members_and_external_slots():
    args = (["a", "x", "b", "y"], {"a", "b", "c"}, ["x", "y", "z", "w"])
    fixed, draws = random_admission(*args, np.random.default_rng(15), 100)
    assert fixed == ["a", "b"]
    assert all(len(draw) == len(set(draw)) == 2 for draw in draws)
    assert all(set(draw) <= {"x", "y", "z", "w"} for draw in draws)
    assert random_admission(*args, np.random.default_rng(15), 100) == (fixed, draws)
    assert len({tuple(draw) for draw in draws}) > 1


def test_witness_retention_deduplicates_rows_and_does_not_assume_content_correctness():
    record = {
        "positive_evidence_by_target": {"t": ["e1", "e2"]},
        "positive_evidence_rows_by_target": {"t": {"e1": [0, 1], "e2": [1, 2]}},
        "paths_by_target": {"t": [
            {"kind": "evidence", "evidence_id": "e1", "evidence_type": "text"},
            {"kind": "evidence", "evidence_id": "e2", "evidence_type": "image"},
            {"kind": "evidence", "evidence_id": "unknown", "evidence_type": "image"},
        ]},
    }
    retained = witness_details(record, "t", ["e2", "unknown"])
    assert retained["raw_supported_row_count"] == 3
    assert retained["retained_supported_row_count"] == 2
    assert retained["retained_known_witness_ids"] == ["e2"]
    assert retained["value_recovery_verified"] is None
    assert retained["correct_join_verified"] is None
    lost = witness_details(record, "t", ["unknown"])
    assert not lost["retention_keeps_known_witness"]
    assert lost["retained_supported_row_count"] == 0


def test_verification_keeps_exact_only_additions_and_unreviewed_states_null(tmp_path):
    output = output_directory(tmp_path)
    output.mkdir(parents=True)
    base = {
        "arm": "B13", "query_id": "q", "source_table_id": "source", "query_kind": "implicit",
        "exact_direct_rank": 101, "in_E": True, "in_C50": {"union_rrf_equal": True},
        "actual_evidence_delivered_by_rule": {"union_rrf_equal": ["e"]},
        "known_qet_witness_ids": ["e"], "retention_keeps_known_witness": True,
        "raw_supported_row_count": 1, "retained_supported_row_count": 1,
    }
    rows = [
        {**base, "target_id": "ann_only", "ann_evidence_only": True, "exact_evidence_only": False},
        {**base, "target_id": "exact_only", "ann_evidence_only": False, "exact_evidence_only": True},
        {**base, "target_id": "neither", "ann_evidence_only": False, "exact_evidence_only": False},
    ]
    write_gzip(output / "all_positive_pairs.jsonl.gz", rows)
    write_gzip(output / "B13_evidence_only_known_witness_cases.jsonl.gz", rows[:1])
    export_verification(tmp_path)
    actual = read_gzip(output / "evidence_only_verification.jsonl.gz")
    assert {row["target_id"] for row in actual} == {"ann_only", "exact_only"}
    assert [row["priority_82_queue_member"] for row in actual] == [True, False]
    assert all(row["chain_step5_correct_join"] is None for row in actual)
    assert all(row["chain_step4_independent_evidence_attribute_support"] is None for row in actual)


def test_original_case_normalization_preserves_false_and_id_membership():
    assert normalize("F1@10", "False") is False
    assert normalize("support_rows", " 2 ") == 2
    assert normalize("qet_known_ids", "b|a") == ["a", "b"]
    assert normalize("qet_known_ids", "a|a") == ["a", "a"]
    with pytest.raises(ValueError, match="Invalid boolean"):
        normalize("QET_known", "unknown")


def test_original_case_keys_reject_duplicates_and_empty_ids():
    row = {"query_id": "q", "target_id": "t"}
    with pytest.raises(ValueError, match="unique"):
        keyed([row, dict(row)])
    with pytest.raises(ValueError, match="nonempty"):
        keyed([{"query_id": " ", "target_id": "t"}])


def test_original_case_mapping_uses_qe_membership_and_rank_cutoffs():
    case = {
        "arm": "B13_S_full_seed13", "query_id": "q", "source_table_id": "s", "target_id": "t",
        "query_kind": "implicit", "known_witness_ids": ["text1", "image1", "absent"],
        "in_ann_D100": False, "in_E": True, "in_U": True, "ann_evidence_only": True,
        "known_qet_witness_ids": ["image1"], "known_qet_modalities": ["image"],
        "known_qet_paths": [{"evidence_type": "image"}], "raw_supported_row_count": 2,
        "final_rank_by_rule": {"f1_union_direct": 20, "union_rrf_equal": None},
    }
    witness = {"arm": "s_full", "positive_denominator": 2, "raw_top_path_known": False}
    mapped = expected_values(case, witness, {"text": {"text1", "unknown"}, "image": {"image1"}})
    assert len(mapped) == 30
    assert mapped["qe_known_ids"] == ["image1", "text1"]
    assert mapped["qet_known_ids"] == ["image1"]
    assert mapped["denominator"] == 2 and mapped["support_rows"] == 2
    assert mapped["F1@10"] is False and mapped["F1@20"] is True
    assert mapped["F1@50"] is True and mapped["RRF@50"] is False
    assert mapped["known_top_path"] is False

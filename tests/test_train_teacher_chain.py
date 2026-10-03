"""``train_teacher_chain``: witness-list merge into T_B records and the chain summary."""
from __future__ import annotations

import pytest

from train_teacher_chain import chain_summary, merge_witness_lists


def test_merge_witness_lists_copies_each_query_s_ta_lists_and_rejects_mismatched_queries():
    ta = [
        {"query_id": "q0", "qet_lists": [{"evidence_id": "e0", "evidence_kind": "text", "positives": ["t0"],
                                          "ignore": [], "candidates": ["t0", "t1"]}]},
        {"query_id": "q1"},
    ]
    tb = [{"query_id": "q1", "targets": ["t1"], "positives": ["t1"], "natural_bags": {"t1": []}, "support_records": []},
          {"query_id": "q0", "targets": ["t0"], "positives": ["t0"], "natural_bags": {"t0": []}, "support_records": []}]
    merged = merge_witness_lists(ta, tb)
    assert [row["query_id"] for row in merged] == ["q1", "q0"]
    assert merged[0]["qet_lists"] == [] and merged[1]["qet_lists"] == ta[0]["qet_lists"]
    assert merged[1]["qet_lists"] is not ta[0]["qet_lists"]  # a copy, the T_A record is untouched
    assert set(merged[1]) == {"query_id", "targets", "positives", "natural_bags", "support_records", "qet_lists"}
    with pytest.raises(ValueError, match="different queries"):
        merge_witness_lists(ta, tb[:1])


def test_chain_summary_reports_views_strict_pairs_and_contrasts_including_the_reference():
    queries = [f"q{i}" for i in range(4)]
    gt = {q: {"G": ["t0"], "kind": "implicit" if i % 2 else "explicit", "source_group": f"g{i}", "W": {}}
          for i, q in enumerate(queries)}
    points = ["TA_epoch1", "TB_end"]
    values = {"TA_epoch1": {"f0": 0.0, "Real": 0.0, "Swap": 0.0}, "TB_end": {"f0": 0.5, "Real": 1.0, "Swap": 0.5}}
    metrics = {
        "per_query": {q: {f"{name}.{view}.R@10": value for name in points for view, value in values[name].items()}
                      for q in queries},
        "teacher": {name: {view: {segment: {"R@10": values[name][view]} for segment in ("overall", "implicit", "explicit")}
                           for view in ("f0", "Real", "Swap")} for name in points},
    }
    reference = {q: {"TB_CQET.f0.R@10": 0.5, "TB_CQET.Real.R@10": 0.25} for q in queries}
    strict = {"TB_end": {"in_C150": 3, "teacher_top10": 2, "teacher_top50": 3}}
    summary = chain_summary(metrics, gt, points, reference, strict)
    assert summary["teacher"]["TB_end"]["Real"]["implicit"] == 1.0
    assert summary["strict_pairs"] == strict
    assert summary["contrasts"]["TB_end.Real_minus_f0"]["mean_delta_pp"] == 50.0
    assert summary["contrasts"]["TB_end.Real_minus_Swap"]["mean_delta_pp"] == 50.0
    assert summary["contrasts"]["TA_epoch1.Real_minus_f0"]["mean_delta_pp"] == 0.0
    assert summary["contrasts"]["TB_end.Real_minus_reference.Real"]["mean_delta_pp"] == 75.0
    assert summary["contrasts"]["TB_end.f0_minus_reference.f0"]["mean_delta_pp"] == 0.0
    # Without a reference covering every query the reference contrasts are left out.
    partial = chain_summary(metrics, gt, points, {q: reference[q] for q in queries[:2]}, strict)
    assert "TB_end.Real_minus_reference.Real" not in partial["contrasts"]

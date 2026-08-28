from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmdd_stage1.evidence_diagnostics import (
    analyze_evidence_annotations,
    summarize_ranks,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _dataset(tmp_path: Path, artifacts: dict[str, list[dict]]) -> Path:
    manifest = {"format": "sharded_jsonl", "artifacts": {}}
    for name, records in artifacts.items():
        relative = f"{name}/part-00000.jsonl"
        _write_jsonl(tmp_path / relative, records)
        manifest["artifacts"][name] = {
            "shards": [{"path": relative, "records": len(records)}]
        }
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return tmp_path


def test_annotation_diagnostic_separates_recovery_and_fallback_sources(tmp_path: Path):
    queries = [
        {"table_id": f"q{index}", "split": "dev", "source_table_id": f"qs{index}"}
        for index in range(1, 5)
    ]
    targets = [
        {"table_id": "t1", "source_table_id": "s1"},
        {"table_id": "t2", "source_table_id": "s2"},
        {"table_id": "t3", "source_table_id": "s3"},
        {"table_id": "t4", "source_table_id": "s4"},
    ]
    assets = [
        {"asset_id": "e1", "asset_type": "text"},
        {"asset_id": "e2", "asset_type": "image"},
        {
            "asset_id": "e3",
            "asset_type": "text",
            "source_table_id": "s3",
        },
    ]
    qrels = [
        {"query_table_id": f"q{index}", "target_table_id": f"t{index}", "split": "dev"}
        for index in range(1, 5)
    ]
    recoveries = [
        {
            "query_table_id": "q1",
            "target_table_id": "t1",
            "split": "dev",
            "recovered_attribute": {"hidden_in_query": True},
            "auto_check": {"supported_attributes": 1},
            "evidence": {"asset_id": "e1"},
        },
        {
            "query_table_id": "q1",
            "target_table_id": "t2",
            "split": "dev",
            "recovered_attribute": {"hidden_in_query": True},
            "auto_check": {"supported_attributes": 1},
            "evidence": {"asset_id": "e2"},
        },
    ]
    root = _dataset(
        tmp_path / "dataset",
        {
            "query_tables": queries,
            "data_lake_tables": targets,
            "bridge_assets": assets,
            "qrels": qrels,
            "evidence_recoveries": recoveries,
        },
    )
    target_lists = tmp_path / "target_lists.jsonl"
    evidence_by_query = {"q1": ["e1"], "q2": ["e2"], "q3": ["e3"], "q4": []}
    _write_jsonl(
        target_lists,
        [
            {
                "query_id": f"q{index}",
                "split": "dev",
                "direct_positive_target_id": f"t{index}",
                "evidence_positive_target_id": f"t{index}",
                "candidates": [
                    {
                        "target_id": f"t{index}",
                        "evidence_ids": evidence_by_query[f"q{index}"],
                    },
                    {"target_id": "negative", "evidence_ids": []},
                ],
            }
            for index in range(1, 5)
        ],
    )

    summary, cases = analyze_evidence_annotations(
        dataset_name="synthetic",
        dataset_root=root,
        target_lists_path=target_lists,
        split="dev",
    )

    assert summary["annotation_source_counts"] == {
        "exact_recovery": 1,
        "none": 1,
        "source_asset_heuristic": 1,
        "target_recovery_fallback": 1,
    }
    assert summary["recovery_pairs_outside_positive_qrels"] == 1
    assert summary["construction_audit"]["positive_candidate_evidence_mismatches"] == 0
    assert {case["annotation_source"] for case in cases} == {
        "exact_recovery",
        "target_recovery_fallback",
        "source_asset_heuristic",
        "none",
    }


def test_summarize_ranks_reports_censored_median():
    summary = summarize_ranks([2, 20, None, None], rank_limit=100)

    assert summary["median_best_positive_rank"] is None
    assert summary["median_best_positive_rank_label"] == ">100"
    assert summary["censored_above_rank_limit"] == 2
    assert summary["recall_at_rank"]["10"] == 0.25

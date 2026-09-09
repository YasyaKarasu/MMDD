from __future__ import annotations

import json

import torch
from mmdd_stage2.data import Stage2ObjectIndex
from mmdd_stage2.pipeline import RowPrediction
from mmdd_stage2.r12_task_f import (
    TaskFInputs,
    _queue_retrieval,
    _row_equivalence,
    _wrong_attribute_evidence,
    prepare_task_f_human_audit,
    run_full_chain,
    select_task_f_audit_cases,
)


def test_task_f_queue_preserves_selected_evidence_order_and_direct_path():
    pool = {
        "query_id": "q",
        "paths_by_target": {
            "t": [
                {"kind": "evidence", "evidence_id": "e1", "path_score": 1.0},
                {"kind": "direct", "path_score": 0.8},
                {"kind": "evidence", "evidence_id": "e2", "path_score": 0.9},
            ]
        },
    }
    queue, details = _queue_retrieval(
        [
            {
                "target_id": "t",
                "final_rank": 3,
                "selected_evidence_ids": ["e2", "e1"],
            }
        ],
        pool,
    )

    assert [path.get("evidence_id") for path in queue[0]["paths"]] == ["e2", "e1", None]
    assert queue[0]["stage2_table_score"] == -3.0
    assert details[0]["target_id"] == "t"


def test_wrong_attribute_evidence_requires_an_equal_count_disjoint_donor():
    record = {
        "query_id": "q",
        "positive_evidence_by_target": {
            "gold": ["shared", "own"],
            "other": ["shared", "wrong1", "wrong2"],
        },
    }

    selected = _wrong_attribute_evidence(record, "gold", count=2, seed=13)

    assert set(selected) == {"wrong1", "wrong2"}
    assert _wrong_attribute_evidence(record, "gold", count=3, seed=13) == ()


def test_task_f_row_equivalence_keeps_model_correctness_separate_from_support():
    class Backend:
        def embed_texts(self, values):
            return torch.tensor(
                [[1.0, 0.0] if value.casefold() == "barcelona" else [0.0, 1.0] for value in values]
            )

    rows = (
        RowPrediction(0, "Barcelona", {"evidence_id": "good"}),
        RowPrediction(1, "Barcelona", {"evidence_id": "wrong"}),
    )
    metrics = _row_equivalence(
        Backend(),
        rows,
        query_id="q",
        target_id="t",
        truth={
            ("q", "t", 0): {"value": "Barcelona"},
            ("q", "t", 1): {"value": "Barcelona"},
        },
        supports={"good": {0}},
        threshold=0.8,
    )

    assert [row["model_value_correct"] for row in metrics] == [True, True]
    assert [row["correct_value_recovery"] for row in metrics] == [True, False]


def test_full_chain_reuses_identical_frozen_queue_result(tmp_path):
    class Result:
        def to_dict(self):
            return {"reranked_candidates": [], "unattempted_candidates": []}

    class Verifier:
        def __init__(self):
            self.calls = 0

        def verify(self, *_args, **_kwargs):
            self.calls += 1
            return Result()

    queue = {"q": [{"target_id": "t", "paths": [{"kind": "direct"}]}]}
    details = {"q": [{"target_id": "t", "raw_d100_member": True}]}
    inputs = TaskFInputs(
        sample=({"query_id": "q", "source_table_id": "s", "kind": "explicit"},),
        supervision={"q": {"positive_target_ids": ["t"]}},
        queues={"f1_union_direct": queue, "frozen_fusion": queue},
        queue_details={"f1_union_direct": details, "frozen_fusion": details},
        frozen_fusion_id="f1_union_direct",
        queues_identical=True,
        objects=Stage2ObjectIndex({"q": {}}, {"t": {}}, {}),
        qrels={},
        recovery_values={},
        manifest={},
    )
    verifier = Verifier()
    output = tmp_path / "predictions.jsonl"

    run_full_chain(verifier, inputs, output)
    records = [json.loads(line) for line in output.read_text().splitlines()]

    assert verifier.calls == 1
    assert [record["system"] for record in records] == [
        "f1_union_direct",
        "frozen_fusion",
    ]
    assert records[1]["identical_queue_result_reused"] is True


def test_task_f_audit_selects_all_successes_and_seeded_failure_fraction():
    fa_rows = [
        {"query_id": "q1", "target_id": "t1", "evidence_enabled_join": True},
        {"query_id": "q2", "target_id": "t2", "evidence_enabled_join": False},
        {"query_id": "q3", "target_id": "t3", "evidence_enabled_join": False},
        {"query_id": "q4", "target_id": "t4", "evidence_enabled_join": False},
    ]
    query_metrics = [
        {
            "query_id": "q2",
            "candidates": [
                {"target_id": "t2", "outside_direct_final_join": True}
            ],
        }
    ]

    selected = select_task_f_audit_cases(
        fa_rows,
        query_metrics,
        failure_fraction=0.5,
        seed=13,
    )
    by_pair = {
        (row["query_id"], row["target_id"]): row["selection_reasons"]
        for row in selected
    }

    assert "evidence_enabled_join_success" in by_pair[("q1", "t1")]
    assert "outside_direct_final_join_success" in by_pair[("q2", "t2")]
    assert sum(
        "evidence_enabled_join_failure_sample" in reasons
        for reasons in by_pair.values()
    ) == 1


def test_task_f_human_audit_writes_blinded_packets_and_separate_labels(tmp_path):
    def table(table_id, columns, rows):
        return {
            "table_id": table_id,
            "columns": [
                {"column_index": index, "column_name": name}
                for index, name in enumerate(columns)
            ],
            "rows": [
                {
                    "row_id": row_index,
                    "cells": [
                        {"column_index": column_index, "text": value}
                        for column_index, value in enumerate(values)
                    ],
                }
                for row_index, values in enumerate(rows)
            ],
        }

    inputs = TaskFInputs(
        sample=({"query_id": "q", "source_table_id": "s", "kind": "implicit"},),
        supervision={"q": {"query_kind": "implicit", "positive_target_ids": ["t"]}},
        queues={},
        queue_details={},
        frozen_fusion_id="f1_union_direct",
        queues_identical=True,
        objects=Stage2ObjectIndex(
            {"q": table("q", ["player"], [["Messi"]])},
            {"t": table("t", ["country"], [["Argentina"]])},
            {
                "e": {
                    "asset_id": "e",
                    "asset_type": "text",
                    "content": "Messi represents Argentina.",
                }
            },
        ),
        qrels={},
        recovery_values={},
        manifest={},
    )
    fa_path = tmp_path / "fa.jsonl"
    fa_path.write_text(
        json.dumps(
            {
                "query_id": "q",
                "target_id": "t",
                "gold_column_index": 0,
                "gold_column_name": "country",
                "evidence_enabled_join": True,
                "conditions": {
                    "retrieved": {
                        "evidence_ids": ["e"],
                        "joinable": True,
                        "rows": [
                            {
                                "row_id": 0,
                                "generated_value": "Argentina",
                                "evidence_id": "e",
                            }
                        ],
                    }
                },
            }
        )
        + "\n"
    )
    full_path = tmp_path / "full.jsonl"
    full_path.write_text(
        json.dumps(
            {
                "system": "f1_union_direct",
                "query_id": "q",
                "source_table_id": "s",
                "query_kind": "implicit",
                "positive_target_ids": ["t"],
                "queue_details": [{"target_id": "t", "raw_d100_member": True}],
                "stage2": {
                    "reranked_candidates": [
                        {
                            "target_id": "t",
                            "verification": {"joinable": True},
                            "selection": None,
                            "branches": {},
                            "final_branch": "direct",
                        }
                    ],
                    "unattempted_candidates": [],
                },
            }
        )
        + "\n"
    )

    manifest = prepare_task_f_human_audit(
        inputs,
        object(),
        fa_path=fa_path,
        full_chain_path=full_path,
        output_dir=tmp_path / "audit",
    )
    packet = json.loads((tmp_path / "audit/review_packets.jsonl").read_text())
    automated = json.loads(
        (tmp_path / "audit/automated_selection_do_not_show_reviewers.jsonl").read_text()
    )

    assert manifest["status"] == "prepared_not_reviewed"
    assert manifest["success_cases"] == 1
    assert packet["evidence"][0]["evidence_text"] == "Messi represents Argentina."
    assert "selection_reasons" not in packet
    assert automated["selection_reasons"] == ["evidence_enabled_join_success"]
    assert automated["independently_confirmed"] is False

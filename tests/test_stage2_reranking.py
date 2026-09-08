from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_stage2
from mmdd_stage1.artifacts import checkpoint_fingerprint
from mmdd_stage2.data import Stage2Objects, validate_retrieval_path_budget
from mmdd_stage2.pipeline import LocalizedEvidence, Stage2Verifier
from mmdd_stage2.verifier import CandidateColumnScorer, semantic_joinability


def table(table_id, columns, rows):
    return {
        "table_id": table_id,
        "columns": [
            {"column_index": index, "column_name": name}
            for index, name in enumerate(columns)
        ],
        "rows": [
            {
                "row_id": index + 10,
                "cells": [
                    {"column_index": column, "text": value}
                    for column, value in enumerate(values)
                ],
            }
            for index, values in enumerate(rows)
        ],
    }


def retrieved(target_id, *, prior=0.0, evidence_ids=(), direct=False, **extra):
    return {
        "target_id": target_id,
        "score": 1.0,
        "evidence_score": prior if evidence_ids else None,
        "paths": [
            *([{"kind": "direct"}] if direct else []),
            *({"kind": "evidence", "evidence_id": item} for item in evidence_ids),
        ],
        **extra,
    }


def assets(values):
    return {
        key: {"asset_id": key, "asset_type": "text", "content": value}
        for key, value in values.items()
    }


class PoolBackend:
    hidden_dim = 1
    device = torch.device("cpu")

    def __init__(self, logits):
        self.logits = logits
        self.events = []
        self.reader_queries = []
        self.localized_rows = []
        self.generated_rows = []
        self.embed_batches = []

    def reader_states(self, query, target, evidence):
        self.events.append(("reader", target["table_id"]))
        self.reader_queries.append(copy.deepcopy(query))
        states = torch.tensor(self.logits[target["table_id"]], dtype=torch.float32)[:, None]
        return states, torch.zeros_like(states)

    def localize_evidence(self, row, *, attribute_name, evidence):
        self.events.append(("localize", evidence["asset_id"]))
        self.localized_rows.append(dict(row))
        return LocalizedEvidence(
            evidence["asset_id"], "text", text=evidence["content"], text_span_relevance=0.9
        )

    def evidence_logits(self, row, *, attribute_name, candidates):
        return torch.zeros(len(candidates))

    def generate_value(self, row, *, attribute_name, evidence):
        self.events.append(("generate", evidence.evidence_id))
        self.generated_rows.append(dict(row))
        return evidence.text

    def embed_texts(self, values):
        self.embed_batches.append(tuple(values))
        vectors = {"Near": [0.5, 3**0.5 / 2], "Foxtrot": [1.0, 0.0]}
        return torch.tensor([vectors.get(value, [0.0, 0.0]) for value in values]).reshape(-1, 2)


class PoolRouter:
    def __init__(self, positions, events):
        self.positions = positions
        self.events = events
        self.calls = []

    def assign(self, query_id, evidence_ids, *, row_count):
        self.events.append(("route", tuple(evidence_ids)))
        self.calls.append((query_id, tuple(evidence_ids), row_count))
        return {evidence_id: self.positions[evidence_id] for evidence_id in evidence_ids}


def scorer():
    head = CandidateColumnScorer(1)
    with torch.no_grad():
        head.weight.weight.copy_(torch.tensor([[1.0, 0.0]]))
        head.weight.bias.zero_()
    return head


def test_full_evidence_pool_scored_before_max_priority_recovery():
    query = table("q", ["Name"], [["Ada"], ["Bo"]])
    targets = {
        "early": table("early", ["A", "B", "C", "D"], [["Alpha"] * 4]),
        "late": table("late", ["Name", "Bridge"], [["unused", "Beta"]]),
        "tied": table("tied", ["Name", "Bridge"], [["unused", "Gamma"]]),
    }
    results = [
        retrieved("early", prior=2.0, evidence_ids=["e0"]),
        retrieved("late", prior=-50.0, stage2_table_score=1.0, evidence_ids=["e1"]),
        retrieved("tied", prior=1.0, evidence_ids=["e2"]),
    ]
    backend = PoolBackend({"early": [0, 0, 0, 0], "late": [0, 4], "tied": [0, 4]})
    router = PoolRouter({"e0": 0, "e1": 0, "e2": 1}, backend.events)
    result = Stage2Verifier(backend, scorer(), evidence_router=router).verify(
        query, results, targets, assets({"e0": "Alpha", "e1": "Beta", "e2": "Gamma"}),
        recovery_budget=1,
    )

    assert backend.events == [
        ("reader", "early"), ("reader", "late"), ("reader", "tied"),
        ("route", ("e1",)), ("localize", "e1"), ("generate", "e1"),
    ]
    assert router.calls == [("q", ("e1",), 2)]
    selected = result.reranked_candidates[0]
    assert selected.target_id == "late"
    assert selected.stage1_rank == 2
    assert selected.table_prior == 1.0
    assert selected.scores.selection.column_index == 1
    assert selected.semantic_joinability.coverage == 0.5
    assert selected.semantic_joinability.joinable is False
    assert [row.value for row in selected.evidence.rows] == ["Beta", ""]
    assert backend.embed_batches == [("Beta", "", "Beta")]
    early, tied = result.unattempted_candidates
    assert early.scores.table_probability > selected.scores.table_probability
    assert early.scores.recovery_priority < selected.scores.recovery_priority
    assert tied.scores.recovery_priority == selected.scores.recovery_priority
    for candidate in (early, selected, tied):
        assert candidate.scores.recovery_priority == max(candidate.scores.joint_probabilities)
    assert sum(sum(c.scores.joint_probabilities) for c in (early, selected, tied)) == pytest.approx(1)
    assert [c.target_id for c in result.unattempted_candidates] == ["early", "tied"]


def test_recovery_budget_counts_unique_tables_not_flattened_pairs():
    query = table("q", ["Name"], [["Ada"]])
    targets = {
        "dominant": table("dominant", ["First", "Second"], [["Alpha", "Beta"]]),
        "second": table("second", ["Bridge"], [["Gamma"]]),
        "third": table("third", ["Bridge"], [["Delta"]]),
    }
    for position, column in enumerate(targets["dominant"]["columns"]):
        column["column_index"] = 7 + position * 4
        targets["dominant"]["rows"][0]["cells"][position]["column_index"] = column["column_index"]
    results = [
        retrieved("dominant", prior=10.0, evidence_ids=["e0"]),
        retrieved("second", evidence_ids=["e1"]),
        retrieved("third", evidence_ids=["e2"]),
    ]
    backend = PoolBackend({"dominant": [0, 0], "second": [0], "third": [0]})
    router = PoolRouter({"e0": 0, "e1": 0, "e2": 0}, backend.events)
    result = Stage2Verifier(backend, scorer(), evidence_router=router).verify(
        query, results, targets, assets({"e0": "Alpha", "e1": "Gamma", "e2": "Delta"}),
        recovery_budget=2,
    )

    dominant, second = result.reranked_candidates
    assert dominant.target_id == "dominant"
    assert dominant.scores.column_indices == (7, 11)
    assert dominant.scores.selection.column_index == 7
    assert min(dominant.scores.joint_probabilities) > second.scores.recovery_priority
    assert second.target_id == "second"
    assert len(second.scores.joint_probabilities) == 1
    assert [call[1] for call in router.calls] == [("e0",), ("e1",)]
    assert result.unattempted_candidates[0].target_id == "third"


def test_branches_merge_by_joinability_and_keep_independent_recovery_provenance():
    query = table("q", ["Name"], [["Ada"], ["Bo"], [""]])
    original = copy.deepcopy(query)
    targets = {
        "direct_failed": table("direct_failed", ["Name"], [["Ada"]]),
        "both_direct": table("both_direct", ["Name", "Bridge"], [["Ada", "Delta"], ["Bo", "Other"]]),
        "evidence_mid": table("evidence_mid", ["Bridge"], [["Foxtrot"]]),
        "both_evidence": table("both_evidence", ["Name", "Bridge"], [["Ada", "Echo"]]),
        "evidence_full": table("evidence_full", ["Bridge"], [["Kappa"]]),
        "unselected_both": table("unselected_both", ["Bridge"], [["Zed"]]),
        "unselected_only": table("unselected_only", ["Bridge"], [["Zulu"]]),
    }
    results = [
        retrieved("direct_failed", direct=True, implicit=True),
        retrieved("both_direct", direct=True, prior=5, evidence_ids=["d"]),
        retrieved("evidence_mid", prior=4, evidence_ids=["m0", "m1"], explicit=True),
        retrieved("both_evidence", direct=True, prior=3, evidence_ids=["b0", "b1"]),
        retrieved("evidence_full", prior=2, evidence_ids=["f0", "f1", "f2"]),
        retrieved("unselected_both", direct=True, prior=-10, evidence_ids=["u"]),
        retrieved("unselected_only", prior=-20, evidence_ids=["v"]),
    ]
    backend = PoolBackend({key: [0, 8] if len(value["columns"]) == 2 else [0] for key, value in targets.items()})
    positions = {"d": 0, "m0": 0, "m1": 1, "b0": 0, "b1": 1, "f0": 0, "f1": 1, "f2": 2}
    router = PoolRouter(positions, backend.events)
    evidence = assets({
        "d": "Delta", "m0": "Foxtrot", "m1": "Near", "b0": "Echo", "b1": "Echo",
        "f0": "Kappa", "f1": "Kappa", "f2": "Kappa", "u": "Zed", "v": "Zulu",
    })
    result = Stage2Verifier(backend, scorer(), evidence_router=router).verify(
        query, results, targets, evidence, recovery_budget=4,
    )

    assert [c.target_id for c in result.reranked_candidates] == [
        "evidence_full", "both_direct", "both_evidence", "evidence_mid", "direct_failed", "unselected_both",
    ]
    assert [c.rerank_rank for c in result.reranked_candidates] == list(range(1, 7))
    by_id = {c.target_id: c for c in result.reranked_candidates}
    assert by_id["both_direct"].final_branch == "direct"
    assert by_id["both_evidence"].final_branch == "evidence"
    assert by_id["both_direct"].evidence.rows[0].value == "Delta"
    assert by_id["both_evidence"].evidence.rows[0].value == "Echo"
    assert by_id["both_direct"].direct.semantic_joinability.coverage == pytest.approx(2 / 3)
    assert by_id["both_direct"].evidence.semantic_joinability.coverage == pytest.approx(1 / 3)
    assert by_id["evidence_mid"].semantic_joinability.coverage == pytest.approx(1 / 3)
    assert by_id["evidence_mid"].semantic_joinability.mean_similarity == pytest.approx(0.5)
    assert by_id["evidence_mid"].semantic_joinability.joinable is False
    assert sum(c.selected_for_recovery for c in result.reranked_candidates) == 4
    assert len(router.calls) == 4
    assert sum(c.direct is not None for c in result.reranked_candidates) == 4
    assert query == original
    assert all(snapshot == original for snapshot in backend.reader_queries)
    assert all(set(row) == {"Name"} for row in backend.localized_rows + backend.generated_rows)

    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    serialized = {c["target_id"]: c for c in payload["reranked_candidates"]}
    assert serialized["both_direct"]["branches"]["evidence"]["rows"][0] == {
        "row_id": 10, "value": "Delta",
        "evidence": {"evidence_id": "d", "evidence_type": "text", "text_span": "Delta", "text_span_relevance": 0.9},
    }
    assert serialized["direct_failed"]["verification"]["joinable"] is False
    assert serialized["unselected_both"]["branches"]["evidence"]["verification"] is None
    unattempted, = payload["unattempted_candidates"]
    assert unattempted["target_id"] == "unselected_only"
    assert unattempted["status"] == "not_attempted"
    assert unattempted["verification"] is None
    assert unattempted["rerank_rank"] is None
    assert unattempted["final_branch"] is None
    assert unattempted["selection"]["column_name"] == "Bridge"
    assert unattempted["recovery_priority"] > 0
    assert unattempted["selected_for_recovery"] is False
    assert unattempted["branches"]["evidence"]["rows"] == []
    assert unattempted["branches"]["evidence"]["not_attempted_reason"] == "recovery_budget"
    assert payload["input_candidate_count"] == 7


@pytest.mark.parametrize("with_evidence", [False, True])
def test_all_direct_targets_verified_without_using_recovery_budget_or_router(with_evidence):
    query = table("q", ["Name"], [["Ada"], [""]])
    targets = {f"d{index}": table(f"d{index}", ["Name"], [["Ada"]]) for index in range(7)}
    results = [retrieved(key, direct=True) for key in targets]
    if with_evidence:
        results[-1] = retrieved("d6", direct=True, evidence_ids=["e"])
    backend = PoolBackend({"d6": [0]})
    result = Stage2Verifier(backend, scorer(), min_row_coverage=0.6).verify(
        query, results, targets, assets({"e": "Ada"}), recovery_budget=0,
    )

    assert [c.target_id for c in result.reranked_candidates] == list(targets)
    assert all(c.direct.semantic_joinability.coverage == 0.5 for c in result.reranked_candidates)
    assert all(c.direct.semantic_joinability.joinable is False for c in result.reranked_candidates)
    assert not any(c.selected_for_recovery for c in result.reranked_candidates)
    assert backend.events == ([("reader", "d6")] if with_evidence else [])
    assert not result.unattempted_candidates


def test_empty_values_never_match_even_at_zero_similarity_threshold():
    check = semantic_joinability(
        ["", "Ada"], ["Ada"], query_embeddings=torch.zeros(2, 1),
        target_embeddings=torch.zeros(1, 1), similarity_threshold=0.0, min_coverage=0.6,
    )
    assert check.coverage == 0.5
    assert check.mean_similarity == 0.5
    assert check.joinable is False


@pytest.mark.parametrize("recovered_value,final_branch", [("Near", "evidence"), ("", "direct")])
def test_dual_branch_mean_similarity_breaks_coverage_ties(recovered_value, final_branch):
    query = table("q", ["Name"], [["Ada"], [""]])
    targets = {"t": table("t", ["Name", "Bridge"], [["Ada", "Foxtrot"]])}
    backend = PoolBackend({"t": [0, 4]})
    router = PoolRouter({"a": 0, "b": 1}, backend.events)
    result = Stage2Verifier(backend, scorer(), evidence_router=router).verify(
        query, [retrieved("t", direct=True, evidence_ids=["a", "b"])],
        targets, assets({"a": "Foxtrot", "b": recovered_value}),
    )

    candidate, = result.reranked_candidates
    assert result.recovery_budget == 20
    assert len(router.calls) == 1
    assert candidate.direct.semantic_joinability.coverage == candidate.evidence.semantic_joinability.coverage == 0.5
    assert candidate.evidence.semantic_joinability.mean_similarity == (0.75 if recovered_value else 0.5)
    assert candidate.final_branch == final_branch
    assert candidate.direct is not None and candidate.evidence is not None


@pytest.mark.parametrize("paths", [None, [], [{"kind": "evidence"}], [{"kind": "other"}]])
def test_missing_or_invalid_path_detail_rejected_without_metadata(paths):
    result = {"target_id": "t"}
    if paths is not None:
        result["paths"] = paths
    with pytest.raises(ValueError, match="path detail"):
        validate_retrieval_path_budget({"results": [result]}, max_targets=50, top_k_evidence=4)


def test_path_validation_rejects_duplicate_target_ids_and_missing_branch():
    result = retrieved("t", direct=True)
    with pytest.raises(ValueError, match="Duplicate Stage-1 target_id"):
        validate_retrieval_path_budget({"results": [result, result]}, max_targets=50, top_k_evidence=4)
    result["evidence_score"] = 1.0
    with pytest.raises(ValueError, match="evidence_score has no corresponding evidence path"):
        validate_retrieval_path_budget({"results": [result]}, max_targets=50, top_k_evidence=4)


def cli_args(tmp_path, *extra):
    return run_stage2.parse_args([
        "--dataset-root", str(tmp_path / "synthetic-dataset"),
        "--retrieval-results", str(tmp_path / "retrieval.json"),
        "--stage1-gate", str(tmp_path / "gate.json"),
        "--scorer-checkpoint", str(tmp_path / "head.pt"),
        "--stage1-features", str(tmp_path / "features"),
        "--model-dir", str(tmp_path / "fake-model"),
        "--device", "cpu", "--dtype", "fp32",
        "--output", str(tmp_path / "result.json"),
        *extra,
    ])


def write_gated_retrieval(tmp_path, record, *, allowed=True):
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"synthetic checkpoint, never loaded as a model")
    fingerprint = checkpoint_fingerprint(checkpoint)
    gate = {
        "format_version": 1,
        "completed_stage": "student-path", "selection_split": "dev", "stage2_allowed": allowed,
        "best_checkpoint": str(checkpoint), "best_checkpoint_sha256": fingerprint,
    }
    (tmp_path / "gate.json").write_text(json.dumps(gate), encoding="utf-8")
    (tmp_path / "retrieval.json").write_text(
        json.dumps({**record, "student_checkpoint_sha256": fingerprint}), encoding="utf-8"
    )


def test_cli_defaults_separate_input_and_recovery_budgets(tmp_path):
    args = cli_args(tmp_path)
    assert args.input_candidate_budget == 50
    assert args.recovery_budget == 20
    assert not hasattr(args, "max_direct_targets")
    args = cli_args(tmp_path, "--max-targets", "30", "--recovery-budget", "4")
    assert args.input_candidate_budget == 30
    assert args.recovery_budget == 4


@pytest.mark.parametrize("gate_allowed", [False, True])
def test_cli_checks_gate_and_full_n_path_budget_before_loading_model(tmp_path, monkeypatch, gate_allowed):
    def unexpected(*args, **kwargs):
        pytest.fail("Gate and path validation must precede data/model loading")

    monkeypatch.setattr(run_stage2, "load_stage2_objects", unexpected)
    monkeypatch.setattr(run_stage2, "QwenStage2Backend", unexpected)
    monkeypatch.setattr(run_stage2, "load_candidate_scorer", unexpected)
    write_gated_retrieval(tmp_path, {
        "query_id": "q", "path_aggregation": {"path_result_k": 10, "evidence_path_k": 4},
        "results": [retrieved(f"t{index}", direct=True) for index in range(50)],
    }, allowed=gate_allowed)
    with pytest.raises(ValueError, match="only 10" if gate_allowed else "Stage 2 is blocked"):
        run_stage2.run(cli_args(tmp_path, "--recovery-budget", "1"))


@pytest.mark.parametrize("flag,value", [
    ("--input-candidate-budget", "0"), ("--recovery-budget", "-1"), ("--top-k-evidence", "0"),
])
def test_cli_rejects_invalid_budgets_before_other_work(tmp_path, flag, value):
    with pytest.raises(ValueError, match=flag):
        run_stage2.run(cli_args(tmp_path, flag, value))


def test_cli_honors_n_and_m_and_serializes_full_pool_with_fake_boundaries(tmp_path, monkeypatch):
    query = table("q", ["Name"], [["Ada"]])
    targets = {
        "d": table("d", ["Name"], [["Ada"]]),
        "e1": table("e1", ["Bridge"], [["Beta"]]),
        "e2": table("e2", ["Bridge"], [["Gamma"]]),
    }
    results = [
        retrieved("d", direct=True), retrieved("e1", prior=1, evidence_ids=["a"]),
        retrieved("e2", prior=2, evidence_ids=["b"]), {"target_id": "outside_n"},
    ]
    write_gated_retrieval(tmp_path, {
        "query_id": "q", "results": results,
        "path_aggregation": {"path_result_k": 3, "evidence_path_k": 4},
    })
    backend = PoolBackend({"e1": [0], "e2": [0]})
    router = PoolRouter({"a": 0, "b": 0}, backend.events)

    def load_objects(root, query_id, bundles, *, extra_target_ids):
        assert [bundle.target_id for bundle in bundles] == ["e1", "e2"]
        assert extra_target_ids == ["d"]
        return Stage2Objects(query, targets, assets({"a": "Beta", "b": "Gamma"}))

    monkeypatch.setattr(run_stage2, "load_stage2_objects", load_objects)
    monkeypatch.setattr(run_stage2, "load_candidate_scorer", lambda *a, **kw: scorer())
    monkeypatch.setattr(run_stage2, "QwenStage2Backend", lambda *a, **kw: backend)
    monkeypatch.setattr(run_stage2.FeatureStore, "from_path", lambda *a: object())
    monkeypatch.setattr(run_stage2, "SimilarityEvidenceRouter", lambda *a: router)
    payload = run_stage2.run(cli_args(tmp_path, "--input-candidate-budget", "3", "--recovery-budget", "1"))

    assert payload["input_candidate_budget"] == payload["input_candidate_count"] == 3
    assert payload["recovery_budget"] == 1
    assert [c["target_id"] for c in payload["reranked_candidates"]] == ["d", "e2"]
    assert [c["target_id"] for c in payload["unattempted_candidates"]] == ["e1"]
    assert backend.events[:2] == [("reader", "e1"), ("reader", "e2")]
    assert router.calls == [("q", ("b",), 1)]
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == payload

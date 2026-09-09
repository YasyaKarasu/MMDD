from __future__ import annotations

import sys
import argparse
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_stage1_r11_protocol import _expansion_epoch
from mmdd_stage1 import scoring
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import EdgeExample
from mmdd_stage1.features import FeatureStore, ObjectFeatures
from mmdd_stage1.models import StudentJoinabilityModel
from mmdd_stage1.training import (
    checkpoint, student_anchor_loss, student_projection_drift,
    student_projection_references,
    train_student_edges,
)
from mmdd_stage1.objectives import PathAggregator
from train_stage1 import _EpochController
from reevaluate_stage1_r12_checkpoints import (
    _explicit_checkpoints,
    _optimizer_updates,
    _r12_checkpoints,
    exact_relation,
)
from prepare_stage1_r12_candidates import materialize_batch
from prepare_stage1_r12_extension_candidates import _batches as extension_batches
from prepare_stage1_r12_attribute_audit import sample_stratum
from run_stage1_r11_task_f import _scale
from collections import Counter
from run_stage1_r11_task_e import (
    _accumulate,
    _empty,
    _finalize,
    empty_intervention_stats,
    select_evidence,
)
from run_stage1_r11_task_f import (
    R12_FUSION_IDS,
    _quality_selection,
    _selected_valid_rows,
    _target_channels,
    reserved_channel_fusion,
)
from run_stage1_r12_task_c import (
    _apply_function_gradients,
    _examples as r12_examples,
    _sample_function_pairs,
    _training_batch_order,
)
from score_stage1_r12_extension_teacher_pairs import _shard_count


def test_checkpoint_preserves_full_chain_and_stage_projection_references(tmp_path):
    model = StudentJoinabilityModel(4, 4, initialization="identity")
    with torch.no_grad():
        model.projections["table"].weight.add_(0.1)
    model.reset_projection_anchors()
    with torch.no_grad():
        model.projections["text"].weight.add_(0.2)
    path = tmp_path / "student.pt"
    torch.save(checkpoint(model, "student-path"), path)
    loaded = load_student(path, torch.device("cpu"))
    source, target = torch.arange(4.0), torch.arange(4.0) + 1
    assert torch.equal(
        model.score_embeddings(source, "table", target, "text"),
        loaded.score_embeddings(source, "table", target, "text"),
    )
    assert student_projection_references(model) == student_projection_references(loaded)
    assert torch.equal(student_anchor_loss(model), student_anchor_loss(loaded))
    assert student_projection_drift(loaded)["table"] == pytest.approx(0.1)
    assert student_projection_drift(loaded, reference="stage_start")["table"] == 0
    assert student_projection_drift(loaded, reference="stage_start")["text"] == pytest.approx(0.2)


def test_legacy_checkpoint_inference_does_not_fabricate_pca_reference(tmp_path):
    model = StudentJoinabilityModel(4, 4, initialization="identity")
    payload = checkpoint(model, "student-edge")
    payload["state_dict"] = {
        key: value for key, value in payload["state_dict"].items()
        if key not in {"initial_projection_weights", "stage_initial_projection_weights"}
    }
    payload.pop("projection_references")
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)
    loaded = load_student(path, torch.device("cpu"))
    assert not student_projection_references(loaded)["P_PCA"]["available"]
    assert student_projection_drift(loaded)["table"] is None
    assert torch.isfinite(loaded.score_embeddings(torch.ones(4), "table", torch.ones(4), "text"))
    with pytest.raises(ValueError, match="full-chain projection reference"):
        student_anchor_loss(loaded)


def _edges():
    return [
        EdgeExample("q", (positive, negative), 0, source_type="table", destination_type="table")
        for positive, negative in (("p1", "n1"), ("p2", "n2"))
    ]


def test_real_expansion_audit_detects_broken_global_mask(monkeypatch):
    kwargs = dict(seed=13, epoch=0, batch_size=2, cap=256)
    correct = _expansion_epoch(_edges(), **kwargs)
    assert correct["fixed_known_positive_as_negative"] == 0
    assert correct["legacy_known_positive_as_negative"] == 2
    original = scoring._candidate_positive_mask

    def broken(*args, **kwargs):
        return torch.zeros_like(original(*args, **kwargs))

    monkeypatch.setattr(scoring, "_candidate_positive_mask", broken)
    failed = _expansion_epoch(_edges(), **kwargs)
    assert failed["fixed_known_positive_as_negative"] > 0


def test_expansion_counts_post_cap_candidates_not_potential_pool():
    result = _expansion_epoch(_edges(), seed=13, epoch=0, batch_size=2, cap=0)
    assert result["legacy_known_positive_as_negative"] == 0
    assert result["fixed_known_positive_as_negative"] == 0


class _DirectScorer:
    def score(self, query_id, target_ids, *, batch_size):
        return {target_id: float(index) for index, target_id in enumerate(target_ids)}


def _channels(record, intervention):
    return _target_channels(
        record, retention="e2_row_coverage", scorer=_DirectScorer(),
        store=FeatureStore({"q": ObjectFeatures("q", "table", torch.ones(2))}),
        content_keys={}, top_l=20, evidence_budget=4,
        pair_batch_size=16, intervention=intervention,
        intervention_stats=empty_intervention_stats(),
    )


def test_direct_only_target_cannot_receive_evidence_vote():
    direct, evidence = _channels({
        "query_id": "q", "paths_by_target": {"t": [{"kind": "direct"}]},
    }, "original_mixed")
    assert len(direct) == 1
    assert not evidence
    assert direct[0]["evidence_score"] is None


def test_removing_all_evidence_retains_union_direct_and_f5_refills():
    record = {
        "query_id": "q", "paths_by_target": {
            "t0": [{"kind": "direct"}],
            **{f"t{index}": [{
                "kind": "evidence", "evidence_id": f"e{index}",
                "evidence_type": "image", "path_score": 1.0,
            }] for index in range(1, 12)},
        },
    }
    direct, evidence = _channels(record, "remove_image")
    assert len(direct) == 12
    assert not evidence
    assert sum(row["original_direct_member"] for row in direct) == 1
    assert [row["target_id"] for row in reserved_channel_fusion(direct, evidence, k=10)] == [
        row["target_id"] for row in direct[:10]
    ]


def test_exact_content_copy_does_not_increase_row_coverage():
    store = FeatureStore({
        "q": ObjectFeatures("q", "table", torch.ones(2), row_embeddings=torch.eye(2)),
        "e": ObjectFeatures("e", "text", torch.tensor([1.0, 0.0])),
        "copy": ObjectFeatures("copy", "text", torch.tensor([1.0, 0.0])),
    })
    paths = [{"kind": "evidence", "evidence_id": "e", "path_score": 1.0}]
    kwargs = dict(query_id="q", store=store, content_keys={"e": "same", "copy": "same"},
                  top_l=20, budget=4, support_cache={})
    baseline = select_evidence("e2_row_coverage", paths, **kwargs)
    copied = select_evidence("e2_row_coverage", [*paths, {
        "kind": "evidence", "evidence_id": "copy", "path_score": 0.9,
    }], **kwargs)
    assert copied == baseline


def test_d2_unique_argmax_keeps_one_highest_quality_evidence_per_row():
    store = FeatureStore({
        "q": ObjectFeatures("q", "table", torch.ones(2), row_embeddings=torch.eye(2)),
        "e0": ObjectFeatures("e0", "text", torch.tensor([1.0, 0.0])),
        "e0_low": ObjectFeatures("e0_low", "text", torch.tensor([0.9, 0.1])),
        "e1": ObjectFeatures("e1", "image", torch.tensor([0.0, 1.0])),
    })
    selected, score = select_evidence(
        "d2_unique_argmax",
        [
            {"kind": "evidence", "evidence_id": "e0", "path_score": 1.0},
            {"kind": "evidence", "evidence_id": "e0_low", "path_score": 0.5},
            {"kind": "evidence", "evidence_id": "e1", "path_score": 0.8},
        ],
        query_id="q",
        store=store,
        content_keys={"e0": "a", "e0_low": "b", "e1": "c"},
        top_l=20,
        budget=4,
        support_cache={},
    )
    assert selected == ["e0", "e1"]
    assert score is not None and score > 0


def test_routed_support_counts_missing_selected_evidence_as_zero():
    values = _empty()
    _accumulate(
        values,
        {
            "query_id": "q",
            "query_kind": "implicit",
            "query_row_count": 2,
            "positive_evidence_by_target": {"t": ["e"]},
            "positive_evidence_rows_by_target": {"t": {"e": [0]}},
        },
        "t",
        [],
        {},
        {},
    )
    result = _finalize(values)
    assert result["actual_routed_row_b"] == 0
    assert result["matching_upper_row_b"] == 0


def test_edge_epoch_zero_has_real_metrics_and_requested_step_snapshots(tmp_path):
    model = StudentJoinabilityModel(2, 2, initialization="identity")
    store = FeatureStore({
        value: ObjectFeatures(value, "table", torch.tensor(vector, dtype=torch.float32))
        for value, vector in {"q": [1, 0], "p1": [1, 0.1], "p2": [0.9, 0],
                              "n1": [0, 1], "n2": [0.1, 0.9]}.items()
    })
    controller = _EpochController(
        output=tmp_path / "edge.pt", stage="student-edge", aggregator=PathAggregator(),
        primary_metric="dev_edge.macro_recall@1", min_delta=0, patience=0,
        store=store, device=torch.device("cpu"), dev_examples=_edges(),
        corpus_path=None, index_root=None, raw_index_root=None, args=argparse.Namespace(),
    )
    history = train_student_edges(
        model, _edges(), store, torch.optim.AdamW(model.parameters(), lr=1e-3),
        device=torch.device("cpu"), epochs=1, batch_size=1, seed=13, temperature=1,
        distillation_weight=0, dev_examples=_edges(), eval_epoch_zero=True,
        epoch_callback=controller, checkpoint_steps=(1, 2), step_callback=controller.save_step,
    )
    assert history[0]["epoch"] == 0
    assert history[0]["optimizer_updates_total"] == 0
    assert history[0]["dev_edge"]["by_relation"]["table_to_table"]["lists"] == 2
    assert history[0]["dev_loss"] > 0
    initial = load_student(tmp_path / "edge.epochs/epoch_000.pt", torch.device("cpu"))
    assert torch.equal(initial.projections["table"].weight, torch.eye(2))
    first = load_student(tmp_path / "edge.steps/step_000001.pt", torch.device("cpu"))
    last = load_student(tmp_path / "edge.steps/step_000002.pt", torch.device("cpu"))
    assert not torch.equal(initial.projections["table"].weight, first.projections["table"].weight)
    assert not torch.equal(first.projections["table"].weight, last.projections["table"].weight)
    assert student_projection_references(initial) == student_projection_references(last)


def test_exact_diagnostics_distinguish_all_positive_recall_from_any_hit():
    model = StudentJoinabilityModel(2, 2, initialization="identity")
    store = FeatureStore({
        value: ObjectFeatures(value, "table", torch.tensor(vector, dtype=torch.float32))
        for value, vector in {"q": [1, 0], "p1": [1, 0], "p2": [0, 1], "n": [-1, 0]}.items()
    })

    class Index:
        def search_many(self, source_ids, destination_type, k):
            return [[("p1", 1.0)]]

    result = exact_relation(
        model, Index(), store, {"table": ["p1", "p2", "n"]},
        ("table_to_table", ["q"], "table", "table", 1, {("q", "table"): {"p1", "p2"}}),
        torch.eye(2), torch.device("cpu"),
    )
    assert result["ann_any_positive_hit_rate"] == 1
    assert result["ann_positive_recall_micro"] == 0.5
    assert result["exact_positive_recall_micro"] == 0.5
    assert result["per_source"][0]["gt_edges"][1]["rank_min"] == 2


def test_r12_candidate_intervention_matches_real_base_and_excludes_global_positives():
    examples = _edges()
    known = scoring.global_edge_positive_ids(examples)
    hits = {scoring.edge_positive_key(examples[0]): [("p2", 9), ("hard", 8), ("n1", 7), ("n2", 6)]}
    arms = materialize_batch(examples, known, hits, 13, 0, 1)
    store = FeatureStore({value: ObjectFeatures(value, "table", torch.ones(2))
                          for value in ("q", "p1", "p2", "n1", "n2")})
    scored = scoring.score_edge_batch_in_batch(
        StudentJoinabilityModel(2, 2), examples, store, torch.device("cpu"),
        known_positive_ids=known, sampling_seed=13, sampling_context="epoch=0:step=1",
    )
    assert [row["candidate_ids"] for row in arms["base"]] == [list(row) for row in scored.candidate_ids]
    for base, changed in zip(arms["base"], arms["candidates"]):
        assert len(base["candidate_ids"]) == len(changed["candidate_ids"])
        assert base["positive_ids"] == changed["positive_ids"]
        assert set(changed["candidate_ids"]) & {"p1", "p2"} == set(changed["positive_ids"])
        assert "hard" in changed["candidate_ids"]
        assert not any(value == 0 for value in changed["confirmed_labels"])


def test_group_sample_keeps_cap_and_conditional_sampling_probability():
    rows = [{"query_id": group, "target_id": "t", "source_column_id": 0, "row_id": 0,
             "evidence_id": str(index), "source_table_id": group}
            for group in ("a", "b") for index in range(8)]
    used = Counter({"a": 3})
    selected, audit = sample_stratum(rows, 6, used, set(), "test")
    assert len(selected) == 5
    assert used == {"a": 4, "b": 4}
    assert audit["conditional_candidate_inclusion_probability_by_group"] == {"a": 1 / 8, "b": 4 / 8}


def test_empty_calibration_channel_has_no_scale():
    assert _scale([]) is None


def test_r12_teacher_scores_follow_candidate_pair_ids():
    rows = [{
        "query_id": "q",
        "candidate_ids": ["a", "b"],
        "candidate_pair_ids": [2, 0],
        "positive_id": "b",
        "positive_ids": ["b"],
        "confirmed_labels": [None, 1],
        "source_type": "table",
        "destination_type": "text",
    }]
    examples = r12_examples(rows, torch.tensor([1.0, 2.0, 3.0]), "teacher")
    assert examples[0].teacher_logits == (3.0, 1.0)
    with pytest.raises(ValueError, match="align"):
        r12_examples([{**rows[0], "candidate_pair_ids": [0]}], torch.ones(3), "teacher")


def test_r12_student_seed_permutations_preserve_all_frozen_batches():
    assert _training_batch_order(8, 13) == list(range(8))
    seed17 = _training_batch_order(8, 17)
    assert seed17 == _training_batch_order(8, 17)
    assert seed17 != list(range(8))
    assert sorted(seed17) == list(range(8))
    assert seed17 != _training_batch_order(8, 23)


def test_function_reference_balances_positive_and_random_pairs():
    relations = (
        ("table", "table"),
        ("table", "text"),
        ("table", "image"),
        ("text", "table"),
        ("image", "table"),
    )
    examples = []
    features = {}
    for source_type, destination_type in relations:
        for index in range(2):
            source_id = f"{source_type}_source_{destination_type}_{index}"
            positive_id = f"{destination_type}_positive_{source_type}_{index}"
            examples.append(EdgeExample(
                source_id,
                (positive_id,),
                0,
                source_type=source_type,
                destination_type=destination_type,
            ))
            features[source_id] = ObjectFeatures(source_id, source_type, torch.ones(2))
            features[positive_id] = ObjectFeatures(
                positive_id, destination_type, torch.ones(2)
            )
    for object_type in ("table", "text", "image"):
        for index in range(8):
            object_id = f"{object_type}_random_{index}"
            features[object_id] = ObjectFeatures(object_id, object_type, torch.ones(2))
    rows = _sample_function_pairs(
        examples, FeatureStore(features), pairs_per_kind=2
    )
    counts = Counter(
        (f"{row['source_type']}_to_{row['destination_type']}", row["reference_kind"])
        for row in rows
    )
    assert all(counts[(relation, kind)] == 2 for relation in (
        "table_to_table", "table_to_text", "table_to_image",
        "text_to_table", "image_to_table",
    ) for kind in ("positive_neighbor", "random_object"))


def test_function_gradient_ratio_uses_weighted_function_term():
    model = StudentJoinabilityModel(2, 2, initialization="identity")
    projection = model.projections["table"].weight
    relation = model.relations["table_to_table"]
    diagnostics = _apply_function_gradients(
        model, projection.sum(), 0.25 * relation.sum()
    )
    assert diagnostics["protocol"]["total"] == pytest.approx(2.0)
    assert diagnostics["weighted_function"]["total"] == pytest.approx(0.5)
    assert diagnostics["function_to_protocol_total_ratio"] == pytest.approx(0.25)
    assert torch.equal(projection.grad, torch.ones_like(projection))
    assert torch.equal(relation.grad, torch.full_like(relation, 0.25))


def test_r12_full_lake_checkpoint_resolution(tmp_path):
    resolved = _r12_checkpoints(tmp_path, "r12_candidates")
    assert [name for name, _path in resolved] == ["step0", "step178", "step356"]
    assert resolved[-1][1] == (
        tmp_path
        / "taskC_training/c_candidates_seed13/checkpoints/step_000356.pt"
    )
    explicit = _explicit_checkpoints([f"rescue={tmp_path / 'rescue.pt'}"])
    assert explicit == [("rescue", tmp_path / "rescue.pt")]
    assert _optimizer_updates("step659", tmp_path / "missing.pt") == 659
    with pytest.raises(ValueError, match="ID=PATH"):
        _explicit_checkpoints(["missing_separator"])


def test_conditional_kd_off_arm_uses_base_schedule(tmp_path):
    resolved = _r12_checkpoints(tmp_path, "r12_kd_off")
    assert resolved[1][1] == (
        tmp_path / "taskC_training/c_kd_off_seed13/checkpoints/step_000178.pt"
    )
    extension = _r12_checkpoints(tmp_path, "r12_base_extension")
    assert [name for name, _path in extension] == ["step659", "step1318"]


def test_r12_quality_gate_includes_explicit_recall():
    baseline = {
        "recall@10": 0.50,
        "implicit_recall@10": 0.40,
        "explicit_recall@10": 0.60,
        "actual_routed_support@10,4": 0.10,
        "valid_path@10,4": 0.10,
        "valid_discovery_count@10,4": 1,
    }
    results = {
        "f1_union_direct": baseline,
        "f3_union_rrf_equal": {
            **baseline,
            "explicit_recall@10": 0.581,
            "actual_routed_support@10,4": 0.20,
        },
        "f5_reserved_half": {
            **baseline,
            "explicit_recall@10": 0.57,
            "actual_routed_support@10,4": 0.30,
        },
    }
    selection = _quality_selection(
        results, fusion_ids=R12_FUSION_IDS, require_explicit=True
    )
    assert selection["eligible"] == ["f1_union_direct", "f3_union_rrf_equal"]
    assert selection["selected"] == "f3_union_rrf_equal"


def test_actual_routed_support_requires_correct_evidence_row():
    record = {
        "positive_evidence_by_target": {"t": ["e0", "e1"]},
        "positive_evidence_rows_by_target": {
            "t": {"e0": [0], "e1": [1]}
        },
    }
    valid, supported, routed = _selected_valid_rows(
        record,
        {
            "selected_evidence_ids": ["e0", "e1"],
            "routed_rows": {"e0": 0, "e1": 0},
        },
        "t",
    )
    assert valid == {"e0", "e1"}
    assert supported == {0, 1}
    assert routed == {0}


def test_target_channels_records_argmax_route_for_selected_evidence():
    store = FeatureStore({
        "q": ObjectFeatures(
            "q", "table", torch.ones(2), row_embeddings=torch.eye(2)
        ),
        "e": ObjectFeatures("e", "text", torch.tensor([0.0, 1.0])),
    })
    direct, evidence = _target_channels(
        {
            "query_id": "q",
            "paths_by_target": {
                "t": [{
                    "kind": "evidence",
                    "evidence_id": "e",
                    "evidence_type": "text",
                    "path_score": 1.0,
                }]
            },
        },
        retention="d0_content_dedup",
        scorer=_DirectScorer(),
        store=store,
        content_keys={"e": "e"},
        top_l=20,
        evidence_budget=4,
        pair_batch_size=16,
        intervention="original_mixed",
        intervention_stats=empty_intervention_stats(),
    )
    assert direct[0]["routed_rows"] == {"e": 1}
    assert evidence[0]["evidence_rank"] == 1


def test_extension_schedule_is_exact_suffix_of_seeded_schedule():
    examples = _edges()
    complete = extension_batches(examples, 1, 5, 13)
    suffix = extension_batches(examples, 3, 5, 13)
    assert [step for step, *_rest in suffix] == [3, 4, 5]
    assert [
        [row.candidate_ids for row in batch] for _step, _epoch, _index, batch in suffix
    ] == [
        [row.candidate_ids for row in batch]
        for _step, _epoch, _index, batch in complete[2:]
    ]


def test_extension_teacher_shards_cover_global_suffix():
    counts = [_shard_count(7, 22, index, 2) for index in range(2)]
    assert counts == [7, 8]
    assert sum(counts) == 15

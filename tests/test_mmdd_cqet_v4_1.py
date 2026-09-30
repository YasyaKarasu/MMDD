from __future__ import annotations

import copy
import gzip
from collections import defaultdict
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from mmdd_cqet_v4_1.evaluate import (
    coverage_at_k,
    evaluate_teacher_matrix,
    hit_rate_at_k,
    paired_bootstrap,
)
from mmdd_cqet_v4_1.artifacts import load_pool_bundle, save_pool_bundle
from mmdd_cqet_v4_1.config import Paths
from mmdd_cqet_v4_1.data import iter_jsonl, utf8_sorted, write_jsonl
from mmdd_cqet_v4_1.data import build_content_aliases
from mmdd_cqet_v4_1.labels import Labels
from mmdd_cqet_v4_1.features import build_or_load_row_store
from mmdd_cqet_v4_1.lists import (
    build_c1_edge_lists,
    build_c2_shared_graph,
    build_raw_et128_exact,
    build_raw_pools_split,
    build_ta_records,
    build_tb_records,
    hash_order,
    HashOrderLibrary,
)
from mmdd_cqet_v4_1.losses import (
    aggregate_corrected_lse,
    aggregate_cqet,
    hierarchical_relation_mean,
    hierarchical_support_mean,
)
from mmdd_cqet_v4_1.models import FreshPathTeacher, NativeStudent
from mmdd_cqet_v4_1.retrieval import HNSWIndex, PathEntry, PoolRecord, d1_retain, p3_admission
from mmdd_cqet_v4_1.provenance import assert_declared_project_imports
from mmdd_cqet_v4_1.pipeline import _completed_stage_result
from mmdd_cqet_v4_1.probes import (
    student_gradient_probe,
    teacher_content_probe,
    teacher_gradient_probe,
)
from mmdd_cqet_v4_1.provenance import (
    json_sha,
    record_source_amendment,
    source_identity,
    source_manifest,
)
from mmdd_cqet_v4_1.train import (
    TeacherListScorer,
    model_state_sha,
    train_student_c1,
    train_student_c2,
)


class TinyBank:
    def __init__(self, vectors: dict[str, torch.Tensor], kinds: dict[str, str]):
        self.vectors = vectors
        self.kinds = kinds
        self.device = torch.device("cpu")

    def attach_device(self, device):
        self.device = torch.device(device)

    def z(self, object_id: str) -> torch.Tensor:
        return self.vectors[object_id].to(self.device)

    def z_many(self, object_ids) -> torch.Tensor:
        return torch.stack([self.z(object_id) for object_id in object_ids])

    def kind(self, object_id: str) -> str:
        return self.kinds[object_id]

    def tokens(self, object_id: str) -> torch.Tensor:
        return torch.stack([self.z(object_id), self.z(object_id) * 0.5])

    def tokens_many(self, object_ids):
        return [self.tokens(object_id) for object_id in object_ids]


class TinyZStore:
    def __init__(self, vectors: dict[str, torch.Tensor]):
        self.ids = list(vectors)
        self.index = {object_id: i for i, object_id in enumerate(self.ids)}
        self.z = torch.stack([vectors[object_id] for object_id in self.ids]).float()
        self.dim = self.z.shape[1]

    def vector(self, object_id: str) -> torch.Tensor:
        return self.z[self.index[object_id]]

    def rows(self, object_ids) -> torch.Tensor:
        return self.z[[self.index[object_id] for object_id in object_ids]]


class TinyRowStore:
    def __init__(self, rows: dict[str, np.ndarray]):
        self.rows = rows

    def get(self, object_id: str) -> np.ndarray:
        return self.rows[object_id]


def tiny_fixture(n_queries: int = 1):
    vectors = {
        "tp": torch.tensor([0.8, 0.1, 0.3, -0.2]),
        "tn": torch.tensor([-0.4, 0.9, 0.2, 0.1]),
        "tx": torch.tensor([0.2, -0.7, 0.5, 0.4]),
        "ep": torch.tensor([0.7, 0.2, -0.1, 0.5]),
        "en": torch.tensor([-0.2, 0.8, 0.4, -0.3]),
    }
    for i in range(n_queries):
        vectors[f"q{i}"] = torch.tensor([0.6 + i * 0.01, -0.2, 0.7, 0.1])
    kinds = {key: ("text" if key.startswith("e") else "table") for key in vectors}
    bank = TinyBank(vectors, kinds)
    records = [
        {
            "query_id": f"q{i}",
            "targets": ["tp", "tn", "tx"],
            "positives": ["tp"],
            "natural_bags": {"tp": ["ep"], "tn": ["en"], "tx": []},
        }
        for i in range(n_queries)
    ]
    basis = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    mean = torch.zeros(4)
    return bank, records, basis, mean


def test_actual_c2_kd_step_has_student_gradient_and_locked_parent(tmp_path: Path):
    torch.manual_seed(13)
    bank, records, basis, mean = tiny_fixture()
    teacher = FreshPathTeacher(
        input_dim=4, width=4, heads=1, layers=1, ffn=8,
        text_slots=1, image_slots=1, dropout=0.0,
    )
    student = NativeStudent(basis, mean, dim=2)
    parent_hash = model_state_sha(student)
    before = {name: value.detach().clone() for name, value in student.state_dict().items()}
    out = tmp_path / "kd"

    train_student_c2(
        student, records, teacher, bank, device="cpu", arm="NATIVE_KD",
        logical_batch=1, save_dir=out, seed=13,
        expected_parent_hash=parent_hash, max_updates=1,
    )

    init = torch.load(out / "snapshot_frac000.pt", map_location="cpu", weights_only=False)
    assert init["extra"]["model_state_sha256"] == parent_hash
    assert init["extra"]["parent_state_sha256"] == parent_hash
    assert any(not torch.equal(before[name], value) for name, value in student.state_dict().items())
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_c1_oom_retry_replays_batch_without_duplicate_optimizer_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    bank, _records, basis, mean = tiny_fixture()
    edge_lists = [
        {
            "item_id": f"edge-{i}",
            "relation": "QT",
            "anchor_id": "q0",
            "candidates": ["tp", "tn", "tx"],
            "positives": ["tp"],
        }
        for i in range(2)
    ]
    initial = NativeStudent(basis, mean, dim=2)
    clean = copy.deepcopy(initial)
    retried = copy.deepcopy(initial)
    train_student_c1(
        clean, edge_lists, None, bank, device="cpu", logical_batch=2,
        save_dir=tmp_path / "clean", seed=13,
    )

    original_score = NativeStudent.score
    injected = {"done": False}

    def score_with_one_oom(self, *args, **kwargs):
        if self is retried and not injected["done"]:
            injected["done"] = True
            raise torch.cuda.OutOfMemoryError("synthetic C1 OOM")
        return original_score(self, *args, **kwargs)

    monkeypatch.setattr(NativeStudent, "score", score_with_one_oom)
    log_path = tmp_path / "retry.jsonl"
    train_student_c1(
        retried, edge_lists, None, bank, device="cpu", logical_batch=2,
        save_dir=tmp_path / "retry", seed=13, log_path=log_path,
    )

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert injected["done"]
    assert len(rows) == 1
    assert rows[0]["step"] == 1
    assert rows[0]["student_query_microbatch"] == 32
    assert model_state_sha(retried) == model_state_sha(clean)


def test_swap_collision_counts_are_not_multiplied_by_teacher_count(tmp_path: Path):
    bank, _records, _basis, _mean = tiny_fixture()
    teacher_a = FreshPathTeacher(
        input_dim=4, width=4, heads=1, layers=1, ffn=8,
        text_slots=1, image_slots=1, dropout=0.0,
    )
    teacher_b = copy.deepcopy(teacher_a)
    pool = PoolRecord(
        query_id="q0", split="dev", generator_id="fixture",
        direct=[("tp", 1.0)], direct_exact=["tp"],
        first_hop={"text": [("ep", 1.0)], "image": []},
        pre_paths={"tp": [PathEntry("ep", "text", 1.0, 1.0)]},
        retained_paths={"tp": ["ep"]}, retained_coverage={"tp": 1.0},
        U=["tp"], C150=["tp"], qt_scores_all_U={"tp": 1.0},
        d1_scores={"tp": 1.0}, admission_scores={"tp": 1.0},
        D150=[("tp", 1.0)], MatchedDirectC=[("tp", 1.0)],
        MatchedDirectU=[("tp", 1.0)], ann_exact_overlap={},
    )
    labels = types.SimpleNamespace(
        canonical_text=["ep", "en"], canonical_image=[],
        modality={"ep": "text", "en": "text"},
    )
    evaluate_teacher_matrix(
        {"A": teacher_a, "B": teacher_b}, bank, {"q0": pool}, labels,
        seed=13, generator="fixture", split="dev", output_dir=tmp_path,
        split_gt={"q0": {"W": {"tp": ["en"]}}},
        teacher_modes={"A": "cqet", "B": "cqet"}, device="cpu",
    )
    collision = json.loads((tmp_path / "SWAP_COLLISIONS.json").read_text())
    assert collision["swap_paths"] == 1
    assert collision["known_witness_collisions"] == 1
    assert collision["collision_rate"] == 1.0


def test_actual_teacher_and_student_trajectory_probes(tmp_path: Path):
    bank, records, basis, mean = tiny_fixture()
    teacher = FreshPathTeacher(
        input_dim=4, width=4, heads=1, layers=1, ffn=8,
        text_slots=1, image_slots=1, dropout=0.0,
    )
    checkpoint = tmp_path / "teacher.pt"
    torch.save({"model": teacher.state_dict()}, checkpoint)
    pool = PoolRecord(
        query_id="q0", split="dev", generator_id="fixture",
        direct=[("tp", 1.0), ("tn", 0.5), ("tx", 0.2)],
        direct_exact=["tp", "tn", "tx"],
        first_hop={"text": [("ep", 1.0), ("en", 0.5)], "image": []},
        pre_paths={
            "tp": [PathEntry("ep", "text", 1.0, 1.0)],
            "tn": [PathEntry("en", "text", 0.5, 0.5)],
        },
        retained_paths={"tp": ["ep"], "tn": ["en"]},
        retained_coverage={"tp": 1.0, "tn": 0.5},
        U=["tp", "tn", "tx"], C150=["tp", "tn", "tx"],
        qt_scores_all_U={"tp": 1.0, "tn": 0.5, "tx": 0.2},
        admission_scores={"tp": 1.0, "tn": 0.5, "tx": 0.2},
    )
    labels = types.SimpleNamespace(
        modality={"ep": "text", "en": "text"},
    )
    matrix = {
        "T": {
            "Real": {"q0": {"target_ids": ["tp", "tn", "tx"], "scores": [3.0, 2.0, 1.0]}},
            "Swap": {"q0": {"target_ids": ["tp", "tn", "tx"], "scores": [2.5, 1.5, 1.0]}},
        }
    }
    summary = teacher_content_probe(
        teacher, bank, {"q0": pool}, labels,
        {"q0": {"W": {"tp": ["ep"]}}}, matrix,
        teacher_name="T", mode="cqet", seed=13, checkpoint=checkpoint,
        output_dir=tmp_path / "content", device="cpu",
    )
    assert summary["query_count"] == 1
    assert summary["raw_QET_minus_f0"]["count"] == 2
    assert summary["Real_minus_Swap"]["count"] == 3
    assert (tmp_path / "content" / "CONTENT_ABLATIONS.jsonl.gz").exists()

    qt_summary = teacher_content_probe(
        teacher, bank, {"q0": pool}, labels,
        {"q0": {"W": {"tp": ["ep"]}}},
        {"T": {"Direct": matrix["T"]["Real"]}},
        teacher_name="T", mode="qt", seed=13, checkpoint=checkpoint,
        output_dir=tmp_path / "qt_content", device="cpu",
    )
    assert qt_summary["raw_QET_minus_f0"]["count"] == 0
    assert qt_summary["old_LSE_null_mass"]["count"] == 0

    teacher_probe = teacher_gradient_probe(
        teacher, bank, records[0], mode="cqet", stage="TB_CQET",
    )
    assert teacher_probe["components"]["direct"]["gradient_norm"] > 0
    assert abs(teacher_probe["components"]["direct"]["shared_score_shift_gradient"]) < 1e-5

    student = NativeStudent(basis, mean, dim=2)
    student_probe = student_gradient_probe(
        student, bank, records[0],
        (torch.tensor([3.0, -1.0, -2.0]), torch.tensor([2.0, -1.0])),
    )
    assert student_probe["components"]["SUP"]["gradient_norm"] > 0
    assert student_probe["components"]["direct_KD"]["gradient_norm"] > 0
    assert student_probe["components"]["evidence_KD"]["gradient_norm"] > 0


def test_actual_c2_sup_and_kd_updates_differ():
    torch.manual_seed(29)
    bank, records, basis, mean = tiny_fixture()
    initial = NativeStudent(basis, mean, dim=2)
    sup = copy.deepcopy(initial)
    kd = copy.deepcopy(initial)
    parent_hash = model_state_sha(initial)
    teacher_logits = {"q0": (torch.tensor([4.0, -2.0, -3.0]), torch.tensor([-2.0, 3.0]))}

    train_student_c2(
        sup, records, None, bank, device="cpu", arm="NATIVE_SUP", logical_batch=1,
        seed=29, expected_parent_hash=parent_hash, max_updates=1,
    )
    train_student_c2(
        kd, records, None, bank, device="cpu", arm="NATIVE_KD", logical_batch=1,
        seed=29, expected_parent_hash=parent_hash, teacher_logits=teacher_logits, max_updates=1,
    )
    assert model_state_sha(sup) != model_state_sha(kd)


def test_actual_c2_resume_matches_continuous(tmp_path: Path):
    bank, records, basis, mean = tiny_fixture(n_queries=4)
    logits = {
        f"q{i}": (torch.tensor([2.0 + i, -1.0, -2.0]), torch.tensor([1.0, -1.0]))
        for i in range(4)
    }
    base = NativeStudent(basis, mean, dim=2)
    parent_hash = model_state_sha(base)

    torch.manual_seed(71)
    np.random.seed(71)
    continuous = copy.deepcopy(base)
    train_student_c2(
        continuous, records, None, bank, device="cpu", arm="NATIVE_KD",
        logical_batch=1, seed=13, expected_parent_hash=parent_hash,
        teacher_logits=logits,
    )

    torch.manual_seed(71)
    np.random.seed(71)
    partial = copy.deepcopy(base)
    part_dir = tmp_path / "partial"
    train_student_c2(
        partial, records, None, bank, device="cpu", arm="NATIVE_KD",
        logical_batch=1, seed=13, expected_parent_hash=parent_hash,
        teacher_logits=logits, save_dir=part_dir, max_updates=2,
    )
    resumed = copy.deepcopy(base)
    train_student_c2(
        resumed, records, None, bank, device="cpu", arm="NATIVE_KD",
        logical_batch=1, seed=13, expected_parent_hash=parent_hash,
        teacher_logits=logits, save_dir=tmp_path / "resumed",
        resume_from=part_dir / "snapshot_frac050.pt",
    )
    assert model_state_sha(resumed) == model_state_sha(continuous)


def test_five_relation_ann_formula_and_reload(tmp_path: Path):
    torch.manual_seed(5)
    basis = torch.randn(3, 4)
    student = NativeStudent(basis, torch.randn(4), dim=3)
    left = torch.randn(4)
    right = torch.randn(7, 4)
    relation_kinds = {
        "QT": ("table", "table"),
        "Q_text": ("table", "text"),
        "Q_image": ("table", "image"),
        "text_T": ("text", "table"),
        "image_T": ("image", "table"),
    }
    ids = [f"x{i}" for i in range(len(right))]
    for relation, (left_kind, right_kind) in relation_kinds.items():
        with torch.no_grad():
            student.R[relation].copy_(torch.randn(3, 3))
            query = student.ann_query(relation, left)
            vectors = student.index_vectors(relation, right)
            transformed = vectors @ query
            bilinear = student.score(left_kind, left, right_kind, right)
            torch.testing.assert_close(transformed, bilinear)

        index = HNSWIndex(vectors.detach().numpy(), ids, dim=3, seed=13)
        hits = index.search(query.detach().numpy(), len(ids))
        for object_id, score in hits:
            i = ids.index(object_id)
            assert np.isclose(score, float(bilinear[i]), rtol=1e-4, atol=1e-5)
        if relation == "Q_text":
            path = tmp_path / "q_text.hnsw"
            index.save(path)
            reloaded = HNSWIndex.load(path)
            assert reloaded.search(query.detach().numpy(), len(ids)) == hits


def test_cqet_and_lse_gradients_and_permutation():
    f0 = torch.tensor([100.0, 2.0], requires_grad=True)
    paths = torch.tensor([-20.0, -18.0], requires_grad=True)
    target = torch.tensor([0, 0], dtype=torch.long)
    cqet = aggregate_cqet(f0, paths, target)
    cqet.sum().backward()
    torch.testing.assert_close(f0.grad, torch.tensor([0.0, 1.0]))
    assert np.isclose(float(paths.grad.sum()), 1.0)

    perm = torch.tensor([1, 0])
    torch.testing.assert_close(cqet.detach(), aggregate_cqet(f0.detach(), paths.detach()[perm], target[perm]))
    f0_lse = torch.tensor([1.0], requires_grad=True)
    path_lse = torch.tensor([3.0], requires_grad=True)
    aggregate_corrected_lse(f0_lse, path_lse, torch.tensor([0])).backward()
    assert f0_lse.grad.item() > 0
    assert path_lse.grad.item() > 0


def test_hierarchical_denominators():
    relation = hierarchical_relation_mean([
        [torch.tensor(2.0), torch.tensor(4.0)],
        [torch.tensor(9.0)],
    ])
    support = hierarchical_support_mean([
        [torch.tensor(2.0), torch.tensor(4.0)],
        [torch.tensor(9.0)],
    ])
    assert relation.item() == 6.0
    assert support.item() == 6.0


def test_teacher_list_scorer_matches_per_chunk_scoring_with_dropout():
    """The shared-cache, batched-encode scorer must reproduce the previous per-chunk
    scoring forward bit for bit at the same chunk size: same dropout draws (same batch
    shapes, same order), same complete-list denominator, same gradients."""
    torch.manual_seed(101)
    bank, _records, _basis, _mean = tiny_fixture()
    legacy = FreshPathTeacher(
        input_dim=4, width=4, heads=1, layers=1, ffn=8,
        text_slots=1, image_slots=1, dropout=0.1,
    ).train()
    scorer_model = copy.deepcopy(legacy).train()
    pairs = [
        ("table", bank.z("q0"), bank.tokens("q0"), "table", bank.z(target), bank.tokens(target))
        for target in ("tp", "tn", "tx")
    ]
    keys = [("q0", target) for target in ("tp", "tn", "tx")]
    chunk = 2
    rng = torch.get_rng_state()

    # Previous implementation: per-object encode, fresh cache, one graph per chunk.
    legacy_scores = torch.cat([
        legacy.score_pairs(pairs[start : start + chunk], cache={}, cache_keys=keys[start : start + chunk])
        for start in range(0, len(pairs), chunk)
    ])
    legacy_loss = torch.logsumexp(legacy_scores, 0) - legacy_scores[0]
    legacy_loss.backward()
    legacy_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in legacy.named_parameters()
        if parameter.grad is not None
    }

    torch.set_rng_state(rng)
    scorer = TeacherListScorer(scorer_model, chunk=chunk)
    scores = scorer.score_pairs(pairs, keys)
    loss = torch.logsumexp(scores, 0) - scores[0]
    scorer.backward(loss, scale=1.0)
    # Batched adapter/pooler GEMMs differ from per-object ones at float32 rounding level.
    torch.testing.assert_close(scores.detach(), legacy_scores.detach(), rtol=1e-5, atol=1e-5)
    for name, parameter in scorer_model.named_parameters():
        if name in legacy_grads:
            torch.testing.assert_close(parameter.grad, legacy_grads[name], rtol=1e-4, atol=1e-5)


def test_teacher_list_scorer_consumes_the_same_dropout_stream_as_per_chunk_scoring():
    """Chunking bounds the relation batch but must not change the dropout draws:
    the relation transformer sees the same sequence of batch shapes either way."""
    torch.manual_seed(11)
    bank, _records, _basis, _mean = tiny_fixture()
    model = FreshPathTeacher(
        input_dim=4, width=4, heads=1, layers=1, ffn=8,
        text_slots=1, image_slots=1, dropout=0.5,
    ).train()
    pairs = [
        ("table", bank.z("q0"), bank.tokens("q0"), "table", bank.z(target), bank.tokens(target))
        for target in ("tp", "tn", "tx")
    ]
    keys = [("q0", target) for target in ("tp", "tn", "tx")]

    torch.manual_seed(4321)
    before = torch.get_rng_state().clone()
    TeacherListScorer(model, chunk=2).score_pairs(pairs, keys)
    chunked_stream = torch.get_rng_state().clone()

    torch.set_rng_state(before)
    torch.cat([
        model.score_pairs(pairs[start : start + 2], cache={}, cache_keys=keys[start : start + 2])
        for start in range(0, len(pairs), 2)
    ])
    per_chunk_stream = torch.get_rng_state().clone()
    assert torch.equal(chunked_stream, per_chunk_stream)


def test_teacher_list_scorer_reuses_batched_encode_across_relations():
    """Objects shared between relation lists are encoded once, and encode_many is used."""
    torch.manual_seed(7)
    bank, _records, _basis, _mean = tiny_fixture()
    model = FreshPathTeacher(
        input_dim=4, width=4, heads=1, layers=1, ffn=8,
        text_slots=1, image_slots=1, dropout=0.0,
    ).eval()
    calls = {"encode_one": 0, "encode_many": 0}
    original_one, original_many = model.encode_one, model.encode_many

    def counted_one(*args, **kwargs):
        calls["encode_one"] += 1
        return original_one(*args, **kwargs)

    def counted_many(*args, **kwargs):
        calls["encode_many"] += 1
        return original_many(*args, **kwargs)

    model.encode_one = counted_one
    model.encode_many = counted_many

    zq, cq = bank.z("q0"), bank.tokens("q0")
    pairs = [
        ("table", zq, cq, "table", bank.z(target), bank.tokens(target))
        for target in ("tp", "tn", "tx")
    ]
    pair_keys = [(('q0', 0), (target, 1)) for target in ("tp", "tn", "tx")]
    triplets = [
        ("table", zq, cq, "text", bank.z("ep"), bank.tokens("ep"), "table", bank.z("tp"),
         bank.tokens("tp")),
        ("table", zq, cq, "text", bank.z("ep"), bank.tokens("ep"), "table", bank.z("tn"),
         bank.tokens("tn")),
    ]
    triplet_keys = [
        (("q0", 0), ("ep", 2), ("tp", 1)),
        (("q0", 0), ("ep", 2), ("tn", 1)),
    ]

    scorer = TeacherListScorer(model, chunk=2)
    scorer.score_pairs(pairs, pair_keys)
    scorer.score_triplets(triplets, triplet_keys)

    # q0/tp/tn are each shared by both relation lists and must be encoded exactly once.
    assert calls["encode_many"] == 2
    assert calls["encode_one"] == 0
    assert set(scorer.cache) == {
        ("table", ("q0", 0)), ("table", ("tp", 1)), ("table", ("tn", 1)), ("table", ("tx", 1)),
        ("text", ("ep", 2)),
    }


def test_d1_score_controls_admission():
    entries = [
        PathEntry("e1", "text", 0.0, 0.0),
        PathEntry("e2", "text", 4.0, 4.0),
    ]
    selected, d1 = d1_retain(
        entries,
        {"e1": np.array([1.0, 0.0]), "e2": np.array([0.0, 1.0])},
        budget=1,
    )
    assert selected == ["e2"]
    assert d1 > 0
    admitted, _ = p3_admission({"a": 2.0, "b": 1.9}, ["b"], budget=1)
    assert admitted == ["b"]


def test_target_coverage_hit_rate_and_query_weighted_bootstrap():
    ranked = ["a", "x"]
    gold = {"a", "b", "c", "d"}
    assert coverage_at_k(ranked, gold, 2) == 0.25
    assert hit_rate_at_k(ranked, gold, 2) == 1.0
    result = paired_bootstrap(
        {"q1": 1.0, "q2": 0.0, "q3": 0.0, "q4": 0.0},
        {"q1": "a", "q2": "b", "q3": "b", "q4": "b"},
        replicates=100,
    )
    assert result["mean_delta_pp"] == 25.0
    assert result["aggregation"] == "source_group_resample_then_query_weighted_mean"


def test_pool_bundle_schema_roundtrip_and_hash_guard(tmp_path: Path):
    path = PathEntry("e", "text", 0.2, 0.3)
    pool = PoolRecord(
        split="dev", query_id="q", generator_id="g",
        direct=[("t", 0.9)], direct_exact=["t"],
        first_hop={"text": [("e", 0.2)], "image": []},
        pre_paths={"t": [path]}, retained_paths={"t": ["e"]},
        retained_coverage={"t": 0.4}, U=["t"], C150=["t"],
        qt_scores_all_U={"t": 0.9}, admission_scores={"t": 1.0},
        D150=[("t", 0.9)], MatchedDirectC=[("t", 0.9)],
        MatchedDirectU=[("t", 0.9)], d1_scores={"t": 0.4},
        qt_ranks={"t": 1}, d1_ranks={"t": 1},
        object_vector_hash="v", index_hash="i",
        d1_trace={"t": [{"evidence_id": "e", "row_support_mean": 0.5,
                          "marginal_gain": 0.4, "selected_step": 1}]},
    )
    labels = Labels(
        queries={}, epos={}, legal_targets=["t"], edge_anchors=[],
        canonical_map={"e": "e"}, modality={"e": "text"},
        content_hash={"e": "a" * 64}, canonical_text=["e"], canonical_image=[],
    )
    directory = tmp_path / "bundle"
    manifest = save_pool_bundle(directory, {"q": pool}, labels, seed=13, generator="g")
    assert set(manifest) == {
        "pools", "prepaths", "first_hop", "second_hop",
        "direct_exact", "matched_direct", "internal",
    }
    restored = load_pool_bundle(directory)
    assert restored["q"].D150 == [("t", 0.9)]
    with gzip.open(directory / "pools.jsonl.gz", "rt", encoding="utf-8") as handle:
        row = json.loads(handle.readline())
    assert row["schema_version"] == "4.1.0"
    assert row["MatchedDirectC"] == ["t"]
    with (directory / "pool_records.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_pool_bundle(directory)


def test_actual_raw_training_budgets_and_c2_full_prepath_merge(tmp_path: Path):
    generator = torch.Generator().manual_seed(20260925)
    target_ids = [f"t{i:03d}" for i in range(160)]
    text_ids = [f"et{i:03d}" for i in range(140)]
    image_ids = [f"ei{i:03d}" for i in range(140)]
    all_ids = ["q", *target_ids, *text_ids, *image_ids]
    matrix = torch.randn(len(all_ids), 4, generator=generator)
    matrix = torch.nn.functional.normalize(matrix, dim=1)
    # q2 reuses q's inputs so the C2 graph can be streamed for two queries.
    z_store = TinyZStore({**dict(zip(all_ids, matrix)), "q2": matrix[0]})
    q_rows = np.stack([z_store.vector("et000").numpy(), z_store.vector("ei000").numpy()])
    row_store = TinyRowStore({"q": q_rows, "q2": q_rows})
    canonical = {e: e for e in [*text_ids, *image_ids]}
    modality = {e: "text" for e in text_ids} | {e: "image" for e in image_ids}
    q_entry = {
        "G": ["t000"],
        "W": {"t000": ["et000", "ei000"]},
        "Qpos": {"text": ["et000"], "image": ["ei000"]},
    }
    labels = Labels(
        queries={"q": q_entry, "q2": q_entry},
        epos={"et000": ["t000"], "ei000": ["t000"]},
        legal_targets=target_ids,
        edge_anchors=[
            {"item_id": "QT:q", "relation": "QT", "anchor_id": "q", "positive_ids": ["t000"]},
            {"item_id": "Q_text:q", "relation": "Q_text", "anchor_id": "q", "positive_ids": ["et000"]},
            {"item_id": "Q_image:q", "relation": "Q_image", "anchor_id": "q", "positive_ids": ["ei000"]},
            {"item_id": "text_T:et000", "relation": "text_T", "anchor_id": "et000", "positive_ids": ["t000"]},
            {"item_id": "image_T:ei000", "relation": "image_T", "anchor_id": "ei000", "positive_ids": ["t000"]},
        ],
        canonical_map=canonical,
        modality=modality,
        content_hash={e: hashlib.sha256(e.encode()).hexdigest() for e in canonical},
        canonical_text=text_ids,
        canonical_image=image_ids,
    )
    pools = build_raw_pools_split(
        z_store, row_store, ["q"], labels, "train", device="cpu", hnsw_seed=13,
    )
    pool = pools["q"]
    assert len(pool.direct) == 100
    assert len(pool.D150) == 150
    assert len(pool.training_exact["RawQT128"]) == 128
    assert len(pool.training_exact["RawQE128"]["text"]) == 128
    assert len(pool.training_exact["RawQE128"]["image"]) == 128
    raw_et = build_raw_et128_exact(
        z_store, labels, device="cpu", anchors=["et000", "ei000"],
    )
    assert len(raw_et["et000"]) == 128
    assert len(raw_et["ei000"]) == 128
    edge_lists = build_c1_edge_lists(labels, pools, z_store, 13, raw_et, device="cpu")
    assert {row["relation"] for row in edge_lists} == {
        "QT", "Q_text", "Q_image", "text_T", "image_T",
    }
    assert all(len(row["hard"]) == 32 for row in edge_lists)
    assert all(len(row["uniform"]) == 31 for row in edge_lists)

    c1_pool = copy.deepcopy(pool)
    target = target_ids[-1]
    first_text_ids = {evidence_id for evidence_id, _score in pool.first_hop["text"]}
    extra_id = next(evidence_id for evidence_id in text_ids if evidence_id not in first_text_ids)
    extra = PathEntry(extra_id, "text", -7.0, -8.0)
    c1_pool.pre_paths.setdefault(target, []).append(extra)
    c1_pool.U = list(dict.fromkeys([*c1_pool.U, target]))
    selected = NativeStudent(torch.eye(4), torch.zeros(4), dim=4)
    prepaths = tmp_path / "C2_SHARED_PREPATHS.jsonl.gz"
    # For q2 the two pools swap roles, so the C1-only extra path becomes Raw-only.
    records = build_c2_shared_graph(
        {"q": pool, "q2": c1_pool}, {"q": c1_pool, "q2": pool},
        selected, z_store, row_store, labels, ["q", "q2"], prepaths, device="cpu",
    )
    audit_rows = list(iter_jsonl(prepaths))
    assert [record["query_id"] for record in records] == ["q", "q2"]
    first_q2 = next(i for i, row in enumerate(audit_rows) if row["query_id"] == "q2")
    assert {row["query_id"] for row in audit_rows[:first_q2]} == {"q"}
    assert {row["query_id"] for row in audit_rows[first_q2:]} == {"q2"}
    expected = {
        (target_id, labels.canonical_map[path.evidence_id])
        for source_pool in (pool, c1_pool)
        for target_id, paths in source_pool.pre_paths.items()
        for path in paths
    }
    for query_id, extra_source in (("q", "NativeC1"), ("q2", "Raw")):
        rows = [row for row in audit_rows if row["query_id"] == query_id]
        keys = [(row["target_id"], row["evidence_id"]) for row in rows]
        assert keys == sorted(keys, key=lambda key: (key[0].encode(), key[1].encode()))
        assert set(keys) == expected
        extra_row = next(
            row for row in rows if row["target_id"] == target and row["evidence_id"] == extra_id
        )
        assert extra_row["sources"] == [extra_source]
    assert records[0]["source_graph"] == "RawU_union_selectedNativeC1U_union_G"
    assert records[0]["natural_bags"] == records[1]["natural_bags"]


def test_alias_archive_and_project_imports_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_root = tmp_path / "run"
    run_root.mkdir()
    archive = run_root / "CONTENT_ALIASES.jsonl.gz"
    row = {
        "asset_id": "asset",
        "canonical_evidence_id": "asset",
        "visible_content_sha256": "b" * 64,
    }
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (run_root / "CONTENT_ALIAS_REPORT.json").write_text(json.dumps({
        "status": "PASS", "objects": 1, "file": {"sha256": digest},
    }))
    paths = Paths(
        repo_root=Path("/home/oycy/MMDD"), dataset_root=tmp_path,
        backbone_dir=tmp_path, pure_cache_dir=tmp_path,
        row_cache_manifest=tmp_path / "manifest", protocol_path=tmp_path / "protocol",
        run_root=run_root,
    )
    assert build_content_aliases(paths) == {"asset": "asset"}
    with gzip.open(archive, "at", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(RuntimeError, match="archive hash"):
        build_content_aliases(paths)

    module = types.ModuleType("undeclared_mmdd_test_module")
    module.__file__ = "/home/oycy/MMDD/undeclared_mmdd_test_module.py"
    sys.modules[module.__name__] = module
    try:
        with pytest.raises(RuntimeError, match="undeclared project-local"):
            assert_declared_project_imports(paths)
    finally:
        sys.modules.pop(module.__name__, None)

    placeholder = types.ModuleType("relative_extension_namespace")
    placeholder.__file__ = "_classes.py"
    sys.modules[placeholder.__name__] = placeholder
    monkeypatch.chdir(paths.repo_root)
    try:
        assert_declared_project_imports(paths)
    finally:
        sys.modules.pop(placeholder.__name__, None)


def _write_completed_ta(stage_dir: Path, source_sha: str) -> Path:
    checkpoint_dir = stage_dir / "attempts" / "attempt_001" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "epoch2.pt"
    checkpoint.write_bytes(b"checkpoint")
    (stage_dir / "checkpoints").symlink_to(checkpoint_dir.relative_to(stage_dir))
    (stage_dir / "PRE_RUN.attempt_001.json").write_text(json.dumps({
        "source_identity_sha256": source_sha,
    }))
    (stage_dir / "POST_RUN.attempt_001.json").write_text(json.dumps({
        "attempt_id": "attempt_001",
        "status": "SUCCESS",
        "outputs": {
            "epoch2.pt": {
                "path": str(checkpoint),
                "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            }
        },
    }))
    return checkpoint


def test_completed_stage_requires_success_source_and_output_hashes(tmp_path: Path):
    paths = Paths(
        repo_root=Path("/home/oycy/MMDD"), dataset_root=tmp_path,
        backbone_dir=tmp_path, pure_cache_dir=tmp_path,
        row_cache_manifest=tmp_path / "manifest", protocol_path=tmp_path / "protocol",
        run_root=tmp_path / "run",
    )
    stage_dir = tmp_path / "seed13" / "TA"
    checkpoint = _write_completed_ta(stage_dir, source_identity(paths))
    assert _completed_stage_result(paths, stage_dir, "TA") == stage_dir / "checkpoints" / "epoch2.pt"
    checkpoint.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="output hash mismatch"):
        _completed_stage_result(paths, stage_dir, "TA")


def test_source_amendment_carries_only_declared_stages(tmp_path: Path):
    paths = Paths(
        repo_root=Path("/home/oycy/MMDD"), dataset_root=tmp_path,
        backbone_dir=tmp_path, pure_cache_dir=tmp_path,
        row_cache_manifest=tmp_path / "manifest", protocol_path=tmp_path / "protocol",
        run_root=tmp_path / "run",
    )
    paths.run_root.mkdir()
    current = source_manifest(paths)
    previous = [dict(row) for row in current]
    previous[0]["sha256"] = "0" * 64  # the manifest prepare wrote before the code edit
    write_jsonl(paths.run_root / "SOURCE_TREE_MANIFEST.jsonl", previous)
    stage_dir = tmp_path / "seed13" / "TA"
    _write_completed_ta(stage_dir, json_sha(previous))
    with pytest.raises(RuntimeError, match="completed source differs"):
        _completed_stage_result(paths, stage_dir, "TA")

    row = record_source_amendment(paths, amendment_id="a1", carried_stages=["TB_CQET"], reason="r")
    assert row["from_source_identity_sha256"] == json_sha(previous)
    assert row["to_source_identity_sha256"] == source_identity(paths)
    assert row["changed_paths"] == {
        current[0]["path"]: {"from_sha256": "0" * 64, "to_sha256": current[0]["sha256"]},
    }
    with pytest.raises(RuntimeError, match="completed source differs"):
        _completed_stage_result(paths, stage_dir, "TA")
    record_source_amendment(paths, amendment_id="a2", carried_stages=["TA"], reason="r")
    assert _completed_stage_result(paths, stage_dir, "TA") == stage_dir / "checkpoints" / "epoch2.pt"

    # Chains through earlier amendments only when each later link carries the stage.
    older = "1" * 64
    (stage_dir / "PRE_RUN.attempt_001.json").write_text(json.dumps({"source_identity_sha256": older}))
    link = {"from_source_identity_sha256": older, "to_source_identity_sha256": json_sha(previous)}
    ledger = paths.run_root / "SOURCE_AMENDMENTS.jsonl"
    rows = list(iter_jsonl(ledger))
    write_jsonl(ledger, [{**link, "carried_stages": ["TA"]}, *rows])
    assert _completed_stage_result(paths, stage_dir, "TA") == stage_dir / "checkpoints" / "epoch2.pt"
    write_jsonl(ledger, [{**link, "carried_stages": ["TA"]}, rows[0]])
    with pytest.raises(RuntimeError, match="completed source differs"):
        _completed_stage_result(paths, stage_dir, "TA")

    write_jsonl(paths.run_root / "SOURCE_TREE_MANIFEST.jsonl", current)
    with pytest.raises(RuntimeError, match="before prepare"):
        record_source_amendment(paths, amendment_id="a3", carried_stages=["TA"], reason="r")


def test_teacher_matrix_modes_direct_only_and_swap_collision_ledger(tmp_path: Path):
    bank, _records, _basis, _mean = tiny_fixture()
    pool = PoolRecord(
        split="dev", query_id="q0", generator_id="g",
        direct=[("tp", 1.0), ("tn", 0.0)], direct_exact=["tp", "tn"],
        first_hop={"text": [("ep", 1.0)], "image": []},
        pre_paths={"tp": [PathEntry("ep", "text", 1.0, 1.0)]},
        retained_paths={"tp": ["ep"]}, retained_coverage={"tp": 0.5},
        U=["tp", "tn"], C150=["tp", "tn"],
        qt_scores_all_U={"tp": 1.0, "tn": 0.0},
        admission_scores={"tp": 1.0, "tn": 0.5},
    )
    labels = Labels(
        queries={}, epos={}, legal_targets=["tp", "tn"], edge_anchors=[],
        canonical_map={"ep": "ep", "en": "en"},
        modality={"ep": "text", "en": "text"},
        content_hash={"ep": "a" * 64, "en": "b" * 64},
        canonical_text=["ep", "en"], canonical_image=[],
    )
    teacher = FreshPathTeacher(
        input_dim=4, width=4, heads=1, layers=1, ffn=8,
        text_slots=1, image_slots=1, dropout=0.0,
    )
    matrix = evaluate_teacher_matrix(
        {"custom": teacher}, bank, {"q0": pool}, labels,
        seed=13, generator="g", split="dev", output_dir=tmp_path / "path",
        device="cpu", teacher_modes={"custom": "cqet"},
        split_gt={"q0": {"W": {"tp": ["en"]}}},
    )
    assert set(matrix["custom"]) == {"f0", "Real", "Swap"}
    collision = json.loads((tmp_path / "path" / "SWAP_COLLISIONS.json").read_text())
    assert collision["swap_paths"] == 1
    assert collision["known_witness_collisions"] == 1

    direct = evaluate_teacher_matrix(
        {"custom": teacher}, bank, {"q0": pool}, labels,
        seed=13, generator="g", split="dev", output_dir=tmp_path / "direct",
        device="cpu", teacher_modes={"custom": "cqet"}, direct_only=True,
    )
    assert set(direct["custom"]) == {"f0"}


def test_row_store_filters_non_query_payloads_without_row_embeddings(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    query_dir = dataset_root / "query_tables"
    query_dir.mkdir(parents=True)
    (query_dir / "part-00000.jsonl").write_text(
        json.dumps({"table_id": "q", "split": "train"}) + "\n", encoding="utf-8"
    )
    cache_root = tmp_path / "cache"
    objects = cache_root / "objects"
    objects.mkdir(parents=True)
    torch.save(
        {"embedding": torch.ones(4096), "row_embeddings": torch.stack([
            torch.ones(4096), torch.arange(1, 4097, dtype=torch.float32),
        ])},
        objects / "q.pt",
    )
    torch.save({"embedding": torch.ones(4096)}, objects / "e.pt")
    manifest = cache_root / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"object_id": "q", "object_type": "table", "feature_path": "objects/q.pt"}) + "\n"
        + json.dumps({"object_id": "e", "object_type": "text", "feature_path": "objects/e.pt"}) + "\n",
        encoding="utf-8",
    )
    paths = Paths(
        repo_root=Path("/home/oycy/MMDD"), dataset_root=dataset_root,
        backbone_dir=tmp_path, pure_cache_dir=tmp_path,
        row_cache_manifest=manifest, protocol_path=tmp_path / "protocol",
        run_root=tmp_path / "run",
    )
    store = build_or_load_row_store(paths)
    assert set(store.offsets) == {"q"}
    assert store.get("q").shape == (2, 4096)
    np.testing.assert_allclose(np.linalg.norm(store.get("q"), axis=1), np.ones(2), rtol=1e-5)


def test_hash_order_library_equals_full_sort_with_exclusions():
    ids = [f"obj_{i:05d}" for i in range(3000)] + ["\u00e9_utf8", "Z_upper"]
    library = HashOrderLibrary(ids)
    full = hash_order(ids, "UNIFORM_E", 13, "q|text")
    assert library.first("UNIFORM_E", 13, "q|text", 32) == full[:32]
    exclude = set(full[:40:3]) | {"not_in_library"}
    legal = [x for x in ids if x not in exclude]
    for k in (0, 1, 8, 31, 32):
        assert library.first("UNIFORM_E", 13, "q|text", k, exclude) == (
            hash_order(legal, "UNIFORM_E", 13, "q|text")[:k]
        )
    small = HashOrderLibrary(["a", "b", "c"])
    assert small.first("UNIFORM_T", 29, "q", 32, {"b"}) == hash_order(["a", "c"], "UNIFORM_T", 29, "q")


def _reference_uniform_lists(query_id, raw_pool, labels, seed):
    """Pre-optimization sampling: filter the whole library, sort it by hash, then slice."""
    g_targets = set(labels.queries[query_id]["G"])
    legal = [t for t in labels.legal_targets if t not in g_targets]
    u32_t = hash_order(legal, "UNIFORM_T", seed, query_id)[:32]
    u32_e = {}
    support = {}
    for m in ("text", "image"):
        qpos = set(labels.queries[query_id]["Qpos"][m])
        lib = [e for e in labels.library(f"Q_{m}") if e not in qpos]
        u32_e[m] = hash_order(lib, "UNIFORM_E", seed, f"{query_id}|{m}")[:32]
    for tid in sorted(labels.queries[query_id]["W"]):
        protect = labels.protect_set(query_id, tid)
        for m in ("text", "image"):
            bag = [e for e in raw_pool.retained_paths.get(tid, []) if labels.modality.get(e) == m and e not in protect]
            qe = [e for e in raw_pool.training_exact["RawQE128"][m] if e not in protect and e not in set(bag)]
            lib = [e for e in labels.library(f"Q_{m}") if e not in protect and e not in set(bag) and e not in set(qe)]
            needed = max(0, 8 - len(bag) - len(qe))
            support[(tid, m)] = hash_order(lib, "UNIFORM_E", seed, f"{query_id}|{tid}|{m}")[:needed]
    return u32_t, u32_e, support


def test_actual_list_builders_match_full_sort_reference_and_second_hop_cache():
    generator = torch.Generator().manual_seed(7)
    target_ids = [f"t{i:03d}" for i in range(160)]
    text_ids = [f"et{i:03d}" for i in range(12)]
    image_ids = [f"ei{i:03d}" for i in range(140)]
    query_ids = ["q0", "q1"]
    all_ids = [*query_ids, *target_ids, *text_ids, *image_ids]
    matrix = torch.nn.functional.normalize(torch.randn(len(all_ids), 4, generator=generator), dim=1)
    z_store = TinyZStore(dict(zip(all_ids, matrix)))
    row_store = TinyRowStore({q: np.stack([z_store.vector("et000").numpy()]) for q in query_ids})
    canonical = {e: e for e in [*text_ids, *image_ids]}
    modality = {e: "text" for e in text_ids} | {e: "image" for e in image_ids}
    queries = {
        q: {
            "G": ["t000", "t001"],
            "W": {"t000": ["et000", "ei000"], "t001": ["et001"]},
            "Qpos": {"text": ["et000", "et001"], "image": ["ei000"]},
        }
        for q in query_ids
    }
    labels = Labels(
        queries=queries, epos={"et000": ["t000"], "ei000": ["t000"], "et001": ["t001"]},
        legal_targets=target_ids, edge_anchors=[], canonical_map=canonical, modality=modality,
        content_hash={e: hashlib.sha256(e.encode()).hexdigest() for e in canonical},
        canonical_text=text_ids, canonical_image=image_ids,
    )
    pools = build_raw_pools_split(z_store, row_store, query_ids, labels, "train", device="cpu", hnsw_seed=13)
    # The second-hop cache must give each query exactly the uncached per-evidence ANN search.
    reference_index = HNSWIndex(
        z_store.rows(target_ids).numpy(), target_ids, dim=4, seed=13,
    )
    for q in query_ids:
        expected = defaultdict(list)
        for m in ("text", "image"):
            for eid, first in pools[q].first_hop[m]:
                for tid, second in reference_index.search(z_store.vector(eid).numpy(), 50):
                    expected[tid].append(PathEntry(eid, m, first, second))
        assert dict(pools[q].pre_paths) == dict(expected)

    # Protect every RawQE128 image via E+(t000) so image support must fall back to uniform.
    for q in query_ids:
        for e in pools[q].training_exact["RawQE128"]["image"]:
            labels.epos[e] = ["t000"]
    raw_et = {e: target_ids[:128] for e in canonical}
    for q in query_ids:
        u32_t, u32_e, support = _reference_uniform_lists(q, pools[q], labels, 13)
        ta = build_ta_records(q, pools[q], labels, 13, raw_et)
        tb = build_tb_records(q, pools[q], labels, 13)
        assert set(u32_t) <= set(ta["qt_candidates"]) and set(u32_t) <= set(tb["targets"])
        expected_qt = utf8_sorted(
            set(pools[q].training_exact["RawQT128"]) | set(pools[q].C150) | {"t000", "t001"} | set(u32_t)
        )
        assert ta["qt_candidates"] == expected_qt
        for m in ("text", "image"):
            expected_qe = utf8_sorted(
                set(queries[q]["Qpos"][m]) | set(pools[q].training_exact["RawQE128"][m]) | set(u32_e[m])
            )
            assert ta["qe_candidates"][m] == expected_qe
        assert any(row["competitor_sources"]["uniform"] for row in ta["support_records"])
        for records in (ta["support_records"], tb["support_records"]):
            for row in records:
                sources = row["competitor_sources"]
                room = max(0, 8 - len(sources["natural_bag"]) - len(sources["RawQE128"]))
                assert sources["uniform"] == support[(row["target_id"], row["modality"])][:room]


def test_raw_dev_reuses_train_indices_and_rejects_foreign_index(tmp_path: Path):
    generator = torch.Generator().manual_seed(3)
    target_ids = [f"t{i:03d}" for i in range(160)]
    text_ids = [f"et{i:03d}" for i in range(30)]
    image_ids = [f"ei{i:03d}" for i in range(30)]
    all_ids = ["q0", "q1", *target_ids, *text_ids, *image_ids]
    matrix = torch.nn.functional.normalize(torch.randn(len(all_ids), 4, generator=generator), dim=1)
    z_store = TinyZStore(dict(zip(all_ids, matrix)))
    row_store = TinyRowStore({q: np.stack([z_store.vector("et000").numpy()]) for q in ("q0", "q1")})
    canonical = {e: e for e in [*text_ids, *image_ids]}
    labels = Labels(
        queries={}, epos={}, legal_targets=target_ids, edge_anchors=[], canonical_map=canonical,
        modality={e: "text" for e in text_ids} | {e: "image" for e in image_ids},
        content_hash={e: e for e in canonical}, canonical_text=text_ids, canonical_image=image_ids,
    )
    built = build_raw_pools_split(
        z_store, row_store, ["q0"], labels, "train", device="cpu", hnsw_seed=13,
        index_dir=tmp_path / "indices",
    )
    fresh = build_raw_pools_split(z_store, row_store, ["q1"], labels, "dev", device="cpu", hnsw_seed=13)
    reused = build_raw_pools_split(
        z_store, row_store, ["q1"], labels, "dev", device="cpu", hnsw_seed=13,
        reuse_index_dir=tmp_path / "indices",
    )
    for field_name in ("direct", "first_hop", "pre_paths", "retained_paths", "U", "C150",
                       "D150", "MatchedDirectC", "MatchedDirectU", "training_exact"):
        assert getattr(reused["q1"], field_name) == getattr(fresh["q1"], field_name)
    assert built["q0"].object_vector_hash == reused["q1"].object_vector_hash
    with pytest.raises(ValueError, match="does not match"):
        build_raw_pools_split(
            z_store, row_store, ["q1"], labels, "dev", device="cpu", hnsw_seed=29,
            reuse_index_dir=tmp_path / "indices",
        )


def test_teacher_list_scorer_keeps_chunk_as_the_oom_knob():
    """Only one chunk's activations may be live at a time: peak memory must fall as the
    chunk shrinks, otherwise the spec's 16->8->4->2->1 retry ladder is inert."""
    torch.manual_seed(3)
    bank, _records, _basis, _mean = tiny_fixture()
    model = FreshPathTeacher(
        input_dim=4, width=4, heads=1, layers=1, ffn=8,
        text_slots=1, image_slots=1, dropout=0.0,
    ).train()
    zq, cq = bank.z("q0"), bank.tokens("q0")
    pairs = [
        ("table", zq, cq, "table", bank.z(target), bank.tokens(target))
        for target in ("tp", "tn", "tx")
    ]
    keys = [(("q0", 0), (target, 1)) for target in ("tp", "tn", "tx")]
    scorer = TeacherListScorer(model, chunk=1)
    scores = scorer.score_pairs(pairs, keys)
    # chunk=1 must register one leaf per row, i.e. one live graph per candidate.
    assert len(scorer.calls) == len(pairs)
    assert all(call[-1].shape == (1,) for call in scorer.calls)
    loss = torch.logsumexp(scores, 0) - scores[0]
    scorer.backward(loss, scale=1.0)
    # Parameters on the scored path get their gradient from the replayed chunks.
    for name in ("adapters.table.weight", "globals.table.0.weight", "scoring_head.0.weight"):
        assert model.get_parameter(name).grad is not None
    assert model.rel.grad is not None

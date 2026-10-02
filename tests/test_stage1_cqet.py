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

from mmdd_stage1.evaluate import evaluate_teacher_matrix, paired_bootstrap
from mmdd_stage1.metrics import rank_metrics
from mmdd_stage1.artifacts import load_pool_bundle, save_pool_bundle
from mmdd_stage1.config import Paths
from mmdd_stage1.data import iter_jsonl, utf8_sorted, write_jsonl
from mmdd_stage1.data import build_content_aliases
from mmdd_stage1.labels import Labels
from mmdd_stage1.features import build_or_load_row_store
from mmdd_stage1.lists import (
    build_c1_edge_lists,
    build_c2_shared_graph,
    build_raw_et128_exact,
    build_raw_pools_split,
    build_ta_records,
    build_tb_records,
    hash_order,
    HashOrderLibrary,
)
from mmdd_stage1.losses import (
    aggregate_corrected_lse,
    aggregate_cqet,
    hierarchical_relation_mean,
    hierarchical_support_mean,
)
from mmdd_stage1.models import FreshPathTeacher, NativeStudent
from mmdd_stage1.retrieval import HNSWIndex, PathEntry, PoolRecord, d1_retain, p3_admission
from mmdd_stage1.provenance import assert_declared_project_imports
from mmdd_stage1.pipeline import _completed_stage_result
from mmdd_stage1.probes import (
    student_gradient_probe,
    teacher_content_probe,
    teacher_gradient_probe,
)
from mmdd_stage1.data import json_identity
from mmdd_stage1.provenance import (
    record_source_amendment,
    source_identity,
    source_manifest,
)
from mmdd_stage1.train import (
    StudentRecipe,
    TeacherListScorer,
    _scored_list_sha,
    build_teacher_logits_cache,
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


def teacher_cache(records, logits) -> dict:
    """Hand-written Teacher logits in the ``build_teacher_logits_cache`` layout, aligned with the records."""
    rows = {row["query_id"]: row for row in records}
    return {
        q: {"direct": direct, "evidence": evidence, "list_sha256": _scored_list_sha(rows[q])}
        for q, (direct, evidence) in logits.items()
    }


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

    logits = build_teacher_logits_cache(teacher, bank, records, device="cpu")
    assert logits["q0"]["direct"].shape == (3,) and logits["q0"]["evidence"].shape == (2,)
    train_student_c2(
        student, records, bank, device="cpu", arm="NATIVE_KD",
        logical_batch=1, save_dir=out, seed=13,
        expected_parent_hash=parent_hash, max_updates=1, teacher_logits=logits,
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
        clean, edge_lists, bank, device="cpu", logical_batch=2,
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
        retried, edge_lists, bank, device="cpu", logical_batch=2,
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
        teacher_cache(records, {"q0": (torch.tensor([3.0, -1.0, -2.0]), torch.tensor([2.0, -1.0]))}),
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
    teacher_logits = teacher_cache(records, {"q0": (torch.tensor([4.0, -2.0, -3.0]), torch.tensor([-2.0, 3.0]))})

    train_student_c2(
        sup, records, bank, device="cpu", arm="NATIVE_SUP", logical_batch=1,
        seed=29, expected_parent_hash=parent_hash, max_updates=1,
    )
    train_student_c2(
        kd, records, bank, device="cpu", arm="NATIVE_KD", logical_batch=1,
        seed=29, expected_parent_hash=parent_hash, teacher_logits=teacher_logits, max_updates=1,
    )
    assert model_state_sha(sup) != model_state_sha(kd)


def test_actual_c2_resume_matches_continuous(tmp_path: Path):
    bank, records, basis, mean = tiny_fixture(n_queries=4)
    logits = teacher_cache(records, {
        f"q{i}": (torch.tensor([2.0 + i, -1.0, -2.0]), torch.tensor([1.0, -1.0]))
        for i in range(4)
    })
    base = NativeStudent(basis, mean, dim=2)
    parent_hash = model_state_sha(base)

    torch.manual_seed(71)
    np.random.seed(71)
    continuous = copy.deepcopy(base)
    train_student_c2(
        continuous, records, bank, device="cpu", arm="NATIVE_KD",
        logical_batch=1, seed=13, expected_parent_hash=parent_hash,
        teacher_logits=logits,
    )

    torch.manual_seed(71)
    np.random.seed(71)
    partial = copy.deepcopy(base)
    part_dir = tmp_path / "partial"
    train_student_c2(
        partial, records, bank, device="cpu", arm="NATIVE_KD",
        logical_batch=1, seed=13, expected_parent_hash=parent_hash,
        teacher_logits=logits, save_dir=part_dir, max_updates=2,
    )
    resumed = copy.deepcopy(base)
    train_student_c2(
        resumed, records, bank, device="cpu", arm="NATIVE_KD",
        logical_batch=1, seed=13, expected_parent_hash=parent_hash,
        teacher_logits=logits, save_dir=tmp_path / "resumed",
        resume_from=part_dir / "snapshot_frac050.pt",
    )
    assert model_state_sha(resumed) == model_state_sha(continuous)


def test_student_multiple_epochs_preserve_batches_and_optimizer_state(tmp_path: Path):
    bank, records, basis, mean = tiny_fixture(n_queries=4)
    base = NativeStudent(basis, mean, dim=2)
    parent_hash = model_state_sha(base)
    common = dict(device="cpu", arm="NATIVE_SUP", logical_batch=3, epochs=3,
                  expected_parent_hash=parent_hash, seed=13)
    continuous = copy.deepcopy(base)
    points = train_student_c2(continuous, records, bank, **common,
        save_dir=tmp_path / "continuous", log_path=tmp_path / "steps.jsonl")
    rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [1, 1, 2, 2, 3, 3]
    assert [r["batch_queries"] for r in rows] == [3, 1, 3, 1, 3, 1]
    for epoch in (1, 2, 3):
        assert sorted(q for r in rows if r["epoch"] == epoch for q in r["record_ids"]) == ["q0", "q1", "q2", "q3"]
    endpoint = torch.load(points[1.0], map_location="cpu", weights_only=False)
    assert all(int(state["step"]) == 6 for state in endpoint["optimizer"]["state"].values())
    partial = copy.deepcopy(base)
    train_student_c2(partial, records, bank, **common,
                     save_dir=tmp_path / "partial", max_updates=2)
    resumed = copy.deepcopy(base)
    train_student_c2(resumed, records, bank, **common,
                     resume_from=tmp_path / "partial/snapshot_epoch001.pt")
    assert model_state_sha(resumed) == model_state_sha(continuous)
    edges = [{"item_id": r["query_id"], "relation": "QT", "anchor_id": r["query_id"],
              "candidates": r["targets"], "positives": r["positives"]} for r in records]
    train_student_c1(copy.deepcopy(base), edges, bank, device="cpu", epochs=3,
                     logical_batch=3, log_path=tmp_path / "c1_steps.jsonl")
    c1_rows = [json.loads(line) for line in (tmp_path / "c1_steps.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in c1_rows] == [1, 1, 2, 2, 3, 3]


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


def _per_object_cache(model: FreshPathTeacher, rows, keys) -> dict:
    """Reference encoding: every object through ``encode_one`` and ``tag`` individually."""
    cache = {}
    for row, key in zip(rows, keys):
        for slot, role in enumerate((0, 1) if len(row) == 6 else (0, 2, 1)):
            ref = (row[3 * slot], key[slot])
            if ref not in cache:
                tokens, g = model.encode_one(row[3 * slot], row[3 * slot + 1], row[3 * slot + 2])
                cache[ref] = (model.tag(row[3 * slot], role, tokens, g), g)
    return cache


def _refs(rows, keys):
    return [tuple((row[3 * slot], key[slot]) for slot in range(len(row) // 3)) for row, key in zip(rows, keys)]


def test_teacher_list_scorer_matches_per_chunk_scoring_with_dropout():
    """The shared-cache, batched-encode scorer must reproduce per-object encoding with one
    graph per chunk at the same chunk size: same dropout draws (same batch shapes, same
    order), same complete-list denominator, same gradients."""
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
    keys = [(("q0", 0), (target, 1)) for target in ("tp", "tn", "tx")]
    chunk = 2
    rng = torch.get_rng_state()

    # Reference: per-object encode, fresh cache, one graph per chunk.
    legacy_scores = torch.cat([
        legacy.score_pairs(_per_object_cache(legacy, pairs[start : start + chunk], keys[start : start + chunk]),
                           _refs(pairs[start : start + chunk], keys[start : start + chunk]))
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
    keys = [(("q0", 0), (target, 1)) for target in ("tp", "tn", "tx")]

    torch.manual_seed(4321)
    before = torch.get_rng_state().clone()
    TeacherListScorer(model, chunk=2).score_pairs(pairs, keys)
    chunked_stream = torch.get_rng_state().clone()

    torch.set_rng_state(before)
    torch.cat([
        model.score_pairs(_per_object_cache(model, pairs[start : start + 2], keys[start : start + 2]),
                          _refs(pairs[start : start + 2], keys[start : start + 2]))
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
    assert rank_metrics(ranked, gold)["target_coverage"] == 0.25
    assert rank_metrics(ranked, gold)["query_hit_rate"] == 1.0
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
    repo = Path(__file__).resolve().parents[1]
    paths = Paths(
        repo_root=repo, dataset_root=tmp_path,
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
    module.__file__ = str(repo / "undeclared_mmdd_test_module.py")
    sys.modules[module.__name__] = module
    try:
        with pytest.raises(RuntimeError, match="undeclared project-local"):
            assert_declared_project_imports(paths)
    finally:
        sys.modules.pop(module.__name__, None)

    placeholder = types.ModuleType("relative_extension_namespace")
    placeholder.__file__ = "_classes.py"
    # Only declared Stage-1 modules plus a relative-path placeholder: other tests in the
    # same process may already have imported unrelated project modules.
    import mmdd_stage1.pipeline
    monkeypatch.setattr(sys, "modules", {
        "mmdd_stage1.pipeline": mmdd_stage1.pipeline, placeholder.__name__: placeholder,
    })
    monkeypatch.chdir(paths.repo_root)
    assert_declared_project_imports(paths)


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
        repo_root=Path(__file__).resolve().parents[1], dataset_root=tmp_path,
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
        repo_root=Path(__file__).resolve().parents[1], dataset_root=tmp_path,
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
    _write_completed_ta(stage_dir, json_identity(previous))
    with pytest.raises(RuntimeError, match="completed source differs"):
        _completed_stage_result(paths, stage_dir, "TA")

    row = record_source_amendment(paths, amendment_id="a1", carried_stages=["TB_CQET"], reason="r")
    assert row["from_source_identity_sha256"] == json_identity(previous)
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
    link = {"from_source_identity_sha256": older, "to_source_identity_sha256": json_identity(previous)}
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
        repo_root=Path(__file__).resolve().parents[1], dataset_root=dataset_root,
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


# ---------------------------------------------------------------------------
# V4.2 student recipe: scaled logits, tempered KD, random negatives, protocol wiring
# ---------------------------------------------------------------------------

def _mixed_evidence_fixture():
    """Two targets with text and image paths, a third with no bag; random P/R so the
    bilinear scores are not trivially symmetric."""
    torch.manual_seed(7)
    vectors = {
        "q0": torch.randn(4), "tp": torch.randn(4), "tn": torch.randn(4), "tx": torch.randn(4),
        "ea": torch.randn(4), "eb": torch.randn(4), "ec": torch.randn(4),
    }
    kinds = {"q0": "table", "tp": "table", "tn": "table", "tx": "table",
             "ea": "text", "eb": "image", "ec": "text"}
    bank = TinyBank(vectors, kinds)
    row = {"query_id": "q0", "targets": ["tp", "tn", "tx"], "positives": ["tp"],
           "natural_bags": {"tp": ["ea", "eb"], "tn": ["ec", "eb"], "tx": []}}
    student = NativeStudent(torch.randn(2, 4), torch.zeros(4), dim=2)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(0.3 * torch.randn_like(parameter))
    return bank, row, student


def _reference_c2_scores(student, bank, row, scale):
    """The original per-path Python loop, kept as the oracle for the gathered version."""
    from mmdd_stage1.losses import aggregate_cqet as agg

    targets = row["targets"]
    uq = student.u("table", bank.z(row["query_id"]))
    ut = student.u("table", bank.z_many(targets))
    direct = scale * (uq @ student.R["QT"] * ut).sum(-1)
    flat = [(i, e) for i, t in enumerate(targets) for e in row["natural_bags"].get(t, [])]
    scores = []
    for target_i, e in flat:
        modality = bank.kind(e)
        ue = student.u(modality, bank.z(e))
        qe = (uq @ student.R[f"Q_{modality}"] * ue).sum()
        et = ((ue @ student.R[f"{modality}_T"]) * ut[target_i]).sum()
        scores.append(scale * (qe + et))
    aggregate = agg(direct, torch.stack(scores), torch.tensor([i for i, _ in flat]))
    bag_targets = [i for i, t in enumerate(targets) if row["natural_bags"].get(t)]
    return direct, aggregate[bag_targets]


def test_student_c2_scores_gathered_paths_match_per_path_loop():
    from mmdd_stage1.train import _student_c2_scores

    bank, row, student = _mixed_evidence_fixture()
    direct, evidence, bag_targets = _student_c2_scores(student, bank, row, logit_scale=20.0)
    ref_direct, ref_evidence = _reference_c2_scores(student, bank, row, 20.0)
    assert torch.allclose(direct, ref_direct, atol=1e-6)
    assert torch.allclose(evidence, ref_evidence, atol=1e-6)
    assert [t for _, t in bag_targets] == ["tp", "tn"]
    # scale enters every bilinear term, so halving it halves the direct logits exactly
    half, _, _ = _student_c2_scores(student, bank, row, logit_scale=10.0)
    assert torch.allclose(half, direct / 2)


def test_random_negatives_are_deterministic_and_exclude_pool_and_positives():
    from mmdd_stage1.train import _random_negative_ids

    pool = [f"t{i}" for i in range(50)]
    first = _random_negative_ids(pool, {"t3", "t7"}, 8, "C2_SHARED", 13, 1, "q0")
    again = _random_negative_ids(pool, {"t3", "t7"}, 8, "C2_SHARED", 13, 1, "q0")
    assert first == again and len(first) == 8 and len(set(first)) == 8
    assert not {"t3", "t7"} & set(first)
    assert first != _random_negative_ids(pool, {"t3", "t7"}, 8, "C2_SHARED", 13, 2, "q0")
    assert first != _random_negative_ids(pool, {"t3", "t7"}, 8, "C2_SHARED", 13, 1, "q1")
    # a pool that is fully excluded yields nothing instead of looping forever
    assert _random_negative_ids(["a", "b"], {"a", "b"}, 4, "C2_SHARED", 13, 1, "q0") == []


def test_c2_random_negatives_enter_the_sup_loss_and_are_logged(tmp_path: Path):
    bank, records, basis, mean = tiny_fixture()
    bank.vectors["far"] = torch.tensor([-0.9, -0.1, -0.3, 0.2])
    bank.kinds["far"] = "table"
    base = NativeStudent(basis, mean, dim=2)
    parent_hash = model_state_sha(base)
    common = dict(device="cpu", arm="NATIVE_SUP", logical_batch=1, seed=13,
                  expected_parent_hash=parent_hash, max_updates=1)
    with_negs = copy.deepcopy(base)
    train_student_c2(with_negs, records, bank, recipe=StudentRecipe(random_negatives=4),
                     negative_pool=["tp", "tn", "tx", "far"], log_path=tmp_path / "negs.jsonl", **common)
    without = copy.deepcopy(base)
    train_student_c2(without, records, bank, recipe=StudentRecipe(random_negatives=0), **common)
    assert model_state_sha(with_negs) != model_state_sha(without)
    row = json.loads((tmp_path / "negs.jsonl").read_text().splitlines()[0])
    assert row["random_negatives"] == 4 and row["negative_pool_size"] == 4
    assert row["logit_scale"] == 20.0 and row["P_lr"] == 1e-4 and row["R_lr"] == 1e-3
    assert row["sigma1_R_QT_minus_I"] >= row["sigma2_R_QT_minus_I"] >= 0.0


def test_kd_temperature_divides_teacher_logits():
    bank, records, basis, mean = tiny_fixture()
    base = NativeStudent(basis, mean, dim=2)
    parent_hash = model_state_sha(base)
    direct, evidence = torch.tensor([4.0, -2.0, -3.0]), torch.tensor([-2.0, 3.0])
    hot = teacher_cache(records, {"q0": (direct, evidence)})
    cooled = teacher_cache(records, {"q0": (direct / 5.0, evidence / 5.0)})
    common = dict(device="cpu", arm="NATIVE_KD", logical_batch=1, seed=29,
                  expected_parent_hash=parent_hash, max_updates=1)
    tempered = copy.deepcopy(base)
    train_student_c2(tempered, records, bank, teacher_logits=hot,
                     recipe=StudentRecipe(kd_temperature=5.0, random_negatives=0), **common)
    pre_divided = copy.deepcopy(base)
    train_student_c2(pre_divided, records, bank, teacher_logits=cooled,
                     recipe=StudentRecipe(kd_temperature=1.0, random_negatives=0), **common)
    assert model_state_sha(tempered) == model_state_sha(pre_divided)
    untempered = copy.deepcopy(base)
    train_student_c2(untempered, records, bank, teacher_logits=hot,
                     recipe=StudentRecipe(kd_temperature=1.0, random_negatives=0), **common)
    assert model_state_sha(tempered) != model_state_sha(untempered)


def test_c1_uses_logit_scale_and_optional_anchor(tmp_path: Path):
    bank, _records, basis, mean = tiny_fixture()
    edges = [{"item_id": "e0", "relation": "QT", "anchor_id": "q0",
              "candidates": ["tp", "tn", "tx"], "positives": ["tp"]}]
    base = NativeStudent(basis, mean, dim=2)
    scaled = copy.deepcopy(base)
    train_student_c1(scaled, edges, bank, device="cpu", logical_batch=1, recipe=StudentRecipe(logit_scale=20.0),
                     log_path=tmp_path / "c1.jsonl")
    flat = copy.deepcopy(base)
    train_student_c1(flat, edges, bank, device="cpu", logical_batch=1, recipe=StudentRecipe(logit_scale=1.0))
    assert model_state_sha(scaled) != model_state_sha(flat)
    row = json.loads((tmp_path / "c1.jsonl").read_text().splitlines()[0])
    assert row["logit_scale"] == 20.0 and row["anchor_weight"] == 0.0
    assert "sigma1_R_QT_minus_I" in row


def test_protocol_template_validates_and_binds_paths_and_gpu(tmp_path: Path):
    from mmdd_stage1 import EXPERIMENT_ID, VERSION
    from mmdd_stage1.config import load_protocol, resolve_default_paths, validate_protocol
    from mmdd_stage1.train import StudentRecipe

    repo = Path(__file__).resolve().parents[1]
    template = repo / "configs" / "mmdd_stage1_cqet_protocol.json"
    protocol = json.loads(template.read_text())
    assert protocol["experiment_id"] == EXPERIMENT_ID and protocol["version"] == VERSION
    validate_protocol(protocol)
    recipe = StudentRecipe.from_protocol(protocol)
    assert (recipe.lr_p, recipe.lr_r, recipe.logit_scale) == (1e-4, 1e-3, 20.0)
    assert (recipe.kd_weight, recipe.kd_temperature, recipe.random_negatives) == (1.0, 10.0, 256)
    assert recipe.anchor_weight == 0.0
    assert (recipe.lr_schedule, recipe.kd_normalization, recipe.kd_top_k) == ("cosine", "temperature", 50)
    assert recipe.teacher_scored_negatives is True and recipe.evidence_random_negatives == 256

    protocol["hardware"]["uuid"] = "GPU-deadbeef"
    protocol["paths"]["run_root"] = str(tmp_path / "run")
    protocol["paths"]["dataset_root"] = "relative/dataset"
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    paths = resolve_default_paths(protocol_path, tmp_path / "run")
    assert paths.gpu_uuid == "GPU-deadbeef" and paths.gpu_physical_index == 0
    assert paths.dataset_root == repo / "relative" / "dataset"
    assert paths.package_dir is None  # preflight then defaults to <run_root>/protocol_package
    with pytest.raises(ValueError, match="run root is fixed by protocol"):
        resolve_default_paths(protocol_path, tmp_path / "elsewhere")

    for broken, match in (
        ({"paths": {k: v for k, v in protocol["paths"].items() if k != "dataset_root"}}, "paths block is missing"),
        ({"hardware": {**protocol["hardware"], "uuid": "0"}}, "full GPU UUID"),
        ({"student": {**protocol["student"], "temperature": 0}}, "must be positive"),
        ({"student": {k: v for k, v in protocol["student"].items() if k != "random_negatives"}}, "random_negatives"),
        ({"student": {**protocol["student"], "lr_schedule": "step"}}, "lr_schedule"),
        ({"student": {**protocol["student"], "evidence_random_negatives": 300}}, "cannot exceed"),
        ({"version": "4.1.0"}, "Unexpected version"),
    ):
        with pytest.raises(ValueError, match=match):
            validate_protocol({**protocol, **broken})
    assert load_protocol(protocol_path)["student"]["logit_scale"] == 20.0


def test_list_kl_divergence_takes_final_logits_without_hidden_temperature():
    import inspect

    import torch.nn.functional as F
    from mmdd_stage1.losses import list_kl_divergence

    assert "temperature" not in inspect.signature(list_kl_divergence).parameters
    student = torch.tensor([0.4, -1.2, 2.0], requires_grad=True)
    teacher = torch.tensor([3.0, -2.0, 1.0])
    p_t = F.softmax(teacher, 0)
    expected = (p_t * (F.log_softmax(teacher, 0) - F.log_softmax(student, 0))).sum()
    assert torch.allclose(list_kl_divergence(student, teacher), expected)


def test_run_stage1_init_binds_run_layout_and_gpu_into_valid_protocol(tmp_path: Path):
    import run_stage1
    from mmdd_stage1.config import resolve_default_paths, validate_protocol

    template = json.loads(run_stage1.TEMPLATE.read_text())
    template["seeds"] = [13, 17]
    run = tmp_path / "run"
    features = tmp_path / "shared_features"  # outside the run: several runs may train on it
    protocol = run_stage1.build_protocol(
        template, dataset_root=tmp_path / "dataset", run_root=run, features_dir=features,
        backbone_dir=tmp_path / "qwen",
        hardware={"physical_index": 1, "uuid": "GPU-abc", "model": "RTX 4090"},
    )
    validate_protocol(protocol)
    assert protocol["max_registered_stages"] == 16
    assert template["hardware"]["physical_index"] == 0  # template itself is not mutated
    run.mkdir()
    (run / "protocol.json").write_text(json.dumps(protocol))
    paths = resolve_default_paths(run / "protocol.json", run)
    assert paths.pure_cache_dir == features / "features"
    assert paths.row_cache_manifest == features / "encoder" / "manifest.jsonl"
    assert paths.upstream_data_dir == features / "data"
    assert run_stage1.features_root(protocol) == features
    assert not any(str(value).startswith(str(run) + "/") for key, value in protocol["paths"].items()
                   if key != "run_root")
    assert (paths.gpu_uuid, paths.gpu_physical_index) == ("GPU-abc", 1)
    # protocols written before features_dir existed still resolve to the data directory's parent
    legacy = {"paths": {k: v for k, v in protocol["paths"].items() if k != "features_dir"}}
    assert run_stage1.features_root(legacy) == features
    # build-data and encode refuse to overwrite a shared feature directory
    (features / "data").mkdir(parents=True)
    (features / "data" / "stage1_objects.jsonl").write_text("")
    with pytest.raises(FileExistsError, match="shared across runs"):
        run_stage1.build_data(protocol)
    (features / "features" / "z").mkdir(parents=True)
    (features / "features" / "z" / "z.f32.npy").write_bytes(b"")
    with pytest.raises(FileExistsError, match="already encoded"):
        run_stage1.encode(run, protocol, [1])


def test_stage2_gate_rejects_retrieval_from_another_checkpoint(tmp_path: Path):
    from mmdd_stage1.data import sha256_file
    from mmdd_stage1.export import validate_stage2_gate

    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"selected")
    gate = {
        "format_version": 1, "completed_stage": "student-path", "selection_split": "dev",
        "stage2_allowed": True, "best_checkpoint": str(checkpoint),
        "best_checkpoint_sha256": sha256_file(checkpoint),
    }
    gate_path = tmp_path / "stage1_gate.json"
    gate_path.write_text(json.dumps(gate))
    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps({"student_checkpoint_sha256": gate["best_checkpoint_sha256"]}) + "\n")
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"student_checkpoint_sha256": "other"}) + "\n")
    assert validate_stage2_gate(gate_path, [good])["stage2_allowed"]
    with pytest.raises(ValueError, match="not produced by the dev-gated"):
        validate_stage2_gate(gate_path, [bad])
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        validate_stage2_gate(gate_path)


def test_declared_sources_cover_the_main_flow_and_exclude_removed_packages():
    from mmdd_stage1.provenance import source_files

    repo = Path(__file__).resolve().parents[1]
    paths = types.SimpleNamespace(repo_root=repo, backbone_dir=repo / "missing_backbone")
    declared = {path.relative_to(repo).as_posix() for path in source_files(paths)}
    assert {"src/run_stage1.py", "src/cache_stage1_features.py", "src/mmdd_stage1/content.py",
            "tests/test_stage1_reference_contracts.py"} <= declared
    assert all(path.is_file() for path in source_files(paths))
    assert not any("cqet_v4_1" in path or "fresh_path" in path or path.startswith("audit/") for path in declared)


def test_run_stage1_pack_merges_encoder_tiers_into_feature_store(tmp_path: Path):
    import run_stage1
    from mmdd_stage1.content import ContentStore, write_chunk
    from mmdd_stage1.features import INPUT_DIM

    run = tmp_path / "features"  # the feature directory, not a run root
    objects = [("t1", "table"), ("e1", "text"), ("i1", "image")]
    run_stage1.write_rows(run / "data" / "stage1_objects.jsonl",
                          [{"object_id": oid, "object_type": kind} for oid, kind in objects])
    encoder = run / "encoder"
    (encoder / "objects").mkdir(parents=True)
    (encoder / "teacher_objects").mkdir()
    manifest, teacher_manifest = [], []
    for position, (oid, kind) in enumerate(objects):
        embedding = torch.zeros(INPUT_DIM)
        embedding[position] = 1.0
        torch.save({"embedding": embedding}, encoder / "objects" / f"{oid}.pt")
        manifest.append({"object_id": oid, "object_type": kind, "feature_path": f"objects/{oid}.pt"})
    torch.save({"hidden_states": torch.ones(2, INPUT_DIM)}, encoder / "teacher_objects" / "t1.pt")
    teacher_manifest.append({"object_id": "t1", "object_type": "table",
                             "teacher_feature_path": "teacher_objects/t1.pt"})
    run_stage1.write_rows(encoder / "manifest.jsonl", manifest)
    run_stage1.write_rows(encoder / "teacher_manifest.jsonl", teacher_manifest)
    for shard, (oid, kind) in enumerate(objects[1:]):
        write_chunk(run / f"content_shard{shard}" / "chunks", 0,
                    [(oid, kind, np.full((3, INPUT_DIM), shard + 2, dtype=np.float16))])

    run_stage1.pack(run)
    index = json.loads((run / "features" / "z" / "z_index.json").read_text())
    z = np.load(run / "features" / "z" / "z.f32.npy")
    assert index["ids"] == ["t1", "e1", "i1"] and z[index["ids"].index("i1"), 2] == 1.0
    store = ContentStore(run / "features" / "content")
    assert tuple(store.get("t1").shape) == (2, INPUT_DIM)
    assert float(store.get("i1")[0, 0]) == 3.0

    (run / "content_shard1" / "chunks" / "chunk_000000.ids.npy").unlink()
    with pytest.raises(ValueError, match="content tokens missing"):
        run_stage1.pack(run)


# ------------------------------------------------- diagnosis plans A-D (2026-10-01) --

def _negatives_fixture():
    """``tiny_fixture`` plus out-of-pool targets and evidence for random negatives; only the gold
    target has a bag, so without evidence negatives the evidence list has no negative."""
    bank, records, basis, mean = tiny_fixture(n_queries=2)
    for i, vector in enumerate(([-0.9, -0.1, -0.3, 0.2], [0.1, 0.4, -0.8, 0.3], [0.5, -0.5, 0.1, -0.6])):
        bank.vectors[f"far{i}"] = torch.tensor(vector)
        bank.kinds[f"far{i}"] = "table"
    bank.vectors["ex"] = torch.tensor([0.3, 0.3, -0.6, 0.2])
    bank.kinds["ex"] = "text"
    for row in records:
        row["natural_bags"] = {"tp": ["ep"], "tn": [], "tx": []}
    pools = dict(negative_pool=["tp", "tn", "tx", "far0", "far1", "far2"], evidence_pool=["ep", "en", "ex"])
    return bank, records, basis, mean, pools


def test_batch_scores_match_single_record_scores():
    from mmdd_stage1.models import QTStudent
    from mmdd_stage1.train import _student_c2_batch_scores, _student_c2_scores

    bank, row, student = _mixed_evidence_fixture()
    other = {"query_id": "tp", "targets": ["tn", "q0", "tx"], "positives": ["q0"],
             "natural_bags": {"q0": ["eb"], "tn": ["ea", "ec"], "tx": []}}
    batched = _student_c2_batch_scores(student, bank, [row, other], logit_scale=20.0)
    for record, (direct, evidence, bag_targets) in zip([row, other], batched):
        single_direct, single_evidence, single_bags = _student_c2_scores(student, bank, record, 20.0)
        assert torch.allclose(direct, single_direct, atol=1e-6)
        assert torch.allclose(evidence, single_evidence, atol=1e-6)
        assert bag_targets == single_bags
    qt = QTStudent(torch.randn(2, 4), torch.zeros(4), dim=2)
    qt_direct, qt_evidence, _ = _student_c2_batch_scores(qt, bank, [row, other], 20.0)[1]
    assert qt_evidence is None
    assert torch.allclose(qt_direct, 20.0 * qt.score(bank.z("tp"), bank.z_many(["tn", "q0", "tx"])), atol=1e-6)


def test_cosine_schedule_decays_lr_and_resume_follows_it(tmp_path: Path):
    bank, records, basis, mean = tiny_fixture(n_queries=4)
    base = NativeStudent(basis, mean, dim=2)
    common = dict(device="cpu", arm="NATIVE_SUP", logical_batch=1, seed=13,
                  expected_parent_hash=model_state_sha(base), recipe=StudentRecipe(lr_schedule="cosine"))
    continuous = copy.deepcopy(base)
    train_student_c2(continuous, records, bank, log_path=tmp_path / "c2.jsonl", **common)
    rows = [json.loads(line) for line in (tmp_path / "c2.jsonl").read_text().splitlines()]
    assert [round(r["lr_factor"], 6) for r in rows] == [round(0.5 * (1 + np.cos(np.pi * k / 4)), 6) for k in range(4)]
    assert rows[0]["lr_schedule"] == "cosine"
    partial = copy.deepcopy(base)
    train_student_c2(partial, records, bank, save_dir=tmp_path / "partial", max_updates=2, **common)
    resumed = copy.deepcopy(base)
    train_student_c2(resumed, records, bank, resume_from=tmp_path / "partial/snapshot_frac050.pt", **common)
    assert model_state_sha(resumed) == model_state_sha(continuous)
    constant = copy.deepcopy(base)
    train_student_c2(constant, records, bank, **{**common, "recipe": StudentRecipe()})
    assert model_state_sha(constant) != model_state_sha(continuous)


def test_top_k_kd_is_full_kl_on_short_lists_and_ranks_the_tail_below_the_top_k():
    from mmdd_stage1.losses import list_kl_divergence, rank_mass_loss, top_k_list_kd

    student = torch.tensor([0.4, -1.2, 2.0, 0.3, -0.5], requires_grad=True)
    teacher = torch.tensor([3.0, -2.0, 1.0, 2.5, -4.0])
    assert torch.allclose(top_k_list_kd(student, teacher, 0), list_kl_divergence(student, teacher))
    assert torch.allclose(top_k_list_kd(student, teacher, 5), list_kl_divergence(student, teacher))
    top = torch.tensor([True, False, True, True, False])
    expected = list_kl_divergence(student, teacher, top) + rank_mass_loss(student, top)
    assert torch.allclose(top_k_list_kd(student, teacher, 3), expected)
    # the tail's internal order carries no KD signal
    reordered = torch.tensor([3.0, -4.0, 1.0, 2.5, -2.0])
    assert torch.allclose(top_k_list_kd(student, reordered, 3), expected)


def test_kd_zscore_normalisation_removes_the_teacher_logit_scale():
    student = torch.tensor([0.4, -1.2, 2.0, 0.3], requires_grad=True)
    teacher = torch.tensor([3.0, -2.0, 1.0, 2.5])
    zscore = StudentRecipe(kd_normalization="zscore", kd_temperature=1.0)
    assert torch.allclose(zscore.kd_loss(student, teacher), zscore.kd_loss(student, 7.0 * teacher - 3.0))
    tempered = StudentRecipe(kd_temperature=10.0)
    assert not torch.allclose(tempered.kd_loss(student, teacher), tempered.kd_loss(student, 7.0 * teacher))


def test_teacher_scored_negatives_are_fixed_scored_by_the_teacher_and_inside_kd(tmp_path: Path):
    from mmdd_stage1.train import c2_teacher_rows, c2_training_row

    torch.manual_seed(13)
    bank, records, basis, mean, pools = _negatives_fixture()
    recipe = StudentRecipe(random_negatives=2, evidence_random_negatives=1, teacher_scored_negatives=True)
    row = records[0]
    first = c2_training_row(row, recipe, pools["negative_pool"], pools["evidence_pool"], 13, epoch=1)
    assert first == c2_training_row(row, recipe, pools["negative_pool"], pools["evidence_pool"], 13, epoch=2)
    negatives = first["targets"][3:]
    assert first["targets"][:3] == row["targets"] and len(negatives) == 2
    assert set(negatives) <= {"far0", "far1", "far2"}
    assert first["natural_bags"][negatives[0]][0] in {"en", "ex"} and negatives[1] not in first["natural_bags"]
    per_epoch = StudentRecipe(random_negatives=2, evidence_random_negatives=1)
    assert any(
        c2_training_row(r, per_epoch, pools["negative_pool"], pools["evidence_pool"], 13, epoch=1)
        != c2_training_row(r, per_epoch, pools["negative_pool"], pools["evidence_pool"], 13, epoch=2)
        for r in [{**row, "query_id": f"q{i}"} for i in range(8)]
    )

    teacher = FreshPathTeacher(input_dim=4, width=4, heads=1, layers=1, ffn=8,
                               text_slots=1, image_slots=1, dropout=0.0)
    scored_rows = c2_teacher_rows(records, recipe, pools["negative_pool"], pools["evidence_pool"], 13)
    assert scored_rows[0] == first
    logits = build_teacher_logits_cache(teacher, bank, scored_rows, device="cpu")
    assert logits["q0"]["direct"].shape == (5,)  # 3 graph targets + 2 negatives
    assert logits["q0"]["evidence"].shape == (2,)  # the gold bag + one random-evidence negative
    base = NativeStudent(basis, mean, dim=2)
    common = dict(device="cpu", arm="NATIVE_KD", logical_batch=2, seed=13, max_updates=1,
                  expected_parent_hash=model_state_sha(base), **pools)
    student = copy.deepcopy(base)
    train_student_c2(student, records, bank, recipe=recipe, teacher_logits=logits,
                     log_path=tmp_path / "kd.jsonl", **common)
    step = json.loads((tmp_path / "kd.jsonl").read_text().splitlines()[0])
    assert step["direct_kd_denominator"] == 2 and step["evidence_kd_denominator"] == 2
    assert step["teacher_scored_negatives"] is True
    unextended = build_teacher_logits_cache(teacher, bank, records, device="cpu")
    with pytest.raises(ValueError, match="different C2 list"):
        train_student_c2(copy.deepcopy(base), records, bank, recipe=recipe, teacher_logits=unextended, **common)


def test_evidence_random_negatives_activate_the_evidence_list(tmp_path: Path):
    bank, records, basis, mean, pools = _negatives_fixture()
    base = NativeStudent(basis, mean, dim=2)
    common = dict(device="cpu", arm="NATIVE_SUP", logical_batch=2, seed=13, max_updates=1,
                  expected_parent_hash=model_state_sha(base), **pools)
    denominators = {}
    for count in (0, 2):
        student = copy.deepcopy(base)
        log = tmp_path / f"evidence{count}.jsonl"
        train_student_c2(student, records, bank, log_path=log,
                         recipe=StudentRecipe(random_negatives=2, evidence_random_negatives=count), **common)
        denominators[count] = json.loads(log.read_text().splitlines()[0])["evidence_sup_denominator"]
        if count:
            # Q-E and E-T relations receive gradient only through the evidence list
            assert not torch.equal(student.R["Q_text"], base.R["Q_text"])
            assert not torch.equal(student.R["text_T"], base.R["text_T"])
    assert denominators == {0: 0, 2: 2}


def test_reports_add_evidence_and_student_diagnostics_without_changing_gates(tmp_path: Path):
    import types as _types

    from mmdd_stage1.pipeline import _reports

    contrast = lambda value: {"mean_delta_pp": value}  # noqa: E731
    candidate = {"C150_target_coverage": 0.8, "MatchedDirectC_target_coverage": 0.7,
                 "Direct_ANN_R10": 0.45, "E_target_coverage": 0.3}
    split = {
        "native_kd": {"candidate": {"overall": candidate}},
        "native_sup": {"candidate": {"overall": {**candidate, "C150_target_coverage": 0.79}}},
        "narrative": {"native_kd": {"implicit": {"TB_CQET.Real.R@10": None}}},
        "contrasts": {
            "content_CQET_Real_minus_Swap_implicit": contrast(1.0),
            "CQET_Real_minus_TB_QT_overall": contrast(0.0),
            "KD_minus_SUP_same_TB_CQET_overall": contrast(0.5),
            "evidence_CQET_Real_minus_f0_overall": contrast(0.8),
            "evidence_CQET_Real_minus_f0_implicit": contrast(None),
            "KD_minus_SUP_same_TB_CQET_implicit": contrast(None),
            "KD_minus_SUP_student_Direct_R10_overall": contrast(4.0),
            "KD_minus_SUP_E_target_coverage_overall": contrast(-0.2),
        },
    }
    protocol = json.loads((Path(__file__).resolve().parents[1] / "configs/mmdd_stage1_cqet_protocol.json").read_text())
    rt = _types.SimpleNamespace(paths=_types.SimpleNamespace(run_root=tmp_path), protocol=protocol)
    decision = _reports(rt, {13: {"splits": {"dev": split, "test": split}}})
    row = decision["candidate_gain"]["rows"][0]
    assert row["evidence_real_minus_f0_pp"] == 0.8 and row["kd_student_direct_pp"] == 4.0
    assert row["KD_student_Direct_R10"] == 0.45 and row["KD_implicit_Real_R10"] is None
    assert decision["KD_gain"]["pass"] is True
    text = (tmp_path / "reports" / "RESULTS.md").read_text()
    assert "Evidence and Student diagnostics (not gates)" in text
    assert "| 13 | dev | 0.8000 | n/a | n/a | 0.4500 | 4.0000 | 0.3000 | -0.2000 | n/a |" in text


def test_lock_content_aliases_hash_each_image_once_across_workers(tmp_path: Path) -> None:
    from PIL import Image

    from mmdd_stage1 import preflight

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (3, 2), (10, 20, 30)).save(images / "a.png")
    Image.new("RGB", (3, 2), (10, 20, 30)).save(images / "a_copy.png", compress_level=0)
    Image.new("RGB", (2, 2), (200, 0, 0)).save(images / "b.png")
    rows = [
        {"asset_id": "img_b", "asset_type": "image", "local_path": str(images / "b.png")},
        {"asset_id": "img_a2", "asset_type": "image", "local_path": str(images / "a_copy.png")},
        {"asset_id": "img_a1", "asset_type": "image", "local_path": str(images / "a.png")},
        {"asset_id": "img_a1_again", "asset_type": "image", "local_path": str(images / "a.png")},
        {"asset_id": "txt_2", "asset_type": "text", "content": "x\r\ny"},
        {"asset_id": "txt_1", "asset_type": "text", "content": "x\ny"},
    ]
    write_jsonl(tmp_path / "bridge_assets" / "part-00000.jsonl", rows)
    paths = Paths(
        repo_root=tmp_path, dataset_root=tmp_path, backbone_dir=tmp_path, pure_cache_dir=tmp_path,
        row_cache_manifest=tmp_path / "manifest", protocol_path=tmp_path / "protocol",
        run_root=tmp_path / "run",
    )

    report = preflight.build_content_aliases(paths, workers=2)
    out = {row["asset_id"]: row for row in iter_jsonl_gz(paths.run_root / "CONTENT_ALIASES.jsonl.gz")}
    assert list(out) == utf8_sorted(out)
    assert report["objects"] == 6 and report["canonical_objects"] == 3 and report["alias_groups"] == 2
    # Same pixels in a different file encoding alias to one canonical image.
    assert {out[key]["canonical_evidence_id"] for key in ("img_a1", "img_a1_again", "img_a2")} == {"img_a1"}
    assert out["img_a1"]["raw_file_sha256"] != out["img_a2"]["raw_file_sha256"]
    assert out["img_a1"]["alias_count"] == 3 and (out["img_a1"]["width"], out["img_a1"]["height"]) == (3, 2)
    pixels = hashlib.sha256((3).to_bytes(8, "big") + (2).to_bytes(8, "big") + bytes([10, 20, 30]) * 6)
    assert out["img_a1"]["pixel_sha256"] == pixels.hexdigest()
    assert out["img_b"]["canonical_evidence_id"] == "img_b" and out["img_b"]["alias_count"] == 1
    assert out["txt_2"]["canonical_evidence_id"] == "txt_1"

    rows[3]["sha256"] = "0" * 64
    write_jsonl(tmp_path / "bridge_assets" / "part-00000.jsonl", rows)
    with pytest.raises(ValueError, match="image hash mismatch for img_a1_again"):
        preflight.build_content_aliases(paths, workers=2)


def iter_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_protocol_side_gpu_is_bound_only_for_two_gpu_runs(tmp_path: Path):
    import run_stage1
    from mmdd_stage1.config import STAGE_ORDER, resolve_default_paths, validate_protocol

    assert "TB_LSE" not in STAGE_ORDER and len(STAGE_ORDER) == 8
    template = json.loads(run_stage1.TEMPLATE.read_text())
    run = tmp_path / "run"
    hardware = {"physical_index": 0, "uuid": "GPU-main", "model": "RTX 4090",
                "gpu_processes": 2, "side_physical_index": 1, "side_uuid": "GPU-side"}
    protocol = run_stage1.build_protocol(
        template, dataset_root=tmp_path / "dataset", run_root=run, features_dir=tmp_path / "features",
        backbone_dir=tmp_path / "qwen", hardware=hardware,
    )
    validate_protocol(protocol)
    run.mkdir()
    (run / "protocol.json").write_text(json.dumps(protocol))
    paths = resolve_default_paths(run / "protocol.json", run)
    assert (paths.gpu_uuid, paths.side_gpu_uuid) == ("GPU-main", "GPU-side")

    bound = protocol["hardware"]
    single = {**{k: v for k, v in bound.items() if not k.startswith("side_")}, "gpu_processes": 1}
    for broken, match in (
        ({**single, "gpu_processes": 2}, "side_uuid is required"),
        ({**single, "side_uuid": "GPU-side", "side_physical_index": 1}, "side_uuid is required"),
        ({**bound, "side_uuid": "GPU-main"}, "second GPU"),
        ({**bound, "gpu_processes": 3}, "one or two"),
    ):
        with pytest.raises(ValueError, match=match):
            validate_protocol({**protocol, "hardware": broken})
    protocol["hardware"] = single
    (run / "protocol.json").write_text(json.dumps(protocol))
    assert resolve_default_paths(run / "protocol.json", run).side_gpu_uuid is None


def test_peer_wait_returns_when_ready_and_fails_when_the_peer_is_gone(tmp_path: Path, monkeypatch):
    import subprocess as _subprocess

    from mmdd_stage1 import pipeline

    monkeypatch.setattr(pipeline, "PEER_POLL_SECONDS", 0)
    paths = types.SimpleNamespace(run_root=tmp_path, gpu_uuid="GPU-side")
    marker = tmp_path / "marker"
    calls = []

    def ready():
        calls.append(1)
        if len(calls) == 3:
            marker.write_text("x")
        return marker.exists()

    pipeline._wait_for(paths, "side", ready, "marker")  # peer not registered yet: keep waiting
    assert len(calls) == 3
    with pytest.raises(RuntimeError, match="missing"):
        pipeline._wait_for(paths, None, lambda: False, "graph")

    pipeline._set_process_state(paths, "side", "FAILED")
    with pytest.raises(RuntimeError, match="side process is FAILED"):
        pipeline._wait_for(paths, "side", lambda: False, "graph")
    dead = _subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    pipeline.write_json(pipeline._process_file(paths, "side"), {"pid": dead.pid, "status": "RUNNING"})
    with pytest.raises(RuntimeError, match="died before producing graph"):
        pipeline._wait_for(paths, "side", lambda: False, "graph")


def test_selection_is_reused_only_while_its_checkpoint_is_unchanged(tmp_path: Path):
    from mmdd_stage1 import pipeline
    from mmdd_stage1.data import sha256_file

    checkpoint = tmp_path / "frac050.pt"
    checkpoint.write_bytes(b"a")
    calls = []

    def select():
        calls.append(1)
        return {"selected_checkpoint": str(checkpoint), "selected_checkpoint_sha256": sha256_file(checkpoint),
                "call": len(calls)}

    path = tmp_path / "QT_C1.json"
    assert pipeline._selection(path, select)["call"] == 1
    assert pipeline._selection(path, select)["call"] == 1  # written by the other process: reused
    checkpoint.write_bytes(b"b")
    assert pipeline._selection(path, select)["call"] == 2

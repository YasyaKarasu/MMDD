"""Repository-level integration tests for fresh_path (SPEC 14.3/14.4 P1).

These are synthetic and CPU-only: they exercise the real stage functions and
real models, not the package's reference maths.  The real-data smoke lives in
``tests/test_fresh_path_real_smoke.py`` and is skipped unless data is present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fresh_path import candidates, config, contracts, features, lineage, train_student, train_teacher  # noqa: E402
from fresh_path.candidates import RawStore  # noqa: E402
from fresh_path.inputs import TrainLabels  # noqa: E402
from fresh_path.models import FreshPathTeacher, PCAStudent, QTOnlyStudent  # noqa: E402
from fresh_path.score import ObjectBank  # noqa: E402
from fresh_path.teacher_cache import FrozenTeacherLogitCache  # noqa: E402
import run_stage1_fresh_path as fresh_path_cli  # noqa: E402

torch.set_num_threads(1)
DIM = 4096
STUDENT_DIM = 8
def _find_protocol() -> Path:
    p = Path(__file__).resolve().parents[1] / "MMDD_STAGE1_FRESH_PATH_v2_1_20260920" / "protocol.json"
    if p.exists():
        return p
    return Path(__file__).resolve().parents[1] / "audit" / "MMDD_STAGE1_FRESH_PATH_v2_1_20260920" / "protocol.json"


PROTOCOL = _find_protocol()


class FakeContent:
    def __init__(self, ids, tokens=4):
        self.ids = list(ids)
        self.tokens = tokens

    def has(self, object_id):
        return object_id in self.ids

    def get(self, object_id):
        g = torch.Generator().manual_seed(abs(hash(object_id)) % 2**31)
        return torch.randn(self.tokens, DIM, generator=g)


def build_world(tmp_path: Path):
    ids = ["query_a", "query_b", "target_1", "target_2", "target_3", "asset_text_x", "asset_img_y"]
    types = ["table", "table", "table", "table", "table", "text", "image"]
    g = torch.Generator().manual_seed(7)
    z = torch.nn.functional.normalize(torch.randn(len(ids), DIM, generator=g), dim=-1)
    store = RawStore(ids=ids, types=types, z=z, index={o: i for i, o in enumerate(ids)})
    bank = ObjectBank(store, FakeContent(ids), lru_bytes=0)

    labels = TrainLabels()
    labels.legal = ["target_1", "target_2", "target_3"]
    labels.epos = {"asset_text_x": ["target_1", "target_2"], "asset_img_y": ["target_3"]}
    labels.queries = {
        "query_a": {
            "query_id": "query_a", "source_group": "st_a", "query_kind": "implicit",
            "G": ["target_1", "target_2"], "D": [], "W": {"target_1": ["asset_text_x"], "target_3": ["asset_img_y"]},
            "Qpos": {"text": ["asset_text_x"], "image": ["asset_img_y"]},
        },
        "query_b": {
            "query_id": "query_b", "source_group": "st_b", "query_kind": "explicit",
            "G": ["target_3"], "D": ["target_3"], "W": {},
            "Qpos": {"text": [], "image": []},
        },
    }
    raw = {
        "legal": labels.legal,
        "targets": ["target_1", "target_2", "target_3"],
        "text_assets": ["asset_text_x"],
        "image_assets": ["asset_img_y"],
        "qt_top256": {"query_a": ["target_1", "target_2", "target_3"], "query_b": ["target_3", "target_2", "target_1"]},
        "qt_reservoir": {"query_a": ["target_1", "target_2", "target_3"], "query_b": ["target_3", "target_2", "target_1"]},
        "qe_reservoir": {"query_a": {"text": ["asset_text_x"], "image": ["asset_img_y"]},
                         "query_b": {"text": ["asset_text_x"], "image": ["asset_img_y"]}},
        "et_reservoir": {"asset_text_x": ["target_1", "target_2", "target_3"],
                         "asset_img_y": ["target_3", "target_1", "target_2"]},
        "admission": {
            "query_a": {"direct": ["target_1", "target_2"], "evidence": ["target_3"], "U": ["target_1", "target_2", "target_3"],
                        "C100": ["target_1", "target_2", "target_3"],
                        "paths": {"target_1": [["asset_text_x", 1.0]], "target_3": [["asset_img_y", 0.5]]}},
            "query_b": {"direct": ["target_3"], "evidence": ["target_1"], "U": ["target_1", "target_3"],
                        "C100": ["target_3", "target_1"], "paths": {"target_1": [["asset_text_x", 1.0]]}},
        },
        "params": {"direct": 100, "hard_pool": 256, "first_hop": 20, "targets_per_evidence": 20,
                   "retained_paths": 4, "rrf_k": 60},
    }
    return store, bank, labels, raw


def build_edges(labels, raw):
    edge = candidates.build_edge_lists(labels, raw)
    positives = {
        qid: {
            key: (entry["G"] if key == "QT" else entry["Qpos"][key[2:]] if key.startswith("Q_")
                  else [t for t in labels.epos.get(key[2:], []) if t in set(raw["legal"])])
            for key in groups
        }
        for qid, entry in labels.queries.items()
        for groups in [edge[qid]]
    }
    return edge, positives


def tiny_teacher(seed=17):
    torch.manual_seed(seed)
    return FreshPathTeacher(input_dim=DIM, width=32, heads=4, layers=2, ffn=64, text_slots=2, image_slots=2).float()


def tiny_student(adapter=True):
    g = torch.Generator().manual_seed(3)
    basis = torch.linalg.qr(torch.randn(DIM, STUDENT_DIM, generator=g))[0].T
    return PCAStudent(basis, adapter=adapter)


def test_edge_lists_include_all_positives_and_protect_scope():
    store, bank, labels, raw = build_world(Path("/tmp"))
    edge, positives = build_edges(labels, raw)
    assert set(labels.queries["query_a"]["G"]) <= set(edge["query_a"]["QT"])
    assert "asset_text_x" in edge["query_a"]["Q_text"]
    assert "E_asset_text_x" in edge["query_a"]


def test_teacher_edge_training_updates_all_param_groups(tmp_path):
    store, bank, labels, raw = build_world(tmp_path)
    edge, positives = build_edges(labels, raw)
    model = tiny_teacher()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    losses = train_teacher.edge_relation_losses(model, bank, "query_a", edge["query_a"], positives, labels, "cpu")
    total = sum(v for v in losses.values() if v is not None)
    total.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert any("adapters.text" in n and g.norm() > 0 for n, g in grads.items())
    assert any("poolers.image" in n and g.norm() > 0 for n, g in grads.items())
    assert any("globals.table" in n and g.norm() > 0 for n, g in grads.items())
    counters = train_teacher.train_edge(model, bank, labels, edge, positives, seed=13, epochs=1, lr=1e-3,
                                        logical_batch=2, device="cpu", out_dir=tmp_path / "edge",
                                        protocol_path=PROTOCOL, log=lambda *_: None)
    assert counters["updates"] == 1
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    assert changed, "T_EDGE must update at least one trainable tensor"


def test_path_and_qt_branches_share_parent_but_not_storage(tmp_path):
    store, bank, labels, raw = build_world(tmp_path)
    edge, positives = build_edges(labels, raw)
    edge_model = tiny_teacher()
    graph = train_teacher.build_train_graph(edge_model, bank, labels, raw, device="cpu")
    path_model = tiny_teacher()
    path_model.load_state_dict(edge_model.state_dict())
    qt_model = tiny_teacher()
    qt_model.load_state_dict(edge_model.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(path_model.parameters(), qt_model.parameters()))
    train_teacher.train_path(path_model, bank, labels, edge, positives, graph, raw, seed=13, epochs=1, lr=1e-3,
                             logical_batch=2, device="cpu", out_dir=tmp_path / "path",
                             protocol_path=PROTOCOL, parent_dirs=[], log=lambda *_: None)
    moved = [n for (n, a), b in zip(path_model.named_parameters(), qt_model.parameters()) if not torch.equal(a, b)]
    assert moved, "train_path must move at least one shared-parent tensor"
    # the untouched branch stays bit-identical to the common parent
    assert all(torch.equal(a, b) for a, b in zip(qt_model.parameters(), edge_model.parameters()))
    # QET forward must read evidence content
    a = path_model.score_triplets([
        ("table", bank.z("query_a"), bank.tokens("query_a"), "text", bank.z("asset_text_x"),
         bank.tokens("asset_text_x"), "table", bank.z("target_1"), bank.tokens("target_1"))])
    b = path_model.score_triplets([
        ("table", bank.z("query_a"), bank.tokens("query_a"), "text", bank.z("asset_text_x"),
         bank.tokens("asset_text_x") * 3 + 1, "table", bank.z("target_1"), bank.tokens("target_1"))])
    assert not torch.allclose(a, b)


def test_post_edge_refresh_and_conditional_registry_match_dynamic_lists(tmp_path):
    _, bank, labels, raw = build_world(tmp_path)
    model = tiny_teacher().eval()
    refreshed = train_teacher.refreshed_hard(model, bank, labels, raw, device="cpu")
    assert ("query_a", "QT") in refreshed
    assert ("query_a", "E_asset_text_x") in refreshed

    registry = train_teacher.build_conditional_registry(labels, raw)
    for query_id, by_asset in registry.items():
        for asset in by_asset:
            dynamic = train_teacher.conditional_list(labels, raw, query_id, asset)
            prepared = train_teacher.conditional_list(labels, raw, query_id, asset, registry=registry)
            assert prepared == dynamic
    anchors = train_teacher.build_anchor_registry(labels, 3)
    for epoch, by_query in anchors.items():
        for query_id, anchor in by_query.items():
            assert anchor == train_teacher.anchor_choice(labels, query_id, epoch)


def test_frozen_teacher_cache_hit_matches_online_forward_and_is_branch_scoped(tmp_path):
    _, bank, _, _ = build_world(tmp_path)
    model = tiny_teacher(seed=29).eval()
    candidates = ["target_1", "target_2", "target_3"]
    cache_path = tmp_path / "teacher_logits.pt"
    cache = FrozenTeacherLogitCache.for_model(
        model, PROTOCOL, teacher_branch="T_PATH", path=cache_path,
    )

    with torch.no_grad():
        online = train_teacher.pair_logits(model, bank, "query_a", candidates, "cpu", view="cache_test")
        first = train_teacher.pair_logits(
            model, bank, "query_a", candidates, "cpu", teacher_cache=cache, view="cache_test",
        )
        hit = train_teacher.pair_logits(
            model, bank, "query_a", candidates, "cpu", teacher_cache=cache, view="cache_test",
        )
    assert torch.equal(online, first)
    assert torch.equal(first, hit)
    assert (cache.hits, cache.misses) == (1, 1)
    metadata = next(iter(cache.entries.values()))["metadata"]
    assert metadata["teacher_branch"] == "T_PATH"
    assert metadata["teacher_hash"] == cache.teacher_hash
    assert metadata["protocol_hash"] == cache.protocol_hash
    assert metadata["candidate_ids"] == tuple(candidates)
    assert metadata["mask"] == (True, True, True)
    assert metadata["view"] == "cache_test"

    with torch.no_grad():
        train_teacher.pair_logits(
            model, bank, "query_a", list(reversed(candidates)), "cpu",
            teacher_cache=cache, view="cache_test",
        )
    assert cache.misses == 2
    cache.save()
    with pytest.raises(ValueError, match="metadata mismatch"):
        FrozenTeacherLogitCache.load(
            cache_path, teacher_branch="T_QT", teacher_hash=cache.teacher_hash,
            protocol_hash=cache.protocol_hash,
        )


def test_static_loss_activity_avoids_tensor_item_sync():
    import inspect

    logits = torch.tensor([0.2, -0.1, 0.4])
    positive = torch.tensor([True, False, False])
    allowed = torch.tensor([True, True, True])
    direct = contracts.rank_loss(logits, positive, allowed, active=True)
    expected = torch.logsumexp(logits, 0) - logits[0]
    assert torch.equal(direct, expected)

    query = torch.tensor([0.3, -0.2])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    streamed = contracts.streamed_full_loss(
        query, keys, positive, allowed, 2,
        active=True, chunk_activity=((True, True), (True, False)),
    )
    full_scores = keys @ query
    assert torch.allclose(
        streamed, torch.logsumexp(full_scores, 0) - full_scores[0], atol=1e-5, rtol=1e-4,
    )
    assert ".item()" not in inspect.getsource(contracts.rank_loss)
    assert ".item()" not in inspect.getsource(contracts.streamed_full_loss)


def test_qt_static_registry_scores_frozen_teacher_once_across_epochs(tmp_path, monkeypatch):
    _, bank, labels, raw = build_world(tmp_path)
    teacher = tiny_teacher(seed=41).eval()
    sup_registry = train_student.build_qt_static_registry(
        None, bank, labels, raw, stage="QT_SUP_C1", device="cpu", log=lambda *_: None,
    )
    for query_id, prepared in sup_registry.items():
        assert list(prepared.candidates) == train_student.qt_c1_list(labels, raw, query_id)
        assert prepared.teacher_logits is None

    calls = []
    original = train_teacher.pair_logits

    def record(*args, **kwargs):
        calls.append((args[2], kwargs.get("view")))
        return original(*args, **kwargs)

    monkeypatch.setattr(train_teacher, "pair_logits", record)
    basis = torch.linalg.qr(torch.randn(DIM, STUDENT_DIM, generator=torch.Generator().manual_seed(43)))[0].T
    model = QTOnlyStudent(basis)
    train_student.train_qt_student(
        model, teacher, bank, labels, raw, seed=13, stage="QT_KD_C2", epochs=3, lr=1e-3,
        logical_batch=2, device="cpu", out_dir=tmp_path / "qt_kd_registry",
        protocol_path=PROTOCOL, parent_dirs=[], log=lambda *_: None,
    )
    query_count = len(labels.queries)
    assert sum(view == "qt_c2_hard" for _, view in calls) == query_count
    assert sum(view == "qt_c2_kd" for _, view in calls) == query_count


def test_scheduler_releases_c1_at_its_declared_dependencies():
    dependencies = fresh_path_cli.RUN_DEPENDENCIES
    assert dependencies["S_SUP_C1"] == ("post_edge",)
    assert set(dependencies["S_KD_C1"]) == {"post_edge", "T_PATH"}
    command = fresh_path_cli._stage_command(["python", "runner.py"], "T_QT", 13, 1024)
    assert command[-4:] == ["--device", "cuda:0", "--path-batch", "1024"]


def test_completed_legacy_qt_stage_is_reused_but_not_resumable(tmp_path):
    class Paths:
        protocol = PROTOCOL

        @staticmethod
        def stage_dir(seed, stage):
            return tmp_path / f"seed{seed}" / stage

    stage = "S_QT_SUP_C1"
    stage_dir = Paths.stage_dir(13, stage)
    stage_dir.mkdir(parents=True)
    legacy = {
        "run_id": lineage.RUN_ID,
        "stage": "S_S_QT_SUP_C1",
        "model_seed": 13,
        "epoch": 1,
        "protocol_hash": lineage.protocol_hash(PROTOCOL),
        "state_dict": {"weight": torch.ones(1)},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "counters": {"updates": 1},
    }
    torch.save(legacy, stage_dir / "checkpoint.pt")
    (stage_dir / "receipt.json").write_text(json.dumps({"stage": legacy["stage"], "epoch": 1}))
    assert fresh_path_cli._stage_completed(Paths, 13, stage, 1)

    legacy["stage"] = stage
    legacy["epoch"] = 0
    torch.save(legacy, stage_dir / "checkpoint.pt")
    with pytest.raises(ValueError, match="no strict-resume metadata"):
        fresh_path_cli._resume_stage(
            torch.nn.Linear(1, 1), stage_dir=stage_dir, stage=stage, seed=13, epochs=1,
            protocol_path=PROTOCOL,
            make_optimizer=lambda: torch.optim.AdamW(torch.nn.Linear(1, 1).parameters()),
        )


def test_train_path_uses_separate_dropout_for_natural_and_augmented_views(tmp_path, monkeypatch):
    _, bank, labels, raw = build_world(tmp_path)
    edge, positives = build_edges(labels, raw)
    graph = train_teacher.build_train_graph(tiny_teacher().eval(), bank, labels, raw, device="cpu")
    model = tiny_teacher()
    calls = []
    original = train_teacher.target_view_scores

    def record(*args, **kwargs):
        calls.append(args[6])
        return original(*args, **kwargs)

    monkeypatch.setattr(train_teacher, "target_view_scores", record)
    monkeypatch.setattr(
        train_teacher,
        "augmented_scores_from_natural",
        lambda *args, **kwargs: pytest.fail("training must not reuse Natural logits"),
    )
    train_teacher.train_path(
        model, bank, labels, edge, positives, graph, raw, seed=13, epochs=1, lr=1e-3,
        logical_batch=2, device="cpu", out_dir=tmp_path / "path", protocol_path=PROTOCOL,
        parent_dirs=[], log=lambda *_: None,
    )
    assert calls.count(None) == len(labels.queries)
    assert "asset_text_x" in calls


def test_teacher_epoch_resume_restores_dropout_trajectory(tmp_path):
    _, bank, labels, raw = build_world(tmp_path)
    edge, positives = build_edges(labels, raw)

    full = tiny_teacher(seed=31)
    torch.manual_seed(2026)
    train_teacher.train_edge(
        full, bank, labels, edge, positives, seed=13, epochs=2, lr=1e-3, logical_batch=2,
        device="cpu", out_dir=tmp_path / "full", protocol_path=PROTOCOL, log=lambda *_: None,
    )
    expected = {name: value.detach().clone() for name, value in full.state_dict().items()}

    partial = tiny_teacher(seed=31)
    torch.manual_seed(2026)
    train_teacher.train_edge(
        partial, bank, labels, edge, positives, seed=13, epochs=1, lr=1e-3, logical_batch=2,
        device="cpu", out_dir=tmp_path / "partial", protocol_path=PROTOCOL, log=lambda *_: None,
    )
    payload = lineage.load_checkpoint(tmp_path / "partial" / "checkpoint.pt")
    assert (tmp_path / "partial" / "init.pt").is_file()
    assert (tmp_path / "partial" / "epoch1.pt").is_file()
    assert payload["data_state"]["order_cursor"] == len(payload["data_state"]["order"])

    resumed = tiny_teacher(seed=99)  # construction must not perturb restored RNG.
    start_epoch, optimizer, counters = fresh_path_cli._resume_stage(
        resumed, stage_dir=tmp_path / "partial", stage="T_EDGE", seed=13, epochs=2,
        protocol_path=PROTOCOL,
        make_optimizer=lambda: torch.optim.AdamW(
            resumed.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
        ),
    )
    assert start_epoch == 2
    assert counters == payload["counters"]
    train_teacher.train_edge(
        resumed, bank, labels, edge, positives, seed=13, epochs=2, lr=1e-3, logical_batch=2,
        device="cpu", out_dir=tmp_path / "partial", protocol_path=PROTOCOL, optimizer=optimizer,
        start_epoch=start_epoch, counters=counters, log=lambda *_: None,
    )
    assert (tmp_path / "partial" / "epoch2.pt").is_file()
    assert all(torch.equal(expected[name], value) for name, value in resumed.state_dict().items())


def test_students_have_declared_trainable_scopes(tmp_path):
    store, bank, labels, raw = build_world(tmp_path)
    adapter_student = tiny_student(adapter=True)
    native_student = tiny_student(adapter=False)
    assert adapter_student.adapter is not None
    assert native_student.adapter is None
    assert set(dict(native_student.named_parameters())) >= {"projections.table.weight", "relations.QT"}


def test_c1_c2_native_and_qt_train_without_nan(tmp_path):
    store, bank, labels, raw = build_world(tmp_path)
    edge, positives = build_edges(labels, raw)
    teacher = tiny_teacher()
    graph = train_teacher.build_train_graph(teacher, bank, labels, raw, device="cpu")
    for kd in (False, True):
        student = tiny_student(adapter=True)
        out = tmp_path / f"c1_{kd}"
        counters = train_student.train_c1(student, teacher, bank, labels, edge, positives, seed=13, epochs=1,
                                          lr=1e-3, logical_batch=2, device="cpu", out_dir=out,
                                          protocol_path=PROTOCOL, kd=kd, log=lambda *_: None)
        assert counters["updates"] == 1
    # conditional C2 (SUP and KD) trains only the adapter
    for kd, eonly in ((False, False), (True, False), (True, True)):
        student = tiny_student(adapter=True)
        before = {n: p.detach().clone() for n, p in student.named_parameters()}
        counters = train_student.train_conditional_c2(
            student, teacher, bank, labels, raw, graph, seed=13, epochs=1, lr=1e-3, logical_batch=2,
            device="cpu", out_dir=tmp_path / f"c2_{kd}_{eonly}", protocol_path=PROTOCOL,
            kd=kd, eonly=eonly, parent_dirs=[], log=lambda *_: None)
        assert counters["updates"] == 1
        assert torch.equal(before["projections.table.weight"], student.projections.table.weight)
        assert torch.equal(before["relations.QT"], student.relations["QT"])
        assert not torch.equal(before["adapter.output.weight"], student.adapter.output.weight)
    # NATIVE updates P/R and has no adapter
    native = tiny_student(adapter=False)
    before = native.projections.table.weight.detach().clone()
    train_student.train_native(native, teacher, bank, labels, raw, graph, seed=13, epochs=1, lr=1e-3,
                               logical_batch=2, device="cpu", out_dir=tmp_path / "native",
                               protocol_path=PROTOCOL, parent_dirs=[], log=lambda *_: None)
    assert not torch.equal(before, native.projections.table.weight)
    # QT-only C1 and C2
    qt = QTOnlyStudent(torch.linalg.qr(torch.randn(DIM, STUDENT_DIM, generator=torch.Generator().manual_seed(4)))[0].T)
    for stage in ("QT_SUP_C1", "QT_KD_C1", "QT_SUP_C2", "QT_KD_C2"):
        model = QTOnlyStudent(qt.P_table.detach().clone())
        counters = train_student.train_qt_student(
            model, teacher, bank, labels, raw, seed=13, stage=stage, epochs=1, lr=1e-3, logical_batch=2,
            device="cpu", out_dir=tmp_path / stage, protocol_path=PROTOCOL,
            parent_dirs=[], log=lambda *_: None)
        assert counters["updates"] == 1


def test_native_et_is_query_independent():
    store, bank, labels, raw = build_world(Path("/tmp"))
    model = tiny_student(adapter=False)
    a = model.native_second_hop("text", bank.z("asset_text_x"), bank.z("target_1"))
    b = model.native_second_hop("text", bank.z("asset_text_x"), bank.z("target_1"))
    assert torch.equal(a, b)
    # changing the query projection cannot change the ET term
    u_e = model.u("text", bank.z("asset_text_x"))
    v = u_e @ model.relations["text_to_T"]
    assert torch.allclose(a, v @ model.u("table", bank.z("target_1")).T)


def test_native_teacher_batches_first_hop_without_changing_scores(tmp_path, monkeypatch):
    _, bank, _, raw = build_world(tmp_path)
    teacher = tiny_teacher(seed=37).eval()
    query_id = "query_a"
    graph = ["target_1", "target_2", "target_3"]
    per_target = train_teacher.paths_for_targets(raw, query_id, graph, None)
    assets = sorted({e for paths in per_target.values() for e in paths}, key=lambda x: x.encode("utf-8"))
    original = train_teacher.pair_logits

    with torch.no_grad():
        f0 = original(teacher, bank, query_id, graph, "cpu", view="legacy:target")
        f_qe = {
            e: original(teacher, bank, query_id, [e], "cpu", batch=1, view=f"legacy:first:{e}")[0]
            for e in assets
        }
        f_et = {}
        for e in assets:
            targets = sorted(
                {t for t, paths in per_target.items() if e in paths}, key=lambda x: x.encode("utf-8"),
            )
            scores = original(teacher, bank, e, targets, "cpu", view=f"legacy:second:{e}")
            f_et[e] = dict(zip(targets, scores))
        expected = torch.stack([
            torch.logsumexp(torch.cat([
                f0[i].reshape(1),
                torch.stack([f_qe[e] + f_et[e][target] for e in per_target[target]]),
            ]), 0) if per_target[target] else f0[i]
            for i, target in enumerate(graph)
        ])

    calls = []

    def record(*args, **kwargs):
        calls.append((args[2], tuple(args[3]), kwargs.get("view"), kwargs.get("batch")))
        return original(*args, **kwargs)

    monkeypatch.setattr(train_teacher, "pair_logits", record)
    with torch.no_grad():
        actual = train_student.native_teacher_paths(
            teacher, bank, raw, query_id, graph, None, "cpu", batch=16,
        )
    first_hop = [call for call in calls if call[2] == "native:first_hop"]
    assert first_hop == [(query_id, tuple(assets), "native:first_hop", 16)]
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_conditional_adapter_output_layer_starts_at_base():
    store, bank, labels, raw = build_world(Path("/tmp"))
    student = tiny_student(adapter=True)
    v = train_student.adapter_vector(student, bank, "query_a", "asset_text_x", "cpu")
    base = student.v("text", bank.z("asset_text_x"))
    assert torch.equal(v, base)


def _perturbed_adapter(student):
    """Move the adapter off its zero-output start so the branch is observable."""
    g = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for name, param in student.adapter.named_parameters():
            if name == "output.weight":
                param.copy_(torch.randn(param.shape, generator=g) * 0.05)
            elif name == "hidden.weight":
                param.copy_(torch.randn(param.shape, generator=g) * 0.05)
            elif name == "output.bias":
                param.copy_(torch.randn(param.shape, generator=g) * 0.05)


def test_ann_second_hop_uses_the_training_adapter_vector():
    """SPEC 8.3: train/full exact/ANN/cache all call the same second-hop function.

    The ANN retriever must build the conditional ``v_qe`` via the adapter, not the
    unconditional ``u_e @ R_{e->T}``.  The unconditional vector is what this test
    guards against: it was the shape of the original bug.
    """
    from fresh_path import retrieval

    store, bank, labels, raw = build_world(Path("/tmp"))
    student = tiny_student(adapter=True)
    _perturbed_adapter(student)
    retriever = retrieval.OwnRetriever(student, bank, labels.legal, raw["text_assets"],
                                       raw["image_assets"], device="cpu", read_q=True)

    got = retriever._second_hop_vector("query_a", "asset_text_x", "text")
    expected = train_student.adapter_vector(student, bank, "query_a", "asset_text_x", "cpu")
    assert np.allclose(got, expected.detach().float().cpu().numpy(), atol=1e-6)

    # and it must not equal the adapter-free vector that the bug produced
    unconditional = student.v("text", bank.z("asset_text_x"))
    assert not np.allclose(got, unconditional.detach().float().cpu().numpy(), atol=1e-6)


def test_ann_second_hop_distinguishes_query_conditioned_from_eonly():
    """KD-QE and KD-EONLY share P/R; only the second-hop query vector may differ."""
    from fresh_path import retrieval

    store, bank, labels, raw = build_world(Path("/tmp"))
    student = tiny_student(adapter=True)
    _perturbed_adapter(student)
    qe = retrieval.OwnRetriever(student, bank, labels.legal, raw["text_assets"], raw["image_assets"],
                                device="cpu", read_q=True)
    eonly = retrieval.OwnRetriever(student, bank, labels.legal, raw["text_assets"], raw["image_assets"],
                                   device="cpu", read_q=False)

    a = qe._second_hop_vector("query_a", "asset_text_x", "text")
    b = eonly._second_hop_vector("query_a", "asset_text_x", "text")
    assert not np.allclose(a, b, atol=1e-6)
    # E-only must not read the query at all
    assert np.allclose(b, train_student.adapter_vector(student, bank, "query_a", "asset_text_x", "cpu",
                                                       read_q=False).detach().float().cpu().numpy(), atol=1e-6)
    other = eonly._second_hop_vector("query_b", "asset_text_x", "text")
    assert np.allclose(b, other, atol=1e-6)


def test_ann_second_hop_falls_back_to_native_when_no_adapter():
    """KD-NATIVE has no adapter; its second hop stays the unconditional vector."""
    from fresh_path import retrieval

    store, bank, labels, raw = build_world(Path("/tmp"))
    native = tiny_student(adapter=False)
    retriever = retrieval.OwnRetriever(native, bank, labels.legal, raw["text_assets"], raw["image_assets"],
                                       device="cpu")
    got = retriever._second_hop_vector("query_a", "asset_text_x", "text")
    expected = native.v("text", bank.z("asset_text_x"))
    assert np.allclose(got, expected.detach().float().cpu().numpy(), atol=1e-6)


def test_probe_et_respects_read_q(monkeypatch):
    """The fixed probe must forward read_q to the same conditional vector function."""
    from fresh_path import evaluate
    from fresh_path import train_student as ts

    store, bank, labels, raw = build_world(Path("/tmp"))
    student = tiny_student(adapter=True)
    _perturbed_adapter(student)
    seen = []
    original = ts.adapter_vector

    def record(model, bank_, qid, asset, device, *, read_q=True):
        seen.append(read_q)
        return original(model, bank_, qid, asset, device, read_q=read_q)

    monkeypatch.setattr(ts, "adapter_vector", record)
    pairs = [("query_a", "asset_text_x", ["target_1", "target_2"])]
    full = evaluate.probe_et(student, bank, labels.legal, pairs, "cpu")
    eonly = evaluate.probe_et(student, bank, labels.legal, pairs, "cpu", read_q=False)
    assert full["pairs"] == 1 and eonly["pairs"] == 1
    assert seen == [True, False]


def test_qt_only_loader_scope_has_no_evidence_relations():
    qt = QTOnlyStudent(torch.randn(STUDENT_DIM, DIM))
    assert set(dict(qt.named_parameters())) == {"P_table", "R_QT"}


def test_protocol_rejects_legacy_roles(tmp_path):
    payload = json.loads(PROTOCOL.read_text())
    config.validate_protocol(payload)
    with pytest.raises(ValueError):
        bad = dict(payload)
        bad["lineage"] = dict(payload["lineage"], external_task_checkpoint=True)
        config.validate_protocol(bad)


def test_teacher_eval_is_permutation_invariant():
    """Padding does not change any sequence's own eval score."""
    store, bank, labels, raw = build_world(Path("/tmp"))
    torch.manual_seed(11)
    model = FreshPathTeacher(input_dim=DIM, width=32, heads=4, layers=2, ffn=64,
                             text_slots=2, image_slots=2).float().eval()
    pairs = [
        ("table", bank.z("query_a"), bank.tokens("query_a"), "table", bank.z(t), bank.tokens(t))
        for t in ["target_1", "target_2", "target_3"]
    ]
    pairs.append(("table", bank.z("query_b"), bank.tokens("query_b"), "text",
                  bank.z("asset_text_x"), bank.tokens("asset_text_x")))
    perm = [3, 0, 2, 1]
    with torch.no_grad():
        a = model.score_pairs(pairs)
        b = model.score_pairs([pairs[i] for i in perm])
    back = [perm.index(i) for i in range(len(perm))]
    assert torch.equal(a, b[back]), (a, b, back)


def test_teacher_training_preserves_pair_input_order(monkeypatch):
    _, bank, _, _ = build_world(Path("/tmp"))
    model = tiny_teacher().train()
    pairs = [
        ("table", bank.z("query_a"), bank.tokens("query_a"), "table", bank.z(target), bank.tokens(target))
        for target in ["target_1", "target_2", "target_3"]
    ]
    pairs.append(("table", bank.z("query_b"), bank.tokens("query_b"), "text",
                  bank.z("asset_text_x"), bank.tokens("asset_text_x")))
    seen = []
    original = model.relation.forward

    def record(x, *args, **kwargs):
        seen.append((~kwargs["src_key_padding_mask"]).sum(dim=1).tolist())
        return original(x, *args, **kwargs)

    monkeypatch.setattr(model.relation, "forward", record)
    model.score_pairs(pairs)
    assert seen == [[10, 10, 10, 8]]


def test_content_store_keeps_float16_lru_and_refreshes_hits(tmp_path):
    chunk_dir = tmp_path / "content" / "chunks"
    rows = [(f"id_{i}", "table", torch.full((2, 3), i, dtype=torch.float16).numpy()) for i in range(3)]
    features.write_chunk(chunk_dir, 0, rows)
    features.build_chunk_index(chunk_dir)
    store = features.ContentStore(tmp_path / "content", lru_bytes=24)
    assert store.get("id_0").dtype == torch.float16
    store.get("id_1")
    store.get("id_0")  # hit moves id_0 to the LRU tail
    store.get("id_2")
    assert list(store._lru) == ["id_0", "id_2"]
    assert store._lru_used == 24


def test_augmented_view_reuse_is_exact():
    """Incremental augmented view must equal the from-scratch augmented view."""
    from fresh_path import train_teacher
    store, bank, labels, raw = build_world(Path("/tmp"))
    torch.manual_seed(23)
    model = tiny_teacher(seed=23).eval()
    q = "query_a"
    graph = sorted(labels.queries[q]["G"]) + ["target_3"]
    natural = train_teacher.target_view_scores(model, bank, labels, raw, q, graph, None, "cpu")
    e_star = "asset_text_x"
    reused = train_teacher.augmented_scores_from_natural(model, bank, raw, q, graph, natural, e_star, "cpu")
    direct = train_teacher.target_view_scores(model, bank, labels, raw, q, graph, e_star, "cpu")
    # Mathematically identical (LSE(S_nat, f(q,e*,t))); the only difference is
    # float32 reduction order, because the reused path evaluates fewer QET
    # sequences and therefore reduces in a different batch layout.  SPEC 14.5
    # sets the float32 comparison band at atol 1e-5 / rtol 1e-4.
    diff = float((reused - direct).abs().max().detach())
    assert diff < 1e-5, diff
    assert torch.allclose(reused, direct, atol=1e-5, rtol=1e-4)


def test_grouped_student_path_scores_restore_original_slots_and_gradients():
    """Evidence batching must not change target-major values or gradients."""
    store, bank, labels, raw = build_world(Path("/tmp"))
    model = tiny_student(adapter=True)
    flat = [("asset_text_x", "target_1"), ("asset_img_y", "target_1"),
            ("asset_text_x", "target_2"), ("asset_img_y", "target_2")]
    got = train_student._conditional_path_scores(model, bank, "query_a", flat, "cpu")
    expected = torch.stack([
        train_student.first_hop_logits(model, bank, "query_a", bank.kind(e), [e], "cpu")[0]
        + train_student.adapter_vector(model, bank, "query_a", e, "cpu")
        @ train_student.target_keys(model, bank, [t], "cpu").T
        for e, t in flat
    ]).reshape(-1)
    assert torch.allclose(got, expected)
    loss_got = got.square().sum()
    loss_expected = expected.square().sum()
    grad_got = torch.autograd.grad(loss_got, model.relations["text_to_T"], retain_graph=True)[0]
    grad_expected = torch.autograd.grad(loss_expected, model.relations["text_to_T"])[0]
    assert torch.allclose(grad_got, grad_expected)


def test_own_candidate_rerank_rejects_unbound_raw_paths():
    """An own-candidate rerank cannot silently consume raw-split paths."""
    from fresh_path import evaluate

    with pytest.raises(ValueError, match="PathPool"):
        evaluate.rerank_path(
            None, None, "query_a", ["target_1"], {"target_1": []}, "cpu",
            expected_generator="student:S_KD_NATIVE",
        )

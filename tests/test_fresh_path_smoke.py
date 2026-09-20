"""Repository-level integration tests for fresh_path (SPEC 14.3/14.4 P1).

These are synthetic and CPU-only: they exercise the real stage functions and
real models, not the package's reference maths.  The real-data smoke lives in
``tests/test_fresh_path_real_smoke.py`` and is skipped unless data is present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fresh_path import candidates, config, features, lineage, train_student, train_teacher  # noqa: E402
from fresh_path.candidates import RawStore  # noqa: E402
from fresh_path.inputs import TrainLabels  # noqa: E402
from fresh_path.models import FreshPathTeacher, PCAStudent, QTOnlyStudent  # noqa: E402
from fresh_path.score import ObjectBank  # noqa: E402
import run_stage1_fresh_path as fresh_path_cli  # noqa: E402

torch.set_num_threads(1)
DIM = 4096
STUDENT_DIM = 8
PROTOCOL = Path(__file__).resolve().parents[1] / "MMDD_STAGE1_FRESH_PATH_v2_1_20260920" / "protocol.json"


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
    assert len(changed) == len(before)


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


def test_scheduler_releases_c1_at_its_declared_dependencies():
    dependencies = fresh_path_cli.RUN_DEPENDENCIES
    assert dependencies["S_SUP_C1"] == ("post_edge",)
    assert set(dependencies["S_KD_C1"]) == {"post_edge", "T_PATH"}


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


def test_conditional_adapter_output_layer_starts_at_base():
    store, bank, labels, raw = build_world(Path("/tmp"))
    student = tiny_student(adapter=True)
    v = train_student.adapter_vector(student, bank, "query_a", "asset_text_x", "cpu")
    base = student.v("text", bank.z("asset_text_x"))
    assert torch.equal(v, base)


def test_qt_only_loader_scope_has_no_evidence_relations():
    qt = QTOnlyStudent(torch.randn(STUDENT_DIM, DIM))
    assert set(dict(qt.named_parameters())) == {"P_table", "R_QT"}


def test_protocol_rejects_legacy_roles(tmp_path):
    payload = json.loads(Path(__file__).resolve().parents[1].joinpath(
        "MMDD_STAGE1_FRESH_PATH_v2_1_20260920", "protocol.json").read_text())
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

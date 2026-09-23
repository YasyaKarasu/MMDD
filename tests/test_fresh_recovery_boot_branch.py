"""The additional T_BOOT continuation is evaluated without replacing the Path arm."""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fresh_recovery.config import Paths
from fresh_recovery.dag import build_nodes
from fresh_recovery.final_eval import _contrasts, _k20_pools, _strict_150, _witness_funnel, cmd_evaluate
from fresh_recovery.evaluate import swap_map, teacher_readouts
from fresh_recovery.stages import cmd_build_raw, cmd_freeze_models
from fresh_recovery.teacher import matched_final_order
from fresh_recovery.recompute import coverage_from_ids, oracle_from_ids, recall_from_ids
from fresh_recovery.package import acceptance_ledger


def test_boot_continuation_is_a_separate_prerequisite_and_readout() -> None:
    nodes = {node.name: node for node in build_nodes()}
    assert nodes["T_QT_FROM_BOOT"].parents == ["HARD32", "T_BOOT"]
    assert nodes["T_PATH"].parents == ["T_QT", "SELECT_native_C2"]
    assert "T_QT_FROM_BOOT" in nodes["EVAL_DEV"].parents
    assert "T_QT_FROM_BOOT" in nodes["FREEZE_MODELS"].parents
    teachers = inspect.signature(cmd_evaluate).parameters["teachers"].default
    assert teachers == ("T_QT", "T_QT_FROM_BOOT", "T_QT_CONT", "T_PATH")


def test_boot_continuation_compared_on_identical_candidate_ids() -> None:
    gold = {"q1": ["t1"], "q2": ["t1"]}
    kinds = {"q1": "implicit", "q2": "explicit"}
    groups = {"q1": "s1", "q2": "s2"}
    ranks = {}
    for gen in ("S_KD_NATIVE", "Raw"):
        ranks[f"{gen}+T_QT_FROM_BOOT|C150"] = {"q1": ["t1"], "q2": ["t1"]}
        ranks[f"{gen}+T_QT|C150"] = {"q1": ["t2"], "q2": ["t2"]}
    contrasts = _contrasts(gold, kinds, groups, ranks)
    for key in ("BOOT_init_vs_fresh_same_pool", "BOOT_init_vs_fresh_Raw"):
        assert contrasts[key]["overall"]["point_estimate"] == 1.0
        assert contrasts[key]["implicit"]["point_estimate"] == 1.0
        assert contrasts[key]["overall"]["seed"] == 20260923


def test_model_lock_requires_boot_continuation(tmp_path: Path) -> None:
    paths = Paths(tmp_path, tmp_path, tmp_path, tmp_path / "protocol.json", tmp_path)
    with pytest.raises(RuntimeError, match="T_QT_FROM_BOOT"):
        cmd_freeze_models(paths, seed=13)


def test_independent_recall_requires_unique_ranked_ids() -> None:
    assert recall_from_ids(["a", "b"], ["b", "c"], 1) == 0.5
    assert coverage_from_ids(["a", "b"], ["b", "c"]) == 0.5
    assert oracle_from_ids(["a", "b"], ["a", "b", "c"], 1) == 0.5
    with pytest.raises(ValueError, match="duplicate"):
        recall_from_ids(["a"], ["a", "a"], 10)


def test_a29_requires_both_successful_recomputations(tmp_path: Path) -> None:
    paths = Paths(tmp_path, tmp_path, tmp_path, tmp_path / "protocol.json", tmp_path)
    ledger = {row["id"]: row for row in acceptance_ledger(paths, 13)}
    assert ledger["A29"]["status"] == "NOT_REACHED"


def test_test_inputs_reject_before_model_lock(tmp_path: Path) -> None:
    paths = Paths(tmp_path, tmp_path, tmp_path, tmp_path / "protocol.json", tmp_path)
    with pytest.raises(RuntimeError, match="MODEL_LOCK.json"):
        cmd_build_raw(paths, split="test", device="cpu")
    with pytest.raises(RuntimeError, match="MODEL_LOCK.json"):
        cmd_evaluate(paths, seed=13, split="test", device="cpu")


def test_final_teacher_order_is_shared_and_seeded() -> None:
    queries = [f"q{i:02}" for i in range(40)]
    first = matched_final_order(queries, root="original-root", seed=13, epoch=1)
    assert first == matched_final_order(reversed(queries), root="original-root", seed=13, epoch=1)
    assert set(first) == set(queries)
    assert first != matched_final_order(queries, root="original-root", seed=13, epoch=2)


def test_strict_150_excludes_ann_and_exact_direct_of_every_generator() -> None:
    from types import SimpleNamespace

    gold = {"q": ["t_ann", "t_exact", "t_remaining"]}
    pools = {gen: {"q": SimpleNamespace(evidence_ids=["t_remaining"], U=["t_remaining"],
                                         C150=["t_remaining"])} for gen in ("Raw", "S_SUP_NATIVE", "S_KD_NATIVE")}
    d150 = {gen: {"q": ["t_ann"]} for gen in pools}
    d150_exact = {gen: {"q": ["t_exact"]} for gen in pools}
    strict = _strict_150(gold, pools, {}, d150, d150_exact)
    assert strict["fixed_cohort_queue"] == {"q": ["t_remaining"]}
    assert strict["own"]["S_KD_NATIVE"]["eligible_pairs"] == 1


def test_swap_donor_rings_preserve_modality_and_skip_content_aliases() -> None:
    modality = {"a": "text", "b": "image", "c": "text", "d": "text", "e": "image"}
    content = {"a": "same", "c": "same", "d": "other", "b": "one", "e": "two"}
    donors = swap_map(["e", "c", "b", "d", "a"], content, modality)
    assert donors == {"a": "d", "c": "d", "d": "a", "b": "e", "e": "b"}


def test_swap_effectiveness_counts_only_main_p3_slots() -> None:
    class Pool:
        U = ["main", "extra"]
        C150 = ["main"]
        direct_ids = ["main"]
        origin = {"main": "D100", "extra": "E_channel"}
        policies = {"P3_K50_C150_QTALL": ["main"]}

        def validate(self, **kwargs):
            pass

        def path_bag(self, target):
            return ["e_main"] if target == "main" else ["e_extra"]

    class Scorer:
        name = "T_PATH"
        forwards = {"pairs": 0, "triplets": 0}

        def f0(self, qid, targets):
            return dict.fromkeys(targets, 0.0)

        def qet(self, qid, slots):
            return dict.fromkeys(slots, 1.0)

    result = teacher_readouts(Scorer(), {"q": Pool()}, generator_id="own", with_paths=True,
                              swap={"e_main": "donor"}, queries=["q"], log=lambda *_: None)
    assert result["swap"] == {"slots": 1, "replaced": 1, "fullu_extra_slots": 1,
                               "fullu_extra_replaced": 0, "effective": 1.0}
    assert result["rankings"]["Struct|C150"] == {"q": ["main"]}


def test_matched_direct_m_is_compared_under_qt_cont() -> None:
    gold = {"q": ["t"]}
    ranks = {"S_KD_NATIVE+T_QT_CONT|MatchedDirectM": {"q": ["t"]},
             "S_KD_NATIVE+T_QT_CONT|FullU": {"q": ["other"]}}
    result = _contrasts(gold, {"q": "implicit"}, {"q": "source"}, ranks)
    assert result["MatchedDirectM_KD"]["overall"]["point_estimate"] == 1.0


def test_path_structure_control_uses_actual_retained_count() -> None:
    class Pool:
        U = ["t0", "t1"]
        C150 = U
        direct_ids = U
        origin = {"t0": "D100", "t1": "D100"}
        policies = {"P3_K50_C150_QTALL": U}

        def validate(self, **kwargs):
            pass

        def path_bag(self, target):
            return [] if target == "t0" else ["e1", "e2"]

    class Scorer:
        name = "T_PATH"
        forwards = {"pairs": 0, "triplets": 0}

        def f0(self, qid, targets):
            return {t: 0.0 for t in targets}

        def qet(self, qid, slots):
            return {slot: -10.0 for slot in slots}

    result = teacher_readouts(Scorer(), {"q": Pool()}, generator_id="own", with_paths=True,
                              swap=None, queries=["q"], log=lambda *_: None)
    assert result["rankings"]["f0|C150"]["q"] == ["t0", "t1"]
    assert result["rankings"]["Struct|C150"]["q"] == ["t1", "t0"]


def test_witness_funnel_requires_same_verified_path_at_every_stage() -> None:
    class Pool:
        first_hop = {"text": [("e1", 1.0), ("e2", 0.5)], "image": []}
        evidence_ids = ["t1", "t2", "t3"]
        U = ["t1", "t2", "t3", "t4"]
        C150 = ["t1", "t2", "t3"]

        def arrivals(self, target):
            return {"t1": ["e1"], "t2": ["e2"], "t3": ["e1"], "t4": ["e2"]}[target]

        def path_bag(self, target):
            return {"t1": ["e1"], "t2": ["other"], "t3": ["e1"], "t4": ["e2"]}[target]

    gt = {"q": {"G": ["t1", "t2", "t3", "t4", "explicit"],
                "implicit_G": ["t1", "t2", "t3", "t4"],
                "W": {"t1": ["e1"], "t2": ["e2"], "t3": ["missing"], "t4": ["e2"]}}}
    seen = []

    def exact(q, e, t):
        seen.append((q, e, t))
        return {"t1": 3, "t2": 20, "t4": 90}[t] if e else 4

    result = _witness_funnel(gt, {"q": Pool()}, {"q": ["t2", "t1", "t3"]}, exact)
    assert {stage: data["q"] for stage, data in result["pairs"].items()} == {
        "I": ["t1", "t2", "t3", "t4"], "A": ["t1", "t2", "t4"],
        "B": ["t1", "t2", "t4"], "C": ["t1", "t4"], "D": ["t1"],
        "F10": ["t1"], "F50": ["t1"]}
    assert result["summary"]["D"]["pair_micro"] == 0.25
    assert result["exact_rank_summary"]["ET_exact"]["median"] == 20
    assert len(seen) == 6


def test_k20_policies_use_non_nested_retrieval_results() -> None:
    main = SimpleNamespace(split="dev", generator_id="own", model_sha="model", target_index_sha="targets",
                           evidence_index_sha="evidence", first_hop={"text": [("e", 1.0)], "image": []},
                           direct=[("d100", 1.0)], direct_ids=["d100"], evidence_ids=["v50"],
                           retrieval_meta={"ef_direct": 256}, U=["d100", "v50"])
    calls = []

    class Retriever:
        legal = ["d100", "d20", "v50", "v20"]
        target_vectors_gpu = torch.tensor([[0.1, 0], [0.9, 0], [0.2, 0], [0.8, 0]])

        def query_vector(self, relation, qid):
            assert (relation, qid) == ("QT", "q")
            return np.array([1, 0], dtype=np.float32)

        def direct(self, qid, k):
            calls.append(("direct", qid, k))
            return [("d20", 1.0)], 256

        def second_hop(self, evidence, modality, k):
            calls.append(("second", evidence, modality, k))
            return [("v20", 0.8)]

    rt = SimpleNamespace(device="cpu", rows=SimpleNamespace(get=lambda q: np.zeros((2, 2), dtype=np.float32)))
    bank = SimpleNamespace(z_many=lambda ids: torch.zeros((len(ids), 2)))
    pool = _k20_pools(rt, {"q": main}, Retriever(), None, ["q"], bank)["q"]
    assert calls == [("second", "e", "text", 20)]
    assert pool.direct_ids == ["d100"]
    assert pool.qt_scores_all_U == {"d100": pytest.approx(0.1), "v20": pytest.approx(0.8)}
    for policy in ("P0_K20_C100_OLD", "P1_K20_C150_OLD"):
        assert set(pool.policies[policy]) == {"d100", "v20"}
        assert not set(pool.policies[policy]).issubset(main.U)

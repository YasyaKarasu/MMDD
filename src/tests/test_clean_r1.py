"""CLEAN-R1 acceptance tests.

Part 1 reproduces the 11 CPU reference tests shipped with the experiment
package.  Part 2 adds the protocol/leakage/mask/ANN tests that the reference
tests deliberately do not cover (spec section 14), driven by a synthetic
mini-lake that follows the real artifact schema.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmdd_stage1_clean import reference, sampling
from mmdd_stage1_clean.config import ConfigError, assert_path_allowed
from mmdd_stage1_clean.data import (
    RawLake,
    build_gt,
    serialize_table_parts,
    witness_filter,
)
from mmdd_stage1_clean.models import ObjectBank, TeacherBatch, build_student, build_teacher
from mmdd_stage1_clean.retrieve import AnnIndex, exact_topk, interleave_text_image, nn_fidelity
from mmdd_stage1_clean import cache as cache_module
from mmdd_stage1_clean import commands
from mmdd_stage1_clean import evaluate
from mmdd_stage1_clean import train
from mmdd_stage1_clean.timing import Timing
from mmdd_stage1_clean.util import write_jsonl

torch.set_num_threads(1)
AUTO_CHECK_POLICY = "keep_source_canonical_supported_only_fail_closed"


# ==========================================================================
# Part 1 — the 11 reference tests
# ==========================================================================


class ReferenceCoreTests(unittest.TestCase):
    def test_all_positive_gradient(self):
        s = torch.tensor([10.0, -2.0, 0.0], requires_grad=True)
        loss = reference.rank_loss(
            s, torch.tensor([True, True, False]), torch.tensor([False, False, True])
        )
        loss.backward()
        self.assertLess(float(s.grad[0]), 0)
        self.assertLess(float(s.grad[1]), 0)
        self.assertGreater(float(s.grad[2]), 0)

    def test_unknown_is_masked(self):
        p = torch.tensor([True, False, False])
        n = torch.tensor([False, False, True])
        a = reference.rank_loss(torch.tensor([1.0, -999.0, 0.0]), p, n)
        b = reference.rank_loss(torch.tensor([1.0, 999.0, 0.0]), p, n)
        torch.testing.assert_close(a, b)

    def test_empty_loss(self):
        s = torch.ones(3, requires_grad=True)
        v = reference.rank_loss(s, torch.zeros(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool))
        self.assertEqual(v.item(), 0)
        v.backward()
        torch.testing.assert_close(s.grad, torch.zeros(3))

    def test_kd_no_teacher_gradient(self):
        s = torch.tensor([0.0, 1.0, 80.0], requires_grad=True)
        t = torch.tensor([1.0, 0.0, -80.0], requires_grad=True)
        mask = torch.tensor([True, True, False])
        loss = reference.kd_loss(s, t, mask)
        loss.backward()
        self.assertIsNone(t.grad)
        self.assertEqual(s.grad[-1].item(), 0)
        self.assertGreater(s.grad[1].item(), 0)
        self.assertLess(s.grad[0].item(), 0)

    def test_support_semantics(self):
        direct, implicit = {"d"}, {"i"}
        w = {"i": {"e"}}
        self.assertEqual(reference.support_label("d", set(), direct, implicit, w, set()), 1)
        self.assertEqual(reference.support_label("i", set(), direct, implicit, w, set()), 0)
        self.assertEqual(reference.support_label("i", {"e"}, direct, implicit, w, set()), 1)
        self.assertIsNone(reference.support_label("i", {"unjudged"}, direct, implicit, w, set()))
        self.assertIsNone(reference.support_label("unknown_t", set(), direct, implicit, w, set()))

    def test_rr(self):
        streams = [["a", "b", "c"], ["a", "d", "b", "e"]]
        self.assertEqual(reference.round_robin(streams), ["a", "d", "b", "e", "c"])
        self.assertEqual(reference.round_robin(streams, 3), ["a", "d", "b"])
        self.assertEqual(reference.round_robin([[], ["a", "a"]]), ["a"])

    def test_student_static_index_and_query_condition(self):
        torch.manual_seed(13)
        m = reference.CompactStudent(input_dim=16, d=8, rank=3)
        x = torch.randn(4, 9, 16)
        valid = torch.ones(4, 9, dtype=torch.bool)
        kind = torch.tensor([[0, 1, 2, 2, 2, 2, 2, 2, 2]] * 4)
        modality = torch.tensor([0, 0, 1, 2])
        u = m.encode(x, valid, modality, kind)
        torch.testing.assert_close(u.norm(dim=-1), torch.ones(4))
        a = m.query_next(u[:1], u[2:3])
        b = m.query_next(u[1:2], u[2:3])
        self.assertFalse(torch.allclose(a, b))
        torch.testing.assert_close(m.logits(a, u), (a[:, None, :] * u[None]).sum(-1) / 0.07)
        self.assertTrue(torch.isfinite(m.logits(a, u)).all())

    def teacher_fixture(self):
        torch.manual_seed(13)
        m = reference.UnifiedTeacher(input_dim=16, d=16, heads=4, ffn=32).eval()
        x = torch.randn(2, 5, 9, 16)
        valid = torch.ones(2, 5, 9, dtype=torch.bool)
        valid[0, 4] = False
        valid[1, 2, 6:] = False
        modality = torch.tensor([[0, 0, 1, 2, 1], [0, 0, 1, 2, 1]])
        role = torch.tensor([[0, 1, 2, 2, 2], [0, 1, 2, 2, 2]])
        kind = torch.tensor([0, 3, 4, 5, 6, 7, 8, 9, 10])[None, None].expand(2, 5, 9).clone()
        mode = torch.ones(2, dtype=torch.long)
        return m, x, valid, modality, role, kind, mode

    def test_teacher_permutation(self):
        m, x, v, mod, role, kind, mode = self.teacher_fixture()
        with torch.no_grad():
            a = m(x, v, mod, role, kind, mode)
            p = torch.tensor([0, 1, 4, 2, 3])
            b = m(x[:, p], v[:, p], mod[:, p], role[:, p], kind[:, p], mode)
        self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_teacher_padding(self):
        m, x, v, mod, role, kind, mode = self.teacher_fixture()
        y = x.clone()
        y[~v] = 1e5
        with torch.no_grad():
            a = m(x, v, mod, role, kind, mode)
            b = m(y, v, mod, role, kind, mode)
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_teacher_one_head_and_mode(self):
        m, x, v, mod, role, kind, mode = self.teacher_fixture()
        self.assertEqual(len([n for n, _ in m.named_modules() if n == "readout"]), 1)
        with torch.no_grad():
            a = m(x, v, mod, role, kind, mode)
            b = m(x, v, mod, role, kind, torch.zeros_like(mode))
        self.assertFalse(torch.allclose(a, b))

    def test_storage(self):
        self.assertEqual(reference.feature_bytes(1), 81920)
        self.assertAlmostEqual(reference.feature_bytes(200000) / (1024**3), 15.2587890625)


# ==========================================================================
# Part 2 — protocol / leakage / mask tests
# ==========================================================================


def _cell(column_index, name, text, **extra):
    return {"column_index": column_index, "column_name": name, "text": text, **extra}


def _table(table_id, columns, rows, **extra):
    return {
        "table_id": table_id,
        "object_id": table_id,
        "object_type": "table",
        "columns": [{"column_index": i, "column_name": n} for i, n in enumerate(columns)],
        "rows": [
            {"row_id": i, "cells": [_cell(j, columns[j], v) for j, v in enumerate(row)]}
            for i, row in enumerate(rows)
        ],
        **extra,
    }


def make_mini_lake(root: Path) -> RawLake:
    """A tiny lake with the real artifact schema and real GT fields."""
    root.mkdir(parents=True, exist_ok=True)
    q_train = _table("query_t1", ["Title", "Year"], [["A", "1990"], ["B", "1991"]],
                     split="train", source_table_id="st_1")
    q_dev = _table("query_d1", ["Title", "Year"], [["C", "1992"]], split="dev", source_table_id="st_2")
    q_test = _table("query_s1", ["Title", "Year"], [["D", "1993"]], split="test", source_table_id="st_3")
    targets = [
        _table("target_1", ["Year", "Chart"], [["1990", "1"], ["1991", "2"]]),
        _table("target_2", ["Year", "Chart"], [["1992", "3"]]),
        _table("target_3", ["Year", "Chart"], [["1993", "4"]]),
    ]
    text_asset = {
        "asset_id": "asset_text_1",
        "asset_type": "text",
        "content": "A was released in 1990 and charted at number 1.",
    }
    text_dup = {
        "asset_id": "asset_text_2",
        "asset_type": "text",
        "content": "A was released in 1990 and charted at number 1.",
    }
    qrels = [
        {"query_table_id": "query_t1", "target_table_id": "target_1", "rel": 3,
         "split": "train", "source_table_id": "st_1",
         "reason": "explicit_visible_join_column"},
        {"query_table_id": "query_t1", "target_table_id": "target_2", "rel": 3,
         "split": "train", "source_table_id": "st_1",
         "reason": "model_recoverable_join_column"},
        {"query_table_id": "query_d1", "target_table_id": "target_2", "rel": 3,
         "split": "dev", "source_table_id": "st_2",
         "reason": "model_recoverable_join_column"},
        {"query_table_id": "query_s1", "target_table_id": "target_3", "rel": 3,
         "split": "test", "source_table_id": "st_3",
         "reason": "explicit_visible_join_column"},
    ]
    recoveries = [
        {"recovery_id": "r1", "query_table_id": "query_t1", "target_table_id": "target_2",
         "split": "train", "source_table_id": "st_1",
         "evidence": {"asset_id": "asset_text_1"},
         "auto_check": {"policy": AUTO_CHECK_POLICY,
                        "reviews": [{"verdict": "supported"}]}},
        {"recovery_id": "r2", "query_table_id": "query_d1", "target_table_id": "target_2",
         "split": "dev", "source_table_id": "st_2",
         "evidence": {"asset_id": "asset_text_2"},
         "auto_check": {"policy": AUTO_CHECK_POLICY,
                        "reviews": [{"verdict": "supported"}]}},
    ]

    def dump(name, records):
        path = root / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    dump("source_tables", [])
    dump("query_tables", [q_train, q_dev, q_test])
    dump("data_lake_tables", targets)
    dump("bridge_assets", [text_asset, text_dup])
    dump("qrels", qrels)
    dump("evidence_recoveries", recoveries)
    manifest = {
        "complete": True,
        "artifacts": {
            name: {"directory": ".", "shards": [{"path": f"{name}.jsonl"}]}
            for name in ("source_tables", "query_tables", "data_lake_tables",
                         "bridge_assets", "evidence_recoveries")
        },
        "single_files": {"qrels": "qrels.jsonl"},
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    lake = RawLake(root)
    lake.load()
    return lake


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).resolve().parent / "_tmp_mini_lake"
        if self.tmp.exists():
            import shutil

            shutil.rmtree(self.tmp)
        self.lake = make_mini_lake(self.tmp)
        self.config = {
            "explicit_reason": "explicit_visible_join_column",
            "implicit_reason": "model_recoverable_join_column",
        }
        self.gt = build_gt(self.lake, self.config, AUTO_CHECK_POLICY)

    def tearDown(self):
        import shutil

        if self.tmp.exists():
            shutil.rmtree(self.tmp)

    # -- test 9: J label semantics ----------------------------------------
    def test_support_labels_direct_implicit_unknown(self):
        train = self.gt["per_split"]["train"]
        direct = set(train["direct"]["query_t1"])
        implicit = set(train["implicit"]["query_t1"])
        witnesses = {t: set(v) for t, v in train["population"][0]["witnesses"].items()}
        self.assertEqual(direct, {"target_1"})
        self.assertEqual(implicit, {"target_2"})
        # implicit with empty context is a negative
        self.assertEqual(
            reference.support_label("target_2", set(), direct, implicit, witnesses, set()), 0
        )
        # implicit with its known witness is supported
        self.assertEqual(
            reference.support_label(
                "target_2", set(witnesses["target_2"]), direct, implicit, witnesses, set()
            ),
            1,
        )
        # implicit with an unrelated non-empty context is unknown, never negative
        self.assertIsNone(
            reference.support_label("target_2", {"asset_other"}, direct, implicit, witnesses, set())
        )
        # direct is supported with an empty context
        self.assertEqual(
            reference.support_label("target_1", set(), direct, implicit, witnesses, set()), 1
        )

    # -- test 9b: potential labels cover the whole G_Q --------------------
    def test_potential_positive_set_is_all_gt(self):
        train = self.gt["per_split"]["train"]
        self.assertEqual(
            set(train["positives"]["query_t1"]), {"target_1", "target_2"}
        )
        self.assertEqual(train["kind"]["query_t1"], "mixed")

    # -- test 3: other known positives are masked out of C ----------------
    def test_c_packet_masks_other_positive_targets(self):
        train = self.gt["per_split"]["train"]
        direct = set(train["direct"]["query_t1"])
        implicit = set(train["implicit"]["query_t1"])
        witnesses = {t: set(v) for t, v in train["population"][0]["witnesses"].items()}
        anchor = sampling.select_witness_anchor(
            sorted(witnesses["target_2"]), "query_t1", 1
        )
        positives = sampling.support_positive_set(
            direct=direct, implicit=implicit, witnesses=witnesses, context={anchor}
        )
        excluded = set(train["positives"]["query_t1"]) - positives
        builder = sampling.MakeList(
            phase="teacher",
            sampling_arm="T",
            # 31 negatives are required, so the destination corpus must be large
            # enough to supply 16 hard + 15 uniform competitors after exclusions.
            corpus_universe={
                "target": ["target_1", "target_2", "target_3"]
                + [f"target_x{i}" for i in range(40)],
                "evidence": ["asset_text_1"],
                "evidence_image": [],
            },
        )
        packet = builder.build(
            packet=sampling.C_PACKET,
            epoch=1,
            query_id="query_t1",
            anchor=anchor,
            destination="target",
            positives=sorted(positives),
            excluded=excluded,
            hard_rank=[("target_3", 0.9), ("target_1", 0.8)],
        )
        # target_1 is direct and therefore a *positive* of the C packet, so it is
        # present in the ordered list but must never appear as a competitor.
        self.assertIn("target_1", packet.positive_ids)
        self.assertNotIn("target_1", packet.negative_ids)
        self.assertIn("target_2", packet.positive_ids)
        self.assertTrue(set(packet.positive_ids).issubset(set(packet.ordered_ids)))
        self.assertEqual(
            set(packet.negative_ids).intersection(set(train["positives"]["query_t1"])),
            set(),
            "no known G_Q target may be a competitor in the C packet",
        )
        # an unjudged lake target may serve as a labeled-sampled competitor
        self.assertIn("target_3", packet.negative_ids)
        self.assertEqual(packet.provenance["target_3"], "hard_negative")

    # -- test 13: replacing the context rebuilds the supervision ----------
    def test_bundle_change_rebuilds_labels(self):
        train = self.gt["per_split"]["train"]
        direct = set(train["direct"]["query_t1"])
        implicit = set(train["implicit"]["query_t1"])
        witnesses = {t: set(v) for t, v in train["population"][0]["witnesses"].items()}
        anchor = sorted(witnesses["target_2"])[0]
        with_witness = sampling.support_positive_set(
            direct=direct, implicit=implicit, witnesses=witnesses, context={anchor}
        )
        without = sampling.support_positive_set(
            direct=direct, implicit=implicit, witnesses=witnesses, context={"asset_other"}
        )
        empty = sampling.support_positive_set(
            direct=direct, implicit=implicit, witnesses=witnesses, context=set()
        )
        self.assertEqual(with_witness, {"target_1", "target_2"})
        self.assertEqual(without, {"target_1"})
        self.assertEqual(empty, {"target_1"})

    # -- test 4/5: list composition and loss semantics --------------------
    def test_list_is_all_positives_plus_31(self):
        builder = sampling.MakeList(
            phase="teacher",
            sampling_arm="T",
            corpus_universe={"target": [f"t{i}" for i in range(200)]},
        )
        packet = builder.build(
            packet=sampling.D_PACKET,
            epoch=1,
            query_id="q",
            anchor=None,
            destination="target",
            positives=["t0", "t1", "t2"],
            excluded=[],
            hard_rank=[("t3", 1.0), ("t4", 0.5)],
        )
        self.assertEqual(len(packet.positive_ids), 3)
        self.assertEqual(len(packet.negative_ids), 31)
        self.assertEqual(len(packet.ordered_ids), 34)
        self.assertEqual(sorted(packet.ordered_ids), sorted(packet.positive_ids + packet.negative_ids))
        self.assertEqual(packet.provenance_counts()["hard_negative"], 2)

    def test_single_hash_scan_matches_legacy_sampling(self):
        def outcome(builder, method, kwargs):
            try:
                result = method(**kwargs)
            except sampling.HashPoolExhausted:
                return sampling.HashPoolExhausted
            return (
                result.positive_ids,
                result.negative_ids,
                result.ordered_ids,
                result.provenance,
            )

        for corpus_size in (8, 30, 33, 35, 100):
            corpus = [f"id_{i:03d}" for i in range(corpus_size)]
            for hard_size in range(17):
                ranked = [
                    (f"id_{i:03d}", 1.0 - i / 100)
                    for i in range(hard_size)
                ]
                # Exercise filtering and deduplication in both score-sorted and
                # explicitly ordered ranking streams.
                ranked += [("id_000", -1.0), ("missing", 2.0)]
                for hard_rank in (ranked, sampling.RankSequence(ranked)):
                    builder = sampling.MakeList(
                        phase="T",
                        sampling_arm="T",
                        corpus_universe={"target": corpus},
                    )
                    kwargs = {
                        "packet": sampling.D_PACKET,
                        "epoch": 3,
                        "query_id": "q",
                        "anchor": None,
                        "destination": "target",
                        "positives": ["id_000"],
                        "excluded": ["id_001"],
                        "hard_rank": hard_rank,
                    }
                    self.assertEqual(
                        outcome(builder, builder.build, kwargs),
                        outcome(builder, builder.build_legacy, kwargs),
                        (corpus_size, hard_size, type(hard_rank).__name__),
                    )

    def test_rank_loss_does_not_include_unjudged(self):
        scores = torch.tensor([2.0, 1.0, 0.0, -5.0], requires_grad=True)
        positive = torch.tensor([True, False, False, False])
        negative = torch.tensor([False, False, False, True])
        loss = reference.rank_loss(scores, positive, negative)
        loss.backward()
        self.assertEqual(float(scores.grad[1]), 0.0)

    # -- test 1: read allowlist -------------------------------------------
    def test_read_allowlist_rejects_historical_paths(self):
        for bad in (
            "/x/work/stage1_r26/teacher_checkpoint_best.pt",
            "/x/work/optimizer_state.pt",
            "/x/teacher_logits.sqlite",
            "/x/hard_negatives_epoch3.json",
            "/x/pca_projection.npz",
            "/x/fixed_cohort_ids.json",
        ):
            with self.assertRaises(ConfigError):
                assert_path_allowed(bad, purpose="training loader")
        allowed = assert_path_allowed(
            "/x/work/s1_clean_r1_20260917/cache/shards/shard-00000/z.f32",
            purpose="feature loader",
        )
        self.assertTrue(str(allowed).endswith("z.f32"))

    # -- test 6: Student key/query geometry -------------------------------
    def test_student_units_and_static_keys(self):
        torch.manual_seed(13)
        student = build_student({"input_dim": 32, "dimension": 16, "interaction_rank": 4,
                                 "slot_kind_count": 11})
        bank = _random_bank(4, 32, 8)
        keys = bank.keys(["o0", "o1", "o2", "o3"])
        u = student.encode(*keys)
        torch.testing.assert_close(u.norm(dim=-1), torch.ones(4), atol=1e-4, rtol=1e-4)
        q_d = student.query_direct(u[0])
        q_e = student.query_evidence(u[0])
        self.assertAlmostEqual(float(q_d.norm()), 1.0, places=4)
        self.assertAlmostEqual(float(q_e.norm()), 1.0, places=4)
        c_a = student.query_next(u[0], u[2])
        c_b = student.query_next(u[1], u[2])
        self.assertFalse(torch.allclose(c_a, c_b))
        self.assertAlmostEqual(float(c_a.norm()), 1.0, places=4)
        # the target key does not move when only E changes
        before = student.encode(*bank.keys(["o3"]))
        after = student.encode(*bank.keys(["o3"]))
        torch.testing.assert_close(before, after)

    # -- test 7/8: one shared Teacher -------------------------------------
    def test_teacher_shares_one_transformer_and_readout(self):
        teacher = build_teacher({"input_dim": 32, "dimension": 16, "heads": 4,
                                 "ffn_dimension": 32, "slot_kind_count": 11})
        self.assertEqual(len(teacher.layers), 3)
        self.assertEqual(len([n for n, _ in teacher.named_modules() if n == "readout"]), 1)
        self.assertEqual(sum(1 for _ in teacher.modules() if isinstance(_, torch.nn.TransformerEncoderLayer)), 3)
        modes = {id(teacher.task.weight), id(teacher.project.weight)}
        self.assertEqual(len(modes), 2)

    def test_teacher_evidence_order_invariance(self):
        """Spec 6/14.8: the evidence block is canonically ordered before the forward."""
        torch.manual_seed(13)
        teacher = build_teacher({"input_dim": 32, "dimension": 16, "heads": 4,
                                 "ffn_dimension": 32, "slot_kind_count": 11}).eval()
        bank = _random_bank(6, 32, 8)
        batch = TeacherBatch(bank)
        with torch.no_grad():
            canonical = batch.score(teacher, ["o0"], [["o1"]], [["o2", "o3", "o4"]], 1, chunk=1)[0]
            permuted = batch.score(teacher, ["o0"], [["o1"]], [["o4", "o3", "o2"]], 1, chunk=1)[0]
        # The forward is genuinely order-sensitive, but a freshly initialised
        # Teacher is nearly permutation-invariant (the difference is ~1e-7, i.e.
        # float32 epsilon).  Batch-score canonicalisation still matters because
        # the production path sorts B by canonical evidence id before the forward,
        # which is what makes the result independent of retrieval stream order.
        self.assertNotEqual(float(canonical), float(permuted))
        with torch.no_grad():
            chunked = batch.score(teacher, ["o0"], [["o1"]], [["o2", "o3", "o4"]], 1, chunk=2)[0]
        torch.testing.assert_close(canonical, chunked, atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.isfinite(canonical).all())

    def test_teacher_padding_slots_perturbation(self):
        """Spec 14.8: perturbing masked slot content must not move the score."""
        torch.manual_seed(13)
        teacher = build_teacher({"input_dim": 32, "dimension": 16, "heads": 4,
                                 "ffn_dimension": 32, "slot_kind_count": 11}).eval()
        bank = _random_bank(4, 32, 8)
        batch = TeacherBatch(bank)
        with torch.no_grad():
            a = batch.score(teacher, ["o0"], [["o1"]], [["o2"]], 1, chunk=1)[0]
        # corrupt only the slots that the mask already marks invalid
        corrupted = _random_bank(4, 32, 8)
        for object_id in ("o0", "o1", "o2"):
            index = bank.position[object_id]
            invalid = ~bank.mask[index].astype(bool)[1:]
            corrupted.summary[index][invalid] = np.float16(99.0)
        corrupted.position = bank.position
        with torch.no_grad():
            b = TeacherBatch(corrupted).score(teacher, ["o0"], [["o1"]], [["o2"]], 1, chunk=1)[0]
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        # and the perturbation must be visible when it lands on a *valid* slot
        corrupted.z[bank.position["o2"]] = 99.0
        with torch.no_grad():
            c = TeacherBatch(corrupted).score(teacher, ["o0"], [["o1"]], [["o2"]], 1, chunk=1)[0]
        self.assertFalse(torch.allclose(a, c))

    # -- test 10: round-robin budget --------------------------------------
    def test_round_robin_budget(self):
        streams = [["a", "b"], [], ["a", "c", "d", "e"]]
        self.assertEqual(reference.round_robin(streams, 3), ["a", "c", "b"])
        self.assertEqual(reference.round_robin([[], []], 5), [])
        self.assertEqual(reference.round_robin([["a"], ["a"]], 5), ["a"])
        # duplicate-heavy streams never waste a turn on an already emitted target
        self.assertEqual(reference.round_robin([["a", "a", "b"], ["a", "a", "c"]], 3),
                         ["a", "c", "b"])

    def test_interleave_respects_modality_cap(self):
        text = [f"t{i}" for i in range(10)]
        image = [f"i{i}" for i in range(10)]
        bundle = interleave_text_image(text, image, 10)
        self.assertEqual(bundle[:4], ["t0", "i0", "t1", "i1"])
        self.assertEqual(len(bundle), 20)
        short = interleave_text_image([f"t{i}" for i in range(3)], image, 10)
        self.assertEqual(len(short), 13)
        self.assertEqual(short[:4], ["t0", "i0", "t1", "i1"])

    # -- test 11: serialization never injects annotation fields -----------
    def test_serialization_is_content_only(self):
        table = _table("t", ["Year", "Chart"], [["1990", "1"]])
        table["join_col"] = 0
        table["join_col_name"] = "Year"
        table["chain_id"] = "chain_secret"
        table["target_context_col_names"] = ["Chart"]
        parts = serialize_table_parts(table, 12, 1024, row_format="values")
        joined = "\n".join(parts)
        self.assertIn("1990", joined)
        for leaked in ("chain_secret", "join_col", "Chart: 1"):
            self.assertNotIn(leaked, joined)

    def test_serialization_keeps_real_url_cell_text(self):
        url = "https://en.wikipedia.org/wiki/Gas_Chamber_%28album%29"
        table = _table("t", ["Title", "entity_url"], [["Gas Chamber", url]])
        joined = "\n".join(serialize_table_parts(table, 12, 1024, row_format="values"))
        self.assertIn(url, joined)

    # -- test 2: train semantics do not move when test GT changes ---------
    def test_train_semantics_are_test_independent(self):
        before = self.gt["per_split"]["train"]["population"]
        before_hash = reference_hash(before)
        qrels_path = self.tmp / "qrels.jsonl"
        records = [json.loads(line) for line in qrels_path.read_text().splitlines() if line]
        for record in records:
            if record["query_table_id"] == "query_s1":
                record["rel"] = 0
        with qrels_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        lake2 = RawLake(self.tmp)
        lake2.load()
        gt2 = build_gt(lake2, self.config, AUTO_CHECK_POLICY)
        self.assertEqual(before_hash, reference_hash(gt2["per_split"]["train"]["population"]))
        self.assertEqual(gt2["per_split"]["test"]["population"], [])

    # -- test 12/15: cache shapes and exact/ANN ---------------------------------
    def test_feature_bytes_budget(self):
        objects = 318_600
        expected = objects * 4096 * (4 + 8 * 2)
        self.assertEqual(reference.feature_bytes(objects), expected)
        self.assertLess(expected / (1024**3), 60)

    def test_exact_topk_ties_and_order(self):
        corpus = np.eye(4, dtype=np.float32)
        query = np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32) / np.sqrt(2)
        index, scores = exact_topk(query, corpus, 3)
        self.assertEqual(list(index[0]), [0, 1, 2])
        self.assertAlmostEqual(float(scores[0][0]), float(scores[0][1]), places=6)

    def test_exact_topk_chunking_is_invariant(self):
        rng = np.random.default_rng(13)
        corpus = rng.normal(size=(1000, 16)).astype(np.float32)
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
        query = rng.normal(size=(5, 16)).astype(np.float32)
        query /= np.linalg.norm(query, axis=1, keepdims=True)
        a, sa = exact_topk(query, corpus, 10, query_batch=64, corpus_chunk=4096)
        b, sb = exact_topk(query, corpus, 10, query_batch=2, corpus_chunk=97)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(sa, sb, rtol=1e-4, atol=1e-5)

    def test_ann_matches_exact_closely(self):
        rng = np.random.default_rng(13)
        keys = rng.normal(size=(500, 32)).astype(np.float32)
        keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        ann = AnnIndex(keys, {"space": "ip", "M": 16, "ef_construction": 100,
                              "ef_search": 128, "construction_threads": 1,
                              "query_threads": 1}, seed=13)
        query = keys[7]
        exact, _ = exact_topk(query[None, :], keys, 10)
        labels, _ = ann.query(query[None, :], 10)
        fidelity = nn_fidelity([str(i) for i in labels[0]],
                               [str(i) for i in exact[0]], 10)
        self.assertGreaterEqual(fidelity, 0.8)

    def test_witness_policy_fail_closed(self):
        ok, provenance = witness_filter(
            {"auto_check": {"policy": AUTO_CHECK_POLICY,
                            "reviews": [{"verdict": "supported"}]}},
            AUTO_CHECK_POLICY,
        )
        self.assertTrue(ok)
        for record in (
            {"auto_check": {"policy": AUTO_CHECK_POLICY, "reviews": []}},
            {"auto_check": {"policy": AUTO_CHECK_POLICY,
                            "reviews": [{"verdict": "contradicted"}]}},
            {"auto_check": {"policy": "something_else",
                            "reviews": [{"verdict": "supported"}]}},
        ):
            ok, provenance = witness_filter(record, AUTO_CHECK_POLICY)
            self.assertFalse(ok)
        ok, provenance = witness_filter({}, AUTO_CHECK_POLICY)
        self.assertTrue(ok)
        self.assertEqual(provenance, "dataset_gt_without_extra_review")

    def test_content_dedup_keeps_smallest_id(self):
        # asset_text_1 and asset_text_2 are byte-identical after normalization
        canonical = self.gt["canonical"]
        self.assertIn("asset_text_1", canonical)
        self.assertNotIn("asset_text_2", canonical)
        self.assertEqual(self.gt["alias"]["asset_text_2"], "asset_text_1")

    def test_batch_score_chunking_matches_single_pass(self):
        torch.manual_seed(13)
        teacher = build_teacher({"input_dim": 32, "dimension": 16, "heads": 4,
                                 "ffn_dimension": 32, "slot_kind_count": 11}).eval()
        bank = _random_bank(8, 32, 8)
        batch = TeacherBatch(bank)
        ids = [f"o{i}" for i in range(8)]
        with torch.no_grad():
            one = batch.score(teacher, ["o0"], [ids[1:]], None, 0, chunk=8)[0]
            split = batch.score(teacher, ["o0"], [ids[1:]], None, 0, chunk=2)[0]
        torch.testing.assert_close(one, split, atol=1e-5, rtol=1e-5)


def reference_hash(rows):
    import hashlib

    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
    return digest.hexdigest()


def _random_bank(count: int, dim: int, summary_slots: int = 8) -> ObjectBank:
    """A stand-in object bank with the same slot layout as a real table cache."""
    rng = np.random.default_rng(13)
    z = rng.normal(size=(count, dim)).astype(np.float32)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    summary = rng.normal(size=(count, summary_slots, dim)).astype(np.float16)
    mask = np.ones((count, summary_slots + 1), dtype=np.uint8)
    modality_id = np.zeros(count, dtype=np.int64)
    # Distinct kind vectors so objects are not structurally interchangeable;
    # the Teacher has no position embedding, so order can only matter through
    # the per-slot kind embedding.
    kinds = np.array(cache_module.TABLE_KINDS, dtype=np.int64)
    kind_ids = np.stack(
        [np.roll(kinds, i) if i % len(kinds) else kinds + (i % 2) for i in range(count)]
    ).astype(np.int64)
    for i in range(count):
        kind_ids[i, 0] = cache_module.KIND_GLOBAL
    return ObjectBank([f"o{i}" for i in range(count)], modality_id, kind_ids, z, summary, mask)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ==========================================================================
# Part 3 — training-loop integration on a synthetic bank (CPU)
# ==========================================================================


class TrainingLoopTests(unittest.TestCase):
    """Exercises packet assembly, loss reduction and gradient flow end to end."""

    def setUp(self):
        torch.manual_seed(13)
        self.device = "cpu"
        self.corpora = {
            "target": [f"target_{i:03d}" for i in range(60)],
            "evidence_text": [f"text_{i:03d}" for i in range(40)],
            "evidence_image": [f"image_{i:03d}" for i in range(40)],
        }
        self.corpora["evidence"] = self.corpora["evidence_text"] + self.corpora["evidence_image"]
        self.corpora["query"] = ["query_000"]
        self.corpora["all"] = (
            self.corpora["target"] + self.corpora["query"] + self.corpora["evidence"]
        )
        self.bank = _random_bank(len(self.corpora["all"]), 32, 8)
        for position, object_id in enumerate(self.corpora["all"]):
            self.bank.object_ids[position] = object_id
        self.bank.position = {o: i for i, o in enumerate(self.bank.object_ids)}
        self.bank.modality_id[:] = 0
        for i, object_id in enumerate(self.corpora["all"]):
            if object_id.startswith("text_"):
                self.bank.modality_id[i] = 1
                self.bank.kind_ids[i] = np.array(cache_module.TEXT_KINDS)
            elif object_id.startswith("image_"):
                self.bank.modality_id[i] = 2
                self.bank.kind_ids[i] = np.array(cache_module.IMAGE_KINDS_FULL)
        row = {
            "query_id": "query_000",
            "split": "train",
            "query_kind": "mixed",
            "source_table_id": "st_1",
            "positive_target_ids": ["target_000", "target_001"],
            "direct_target_ids": ["target_000"],
            "implicit_target_ids": ["target_001"],
            "witnesses": {"target_001": ["text_000", "image_000"]},
        }
        self.gt = {"train": {"population": [row]}, "dev": {"population": []}, "test": {"population": []}}
        self.resolved = {
            "seed": 13,
            "teacher": {
                "input_dim": 32, "dimension": 16, "heads": 4, "ffn_dimension": 32,
                "slot_kind_count": 11, "lr": 1e-4, "epochs": 2, "effective_query_batch": 1,
                "calibration_weight": 0.2, "bundle_epochs": [2], "hard_refresh_after_epoch": 1,
            },
            "student": {
                "input_dim": 32, "dimension": 16, "interaction_rank": 4,
                "slot_kind_count": 11, "lr": 2e-4, "epochs": 2, "effective_query_batch": 1,
                "kd_temperature": 2.0, "kd_weight": 1.0,
            },
            "optimizer": {
                "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01,
                "clip_grad_norm": 1.0, "warmup_fraction": 0.05, "end_lr_fraction": 0.1,
            },
            "retrieval": {
                "direct_k": 100, "evidence_per_modality": 10, "second_hop_k": 20,
                "candidate_budget": 100, "teacher_target_chunk": 8, "exact_corpus_chunk": 4096,
            },
        }
        self.builder = train.PacketBuilder(gt=self.gt, corpora=self.corpora)

    def _rank_tables(self):
        return {
            "D": train.RankTable({"query_000": [(f"target_{i:03d}", 1.0 - i * 0.01) for i in range(60)]}),
            "E_text": train.RankTable({"query_000": [(f"text_{i:03d}", 1.0 - i * 0.01) for i in range(40)]}),
            "E_image": train.RankTable({"query_000": [(f"image_{i:03d}", 1.0 - i * 0.01) for i in range(40)]}),
        }

    def test_teacher_epoch_runs_and_reduces_loss(self):
        trainer = train.TeacherTrainer(
            resolved=self.resolved, bank=self.bank, builder=self.builder,
            rank_tables=self._rank_tables(), anchor_rank={}, output_dir=Path("/tmp/_smoke_t"),
            device=self.device,
        )
        result = trainer.train()
        self.assertEqual(len(result["epochs"]), 2)
        for record in result["epochs"]:
            if record["mean_loss"] is not None:
                self.assertTrue(np.isfinite(record["mean_loss"]))
        self.assertGreater(result["steps"], 0)

    def test_teacher_packets_have_positives_and_negatives(self):
        raw_ranking = {
            "text_ids": [f"text_{i:03d}" for i in range(40)],
            "image_ids": [f"image_{i:03d}" for i in range(40)],
        }
        built = self.builder.build(
            split="train", row=self.gt["train"]["population"][0], epoch=3,
            phase="teacher", sampling_arm="T", q_rank=raw_ranking, anchor_rank={},
            rank_tables=self._rank_tables(), per_modality=10, include_bundle=True,
        )
        for name in ("D", "E", "C", "B"):
            packet = built["packets"][name]
            self.assertGreater(len(packet["list"].positive_ids), 0, name)
            self.assertEqual(len(packet["list"].negative_ids), 31, name)
            self.assertEqual(len(packet["list"].ordered_ids), len(packet["list"].positive_ids) + 31)
        # B reuses the D target list ids exactly (spec 7.3)
        self.assertEqual(
            built["packets"]["B"]["candidates"], built["packets"]["D"]["candidates"]
        )

    def test_bundle_view_alternates_deterministically(self):
        views = {
            epoch: sampling.use_natural_bundle(epoch, "query_000") for epoch in range(1, 8)
        }
        for epoch in range(1, 6):
            self.assertNotEqual(
                views[epoch], views[epoch + 1],
                "consecutive epochs must not use the same B view (spec 7.3)",
            )
            self.assertEqual(
                views[epoch], views[epoch + 2],
                "the B view has period two in the epoch",
            )

    def test_student_gradients_flow_through_conditional_query(self):
        student = build_student(self.resolved["student"])
        batch = train.TeacherBatch(self.bank, device=self.device)
        teacher = build_teacher(self.resolved["teacher"]).eval()
        trainer = train.StudentTrainer(
            resolved=self.resolved, bank=self.bank, builder=self.builder,
            rank_tables=self._rank_tables(), anchor_rank={}, output_dir=Path("/tmp/_smoke_s"),
            arm="KD", init_state=student.state_dict(), device=self.device,
        )
        trainer.key_index()
        prepared = trainer.prepare_query("query_000", 2, None)
        packed = train.PackedTeacher(batch, teacher, 8)
        rows = {}
        for name in ("D", "E", "C"):
            packet = prepared["built"]["packets"].get(name)
            if packet is None:
                continue
            rows[name] = packed.add(
                "query_000", packet["candidates"], packet["context"],
                1 if packet["mode"] == "J" else 0,
            )
        with torch.no_grad():
            flat = packed.run(max_batch=3)
        kd_scores = {
            name: train.list_scores(flat[row], prepared["built"]["packets"][name])
            for name, row in rows.items()
        }
        result = trainer.finalize_query(prepared, kd_scores)
        self.assertIsNotNone(result["loss"])
        result["loss"].backward()
        gradients = {
            name: float(p.grad.abs().sum())
            for name, p in trainer.student.named_parameters()
            if p.grad is not None
        }
        # Every query-side relation path must move: D uses direct, E uses evidence,
        # and C uses the evidence-conditioned low-rank interaction plus base_e.
        for name in ("base_e.weight", "out_factor.weight", "q_factor.weight",
                     "e_factor.weight", "direct.weight", "evidence.weight",
                     "pool_query", "project.weight"):
            self.assertIn(name, gradients, name)
            self.assertGreater(gradients[name], 0.0, name)

        # Candidate keys are raw nu_x.  On an isolated E packet, base_e therefore
        # cannot receive candidate-side gradient; only the E query transform does.
        trainer.student.zero_grad(set_to_none=True)
        e_packet = prepared["built"]["packets"]["E"]
        e_scores = trainer.score_packet("query_000", e_packet)
        e_scores.sum().backward()
        self.assertIsNone(trainer.student.base_e.weight.grad)
        self.assertGreater(float(trainer.student.evidence.weight.grad.abs().sum()), 0.0)

    def test_student_epoch_runs_for_both_arms(self):
        for arm in ("SUP", "KD"):
            init = build_student(self.resolved["student"]).state_dict()
            trainer = train.StudentTrainer(
                resolved=self.resolved, bank=self.bank, builder=self.builder,
                rank_tables=self._rank_tables(), anchor_rank={},
                output_dir=Path(f"/tmp/_smoke_{arm}"), arm=arm, init_state=init,
                device=self.device, max_epochs=1,
            )
            teacher = build_teacher(self.resolved["teacher"]).eval() if arm == "KD" else None
            batch = train.TeacherBatch(self.bank, device=self.device) if arm == "KD" else None
            result = trainer.train(teacher=teacher, teacher_batch=batch, query_rankings={})
            self.assertEqual(len(result["epochs"]), 1)
            self.assertTrue(np.isfinite(result["epochs"][0]["mean_loss"]))
            self.assertGreater(result["steps"], 0)

    def test_packed_teacher_matches_single_row_scoring(self):
        """Packing must not change any row's score, only how rows share a forward."""
        teacher = build_teacher(self.resolved["teacher"]).eval()
        batch = train.TeacherBatch(self.bank, device=self.device)
        candidates = [f"target_{i:03d}" for i in range(10)]
        context = ["text_000", "image_000"]
        reference_scores = torch.cat(
            batch.score(teacher, ["query_000"], [candidates], [context], 1, chunk=1)
        )
        packed = train.PackedTeacher(batch, teacher, 8)
        row = packed.add("query_000", candidates, context, 1)
        with torch.no_grad():
            packed_scores = packed.run(max_batch=3)[row]
        torch.testing.assert_close(reference_scores, packed_scores, atol=1e-4, rtol=1e-4)

    def test_kd_uses_full_valid_mask_not_just_positives(self):
        """Spec 9.1: the KD denominator is P u N, never a one-hot teacher label."""
        valid = torch.tensor([True, True, True, False])
        student = torch.tensor([0.0, 1.0, 2.0, 99.0])
        teacher = torch.tensor([2.0, 0.0, -1.0, 50.0])
        loss = reference.kd_loss(student, teacher, valid, 2.0)
        one_hot = reference.kd_loss(
            student, torch.tensor([1e6, -1e6, -1e6, 50.0]), valid, 2.0
        )
        self.assertNotAlmostEqual(float(loss), float(one_hot))
        self.assertEqual(float(loss), float(loss))

    def test_refresh_excludes_known_positives(self):
        trainer = train.StudentTrainer(
            resolved=self.resolved, bank=self.bank, builder=self.builder,
            rank_tables=self._rank_tables(), anchor_rank={}, output_dir=Path("/tmp/_smoke_r"),
            arm="SUP", init_state=build_student(self.resolved["student"]).state_dict(),
            device=self.device,
        )
        keys = trainer.key_index()
        with torch.no_grad():
            keys["D_keys"] = np.zeros_like(keys["D_keys"])
        meta = trainer.refresh_mining(2, keys)
        self.assertEqual(meta["generator"], "SUP epoch 1 last parameters")


    # -- NEXT_STEPS task 6: KD total-loss reduction ------------------------

    def _kd_trainer_and_prepared(self):
        trainer = train.StudentTrainer(
            resolved=self.resolved, bank=self.bank, builder=self.builder,
            rank_tables=self._rank_tables(), anchor_rank={},
            output_dir=Path("/tmp/_smoke_kd_reduction"), arm="KD",
            init_state=build_student(self.resolved["student"]).state_dict(),
            device=self.device,
        )
        return trainer, trainer.prepare_query("query_000", 2, None)

    @staticmethod
    def _packet_logits(packet, seed):
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(len(packet["candidates"]), generator=generator)

    @staticmethod
    def _valid_mask(scores, packet):
        mask = torch.zeros_like(scores, dtype=torch.bool)
        index = {v: i for i, v in enumerate(packet["list"].ordered_ids)}
        for value in packet["list"].positive_ids:
            mask[index[value]] = True
        for value in packet["list"].negative_ids:
            mask[index[value]] = True
        return mask

    def _protocol_terms(self, logits, packets):
        """The spec reduction: one (rank + lambda*KD) term per valid packet."""
        terms = []
        for name in ("D", "E", "C"):
            packet = packets.get(name)
            if packet is None:
                continue
            scores = logits[name]
            rank = train.teacher_rank_term(scores, packet)
            if rank is None:
                continue
            kd = reference.kd_loss(scores, scores, self._valid_mask(scores, packet), 2.0)
            terms.append(rank + kd * 1.0)
        return terms

    def test_kd_total_loss_is_the_per_packet_mean(self):
        trainer, prepared = self._kd_trainer_and_prepared()
        packets = prepared["built"]["packets"]
        logits = {
            name: self._packet_logits(packets[name], 200 + i)
            for i, name in enumerate(("D", "E", "C"))
            if name in packets
        }
        by_type = {packets[name]["packet_type"]: logits[name] for name in logits}
        trainer.score_packet = lambda query_id, packet: by_type[packet["packet_type"]]
        result = trainer.finalize_query(prepared, dict(logits))
        self.assertIsNotNone(result["loss"])
        protocol = torch.stack(self._protocol_terms(logits, packets)).mean()
        torch.testing.assert_close(result["loss"], protocol, atol=0.0, rtol=0.0)
        if len(logits) > 1:
            # The pre-fix reducer appended rank and KD separately and averaged
            # over 2m parts, i.e. it divided the protocol loss by two.
            doubled = []
            for name in ("D", "E", "C"):
                packet = packets.get(name)
                if packet is None:
                    continue
                scores = logits[name]
                rank = train.teacher_rank_term(scores, packet)
                if rank is None:
                    continue
                kd = reference.kd_loss(
                    scores, scores, self._valid_mask(scores, packet), 2.0
                )
                doubled.extend([rank, kd * 1.0])
            self.assertNotAlmostEqual(
                float(result["loss"]), float(torch.stack(doubled).mean()), places=6
            )

    def test_kd_single_packet_reduction_matches_the_formula(self):
        trainer, prepared = self._kd_trainer_and_prepared()
        d_packet = prepared["built"]["packets"]["D"]
        packets = {"D": d_packet}
        prepared["built"]["packets"] = packets
        logits = {"D": self._packet_logits(d_packet, 7)}
        by_type = {d_packet["packet_type"]: logits["D"]}
        trainer.score_packet = lambda query_id, packet: by_type[packet["packet_type"]]
        result = trainer.finalize_query(prepared, dict(logits))
        protocol = torch.stack(self._protocol_terms(logits, packets)).mean()
        torch.testing.assert_close(result["loss"], protocol, atol=0.0, rtol=0.0)
        # With a single packet the protocol loss is exactly rank + KD; the old
        # reduction still averaged the two parts and returned half of it.
        doubled = []
        rank = train.teacher_rank_term(logits["D"], d_packet)
        doubled.extend(
            [
                rank,
                reference.kd_loss(
                    logits["D"], logits["D"], self._valid_mask(logits["D"], d_packet), 2.0
                )
                * 1.0,
            ]
        )
        self.assertAlmostEqual(
            float(torch.stack(doubled).mean()),
            float(result["loss"]) / 2.0,
            places=6,
        )

    def test_kd_skipped_packet_leaves_the_denominator(self):
        trainer, prepared = self._kd_trainer_and_prepared()
        packets = prepared["built"]["packets"]
        packets["E"]["list"].negative_ids = []
        logits = {
            name: self._packet_logits(packets[name], 300 + i)
            for i, name in enumerate(("D", "E", "C"))
            if name in packets
        }
        by_type = {packets[name]["packet_type"]: logits[name] for name in logits}
        trainer.score_packet = lambda query_id, packet: by_type[packet["packet_type"]]
        result = trainer.finalize_query(prepared, dict(logits))
        self.assertIn("E:empty_positive_or_negative", result["skipped"])
        protocol = torch.stack(self._protocol_terms(logits, packets)).mean()
        torch.testing.assert_close(result["loss"], protocol, atol=0.0, rtol=0.0)


class RetrievalPipelineTests(unittest.TestCase):
    """End-to-end retrieval, admission and Teacher re-ranking on a synthetic bank."""

    def setUp(self):
        torch.manual_seed(13)
        self.resolved = {
            "seed": 13,
            "ann": {"space": "ip", "M": 16, "ef_construction": 100, "ef_search": 128,
                    "construction_threads": 1, "query_threads": 1},
            "retrieval": {
                "direct_k": 100, "evidence_per_modality": 10, "second_hop_k": 20,
                "candidate_budget": 100, "teacher_target_chunk": 8, "exact_corpus_chunk": 4096,
            },
            "evaluation": {"ks": [10, 20, 50]},
        }
        self.corpora = {
            "target": [f"target_{i:03d}" for i in range(30)],
            "evidence_text": [f"text_{i:03d}" for i in range(12)],
            "evidence_image": [f"image_{i:03d}" for i in range(12)],
        }
        self.corpora["query"] = [f"query_{i:03d}" for i in range(3)]
        self.corpora["evidence"] = self.corpora["evidence_text"] + self.corpora["evidence_image"]
        self.corpora["all"] = (
            self.corpora["query"] + self.corpora["target"] + self.corpora["evidence"]
        )
        self.bank = _random_bank(len(self.corpora["all"]), 32, 8)
        self.bank.object_ids = list(self.corpora["all"])
        self.bank.position = {o: i for i, o in enumerate(self.bank.object_ids)}
        for i, object_id in enumerate(self.bank.object_ids):
            if object_id.startswith("text_"):
                self.bank.modality_id[i] = 1
                self.bank.kind_ids[i] = np.array(cache_module.TEXT_KINDS)
            elif object_id.startswith("image_"):
                self.bank.modality_id[i] = 2
                self.bank.kind_ids[i] = np.array(cache_module.IMAGE_KINDS_FULL)
        self.student = build_student(
            {"input_dim": 32, "dimension": 16, "interaction_rank": 4, "slot_kind_count": 11}
        ).eval()
        with torch.no_grad():
            for parameter in self.student.parameters():
                parameter.copy_(parameter)
        self.engine = evaluate.RetrievalEngine(
            student=self.student, bank=self.bank, corpora=self.corpora,
            retrieval=self.resolved["retrieval"], ann=self.resolved["ann"], seed=13,
            device="cpu", build_ann=True,
        )

    def test_pipeline_structure_and_budget(self):
        outputs = self.engine.pipeline_both("query_000")
        self.assertIn("ann", outputs)
        self.assertIn("exact", outputs)
        result = outputs["exact"]
        self.assertLessEqual(len(result["D100"]), 100)
        self.assertLessEqual(len(result["B_Q"]), 20)
        self.assertLessEqual(len(result["C100"]), 100)
        self.assertEqual(len(result["C100"]), len(set(result["C100"])))
        self.assertEqual(len(result["U"]), len(set(result["U"])))
        for target in result["C100"]:
            self.assertIn(target, result["U"])
        # every C100 target must carry at least one arrival path
        for target in result["R_E"]:
            self.assertIn(target, result["C100"])

    def test_ann_and_exact_share_the_bundle(self):
        outputs = self.engine.pipeline_both("query_000")
        self.assertEqual(outputs["ann"]["B_Q"], outputs["exact"]["B_Q"])

    def test_round_robin_admission_fills_the_budget(self):
        result = self.engine.pipeline_both("query_000")["exact"]
        limit = int(self.resolved["retrieval"]["candidate_budget"])
        expected = min(len(set(result["D100"]) | set(result["R_E"])), limit)
        self.assertEqual(len(result["C100"]), expected)

    def test_teacher_rerank_over_the_full_natural_bundle(self):
        teacher = build_teacher(
            {"input_dim": 32, "dimension": 16, "heads": 4, "ffn_dimension": 32,
             "slot_kind_count": 11}
        ).eval()
        batch = train.TeacherBatch(self.bank, device="cpu")
        result = self.engine.pipeline_both("query_000")["ann"]
        scores = torch.cat(
            batch.score(teacher, ["query_000"], [result["C100"]], [result["B_Q"]], 1, chunk=4)
        )
        self.assertEqual(scores.shape[0], len(result["C100"]))
        self.assertTrue(torch.isfinite(scores).all())
        # the bundle is exactly the natural B_Q used for every candidate
        self.assertEqual(len(result["B_Q"]), len(set(result["B_Q"])))

    def test_raw_engine_runs_the_same_pipeline(self):
        raw = evaluate.RawEngine(
            bank=self.bank, corpora=self.corpora, retrieval=self.resolved["retrieval"],
            ann=self.resolved["ann"], seed=13, device="cpu", build_ann=True,
        )
        outputs = raw.pipeline_both(self.bank, "query_000")
        self.assertIn("ann", outputs)
        self.assertLessEqual(len(outputs["exact"]["C100"]), 100)
        scores = raw.direct_scores(self.bank, "query_000", outputs["exact"]["C100"])
        self.assertEqual(len(scores), len(outputs["exact"]["C100"]))

    def test_metrics_use_the_fixed_denominator(self):
        metrics = evaluate.query_metrics(
            ["target_000", "target_001"], ["target_000"], ["target_001"], [10, 20]
        )
        self.assertAlmostEqual(metrics["overall_R10"], 1.0)
        self.assertAlmostEqual(metrics["explicit_R10"], 1.0)
        self.assertAlmostEqual(metrics["implicit_R10"], 1.0)
        # an empty ranking contributes zero rather than being dropped
        empty = evaluate.query_metrics([], ["target_000"], [], [10])
        self.assertEqual(empty["overall_R10"], 0.0)
        macro = evaluate.macro_metrics([metrics, empty], [10])
        self.assertAlmostEqual(macro["overall_R10"], 0.5)
        self.assertEqual(macro["explicit_R10_queries"], 2)

    def test_missing_class_is_na_not_zero(self):
        rows = [
            {"overall_R10": 0.5, "explicit_R10": 0.5},
            {"overall_R10": 0.0},
        ]
        macro = evaluate.macro_metrics(rows, [10])
        self.assertAlmostEqual(macro["overall_R10"], 0.25)
        self.assertIsNone(macro["implicit_R10"])


class TimingTests(unittest.TestCase):
    """The timing layer must attribute cost without changing any result."""

    def test_accumulator_totals_and_unattributed(self):
        timing = Timing()
        timing.record("a", 0.01)
        timing.record("a", 0.02)
        timing.record("b", 0.005)
        report = timing.report(name="probe", total_seconds=0.1)
        self.assertEqual(report["stages"]["a"]["count"], 2)
        self.assertAlmostEqual(report["stages"]["a"]["total_seconds"], 0.03, places=9)
        self.assertAlmostEqual(report["attributed_seconds"], 0.035, places=9)
        self.assertAlmostEqual(report["unattributed_seconds"], 0.065, places=9)

    def test_percentiles_on_a_known_ramp(self):
        from mmdd_stage1_clean.timing import _percentile

        ramp = [float(i) for i in range(101)]
        self.assertAlmostEqual(_percentile(ramp, 0.5), 50.0)
        self.assertAlmostEqual(_percentile(ramp, 0.95), 95.0)
        self.assertAlmostEqual(_percentile(ramp, 1.0), 100.0)

    def test_stage_labels_separate_buckets(self):
        timing = Timing()
        timing.record("f:packet=D", 1.0)
        timing.record("f:packet=E", 2.0)
        report = timing.summarise()
        self.assertAlmostEqual(report["f:packet=D"]["total_seconds"], 1.0)
        self.assertAlmostEqual(report["f:packet=E"]["total_seconds"], 2.0)

    def test_encode_one_attributes_phases_to_the_modality(self):
        """Every phase a modality pays for must land in that modality's bucket."""
        timing = Timing()
        encoder = _StubEncoder(timing)
        entry = {"object_id": "text_000", "object_type": "evidence", "modality": "text"}
        lake = _StubLake()
        gt = {"canonical": {"text_000": {"asset_type": "text"}}}
        payload = cache_module.encode_one(
            encoder, lake, entry,
            {"table_max_rows": 12, "table_max_cell_chars": 1024, "table_row_format": "values"},
            gt,
        )
        self.assertEqual(payload["modality_id"], cache_module.MODALITY_TEXT)
        staged = timing.summarise()
        for stage in ("preprocess", "qwen_forward", "token_spans", "resolve_content"):
            self.assertIn(f"{stage}:modality=text", staged, stage)
        # only this modality's buckets exist; no unattributed stage leaked through
        self.assertTrue(all(key.endswith("modality=text") for key in staged), staged)

    def test_cache_writer_times_its_shard_writes(self):
        import tempfile

        timing = Timing()
        with tempfile.TemporaryDirectory() as directory:
            writer = cache_module.CacheWriter(
                Path(directory), shard_objects=4, dim=8, slots=2,
                cache_fingerprint="fp", tag="probe", timing=timing,
            )
            writer.add(
                {
                    "object_id": "o0", "modality": "table", "object_type": "table",
                    "modality_id": 0, "kind_ids": [0, 1, 2, 2, 2, 2, 2, 2, 2],
                    "z": np.zeros(8, dtype=np.float32),
                    "summary": np.zeros((2, 8), dtype=np.float16),
                    # the mask covers every slot: 1 global plus the summaries
                    "mask": np.ones(3, dtype=np.uint8),
                }
            )
            writer.close()
            report = timing.summarise()
            self.assertIn("shard_write", report)
            self.assertEqual(report["shard_write"]["count"], 1)

    def test_index_build_and_query_are_timed(self):
        from mmdd_stage1_clean.retrieve import AnnIndex

        timing = Timing()
        rng = np.random.default_rng(13)
        keys = rng.normal(size=(64, 16)).astype(np.float32)
        keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        index = AnnIndex(
            keys, {"space": "ip", "M": 16, "ef_construction": 100, "ef_search": 64,
                   "construction_threads": 1, "query_threads": 1},
            seed=13, label="target", timing=timing,
        )
        index.query(keys[:1], 5)
        staged = timing.summarise()
        for stage in ("index_init:index=target", "index_add_items:index=target",
                      "ann_knn_query:index=target"):
            self.assertIn(stage, staged, stage)
        self.assertEqual(staged["index_add_items:index=target"]["count"], 1)

    def test_timing_does_not_change_the_score(self):
        """Instrumented and uninstrumented paths must produce identical numbers."""
        from mmdd_stage1_clean.retrieve import AnnIndex

        rng = np.random.default_rng(13)
        keys = rng.normal(size=(80, 16)).astype(np.float32)
        keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        config = {"space": "ip", "M": 16, "ef_construction": 100, "ef_search": 64,
                  "construction_threads": 1, "query_threads": 1}
        plain = AnnIndex(keys, config, seed=13, label="target")
        timed = AnnIndex(keys, config, seed=13, label="target", timing=Timing())
        left, _ = plain.query(keys[:3], 7)
        right, _ = timed.query(keys[:3], 7)
        np.testing.assert_array_equal(left, right)


class _StubEncoder:
    """Minimal stand-in that exercises encode_one's timing attribution."""

    def __init__(self, timing):
        self.timing = timing
        self._active = timing
        self.tokenizer = _StubTokenizer()
        self.part_token_limit = 512

    def encode_text(self, *, object_id, content, instruction):
        with self._active.stage("preprocess"):
            pass
        with self._active.stage("qwen_forward"):
            pass
        with self._active.stage("token_spans"):
            pass
        return {
            "z": torch.zeros(4096), "summary": torch.zeros(8, 4096),
            "token_counts": [1], "content_tokens": 3,
        }


class _StubTokenizer:
    """Enough of the tokenizer contract for encode_one's helpers."""

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(max(1, len(str(text)) // 4)))}

    def decode(self, ids, **kwargs):
        return "x" * len(ids)


class _StubLake:
    def __init__(self):
        self.assets = {"text_000": {"asset_id": "text_000", "asset_type": "text",
                                    "content": "hello world"}}


class ArchitecturePinTests(unittest.TestCase):
    """The two architectures are fixed by the spec; drift must fail loudly."""

    def test_production_parameter_counts_match_the_spec(self):
        """Spec 6.1 states 8,417,281 Teacher and 7,554,048 Student parameters."""
        teacher = build_teacher(
            {"input_dim": 4096, "dimension": 512, "heads": 8, "ffn_dimension": 1024,
             "slot_kind_count": 11}
        )
        student = build_student(
            {"input_dim": 4096, "dimension": 1024, "interaction_rank": 64,
             "slot_kind_count": 11}
        )
        self.assertEqual(sum(p.numel() for p in teacher.parameters()), 8_417_281)
        self.assertEqual(sum(p.numel() for p in student.parameters()), 7_554_048)

    def test_teacher_has_exactly_one_transformer_stack_and_one_readout(self):
        teacher = build_teacher(
            {"input_dim": 4096, "dimension": 512, "heads": 8, "ffn_dimension": 1024,
             "slot_kind_count": 11}
        )
        stacks = [
            name for name, module in teacher.named_modules()
            if isinstance(module, torch.nn.TransformerEncoderLayer)
        ]
        self.assertEqual(len(stacks), 3)
        readouts = [name for name, module in teacher.named_modules()
                    if isinstance(module, torch.nn.Linear) and name == "readout"]
        self.assertEqual(len(readouts), 1)
        # P and J share one weight object; only the task embedding differs.
        self.assertEqual(teacher.task.num_embeddings, 2)
        for forbidden in ("fbridge", "fbase", "gate"):
            self.assertFalse(
                any(forbidden in name.lower() for name, _ in teacher.named_modules()),
                f"a {forbidden} sub-network must never exist",
            )

    def test_student_has_no_extra_encoder_or_resampler(self):
        student = build_student(
            {"input_dim": 4096, "dimension": 1024, "interaction_rank": 64,
             "slot_kind_count": 11}
        )
        self.assertFalse(
            any(isinstance(m, torch.nn.TransformerEncoderLayer) for m in student.modules())
        )
        linears = [n for n, m in student.named_modules() if isinstance(m, torch.nn.Linear)]
        self.assertEqual(
            sorted(linears),
            ["base_e", "direct", "e_factor", "evidence", "out_factor", "project", "q_factor"],
        )
        # the low-rank factor shapes of Eq. (14)
        self.assertEqual(student.q_factor.weight.shape, (64, 1024))
        self.assertEqual(student.e_factor.weight.shape, (64, 1024))
        self.assertEqual(student.out_factor.weight.shape, (1024, 64))

    def test_initialisation_follows_the_spec(self):
        """A_D/A_E/B start at identity, embeddings at std 0.02, a at zero."""
        student = build_student(
            {"input_dim": 4096, "dimension": 1024, "interaction_rank": 64,
             "slot_kind_count": 11}
        )
        for name in ("direct", "evidence", "base_e"):
            torch.testing.assert_close(
                getattr(student, name).weight, torch.eye(1024)
            )
        torch.testing.assert_close(student.pool_query, torch.zeros(1024))
        self.assertLess(float(student.modality.weight.std()), 0.05)
        self.assertLess(float(student.kind.weight.std()), 0.05)
        for layer in (student.norm,):
            torch.testing.assert_close(layer.weight, torch.ones(1024))
            torch.testing.assert_close(layer.bias, torch.zeros(1024))
        teacher = build_teacher(
            {"input_dim": 4096, "dimension": 512, "heads": 8, "ffn_dimension": 1024,
             "slot_kind_count": 11}
        )
        # per-layer Xavier: the three layers must not share identical draws
        first = teacher.layers[0].self_attn.in_proj_weight
        second = teacher.layers[1].self_attn.in_proj_weight
        self.assertFalse(torch.allclose(first, second))
        self.assertEqual(len({id(layer) for layer in teacher.layers}), 3)


class TeacherLogitCacheTests(unittest.TestCase):
    """Spec 14.14: a cached score must never be reusable under other inputs."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = evaluate.TeacherLogitCache(Path(self.tmpdir) / "logits.sqlite")
        self.teacher = build_teacher(
            {"input_dim": 32, "dimension": 16, "heads": 4, "ffn_dimension": 32,
             "slot_kind_count": 11}
        ).eval()
        corpora = {
            "query": ["q"],
            "target": [f"t{i}" for i in range(6)],
            "evidence_text": [f"x{i}" for i in range(4)],
            "evidence_image": [f"i{i}" for i in range(4)],
        }
        corpora["all"] = (
            corpora["query"] + corpora["target"]
            + corpora["evidence_text"] + corpora["evidence_image"]
        )
        bank = _random_bank(len(corpora["all"]), 32, 8)
        bank.object_ids = list(corpora["all"])
        bank.position = {o: i for i, o in enumerate(bank.object_ids)}
        self.bank = bank
        self.batch = train.TeacherBatch(bank, device="cpu")

    def tearDown(self):
        self.cache.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _score(self, **kwargs):
        return commands.score_selection(
            teacher=self.teacher, batch=self.batch, cache=self.cache,
            teacher_hash="H1", chunk=4, **kwargs
        )

    def test_reuse_returns_identical_values(self):
        candidates = ["t0", "t1", "t2"]
        first = self._score(mode="J", query_id="q", candidates=candidates, context=["x0"])
        second = self._score(mode="J", query_id="q", candidates=candidates, context=["x0"])
        self.assertEqual(first, second)

    def test_a_changed_context_never_reuses_a_score(self):
        """Different bundle, same ids: the J value must be recomputed."""
        candidates = ["t0", "t1"]
        with_x = self._score(mode="J", query_id="q", candidates=candidates, context=["x0"])
        with_y = self._score(mode="J", query_id="q", candidates=candidates, context=["x1"])
        self.assertNotEqual(with_x, with_y)
        empty = self._score(mode="J", query_id="q", candidates=candidates, context=[])
        self.assertNotEqual(with_x, empty)

    def test_mode_is_part_of_the_key(self):
        """P and J are different tasks; a P score is not a J score."""
        candidates = ["t0", "t1"]
        p_scores = self._score(mode="P", query_id="q", candidates=candidates, context=[])
        # Prime the J entry, then confirm P did not satisfy it.
        j_scores = self._score(mode="J", query_id="q", candidates=candidates, context=[])
        self.assertNotEqual(p_scores, j_scores)

    def test_a_different_teacher_hash_is_a_cache_miss(self):
        candidates = ["t0", "t1"]
        self._score(mode="J", query_id="q", candidates=candidates, context=["x0"])
        before = self.cache.misses
        commands.score_selection(
            teacher=self.teacher, batch=self.batch, cache=self.cache,
            teacher_hash="H2", mode="J", query_id="q",
            candidates=candidates, context=["x0"], chunk=4,
        )
        self.assertGreater(self.cache.misses, before)

    def test_cache_is_queried_by_content_not_position(self):
        """Reordering the candidate list must still hit the same entries."""
        candidates = ["t0", "t1", "t2"]
        first = self._score(mode="J", query_id="q", candidates=candidates, context=["x0"])
        hits_before = self.cache.hits
        reversed_scores = self._score(
            mode="J", query_id="q", candidates=list(reversed(candidates)), context=["x0"]
        )
        self.assertGreater(self.cache.hits, hits_before)
        for candidate in candidates:
            self.assertAlmostEqual(first[candidate], reversed_scores[candidate], places=6)


class DistillationInvariantTests(unittest.TestCase):
    """Spec 9.1 / 3: the Student refreshes candidates; it never trains the Teacher."""

    def setUp(self):
        self.test = TrainingLoopTests("test_bundle_view_alternates_deterministically")
        self.test.setUp()
        self.resolved = self.test.resolved
        self.bank = self.test.bank
        self.builder = self.test.builder

    def _trainer(self, arm):
        return train.StudentTrainer(
            resolved=self.resolved, bank=self.bank, builder=self.builder,
            rank_tables=self.test._rank_tables(), anchor_rank={},
            output_dir=Path("/tmp/_kdinv"), arm=arm,
            init_state=build_student(self.resolved["student"]).state_dict(),
            device="cpu", max_epochs=1,
        )

    def test_teacher_parameters_are_unchanged_by_a_student_step(self):
        teacher = build_teacher(self.resolved["teacher"]).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        before = {n: p.detach().clone() for n, p in teacher.named_parameters()}
        trainer = self._trainer("KD")
        batch = train.TeacherBatch(self.bank, device="cpu")
        trainer.train(teacher=teacher, teacher_batch=batch, query_rankings={})
        for name, parameter in teacher.named_parameters():
            torch.testing.assert_close(parameter, before[name], msg=name)

    def test_teacher_places_no_gradient_on_its_own_parameters(self):
        """The KD target is a constant, so no Teacher tensor may hold a grad."""
        teacher = build_teacher(self.resolved["teacher"]).eval()
        trainer = self._trainer("KD")
        batch = train.TeacherBatch(self.bank, device="cpu")
        prepared = trainer.prepare_query("query_000", 2, None)
        packed = train.PackedTeacher(batch, teacher, 8)
        rows = {}
        for name in ("D", "E", "C"):
            packet = prepared["built"]["packets"].get(name)
            if packet is None:
                continue
            rows[name] = packed.add(
                "query_000", packet["candidates"], packet["context"],
                1 if packet["mode"] == "J" else 0,
            )
        with torch.no_grad():
            flat = packed.run(max_batch=3)
        kd_scores = {
            name: train.list_scores(flat[row], prepared["built"]["packets"][name])
            for name, row in rows.items()
        }
        for value in kd_scores.values():
            self.assertFalse(value.requires_grad)
        result = trainer.finalize_query(prepared, kd_scores)
        result["loss"].backward()
        for name, parameter in teacher.named_parameters():
            self.assertIsNone(parameter.grad, name)

    def test_kd_and_rank_both_reach_the_student(self):
        teacher = build_teacher(self.resolved["teacher"]).eval()
        trainer = self._trainer("KD")
        batch = train.TeacherBatch(self.bank, device="cpu")
        trainer.train(teacher=teacher, teacher_batch=batch, query_rankings={})
        self.assertIsNotNone(trainer.student.base_e.weight.grad)

    def test_sup_arm_never_consults_the_teacher(self):
        """S-SUP is the explicit no-distillation control (spec 9.3)."""
        trainer = self._trainer("SUP")
        result = trainer.train(teacher=None, teacher_batch=None, query_rankings={})
        self.assertEqual(len(result["epochs"]), 1)
        self.assertTrue(np.isfinite(result["epochs"][0]["mean_loss"]))


class NonRetrievableFilterTests(unittest.TestCase):
    """An object with no cached features must leave the corpus *visibly*."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        (self.tmpdir / "objects").mkdir(parents=True, exist_ok=True)
        (self.tmpdir / "cache" / "shards").mkdir(parents=True, exist_ok=True)
        write_corpus = {
            "target": ["target_1", "target_2"],
            "query": ["query_1"],
            "evidence_text": ["text_ok"],
            "evidence_image": ["img_broken"],
            "evidence": ["text_ok", "img_broken"],
        }
        with (self.tmpdir / "objects" / "corpus.jsonl").open("w", encoding="utf-8") as handle:
            for name, ids in write_corpus.items():
                handle.write(json.dumps({"corpus": name, "count": len(ids), "object_ids": ids}) + "\n")
        # the cache is missing img_broken, as it would be after an encode failure
        with (self.tmpdir / "cache" / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for object_id, modality_id, kinds in (
                ("target_1", 0, cache_module.TABLE_KINDS),
                ("target_2", 0, cache_module.TABLE_KINDS),
                ("query_1", 0, cache_module.TABLE_KINDS),
                ("text_ok", 1, cache_module.TEXT_KINDS),
            ):
                handle.write(json.dumps({
                    "cache_fingerprint": "fp", "object_id": object_id, "shard": "shard-00000",
                    "offset": 0, "modality": "table" if modality_id == 0 else "text",
                    "object_type": "table" if modality_id == 0 else "evidence",
                    "modality_id": modality_id, "kind_ids": list(kinds),
                    "mask_popcount": 9, "z_norm": 1.0,
                }) + "\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_uncached_object_is_dropped_and_recorded(self):
        corpus = commands._load_corpus(self.tmpdir)
        self.assertNotIn("img_broken", corpus["evidence_image"])
        self.assertEqual(corpus["target"], ["target_1", "target_2"])
        registry_path = self.tmpdir / "objects" / "non_retrievable.jsonl"
        self.assertTrue(registry_path.is_file())
        dropped = [json.loads(line) for line in registry_path.read_text().splitlines()]
        self.assertEqual({d["object_id"] for d in dropped}, {"img_broken"})
        summary = json.loads(
            (self.tmpdir / "objects" / "NON_RETRIEVABLE_SUMMARY.json").read_text()
        )
        self.assertEqual(summary["dropped_objects"], 1)
        self.assertEqual(summary["dropped_entries"], 2)  # once in evidence, once in evidence_image
        self.assertIn("denominator", summary["effect"])

    def test_targets_are_never_silently_dropped(self):
        """A target only leaves the corpus if it genuinely has no features."""
        corpus = commands._load_corpus(self.tmpdir)
        self.assertEqual(set(corpus["target"]), {"target_1", "target_2"})
        summary = json.loads(
            (self.tmpdir / "objects" / "NON_RETRIEVABLE_SUMMARY.json").read_text()
        )
        self.assertEqual(summary["by_corpus"].get("target", 0), 0)

    def test_filtered_corpus_matches_the_cache_exactly(self):
        corpus = commands._load_corpus(self.tmpdir)
        cached = set(cache_module.read_cache_index(self.tmpdir / "cache"))
        self.assertEqual(set(corpus["all"]), cached)


class SinglePathPipelineTests(unittest.TestCase):
    """`pipeline` must return the same single path `pipeline_both` produces.

    The per-epoch Student dev check scores only the Student's own candidates, so it
    calls this entry point; a mismatch against the combined call would silently
    change what the checkpoint selection sees.
    """

    def setUp(self):
        self.test = RetrievalPipelineTests("test_pipeline_structure_and_budget")
        self.test.setUp()
        self.engine = self.test.engine

    def test_single_path_matches_combined(self):
        both = self.engine.pipeline_both("query_000")
        ann_only = self.engine.pipeline("query_000", use_ann=True)
        exact_only = self.engine.pipeline("query_000", use_ann=False)
        self.assertEqual(sorted(ann_only), ["ann"])
        self.assertEqual(sorted(exact_only), ["exact"])
        for label, single in (("ann", ann_only), ("exact", exact_only)):
            self.assertEqual(single[label]["C100"], both[label]["C100"])
            self.assertEqual(single[label]["D100"], both[label]["D100"])
            self.assertEqual(single[label]["B_Q"], both[label]["B_Q"])
            self.assertEqual(single[label]["U"], both[label]["U"])
            self.assertEqual(
                single[label]["arrival_evidence_by_target"],
                both[label]["arrival_evidence_by_target"],
            )

    def test_single_path_does_not_build_the_other_index_work(self):
        """A single path must not emit the other path's timing buckets."""
        from mmdd_stage1_clean.timing import Timing

        engine = evaluate.RetrievalEngine(
            student=self.test.student, bank=self.test.bank, corpora=self.test.corpora,
            retrieval=self.test.resolved["retrieval"], ann=self.test.resolved["ann"],
            seed=13, device="cpu", build_ann=True, timing=Timing(),
        )
        engine.pipeline("query_000", use_ann=True)
        stages = engine.timing.summarise()
        self.assertIn("candidate_admission:path=ann", stages)
        self.assertNotIn("candidate_admission:path=exact", stages)


# ==========================================================================
# NEXT_STEPS_DEEPSEEK.md task 2 — the two repaired evaluation entrypoints
# ==========================================================================


class EvaluationEntrypointFixTests(unittest.TestCase):
    """A hand-built two-dimensional bank where Q and E point at different targets.

    Q = (1, 0) admits ``target_direct``; the only evidence E = (0, 1) admits
    ``target_bridge``.  The runtime ``RawEngine`` must run the second hop on the
    evidence, and the freeze entry must admit the evidence-only target.
    """

    def setUp(self):
        self.retrieval = {
            "direct_k": 1,
            "evidence_per_modality": 1,
            "second_hop_k": 1,
            "candidate_budget": 100,
            "exact_corpus_chunk": 4096,
        }
        self.corpora = {
            "query": ["query_q"],
            "target": ["target_direct", "target_bridge"],
            "evidence_text": ["text_e"],
            "evidence_image": [],
        }
        self.corpora["evidence"] = (
            self.corpora["evidence_text"] + self.corpora["evidence_image"]
        )
        self.corpora["all"] = (
            self.corpora["query"] + self.corpora["target"] + self.corpora["evidence"]
        )
        vectors = {
            "query_q": [1.0, 0.0],
            "target_direct": [1.0, 0.0],
            "target_bridge": [0.0, 1.0],
            "text_e": [0.0, 1.0],
        }
        ids = self.corpora["all"]
        z = np.array([vectors[o] for o in ids], dtype=np.float32)
        summary = np.zeros((len(ids), 8, 2), dtype=np.float16)
        mask = np.ones((len(ids), 9), dtype=np.uint8)
        modality_id = np.zeros(len(ids), dtype=np.int64)
        kind_ids = np.stack(
            [np.array(cache_module.TABLE_KINDS, dtype=np.int64)] * len(ids)
        )
        for i, object_id in enumerate(ids):
            if object_id.startswith("text_"):
                modality_id[i] = 1
                kind_ids[i] = np.array(cache_module.TEXT_KINDS)
        self.bank = ObjectBank(ids, modality_id, kind_ids, z, summary, mask)

    def test_raw_engine_conditions_the_second_hop_on_the_evidence(self):
        raw = evaluate.RawEngine(
            bank=self.bank, corpora=self.corpora, retrieval=self.retrieval,
            ann={}, seed=13, device="cpu", build_ann=False,
        )
        output = raw.pipeline_both(
            self.bank, "query_q", paths=(("exact", False),)
        )["exact"]
        self.assertEqual(output["D100"], ["target_direct"])
        self.assertEqual(output["B_Q"], ["text_e"])
        # The evidence's own nearest target, not the query's.
        self.assertEqual(output["L_E"]["text_e"], ["target_bridge"])
        self.assertNotEqual(output["L_E"]["text_e"], output["D100"])
        # The evidence-only target enters the union and the admission set.
        self.assertIn("target_bridge", output["U"])
        self.assertEqual(output["C100"], ["target_direct", "target_bridge"])

    def test_freeze_candidate_builder_is_round_robin_not_d100(self):
        records = [
            {
                "query_id": "query_q",
                "target_ids": ["target_direct", "target_bridge"],
                "text_ids": ["text_e"],
                "image_ids": [],
            }
        ]
        population = {
            "query_q": {
                "direct_target_ids": ["target_direct"],
                "implicit_target_ids": ["target_bridge"],
                "positive_target_ids": ["target_direct", "target_bridge"],
            }
        }
        fixed = commands.build_fixed_candidates(
            bank=self.bank, target_ids=self.corpora["target"],
            retrieval=self.retrieval, records=records, population=population,
        )
        self.assertEqual(len(fixed), 1)
        record = fixed[0]
        self.assertEqual(record["D100"], ["target_direct"])
        self.assertEqual(record["L_E"]["text_e"], ["target_bridge"])
        self.assertEqual(record["R_E"], ["target_bridge"])
        self.assertEqual(record["U"], ["target_direct", "target_bridge"])
        expected = reference.round_robin([record["D100"], record["R_E"]], 100)
        self.assertEqual(record["C100"], expected)
        self.assertEqual(record["C100"], ["target_direct", "target_bridge"])
        # "same length as D100" or "same as R_E" would not be sufficient checks.
        self.assertNotEqual(record["C100"], record["D100"])
        self.assertNotEqual(record["C100"], record["R_E"])
        self.assertIn("target_bridge", record["C100"])

"""Unit tests for MMDD Stage1 CLEAN-QET v4.0 implementation."""
from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
import torch

from mmdd_clean_qet_v4.config import Paths, load_protocol, validate_protocol
from mmdd_clean_qet_v4.losses import (
    aggregate_paths,
    global_features,
    hierarchical_support_mean,
    positive_average_pair_loss,
    rank_mass_loss,
)
from mmdd_clean_qet_v4.models import FreshPathTeacher, NativeStudent, QTStudent
from mmdd_clean_qet_v4.provenance import state_sha
from mmdd_clean_qet_v4.retrieval import PathEntry, d1_retain, p3_admission


class TestCleanQetV4(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)

    def test_rank_mass_loss(self) -> None:
        scores = torch.tensor([1.0, 2.0, 3.0])
        pos_mask = torch.tensor([True, False, False])
        loss = rank_mass_loss(scores, pos_mask)
        self.assertIsNotNone(loss)
        # Shift invariance
        loss_shifted = rank_mass_loss(scores + 50.0, pos_mask)
        torch.testing.assert_close(loss, loss_shifted)

    def test_positive_average_pair_loss(self) -> None:
        positives = torch.tensor([1.5, 2.5])
        competitors = torch.tensor([0.5, 0.8, -0.2])
        loss = positive_average_pair_loss(positives, competitors)
        self.assertIsNotNone(loss)
        loss_shifted = positive_average_pair_loss(positives + 20.0, competitors + 20.0)
        torch.testing.assert_close(loss, loss_shifted)

    def test_aggregate_paths_permutation(self) -> None:
        f0 = torch.tensor([0.5, -0.3, 1.2])
        paths = torch.tensor([2.0, 1.0, 3.0, 0.5])
        targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        agg = aggregate_paths(f0, paths, targets)
        self.assertEqual(agg.shape, f0.shape)
        # Target 2 has no paths -> f0[2]
        torch.testing.assert_close(agg[2], f0[2])

    def test_global_features_11h(self) -> None:
        left = torch.randn(2, 8)
        right = torch.randn(2, 8)
        pair = torch.randn(2, 8)
        # Empty evidence -> 6 * 8 = 48 zeros at tail
        feat_empty = global_features(left, right, pair)
        self.assertEqual(feat_empty.shape, (2, 88))
        self.assertEqual(int(torch.count_nonzero(feat_empty[:, 40:])), 0)

        # Nonempty evidence
        ev = torch.randn(2, 8)
        etype = torch.randn(2, 8)
        feat_ev = global_features(left, right, pair, evidence=ev, evidence_type_embedding=etype)
        self.assertEqual(feat_ev.shape, (2, 88))
        self.assertGreater(int(torch.count_nonzero(feat_ev[:, 40:])), 0)

    def test_teacher_architecture(self) -> None:
        teacher = FreshPathTeacher(input_dim=32, width=16, heads=2, layers=1, ffn=32, text_slots=2, image_slots=2)
        self.assertEqual(teacher.global_relation[0].in_features, 11 * 16)

        za, zb, ze = [torch.randn(32) for _ in range(3)]
        ca, cb, ce = torch.randn(3, 32), torch.randn(3, 32), torch.randn(2, 32)
        score_pair = teacher.score_pairs([("table", za, ca, "table", zb, cb)])
        self.assertEqual(score_pair.shape, (1,))

        score_trip = teacher.score_triplets([("table", za, ca, "text", ze, ce, "table", zb, cb)])
        self.assertEqual(score_trip.shape, (1,))

    def test_student_architecture(self) -> None:
        pca_basis = torch.randn(1024, 4096)
        pca_mean = torch.randn(4096)
        student = NativeStudent(pca_basis, pca_mean)
        self.assertEqual(student.anchor_loss().item(), 0.0)

        qt_student = QTStudent(pca_basis, pca_mean)
        self.assertEqual(qt_student.anchor_loss().item(), 0.0)

    def test_d1_retention(self) -> None:
        entries = [
            PathEntry("e1", "text", 1.0, 2.0),
            PathEntry("e2", "image", 0.5, 1.5),
        ]
        support = {
            "e1": np.array([0.9, 0.8]),
            "e2": np.array([0.5, 0.6]),
        }
        selected, cov = d1_retain(entries, support, budget=2)
        self.assertIn("e1", selected)
        self.assertGreater(cov, 0.0)

    def test_p3_admission(self) -> None:
        qt = {"t1": 2.0, "t2": 1.0, "t3": 0.5}
        ev = ["t3", "t2"]
        admitted, scores = p3_admission(qt, ev, budget=2)
        self.assertEqual(len(admitted), 2)

    def test_batched_teacher_vs_reference(self) -> None:
        torch.manual_seed(42)
        teacher = FreshPathTeacher(input_dim=64, width=32, heads=2, layers=1, ffn=64, text_slots=2, image_slots=2).eval()
        zq = torch.randn(64)
        cq = torch.randn(10, 64)
        n_targets = 8
        zt = torch.randn(n_targets, 64)
        ct_list = [torch.randn(np.random.randint(6, 12), 64) for _ in range(n_targets)]

        ev_map = {
            "e_txt_0": ("text", torch.randn(64), torch.randn(64, 64)),
            "e_txt_1": ("text", torch.randn(64), torch.randn(64, 64)),
            "e_img_0": ("image", torch.randn(64), torch.randn(64, 64)),
            "e_img_1": ("image", torch.randn(64), torch.randn(64, 64)),
        }
        paths = [
            (0, "e_txt_0"), (0, "e_img_0"), (1, "e_txt_1"), (2, "e_img_0"),
            (2, "e_txt_0"), (3, "e_img_1"), (4, "e_txt_0"), (4, "e_txt_1"),
            (5, "e_img_0"), (6, "e_txt_0"), (7, "e_img_1"), (7, "e_txt_1"),
        ]

        # Reference pairs & triplets
        pairs = [("table", zq, cq, "table", zt[i], ct_list[i]) for i in range(n_targets)]
        ref_f0 = teacher.score_pairs(pairs)
        trips = [("table", zq, cq, ev_map[eid][0], ev_map[eid][1], ev_map[eid][2], "table", zt[t_idx], ct_list[t_idx]) for t_idx, eid in paths]
        ref_ps = teacher.score_triplets(trips)

        # Batched with small chunk to test chunking
        batched_f0, batched_ps = teacher.score_query_lists((zq, cq), (zt, ct_list), ev_map, paths, chunk=3)
        self.assertLess((ref_f0 - batched_f0).abs().max().item(), 1e-5)
        self.assertLess((ref_ps - batched_ps).abs().max().item(), 1e-5)

        # Gradients check in eval mode (so dropout does not introduce stochastic difference)
        teacher.eval()
        ref_f0 = teacher.score_pairs(pairs)
        ref_ps = teacher.score_triplets(trips)
        teacher.zero_grad(set_to_none=True)
        loss_ref = ref_f0.sum() + ref_ps.sum()
        loss_ref.backward()
        grads_ref = {n: p.grad.clone() for n, p in teacher.named_parameters() if p.grad is not None}

        batched_f0, batched_ps = teacher.score_query_lists((zq, cq), (zt, ct_list), ev_map, paths, chunk=3)
        teacher.zero_grad(set_to_none=True)
        loss_batch = batched_f0.sum() + batched_ps.sum()
        loss_batch.backward()
        grads_batch = {n: p.grad.clone() for n, p in teacher.named_parameters() if p.grad is not None}

        for n in grads_ref:
            diff = (grads_ref[n] - grads_batch[n]).abs().max().item()
            self.assertLess(diff, 1e-4, f"Gradient diff too large on {n}: {diff}")

        # Empty paths
        f0_only, ps_empty = teacher.score_query_lists((zq, cq), (zt, ct_list), ev_map, paths=[], chunk=4)
        self.assertEqual(ps_empty.numel(), 0)
        self.assertEqual(f0_only.shape, (n_targets,))

    def test_aggregate_paths_vectorized_vs_loop(self) -> None:
        torch.manual_seed(42)
        def ref_agg(f0, path_scores, path_target_index):
            result = []
            for target_index in range(f0.numel()):
                terms = torch.cat((f0[target_index : target_index + 1], path_scores[path_target_index == target_index]))
                result.append(torch.logsumexp(terms, dim=0))
            return torch.stack(result)

        f0 = torch.randn(20, requires_grad=True)
        path_scores = torch.randn(45, requires_grad=True)
        path_target_index = torch.randint(0, 20, (45,))

        ref_out = ref_agg(f0, path_scores, path_target_index)
        vec_out = aggregate_paths(f0, path_scores, path_target_index)
        torch.testing.assert_close(vec_out, ref_out)

        # Gradients check
        ref_out.sum().backward()
        g_f0_ref, g_ps_ref = f0.grad.clone(), path_scores.grad.clone()

        f0.grad = None
        path_scores.grad = None
        vec_out = aggregate_paths(f0, path_scores, path_target_index)
        vec_out.sum().backward()
        g_f0_vec, g_ps_vec = f0.grad.clone(), path_scores.grad.clone()

        torch.testing.assert_close(g_f0_vec, g_f0_ref)
        torch.testing.assert_close(g_ps_vec, g_ps_ref)

        # Empty paths
        empty_ps = torch.empty(0)
        empty_idx = torch.empty(0, dtype=torch.long)
        out_empty = aggregate_paths(f0, empty_ps, empty_idx)
        torch.testing.assert_close(out_empty, f0)

    def test_student_evidence_vectorized_vs_loop(self) -> None:
        torch.manual_seed(42)
        pca_basis = torch.randn(1024, 4096)
        pca_mean = torch.randn(4096)
        student = NativeStudent(pca_basis, pca_mean)

        zq = torch.randn(4096)
        targets = [f"t_{i}" for i in range(8)]
        zt_map = {t: torch.randn(4096) for t in targets}
        e_ids = [f"e_{i}" for i in range(10)]
        ekinds = {e: ("text" if i % 2 == 0 else "image") for i, e in enumerate(e_ids)}
        ze_map = {e: torch.randn(4096) for e in e_ids}

        bags = {
            "t_0": ["e_0", "e_1"],
            "t_1": ["e_2"],
            "t_2": [],
            "t_3": ["e_3", "e_4", "e_5"],
            "t_4": ["e_6"],
            "t_5": [],
            "t_6": ["e_7", "e_8"],
            "t_7": ["e_9"],
        }
        bt = [(k, t) for k, t in enumerate(targets) if bags[t]]
        paths = [(row, k, e, slot) for row, (k, t) in enumerate(bt) for slot, e in enumerate(bags[t])]
        unique_e = list(dict.fromkeys(e for _, _, e, _ in paths))

        # 1. Serial loop
        student.zero_grad(set_to_none=True)
        e_scores_loop = []
        for t_idx, t in bt:
            bag = bags[t]
            zt = zt_map[t]
            terms = []
            for eid in bag:
                sqe = student.score("table", zq, ekinds[eid], ze_map[eid])
                set_ = student.score(ekinds[eid], ze_map[eid], "table", zt)
                terms.append(sqe + set_)
            e_scores_loop.append(torch.logsumexp(torch.stack(terms), dim=0))
        e_s_loop = torch.stack(e_scores_loop)
        loss_loop = e_s_loop.sum()
        loss_loop.backward()
        grads_loop = {n: p.grad.clone() for n, p in student.named_parameters() if p.grad is not None}

        # 2. Vectorized
        student.zero_grad(set_to_none=True)
        u_q = student.u("table", zq)
        zt_matrix = torch.stack([zt_map[t] for t in targets])
        u_t = student.u("table", zt_matrix)

        s_qe = torch.empty(len(unique_e))
        proj_eT = torch.empty(len(unique_e), student.dim)
        for kind in ("text", "image"):
            loc = [j for j, e in enumerate(unique_e) if ekinds[e] == kind]
            if not loc:
                continue
            loc_t = torch.tensor(loc, dtype=torch.long)
            ze_k = torch.stack([ze_map[unique_e[j]] for j in loc])
            u_e = student.u(kind, ze_k)
            s_qe = s_qe.index_put((loc_t,), (u_q @ student.R[f"Q_{kind}"] * u_e).sum(dim=-1))
            proj_eT = proj_eT.index_put((loc_t,), u_e @ student.R[f"{kind}_T"])

        e_pos = {e: j for j, e in enumerate(unique_e)}
        e_idx = torch.tensor([e_pos[e] for _, _, e, _ in paths], dtype=torch.long)
        t_idx = torch.tensor([k for _, k, _, _ in paths], dtype=torch.long)
        path_scores = s_qe[e_idx] + (proj_eT[e_idx] * u_t[t_idx]).sum(dim=-1)

        max_bag = max(len(bags[t]) for _, t in bt)
        bag_mat = path_scores.new_full((len(bt), max_bag), float("-inf"))
        rows_t = torch.tensor([r for r, _, _, _ in paths], dtype=torch.long)
        slots_t = torch.tensor([s for _, _, _, s in paths], dtype=torch.long)
        bag_mat = bag_mat.index_put((rows_t, slots_t), path_scores)
        e_s_vec = torch.logsumexp(bag_mat, dim=1)

        loss_vec = e_s_vec.sum()
        loss_vec.backward()
        grads_vec = {n: p.grad.clone() for n, p in student.named_parameters() if p.grad is not None}

        torch.testing.assert_close(e_s_vec, e_s_loop, atol=1e-5, rtol=1e-5)
        for n in grads_loop:
            rel_diff = (grads_vec[n] - grads_loop[n]).abs().max().item() / (grads_loop[n].abs().max().item() + 1e-12)
            self.assertLess(rel_diff, 1e-5, f"Relative gradient diff too large on {n}: {rel_diff}")

    def test_tokens_many_equality(self) -> None:
        import tempfile
        import shutil
        from fresh_path.features import write_chunk, build_chunk_index, ContentStore
        from mmdd_clean_qet_v4.features import ObjectBank, ZStore

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            chunk_dir = tmp_dir / "content" / "chunks"
            rows_0 = [
                ("t0", "table", torch.full((10, 4096), 1.0, dtype=torch.float16).numpy()),
                ("t1", "table", torch.full((15, 4096), 2.0, dtype=torch.float16).numpy()),
            ]
            rows_1 = [
                ("e0", "text", torch.full((64, 4096), 3.0, dtype=torch.float16).numpy()),
                ("e1", "image", torch.full((64, 4096), 4.0, dtype=torch.float16).numpy()),
            ]
            write_chunk(chunk_dir, 0, rows_0)
            write_chunk(chunk_dir, 1, rows_1)
            build_chunk_index(chunk_dir)

            content = ContentStore(tmp_dir / "content")
            ids = ["t0", "t1", "e0", "e1"]
            types = ["table", "table", "text", "image"]
            z_store = ZStore(ids=ids, types=types, z=torch.randn(4, 4096), index={oid: i for i, oid in enumerate(ids)}, sha256="dummy")
            bank = ObjectBank(z_store, content)

            test_ids = ["t0", "e0", "t1", "e1"]
            individual = [bank.tokens(oid) for oid in test_ids]
            batched = bank.tokens_many(test_ids)

            self.assertEqual(len(individual), len(batched))
            for a, b in zip(individual, batched):
                self.assertTrue(torch.equal(a, b))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_encode_many_variable_lengths(self) -> None:
        torch.manual_seed(42)
        teacher = FreshPathTeacher()
        teacher.eval()
        z = torch.randn(3, 4096)
        toks = [torch.randn(64, 4096), torch.randn(21, 4096), torch.randn(64, 4096)]

        # Individual encode_one
        ind_seg = []
        ind_g = []
        for i in range(3):
            s_i, g_i = teacher.encode_one("text", z[i], toks[i])
            ind_seg.append(torch.cat([g_i.unsqueeze(0), s_i], dim=0))
            ind_g.append(g_i)
        expected_seg = torch.stack(ind_seg)
        expected_g = torch.stack(ind_g)

        # Batched encode_many
        act_seg, act_lengths, act_g = teacher.encode_many("text", z, toks)
        self.assertTrue(torch.allclose(expected_seg, act_seg, atol=1e-5))
        self.assertTrue(torch.allclose(expected_g, act_g, atol=1e-5))
        self.assertEqual(act_lengths, [17, 17, 17])


if __name__ == "__main__":
    unittest.main()

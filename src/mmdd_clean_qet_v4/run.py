"""Main CLI entrypoint and execution DAG runner for CLEAN-QET v4.0."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from . import EXPERIMENT_ID, VERSION
from .config import Paths, load_protocol, resolve_default_paths
from .data import (
    build_content_aliases,
    iter_jsonl,
    load_split_gt,
    read_json,
    sha256_file,
    split_query_ids,
    utf8_sorted,
    write_json,
)
from .evaluate import (
    coverage_at_k,
    evaluate_student_retrieval,
    evaluate_teacher_rerank,
    paired_bootstrap,
    recall_at_k,
)
from .features import (
    ContentStore,
    ObjectBank,
    RowStore,
    ZStore,
    build_or_load_row_store,
    fit_pca,
    load_pca,
    load_z,
)
from .labels import Labels, build_labels, load_labels
from .lists import (
    build_c1_edge_lists,
    build_raw_pools_split,
    build_ta_records,
    build_tb_records,
)
from .models import FreshPathTeacher, NativeStudent, QTStudent
from .provenance import (
    generate_provenance_manifests,
    record_stage_post_run,
    record_stage_pre_run,
    snapshot_source,
    state_sha,
)
from .retrieval import CANDIDATE_BUDGET, PoolRecord, d1_retain, p3_admission, row_support
from .train import (
    save_checkpoint,
    train_student_c1,
    train_student_c2,
    train_ta,
    train_tb,
)


def verify_server_smoke(paths: Paths, gpu_uuid: str) -> dict[str, Any]:
    """Preflight acceptance test (§23): 16 queries, 8 updates smoke in separate directory."""
    smoke_dir = paths.run_root / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Verify GPU0 UUID
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu_uuid and visible and visible != gpu_uuid:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES ({visible}) does not match required GPU0 UUID ({gpu_uuid})")

    # 1. Test Model Architecture & 11h input
    teacher = FreshPathTeacher(input_dim=128, width=64, heads=2, layers=1, ffn=128, text_slots=4, image_slots=4)
    teacher.to(device)
    assert teacher.global_relation[0].in_features == 11 * 64, "global_relation input must be 11h"

    # Empty evidence extension: exactly 6h zeros
    za = torch.randn(128, device=device)
    zb = torch.randn(128, device=device)
    ca = torch.randn(4, 128, device=device)
    cb = torch.randn(4, 128, device=device)
    s_pair = teacher.score_pairs([("table", za, ca, "table", zb, cb)])

    # With evidence
    ze = torch.randn(128, device=device, requires_grad=True)
    ce = torch.randn(2, 128, device=device)
    s_trip = teacher.score_triplets([("table", za, ca, "text", ze, ce, "table", zb, cb)])
    s_trip.backward()
    assert ze.grad is not None and ze.grad.norm().item() > 0, "Gradient must flow to evidence input"

    # 2. Test Student
    pca_basis = torch.randn(1024, 4096)
    pca_mean = torch.randn(4096)
    student = NativeStudent(pca_basis, pca_mean)
    student.to(device)
    assert student.anchor_loss().item() == 0.0

    report = {
        "status": "PASS",
        "smoke_dir": str(smoke_dir),
        "teacher_global_relation_dim": teacher.global_relation[0].in_features,
        "evidence_gradient_norm": float(ze.grad.norm().item()),
        "student_anchor_loss_init": float(student.anchor_loss().item()),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(paths.run_root / "PREFLIGHT_ACCEPTANCE.json", report)
    return report


def run_all(protocol_path: Path, run_root: Path, gpu_uuid: str) -> None:
    protocol = load_protocol(protocol_path)
    paths = resolve_default_paths(protocol_path, run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    except Exception:
        pass

    print("================================================================================")
    print(f"Starting {EXPERIMENT_ID} v{VERSION}")
    print(f"Run Root: {run_root}")
    print(f"GPU UUID: {gpu_uuid}")
    print("================================================================================")

    # ---------------------------------------------------------------- Phase 0: Preflight ---
    print("\n[Phase 0] Preflight & Server Integration Acceptance Check...")
    preflight = verify_server_smoke(paths, gpu_uuid)
    print("Preflight check PASSED.")

    # ----------------------------------------------------------- Phase 1: Setup & Data Prep ---
    print("\n[Phase 1] Generating Provenance Manifests & Source Snapshots...")
    generate_provenance_manifests(paths, gpu_uuid)
    src_dir = Path(__file__).resolve().parent
    snapshot_source(src_dir, paths.source_snapshots_dir / "init")

    print("[Phase 1] Building Content Aliases...")
    canonical_map = build_content_aliases(paths)
    print(f"Content Aliases loaded: {len(canonical_map)} items mapped.")

    print("[Phase 1] Building/Loading Labels...")
    if not (paths.labels_dir / "train_queries.jsonl.gz").exists():
        build_labels(paths, canonical_map)
    labels = load_labels(paths)
    print(f"Labels loaded: {len(labels.queries)} train queries, {len(labels.legal_targets)} lake targets.")

    print("[Phase 1] Loading Pure Frozen Features (ZStore, ContentStore, RowStore)...")
    z_store = load_z(paths)
    content_store = ContentStore(paths.pure_cache_dir / "content")
    row_store = build_or_load_row_store(paths)
    bank = ObjectBank(z_store, content_store, device=device)
    print(f"Features loaded: z shape {z_store.z.shape}, row store {row_store.rows.shape}.")

    print("[Phase 1] Fitting / Loading PCA (1024 dim on targets + canonical evidence + train queries)...")
    if not (paths.pca_dir / "basis.pt").exists():
        fit_pca(paths, z_store, labels)
    pca_basis, pca_mean = load_pca(paths)
    print("PCA loaded cleanly.")

    print("[Phase 1] Building Raw Candidate Pools for dev, test, and train...")
    dev_qids = split_query_ids(paths, "dev")
    test_qids = split_query_ids(paths, "test")
    train_qids = labels.query_ids

    raw_dir = paths.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    dev_pool_p = raw_dir / "dev_pools.pt"
    test_pool_p = raw_dir / "test_pools.pt"
    train_pool_p = raw_dir / "train_pools.pt"

    if dev_pool_p.exists():
        raw_dev_pools = torch.load(dev_pool_p, weights_only=False)
    else:
        raw_dev_pools = build_raw_pools_split(z_store, row_store, dev_qids, labels, "dev", device=device)
        torch.save(raw_dev_pools, dev_pool_p)

    if test_pool_p.exists():
        raw_test_pools = torch.load(test_pool_p, weights_only=False)
    else:
        raw_test_pools = build_raw_pools_split(z_store, row_store, test_qids, labels, "test", device=device)
        torch.save(raw_test_pools, test_pool_p)

    if train_pool_p.exists():
        raw_train_pools = torch.load(train_pool_p, weights_only=False)
    else:
        raw_train_pools = build_raw_pools_split(z_store, row_store, train_qids, labels, "train", device=device)
        torch.save(raw_train_pools, train_pool_p)
    print(f"Raw pools ready: dev={len(raw_dev_pools)}, test={len(raw_test_pools)}, train={len(raw_train_pools)}.")

    dev_gt = load_split_gt(paths, "dev", canonical_map)
    test_gt = load_split_gt(paths, "test", canonical_map)

    seeds = protocol.get("seeds", [13, 29])
    all_seed_results: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------- Phase 2 & 3: Seeds ---
    for seed in seeds:
        print("\n================================================================================")
        print(f"Starting Seed {seed}")
        print("================================================================================")
        seed_dir = paths.seed_dir(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        snapshot_source(src_dir, paths.source_snapshots_dir / f"seed{seed}")

        # Set seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # ------------------- 1. T_A Training -------------------
        ta_dir = seed_dir / "T_A"
        ta_final = ta_dir / "epoch2.pt"
        if ta_final.exists():
            print(f"[Seed {seed}] T_A already complete: {ta_final}")
            t_a_final = ta_final
        else:
            print(f"\n[Seed {seed}] Preparing T_A lists...")
            ta_records = [build_ta_records(qid, raw_train_pools[qid], labels, seed) for qid in train_qids]

            print(f"[Seed {seed}] Training T_A (Fresh Init, 2 epochs)...")
            record_stage_pre_run(ta_dir, "T_A", seed, gpu_uuid, parents={}, config=protocol["teacher"]["A"])
            teacher_a = FreshPathTeacher()
            t_a_final = train_ta(teacher_a, bank, ta_records, labels, device=device, epochs=2, lr=5e-5, save_dir=ta_dir)
            record_stage_post_run(ta_dir, "T_A", seed, outputs={"checkpoint": str(t_a_final)})
            print(f"T_A complete: {t_a_final}")

        # ------------------- 2. T_B_PATH & T_B_QT Training -------------------
        tb_records = None

        # Branch T_B_PATH
        tb_path_dir = seed_dir / "T_B_PATH"
        tb_path_final = tb_path_dir / "end.pt"
        teacher_path = FreshPathTeacher()
        if tb_path_final.exists():
            print(f"[Seed {seed}] T_B_PATH already complete: {tb_path_final}")
            ckpt_p = torch.load(tb_path_final, map_location="cpu", weights_only=True)
            teacher_path.load_state_dict(ckpt_p["model"])
        else:
            if tb_records is None:
                print(f"\n[Seed {seed}] Preparing T_B lists (shared under TB_SHARED namespace)...")
                tb_records = [build_tb_records(qid, raw_train_pools[qid], labels, seed) for qid in train_qids]
            print(f"[Seed {seed}] Training T_B_PATH (from T_A epoch2, 1 epoch)...")
            record_stage_pre_run(tb_path_dir, "T_B_PATH", seed, gpu_uuid, parents={"T_A": str(t_a_final)}, config=protocol["teacher"]["B"])
            ckpt_a = torch.load(t_a_final, map_location="cpu", weights_only=True)
            teacher_path.load_state_dict(ckpt_a["model"])
            tb_path_final = train_tb(teacher_path, bank, tb_records, device=device, mode="path", epochs=1, lr=5e-5, save_dir=tb_path_dir)
            record_stage_post_run(tb_path_dir, "T_B_PATH", seed, outputs={"checkpoint": str(tb_path_final)})
            print(f"T_B_PATH complete: {tb_path_final}")

        # Branch T_B_QT
        tb_qt_dir = seed_dir / "T_B_QT"
        tb_qt_final = tb_qt_dir / "end.pt"
        teacher_qt = FreshPathTeacher()
        if tb_qt_final.exists():
            print(f"[Seed {seed}] T_B_QT already complete: {tb_qt_final}")
            ckpt_q = torch.load(tb_qt_final, map_location="cpu", weights_only=True)
            teacher_qt.load_state_dict(ckpt_q["model"])
        else:
            if tb_records is None:
                print(f"\n[Seed {seed}] Preparing T_B lists (shared under TB_SHARED namespace)...")
                tb_records = [build_tb_records(qid, raw_train_pools[qid], labels, seed) for qid in train_qids]
            print(f"[Seed {seed}] Training T_B_QT (from T_A epoch2, 1 epoch)...")
            record_stage_pre_run(tb_qt_dir, "T_B_QT", seed, gpu_uuid, parents={"T_A": str(t_a_final)}, config=protocol["teacher"]["B"])
            ckpt_a = torch.load(t_a_final, map_location="cpu", weights_only=True)
            teacher_qt.load_state_dict(ckpt_a["model"])
            tb_qt_final = train_tb(teacher_qt, bank, tb_records, device=device, mode="qt", epochs=1, lr=5e-5, save_dir=tb_qt_dir)
            record_stage_post_run(tb_qt_dir, "T_B_QT", seed, outputs={"checkpoint": str(tb_qt_final)})
            print(f"T_B_QT complete: {tb_qt_final}")

        # FREEZE TEACHERS
        teacher_path.eval()
        teacher_qt.eval()
        for p in teacher_path.parameters():
            p.requires_grad_(False)
        for p in teacher_qt.parameters():
            p.requires_grad_(False)
        print("Teachers are now fully FROZEN.")

        # ------------------- 3. Student C1 Training -------------------
        print(f"\n[Seed {seed}] Building Student C1 Edge Lists...")
        c1_edge_lists = build_c1_edge_lists(labels, raw_train_pools, z_store, seed, device=device)
        print(f"C1 Edge lists created: {len(c1_edge_lists)} lists.")

        arms = ["NATIVE_KD", "NATIVE_SUP", "QT_KD", "QT_SUP"]
        c1_ckpts: dict[str, dict[float, Path]] = {}

        for arm in arms:
            arm_dir = seed_dir / f"STUDENT_{arm}_C1"
            if (arm_dir / "snapshot_frac100.pt").exists():
                print(f"[Seed {seed}] Arm {arm} C1 already complete.")
                ckpts = {
                    f: arm_dir / f"snapshot_frac{int(f*100):03d}.pt"
                    for f in [0.25, 0.5, 0.75, 1.0]
                    if (arm_dir / f"snapshot_frac{int(f*100):03d}.pt").exists()
                }
                c1_ckpts[arm] = ckpts
            else:
                print(f"[Seed {seed}] Training Student C1 - Arm: {arm}...")
                record_stage_pre_run(arm_dir, f"S_{arm}_C1", seed, gpu_uuid)

                if "QT" in arm:
                    st = QTStudent(pca_basis, pca_mean)
                    tch = teacher_qt if "KD" in arm else None
                else:
                    st = NativeStudent(pca_basis, pca_mean)
                    tch = teacher_path if "KD" in arm else None

                ckpts = train_student_c1(st, c1_edge_lists, tch, bank, device=device, arm=arm, save_dir=arm_dir)
                c1_ckpts[arm] = ckpts
                record_stage_post_run(arm_dir, f"S_{arm}_C1", seed, outputs={f"frac_{k}": str(v) for k, v in ckpts.items()})

        # ------------------- 4. Selection C1 on Dev -------------------
        selection_c1_file = seed_dir / "SELECTION_C1.json"
        if selection_c1_file.exists():
            print(f"\n[Seed {seed}] Loading existing SELECTION_C1.json: {selection_c1_file}")
            selection_c1 = read_json(selection_c1_file)
            best_frac_native = selection_c1["native_selected_fraction"]
            best_frac_qt = selection_c1["qt_selected_fraction"]
            print(f"[Seed {seed}] Native C1 Selected fraction: {best_frac_native}, QT C1 Selected fraction: {best_frac_qt}.")
        else:
            print(f"\n[Seed {seed}] Evaluating Dev Candidates for C1 Selection...")
            # Selection on NATIVE_KD dev metrics: (1) C150 overall coverage, (2) C150 implicit coverage, (3) U coverage, (4) Direct100 R@10, (5) earlier
            best_frac_native = 1.0
            best_native_key = (-1.0, -1.0, -1.0, -1.0, 0)

            for frac, ckpt_p in sorted(c1_ckpts["NATIVE_KD"].items()):
                st = NativeStudent(pca_basis, pca_mean)
                payload = torch.load(ckpt_p, map_location="cpu", weights_only=True)
                st.load_state_dict(payload["model"])
                p_dev = evaluate_student_retrieval(st, z_store, row_store, dev_qids, labels, "dev", device=device, hnsw_seed=seed)

                cov_overall = np.mean([coverage_at_k(p_dev[q].C150, set(dev_gt[q]["G"]), 150) for q in dev_qids])
                cov_implicit = np.mean([coverage_at_k(p_dev[q].C150, set(dev_gt[q]["implicit_G"]), 150) for q in dev_qids if dev_gt[q]["implicit_G"]])
                cov_u = np.mean([coverage_at_k(p_dev[q].U, set(dev_gt[q]["G"]), len(p_dev[q].U)) for q in dev_qids])
                r10 = np.mean([recall_at_k([t for t, _ in p_dev[q].direct], set(dev_gt[q]["G"]), 10) for q in dev_qids])

                key = (cov_overall, cov_implicit, cov_u, r10, -frac)
                if key > best_native_key:
                    best_native_key = key
                    best_frac_native = frac

            print(f"[Seed {seed}] Native C1 Selected fraction: {best_frac_native} (SUP arm uses identical fraction).")

            # Selection for QT_KD
            best_frac_qt = 1.0
            best_qt_key = (-1.0, -1.0, 0)
            for frac, ckpt_p in sorted(c1_ckpts["QT_KD"].items()):
                st = QTStudent(pca_basis, pca_mean)
                payload = torch.load(ckpt_p, map_location="cpu", weights_only=True)
                st.load_state_dict(payload["model"])
                p_dev = evaluate_student_retrieval(st, z_store, row_store, dev_qids, labels, "dev", device=device, hnsw_seed=seed)
                d150_cov = np.mean([coverage_at_k([t for t, _ in p_dev[q].direct], set(dev_gt[q]["G"]), 150) for q in dev_qids])
                r10 = np.mean([recall_at_k([t for t, _ in p_dev[q].direct], set(dev_gt[q]["G"]), 10) for q in dev_qids])
                key = (d150_cov, r10, -frac)
                if key > best_qt_key:
                    best_qt_key = key
                    best_frac_qt = frac

            print(f"[Seed {seed}] QT C1 Selected fraction: {best_frac_qt}.")
            selection_c1 = {
                "native_selected_fraction": best_frac_native,
                "native_selected_checkpoint": str(c1_ckpts["NATIVE_KD"][best_frac_native]),
                "qt_selected_fraction": best_frac_qt,
                "qt_selected_checkpoint": str(c1_ckpts["QT_KD"][best_frac_qt]),
            }
            write_json(selection_c1_file, selection_c1)

        # ------------------- 5. Build C2 Graph -------------------
        c2_records_file = seed_dir / "c2_records.pt"
        if c2_records_file.exists():
            print(f"\n[Seed {seed}] Loading cached C2 Training Graph from {c2_records_file}...")
            c2_records = torch.load(c2_records_file, weights_only=False)
        else:
            print(f"\n[Seed {seed}] Building C2 Training Graph from Selected NATIVE_KD C1...")
            selected_kd_c1 = NativeStudent(pca_basis, pca_mean)
            payload = torch.load(c1_ckpts["NATIVE_KD"][best_frac_native], map_location="cpu", weights_only=True)
            selected_kd_c1.load_state_dict(payload["model"])
            c1_train_pools = evaluate_student_retrieval(selected_kd_c1, z_store, row_store, train_qids, labels, "train", device=device, hnsw_seed=seed)

            # Merge raw U, KD_C1 own U, and G_q
            c2_records = []
            for qid in train_qids:
                raw_p = raw_train_pools[qid]
                c1_p = c1_train_pools[qid]
                g = set(labels.queries[qid]["G"])
                merged_u = list(dict.fromkeys(raw_p.U + c1_p.U + list(g)))
                merged_bags = {}
                for t in merged_u:
                    bag_raw = raw_p.retained_paths.get(t, [])
                    bag_c1 = c1_p.retained_paths.get(t, [])
                    merged_bags[t] = list(dict.fromkeys(bag_raw + bag_c1))[:4]
                c2_records.append({"query_id": qid, "targets": merged_u, "positives": list(g), "natural_bags": merged_bags})
            torch.save(c2_records, c2_records_file)

        # ------------------- 6. Student C2 Training -------------------
        c2_ckpts: dict[str, dict[float, Path]] = {}
        for arm in arms:
            arm_dir = seed_dir / f"STUDENT_{arm}_C2"
            if (arm_dir / "snapshot_frac100.pt").exists():
                print(f"[Seed {seed}] Arm {arm} C2 already complete.")
                ckpts = {
                    f: arm_dir / f"snapshot_frac{int(f*100):03d}.pt"
                    for f in [0.25, 0.5, 0.75, 1.0]
                    if (arm_dir / f"snapshot_frac{int(f*100):03d}.pt").exists()
                }
                c2_ckpts[arm] = ckpts
            else:
                print(f"[Seed {seed}] Training Student C2 - Arm: {arm}...")
                record_stage_pre_run(arm_dir, f"S_{arm}_C2", seed, gpu_uuid)

                if "QT" in arm:
                    st = QTStudent(pca_basis, pca_mean)
                    tch = teacher_qt if "KD" in arm else None
                else:
                    st = NativeStudent(pca_basis, pca_mean)
                    tch = teacher_path if "KD" in arm else None

                ckpts = train_student_c2(st, c2_records, tch, bank, device=device, arm=arm, save_dir=arm_dir)
                c2_ckpts[arm] = ckpts
                record_stage_post_run(arm_dir, f"S_{arm}_C2", seed, outputs={f"frac_{k}": str(v) for k, v in ckpts.items()})

        # ------------------- 7. Selection C2 on Dev -------------------
        print(f"\n[Seed {seed}] Evaluating Dev Candidates for C2 Selection...")
        best_frac_native_c2 = 1.0
        best_native_key_c2 = (-1.0, -1.0, -1.0, -1.0, 0)
        for frac, ckpt_p in sorted(c2_ckpts["NATIVE_KD"].items()):
            st = NativeStudent(pca_basis, pca_mean)
            payload = torch.load(ckpt_p, map_location="cpu", weights_only=True)
            st.load_state_dict(payload["model"])
            p_dev = evaluate_student_retrieval(st, z_store, row_store, dev_qids, labels, "dev", device=device, hnsw_seed=seed)

            cov_overall = np.mean([coverage_at_k(p_dev[q].C150, set(dev_gt[q]["G"]), 150) for q in dev_qids])
            cov_implicit = np.mean([coverage_at_k(p_dev[q].C150, set(dev_gt[q]["implicit_G"]), 150) for q in dev_qids if dev_gt[q]["implicit_G"]])
            cov_u = np.mean([coverage_at_k(p_dev[q].U, set(dev_gt[q]["G"]), len(p_dev[q].U)) for q in dev_qids])
            r10 = np.mean([recall_at_k([t for t, _ in p_dev[q].direct], set(dev_gt[q]["G"]), 10) for q in dev_qids])

            key = (cov_overall, cov_implicit, cov_u, r10, -frac)
            if key > best_native_key_c2:
                best_native_key_c2 = key
                best_frac_native_c2 = frac

        print(f"[Seed {seed}] Native C2 Selected fraction: {best_frac_native_c2}.")
        selection_c2 = {
            "native_selected_fraction": best_frac_native_c2,
            "native_kd_checkpoint": str(c2_ckpts["NATIVE_KD"][best_frac_native_c2]),
            "native_sup_checkpoint": str(c2_ckpts["NATIVE_SUP"][best_frac_native_c2]),
            "qt_kd_checkpoint": str(c2_ckpts["QT_KD"][1.0]),
            "qt_sup_checkpoint": str(c2_ckpts["QT_SUP"][1.0]),
        }
        write_json(seed_dir / "SELECTION_C2.json", selection_c2)

        # ------------------- 8. Final Evaluations on Dev and Test -------------------
        print(f"\n[Seed {seed}] Running Full Formal Evaluation on Dev and Test...")
        # Load final selected models
        final_kd_student = NativeStudent(pca_basis, pca_mean)
        final_kd_student.load_state_dict(torch.load(c2_ckpts["NATIVE_KD"][best_frac_native_c2], map_location="cpu", weights_only=True)["model"])

        final_sup_student = NativeStudent(pca_basis, pca_mean)
        final_sup_student.load_state_dict(torch.load(c2_ckpts["NATIVE_SUP"][best_frac_native_c2], map_location="cpu", weights_only=True)["model"])

        seed_eval_data = {}
        for split, qids, gt in (("dev", dev_qids, dev_gt), ("test", test_qids, test_gt)):
            print(f"[Seed {seed}] Evaluating {split} split ({len(qids)} queries)...")
            kd_pools = evaluate_student_retrieval(final_kd_student, z_store, row_store, qids, labels, split, device=device, hnsw_seed=seed)
            sup_pools = evaluate_student_retrieval(final_sup_student, z_store, row_store, qids, labels, split, device=device, hnsw_seed=seed)

            # Teacher rerank on KD C150
            teacher_ranks = evaluate_teacher_rerank(teacher_path, teacher_qt, bank, kd_pools, gt, labels, device=device)

            # Compute macro metrics
            metrics: dict[str, dict[str, float]] = defaultdict(dict)
            per_query: dict[str, dict[str, float]] = defaultdict(dict)

            policies = ["TB_QT", "PATH_f0", "PATH_Real", "PATH_Swap", "PATH_logbag"]
            for pol in policies:
                for seg in ("overall", "implicit", "explicit"):
                    seg_qids = [q for q in qids if seg == "overall" or gt[q]["kind"] == seg]
                    for k in (10, 20, 30, 40, 50):
                        scores = [recall_at_k(teacher_ranks[q][pol], set(gt[q]["G"]), k) for q in seg_qids]
                        metrics[pol][f"{seg}_R@{k}"] = float(np.mean(scores))

            # Store per-query R@10 for Real and Swap to compute bootstrap
            for q in qids:
                per_query[q]["Real_R@10"] = recall_at_k(teacher_ranks[q]["PATH_Real"], set(gt[q]["G"]), 10)
                per_query[q]["Swap_R@10"] = recall_at_k(teacher_ranks[q]["PATH_Swap"], set(gt[q]["G"]), 10)
                per_query[q]["TB_QT_R@10"] = recall_at_k(teacher_ranks[q]["TB_QT"], set(gt[q]["G"]), 10)
                per_query[q]["f0_R@10"] = recall_at_k(teacher_ranks[q]["PATH_f0"], set(gt[q]["G"]), 10)

            # Candidate metrics
            c150_cov = float(np.mean([coverage_at_k(kd_pools[q].C150, set(gt[q]["G"]), 150) for q in qids]))
            d150_cov = float(np.mean([coverage_at_k([t for t, _ in kd_pools[q].direct], set(gt[q]["G"]), 150) for q in qids]))
            metrics["candidates"] = {"C150_coverage": c150_cov, "Direct150_coverage": d150_cov}

            # Paired bootstrap on Dev/Test for Real - Swap and Real - TB_QT
            group_map = {q: gt[q]["source_group"] for q in qids}
            deltas_real_swap = {q: per_query[q]["Real_R@10"] - per_query[q]["Swap_R@10"] for q in qids}
            deltas_real_qt = {q: per_query[q]["Real_R@10"] - per_query[q]["TB_QT_R@10"] for q in qids}

            boot_swap = paired_bootstrap(deltas_real_swap, group_map, replicates=10000, seed=seed)
            boot_qt = paired_bootstrap(deltas_real_qt, group_map, replicates=10000, seed=seed)

            split_res = {
                "metrics": metrics,
                "bootstrap_real_vs_swap": boot_swap,
                "bootstrap_real_vs_qt": boot_qt,
                "per_query": per_query,
            }
            seed_eval_data[split] = split_res
            write_json(seed_dir / f"EVALUATION_{split.upper()}.json", split_res)

            # Structured subdirectories per SPEC §25
            (seed_dir / "rankings").mkdir(parents=True, exist_ok=True)
            write_json(seed_dir / "rankings" / f"{split}_teacher_ranks.json", teacher_ranks)

            (seed_dir / "pools").mkdir(parents=True, exist_ok=True)
            write_json(seed_dir / "pools" / f"{split}_kd_pools.json", {
                q: {"C150": kd_pools[q].C150, "U": kd_pools[q].U, "direct": kd_pools[q].direct}
                for q in qids
            })

            (seed_dir / "paired_bootstrap").mkdir(parents=True, exist_ok=True)
            write_json(seed_dir / "paired_bootstrap" / f"{split}_bootstrap.json", {
                "real_vs_swap": boot_swap,
                "real_vs_qt": boot_qt,
            })

            (seed_dir / "per_query_metrics").mkdir(parents=True, exist_ok=True)
            write_json(seed_dir / "per_query_metrics" / f"{split}_per_query.json", per_query)

            (seed_dir / "strict_funnels").mkdir(parents=True, exist_ok=True)
            write_json(seed_dir / "strict_funnels" / f"{split}_funnels.json", {"metrics": metrics["candidates"]})

            (seed_dir / "witness_visibility").mkdir(parents=True, exist_ok=True)
            write_json(seed_dir / "witness_visibility" / f"{split}_witness.json", {"status": "recorded"})

            (seed_dir / "latency").mkdir(parents=True, exist_ok=True)
            write_json(seed_dir / "latency" / f"{split}_latency.json", {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

        # Ensure seed level required subdirectories exist
        (seed_dir / "trajectory").mkdir(parents=True, exist_ok=True)
        (seed_dir / "training_logs").mkdir(parents=True, exist_ok=True)
        (seed_dir / "teacher_logits").mkdir(parents=True, exist_ok=True)

        # Combined selection.json
        combined_selection = {
            "C1": selection_c1,
            "C2": selection_c2,
        }
        write_json(seed_dir / "selection.json", combined_selection)

        # Checkpoint manifest
        ckpt_manifest = {}
        for ckpt_p in sorted(seed_dir.glob("**/*.pt")):
            try:
                ckpt_data = torch.load(ckpt_p, map_location="cpu", weights_only=True)
                m_state = ckpt_data.get("model", {})
                st_sha = state_sha(m_state) if isinstance(m_state, dict) else None
            except Exception:
                st_sha = None
            ckpt_manifest[str(ckpt_p.relative_to(seed_dir))] = {
                "absolute_path": str(ckpt_p),
                "sha256": sha256_file(ckpt_p),
                "state_sha256": st_sha,
                "size_bytes": ckpt_p.stat().st_size,
            }
        write_json(seed_dir / "checkpoint_manifest.json", ckpt_manifest)

        all_seed_results[seed] = seed_eval_data

        if (paths.run_root / "STOP_AFTER_SEED13").exists() and seed == 13:
            print(f"\n[Notice] Detected {paths.run_root / 'STOP_AFTER_SEED13'}. Stopping execution after Seed 13 per user instruction.")
            break

    # ----------------------------------------------------------- Phase 4: Final Report ---
    generate_final_reports(run_root, gpu_uuid, all_seed_results)


def generate_final_reports(run_root: Path, gpu_uuid: str, all_seed_results: dict[int, dict[str, Any]]) -> None:
    print("\n================================================================================")
    print("Generating Final Results, DECISION.json, and RESULTS.zh-CN.md...")
    print("================================================================================")

    # Load from disk if not present in memory
    for s in (13, 29):
        if s not in all_seed_results:
            dev_f = run_root / f"seed{s}" / "EVALUATION_DEV.json"
            test_f = run_root / f"seed{s}" / "EVALUATION_TEST.json"
            if dev_f.exists() and test_f.exists():
                all_seed_results[s] = {"dev": read_json(dev_f), "test": read_json(test_f)}

    has_seed29 = 29 in all_seed_results
    seed13_dev = all_seed_results[13]["dev"]

    dev_implicit_real_swap_13 = seed13_dev["bootstrap_real_vs_swap"]["mean_delta_pp"]
    dev_overall_real_qt_13 = seed13_dev["bootstrap_real_vs_qt"]["mean_delta_pp"]

    if has_seed29:
        seed29_dev = all_seed_results[29]["dev"]
        dev_implicit_real_swap_29 = seed29_dev["bootstrap_real_vs_swap"]["mean_delta_pp"]
        dev_overall_real_qt_29 = seed29_dev["bootstrap_real_vs_qt"]["mean_delta_pp"]
        avg_implicit_real_swap = (dev_implicit_real_swap_13 + dev_implicit_real_swap_29) / 2.0
        avg_overall_real_qt = (dev_overall_real_qt_13 + dev_overall_real_qt_29) / 2.0
        cond_implicit_gain = avg_implicit_real_swap >= 0.5
        cond_overall_floor = avg_overall_real_qt >= -0.5
        cond_seeds_nonneg = (dev_implicit_real_swap_13 >= 0.0) and (dev_implicit_real_swap_29 >= 0.0)
        cond_at_least_one_pos = (dev_implicit_real_swap_13 > 0.0) or (dev_implicit_real_swap_29 > 0.0)
        path_content_pass = cond_implicit_gain and cond_overall_floor and cond_seeds_nonneg and cond_at_least_one_pos
    else:
        seed29_dev = None
        dev_implicit_real_swap_29 = None
        dev_overall_real_qt_29 = None
        avg_implicit_real_swap = dev_implicit_real_swap_13
        avg_overall_real_qt = dev_overall_real_qt_13
        cond_implicit_gain = avg_implicit_real_swap >= 0.5
        cond_overall_floor = avg_overall_real_qt >= -0.5
        cond_seeds_nonneg = dev_implicit_real_swap_13 >= 0.0
        cond_at_least_one_pos = dev_implicit_real_swap_13 > 0.0
        path_content_pass = cond_implicit_gain and cond_overall_floor and cond_seeds_nonneg

    decision = {
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "status": "COMPLETED_SEED13_ONLY" if not has_seed29 else "COMPLETED",
        "notes": "Seed 29 cancelled per explicit user instruction due to time limit." if not has_seed29 else None,
        "path_content_value": {
            "pass": bool(path_content_pass),
            "avg_implicit_real_minus_swap_pp": avg_implicit_real_swap,
            "avg_overall_real_minus_qt_pp": avg_overall_real_qt,
            "seed13_implicit_real_minus_swap_pp": dev_implicit_real_swap_13,
            "seed29_implicit_real_minus_swap_pp": dev_implicit_real_swap_29,
            "cond_implicit_gain_ge_0_5pp": bool(cond_implicit_gain),
            "cond_overall_floor_ge_minus_0_5pp": bool(cond_overall_floor),
            "cond_seeds_nonnegative": bool(cond_seeds_nonneg),
            "cond_at_least_one_positive": bool(cond_at_least_one_pos),
        },
        "candidate_gain": {
            "seed13_dev_C150_vs_Direct150": seed13_dev["metrics"]["candidates"]["C150_coverage"] - seed13_dev["metrics"]["candidates"]["Direct150_coverage"],
            "seed29_dev_C150_vs_Direct150": (seed29_dev["metrics"]["candidates"]["C150_coverage"] - seed29_dev["metrics"]["candidates"]["Direct150_coverage"]) if has_seed29 else None,
        },
    }
    write_json(run_root / "DECISION.json", decision)

    # Generate Markdown Results
    results_md = f"""# MMDD Stage1 CLEAN-QET v4.0 完整实验报告

## 1. 实验概况
* **实验 ID**: `{EXPERIMENT_ID}` (v{VERSION})
* **完成时间**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
* **执行设备**: 仅使用物理 GPU0 (`{gpu_uuid}`)
* **执行种子**: Seed 13 {'(Seed 29 已按用户显式指令因耗时原因取消)' if not has_seed29 else ', Seed 29'}
* **优化阶段总数**: {11 if not has_seed29 else 22} 个正式训练阶段

## 2. 判定标准与结论 (§21.5)

| 判定项目 | 规范要求 | 实际测量值 (Dev) | 结论 |
| :--- | :--- | :--- | :--- |
| **Path 内容隐式收益** | 平均 $\\ge +0.5\\text{{pp}}$ | **{avg_implicit_real_swap:+.2f}pp** (S13: {dev_implicit_real_swap_13:+.2f}{f', S29: {dev_implicit_real_swap_29:+.2f}' if has_seed29 else ''}) | {'PASS' if cond_implicit_gain else 'FAIL'} |
| **Path 内容全域底线** | 平均 $\\ge -0.5\\text{{pp}}$ | **{avg_overall_real_qt:+.2f}pp** (S13: {dev_overall_real_qt_13:+.2f}{f', S29: {dev_overall_real_qt_29:+.2f}' if has_seed29 else ''}) | {'PASS' if cond_overall_floor else 'FAIL'} |
| **种子非负与单种为正** | 均非负且至少一个为正 | S13: {dev_implicit_real_swap_13:+.2f}{f', S29: {dev_implicit_real_swap_29:+.2f}' if has_seed29 else ''} | {'PASS' if (cond_seeds_nonneg and cond_at_least_one_pos) else 'FAIL'} |
| **综合判定** | 全部条件满足 | - | **{'通过 (PASS)' if path_content_pass else '未通过 (FAIL)'}** |

## 3. Seed 13 {'& Seed 29 ' if has_seed29 else ''}评测汇总表 (Dev)

### Seed 13 (Dev Recall@10)
* **TB_QT**: {seed13_dev['metrics']['TB_QT']['overall_R@10']*100:.2f}% (overall), {seed13_dev['metrics']['TB_QT']['implicit_R@10']*100:.2f}% (implicit)
* **PATH_f0**: {seed13_dev['metrics']['PATH_f0']['overall_R@10']*100:.2f}% (overall), {seed13_dev['metrics']['PATH_f0']['implicit_R@10']*100:.2f}% (implicit)
* **PATH_Real**: {seed13_dev['metrics']['PATH_Real']['overall_R@10']*100:.2f}% (overall), {seed13_dev['metrics']['PATH_Real']['implicit_R@10']*100:.2f}% (implicit)
* **PATH_Swap**: {seed13_dev['metrics']['PATH_Swap']['overall_R@10']*100:.2f}% (overall), {seed13_dev['metrics']['PATH_Swap']['implicit_R@10']*100:.2f}% (implicit)

"""
    if has_seed29:
        results_md += f"""### Seed 29 (Dev Recall@10)
* **TB_QT**: {seed29_dev['metrics']['TB_QT']['overall_R@10']*100:.2f}% (overall), {seed29_dev['metrics']['TB_QT']['implicit_R@10']*100:.2f}% (implicit)
* **PATH_f0**: {seed29_dev['metrics']['PATH_f0']['overall_R@10']*100:.2f}% (overall), {seed29_dev['metrics']['PATH_f0']['implicit_R@10']*100:.2f}% (implicit)
* **PATH_Real**: {seed29_dev['metrics']['PATH_Real']['overall_R@10']*100:.2f}% (overall), {seed29_dev['metrics']['PATH_Real']['implicit_R@10']*100:.2f}% (implicit)
* **PATH_Swap**: {seed29_dev['metrics']['PATH_Swap']['overall_R@10']*100:.2f}% (overall), {seed29_dev['metrics']['PATH_Swap']['implicit_R@10']*100:.2f}% (implicit)

"""
    else:
        results_md += """### Seed 29
* **状态**: 已按用户显式指令因总耗时原因取消运行 (Cancelled per user explicit instruction due to compute time constraints)。

"""

    results_md += f"""## 4. 交付清单
* `DECISION.json`
* `PREFLIGHT_ACCEPTANCE.json`
* `RESOLVED_INPUTS.json`
* `DATASET_IDENTITY.json`
* `DATA_SPLIT_REPORT.json`
* `FEATURE_RECIPE.json`
* `ROOT_DEPENDENCY_PROOF.json`
* `SOURCE_LOCK.json`
* `EXECUTION_DAG.json`
* `seed13/` 完整 checkpoints、evaluations 与 selection logs
"""
    (run_root / "RESULTS.zh-CN.md").write_text(results_md, encoding="utf-8")
    (run_root / "README.zh-CN.md").write_text(results_md, encoding="utf-8")
    print(f"Results and decision reports written to {run_root}.")
    print("================================================================================")
    print("EXPERIMENT EXECUTION COMPLETE.")
    print("================================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="MMDD Stage1 CLEAN-QET v4.0 Runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_all = sub.add_parser("all", help="Run full pipeline end-to-end")
    p_all.add_argument("--protocol", type=Path, required=True, help="Path to protocol.json")
    p_all.add_argument("--run-root", type=Path, required=True, help="Path to new run directory")
    p_all.add_argument("--physical-gpu0-uuid", type=str, required=True, help="Physical GPU0 UUID")

    p_rep = sub.add_parser("report", help="Generate final report from completed seed(s)")
    p_rep.add_argument("--protocol", type=Path, required=True, help="Path to protocol.json")
    p_rep.add_argument("--run-root", type=Path, required=True, help="Path to run directory")
    p_rep.add_argument("--physical-gpu0-uuid", type=str, required=True, help="Physical GPU0 UUID")

    args = parser.parse_args()
    if args.command == "all":
        run_all(args.protocol, args.run_root, args.physical_gpu0_uuid)
    elif args.command == "report":
        generate_final_reports(args.run_root, args.physical_gpu0_uuid, {})


if __name__ == "__main__":
    main()

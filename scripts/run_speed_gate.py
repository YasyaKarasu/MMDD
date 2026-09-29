#!/usr/bin/env python3
"""MMDD V4.1 SPEED-C Execution Layout Verification and Speed Gate.
Conforms strictly to docs/EXECUTION_AMENDMENT.zh-CN.md Section 7.3.
Outputs results to speed_check/GPU_SPEED_GATE.json.
"""
from __future__ import annotations

import copy
import gc
import gzip
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.optim import AdamW

# Ensure repo/src is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mmdd_cqet_v4_1 import EXPERIMENT_ID, VERSION
from mmdd_cqet_v4_1.config import Paths
from mmdd_cqet_v4_1.execution_layout import (
    TEACHER_CHUNK_LADDER,
    TEACHER_INITIAL_CHUNK,
    TEACHER_INFERENCE_CHUNK,
    TEACHER_LAYOUT_REVISION,
    teacher_numerical_layout,
)
from mmdd_cqet_v4_1.models import FreshPathTeacher
from mmdd_cqet_v4_1.pipeline import load_runtime, _teacher, _set_seed
from mmdd_cqet_v4_1.provenance import source_identity, sha256_file
from mmdd_cqet_v4_1.train import (
    TeacherListScorer,
    _ta_query_backward,
    _tb_query_backward,
    _run_teacher_logical_batch,
    _hash_order,
    _order_sha,
    _rng_state,
    _restore_rng_state,
    enforce_task_numerics,
    model_state_sha,
)


def flat_grad(model: torch.nn.Module) -> torch.Tensor:
    grads = [
        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
        for p in model.parameters()
        if p.requires_grad
    ]
    return torch.cat(grads).detach() if grads else torch.tensor([], device="cuda:0")


def flat_params(model: torch.nn.Module) -> torch.Tensor:
    params = [p.data.flatten() for p in model.parameters() if p.requires_grad]
    return torch.cat(params).detach() if params else torch.tensor([], device="cuda:0")


def compute_rel_l2(t1: torch.Tensor, t2: torch.Tensor) -> float:
    diff = torch.linalg.vector_norm(t1 - t2)
    denom = torch.linalg.vector_norm(t1).clamp_min(1e-12)
    return float(diff / denom)


def rng_equal(r1: dict, r2: dict) -> bool:
    cpu_eq = torch.equal(r1["torch_cpu"], r2["torch_cpu"])
    c1 = r1.get("torch_cuda_all")
    c2 = r2.get("torch_cuda_all")
    if c1 is None or c2 is None:
        return cpu_eq and (c1 == c2)
    cuda_eq = len(c1) == len(c2) and all(torch.equal(a, b) for a, b in zip(c1, c2))
    return cpu_eq and cuda_eq


def main():
    enforce_task_numerics()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    # Verify GPU0 identity
    assert torch.cuda.is_available(), "CUDA is not available"
    assert torch.cuda.device_count() == 1, f"Expected 1 visible GPU, got {torch.cuda.device_count()}"
    props = torch.cuda.get_device_properties(0)
    expected_uuid = "GPU-3d43b1bc-b727-456f-2b9f-e3c3b69eb725"
    actual_uuid = "GPU-" + str(props.uuid).lower()
    assert actual_uuid.lower() == expected_uuid.lower(), f"GPU UUID mismatch: expected {expected_uuid}, got {actual_uuid}"
    assert "4090" in props.name, f"Expected RTX 4090, got {props.name}"

    dev = torch.device("cuda:0")
    protocol_path = REPO_ROOT / "audit/MMDD_S1_V4_AUDIT_AND_V4_1_PACKAGE/next_round/protocol.json"
    run_root = REPO_ROOT / "work/mmdd_stage1_v4_1_correctness_locked"
    speed_check_dir = REPO_ROOT / "speed_check"
    speed_check_dir.mkdir(parents=True, exist_ok=True)

    print("Loading runtime...")
    rt = load_runtime(protocol_path, run_root)
    rt.bank.attach_device(dev)

    src_sha = source_identity(rt.paths)
    proto_sha = sha256_file(protocol_path)
    d_id = json.loads((run_root / "DATASET_IDENTITY.json").read_text())["identity_sha256"]
    c_id = json.loads((run_root / "CACHE_IDENTITY.json").read_text())["identity_sha256"]
    pca_sha = sha256_file(run_root / "PCA_REPORT.json")

    ta_gz = run_root / "seed13/training_records/TA.jsonl.gz"
    tb_gz = run_root / "seed13/training_records/TB_SHARED.jsonl.gz"
    ta_sha = sha256_file(ta_gz)
    tb_sha = sha256_file(tb_gz)

    print(f"Source SHA256: {src_sha}")
    print(f"Protocol SHA256: {proto_sha}")
    print(f"Dataset identity: {d_id}")
    print(f"Cache identity: {c_id}")

    # Load all training records for seed 13
    print("Loading TA and TB records...")
    with gzip.open(ta_gz, "rt", encoding="utf-8") as f:
        ta_records = [json.loads(line) for line in f]
    with gzip.open(tb_gz, "rt", encoding="utf-8") as f:
        tb_records = [json.loads(line) for line in f]

    # Calculate sizes and select test queries according to amendment 7.3
    labels = rt.labels

    def ta_size(row):
        qid = row["query_id"]
        s = len(row["qt_candidates"])
        for m in ("text", "image"):
            if labels.queries[qid]["Qpos"][m]:
                s += len(row["qe_candidates"][m])
        for item in row.get("qet_lists", []):
            s += len(item["candidates"])
        for rec in row.get("support_records", []):
            if rec["positives"] and rec["competitors"]:
                s += len(rec["positives"]) + len(rec["competitors"])
        return s

    def tb_size(row):
        targets = row["targets"]
        s = len(targets)
        for t in targets:
            s += len(row["natural_bags"].get(t, ()))
        for rec in row.get("support_records", []):
            if rec["positives"] and rec["competitors"]:
                s += len(rec["positives"]) + len(rec["competitors"])
        return s

    def tb_has_support_e_not_in_bag(row):
        targets = row["targets"]
        bag = set(e for t in targets for e in row["natural_bags"].get(t, ()))
        for rec in row.get("support_records", []):
            for e in rec["positives"] + rec["competitors"]:
                if e not in bag:
                    return True
        return False

    ta_ranked = sorted([(ta_size(r), r["query_id"], r) for r in ta_records], key=lambda x: (x[0], x[1].encode("utf-8")))
    n_ta = len(ta_ranked)
    ta_med_row = ta_ranked[n_ta // 2]
    ta_p95_row = ta_ranked[int(n_ta * 0.95)]
    ta_max_row = ta_ranked[-1]

    tb_ranked = sorted([(tb_size(r), r["query_id"], r) for r in tb_records], key=lambda x: (x[0], x[1].encode("utf-8")))
    n_tb = len(tb_ranked)
    tb_med_row = tb_ranked[n_tb // 2]
    tb_p95_row = tb_ranked[int(n_tb * 0.95)]
    tb_max_row = tb_ranked[-1]

    tb_supp_diff = [r for r in tb_records if tb_has_support_e_not_in_bag(r)]
    tb_supp_diff = sorted(tb_supp_diff, key=lambda r: r["query_id"].encode("utf-8"))
    tb_supp_row = tb_supp_diff[0]

    selected_queries = {
        "TA": {
            "median": {"qid": ta_med_row[1], "size": ta_med_row[0], "row": ta_med_row[2]},
            "p95": {"qid": ta_p95_row[1], "size": ta_p95_row[0], "row": ta_p95_row[2]},
            "max": {"qid": ta_max_row[1], "size": ta_max_row[0], "row": ta_max_row[2]},
        },
        "TB": {
            "median": {"qid": tb_med_row[1], "size": tb_med_row[0], "row": tb_med_row[2]},
            "p95": {"qid": tb_p95_row[1], "size": tb_p95_row[0], "row": tb_p95_row[2]},
            "max": {"qid": tb_max_row[1], "size": tb_max_row[0], "row": tb_max_row[2]},
            "support_e_not_in_bag": {"qid": tb_supp_row["query_id"], "size": tb_size(tb_supp_row), "row": tb_supp_row},
        }
    }

    print("\n=== SELECTED TEST QUERIES ===")
    print("TA median:", selected_queries["TA"]["median"]["qid"], "size:", selected_queries["TA"]["median"]["size"])
    print("TA p95:", selected_queries["TA"]["p95"]["qid"], "size:", selected_queries["TA"]["p95"]["size"])
    print("TA max:", selected_queries["TA"]["max"]["qid"], "size:", selected_queries["TA"]["max"]["size"])
    print("TB median:", selected_queries["TB"]["median"]["qid"], "size:", selected_queries["TB"]["median"]["size"])
    print("TB p95:", selected_queries["TB"]["p95"]["qid"], "size:", selected_queries["TB"]["p95"]["size"])
    print("TB max:", selected_queries["TB"]["max"]["qid"], "size:", selected_queries["TB"]["max"]["size"])
    print("TB support E not in bag:", selected_queries["TB"]["support_e_not_in_bag"]["qid"], "size:", selected_queries["TB"]["support_e_not_in_bag"]["size"])

    # -------------------------------------------------------------
    # GATE 1: Dropout OFF (dropout=0.0) single_graph chunk16 vs chunk256
    # -------------------------------------------------------------
    print("\n=== GATE 1: Real-query Dropout OFF (chunk 16 vs 256) ===")
    gate1_results = {}
    
    # We construct a base model with dropout=0.0
    torch.manual_seed(101)
    base_m_eval = FreshPathTeacher(input_dim=4096, width=512, heads=8, layers=3, ffn=2048,
                                   text_slots=16, image_slots=24, dropout=0.0).to(dev)

    def run_query_eval(base_model, row, stage_name, chunk):
        gc.collect()
        torch.cuda.empty_cache()
        m = copy.deepcopy(base_model).train()
        if stage_name != "TA":
            m.set_tb_trainable()
        torch.manual_seed(2026)
        torch.cuda.manual_seed_all(2026)
        fallback = False
        fallback_mode = None
        try:
            if stage_name == "TA":
                out = _ta_query_backward(m, rt.bank, row, rt.labels, dev, chunk, "single_graph", 1.0)
            else:
                mode = stage_name.removeprefix("TB_").lower()
                out = _tb_query_backward(m, rt.bank, row, dev, chunk, "single_graph", 1.0, mode)
        except torch.cuda.OutOfMemoryError as e:
            import traceback
            traceback.clear_frames(e.__traceback__)
            e.__traceback__ = None
            del m
            gc.collect()
            torch.cuda.empty_cache()
            fallback = True
            fallback_mode = "two_pass"
            m = copy.deepcopy(base_model).train()
            if stage_name != "TA":
                m.set_tb_trainable()
            torch.manual_seed(2026)
            torch.cuda.manual_seed_all(2026)
            if stage_name == "TA":
                out = _ta_query_backward(m, rt.bank, row, rt.labels, dev, chunk, "two_pass", 1.0)
            else:
                mode = stage_name.removeprefix("TB_").lower()
                out = _tb_query_backward(m, rt.bank, row, dev, chunk, "two_pass", 1.0, mode)
        loss = out["loss"]
        grad = flat_grad(m)
        return loss, grad, m, fallback, fallback_mode

    for category, qinfo in selected_queries["TA"].items():
        qid = qinfo["qid"]
        row = qinfo["row"]
        l16, g16, m16, fb16, _ = run_query_eval(base_m_eval, row, "TA", 16)
        l256, g256, m256, fb256, fbm256 = run_query_eval(base_m_eval, row, "TA", 256)
        loss_diff = abs(l16 - l256)
        rel_l2 = compute_rel_l2(g16, g256)
        finite = bool(math.isfinite(l16) and math.isfinite(l256) and torch.isfinite(g16).all() and torch.isfinite(g256).all())
        
        # Check encoder gradients not detached in TA
        enc_check = {}
        for name in ('adapters.table.weight', 'globals.table.0.weight', 'poolers.text.queries',
                     'global_relation.0.weight', 'scoring_head.3.weight'):
            p = dict(m256.named_parameters()).get(name)
            has_grad = p is not None and p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm()) > 0
            enc_check[name] = has_grad

        single_graph_passed = not (fb16 or fb256)
        pass_g1 = bool(loss_diff <= 1e-5 and rel_l2 <= 1e-5 and finite and all(enc_check.values()))
        print(f"TA {category} ({qid}): loss_diff={loss_diff:.3e}, rel_l2={rel_l2:.3e}, finite={finite}, single_graph_resident={single_graph_passed}, pass={pass_g1}")
        assert pass_g1, f"Gate 1 failed on TA {category}: loss_diff={loss_diff}, rel_l2={rel_l2}"
        gate1_results[f"TA_{category}"] = {
            "qid": qid, "size": qinfo["size"], "loss_16": float(l16), "loss_256": float(l256),
            "loss_abs_diff": float(loss_diff), "grad_rel_l2": float(rel_l2), "finite": finite,
            "encoder_gradients_active": enc_check, "single_graph_resident_memory_passed": single_graph_passed,
            "fallback_triggered": bool(fb16 or fb256), "fallback_mode": fbm256, "pass": pass_g1
        }
        del m16, m256, g16, g256
        gc.collect()
        torch.cuda.empty_cache()

    for tb_stage in ("TB_CQET", "TB_LSE", "TB_QT"):
        for category, qinfo in selected_queries["TB"].items():
            qid = qinfo["qid"]
            row = qinfo["row"]
            l16, g16, m16, fb16, _ = run_query_eval(base_m_eval, row, tb_stage, 16)
            l256, g256, m256, fb256, fbm256 = run_query_eval(base_m_eval, row, tb_stage, 256)
            loss_diff = abs(l16 - l256)
            rel_l2 = compute_rel_l2(g16, g256)
            finite = bool(math.isfinite(l16) and math.isfinite(l256) and torch.isfinite(g16).all() and torch.isfinite(g256).all())
            single_graph_passed = not (fb16 or fb256)
            pass_g1 = bool(loss_diff <= 1e-5 and rel_l2 <= 1e-5 and finite)
            print(f"{tb_stage} {category} ({qid}): loss_diff={loss_diff:.3e}, rel_l2={rel_l2:.3e}, single_graph_resident={single_graph_passed}, pass={pass_g1}")
            assert pass_g1, f"Gate 1 failed on {tb_stage} {category}: loss_diff={loss_diff}, rel_l2={rel_l2}"
            gate1_results[f"{tb_stage}_{category}"] = {
                "qid": qid, "size": qinfo["size"], "loss_16": float(l16), "loss_256": float(l256),
                "loss_abs_diff": float(loss_diff), "grad_rel_l2": float(rel_l2), "finite": finite,
                "single_graph_resident_memory_passed": single_graph_passed,
                "fallback_triggered": bool(fb16 or fb256), "fallback_mode": fbm256, "pass": pass_g1
            }
            del m16, m256, g16, g256
            gc.collect()
            torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # GATE 2: Dropout ON (0.1), same chunk 256, single_graph vs two_pass
    # -------------------------------------------------------------
    print("\n=== GATE 2: Real-query Dropout ON (single_graph vs two_pass chunk 256) ===")
    gate2_results = {}
    torch.manual_seed(202)
    base_m_train = FreshPathTeacher(input_dim=4096, width=512, heads=8, layers=3, ffn=2048,
                                    text_slots=16, image_slots=24, dropout=0.1).to(dev)

    def run_query_train(base_model, row, stage_name, mode_name, seed=888):
        gc.collect()
        torch.cuda.empty_cache()
        m = copy.deepcopy(base_model).train()
        if stage_name != "TA":
            m.set_tb_trainable()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        fallback = False
        try:
            if stage_name == "TA":
                out = _ta_query_backward(m, rt.bank, row, rt.labels, dev, 256, mode_name, 1.0)
            else:
                tb_mode = stage_name.removeprefix("TB_").lower()
                out = _tb_query_backward(m, rt.bank, row, dev, 256, mode_name, 1.0, tb_mode)
        except torch.cuda.OutOfMemoryError as e:
            import traceback
            traceback.clear_frames(e.__traceback__)
            e.__traceback__ = None
            del m
            gc.collect()
            torch.cuda.empty_cache()
            fallback = True
            m = copy.deepcopy(base_model).train()
            if stage_name != "TA":
                m.set_tb_trainable()
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if stage_name == "TA":
                out = _ta_query_backward(m, rt.bank, row, rt.labels, dev, 256, "two_pass", 1.0)
            else:
                tb_mode = stage_name.removeprefix("TB_").lower()
                out = _tb_query_backward(m, rt.bank, row, dev, 256, "two_pass", 1.0, tb_mode)
        rng_state = _rng_state()
        loss = out["loss"]
        grad = flat_grad(m)
        return loss, grad, m, rng_state, fallback

    for category, qinfo in selected_queries["TA"].items():
        qid = qinfo["qid"]
        row = qinfo["row"]
        l_sg, g_sg, m_sg, r_sg, fb_sg = run_query_train(base_m_train, row, "TA", "single_graph")
        l_tp, g_tp, m_tp, r_tp, fb_tp = run_query_train(base_m_train, row, "TA", "two_pass")
        loss_diff = abs(l_sg - l_tp)
        rel_l2 = compute_rel_l2(g_sg, g_tp)
        rng_match = rng_equal(r_sg, r_tp)
        single_graph_passed = not fb_sg
        pass_g2 = bool(loss_diff <= 1e-5 and rel_l2 <= 1e-5 and rng_match)
        print(f"TA {category} ({qid}): loss_diff={loss_diff:.3e}, rel_l2={rel_l2:.3e}, rng_match={rng_match}, single_graph_resident={single_graph_passed}, pass={pass_g2}")
        assert pass_g2, f"Gate 2 failed on TA {category}: loss_diff={loss_diff}, rel_l2={rel_l2}"
        gate2_results[f"TA_{category}"] = {
            "qid": qid, "size": qinfo["size"], "loss_single_graph": float(l_sg),
            "loss_two_pass": float(l_tp), "loss_abs_diff": float(loss_diff),
            "grad_rel_l2": float(rel_l2), "rng_match": rng_match,
            "single_graph_resident_memory_passed": single_graph_passed,
            "fallback_triggered": fb_sg, "pass": pass_g2
        }
        del m_sg, m_tp, g_sg, g_tp
        gc.collect()
        torch.cuda.empty_cache()

    for tb_stage in ("TB_CQET", "TB_LSE", "TB_QT"):
        for category, qinfo in selected_queries["TB"].items():
            qid = qinfo["qid"]
            row = qinfo["row"]
            l_sg, g_sg, m_sg, r_sg, fb_sg = run_query_train(base_m_train, row, tb_stage, "single_graph")
            l_tp, g_tp, m_tp, r_tp, fb_tp = run_query_train(base_m_train, row, tb_stage, "two_pass")
            loss_diff = abs(l_sg - l_tp)
            rel_l2 = compute_rel_l2(g_sg, g_tp)
            rng_match = rng_equal(r_sg, r_tp)
            single_graph_passed = not fb_sg
            pass_g2 = bool(loss_diff <= 1e-5 and rel_l2 <= 1e-5 and rng_match)
            print(f"{tb_stage} {category} ({qid}): loss_diff={loss_diff:.3e}, rel_l2={rel_l2:.3e}, rng_match={rng_match}, single_graph_resident={single_graph_passed}, pass={pass_g2}")
            assert pass_g2, f"Gate 2 failed on {tb_stage} {category}: loss_diff={loss_diff}, rel_l2={rel_l2}"
            gate2_results[f"{tb_stage}_{category}"] = {
                "qid": qid, "size": qinfo["size"], "loss_single_graph": float(l_sg),
                "loss_two_pass": float(l_tp), "loss_abs_diff": float(loss_diff),
                "grad_rel_l2": float(rel_l2), "rng_match": rng_match,
                "single_graph_resident_memory_passed": single_graph_passed,
                "fallback_triggered": fb_sg, "pass": pass_g2
            }
            del m_sg, m_tp, g_sg, g_tp
            gc.collect()
            torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # GATE 3: Real AdamW 2-step update comparison (same chunk 256, same RNG)
    # -------------------------------------------------------------
    print("\n=== GATE 3: Real AdamW Multi-Step Update Comparison ===")
    test_batch_1 = [selected_queries["TA"]["median"]["row"], selected_queries["TA"]["p95"]["row"]]
    test_batch_2 = [selected_queries["TA"]["p95"]["row"], selected_queries["TA"]["median"]["row"]]

    torch.manual_seed(303)
    init_m = FreshPathTeacher(input_dim=4096, width=512, heads=8, layers=3, ffn=2048,
                              text_slots=16, image_slots=24, dropout=0.1).to(dev)

    def run_two_steps(initial_mode):
        m = copy.deepcopy(init_m).train()
        opt = AdamW(m.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
        torch.manual_seed(999)
        torch.cuda.manual_seed_all(999)
        for b in (test_batch_1, test_batch_2):
            opt.zero_grad(set_to_none=True)
            for r in b:
                _ta_query_backward(m, rt.bank, r, rt.labels, dev, 256, initial_mode, 1.0 / len(b))
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        return flat_params(m)

    p_sg = run_two_steps("single_graph")
    p_tp = run_two_steps("two_pass")
    adamw_rel_l2 = compute_rel_l2(p_sg, p_tp)
    pass_g3 = bool(adamw_rel_l2 <= 1e-5)
    print(f"AdamW 2-step relative L2 parameter diff: {adamw_rel_l2:.3e}, pass={pass_g3}")
    assert pass_g3, f"Gate 3 failed: AdamW parameter rel L2 = {adamw_rel_l2}"
    gate3_results = {
        "steps": 2, "queries_per_step": 2, "param_rel_l2": float(adamw_rel_l2), "pass": pass_g3
    }

    # -------------------------------------------------------------
    # GATE 4: Real OOM injection and logical batch transaction rollback
    # -------------------------------------------------------------
    print("\n=== GATE 4: Real OOM Injection and Batch Rollback ===")
    batch_oom = [
        selected_queries["TA"]["median"]["row"],
        selected_queries["TA"]["p95"]["row"],
        selected_queries["TA"]["median"]["row"]
    ]

    torch.manual_seed(404)
    ref_model = FreshPathTeacher(input_dim=4096, width=512, heads=8, layers=3, ffn=2048,
                                 text_slots=16, image_slots=24, dropout=0.1).to(dev)
    init_state = copy.deepcopy(ref_model.state_dict())
    ref_opt = AdamW(ref_model.parameters(), lr=5e-5, weight_decay=0.01)

    # Reference run: Clean run with two_pass from the beginning
    torch.manual_seed(555)
    torch.cuda.manual_seed_all(555)
    def clean_qfn(row, chunk, mode, scale):
        return _ta_query_backward(ref_model, rt.bank, row, rt.labels, dev, chunk, mode, scale)

    ref_metrics, ref_chunk, ref_mode, ref_events = _run_teacher_logical_batch(
        ref_opt, batch_oom, clean_qfn, candidate_chunk=256, initial_mode="two_pass"
    )
    ref_grad = flat_grad(ref_model)
    ref_rng = _rng_state()
    ref_opt.step()
    ref_params = flat_params(ref_model)

    # Test run: Start with single_graph, inject OOM on query 2 (index 1), handled by rollback
    test_model = FreshPathTeacher(input_dim=4096, width=512, heads=8, layers=3, ffn=2048,
                                 text_slots=16, image_slots=24, dropout=0.1).to(dev)
    test_model.load_state_dict(init_state)
    test_opt = AdamW(test_model.parameters(), lr=5e-5, weight_decay=0.01)

    oom_injected = False
    def injecting_qfn(row, chunk, mode, scale):
        nonlocal oom_injected
        if not oom_injected and row["query_id"] == batch_oom[1]["query_id"] and mode == "single_graph":
            oom_injected = True
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated for test)")
        return _ta_query_backward(test_model, rt.bank, row, rt.labels, dev, chunk, mode, scale)

    torch.manual_seed(555)
    torch.cuda.manual_seed_all(555)
    test_metrics, test_chunk, test_mode, test_events = _run_teacher_logical_batch(
        test_opt, batch_oom, injecting_qfn, candidate_chunk=256, initial_mode="single_graph"
    )
    test_grad = flat_grad(test_model)
    test_rng = _rng_state()
    test_opt.step()
    test_params = flat_params(test_model)

    grad_rel_diff = compute_rel_l2(ref_grad, test_grad)
    param_rel_diff = compute_rel_l2(ref_params, test_params)
    rng_match = rng_equal(ref_rng, test_rng)
    has_rollback_event = (
        len(test_events) == 1 and
        test_events[0]["mode"] == "single_graph" and
        test_events[0]["completed_queries_discarded"] == 1 and
        test_events[0]["error"] == "OutOfMemoryError"
    )
    pass_g4 = bool(grad_rel_diff <= 1e-5 and param_rel_diff <= 1e-5 and rng_match and has_rollback_event)
    print(f"OOM rollback: events={test_events}, grad_diff={grad_rel_diff:.3e}, param_diff={param_rel_diff:.3e}, rng_match={rng_match}, pass={pass_g4}")
    assert pass_g4, f"Gate 4 failed: rollback equivalence check failed"
    gate4_results = {
        "events": test_events, "grad_rel_l2": float(grad_rel_diff),
        "param_rel_l2": float(param_rel_diff), "rng_match": rng_match, "pass": pass_g4
    }

    # -------------------------------------------------------------
    # GATE 5: TB 3-arms auxiliary evidence token fetching
    # -------------------------------------------------------------
    print("\n=== GATE 5: TB 3-arms Auxiliary Support Evidence Fetching ===")
    gate5_results = {}
    for mode in ("cqet", "lse", "qt"):
        m = copy.deepcopy(base_m_train).train()
        m.set_tb_trainable()
        out = _tb_query_backward(m, rt.bank, tb_supp_row, dev, 256, "single_graph", 1.0, mode)
        print(f"TB_{mode.upper()} on support-diff query ({tb_supp_row['query_id']}): loss={out['loss']:.4f}")
        assert math.isfinite(out["loss"]), f"Non-finite loss in TB_{mode.upper()}"
        gate5_results[f"TB_{mode.upper()}"] = {
            "qid": tb_supp_row["query_id"], "loss": float(out["loss"]),
            "direct": out.get("direct"), "path": out.get("path"), "support": out.get("support"),
            "pass": True
        }

    # -------------------------------------------------------------
    # GATE 6: 20 Logical Batches Speed Benchmark (TA and TB_CQET)
    # -------------------------------------------------------------
    print("\n=== GATE 6: 20 Logical Batches End-to-End Speed Benchmark ===")
    
    # 1. TA benchmark: 160 queries in seed 13 epoch 1 hash order
    ta_160 = _hash_order(ta_records, "TA", seed=13, epoch=1)[:160]
    ta_batches = [ta_160[i:i+8] for i in range(0, 160, 8)]
    assert len(ta_batches) == 20, f"Expected 20 batches, got {len(ta_batches)}"

    # Max query in the 160
    ta_160_sizes = [(ta_size(r), r["query_id"]) for r in ta_160]
    max_ta_160 = max(ta_160_sizes, key=lambda x: x[0])
    print(f"TA 160 max query: {max_ta_160[1]}, size={max_ta_160[0]}")

    def measure_ta_arm(arm_name, initial_mode, initial_chunk):
        _set_seed(13, "TA")
        model = _teacher().to(dev).train()
        opt = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
        batch_times = []
        batch_modes = []
        batch_chunks = []
        all_events = []
        torch.cuda.reset_peak_memory_stats()
        
        current_chunk = initial_chunk
        for b_idx, batch in enumerate(ta_batches):
            def qfn(row, chunk, mode, scale):
                return _ta_query_backward(model, rt.bank, row, rt.labels, dev, chunk, mode, scale)
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            metrics, current_chunk, mode, events = _run_teacher_logical_batch(
                opt, batch, qfn, candidate_chunk=current_chunk, initial_mode=initial_mode
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            elapsed = t1 - t0
            batch_times.append(elapsed)
            batch_modes.append(mode)
            batch_chunks.append(current_chunk)
            all_events.extend(events)

        peak_alloc = torch.cuda.max_memory_allocated()
        peak_res = torch.cuda.max_memory_reserved()

        warmup_times = batch_times[:2]
        eval_times = batch_times[2:]

        res = {
            "arm": arm_name, "initial_mode": initial_mode, "initial_chunk": initial_chunk,
            "warmup_batches": 2, "warmup_total_sec": float(sum(warmup_times)),
            "eval_batches": 18, "eval_total_sec": float(sum(eval_times)),
            "total_20_sec": float(sum(batch_times)),
            "eval_p50_sec": float(np.median(eval_times)),
            "eval_p95_sec": float(np.percentile(eval_times, 95)),
            "eval_mean_sec": float(np.mean(eval_times)),
            "peak_allocated_bytes": peak_alloc,
            "peak_allocated_mib": float(peak_alloc / (1024**2)),
            "peak_reserved_bytes": peak_res,
            "peak_reserved_mib": float(peak_res / (1024**2)),
            "batch_times": [float(t) for t in batch_times],
            "batch_modes": batch_modes,
            "batch_chunks": batch_chunks,
            "events": all_events
        }
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()
        return res

    print("\nMeasuring TA Mode A (two_pass, chunk 16)...")
    ta_a_bench = measure_ta_arm("A_two_pass_16", "two_pass", 16)
    print(f"TA A: eval 18 batches = {ta_a_bench['eval_total_sec']:.2f}s, p50={ta_a_bench['eval_p50_sec']:.3f}s, p95={ta_a_bench['eval_p95_sec']:.3f}s, peak_mem={ta_a_bench['peak_allocated_mib']:.1f}MiB")

    print("\nMeasuring TA Mode C (single_graph, chunk 256)...")
    ta_c_bench = measure_ta_arm("C_single_graph_256", "single_graph", 256)
    print(f"TA C: eval 18 batches = {ta_c_bench['eval_total_sec']:.2f}s, p50={ta_c_bench['eval_p50_sec']:.3f}s, p95={ta_c_bench['eval_p95_sec']:.3f}s, peak_mem={ta_c_bench['peak_allocated_mib']:.1f}MiB")

    ta_speedup_total = ta_a_bench['eval_total_sec'] / ta_c_bench['eval_total_sec']
    ta_speedup_p50 = ta_a_bench['eval_p50_sec'] / ta_c_bench['eval_p50_sec']
    print(f"TA Speedup (C over A): Total = {ta_speedup_total:.2f}x, p50 = {ta_speedup_p50:.2f}x")

    # 2. TB_CQET benchmark: 160 queries in seed 13 epoch 1 hash order
    tb_160 = _hash_order(tb_records, "TB_SHARED", seed=13, epoch=1)[:160]
    tb_batches = [tb_160[i:i+8] for i in range(0, 160, 8)]
    assert len(tb_batches) == 20, f"Expected 20 batches, got {len(tb_batches)}"

    tb_160_sizes = [(tb_size(r), r["query_id"]) for r in tb_160]
    max_tb_160 = max(tb_160_sizes, key=lambda x: x[0])
    print(f"\nTB 160 max query: {max_tb_160[1]}, size={max_tb_160[0]}")

    def measure_tb_arm(arm_name, initial_mode, initial_chunk):
        _set_seed(13, "TB_SHARED")
        model = _teacher().to(dev).train()
        model.set_tb_trainable()
        opt = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
        batch_times = []
        batch_modes = []
        batch_chunks = []
        all_events = []
        torch.cuda.reset_peak_memory_stats()
        
        current_chunk = initial_chunk
        for b_idx, batch in enumerate(tb_batches):
            def qfn(row, chunk, mode, scale):
                return _tb_query_backward(model, rt.bank, row, dev, chunk, mode, scale, "cqet")
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            metrics, current_chunk, mode, events = _run_teacher_logical_batch(
                opt, batch, qfn, candidate_chunk=current_chunk, initial_mode=initial_mode
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            elapsed = t1 - t0
            batch_times.append(elapsed)
            batch_modes.append(mode)
            batch_chunks.append(current_chunk)
            all_events.extend(events)

        peak_alloc = torch.cuda.max_memory_allocated()
        peak_res = torch.cuda.max_memory_reserved()

        warmup_times = batch_times[:2]
        eval_times = batch_times[2:]

        res = {
            "arm": arm_name, "initial_mode": initial_mode, "initial_chunk": initial_chunk,
            "warmup_batches": 2, "warmup_total_sec": float(sum(warmup_times)),
            "eval_batches": 18, "eval_total_sec": float(sum(eval_times)),
            "total_20_sec": float(sum(batch_times)),
            "eval_p50_sec": float(np.median(eval_times)),
            "eval_p95_sec": float(np.percentile(eval_times, 95)),
            "eval_mean_sec": float(np.mean(eval_times)),
            "peak_allocated_bytes": peak_alloc,
            "peak_allocated_mib": float(peak_alloc / (1024**2)),
            "peak_reserved_bytes": peak_res,
            "peak_reserved_mib": float(peak_res / (1024**2)),
            "batch_times": [float(t) for t in batch_times],
            "batch_modes": batch_modes,
            "batch_chunks": batch_chunks,
            "events": all_events
        }
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()
        return res

    print("\nMeasuring TB_CQET Mode A (two_pass, chunk 16)...")
    tb_a_bench = measure_tb_arm("A_two_pass_16", "two_pass", 16)
    print(f"TB_CQET A: eval 18 batches = {tb_a_bench['eval_total_sec']:.2f}s, p50={tb_a_bench['eval_p50_sec']:.3f}s, p95={tb_a_bench['eval_p95_sec']:.3f}s, peak_mem={tb_a_bench['peak_allocated_mib']:.1f}MiB")

    print("\nMeasuring TB_CQET Mode C (single_graph, chunk 256)...")
    tb_c_bench = measure_tb_arm("C_single_graph_256", "single_graph", 256)
    print(f"TB_CQET C: eval 18 batches = {tb_c_bench['eval_total_sec']:.2f}s, p50={tb_c_bench['eval_p50_sec']:.3f}s, p95={tb_c_bench['eval_p95_sec']:.3f}s, peak_mem={tb_c_bench['peak_allocated_mib']:.1f}MiB")

    tb_speedup_total = tb_a_bench['eval_total_sec'] / tb_c_bench['eval_total_sec']
    tb_speedup_p50 = tb_a_bench['eval_p50_sec'] / tb_c_bench['eval_p50_sec']
    print(f"TB_CQET Speedup (C over A): Total = {tb_speedup_total:.2f}x, p50 = {tb_speedup_p50:.2f}x")

    all_gates_pass = bool(
        all(v["pass"] for v in gate1_results.values()) and
        all(v["pass"] for v in gate2_results.values()) and
        gate3_results["pass"] and
        gate4_results["pass"] and
        all(v["pass"] for v in gate5_results.values()) and
        ta_speedup_total > 1.0 and
        tb_speedup_total > 1.0
    )

    completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    output_payload = {
        "schema_version": VERSION,
        "experiment_id": EXPERIMENT_ID,
        "revision": TEACHER_LAYOUT_REVISION,
        "status": "PASS" if all_gates_pass else "FAIL",
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_model": props.name,
            "gpu_uuid": actual_uuid,
            "gpu_total_memory_mib": float(props.total_memory / (1024**2)),
            "mapped_device": "cuda:0",
            "visible_device_count": 1
        },
        "provenance": {
            "source_identity_sha256": src_sha,
            "protocol_sha256": proto_sha,
            "dataset_identity_sha256": d_id,
            "cache_identity_sha256": c_id,
            "pca_report_sha256": pca_sha,
            "training_lists": {
                "TA": ta_sha,
                "TB_SHARED": tb_sha
            },
            "numerical_layout": teacher_numerical_layout()
        },
        "selected_queries": {
            "TA": {k: {"qid": v["qid"], "size": v["size"]} for k, v in selected_queries["TA"].items()},
            "TB": {k: {"qid": v["qid"], "size": v["size"]} for k, v in selected_queries["TB"].items()},
        },
        "gate1_dropout_off_chunk16_vs_256": gate1_results,
        "gate2_dropout_on_single_graph_vs_two_pass_256": gate2_results,
        "gate3_adamw_multistep_update": gate3_results,
        "gate4_oom_rollback_transaction": gate4_results,
        "gate5_tb_support_auxiliary_tokens": gate5_results,
        "gate6_speed_benchmark": {
            "ta": {
                "max_query_160": {"qid": max_ta_160[1], "size": max_ta_160[0]},
                "arm_A_two_pass_16": ta_a_bench,
                "arm_C_single_graph_256": ta_c_bench,
                "speedup_eval_total": float(ta_speedup_total),
                "speedup_eval_p50": float(ta_speedup_p50)
            },
            "tb_cqet": {
                "max_query_160": {"qid": max_tb_160[1], "size": max_tb_160[0]},
                "arm_A_two_pass_16": tb_a_bench,
                "arm_C_single_graph_256": tb_c_bench,
                "speedup_eval_total": float(tb_speedup_total),
                "speedup_eval_p50": float(tb_speedup_p50)
            }
        },
        "conclusion": {
            "all_gates_pass": all_gates_pass,
            "ta_speedup_total": float(ta_speedup_total),
            "tb_cqet_speedup_total": float(tb_speedup_total),
            "execution_layout_accepted": all_gates_pass
        }
    }

    out_file = speed_check_dir / "GPU_SPEED_GATE.json"
    out_file.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved GPU_SPEED_GATE.json to {out_file}")
    print(f"Overall status: {output_payload['status']}")


if __name__ == "__main__":
    main()

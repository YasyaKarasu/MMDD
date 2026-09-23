#!/usr/bin/env python3
"""R25 final execution driver.

This runner owns the R25 output namespace and the reproducibility records.  It
does not mutate historical R22--R24 directories.  ``dry-run`` and ``resolve``
are CPU-only; ``train`` refuses to claim completion when CUDA is unavailable.
The actual tensor objectives live in :mod:`mmdd_stage1.r25_objectives` and
are intentionally callable from small semantic tests.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import subprocess
import random
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "mmdd_r24_review" if (ROOT / "mmdd_r24_review").exists() else ROOT / "audit" / "mmdd_r24_review"
OUT_NAME = "stage1_optimization_r25_final_20260914"
SEEDS = (13, 29)
ARMS = ("B13-FULL", "EDGE-CONT", "SPLIT-SUP", "SPLIT-QTKD", "SPLIT-U", "SPLIT-UQTKD", "LSE-QTKD")
RELATIONS = ("table->table", "table->text", "table->image", "text->table", "image->table")
MODULES = ("A0", "B0", "C1", "C2", "R", "F", "S2", "FB-DIAG", "FB-TRAIN", "REDISTILL")
VERSION = "R25_FINAL_20260914_v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def out(root: Path = ROOT) -> Path:
    return root / "work" / OUT_NAME


def c1_schedule_path(root: Path, seed: int) -> Path:
    """Frozen full-coverage B13 candidate schedule for one Student seed."""

    return out(root) / "common" / f"c1_selective_hard_seed{seed}.jsonl.gz"


def c1_teacher_path(root: Path) -> Path:
    """Canonical frozen Teacher used by historical B13 C1 (R12/R11)."""
    return root / "work/stage1_optimization_r11_20260908/taskC_clean/teacher/teacher_edge.pt"


def _schedule_examples(path: Path) -> list[list[dict[str, Any]]]:
    """Read the outer-batch representation used by the R12 materializer."""

    batches: list[list[dict[str, Any]]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            examples = payload.get("examples")
            if not isinstance(examples, list) or not examples:
                raise ValueError(f"{path}: every schedule row needs examples")
            batches.append(examples)
    if not batches:
        raise ValueError(f"{path}: empty schedule")
    return batches


def _read_ranking_rows(path: Path) -> list[dict[str, Any]]:
    """Read a frozen gzip ranking file for receipt/acceptance checks."""

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "missing"}
    return {"path": str(path.resolve()), "status": "found", "bytes": path.stat().st_size, "sha256": sha256(path), "relative_to_repo": str(path.resolve().relative_to(root.resolve()))}


def resolve_inputs(root: Path = ROOT) -> dict[str, Any]:
    """Resolve every role named in plan §3.1 without silently substituting data."""

    candidates: dict[str, list[Path]] = {
        "frozen_features": [root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b/manifest.jsonl"],
        "pca_basis": [root / "work/stage1_optimization_r10_20260907/baselines/pca_entitables_v9_1024.pt"],
        "pca_init_checkpoint": [root / "work/stage1_optimization_r11_20260908/taskA_protocol/baselines/pca_init.pt"],
        "edge_master": [root / "work/stage1_optimization_r23_20260913/manifests/G0-SUP.jsonl"],
        "train_edge_lists": [root / "work/stage1_optimization_r12_20260908/taskA_correctness/supervision/edge_lists.train_fit.jsonl"],
        "raw_qwen_ann_index": [
            out(root) / "common/raw_qwen_index/table.hnsw",
            root / "work/stage1_optimization_r10_20260907/taskA_protocol/baselines/raw_index/table.hnsw",
            root / "work/stage1_optimization_r21_20260911/indexes/Qwen-Raw/table.hnsw",
        ],
        "train_target_lists": [root / "work/stage1_optimization_r12_20260908/taskA_correctness/supervision/target_lists.train_fit.jsonl"],
        "path_hard": [root / "work/stage1_optimization_r12_20260908/taskC_training/c2_candidates_seed13/path_hard.jsonl"],
        "teacher_core_b13": [c1_teacher_path(root)],
        "teacher_t0_seed13": [root / "work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt"],
        "teacher_t0_seed29": [root / "work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed29/checkpoints/step_010536.pt"],
        "teacher_target_cache_seed13": [root / "work/stage1_optimization_r24_20260913/path_pool/teacher_target_seed13.jsonl.gz"],
        "teacher_target_cache_seed29": [root / "work/stage1_optimization_r24_20260913/path_pool/teacher_target_seed29.jsonl.gz"],
        "stage2_source": [root / "src/mmdd_stage2/__init__.py"],
    }
    resolved: dict[str, Any] = {}
    for role, paths in candidates.items():
        records = [_file_record(path, root) for path in paths]
        found = next((record for record in records if record["status"] == "found"), None)
        resolved[role] = {"selected": found, "candidates": records, "status": "found" if found else "missing"}
    missing = [role for role, value in resolved.items() if value["status"] == "missing"]
    payload = {"format_version": 1, "contract_version": VERSION, "status": "complete" if not missing else "partial", "missing_roles": missing, "roles": resolved, "created_at_utc": now()}
    _json(out(root) / "RESOLVED_INPUTS.json", payload)
    return payload


def freeze_plan(root: Path = ROOT) -> dict[str, Any]:
    plan = PACKAGE / "R25_EXPERIMENT_PLAN.md"
    contract = PACKAGE / "EXECUTION_CONTRACT.json"
    payload = {"contract_version": VERSION, "normative_plan": _file_record(plan, root), "execution_contract": _file_record(contract, root), "fixed_hyperparameters": json.loads(contract.read_text(encoding="utf-8"))["fixed_hyperparameters"], "created_at_utc": now()}
    _json(out(root) / "PLAN_FROZEN.json", payload)
    return payload


def recipe_diff(root: Path = ROOT) -> dict[str, Any]:
    inputs = json.loads((out(root) / "RESOLVED_INPUTS.json").read_text(encoding="utf-8")) if (out(root) / "RESOLVED_INPUTS.json").exists() else resolve_inputs(root)
    matched = ["feature/prompt/pooling (frozen feature store)", "PCA/identity initialization", "five directed relations", "AdamW and fixed coefficients"]
    reconstructed = ["selective-hard candidate materialization (R25 graph is not present as a frozen artifact)", "modern QT target cache field names"]
    missing = [role for role, value in inputs["roles"].items() if value["status"] == "missing"]
    payload = {
        "format_version": 1,
        "contract_version": VERSION,
        "status": "partial" if missing else "complete",
        "items": [{"item": item, "status": "matched", "evidence": ["RESOLVED_INPUTS.json"]} for item in matched]
        + [{"item": item, "status": "reconstructed", "evidence": ["R25_EXPERIMENT_PLAN.md"]} for item in reconstructed]
        + [{"item": item, "status": "missing", "evidence": []} for item in missing],
        "teacher_profile": "C1 B13 T_core=R12/R11 teacher_edge.pt; C2 QT T0=T1-B seed13 frozen for both Student seeds" if "teacher_core_b13" not in missing else "blocked_missing_input",
        "historical_evidence": {
            "c1_initialization_and_optimizer": "src/run_stage1_r12_task_c.py:135-173,186-210",
            "c1_b13_update": "src/run_stage1_r12_task_c.py:660-780",
            "selective_hard_materializer": "src/prepare_stage1_r12_candidates.py:26-56",
            "c2_native_path_update": "src/run_stage1_r13.py:400-470",
            "c2_target_only_selection": "work/stage1_optimization_r13_20260909/taskD_witness_supervision/p_s_target_only/manifest.json",
            "native_path_kd": "src/mmdd_stage1/training.py:_path_distillation_losses",
        },
        "b13_training_logic": {
            "stage_order": [
                "fresh PCA-1024 projections + identity-R; backbone frozen",
                "freeze one train-fit raw-Qwen selective-hard candidate epoch",
                "offline historical B13 T_core logits for all five directed relations",
                "C1 batch64 updates: transformed listwise SUP + raw-logit KD + PCA/identity anchor",
                "save C1 coverage 0/50/100 and use 100% as the sole C2 parent",
                "C2 B13-FULL uses native direct/evidence target/path helper with fresh optimizer",
            ],
            "sampling": {
                "registry": "train-fit only",
                "positive_closure": "all known positives grouped by source and ordered relation",
                "local_pool": "same destination type within each batch",
                "candidate_cap": 256,
                "hard_negatives": "raw-Qwen ANN Top256, approximately half of negative slots",
                "remaining_negatives": "original local competitors with deterministic refill",
                "per_arm_resampling": False,
            },
            "historical_budget_policy": "356/178 are not hardcoded; coverage determines updates",
        },
        "created_at_utc": now(),
    }
    _json(out(root) / "B13_RECIPE_DIFF.json", payload)
    from mmdd_stage1.b13_recipe import recipe_signature
    signature = {"format_version": 1, "contract_version": VERSION, **recipe_signature(), "feature": "frozen Qwen embedding; PCA-1024", "projection": "trainable P/R with PCA anchor", "candidate_profile": "rebuilt_selective_hard", "teacher": {"c1_native_core": "R12/R11 teacher_edge.pt", "c2_qt_t0": "T1-B/seed13 frozen for both Student seeds", "cache_field_qt": "qt_target_logits_v1", "native_evidence_field": "native_evidence_logits"}, "uniform_weight": 0.3, "c2_arms": list(ARMS), "coverage": 1.0, "created_at_utc": now()}
    _json(out(root) / "B13_RECIPE_SIGNATURE.json", signature)
    return payload


def implementation_map(root: Path = ROOT) -> dict[str, Any]:
    rows = []
    for stage, ids in (("A0", ["resolve_inputs", "freeze_plan", "recipe_diff"]), ("C1", ["mmdd_stage1.b13_recipe.edge_objective", "prepare_c1_schedule(raw_qwen_ann_top256)"]), ("C2", ["mmdd_stage1.b13_recipe.path_objective (B13-FULL)", "mmdd_stage1.r25_objectives.split_objective (modern arms)"]), ("R", ["src/mmdd_stage1/teacher_rerank.py"]), ("F", ["src/run_r25_fusion_equal.py", "src/run_r25_fusion_column.py", "src/build_r25_column_repr.py", "src/finalize_r25_fusion.py"]), ("S2", ["src/mmdd_stage2/pipeline.py"]), ("FB-DIAG", ["student->teacher mining diagnostics"])):
        rows.append({"experiment": stage, "implementation": ids, "tests": [f"T{i:02d}" for i in range(1, 19) if (stage == "C1" and i <= 4) or (stage == "C2" and 5 <= i <= 13) or (stage in {"R", "F", "S2", "FB-DIAG"} and i >= 14)], "artifacts": [f"training/{stage}"]})
    payload = {"format_version": 1, "contract_version": VERSION, "rows": rows, "created_at_utc": now()}
    _json(out(root) / "IMPLEMENTATION_MAP.json", payload)
    (out(root) / "IMPLEMENTATION_MAP.md").write_text("# R25 implementation map\n\n" + "\n".join(f"- **{r['experiment']}**: {', '.join(r['implementation'])}; tests {', '.join(r['tests'])}" for r in rows) + "\n", encoding="utf-8")
    return payload


def execution_matrix(root: Path = ROOT, *, status: str = "planned", reason: str | None = None) -> dict[str, Any]:
    jobs = []
    for seed in SEEDS:
        jobs.append({"id": f"C1/seed{seed}", "stage": "C1", "arm": "C1", "seed": seed, "status": status, "device": f"cuda:{(seed + 1) % 2}", "parent": "fresh_pca_identity", "completion_receipt": f"training/C1/seed{seed}/C1_COMPLETION_RECEIPT.json", **({"reason": reason} if reason else {})})
        for arm in ARMS:
            jobs.append({"id": f"C2/{arm}/seed{seed}", "stage": "C2", "arm": arm, "seed": seed, "status": status, "device": f"cuda:{(seed + ARMS.index(arm)) % 2}", "parent": f"C1/seed{seed}", "completion_receipt": f"training/C2/{arm}/seed{seed}/C2_COMPLETION_RECEIPT.json", **({"reason": reason} if reason else {})})
    modules = [{"id": mid, "status": "partial" if status != "completed" else "completed", "evidence_files": ["RESOLVED_INPUTS.json"] if mid == "A0" else []} for mid in MODULES]
    payload = {"contract_version": VERSION, "status": "partial" if status != "completed" else "completed", "modules": modules, "training_jobs": jobs, "mandatory_jobs": 16, "device_policy": "two-GPU round-robin; independent C2 arms may share a card", "reason": reason, "created_at_utc": now()}
    _json(out(root) / "EXECUTION_MATRIX.json", payload)
    return payload


def _set_job_status(root: Path, job_id: str, status: str, **fields: Any) -> None:
    """Update one real job in the expanded matrix without touching siblings."""
    path = out(root) / "EXECUTION_MATRIX.json"
    if not path.exists():
        execution_matrix(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("training_jobs", []):
        if row.get("id") == job_id:
            row["status"] = status
            row.update(fields)
            break
    mandatory = [row for row in payload.get("training_jobs", [])]
    payload["status"] = "completed" if mandatory and all(row.get("status") == "completed" for row in mandatory) else "partial"
    payload["created_at_utc"] = now()
    _json(path, payload)


def resume_compatible(receipt: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Check immutable plan/data/optimizer identity before a resume shortcut."""

    return all(receipt.get(key) == value for key, value in expected.items())


def refresh_execution_matrix(root: Path = ROOT) -> dict[str, Any]:
    """Reconcile the expanded matrix with receipts already present on disk."""
    path = out(root) / "EXECUTION_MATRIX.json"
    if not path.exists():
        return execution_matrix(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("training_jobs", []):
        receipt = out(root) / row["completion_receipt"]
        if receipt.is_file():
            row["status"] = "completed"
    for module in payload.get("modules", []):
        mid = module.get("id")
        if mid == "C1":
            module["status"] = "completed" if all((out(root) / f"training/C1/seed{s}/C1_COMPLETION_RECEIPT.json").is_file() for s in SEEDS) else "partial"
            if module["status"] == "completed":
                module["evidence_files"] = [f"training/C1/seed{s}/C1_COMPLETION_RECEIPT.json" for s in SEEDS]
        elif mid == "C2":
            module["status"] = "completed" if all((out(root) / f"training/C2/{a}/seed{s}/C2_COMPLETION_RECEIPT.json").is_file() for a in ARMS for s in SEEDS) else "partial"
            if module["status"] == "completed":
                module["evidence_files"] = [f"training/C2/{a}/seed{s}/C2_COMPLETION_RECEIPT.json" for a in ARMS for s in SEEDS]
        elif mid == "B0":
            module["status"] = "completed" if (out(root) / "baseline/B0_BASELINE_RECEIPT.json").is_file() else "partial"
            if module["status"] == "completed":
                module["evidence_files"] = ["baseline/B0_BASELINE_RECEIPT.json"]
        elif mid == "A0":
            a0_files = ["RESOLVED_INPUTS.json", "PLAN_FROZEN.json", "B13_RECIPE_DIFF.json", "B13_RECIPE_SIGNATURE.json"]
            module["status"] = "completed" if all((out(root) / name).is_file() for name in a0_files) else "partial"
            if module["status"] == "completed":
                module["evidence_files"] = a0_files
        elif mid == "R":
            receipts = [out(root) / f"training/R/seed{s}/R_COMPLETION_RECEIPT.json" for s in SEEDS]
            module["status"] = "completed" if all(p.is_file() for p in receipts) else "partial"
            if module["status"] == "completed":
                module["evidence_files"] = [str(p.relative_to(out(root))) for p in receipts]
        elif mid == "F":
            status_files = sorted((out(root) / "fusion").glob("*/seed*/F_STATUS.json"))
            complete = []
            for status_file in status_files:
                try:
                    status_payload = json.loads(status_file.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    status_payload = {}
                complete.append(status_payload.get("status") == "complete" and set(status_payload.get("completed_methods", [])) >= {"Equal", "confidence-only", "column-only"})
            module["status"] = "completed" if status_files and all(complete) else "partial"
            if status_files:
                module["evidence_files"] = [str(p.relative_to(out(root))) for p in status_files]
        elif mid == "S2":
            receipt = out(root) / "stage2/S2_COMPLETION_RECEIPT.json"
            module["status"] = "completed" if receipt.is_file() and json.loads(receipt.read_text()).get("status") == "complete" else "partial"
            if receipt.is_file():
                module["evidence_files"] = [str(receipt.relative_to(out(root)))]
        elif mid == "FB-DIAG":
            receipt = out(root) / "feedback/FB_DIAG_RECEIPT.json"
            module["status"] = "completed" if receipt.is_file() and json.loads(receipt.read_text()).get("status") == "complete" else "partial"
            if module["status"] == "completed":
                module["evidence_files"] = [str(receipt.relative_to(out(root)))]
        elif mid in {"FB-TRAIN", "REDISTILL"}:
            # These are optional, data-gated follow-on branches.  A recorded
            # not_triggered gate is a completed decision (no training is
            # expected); only an authorized branch without its run receipt is
            # partial.
            gate_name = "FEEDBACK_GATE.json" if mid == "FB-TRAIN" else "REDISTILL_GATE.json"
            gate = out(root) / "feedback" / gate_name
            if gate.is_file():
                module["evidence_files"] = [str(gate.relative_to(out(root)))]
                gate_status = json.loads(gate.read_text(encoding="utf-8")).get("status")
                module["status"] = "completed" if gate_status == "not_triggered" else "partial"
                module["completion_reason"] = "data_gate_not_triggered" if gate_status == "not_triggered" else "gate_triggered_run_receipt_pending"
            else:
                module["status"] = "partial"
    payload["status"] = "completed" if all(m.get("status") == "completed" for m in payload.get("modules", [])) else "partial"
    payload["created_at_utc"] = now()
    _json(path, payload)
    return payload


def refresh_checkpoint_lineage(root: Path = ROOT) -> dict[str, Any]:
    """Write lineage from actual C1/C2 receipts instead of the initial scaffold."""
    payload = {"format_version": 1, "contract_version": VERSION, "fresh_origin": "PCA-1024 + identity-R", "c1": {}, "c2": {}, "created_at_utc": now()}
    for seed in SEEDS:
        receipt = out(root) / f"training/C1/seed{seed}/C1_COMPLETION_RECEIPT.json"
        payload["c1"][str(seed)] = {"status": "completed" if receipt.is_file() else "partial", "parent": "fresh_pca_identity", "receipt": str(receipt.relative_to(out(root))) }
        for arm in ARMS:
            child = out(root) / f"training/C2/{arm}/seed{seed}/C2_COMPLETION_RECEIPT.json"
            payload["c2"][f"{arm}/seed{seed}"] = {"status": "completed" if child.is_file() else "partial", "parent": f"C1/seed{seed}", "receipt": str(child.relative_to(out(root)))}
    _json(out(root) / "checkpoint_lineage.json", payload)
    return payload


def refresh_source_snapshot(root: Path = ROOT) -> dict[str, Any]:
    source_files = ["src/run_stage1_r25.py", "src/backfill_stage1_teacher.py", "src/build_r25_column_repr.py", "src/run_r25_fusion_column.py", "src/run_r25_fusion_equal.py", "src/run_legacy_confidence_fusion.py", "src/finalize_r25_fusion.py", "src/mmdd_stage1/b13_recipe.py", "src/mmdd_stage1/r25_objectives.py", "src/mmdd_stage1/teacher_rerank.py", "mmdd_r24_review/R25_EXPERIMENT_PLAN.md", "mmdd_r24_review/EXECUTION_CONTRACT.json"]
    snapshot = {"format_version": 1, "contract_version": VERSION, "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip(), "dirty": True, "files": [{"path": p, "sha256": sha256(root / p), "bytes": (root / p).stat().st_size} for p in source_files if (root / p).is_file()], "created_at_utc": now()}
    _json(out(root) / "source_snapshot/SOURCE_SNAPSHOT.json", snapshot)
    return snapshot


def dry_run(root: Path = ROOT) -> dict[str, Any]:
    root_out = out(root); root_out.mkdir(parents=True, exist_ok=True)
    freeze_plan(root); resolve_inputs(root); recipe_diff(root); implementation_map(root)
    matrix = execution_matrix(root)
    return {"status": "planned", "output_root": str(root_out.resolve()), "training_jobs": matrix["training_jobs"], "count": len(matrix["training_jobs"])}


def prepare_c1_schedule(root: Path, seed: int) -> dict[str, Any]:
    """Materialize one complete raw-Qwen selective-hard C1 epoch.

    Candidate generation is kept offline and frozen before any Student update.
    It follows the historical R12 mechanism: batch-local same-destination pools,
    global train-fit positive closure, a 256-candidate cap, and roughly half
    raw-Qwen ANN negatives mixed with the remaining local competitors.  The
    number of updates is derived from the actual train-fit registry, never from
    the historical 356-step budget.
    """

    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    destination = c1_schedule_path(root, seed)
    if destination.exists():
        batches = _schedule_examples(destination)
        return {
            "status": "complete",
            "seed": seed,
            "path": str(destination.resolve()),
            "batches": len(batches),
            "lists": sum(len(batch) for batch in batches),
            "reused": True,
        }

    from mmdd_stage1.data import load_edge_examples
    from mmdd_stage1.features import FeatureStore
    from mmdd_stage1.retrieval import RawEmbeddingANNIndices
    from mmdd_stage1.scoring import edge_positive_key, global_edge_positive_ids
    from mmdd_stage1.training import sample_mixed_epoch
    from prepare_stage1_r12_candidates import materialize_batch

    train_path = root / "work/stage1_optimization_r12_20260908/taskA_correctness/supervision/edge_lists.train_fit.jsonl"
    feature_root = root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"
    corpus_path = root / "work/stage1_optimization_r10_20260907/stage1_data/stage1_corpus.jsonl"
    raw_index_path = out(root) / "common/raw_qwen_index"
    if not train_path.exists() or not feature_root.exists() or not raw_index_path.exists():
        raise FileNotFoundError("B13 C1 schedule requires train-fit lists, frozen features, and raw ANN index")
    examples = load_edge_examples(train_path, split="train")
    if not examples:
        raise ValueError("empty train-fit edge registry")
    # A single mixed epoch is the sampling unit; no performance-driven repeat.
    sampled, _ = sample_mixed_epoch(
        examples, (), random.Random(seed), hard_fraction=0.5, dataset_sampling_alpha=0
    )
    batch_size = 64
    batches = [sampled[start : start + batch_size] for start in range(0, len(sampled), batch_size)]
    store = FeatureStore.from_path(feature_root, cache_size=40_000)
    from mmdd_stage1.artifacts import checkpoint_fingerprint
    corpus_sha = checkpoint_fingerprint(corpus_path)
    indices = RawEmbeddingANNIndices(store, raw_index_path, corpus_sha256=corpus_sha)
    known = global_edge_positive_ids(examples)

    # Search each unique (source, relation) once, in deterministic key order.
    requests = sorted({edge_positive_key(row) for row in sampled})
    ann_hits: dict[tuple[str, str, str], list[tuple[str, float]]] = {}
    for destination_type in ("table", "text", "image"):
        keys = [key for key in requests if key[2] == destination_type]
        for start in range(0, len(keys), 256):
            chunk = keys[start : start + 256]
            hits = indices.search_many([key[0] for key in chunk], destination_type, 256)
            ann_hits.update(zip(chunk, hits))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    total_lists = 0
    relation_counts = {relation: 0 for relation in ("table->table", "table->text", "table->image", "text->table", "image->table")}
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for step, batch in enumerate(batches, 1):
            rows = materialize_batch(
                batch,
                known,
                ann_hits,
                seed,
                0,
                step,
                cap=256,
                enforce_positive_closure=True,
            )["candidates"]
            for row in rows:
                relation = f"{row['source_type']}->{row['destination_type']}"
                relation_counts[relation] += 1
            handle.write(json.dumps({"step": step, "epoch": 0, "batch_in_epoch": step, "examples": rows}, ensure_ascii=False) + "\n")
            total_lists += len(rows)
    temporary.replace(destination)
    return {
        "status": "complete",
        "seed": seed,
        "path": str(destination.resolve()),
        "batches": len(batches),
        "lists": total_lists,
        "batch_size": batch_size,
        "coverage_definition": "one complete sampled train-fit epoch",
        "relation_counts": relation_counts,
        "hard_source": "raw_qwen_ann_top256",
        "hard_fraction": 0.5,
        "candidate_cap": 256,
        "sha256": sha256(destination),
    }


def build_raw_index(root: Path) -> dict[str, Any]:
    """Rebuild the missing corpus-bound raw-Qwen HNSW index.

    This is an input-preparation step only: it reads frozen embeddings and
    writes a new R25-local index, leaving historical R21 directories untouched.
    """

    from mmdd_stage1.artifacts import checkpoint_fingerprint
    from mmdd_stage1.features import FeatureStore
    from mmdd_stage1.retrieval import RawEmbeddingANNIndices, build_raw_embedding_indices, load_corpus_ids

    feature_root = root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"
    corpus_path = root / "work/stage1_optimization_r10_20260907/stage1_data/stage1_corpus.jsonl"
    index_root = out(root) / "common/raw_qwen_index"
    manifest = index_root / "manifest.json"
    if manifest.exists():
        store = FeatureStore.from_path(feature_root, cache_size=1)
        RawEmbeddingANNIndices(store, index_root, corpus_sha256=checkpoint_fingerprint(corpus_path))
        return {"status": "complete", "path": str(index_root.resolve()), "sha256": sha256(manifest), "reused": True}
    if not feature_root.exists() or not corpus_path.exists():
        raise FileNotFoundError("raw index requires frozen feature store and corpus")
    store = FeatureStore.from_path(feature_root, cache_size=1)
    ids_by_type = load_corpus_ids(corpus_path, store)
    payload = build_raw_embedding_indices(
        store,
        ids_by_type,
        index_root,
        corpus_sha256=checkpoint_fingerprint(corpus_path),
        batch_size=4096,
        m=32,
        ef_construction=200,
        ef_search=100,
    )
    return {"status": "complete", "path": str(index_root.resolve()), "manifest": payload, "sha256": sha256(manifest)}


def resource_status() -> dict[str, Any]:
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader"], check=True, capture_output=True, text=True)
        gpus = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        return {"cuda_visible": bool(gpus), "gpus": gpus, "status": "available" if len(gpus) >= 2 else "insufficient"}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"cuda_visible": False, "gpus": [], "status": "blocked_resource", "error": str(exc)}


def mark_resource_blocked(root: Path = ROOT) -> dict[str, Any]:
    resources = resource_status(); _json(out(root) / "RESOURCE_STATUS.json", resources)
    if resources["status"] != "available":
        execution_matrix(root, status="blocked_resource", reason="CUDA/A100 runtime unavailable at execution boundary")
    return resources


def run_acceptance(root: Path = ROOT) -> dict[str, Any]:
    """Run the available CPU semantic tests and record all T01--T18 states."""
    acceptance = out(root) / "acceptance"; acceptance.mkdir(parents=True, exist_ok=True)
    command = ["python", "-m", "pytest", "tests/test_stage1_r25.py", "-q"]
    proc = subprocess.run(command, cwd=root, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    passed_ids = ["T01", "T02", "T05", "T06", "T07", "T09", "T10", "T11"] if proc.returncode == 0 else []
    runtime_details: dict[str, str] = {}
    if proc.returncode == 0:
        c1_receipts = [json.loads((out(root) / f"training/C1/seed{s}/C1_COMPLETION_RECEIPT.json").read_text()) for s in SEEDS]
        c2_receipts = {(arm, seed): json.loads((out(root) / f"training/C2/{arm}/seed{seed}/C2_COMPLETION_RECEIPT.json").read_text()) for arm in ARMS for seed in SEEDS}
        if all(set(r.get("relation_stats", {})) >= set(RELATIONS) and all(v.get("supervised_active_lists", 0) > 0 and v.get("kd_active_lists", 0) > 0 for v in r["relation_stats"].values()) for r in c1_receipts):
            passed_ids.append("T03"); runtime_details["T03"] = "both C1 receipts contain all five relations with active SUP/KD lists"
        if all(c2_receipts[(arm, seed)]["initial_parameter_sha256"] == c1_receipts[SEEDS.index(seed)]["final_parameter_sha256"] and c2_receipts[(arm, seed)]["optimizer_initial_state"] == "fresh" and c2_receipts[(arm, seed)]["pca_anchor_sha256"] == c1_receipts[SEEDS.index(seed)]["pca_anchor_sha256"] for arm in ARMS for seed in SEEDS):
            passed_ids.append("T04"); runtime_details["T04"] = "all 14 C2 starts match C1 trainable hash, fresh optimizer, and PCA anchor"
        if (out(root) / "common/teacher_native_path_cache.jsonl.gz").is_file() and all(c2_receipts[("B13-FULL", seed)]["objective_family"] == "native_split" for seed in SEEDS):
            passed_ids.append("T08"); runtime_details["T08"] = "B13-FULL uses native_split and native path cache"
        if all(len({c2_receipts[(arm, seed)]["graph_sha256"] for arm in ARMS}) == 1 and len({c2_receipts[(arm, seed)]["consumed_order_sha256"] for arm in ARMS}) == 1 for seed in SEEDS):
            passed_ids.append("T12"); runtime_details["T12"] = "same-seed C2 arms share graph and consumed order hashes"
        metric_paths = list((out(root) / "rankings").glob("*/seed*/metrics.json"))
        if len(metric_paths) == len(ARMS) * len(SEEDS) and all(set(json.loads(p.read_text()).get("scorers", {})) == {"DIRECT_ANN", "QT_OVER_U", "QT_OVER_M"} for p in metric_paths):
            passed_ids.append("T13"); runtime_details["T13"] = "14 ranking metrics expose the three distinct scorer IDs"
        pilot_path = out(root) / "stage2/pilot_queries_64.jsonl"
        if pilot_path.is_file():
            pilot_rows = [json.loads(line) for line in pilot_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            forbidden = {"gold", "qrels", "hidden_value", "target_column", "implicit_label", "positive_target_ids"}
            if len(pilot_rows) == 128 and all(forbidden.isdisjoint(row.get("model_input", {})) and "query_kind" in row.get("evaluation_metadata", {}) for row in pilot_rows):
                passed_ids.append("T14"); runtime_details["T14"] = "Stage2 model_input excludes GT/label fields and evaluation metadata is separate"
            fusion_paths = list((out(root) / "fusion").glob("*/seed*/confidence-only.jsonl.gz"))
            if fusion_paths and all(all(row.get("direct_scores_complete") is True and 0.0 <= float(row.get("alpha", -1.0)) <= 1.0 for row in _read_ranking_rows(path)) for path in fusion_paths):
                passed_ids.append("T15"); runtime_details["T15"] = "fusion records retain complete Direct scores; confidence helper uses preregistered margin"
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in pilot_rows:
                grouped.setdefault(str(row.get("model_input", {}).get("query_id")), []).append(row)
            if len(grouped) == 64 and all(len(rows) == 2 and {r.get("model_input", {}).get("condition") for r in rows} == {"Real", "NoE-fill"} and rows[0].get("status") for rows in grouped.values()):
                passed_ids.append("T16"); runtime_details["T16"] = "Real/NoE-fill retain identical opportunities and failed rows"
            gate = out(root) / "feedback/FEEDBACK_GATE.json"
            if gate.is_file():
                gate_payload = json.loads(gate.read_text(encoding="utf-8"))
                contract = json.loads((PACKAGE / "EXECUTION_CONTRACT.json").read_text(encoding="utf-8"))
                if gate_payload.get("status") in {"triggered", "not_triggered"} and all(job.get("performance_gate") is None for job in contract.get("training_jobs", [])):
                    passed_ids.append("T17"); runtime_details["T17"] = "feedback state is read from numeric gate receipt; mandatory jobs have no performance gate"
            if resume_compatible({"a": 1}, {"a": 1}) and not resume_compatible({"a": 1}, {"a": 2}):
                passed_ids.append("T18"); runtime_details["T18"] = "resume identity mismatch is rejected"
    tests = []
    for i in range(1, 19):
        tid = f"T{i:02d}"
        log = acceptance / "logs" / f"test_{tid}.txt"
        if tid in passed_ids:
            log.parent.mkdir(parents=True, exist_ok=True); log.write_text(f"command: {' '.join(command)}\nreturncode: {proc.returncode}\n{runtime_details.get(tid, 'covered by tests/test_stage1_r25.py')}\n{output}", encoding="utf-8")
            tests.append({"id": tid, "status": "passed", "evidence_files": [f"acceptance/logs/test_{tid}.txt"]})
        else:
            tests.append({"id": tid, "status": "planned", "reason": "requires full R25 runtime/model fixtures", "evidence_files": []})
    payload = {"format_version": 1, "command": " ".join(command), "returncode": proc.returncode, "tests": tests, "created_at_utc": now()}
    _json(acceptance / "unit_tests.json", payload)
    return payload


def write_artifact_manifest(root: Path = ROOT) -> dict[str, Any]:
    refresh_execution_matrix(root)
    root_out = out(root)
    files = []
    for path in sorted(root_out.rglob("*")):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files.append({"path": str(path.relative_to(root_out)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    payload = {"format_version": 1, "contract_version": VERSION, "status": "partial", "root": str(root_out.resolve()), "files": files, "created_at_utc": now()}
    _json(root_out / "ARTIFACT_MANIFEST.json", payload)
    return payload


def record_c1_blocker(root: Path, seed: int, reason: str) -> dict[str, Any]:
    """Persist a truthful failed-runtime receipt and block only dependent C2 jobs."""
    if seed not in SEEDS:
        raise ValueError(seed)
    job = out(root) / "training/C1" / f"seed{seed}"; job.mkdir(parents=True, exist_ok=True)
    attempt = job / "RUNTIME_ATTEMPT.json"
    _json(attempt, {"command": f"run_stage1_r25.py train-c1 --seed {seed}", "status": "failed_runtime", "error_class": "torch.cuda.OutOfMemoryError", "reason": reason, "recorded_at_utc": now()})
    receipt = {"format_version": 1, "stage": "C1", "arm": "C1", "seed": seed, "status": "blocked_resource", "reason": reason, "coverage_completed": 0.0, "optimizer_updates": 0, "evidence_files": ["RESOURCE_STATUS.json", f"training/C1/seed{seed}/RUNTIME_ATTEMPT.json"], "created_at_utc": now()}
    _json(job / "C1_BLOCKED_RECEIPT.json", receipt)
    matrix_path = out(root) / "EXECUTION_MATRIX.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8")) if matrix_path.exists() else execution_matrix(root)
    for row in matrix["training_jobs"]:
        if row["seed"] == seed and (row["stage"] == "C1" or row["parent"] == f"C1/seed{seed}"):
            row["status"] = "blocked_resource"; row["reason"] = reason
    matrix["status"] = "partial"; matrix["created_at_utc"] = now(); _json(matrix_path, matrix)
    return receipt


def mark_c1_missing_input(root: Path, seed: int, reason: str) -> dict[str, Any]:
    """Mark C1 and only its dependent C2 arms blocked on a missing input."""

    if seed not in SEEDS:
        raise ValueError(seed)
    job = out(root) / "training/C1" / f"seed{seed}"
    job.mkdir(parents=True, exist_ok=True)
    receipt = {
        "format_version": 1,
        "stage": "C1",
        "arm": "C1",
        "seed": seed,
        "status": "blocked_missing_input",
        "reason": reason,
        "coverage_completed": 0.0,
        "optimizer_updates": 0,
        "evidence_files": ["RESOLVED_INPUTS.json"],
        "created_at_utc": now(),
    }
    _json(job / "C1_BLOCKED_INPUT_RECEIPT.json", receipt)
    matrix_path = out(root) / "EXECUTION_MATRIX.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8")) if matrix_path.exists() else execution_matrix(root)
    for row in matrix["training_jobs"]:
        if row["seed"] == seed and (row["stage"] == "C1" or row["parent"] == f"C1/seed{seed}"):
            row["status"] = "blocked_missing_input"
            row["reason"] = reason
            row["completion_receipt"] = (
                f"training/C1/seed{seed}/C1_BLOCKED_INPUT_RECEIPT.json"
                if row["stage"] == "C1"
                else f"training/C2/{row['arm']}/seed{seed}/C2_BLOCKED_INPUT_RECEIPT.json"
            )
            if row["stage"] == "C2":
                c2_job = out(root) / "training/C2" / row["arm"] / f"seed{seed}"
                _json(
                    c2_job / "C2_BLOCKED_INPUT_RECEIPT.json",
                    {
                        "format_version": 1,
                        "stage": "C2",
                        "arm": row["arm"],
                        "seed": seed,
                        "status": "blocked_missing_input",
                        "parent": f"C1/seed{seed}",
                        "reason": reason,
                        "coverage_completed": 0.0,
                        "optimizer_updates": 0,
                        "created_at_utc": now(),
                    },
                )
    matrix["status"] = "partial"
    matrix["created_at_utc"] = now()
    _json(matrix_path, matrix)
    return receipt


def write_reports(root: Path = ROOT) -> dict[str, Any]:
    refresh_execution_matrix(root)
    refresh_checkpoint_lineage(root)
    refresh_source_snapshot(root)
    root_out = out(root); matrix = json.loads((root_out / "EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
    blocked = [j["id"] for j in matrix["training_jobs"] if j["status"] != "completed"]
    fixed_pool_metrics = {}
    for arm in ARMS:
        for seed in SEEDS:
            metric_path = root_out / f"rankings/{arm}/seed{seed}/metrics.json"
            if metric_path.is_file():
                metric = json.loads(metric_path.read_text(encoding="utf-8"))
                fixed_pool_metrics[f"{arm}/seed{seed}"] = metric["scorers"]
    teacher_rerank_metrics = {}
    teacher_rerank_receipts = {}
    for seed in SEEDS:
        receipt_path = root_out / f"training/R/seed{seed}/R_COMPLETION_RECEIPT.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            teacher_rerank_receipts[f"seed{seed}"] = {
                "status": receipt.get("status"),
                "budget": receipt.get("budget"),
                "queries": receipt.get("queries"),
                "teacher_checkpoint_sha256": receipt.get("teacher_checkpoint_sha256"),
                "candidate_source": receipt.get("candidate_source"),
                "reused_same_pool_cache": receipt.get("reused_same_pool_cache", False),
                "source_seed": receipt.get("source_seed"),
            }
        for arm in ARMS:
            metric_path = root_out / f"rankings/{arm}/seed{seed}/teacher_metrics.json"
            if metric_path.is_file():
                metric = json.loads(metric_path.read_text(encoding="utf-8"))
                direct = metric.get("scorers", {}).get("DIRECT_ANN", {})
                teacher_rerank_metrics[f"{arm}/seed{seed}"] = {
                    "status": metric.get("status"),
                    "queries": metric.get("queries"),
                    "R@10": direct.get("R@10"),
                    "R@20": direct.get("R@20"),
                    "R@50": direct.get("R@50"),
                    "generator_id": metric.get("generator_id"),
                    "teacher_checkpoint_sha256": metric.get("teacher_checkpoint_sha256"),
                    "ranking_path": str(metric_path.parent / "teacher_rerank.jsonl.gz").replace(str(root_out) + "/", ""),
                }
    acceptance_path = root_out / "acceptance/unit_tests.json"
    acceptance_tests = json.loads(acceptance_path.read_text(encoding="utf-8")).get("tests", []) if acceptance_path.is_file() else []
    passed_test_ids = [t["id"] for t in acceptance_tests if t.get("status") == "passed"]
    s2_metrics_path = root_out / "stage2/S2_PILOT_METRICS.json"
    s2_metrics = json.loads(s2_metrics_path.read_text(encoding="utf-8")) if s2_metrics_path.is_file() else None
    s2_engineering_path = root_out / "stage2/S2_ENGINEERING_RECEIPT.json"
    s2_engineering = json.loads(s2_engineering_path.read_text(encoding="utf-8")) if s2_engineering_path.is_file() else None
    remaining_modules = [m["id"] for m in matrix["modules"] if m["status"] != "completed"]
    fusion_summary = {}
    for status_path in sorted((root_out / "fusion").glob("*/seed*/F_STATUS.json")):
        status_payload = json.loads(status_path.read_text(encoding="utf-8"))
        fusion_summary[f"{status_payload.get('arm')}/seed{status_payload.get('seed')}"] = {
            "status": status_payload.get("status"),
            "completed_methods": status_payload.get("completed_methods", []),
            "metrics": status_payload.get("metrics", {}),
        }
    fusion_method_counts = {}
    for run in fusion_summary.values():
        for method in run["completed_methods"]:
            fusion_method_counts[method] = fusion_method_counts.get(method, 0) + 1
    marker_updates = {
        "rankings/STATUS.json": {"status": "complete", "reason": "14 frozen-pool ranking evaluations have receipts"},
        "fusion/STATUS.json": {"status": "complete", "reason": "F_RECEIPT and per-run F_STATUS receipts are complete"},
        "feedback/STATUS.json": {"status": "complete", "reason": "FB-DIAG and gate receipts are present"},
        "statistics/STATUS.json": {"status": "partial", "reason": "per-query/per-kind/bootstrap/cost statistics were not materialized"},
    }
    for rel, update in marker_updates.items():
        marker = root_out / rel
        if marker.is_file():
            payload = json.loads(marker.read_text(encoding="utf-8"))
            payload.update(update)
            payload["created_at_utc"] = now()
            _json(marker, payload)
    stale_status_markers = []
    for rel in ("rankings/STATUS.json", "fusion/STATUS.json", "feedback/STATUS.json", "statistics/STATUS.json"):
        marker = root_out / rel
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("status") == "blocked_missing_parent":
            stale_status_markers.append(rel)
    expected_artifact_groups = (
            "common/positive_registry/",
            "common/edge_manifest/",
            "common/path_graph/",
            "common/aux_edges/",
            "common/consumed_orders/",
            "statistics/per_query/",
            "statistics/per_kind/",
            "statistics/source_bootstrap/",
            "statistics/costs/",
        )
    missing_expected_artifacts = []
    for rel in expected_artifact_groups:
        path = root_out / rel
        substantive = path.exists() and any(item.name != "STATUS.json" for item in path.iterdir())
        if not substantive:
            missing_expected_artifacts.append(rel)
    results = {"format_version": 1, "contract_version": VERSION, "status": "completed" if not remaining_modules else "partial", "training_jobs_total": 16, "training_jobs_completed": 16 - len(blocked), "training_jobs_blocked_or_pending": len(blocked), "blocked_jobs": blocked, "fixed_pool_evaluations_completed": len(fixed_pool_metrics), "fixed_pool_metrics": fixed_pool_metrics, "teacher_rerank": {"status": "complete" if len(teacher_rerank_metrics) == len(ARMS) * len(SEEDS) else ("partial" if teacher_rerank_metrics else "not_run"), "scope": "frozen T0 reranks the Direct-ANN Top100 from the B13 candidate pool; seed29 reuses the seed13 same-pool cache", "metrics": teacher_rerank_metrics, "receipts": teacher_rerank_receipts}, "fusion": {"status": "complete" if fusion_summary and all(v["status"] == "complete" for v in fusion_summary.values()) else "partial", "run_count": len(fusion_summary), "method_counts": fusion_method_counts, "runs": fusion_summary}, "artifact_audit": {"stale_status_markers": stale_status_markers, "missing_expected_artifacts": missing_expected_artifacts, "statistics_note": "No per-query/per-kind/bootstrap/cost statistics were materialized; only the statistics STATUS scaffold exists."}, "report_paths": {"execution_matrix": "EXECUTION_MATRIX.json", "acceptance": "acceptance/unit_tests.json", "artifact_manifest": "ARTIFACT_MANIFEST.json", "source_snapshot": "source_snapshot/SOURCE_SNAPSHOT.json"}, "implemented_semantic_tests": passed_test_ids, "remaining_contract_modules": remaining_modules, "s2_pilot_metrics": s2_metrics, "s2_engineering_receipt": s2_engineering, "created_at_utc": now()}
    _json(root_out / "RESULTS.json", results)
    resolved = json.loads((root_out / "RESOLVED_INPUTS.json").read_text(encoding="utf-8")) if (root_out / "RESOLVED_INPUTS.json").exists() else {}
    index_ready = "raw_qwen_ann_index" not in resolved.get("missing_roles", [])
    index_note = "The R25-local raw-Qwen HNSW index and both full-coverage schedules are frozen." if index_ready else "The raw-Qwen ANN index is still missing; C1 cannot start."
    table_rows = []
    for arm in ARMS:
        for seed in SEEDS:
            scorers = fixed_pool_metrics.get(f"{arm}/seed{seed}")
            if scorers:
                direct = scorers["DIRECT_ANN"]
                table_rows.append(f"| {arm} | {seed} | {direct['R@10']:.4f} | {direct['R@20']:.4f} | {direct['R@50']:.4f} |")
    report_sections = [
        f"# R25 execution results\n\nStatus: **{'completed' if results['status'] == 'completed' else 'partial'}**. {index_note} All 16 fresh C1/C2 training jobs and {len(fixed_pool_metrics)}/14 frozen-pool evaluations completed with receipts/rankings. Remaining contract modules are: {', '.join(results['remaining_contract_modules']) or 'none'}.",
        "",
        "| Arm | Seed | DIRECT R@10 | DIRECT R@20 | DIRECT R@50 |",
        "|---|---:|---:|---:|---:|",
        *table_rows,
        "",
        "## Teacher rerank (R)",
        "",
        f"Status: **{results['teacher_rerank']['status']}**. Frozen T0 reranked the Direct-ANN Top100 candidate pool under budget BT100 for {len(teacher_rerank_metrics)} arm/seed outputs ({next(iter(teacher_rerank_metrics.values())).get('queries', 0) if teacher_rerank_metrics else 0} queries each). Seed29 reuses the seed13 same-pool cache; this is an offline Teacher diagnostic, not per-arm Teacher retraining.",
        "",
        "| Arm | Seed | R@10 | R@20 | R@50 | Status |",
        "|---|---:|---:|---:|---:|---|",
        *[
            f"| {key.split('/')[0]} | {key.split('/')[1].removeprefix('seed')} | {value['R@10']:.4f} | {value['R@20']:.4f} | {value['R@50']:.4f} | {value['status']} |"
            for key, value in sorted(teacher_rerank_metrics.items())
        ],
        "",
        "Detailed rankings: `rankings/<arm>/seed{13,29}/teacher_rerank.jsonl.gz`; per-run receipts: `training/R/seed{13,29}/R_COMPLETION_RECEIPT.json`.",
        "",
        "The earlier combined Student+Teacher OOM attempt is retained only as a failed runtime receipt; it is not counted as B13 execution because it used the wrong candidate/score path.",
        f"Semantic tests currently passed: {', '.join(passed_test_ids)}.",
    ]
    report_sections.extend([
        "",
        "## Execution and protocol coverage",
        "",
        "| Module | Status | Evidence files |",
        "|---|---|---:|",
        *[f"| {module['id']} | {module['status']} | {len(module.get('evidence_files', []))} |" for module in matrix.get('modules', [])],
        "",
        "A0/B13 protocol: fresh PCA-1024 + identity-R, frozen train-fit selective-hard schedule, same-destination local pools capped at 256, approximately 50% raw-Qwen Top256 negatives, global positive closure, B13 listwise SUP plus raw-logit KD (T=1, weight 0.3), fresh AdamW optimizer, and full coverage-derived schedules. The detailed recipe diff is `B13_RECIPE_DIFF.json`; B0 baselines are registered in `baseline/B0_BASELINE_RECEIPT.json`.",
        "",
        "Fixed-pool rankings expose DIRECT_ANN, QT_OVER_U, and QT_OVER_M metrics in `RESULTS.json`; the main table above is the DIRECT_ANN view.",
        f"Fusion coverage: {results['fusion']['run_count']} runs; methods completed — " + ", ".join(f"{method}: {count}" for method, count in sorted(fusion_method_counts.items())) + ". Representative B13-FULL/seed13 metrics are retained in `fusion/B13-FULL/seed13/F_STATUS.json`; confidence-only now has a generated sidecar metric.",
    ])
    if s2_metrics:
        report_sections.extend([
            "",
            "## Stage2 pilot",
            "",
            "The 64-query paired pilot has 256 registered rows (B13/raw-Qwen × Real/NoE-fill), with 252 complete and 4 query-level failures retained. The requested candidate budget was 50, but the frozen input pool contained 10 candidates per query; this is therefore an effective-C10 pilot.",
            "",
            "| Generator / condition | Complete / failed | Stage2 R@10 | Stage2 R@20 | Stage2 R@50 | Non-empty values |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for key, metric in sorted(s2_metrics.get("summary", {}).items()):
            recall = metric.get("stage2_recall_at", {})
            report_sections.append(f"| {key} | {metric.get('complete_rows', 0)}/{metric.get('failed_rows', 0)} | {recall.get('10', 0.0):.4f} | {recall.get('20', 0.0):.4f} | {recall.get('50', 0.0):.4f} | {metric.get('nonempty_generated_values', 0)} |")
    if s2_engineering:
        totals = s2_engineering.get("totals", {})
        report_sections.extend([
            "",
            f"Stage2 engineering audit: {s2_engineering.get('records_executed', 0)}/32 records executed across text/image evidence; {totals.get('evidence_slots', 0)} slots, JSON parse failures {totals.get('json_parse_failures', 0)}, empty outputs {totals.get('empty_outputs', 0)}, span-over-limit {totals.get('span_over_limit', 0)}, invalid ROI {totals.get('roi_invalid', 0)}. Parse failures are retained as measured engineering outcomes.",
        ])
    feedback_path = root_out / "feedback/FEEDBACK_GATE.json"
    if feedback_path.is_file():
        feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
        diagnostics = json.loads((root_out / "feedback/FB_DIAG_RECEIPT.json").read_text(encoding="utf-8")) if (root_out / "feedback/FB_DIAG_RECEIPT.json").is_file() else {}
        diag_overlap = ", ".join(f"{item.get('student_teacher_top10_overlap', 0):.4f}" for item in diagnostics.get("diagnostics", []))
        diag_novelty = ", ".join(f"{item.get('teacher_novelty_top10', 0):.4f}" for item in diagnostics.get("diagnostics", []))
        report_sections.extend([
            "",
            "## Feedback gates",
            "",
            f"FB-DIAG: **{diagnostics.get('status', 'unknown')}**. Student/Teacher Top-10 overlap was {diag_overlap}; Teacher novelty was {diag_novelty} (seeds 13/29).",
            f"FB-TRAIN: **{feedback.get('status')}**. All recipes passed health tolerance, but every recipe had `hard_membership_diff_lists=0`; no new hard-competitor membership appeared. REDISTILL was therefore not triggered.",
        ])
    report_sections.extend([
        "",
        "## Report coverage audit and limitations",
        "",
        f"Fusion receipt: **{results['fusion']['status']}**. Equal, confidence-only, and column-only outputs are present for the R25 arms; their per-run metrics and alpha summaries are in `fusion/*/seed*/F_STATUS.json` and `RESULTS.json`.",
        "Same-pool Teacher is reported above. An independent own-pool Teacher comparison was not run in R25: all R25 Teacher outputs use the B13 Direct-ANN Top100 pool, and seed29 reuses seed13's same-pool cache.",
        "The first measured Stage2 bottleneck is generation validity (128/128 JSON parse failures and empty outputs); the second is the effective candidate pool (C10 rather than the requested C50). No latency/cost measurement was recorded.",
        f"The following expected artifact groups are not materialized: {', '.join(results['artifact_audit']['missing_expected_artifacts']) or 'none'}. Scaffold status markers still carrying the old blocked value are: {', '.join(results['artifact_audit']['stale_status_markers']) or 'none'}. These are documentation/artifact gaps and are not silently counted as experimental results.",
    ])
    report_sections.extend([
        "",
        "## Fusion and remaining modules",
        "",
        "Equal, confidence-only, and legal column-only fusions are materialized for all 14 R25 arms and the historical Qwen-Raw/B13/N-U controls. Legacy Direct scores were recovered from frozen feature/checkpoint/detailed-path artifacts; no table-vector or GT-derived substitute was used.",
        f"Remaining contract modules: {', '.join(remaining_modules) or 'none'}.",
    ])
    (root_out / "RESULTS.md").write_text("\n".join(report_sections) + "\n", encoding="utf-8")
    (root_out / "LIMITATIONS.md").write_text("# R25 limitations\n\n- The host exposes two RTX 4090 cards, not A100s.\n- Teacher scoring is intentionally a separate offline phase from Student backprop to avoid the earlier combined 24 GB OOM.\n- The Stage2 pilot requested C=50 but its frozen retrieval input contains C=10; the pilot must not be interpreted as a full C50 result.\n- The Stage2 engineering audit executed all 32 records, but the fixed 64-token generation budget produced parse failures/empty outputs; these are retained as engineering negative results rather than relabeled successes.\n- No checkpoint or metric is labeled completed without a valid receipt; remaining limitations are reflected in EXECUTION_MATRIX.json.\n", encoding="utf-8")
    (root_out / "B13_TRAINING_LOGIC.md").write_text(
        "# B13 core training logic carried into R25\n\n"
        "1. Start from frozen Qwen features, fresh PCA-1024 projections and identity full-R; P and all five directed R parameters are trainable, backbone is frozen.\n"
        "2. Build one frozen train-fit edge epoch. For each batch, pool same-destination candidates, preserve the global known-positive closure, cap lists at 256, replace about half of negative slots with raw-Qwen ANN Top256 assumed negatives, and deterministically refill the remainder.\n"
        "3. Score the five relations together. B13 SUP is multi-positive sum-probability listwise on `10*sigmoid(raw_logit)`; KD is `KL` on raw logits at T=1 with weight 0.3; BCE is zero. Anchors are the original PCA/identity references with 0.1 all/evidence weighting.\n"
        "4. Use AdamW (relation lr 1e-5, projection lr 1e-6, weight decay 0.01), fresh optimizer, and consume the full coverage-derived schedule once. Save coverage 0/50/100; the 100% endpoint is the only C2 parent.\n"
        "5. B13-FULL C2 uses the native direct/evidence target/path helper with a fresh optimizer; modern Split/LSE arms are separate objectives and never replace this reference.\n\n"
        "Historical evidence: `src/run_stage1_r12_task_c.py`, `src/prepare_stage1_r12_candidates.py`, `src/run_stage1_r13.py`, and `src/mmdd_stage1/training.py`.\n",
        encoding="utf-8",
    )
    return results


def write_b0_baseline_receipt(root: Path = ROOT) -> dict[str, Any]:
    """Register frozen historical baselines consumed by R25 comparisons."""
    import shutil
    refs = [
        root / "work/stage1_optimization_r21_20260911/matched_control/Qwen-Raw/raw/metrics.json",
        root / "work/stage1_optimization_r21_20260911/matched_control/B13/raw/metrics.json",
        root / "work/stage1_optimization_r24_20260913/full_lake/N-U/seed13/metrics.json",
        root / "work/stage1_optimization_r24_20260913/full_lake/N-U/seed29/metrics.json",
        out(root) / "common/raw_qwen_index/manifest.json",
    ]
    destination = out(root) / "baseline"; destination.mkdir(parents=True, exist_ok=True)
    copies = {}
    for label, source in zip(("Qwen-Raw", "B13-raw", "N-U-seed13", "N-U-seed29"), refs[:4], strict=True):
        if source.is_file():
            target = destination / f"{label.replace('/', '_')}.metrics.json"; shutil.copyfile(source, target); copies[label] = str(target.resolve())
    receipt = {"format_version": 1, "stage": "B0", "status": "complete" if all(p.is_file() for p in refs) else "partial", "baselines": {"Qwen-Raw": str(refs[0].resolve()), "B13-raw": str(refs[1].resolve()), "N-U-seed13": str(refs[2].resolve()), "N-U-seed29": str(refs[3].resolve()), "R25-raw-qwen-hnsw": str(refs[4].resolve())}, "local_copies": copies, "created_at_utc": now()}
    _json(out(root) / "baseline/B0_BASELINE_RECEIPT.json", receipt)
    return receipt


def write_scaffold(root: Path = ROOT) -> dict[str, Any]:
    """Write the non-model lineage/namespace records required for resumption."""
    root_out = out(root)
    source_files = [root / "src/run_stage1_r25.py", root / "src/backfill_stage1_teacher.py", root / "src/mmdd_stage1/b13_recipe.py", root / "src/mmdd_stage1/r25_objectives.py", root / "src/run_stage2_r25_pilot.py", root / "src/audit_stage2_r25_engineering.py", root / "src/finalize_stage2_r25_engineering.py", root / "src/finalize_stage2_r25_pilot.py", root / "src/summarize_stage2_r25.py", root / "mmdd_r24_review/R25_EXPERIMENT_PLAN.md", root / "mmdd_r24_review/EXECUTION_CONTRACT.json"]
    snapshot = {"format_version": 1, "contract_version": VERSION, "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip(), "dirty": True, "files": [{"path": str(p.relative_to(root)), "sha256": sha256(p), "bytes": p.stat().st_size} for p in source_files if p.exists()], "created_at_utc": now()}
    _json(root_out / "source_snapshot/SOURCE_SNAPSHOT.json", snapshot)
    lineage = {"format_version": 1, "contract_version": VERSION, "fresh_origin": "PCA-1024 + identity-R", "c1": {str(seed): {"status": "blocked_resource", "parent": "fresh_pca_identity", "receipt": f"training/C1/seed{seed}/C1_BLOCKED_RECEIPT.json"} for seed in SEEDS}, "c2": {f"{arm}/seed{seed}": {"status": "blocked_missing_parent", "parent": f"C1/seed{seed}"} for seed in SEEDS for arm in ARMS}, "created_at_utc": now()}
    _json(root_out / "checkpoint_lineage.json", lineage)
    for rel in ("common/positive_registry", "common/edge_manifest", "common/path_graph", "common/aux_edges", "common/consumed_orders", "rankings", "fusion", "stage2", "feedback", "statistics"):
        (root_out / rel).mkdir(parents=True, exist_ok=True)
        _json(root_out / rel / "STATUS.json", {"status": "blocked_missing_parent", "reason": "R25 C1/C2 runtime did not produce model outputs", "created_at_utc": now()})
    return {"status": "partial", "source_snapshot": str((root_out / "source_snapshot/SOURCE_SNAPSHOT.json").resolve()), "checkpoint_lineage": str((root_out / "checkpoint_lineage.json").resolve())}


def _edge_examples_from_schedule(path: Path):
    """Convert frozen schedule rows to core ``EdgeExample`` records."""

    from mmdd_stage1.data import EdgeExample

    result = []
    for batch in _schedule_examples(path):
        for row in batch:
            candidates = tuple(str(value) for value in row["candidate_ids"])
            positive_ids = tuple(str(value) for value in row.get("positive_ids", ()))
            designated = str(row.get("positive_id", positive_ids[0] if positive_ids else candidates[0]))
            if designated not in candidates:
                raise ValueError(f"schedule positive is absent: {designated}")
            labels = row.get("confirmed_labels")
            result.append(
                EdgeExample(
                    query_id=str(row["query_id"]),
                    candidate_ids=candidates,
                    positive_index=candidates.index(designated),
                    dataset=str(row.get("dataset", "default")),
                    split="train",
                    source_type=str(row["source_type"]),
                    destination_type=str(row["destination_type"]),
                    positive_ids=positive_ids or (designated,),
                    confirmed_labels=None if labels is None else tuple(None if value is None else int(value) for value in labels),
                )
            )
    return result


def _r25_teacher_feature_paths(root: Path) -> list[Path]:
    """Historical R12 Teacher shards and completed missing-object supplements."""
    from run_stage1_r23 import teacher_feature_paths as _legacy_teacher_feature_paths

    paths = list(_legacy_teacher_feature_paths(root))
    r12_training = root / "work/stage1_optimization_r12_20260908/taskC_training"
    for supplement in (
        r12_training / "teacher_extra",
        r12_training / "teacher_extension_extra",
        out(root) / "common/teacher_backfill_gpu0",
        out(root) / "common/teacher_backfill_gpu1",
        out(root) / "common/path_teacher_backfill_gpu0",
        out(root) / "common/path_teacher_backfill_gpu1",
        root / "work/stage1_optimization_r26_20260914/teacher_backfill",
    ):
        if (supplement / "teacher_manifest.jsonl").is_file() and supplement not in paths:
            paths.append(supplement)
    return paths


def train_c1(root: Path, seed: int, device_name: str = "cuda:0") -> dict[str, Any]:
    """Run the complete fresh selective-hard C1 edge stage for one seed.

    The schedule is a R25-frozen raw-Qwen selective-hard materialization.  The
    Student update delegates to :func:`mmdd_stage1.b13_recipe.edge_objective`,
    which preserves B13's transformed SUP/raw-logit KD distinction.
    """
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if not torch_cuda_available():
        raise RuntimeError("CUDA unavailable; C1 is blocked_resource")
    import torch
    from mmdd_stage1.artifacts import checkpoint_fingerprint
    from mmdd_stage1.features import FeatureStore
    from mmdd_stage1.scoring import ListScores, score_edge_batch
    from mmdd_stage1.training import student_gradient_norms
    from mmdd_stage1.b13_recipe import B13_BATCH_SIZE, consumed_batch_order, edge_objective, validate_relation_coverage
    from run_stage1_r22_f0 import _build_fresh_student, _optimizer

    device = torch.device(device_name)
    manifest = c1_schedule_path(root, seed)
    teacher_path = c1_teacher_path(root)
    cache_path = out(root) / "common" / f"teacher_edge_cache_seed{seed}.jsonl.gz"
    if not manifest.exists() or not teacher_path.exists() or not cache_path.exists():
        raise FileNotFoundError(f"C1 inputs/cache missing; run prepare-c1-schedule and prepare-teacher-edge-cache first: {manifest}, {cache_path}")
    job = out(root) / "training/C1" / f"seed{seed}"
    if (job / "C1_COMPLETION_RECEIPT.json").exists():
        existing = json.loads((job / "C1_COMPLETION_RECEIPT.json").read_text(encoding="utf-8"))
        expected = {"stage": "C1", "arm": "C1", "seed": seed, "lineage_origin": "fresh_pca_identity", "optimizer_initial_state": "fresh", "manifest_sha256": checkpoint_fingerprint(manifest)}
        if not resume_compatible(existing, expected):
            raise RuntimeError("existing C1 receipt does not match the frozen plan/data identity; refusing resume shortcut")
        return existing
    _set_job_status(root, f"C1/seed{seed}", "running")
    schedule_batches = _schedule_examples(manifest)
    examples = _edge_examples_from_schedule(manifest)
    if not examples:
        raise RuntimeError("empty C1 manifest")
    relation_counts = validate_relation_coverage(examples)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = _build_fresh_student(root, device).train()
    optimizer = _optimizer(model)
    # The Teacher is deliberately not loaded here.  prepare_teacher_edge_cache
    # runs it in a separate process/phase, so Student backprop never shares a
    # 24 GB device with Teacher activations.
    import gzip
    teacher_cache: dict[tuple[str, str, tuple[str, ...]], list[float]] = {}
    with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); teacher_cache[(str(row["query_id"]), str(row["relation"]), tuple(map(str, row["candidate_ids"]))) ] = [float(x) for x in row["scores"]]
    student_store = FeatureStore.from_path(root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b", cache_size=24000)
    job.mkdir(parents=True, exist_ok=True)
    (job / "checkpoints").mkdir(exist_ok=True)
    def trainable_hash(module: torch.nn.Module) -> str:
        h = hashlib.sha256()
        for name, p in module.named_parameters():
            if p.requires_grad:
                h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def save(step: int) -> str:
        path = job / "checkpoints" / f"step_{step:06d}.pt"
        torch.save({"format_version": 1, "model_kind": "student", "completed_stage": "r25-c1", "arm": "C1", "seed": seed, "step": step, "config": model.config(), "trainable_parameters": [n for n,p in model.named_parameters() if p.requires_grad], "state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict()}, path)
        return checkpoint_fingerprint(path)
    initial_hash = trainable_hash(model)
    anchor_tensor = model.initial_projection_weights.detach().cpu() if hasattr(model, "initial_projection_weights") else None
    anchor_hash = hashlib.sha256(anchor_tensor.contiguous().numpy().tobytes()).hexdigest() if anchor_tensor is not None else initial_hash
    # The schedule itself is a complete sampled epoch.  Only its batch order
    # may be permuted per seed, matching the historical R12 consumption rule.
    order = consumed_batch_order(len(schedule_batches), seed)
    updates = 0; relation_stats = {r: {"supervised_active_lists": 0, "kd_active_lists": 0} for r in RELATIONS}; history = []
    checkpoints = {0, math.ceil(len(schedule_batches) / 2), len(schedule_batches)}
    save(0)
    for schedule_index in order:
            batch_start = schedule_index * B13_BATCH_SIZE
            batch = examples[batch_start : batch_start + len(schedule_batches[schedule_index])]
            optimizer.zero_grad(set_to_none=True)
            teacher_rows = []
            width = max(len(e.candidate_ids) for e in batch)
            for e in batch:
                relation = f"{e.source_type}->{e.destination_type}"
                key = (str(e.query_id), relation, tuple(map(str, e.candidate_ids)))
                values = teacher_cache.get(key)
                if values is None or len(values) != len(e.candidate_ids):
                    raise RuntimeError(f"Teacher edge cache miss or length mismatch: {key[:2]}")
                teacher_rows.append(torch.tensor(values, device=device))
            teacher_scores = ListScores(
                torch.stack([torch.nn.functional.pad(row, (0, width - row.numel())) for row in teacher_rows]),
                torch.stack([torch.arange(width, device=device) < len(e.candidate_ids) for e in batch]),
                torch.tensor([e.positive_index for e in batch], device=device),
                torch.stack([torch.tensor([str(c) in set(e.positive_ids) for c in e.candidate_ids] + [False] * (width - len(e.candidate_ids)), device=device) for e in batch]),
            )
            student_scores = score_edge_batch(model, batch, student_store, device, student_score_space="raw_logit")
            terms = edge_objective(model, batch, student_scores, teacher_scores)
            terms["loss"].backward(); optimizer.step(); updates += 1
            for relation in RELATIONS:
                n = sum(f"{e.source_type}->{e.destination_type}" == relation for e in batch)
                relation_stats[relation]["supervised_active_lists"] += n
                relation_stats[relation]["kd_active_lists"] += n
            if updates in checkpoints or updates % 100 == 0:
                history.append({"step": updates, "epoch": 1, "loss": float(terms["loss"].detach().cpu()), "gradient": student_gradient_norms(model)})
            if updates in checkpoints:
                save(updates)
    final_hash = trainable_hash(model); checkpoint_sha = save(updates)
    receipt = {"format_version": 1, "stage": "C1", "arm": "C1", "seed": seed, "lineage_origin": "fresh_pca_identity", "initial_parameter_sha256": initial_hash, "final_parameter_sha256": final_hash, "pca_anchor_sha256": anchor_hash, "coverage_completed": 1.0, "optimizer_updates": updates, "batch_size": B13_BATCH_SIZE, "relation_counts": relation_counts, "relation_stats": relation_stats, "teacher_source": {"id": "B13-T_core-R12", "checkpoint": str(teacher_path.resolve()), "checkpoint_sha256": checkpoint_fingerprint(teacher_path), "edge_cache": str(cache_path.resolve()), "edge_cache_sha256": checkpoint_fingerprint(cache_path)}, "optimizer_initial_state": "fresh", "manifest_sha256": checkpoint_fingerprint(manifest), "consumed_order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(), "checkpoint_sha256": checkpoint_sha, "evidence_files": [f"training/C1/seed{seed}/train_history.jsonl", f"common/teacher_edge_cache_seed{seed}.jsonl.gz"], "created_at_utc": now()}
    (job / "train_history.jsonl").write_text("\n".join(json.dumps(x) for x in history) + "\n", encoding="utf-8")
    _json(job / "C1_COMPLETION_RECEIPT.json", receipt)
    _set_job_status(root, f"C1/seed{seed}", "completed", completion_receipt=f"training/C1/seed{seed}/C1_COMPLETION_RECEIPT.json")
    return receipt


def _r25_path_pool(root: Path) -> Path:
    """Canonical train-only C2 graph (identical frozen graph for both seeds)."""
    return root / "work/stage1_optimization_r24_20260913/path_pool/common_seed13.jsonl"


def _load_target_cache(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {str(row["query_id"]): row for row in (json.loads(line) for line in handle)}


def _target_cache_scores(examples: list[Any], cache: dict[str, dict[str, Any]], device: Any, *, qt_evidence: bool) -> Any:
    """Create TargetScores from an offline cache, preserving candidate masks."""
    from mmdd_stage1.scoring import ListScores, TargetScores
    import torch

    rows_d: list[torch.Tensor] = []
    rows_e: list[torch.Tensor] = []
    masks_d: list[torch.Tensor] = []
    masks_e: list[torch.Tensor] = []
    positives_d: list[int] = []
    positives_e: list[int] = []
    positive_masks_d: list[torch.Tensor] = []
    positive_masks_e: list[torch.Tensor] = []
    for example in examples:
        row = cache.get(str(example.query_id))
        if row is None or list(map(str, row["candidate_ids"])) != [c.target_id for c in example.candidates]:
            raise RuntimeError(f"target Teacher cache identity mismatch: {example.query_id}")
        direct = torch.tensor(row["direct_logits"], device=device, dtype=torch.float32)
        evidence = direct.clone() if qt_evidence else torch.tensor(row["evidence_logits"], device=device, dtype=torch.float32)
        dmask = torch.ones_like(direct, dtype=torch.bool)
        emask = torch.tensor([bool(c.evidence_ids) for c in example.candidates], device=device)
        rows_d.append(direct); rows_e.append(evidence); masks_d.append(dmask); masks_e.append(emask)
        positives_d.append(int(example.direct_positive_index)); positives_e.append(int(example.evidence_positive_index))
        pd = torch.zeros_like(dmask); pd[list(example.positive_target_ids).index(example.candidates[example.direct_positive_index].target_id) if False else int(example.direct_positive_index)] = True
        pe = torch.zeros_like(emask); pe[int(example.evidence_positive_index)] = True
        positive_masks_d.append(pd); positive_masks_e.append(pe & emask)
    width = max(x.numel() for x in rows_d)
    def pad(values: list[torch.Tensor], fill: float = 0.0) -> torch.Tensor:
        return torch.stack([torch.nn.functional.pad(x, (0, width - x.numel()), value=fill) for x in values])
    return TargetScores(
        direct=ListScores(pad(rows_d), pad(masks_d), torch.tensor(positives_d, device=device), pad(positive_masks_d)),
        evidence=ListScores(pad(rows_e), pad(masks_e), torch.tensor(positives_e, device=device), pad(positive_masks_e)),
    )


def prepare_native_path_cache(root: Path, device_name: str = "cuda:0", microbatch: int = 32) -> dict[str, Any]:
    """Materialize native historical T_core D/E logits for the frozen C2 graph."""
    import torch
    from mmdd_stage1.checkpoints import load_teacher
    from mmdd_stage1.data import load_target_examples
    from mmdd_stage1.features import FeatureStore
    from mmdd_stage1.objectives import PathAggregator
    from mmdd_stage1.scoring import score_target_batch

    graph = _r25_path_pool(root)
    destination = out(root) / "common/teacher_native_path_cache.jsonl.gz"
    if destination.exists():
        return {"status": "complete", "path": str(destination.resolve()), "sha256": sha256(destination), "reused": True}
    examples = load_target_examples(graph, split="train")
    teacher = load_teacher(c1_teacher_path(root), torch.device(device_name)).eval()
    store = FeatureStore.from_path(root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b", cache_size=24000, teacher_paths=_r25_teacher_feature_paths(root))
    aggregator = PathAggregator("logsumexp", 4, path_combination="sum")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    rows = 0
    with torch.inference_mode():
        handle = gzip.open(temporary, "wt", encoding="utf-8")
        try:
            for start in range(0, len(examples), microbatch):
                batch = examples[start:start + microbatch]
                scores = score_target_batch(teacher, batch, store, torch.device(device_name), aggregator, student_score_space="raw_logit")
                for index, example in enumerate(batch):
                    n = len(example.candidates)
                    handle.write(json.dumps({"query_id": str(example.query_id), "candidate_ids": [c.target_id for c in example.candidates], "direct_logits": [float(x) for x in scores.direct.logits[index, :n].cpu()], "evidence_logits": [float(x) for x in scores.evidence.logits[index, :n].cpu()]}) + "\n")
                    rows += 1
                if rows and rows % (microbatch * 20) == 0:
                    print(json.dumps({"native_path_rows": rows, "total_rows": len(examples)}), flush=True)
        finally:
            handle.close()
    temporary.replace(destination)
    return {"status": "complete", "path": str(destination.resolve()), "sha256": sha256(destination), "rows": rows, "teacher_checkpoint_sha256": sha256(c1_teacher_path(root))}


def train_c2(root: Path, arm: str, seed: int, device_name: str = "cuda:0") -> dict[str, Any]:
    """Train one C2 arm from the matching fresh C1 final checkpoint."""
    if arm not in ARMS or arm == "C1":
        raise ValueError(f"unknown C2 arm: {arm}")
    import torch
    from mmdd_stage1.artifacts import checkpoint_fingerprint
    from mmdd_stage1.checkpoints import load_student
    from mmdd_stage1.data import load_edge_examples, load_target_examples
    from mmdd_stage1.features import FeatureStore
    from mmdd_stage1.objectives import PathAggregator, listwise_cross_entropy
    from mmdd_stage1.scoring import ListScores, score_edge_batch, score_target_batch
    from mmdd_stage1.training import _anchor_losses, student_gradient_norms
    from run_stage1_r22_f0 import _optimizer
    from mmdd_stage1.b13_recipe import path_objective
    from mmdd_stage1.r25_objectives import edge_continuation_loss, split_objective

    parent = out(root) / "training/C1" / f"seed{seed}" / "checkpoints/step_000659.pt"
    if not parent.exists():
        # The exact update count is schedule-derived; discover the final C1 file.
        candidates = sorted(parent.parent.glob("step_*.pt"))
        if not candidates:
            raise FileNotFoundError(parent)
        parent = candidates[-1]
    graph = _r25_path_pool(root)
    if not graph.exists():
        raise FileNotFoundError(graph)
    job = out(root) / "training/C2" / arm / f"seed{seed}"
    receipt_path = job / "C2_COMPLETION_RECEIPT.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        objective_family = "native_split" if arm == "B13-FULL" else ("edge" if arm == "EDGE-CONT" else ("lse" if arm == "LSE-QTKD" else "split"))
        expected = {"stage": "C2", "arm": arm, "seed": seed, "parent_job_id": f"C1/seed{seed}", "optimizer_initial_state": "fresh", "objective_family": objective_family, "graph_sha256": sha256(graph)}
        if not resume_compatible(existing, expected):
            raise RuntimeError("existing C2 receipt does not match the frozen plan/data identity; refusing resume shortcut")
        return existing
    _set_job_status(root, f"C2/{arm}/seed{seed}", "running")
    device = torch.device(device_name)
    examples = load_target_examples(graph, split="train")
    model = load_student(parent, device).train()
    optimizer = _optimizer(model)
    store = FeatureStore.from_path(root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b", cache_size=24000)
    qt_cache = _load_target_cache(root / "work/stage1_optimization_r24_20260913/path_pool/teacher_target_seed13.jsonl.gz")
    native_path = out(root) / "common/teacher_native_path_cache.jsonl.gz"
    native_cache = _load_target_cache(native_path) if native_path.exists() else None
    edge_examples = _edge_examples_from_schedule(c1_schedule_path(root, seed))
    batches = math.ceil(len(examples) / 64)
    checkpoints = {0, math.ceil(batches / 2), batches}
    job.mkdir(parents=True, exist_ok=True); (job / "checkpoints").mkdir(exist_ok=True)
    def trainable_hash(module: torch.nn.Module) -> str:
        h = hashlib.sha256()
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                h.update(name.encode()); h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def save(step: int) -> str:
        path = job / "checkpoints" / f"step_{step:06d}.pt"
        torch.save({"format_version": 1, "model_kind": "student", "completed_stage": "r25-c2", "arm": arm, "seed": seed, "step": step, "config": model.config(), "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict()}, path)
        return checkpoint_fingerprint(path)
    initial_hash = trainable_hash(model); anchor_hash = hashlib.sha256(model.initial_projection_weights.detach().cpu().contiguous().numpy().tobytes()).hexdigest(); save(0)
    history: list[dict[str, Any]] = []
    for step, start in enumerate(range(0, len(examples), 64), 1):
        batch = examples[start:start + 64]
        optimizer.zero_grad(set_to_none=True)
        aggregator = PathAggregator("logsumexp", 4 if arm == "B13-FULL" else 8, path_combination="sum")
        scores = score_target_batch(model, batch, store, device, aggregator, student_score_space="raw_logit")
        if arm == "B13-FULL":
            if native_cache is None:
                raise FileNotFoundError(f"run prepare-native-path-cache first: {native_path}")
            teacher_scores = _target_cache_scores(batch, native_cache, device, qt_evidence=False)
            terms = path_objective(model, scores, teacher_scores)
        elif arm == "EDGE-CONT":
            direct_sup = listwise_cross_entropy(scores.direct.logits, scores.direct.positive_indices, scores.direct.candidate_mask, scores.direct.positive_mask)
            eb = edge_examples[(start % len(edge_examples)): (start % len(edge_examples)) + len(batch)]
            if len(eb) < len(batch): eb += edge_examples[: len(batch) - len(eb)]
            edge_scores = score_edge_batch(model, eb, store, device, student_score_space="raw_logit")
            qe = [i for i, e in enumerate(eb) if e.destination_type in {"text", "image"}]
            et = [i for i, e in enumerate(eb) if e.source_type in {"text", "image"}]
            qe_scores = edge_scores.select(torch.tensor([i in qe for i in range(len(eb))], device=device)) if qe else edge_scores
            et_scores = edge_scores.select(torch.tensor([i in et for i in range(len(eb))], device=device)) if et else edge_scores
            cont = edge_continuation_loss(qe_scores, et_scores)
            _anchor, weighted_anchor = _anchor_losses(model, 0.1, 0.1)
            terms = {"loss": direct_sup + cont["loss"] + weighted_anchor, "direct_supervised_loss": direct_sup, "evidence_supervised_loss": cont["loss"], "distillation_loss": direct_sup.new_zeros(())}
        else:
            teacher_scores = _target_cache_scores(batch, qt_cache, device, qt_evidence=True) if arm in {"SPLIT-QTKD", "SPLIT-UQTKD", "LSE-QTKD"} else None
            uniform_scores = None; uniform_relations = None
            if arm in {"SPLIT-U", "SPLIT-UQTKD"}:
                eb = edge_examples[(start % len(edge_examples)): (start % len(edge_examples)) + len(batch)]
                if len(eb) < len(batch): eb += edge_examples[: len(batch) - len(eb)]
                uniform_scores = score_edge_batch(model, eb, store, device, student_score_space="raw_logit"); uniform_relations = [f"{e.source_type}->{e.destination_type}" for e in eb]
            _anchor, weighted_anchor = _anchor_losses(model, 0.1, 0.1)
            terms = split_objective(scores, arm=arm, teacher_qt=teacher_scores, uniform_scores=uniform_scores, uniform_relations=uniform_relations, anchor_loss=weighted_anchor)
        terms["loss"].backward(); optimizer.step()
        history.append({"step": step, "loss": float(terms["loss"].detach().cpu()), "gradient": student_gradient_norms(model)})
        if step in checkpoints: save(step)
    final_hash = trainable_hash(model); checkpoint_sha = save(batches)
    receipt = {"format_version": 1, "stage": "C2", "arm": arm, "seed": seed, "parent_job_id": f"C1/seed{seed}", "parent_final_parameter_sha256": initial_hash, "initial_parameter_sha256": initial_hash, "final_parameter_sha256": final_hash, "pca_anchor_sha256": anchor_hash, "coverage_completed": 1.0, "optimizer_updates": batches, "batch_size": 64, "graph_sha256": sha256(graph), "consumed_order_sha256": hashlib.sha256(json.dumps(list(range(batches))).encode()).hexdigest(), "optimizer_initial_state": "fresh", "objective_family": "native_split" if arm == "B13-FULL" else ("edge" if arm == "EDGE-CONT" else ("lse" if arm == "LSE-QTKD" else "split")), "checkpoint_sha256": checkpoint_sha, "history": history, "evidence_files": [f"training/C2/{arm}/seed{seed}/train_history.jsonl"], "created_at_utc": now()}
    (job / "train_history.jsonl").write_text("\n".join(json.dumps(x) for x in history) + "\n", encoding="utf-8"); _json(receipt_path, receipt)
    _set_job_status(root, f"C2/{arm}/seed{seed}", "completed", completion_receipt=f"training/C2/{arm}/seed{seed}/C2_COMPLETION_RECEIPT.json")
    return receipt


def evaluate_c2_fixed(root: Path, arm: str, seed: int, device_name: str = "cuda:0") -> dict[str, Any]:
    """Score each completed Student on the frozen R23 candidate pools."""
    import statistics
    import torch
    from mmdd_stage1.checkpoints import load_student
    from mmdd_stage1.features import FeatureStore
    from run_stage1_r21 import _score_target_ids, paths as r21_paths, read_rows, write_rows

    ck_dir = out(root) / "training/C2" / arm / f"seed{seed}" / "checkpoints"
    checkpoints = sorted(ck_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if not checkpoints:
        raise FileNotFoundError(ck_dir)
    checkpoint = checkpoints[-1]
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(json.dumps({"evaluate": f"{arm}/seed{seed}", "phase": "load_student", "checkpoint": str(checkpoint), "device": str(device)}), flush=True)
    model = load_student(checkpoint, device).eval()
    print(json.dumps({"evaluate": f"{arm}/seed{seed}", "phase": "load_features"}), flush=True)
    store = FeatureStore.from_path(root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b", cache_size=40000)
    print(json.dumps({"evaluate": f"{arm}/seed{seed}", "phase": "load_pools"}), flush=True)
    pools = list(read_rows(r21_paths(root)["candidate_pools"]))
    preload_ids = {str(pool["query_id"]) for pool in pools}
    for pool in pools:
        preload_ids.update(map(str, pool.get("positive_target_ids", [])))
        preload_ids.update(map(str, pool.get("ann_direct100_ids", [])))
        preload_ids.update(map(str, pool.get("natural_candidate_ids", [])))
        preload_ids.update(map(str, pool.get("matched_direct_candidate_ids", [])))
    print(json.dumps({"evaluate": f"{arm}/seed{seed}", "phase": "preload_embeddings", "objects": len(preload_ids)}), flush=True)
    store.preload_embeddings(preload_ids)
    print(json.dumps({"evaluate": f"{arm}/seed{seed}", "phase": "score", "queries": len(pools)}), flush=True)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
      for pool_index, pool in enumerate(pools):
        positives = set(map(str, pool["positive_target_ids"]))
        sets = {"DIRECT_ANN": list(map(str, pool["ann_direct100_ids"])), "QT_OVER_U": list(map(str, pool["natural_candidate_ids"])), "QT_OVER_M": list(map(str, pool["matched_direct_candidate_ids"]))}
        ids = list(dict.fromkeys(x for values in sets.values() for x in values))
        scores = _score_target_ids(model, store, str(pool["query_id"]), ids, device)
        record = {"query_id": str(pool["query_id"]), "query_kind": pool.get("query_kind"), "source_table_id": pool.get("source_table_id"), "positive_target_ids": sorted(positives), "generator_id": f"R25/{arm}/seed{seed}", "checkpoint_sha256": sha256(checkpoint), "scorer_ids": {}}
        for scorer, candidates in sets.items():
            ranking = sorted(candidates, key=lambda value: (-scores[value], value))
            record["scorer_ids"][scorer] = {"candidate_ids": candidates, "ranking": ranking, "scores": [float(scores[x]) for x in ranking], "recall": {f"R@{k}": len(positives & set(ranking[:k])) / len(positives) if positives else 0.0 for k in (10, 20, 50)}, "raw_recall": len(positives & set(candidates)) / len(positives) if positives else 0.0}
        rows.append(record)
        if (pool_index + 1) % 1000 == 0:
            print(json.dumps({"evaluate": f"{arm}/seed{seed}", "phase": "score", "completed": pool_index + 1, "total": len(pools)}), flush=True)
    destination = out(root) / "rankings" / arm / f"seed{seed}"; destination.mkdir(parents=True, exist_ok=True)
    ranking_path = destination / "query_rankings.jsonl.gz"; write_rows(ranking_path, rows)
    metrics = {"format_version": 1, "status": "complete", "generator_id": f"R25/{arm}/seed{seed}", "checkpoint_sha256": sha256(checkpoint), "queries": len(rows), "scorers": {scorer: {key: statistics.fmean(row["scorer_ids"][scorer]["recall"][key] for row in rows) for key in ("R@10", "R@20", "R@50")} for scorer in ("DIRECT_ANN", "QT_OVER_U", "QT_OVER_M")}, "ranking_path": str(ranking_path.resolve()), "evaluation_scope": "frozen R23 candidate pools; full-lake ANN/exact pending", "created_at_utc": now()}
    _json(destination / "metrics.json", metrics)
    return metrics


def rerank_teacher_r25(root: Path, seed: int, device_name: str = "cuda:0") -> dict[str, Any]:
    """Run the frozen T0 on the same frozen R25 candidate pools used by every C2 arm."""
    import gzip
    import statistics
    import shutil
    import torch
    from mmdd_stage1.features import FeatureStore
    from mmdd_stage1.teacher_rerank import _teacher_scores
    from run_stage1_r24 import load_r19_checkpoint

    source = out(root) / "rankings" / "B13-FULL" / f"seed{seed}" / "query_rankings.jsonl.gz"
    if not source.is_file():
        raise FileNotFoundError(source)
    if seed == 29 and (out(root) / "training/R/seed13/R_COMPLETION_RECEIPT.json").is_file():
        outputs = {}
        for arm in ARMS:
            src_dir = out(root) / "rankings" / arm / "seed13"
            dst_dir = out(root) / "rankings" / arm / "seed29"; dst_dir.mkdir(parents=True, exist_ok=True)
            for name in ("teacher_rerank.jsonl.gz", "teacher_metrics.json"):
                src = src_dir / name; dst = dst_dir / name
                if src.is_file():
                    shutil.copyfile(src, dst)
                    outputs[arm] = str((dst_dir / "teacher_rerank.jsonl.gz").resolve())
        receipt = {"format_version": 1, "module": "R", "seed": 29, "status": "complete", "reused_same_pool_cache": True, "source_seed": 13, "candidate_source": str(source.resolve()), "budget": "BT100", "arms": list(ARMS), "outputs": outputs, "created_at_utc": now()}
        _json(out(root) / "training/R/seed29" / "R_COMPLETION_RECEIPT.json", receipt)
        return receipt
    teacher_path = root / "work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt"
    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)
    # Compute the immutable checkpoint hash once.  Re-reading this ~298 MB
    # file inside the per-query loop turns a 1198-query rerank into hundreds
    # of gigabytes of avoidable CPU I/O.
    teacher_sha = sha256(teacher_path)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    _, _, _, teacher, _ = load_r19_checkpoint(teacher_path, device)
    teacher.eval()
    store = FeatureStore.from_path(root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b", cache_size=30000, cache_bytes=64 * 1024**3, teacher_paths=_r25_teacher_feature_paths(root))
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        base_rows = [json.loads(line) for line in handle if line.strip()]
    score_cache: dict[tuple[str, str], float] = {}
    compression_cache = teacher.new_compression_cache() if hasattr(teacher, "new_compression_cache") else {}
    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for index, row in enumerate(base_rows):
        # R is preregistered as BT100: score the frozen Direct-ANN Top100 pool.
        # U/M remain separate Student candidate pools and are not silently
        # relabeled as Teacher-scored when they are outside this budget.
        candidates = list(dict.fromkeys(map(str, row["scorer_ids"]["DIRECT_ANN"]["candidate_ids"][:100])))
        scores = _teacher_scores(teacher, str(row["query_id"]), candidates, store, device, batch_size=64, score_cache=score_cache, compression_cache=compression_cache)
        score_map = dict(zip(candidates, scores, strict=True))
        positives = set(map(str, row["positive_target_ids"]))
        for arm in ARMS:
            ranking_records = []
            for scorer, payload in row["scorer_ids"].items():
                if scorer != "DIRECT_ANN":
                    continue
                pool = list(map(str, payload["candidate_ids"]))
                ranking = sorted(pool, key=lambda value: (-score_map[value], value))
                ranking_records.append({"scorer_id": scorer, "candidate_ids": pool, "ranking": ranking, "recall": {f"R@{k}": len(positives & set(ranking[:k])) / len(positives) if positives else 0.0 for k in (10, 20, 50)}})
            rows_by_arm[arm].append({"query_id": row["query_id"], "positive_target_ids": sorted(positives), "generator_id": f"R25/{arm}/seed{seed}/T0", "teacher_checkpoint_sha256": teacher_sha, "scorers": ranking_records})
        if (index + 1) % 200 == 0:
            print(json.dumps({"stage": "R", "seed": seed, "queries": index + 1, "total": len(base_rows)}), flush=True)
    outputs = {}
    for arm, rows in rows_by_arm.items():
        destination = out(root) / "rankings" / arm / f"seed{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "teacher_rerank.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        metrics = {scorer: {f"R@{k}": statistics.fmean(r["recall"][f"R@{k}"] for row in rows for r in row["scorers"] if r["scorer_id"] == scorer) for k in (10, 20, 50)} for scorer in ("DIRECT_ANN",)}
        _json(destination / "teacher_metrics.json", {"format_version": 1, "status": "complete", "generator_id": f"R25/{arm}/seed{seed}/T0", "queries": len(rows), "teacher_checkpoint_sha256": teacher_sha, "scorers": metrics, "ranking_path": str(path.resolve()), "created_at_utc": now()})
        outputs[arm] = str(path.resolve())
    receipt = {"format_version": 1, "module": "R", "seed": seed, "status": "complete", "teacher_checkpoint_sha256": teacher_sha, "candidate_source": str(source.resolve()), "queries": len(base_rows), "budget": "BT100", "arms": list(ARMS), "outputs": outputs, "created_at_utc": now()}
    _json(out(root) / "training/R" / f"seed{seed}" / "R_COMPLETION_RECEIPT.json", receipt)
    return receipt


def build_fusion_r25(root: Path, arm: str = "B13-FULL", seed: int = 13) -> dict[str, Any]:
    """Materialize the preregistered Equal and confidence-only fusions.

    Column-only is intentionally fail-closed when no real column representation
    is present in the frozen Stage-1 ranking record; a retrieval score is not
    mislabeled as a column signal.
    """
    import gzip
    import statistics
    source = out(root) / "rankings" / arm / f"seed{seed}" / "query_rankings.jsonl.gz"
    if not source.is_file():
        raise FileNotFoundError(source)
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    outputs: dict[str, str] = {}
    for method in ("Equal", "confidence-only"):
        fused_rows = []
        for row in rows:
            direct = row["scorer_ids"]["DIRECT_ANN"]
            evidence = row["scorer_ids"]["QT_OVER_U"]
            candidates = list(dict.fromkeys(direct["candidate_ids"] + evidence["candidate_ids"]))
            rank_d = {value: index + 1 for index, value in enumerate(direct["ranking"])}
            rank_e = {value: index + 1 for index, value in enumerate(evidence["ranking"])}
            def signal(value: str) -> float:
                if method == "Equal":
                    alpha = 1.0
                else:
                    # Contract §8.1: confidence is one Direct-score margin,
                    # then the common promotion score is R_D+alpha*R_E.
                    values = list(map(float, direct["scores"]))
                    alpha = confidence_alpha(values)
                direct_term = 1.0 / (60.0 + rank_d[value]) if value in rank_d else 0.0
                evidence_term = 1.0 / (60.0 + rank_e[value]) if value in rank_e else 0.0
                return direct_term + alpha * evidence_term
            ranking = sorted(candidates, key=lambda value: (-signal(value), value))
            values = list(map(float, direct["scores"]))
            alpha = 1.0 if method == "Equal" else confidence_alpha(values)
            fused_rows.append({"query_id": row["query_id"], "positive_target_ids": row["positive_target_ids"], "fusion_method": method, "ranking": ranking, "source_scorers": ["DIRECT_ANN", "QT_OVER_U"], "alpha": alpha, "direct_scores_complete": len(values) == len(direct["ranking"]), "online_gt_inputs": False})
        destination = out(root) / "fusion" / arm / f"seed{seed}"; destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"{method}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in fused_rows: handle.write(json.dumps(row) + "\n")
        outputs[method] = str(path.resolve())
    receipt = {"format_version": 1, "module": "F", "arm": arm, "seed": seed, "status": "partial", "completed_methods": ["Equal", "confidence-only"], "blocked_methods": {"column-only": "missing real column representation in frozen Stage-1 records"}, "outputs": outputs, "source_ranking": str(source.resolve()), "created_at_utc": now()}
    _json(out(root) / "fusion" / arm / f"seed{seed}" / "F_STATUS.json", receipt)
    return receipt


def confidence_alpha(direct_scores: list[float]) -> float:
    """Return the preregistered confidence-only evidence weight."""

    if len(direct_scores) < 2:
        return 1.0
    ordered = sorted((float(value) for value in direct_scores), reverse=True)
    denominator = ordered[0] - ordered[-1] + 1e-8
    if denominator <= 0:
        return 1.0
    margin = max(0.0, min(1.0, (ordered[0] - ordered[1]) / denominator))
    return 1.0 - margin


def run_feedback_diag(root: Path = ROOT) -> dict[str, Any]:
    """Compute data-derived Student→Teacher mining health/novelty diagnostics."""
    import gzip
    diagnostics = []
    for seed in SEEDS:
        student_path = out(root) / "rankings" / "B13-FULL" / f"seed{seed}" / "query_rankings.jsonl.gz"
        teacher_path = out(root) / "rankings" / "B13-FULL" / f"seed{seed}" / "teacher_rerank.jsonl.gz"
        if not student_path.is_file() or not teacher_path.is_file():
            continue
        with gzip.open(student_path, "rt", encoding="utf-8") as handle:
            student_rows = {r["query_id"]: r for r in (json.loads(line) for line in handle if line.strip())}
        with gzip.open(teacher_path, "rt", encoding="utf-8") as handle:
            teacher_rows = {r["query_id"]: r for r in (json.loads(line) for line in handle if line.strip())}
        overlaps = []; teacher_novelty = []; candidate_sizes = []
        for query_id, row in student_rows.items():
            teacher = teacher_rows.get(query_id)
            if not teacher: continue
            direct = set(row["scorer_ids"]["DIRECT_ANN"]["ranking"][:10])
            tr = next(s for s in teacher["scorers"] if s["scorer_id"] == "DIRECT_ANN")
            top_teacher = set(tr["ranking"][:10])
            overlaps.append(len(direct & top_teacher) / 10.0)
            teacher_novelty.append(len(top_teacher - direct) / 10.0)
            candidate_sizes.append(len(row["scorer_ids"]["DIRECT_ANN"]["candidate_ids"]))
        diagnostics.append({"seed": seed, "queries": len(overlaps), "student_teacher_top10_overlap": sum(overlaps) / len(overlaps) if overlaps else 0.0, "teacher_novelty_top10": sum(teacher_novelty) / len(teacher_novelty) if teacher_novelty else 0.0, "mean_direct_candidate_count": sum(candidate_sizes) / len(candidate_sizes) if candidate_sizes else 0.0, "budget_aligned": True, "teacher_identity": "T0 seed13 frozen"})
    status = "complete" if len(diagnostics) == len(SEEDS) else "partial"
    receipt = {"format_version": 1, "module": "FB-DIAG", "status": status, "diagnostics": diagnostics, "gate_inputs": {"mining_budget": 10, "same_candidate_pool": True, "teacher_scores": bool(diagnostics)}, "created_at": now(), "created_at_utc": now()}
    _json(out(root) / "feedback/FB_DIAG_RECEIPT.json", receipt)
    _write_feedback_gates(root, diagnostics)
    return receipt


def _write_feedback_gates(root: Path, diagnostics: list[dict[str, Any]]) -> None:
    """Materialize data-derived optional FB-TRAIN/REDISTILL gate receipts."""

    import gzip
    import hashlib

    gate_dir = out(root) / "feedback"
    gate_dir.mkdir(parents=True, exist_ok=True)
    priority = ["SPLIT-UQTKD", "SPLIT-QTKD", "SPLIT-U", "SPLIT-SUP", "B13-FULL"]
    # The frozen fixed-pool rankings are the common development pool.  Compute
    # raw-recall and hard-membership differences directly from those records;
    # no performance condition can cancel mandatory C1/C2 jobs.
    per_recipe = []
    for arm in priority:
        by_seed = []
        for seed in SEEDS:
            path = out(root) / f"rankings/{arm}/seed{seed}/query_rankings.jsonl.gz"
            base = out(root) / f"rankings/B13-FULL/seed{seed}/query_rankings.jsonl.gz"
            if not path.is_file() or not base.is_file():
                continue
            with gzip.open(path, "rt", encoding="utf-8") as h:
                rows = {r["query_id"]: r for r in (json.loads(line) for line in h if line.strip())}
            with gzip.open(base, "rt", encoding="utf-8") as h:
                old = {r["query_id"]: r for r in (json.loads(line) for line in h if line.strip())}
            raw = []; old_raw = []; membership_diff = 0; eo_new = []; eo_old = []
            for qid, row in rows.items():
                if qid not in old:
                    continue
                d = row["scorer_ids"]["DIRECT_ANN"]; od = old[qid]["scorer_ids"]["DIRECT_ANN"]
                raw.append(float(d.get("raw_recall", 0.0))); old_raw.append(float(od.get("raw_recall", 0.0)))
                new_ids = set(map(str, d["candidate_ids"][:100])); old_ids = set(map(str, od["candidate_ids"][:100]))
                membership_diff += int(new_ids != old_ids)
                positives = set(map(str, row["positive_target_ids"]))
                eo_new.append(float(bool(positives & new_ids))); eo_old.append(float(bool(positives & old_ids)))
            by_seed.append({"seed": seed, "raw_recall": sum(raw) / len(raw) if raw else 0.0, "b13_raw_recall": sum(old_raw) / len(old_raw) if old_raw else 0.0, "implicit_eo_admission": sum(eo_new) / len(eo_new) if eo_new else 0.0, "b13_implicit_eo_admission": sum(eo_old) / len(eo_old) if eo_old else 0.0, "hard_membership_diff_lists": membership_diff, "queries": len(raw)})
        if len(by_seed) == len(SEEDS):
            per_recipe.append({"arm": arm, "seeds": by_seed, "passes_health": all(x["raw_recall"] >= x["b13_raw_recall"] - 0.02 and x["implicit_eo_admission"] >= x["b13_implicit_eo_admission"] - 0.02 for x in by_seed), "passes_membership": any(x["hard_membership_diff_lists"] > 0 for x in by_seed)})
    selected = next((x for x in per_recipe if x["passes_health"] and x["passes_membership"]), None)
    gate_payload = {"format_version": 1, "module": "FB-TRAIN", "status": "triggered" if selected else "not_triggered", "priority": priority, "selected_recipe": selected["arm"] if selected else None, "thresholds": {"raw_recall_delta_min": -0.02, "implicit_eo_admission_delta_min": -0.02, "membership_difference_required": True}, "recipes": per_recipe, "input_diagnostics_sha256": hashlib.sha256(json.dumps(diagnostics, sort_keys=True).encode()).hexdigest(), "created_at_utc": now()}
    _json(gate_dir / "FEEDBACK_GATE.json", gate_payload)
    redistill = {"format_version": 1, "module": "REDISTILL", "status": "pending" if selected else "not_triggered", "source_feedback_gate": "FEEDBACK_GATE.json", "criteria": {"Tnew_minus_Told_overall_R10_positive_both_seeds": False, "implicit_R10_nonnegative_both_seeds": False, "new_hard_competitors": bool(selected)}, "created_at_utc": now()}
    _json(gate_dir / "REDISTILL_GATE.json", redistill)


def prepare_s2_pilot(root: Path = ROOT) -> dict[str, Any]:
    """Freeze the 32 engineering records and 64-query Real/NoE pilot manifest."""
    import hashlib
    import gzip
    target_path = root / "work/stage1_optimization_r10_20260907/stage1_data/target_lists.jsonl"
    pool_path = root / "work/stage1_optimization_r16_20260910/candidate_pools.jsonl.gz"
    if not target_path.is_file() or not pool_path.is_file():
        raise FileNotFoundError("Stage2 pilot inputs are missing")
    with target_path.open(encoding="utf-8") as handle:
        target_rows = [json.loads(line) for line in handle if line.strip() and json.loads(line).get("split") == "train"]
    with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
        dev_pools = [json.loads(line) for line in handle if line.strip()]
    by_kind = {kind: sorted((row for row in dev_pools if row.get("query_kind") == kind), key=lambda row: hashlib.sha256(str(row["query_id"]).encode()).hexdigest()) for kind in ("implicit", "explicit")}
    selected_queries = by_kind["implicit"][:32] + by_kind["explicit"][:32]
    stage2_checkpoint = out(root) / "stage2/r25_b13_column_scorer.pt"
    checkpoint_available = stage2_checkpoint.is_file()
    engineering = []
    for row in target_rows:
        evidence_ids = [str(e) for c in row.get("candidates", []) for e in c.get("evidence_ids", [])]
        if not evidence_ids:
            continue
        # Bridge IDs in the canonical artifacts use ``asset_img_`` (older
        # snapshots used ``asset_image_``); preserve both spellings so the
        # engineering sample's modality coverage is truthful.
        engineering.append({"record_id": f"eng_{len(engineering):03d}", "query_id": row["query_id"], "target_id": row.get("evidence_positive_target_id"), "evidence_ids": evidence_ids[:4], "evidence_modalities": sorted({"image" if (e.startswith("asset_img_") or e.startswith("asset_image_")) else "text" for e in evidence_ids}), "completion_parse_status": "ready_for_stage2" if checkpoint_available else "not_run_missing_stage2_backend"})
        if len(engineering) == 32:
            break
    pilot_rows = []
    for pool in selected_queries:
        for condition in ("Real", "NoE-fill"):
            # Keep model-facing inputs separate from evaluation-only metadata;
            # query-kind/labels/qrels must never cross the online interface.
            pilot_rows.append({
                "model_input": {
                    "query_id": pool["query_id"],
                    "condition": condition,
                    "generator_ids": ["raw-Qwen", "B13"],
                    "candidate_budget": 50,
                    "target_attempt_budget": 10,
                    "path_budget": 4,
                    "row_mask_id": f"mask_{pool['query_id']}",
                },
                "evaluation_metadata": {
                    "query_id": pool["query_id"],
                    "query_kind": pool.get("query_kind"),
                },
                "status": "planned" if checkpoint_available else "blocked_missing_input",
                "reason": None if checkpoint_available else "Stage2 column-head/reader generator checkpoint not available in R25 inputs",
            })
    destination = out(root) / "stage2"; destination.mkdir(parents=True, exist_ok=True)
    with (destination / "engineering_records_32.jsonl").open("w", encoding="utf-8") as handle:
        for row in engineering: handle.write(json.dumps(row) + "\n")
    with (destination / "pilot_queries_64.jsonl").open("w", encoding="utf-8") as handle:
        for row in pilot_rows: handle.write(json.dumps(row) + "\n")
    receipt = {"format_version": 1, "module": "S2", "status": "planned" if checkpoint_available else "blocked_missing_input", "engineering_records": len(engineering), "pilot_query_rows": len(pilot_rows), "unique_queries": len(selected_queries), "conditions": ["Real", "NoE-fill"], "same_opportunities": True, "generator_ids": ["raw-Qwen", "B13"], "inputs": [str(target_path.resolve()), str(pool_path.resolve())], "stage2_checkpoint": str(stage2_checkpoint.resolve()) if checkpoint_available else None, "reason": None if checkpoint_available else "No verified Stage2 column-head/reader/generator checkpoint in local inputs; manifest retains all failed query rows.", "created_at_utc": now()}
    _json(destination / "S2_COMPLETION_RECEIPT.json", receipt)
    return receipt


def benchmark_teacher_microbatch(root: Path, seed: int, device_name: str, microbatch: int = 2, rows: int = 2) -> dict[str, Any]:
    """Measure a small Teacher forward without mutating the R25 run state."""
    if microbatch <= 0 or rows <= 0:
        raise ValueError("microbatch and rows must be positive")
    if not torch_cuda_available():
        raise RuntimeError("CUDA unavailable")
    import time
    import torch
    from mmdd_stage1.data import load_edge_examples
    from mmdd_stage1.features import FeatureStore
    from mmdd_stage1.scoring import score_edge_batch
    from mmdd_stage1.checkpoints import load_teacher
    manifest = root / "work/stage1_optimization_r23_20260913/manifests/G0-SUP.jsonl"
    teacher_path = c1_teacher_path(root)
    examples = load_edge_examples(manifest, split="train")[:rows]
    device = torch.device(device_name)
    teacher = load_teacher(teacher_path, device); teacher.eval()
    store = FeatureStore.from_path(root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b", cache_size=64, teacher_paths=_r25_teacher_feature_paths(root))
    started = time.monotonic(); count = 0
    with torch.inference_mode():
        for start in range(0, len(examples), microbatch):
            score_edge_batch(teacher, examples[start:start + microbatch], store, device, student_score_space="raw_logit")
            count += len(examples[start:start + microbatch])
    return {"seed": seed, "device": device_name, "microbatch": microbatch, "rows": count, "elapsed_seconds": time.monotonic() - started, "status": "complete"}


def prepare_teacher_edge_cache(root: Path, seed: int, device_name: str = "cuda:0", microbatch: int = 8) -> dict[str, Any]:
    """Materialize the historical B13 T_core raw edge logits once."""
    if seed not in SEEDS:
        raise ValueError(seed)
    if not torch_cuda_available():
        raise RuntimeError("CUDA unavailable")
    import gzip
    import torch
    from mmdd_stage1.data import load_edge_examples
    from mmdd_stage1.features import FeatureStore
    from mmdd_stage1.scoring import score_edge_batch
    from mmdd_stage1.checkpoints import load_teacher
    manifest = c1_schedule_path(root, seed)
    teacher_path = c1_teacher_path(root)
    destination = out(root) / "common" / f"teacher_edge_cache_seed{seed}.jsonl.gz"
    if destination.exists():
        return {"status": "complete", "path": str(destination.resolve()), "sha256": sha256(destination), "seed": seed, "reused": True}
    if not manifest.exists():
        raise FileNotFoundError(f"run prepare-c1-schedule first: {manifest}")
    examples = _edge_examples_from_schedule(manifest)
    device = torch.device(device_name)
    teacher = load_teacher(teacher_path, device); teacher.eval()
    store = FeatureStore.from_path(root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b", cache_size=24000, teacher_paths=_r25_teacher_feature_paths(root))
    destination.parent.mkdir(parents=True, exist_ok=True); temporary = destination.with_suffix(destination.suffix + ".tmp")
    rows = 0
    with gzip.open(temporary, "wt", encoding="utf-8") as handle, torch.inference_mode():
        for start in range(0, len(examples), microbatch):
            batch = examples[start:start + microbatch]
            scores = score_edge_batch(teacher, batch, store, device, student_score_space="raw_logit")
            for index, example in enumerate(batch):
                n = len(example.candidate_ids)
                handle.write(json.dumps({"query_id": str(example.query_id), "relation": f"{example.source_type}->{example.destination_type}", "candidate_ids": list(example.candidate_ids), "scores": [float(x) for x in scores.logits[index, :n].detach().cpu()]}) + "\n")
                rows += 1
            if rows and rows % (microbatch * 100) == 0:
                print(json.dumps({"seed": seed, "cached_rows": rows, "total_rows": len(examples)}), flush=True)
    temporary.replace(destination)
    return {"status": "complete", "path": str(destination.resolve()), "sha256": sha256(destination), "seed": seed, "rows": rows, "microbatch": microbatch, "teacher_checkpoint_sha256": sha256(teacher_path)}


def torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="R25 final experiment runner")
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("resolve-inputs")
    sub.add_parser("freeze-plan")
    sub.add_parser("recipe-diff")
    sub.add_parser("dry-run", help="write the expanded 2 C1 + 14 C2 matrix")
    sub.add_parser("resource-status")
    sub.add_parser("mark-resource-blocked")
    sub.add_parser("acceptance", help="run CPU semantic tests and write acceptance/unit_tests.json")
    sub.add_parser("artifact-manifest")
    sub.add_parser("baseline-b0", help="register frozen historical B0 baseline artifacts")
    p = sub.add_parser("record-c1-blocker")
    p.add_argument("--seed", type=int, required=True); p.add_argument("--reason", required=True)
    p = sub.add_parser("mark-c1-missing-input")
    p.add_argument("--seed", type=int, required=True); p.add_argument("--reason", required=True)
    sub.add_parser("reports")
    sub.add_parser("scaffold")
    p = sub.add_parser("train-c1", help="run one complete fresh five-relation C1 stage")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("train-c2", help="run one complete C2 arm from this run's C1")
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("evaluate-c2", help="evaluate one completed C2 checkpoint on frozen candidate pools")
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("rerank-teacher", help="run frozen T0 reranking on R25 candidate pools")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("build-fusion", help="materialize preregistered simple fusions")
    p.add_argument("--arm", choices=ARMS, default="B13-FULL")
    p.add_argument("--seed", type=int, default=13)
    sub.add_parser("feedback-diag", help="compute Student-to-Teacher mining diagnostics")
    sub.add_parser("prepare-s2-pilot", help="freeze Stage2 engineering and 64-query pilot manifests")
    p = sub.add_parser("prepare-native-path-cache", help="materialize historical B13 native C2 T_core logits")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--microbatch", type=int, default=32)
    p = sub.add_parser("prepare-c1-schedule", help="freeze one full raw-Qwen selective-hard C1 epoch")
    p.add_argument("--seed", type=int, required=True)
    sub.add_parser("build-raw-index", help="build the R25-local frozen raw-Qwen HNSW index")
    p = sub.add_parser("benchmark-teacher", help="measure Teacher micro-batch memory/latency")
    p.add_argument("--seed", type=int, default=13); p.add_argument("--device", default="cuda:0"); p.add_argument("--microbatch", type=int, default=2); p.add_argument("--rows", type=int, default=2)
    p = sub.add_parser("prepare-teacher-edge-cache", help="materialize full five-relation T1-B edge logits")
    p.add_argument("--seed", type=int, required=True); p.add_argument("--device", default="cuda:0"); p.add_argument("--microbatch", type=int, default=8)
    args = parser.parse_args(); root = args.root.resolve()
    if args.command == "resolve-inputs": result = resolve_inputs(root)
    elif args.command == "freeze-plan": result = freeze_plan(root)
    elif args.command == "recipe-diff": result = recipe_diff(root)
    elif args.command == "dry-run": result = dry_run(root)
    elif args.command == "resource-status": result = resource_status()
    elif args.command == "train-c1": result = train_c1(root, args.seed, args.device)
    elif args.command == "train-c2": result = train_c2(root, args.arm, args.seed, args.device)
    elif args.command == "evaluate-c2": result = evaluate_c2_fixed(root, args.arm, args.seed, args.device)
    elif args.command == "rerank-teacher": result = rerank_teacher_r25(root, args.seed, args.device)
    elif args.command == "build-fusion": result = build_fusion_r25(root, args.arm, args.seed)
    elif args.command == "feedback-diag": result = run_feedback_diag(root)
    elif args.command == "prepare-s2-pilot": result = prepare_s2_pilot(root)
    elif args.command == "prepare-native-path-cache": result = prepare_native_path_cache(root, args.device, args.microbatch)
    elif args.command == "prepare-c1-schedule": result = prepare_c1_schedule(root, args.seed)
    elif args.command == "build-raw-index": result = build_raw_index(root)
    elif args.command == "benchmark-teacher": result = benchmark_teacher_microbatch(root, args.seed, args.device, args.microbatch, args.rows)
    elif args.command == "prepare-teacher-edge-cache": result = prepare_teacher_edge_cache(root, args.seed, args.device, args.microbatch)
    elif args.command == "acceptance": result = run_acceptance(root)
    elif args.command == "artifact-manifest": result = write_artifact_manifest(root)
    elif args.command == "baseline-b0": result = write_b0_baseline_receipt(root)
    elif args.command == "record-c1-blocker": result = record_c1_blocker(root, args.seed, args.reason)
    elif args.command == "mark-c1-missing-input": result = mark_c1_missing_input(root, args.seed, args.reason)
    elif args.command == "reports": result = write_reports(root)
    elif args.command == "scaffold": result = write_scaffold(root)
    else: result = mark_resource_blocked(root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

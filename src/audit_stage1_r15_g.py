#!/usr/bin/env python
"""Audit R14 residual checkpoints before R15 interaction training."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import load_edge_examples, load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.pca import load_pca_projection
from mmdd_stage1.retrieval import StudentANNIndices, build_indices, load_corpus_ids
from reevaluate_stage1_r12_checkpoints import _requests, exact_relation
from run_stage1_r13 import _paths as r13_paths
from run_stage1_r15 import ARM_SPECS, freeze_plan, optimizer_audit, output_root


FP32_ATOL = 1e-4
FP32_RTOL = 1e-5


def merged_linear_model(model: Any) -> Any:
    """Return an inference copy with P + BA/c folded into the base projection."""

    if model.projection_adapter != "linear":
        raise ValueError("Only a linear residual can be merged exactly")
    merged = copy.deepcopy(model)
    with torch.no_grad():
        for key in merged.projection_keys:
            effective = (
                merged.projections[key].weight
                + merged.projection_residual_outputs[key].weight
                @ merged.projection_residual_inputs[key].weight
                / merged.projection_scales[key]
            )
            merged.projections[key].weight.copy_(effective)
    merged.projection_adapter = "none"
    return merged


def adapter_off_model(model: Any) -> Any:
    """Keep the trained P/R endpoint and disable only the residual branch."""

    disabled = copy.deepcopy(model)
    disabled.projection_adapter = "none"
    return disabled


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    points = torch.tensor(
        [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0],
        device=values.device,
        dtype=torch.float32,
    )
    result = torch.quantile(values.float().reshape(-1), points).cpu().tolist()
    return dict(zip(("min", "p01", "p10", "p50", "p90", "p99", "max"), result))


@torch.inference_mode()
def projection_diagnostics(
    model: Any,
    step0: Any,
    s0: Any,
    store: FeatureStore,
    ids_by_type: dict[str, list[str]],
    device: torch.device,
    *,
    sample_size: int = 1024,
) -> dict[str, Any]:
    by_type = {}
    for object_type, object_ids in ids_by_type.items():
        selected = object_ids[:sample_size]
        embeddings = torch.stack(
            [store.embedding_features(object_id).embedding for object_id in selected]
        ).to(device=device, dtype=torch.float32)
        key = model.projection_key(object_type)
        base = model.projections[key](embeddings)
        raw_hidden = model.projection_residual_inputs[key](embeddings)
        scaled_hidden = raw_hidden / model.projection_scales[key]
        gelu_hidden = F.gelu(scaled_hidden)
        used = gelu_hidden if model.projection_adapter == "gelu" else scaled_hidden
        residual = model.projection_residual_outputs[key](used)
        if model.projection_adapter == "none":
            residual = torch.zeros_like(base)
        full = base + residual
        s0_full = s0.project(embeddings, object_type)
        centered = full - full.mean(dim=0, keepdim=True)
        singular = torch.linalg.svdvals(centered)
        energy = singular.square()
        probabilities = energy / energy.sum().clamp_min(torch.finfo(energy.dtype).tiny)
        effective_rank = float(
            torch.exp(-(probabilities * probabilities.clamp_min(1e-30).log()).sum())
        )
        pair_cosine = F.cosine_similarity(full[0::2], full[1::2], dim=1)
        base_mean = base.mean(dim=0)
        full_mean = full.mean(dim=0)
        s0_mean = s0_full.mean(dim=0)
        base_rms = float(base.square().mean().sqrt())
        residual_rms = float(residual.square().mean().sqrt())
        input_weight = model.projection_residual_inputs[key].weight
        output_weight = model.projection_residual_outputs[key].weight
        by_type[object_type] = {
            "residual_enabled": model.projection_adapter != "none",
            "samples": len(selected),
            "sample_ids_sha256": checkpoint_fingerprint_from_ids(selected),
            "projection_scale_c_tau": float(model.projection_scales[key]),
            "raw_Az": {
                "rms": float(raw_hidden.square().mean().sqrt()),
                "quantiles": _quantiles(raw_hidden),
            },
            "scaled_Az": {
                "rms": float(scaled_hidden.square().mean().sqrt()),
                "quantiles": _quantiles(scaled_hidden),
            },
            "gelu_scaled_Az": {
                "rms": float(gelu_hidden.square().mean().sqrt()),
                "quantiles": _quantiles(gelu_hidden),
            },
            "base_output": {
                "rms": base_rms,
                "norm_quantiles": _quantiles(base.norm(dim=1)),
                "mean_vector": base_mean.cpu().tolist(),
                "mean_vector_norm": float(base_mean.norm()),
            },
            "residual_output": {
                "rms": residual_rms,
                "norm_quantiles": _quantiles(residual.norm(dim=1)),
            },
            "full_output": {
                "rms": float(full.square().mean().sqrt()),
                "norm_quantiles": _quantiles(full.norm(dim=1)),
                "mean_vector": full_mean.cpu().tolist(),
                "mean_vector_norm": float(full_mean.norm()),
                "random_pair_cosine": _quantiles(pair_cosine),
                "effective_rank": effective_rank,
                "stable_rank": float(energy.sum() / energy.max()),
                "top_singular_values": singular[:20].cpu().tolist(),
            },
            "s0_output": {
                "mean_vector": s0_mean.cpu().tolist(),
                "mean_vector_norm": float(s0_mean.norm()),
            },
            "residual_to_base_rms": residual_rms / base_rms if base_rms else None,
            "full_vs_s0_direction_cosine": _quantiles(
                F.cosine_similarity(full, s0_full, dim=1)
            ),
            "parameters": {
                "A_weight_norm": float(input_weight.norm()),
                "A_update_norm_from_step0": float(
                    (input_weight - step0.projection_residual_inputs[key].weight).norm()
                ),
                "B_weight_norm": float(output_weight.norm()),
                "B_update_norm_from_step0": float(
                    (output_weight - step0.projection_residual_outputs[key].weight).norm()
                ),
                "P_update_norm_from_step0": float(
                    (
                        model.projections[key].weight
                        - step0.projections[key].weight
                    ).norm()
                ),
            },
        }
    return {"sample_size_per_type": sample_size, "by_type": by_type}


def checkpoint_fingerprint_from_ids(values: list[str]) -> str:
    import hashlib

    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@torch.inference_mode()
def score_path_consistency(
    model: Any,
    reloaded: Any,
    indices: StudentANNIndices,
    store: FeatureStore,
    requests: list[tuple[Any, ...]],
    device: torch.device,
) -> dict[str, Any]:
    relations = {}
    for request in requests:
        name, source_ids, source_type, destination_type, _k, positives = request
        destination_corpus = indices.object_ids[destination_type]
        selected_sources = source_ids[:16]
        selected_destinations = []
        for source_id in selected_sources:
            known = sorted(
                set(positives.get((source_id, destination_type), set()))
                & set(destination_corpus)
            )
            selected_destinations.append(
                known[0] if known else destination_corpus[len(selected_destinations)]
            )
        source_embeddings = torch.stack(
            [store.embedding_features(value).embedding for value in selected_sources]
        ).to(device=device, dtype=torch.float32)
        destination_embeddings = torch.stack(
            [
                store.embedding_features(value).embedding
                for value in selected_destinations
            ]
        ).to(device=device, dtype=torch.float32)
        forward = model.score_embeddings(
            source_embeddings,
            source_type,
            destination_embeddings,
            destination_type,
        )
        query_vectors = model.relation_query(
            source_embeddings,
            source_type,
            destination_type,
            source_role="query" if source_type == "table" else None,
        )
        destination_vectors = model.index_vector(
            destination_embeddings,
            destination_type,
            destination_role="target" if destination_type == "table" else None,
        )
        vector_ip = (query_vectors * destination_vectors).sum(dim=-1)
        reloaded_scores = reloaded.score_embeddings(
            source_embeddings,
            source_type,
            destination_embeddings,
            destination_type,
        )
        ann = indices.search_many(selected_sources, destination_type, 5)
        ann_score_errors = []
        ann_score_magnitudes = []
        for row, source_embedding in zip(ann, source_embeddings):
            ids = [value for value, _score in row]
            embeddings = torch.stack(
                [store.embedding_features(value).embedding for value in ids]
            ).to(device=device, dtype=torch.float32)
            exact = model.score_embedding_matrix(
                source_embedding.unsqueeze(0),
                source_type,
                embeddings,
                destination_type,
            )[0]
            ann_score_errors.extend(
                abs(float(saved) - float(actual))
                for (_value, saved), actual in zip(row, exact)
            )
            ann_score_magnitudes.extend(abs(float(value)) for value in exact)
        max_forward_ip = float((forward - vector_ip).abs().max())
        max_reload = float((forward - reloaded_scores).abs().max())
        max_ann_rescore = max(ann_score_errors, default=0.0)
        max_ann_magnitude = max(ann_score_magnitudes, default=1.0)
        relations[name] = {
            "pairs": len(selected_sources),
            "max_abs_forward_vs_vector_ip": max_forward_ip,
            "max_abs_forward_vs_checkpoint_reload": max_reload,
            "max_abs_ann_saved_score_vs_exact_rescore": max_ann_rescore,
            "passed": bool(
                torch.allclose(forward, vector_ip, atol=FP32_ATOL, rtol=FP32_RTOL)
                and torch.allclose(
                    forward, reloaded_scores, atol=FP32_ATOL, rtol=FP32_RTOL
                )
                and max_ann_rescore <= FP32_ATOL
                + FP32_RTOL * max(max_ann_magnitude, 1.0)
            ),
        }
    return {
        "dtype": "float32",
        "atol": FP32_ATOL,
        "rtol": FP32_RTOL,
        "relations": relations,
        "passed": all(row["passed"] for row in relations.values()),
    }


@torch.inference_mode()
def full_dev_direct_exact_ann(
    model: Any,
    indices: StudentANNIndices | None,
    store: FeatureStore,
    examples: list[Any],
    target_ids: list[str],
    device: torch.device,
    *,
    batch_size: int = 16,
) -> dict[str, Any]:
    destination_embeddings = torch.stack(
        [store.embedding_features(value).embedding for value in target_ids]
    ).to(device=device, dtype=torch.float32)
    destination_vectors = model.index_vector(
        destination_embeddings, "table", destination_role="target"
    )
    target_position = {value: index for index, value in enumerate(target_ids)}
    per_query = []
    exact_hubs: Counter[str] = Counter()
    ann_hubs: Counter[str] = Counter()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        query_ids = [row.query_id for row in batch]
        query_embeddings = torch.stack(
            [store.embedding_features(value).embedding for value in query_ids]
        ).to(device=device, dtype=torch.float32)
        query_vectors = model.relation_query(
            query_embeddings, "table", "table", source_role="query"
        )
        scores = query_vectors @ destination_vectors.T
        ann_rows = (
            indices.search_many(query_ids, "table", 100)
            if indices is not None
            else [None] * len(batch)
        )
        for offset, (example, ann_row) in enumerate(zip(batch, ann_rows)):
            row_scores = scores[offset]
            top_values, top_positions = torch.topk(row_scores, 101)
            exact_pairs = sorted(
                zip(top_positions.cpu().tolist(), top_values.cpu().tolist()),
                key=lambda pair: (-pair[1], target_ids[pair[0]]),
            )
            exact_ids = [target_ids[position] for position, _score in exact_pairs[:100]]
            exact_hubs.update(exact_ids[:1])
            ann_ids = [value for value, _score in ann_row] if ann_row else []
            ann_hubs.update(ann_ids[:1])
            positives = set(example.positive_target_ids)
            positive_rows = []
            for positive in sorted(positives):
                position = target_position.get(positive)
                if position is None:
                    positive_rows.append(
                        {"target_id": positive, "present": False, "exact_rank": None}
                    )
                    continue
                score = row_scores[position]
                greater = int((row_scores > score).sum())
                equal = int((row_scores == score).sum())
                nonpositive = row_scores.clone()
                for other in positives:
                    other_position = target_position.get(other)
                    if other_position is not None:
                        nonpositive[other_position] = -torch.inf
                positive_rows.append(
                    {
                        "target_id": positive,
                        "present": True,
                        "score": float(score),
                        "exact_rank_min": greater + 1,
                        "exact_rank_max": greater + equal,
                        "ann_rank": ann_ids.index(positive) + 1 if positive in ann_ids else None,
                        "margin_over_best_nonpositive": float(score - nonpositive.max()),
                    }
                )
            record = {
                "query_id": example.query_id,
                "query_kind": example.query_kind,
                "positive_denominator": len(positives),
                "positive_targets": positive_rows,
                "exact_ids": exact_ids,
                "ann_ids": ann_ids,
                "ann_exact_overlap@100": (
                    len(set(ann_ids) & set(exact_ids)) / 100 if ann_ids else None
                ),
            }
            for k in (10, 20, 50, 100):
                record[f"exact_recall@{k}"] = len(positives & set(exact_ids[:k])) / len(positives)
                record[f"ann_recall@{k}"] = (
                    len(positives & set(ann_ids[:k])) / len(positives)
                    if ann_ids
                    else None
                )
            per_query.append(record)
    aggregate = {}
    for kind in ("all", "implicit", "explicit"):
        rows = [row for row in per_query if kind == "all" or row["query_kind"] == kind]
        aggregate[kind] = {
            f"{mode}_recall@{k}": float(np.mean([row[f"{mode}_recall@{k}"] for row in rows]))
            for mode in (("exact", "ann") if indices is not None else ("exact",))
            for k in (10, 20, 50, 100)
        }
        if indices is not None:
            aggregate[kind]["ann_exact_overlap@100"] = float(
                np.mean([row["ann_exact_overlap@100"] for row in rows])
            )
    return {
        "queries": len(per_query),
        "legal_targets": len(target_ids),
        "aggregate": aggregate,
        "exact_top1_hubs": exact_hubs.most_common(20),
        "ann_top1_hubs": ann_hubs.most_common(20),
        "exact_distinct_top1": len(exact_hubs),
        "ann_distinct_top1": len(ann_hubs),
        "per_query": per_query,
    }


def _historical_optimizer_audit(root: Path, arm: str, model: Any) -> dict[str, Any]:
    from run_stage1_r14 import _optimizer

    manifest_path = (
        root
        / "work/stage1_optimization_r14_20260909/stage1_M_projection_capacity"
        / ARM_SPECS[arm]["r14_arm_id"]
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    optimizer = _optimizer(model)
    audit = optimizer_audit(optimizer, model)
    code_path = root / "src/run_stage1_r14.py"
    audit.update(
        {
            "historical_optimizer_tensor_state_available": False,
            "historical_state_limitation": (
                "R14 checkpoints did not serialize Adam moments; parameter groups are "
                "reconstructed from current source and the historical manifest. "
                "Check executed_source_matches_current before claiming source identity."
            ),
            "executed_code_sha256": manifest["code_sha256"],
            "current_code_sha256": checkpoint_fingerprint(code_path),
            "executed_source_matches_current": manifest["code_sha256"]
            == checkpoint_fingerprint(code_path),
            "manifest_evidence_loss_weight": manifest["evidence_loss_weight"],
            "manifest_base_anchor_only": manifest["projection_adapter"][
                "base_anchor_only"
            ],
        }
    )
    return audit


def _existing_step178_index(root: Path, arm: str) -> Path:
    return _r14_directory(root, arm) / "evaluation_step178/index"


def _r14_directory(root: Path, arm: str) -> Path:
    return (
        root
        / "work/stage1_optimization_r14_20260909/stage1_M_projection_capacity"
        / ARM_SPECS[arm]["r14_arm_id"]
    )


def _prepare_index(
    model: Any,
    store: FeatureStore,
    ids_by_type: dict[str, list[str]],
    index_model_sha256: str,
    corpus_sha256: str,
    output: Path,
    device: torch.device,
    index_threads: int,
    *,
    reuse: Path | None = None,
) -> tuple[Path, dict[str, Any], bool, float]:
    if reuse is not None:
        manifest = json.loads((reuse / "manifest.json").read_text(encoding="utf-8"))
        if manifest["student_checkpoint_sha256"] != index_model_sha256:
            raise ValueError("Reused index does not match the checkpoint")
        return reuse, manifest, True, 0.0
    index_dir = output / "index_in_progress"
    if index_dir.exists():
        raise FileExistsError(f"Inspect incomplete temporary index: {index_dir}")
    started = time.monotonic()
    manifest = build_indices(
        model,
        store,
        ids_by_type,
        index_dir,
        device=device,
        checkpoint_sha256=index_model_sha256,
        corpus_sha256=corpus_sha256,
        batch_size=4096,
        num_threads=index_threads,
    )
    return index_dir, manifest, False, time.monotonic() - started


def audit_checkpoint(
    args: argparse.Namespace,
    *,
    arm: str,
    checkpoint_id: str,
    checkpoint_path: Path,
    model_override: Any | None = None,
    reuse_index: Path | None = None,
    output_base: Path | None = None,
    step0_path: Path | None = None,
    optimizer_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = (output_base or (output_root(args.root) / "stageG_correctness" / arm)) / checkpoint_id
    metrics_path = output / "metrics.json"
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            print(json.dumps({"status": "retained", "arm": arm, "checkpoint": checkpoint_id}))
            return payload
        raise FileExistsError(f"Inspect incomplete G output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    paths = r13_paths(args.root)
    targets = load_target_examples(paths["dev_targets"], split="dev")
    edges = load_edge_examples(paths["dev_edges"], split="dev")
    supervision_manifest = json.loads(
        (
            args.root
            / "work/stage1_optimization_r12_20260908/taskA_correctness/supervision/manifest.json"
        ).read_text(encoding="utf-8")
    )
    requests = _requests(targets, edges, Path(supervision_manifest["dataset_root"]))
    store = FeatureStore.from_path(paths["features"], cache_size=260_000)
    ids_by_type = load_corpus_ids(paths["corpus"], store)
    store.preload_embeddings(
        [
            *(value for ids in ids_by_type.values() for value in ids),
            *(row.query_id for row in targets),
        ]
    )
    model = model_override or load_student(checkpoint_path, device).eval()
    reloaded = (
        copy.deepcopy(model).eval()
        if model_override is not None
        else load_student(checkpoint_path, device).eval()
    )
    step0 = load_student(
        step0_path
        or Path(freeze_plan(args.root)["r14_checkpoints"][arm]["0"]["path"]),
        device,
    ).eval()
    s0 = load_student(paths["s0"], device).eval()
    corpus_sha256 = checkpoint_fingerprint(paths["corpus"])
    checkpoint_sha256 = checkpoint_fingerprint(checkpoint_path)
    index_model_sha256 = (
        hashlib.sha256(f"{checkpoint_sha256}:adapter_off".encode()).hexdigest()
        if model_override is not None
        else checkpoint_sha256
    )
    index_dir, index_manifest, reused, index_seconds = _prepare_index(
        model,
        store,
        ids_by_type,
        index_model_sha256,
        corpus_sha256,
        output,
        device,
        args.index_threads,
        reuse=reuse_index,
    )
    indices = StudentANNIndices(
        model,
        store,
        index_dir,
        device=device,
        checkpoint_sha256=index_model_sha256,
        corpus_sha256=corpus_sha256,
        score_space="raw_logit",
    )
    started = time.monotonic()
    consistency = score_path_consistency(
        model, reloaded, indices, store, requests, device
    )
    basis = load_pca_projection(
        args.root
        / "work/stage1_optimization_r10_20260907/baselines/pca_entitables_v9_1024.pt",
        input_dim=store.embedding_dimension(),
        student_dim=1024,
    ).to(device)
    exact_panels = {
        request[0]: exact_relation(
            model, indices, store, ids_by_type, request, basis, device
        )
        for request in requests
    }
    for panel in exact_panels.values():
        for row in panel["per_source"]:
            for edge in row["gt_edges"]:
                edge["margin_over_exact_cutoff"] = edge["score"] - row["cutoff_score"]
    direct = full_dev_direct_exact_ann(
        model,
        indices,
        store,
        targets,
        sorted(ids_by_type["table"]),
        device,
        batch_size=args.query_batch_size,
    )
    geometry = projection_diagnostics(
        model, step0, s0, store, ids_by_type, device
    )
    index_bytes = sum(
        value.stat().st_size for value in index_dir.iterdir() if value.is_file()
    )
    payload = {
        "format_version": 1,
        "status": "complete",
        "arm": arm,
        "checkpoint_id": checkpoint_id,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "index_model_sha256": index_model_sha256,
        "model_override": model_override is not None,
        "score_path_consistency": consistency,
        "full_dev_direct": direct,
        "exact_relation_panels": exact_panels,
        "projection_and_parameter_diagnostics": geometry,
        "optimizer": optimizer_payload or _historical_optimizer_audit(args.root, arm, model),
        "normalization": {
            "project_function": "no output normalization",
            "index_function": "model.index_vector; no second normalization",
            "score_space": "raw_logit inner product",
        },
        "index": {
            "reused": reused,
            "manifest": index_manifest,
            "bytes": index_bytes,
            "temporary_index_retained": reused,
        },
        "cost": {
            "index_build_seconds": index_seconds,
            "audit_seconds": time.monotonic() - started,
            "device": args.device,
            "cpu_threads": args.cpu_threads,
            "index_threads": args.index_threads,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(metrics_path, payload)
    if not reused:
        shutil.rmtree(index_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "arm": arm,
                "checkpoint": checkpoint_id,
                "exact_R@10": direct["aggregate"]["all"]["exact_recall@10"],
                "ann_R@10": direct["aggregate"]["all"]["ann_recall@10"],
                "score_consistency": consistency["passed"],
            }
        ),
        flush=True,
    )
    del indices, model, reloaded, step0, s0, store, exact_panels, direct
    gc.collect()
    torch.cuda.empty_cache()
    return payload


@torch.inference_mode()
def audit_linear_merge(args: argparse.Namespace) -> dict[str, Any]:
    output = output_root(args.root) / "stageG_correctness/l_eoff/linear_merge_step178.json"
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("criterion_version") == 2:
            return existing
    plan = freeze_plan(args.root)
    checkpoint_path = Path(plan["r14_checkpoints"]["l_eoff"]["178"]["path"])
    device = torch.device(args.device)
    original = load_student(checkpoint_path, device).eval()
    merged = merged_linear_model(original).eval()
    paths = r13_paths(args.root)
    store = FeatureStore.from_path(paths["features"], cache_size=260_000)
    targets = load_target_examples(paths["dev_targets"], split="dev")
    ids_by_type = load_corpus_ids(paths["corpus"], store)
    store.preload_embeddings(
        [
            *(value for ids in ids_by_type.values() for value in ids),
            *(row.query_id for row in targets),
        ]
    )
    relation_differences = {}
    pairs = {
        ("table", "table"),
        ("table", "text"),
        ("table", "image"),
        ("text", "table"),
        ("image", "table"),
    }
    for source_type, destination_type in sorted(pairs):
        source_ids = ids_by_type[source_type][:32]
        destination_ids = ids_by_type[destination_type][:32]
        source_embeddings = torch.stack(
            [store.embedding_features(value).embedding for value in source_ids]
        ).to(device=device, dtype=torch.float32)
        destination_embeddings = torch.stack(
            [store.embedding_features(value).embedding for value in destination_ids]
        ).to(device=device, dtype=torch.float32)
        first = original.score_embedding_matrix(
            source_embeddings, source_type, destination_embeddings, destination_type
        )
        second = merged.score_embedding_matrix(
            source_embeddings, source_type, destination_embeddings, destination_type
        )
        relation_differences[f"{source_type}_to_{destination_type}"] = {
            "max_abs_difference": float((first - second).abs().max()),
            "max_abs_reference_score": float(first.abs().max()),
        }
    original_direct = full_dev_direct_exact_ann(
        original,
        None,
        store,
        targets,
        sorted(ids_by_type["table"]),
        device,
        batch_size=args.query_batch_size,
    )
    merged_direct = full_dev_direct_exact_ann(
        merged,
        None,
        store,
        targets,
        sorted(ids_by_type["table"]),
        device,
        batch_size=args.query_batch_size,
    )
    same_rankings = sum(
        left["exact_ids"] == right["exact_ids"]
        for left, right in zip(
            original_direct["per_query"], merged_direct["per_query"]
        )
    )
    overlaps = [
        len(set(left["exact_ids"]) & set(right["exact_ids"])) / 100
        for left, right in zip(
            original_direct["per_query"], merged_direct["per_query"]
        )
    ]
    aggregate_identical = original_direct["aggregate"] == merged_direct["aggregate"]
    score_tolerance_passed = all(
        row["max_abs_difference"]
        <= FP32_ATOL + FP32_RTOL * max(row["max_abs_reference_score"], 1.0)
        for row in relation_differences.values()
    )
    payload = {
        "format_version": 1,
        "criterion_version": 2,
        "status": "complete",
        "identity": "P_eff = P + BA/c_tau",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_fingerprint(checkpoint_path),
        "max_abs_score_difference_by_relation": relation_differences,
        "full_dev_exact_top100_identical_queries": same_rankings,
        "full_dev_exact_top100_mean_overlap": float(np.mean(overlaps)),
        "full_dev_exact_top100_min_overlap": min(overlaps),
        "queries": len(targets),
        "score_tolerance_passed": score_tolerance_passed,
        "aggregate_exact_metrics_identical": aggregate_identical,
        "passed": score_tolerance_passed
        and aggregate_identical
        and float(np.mean(overlaps)) >= 0.999,
        "original_exact": original_direct["aggregate"],
        "merged_exact": merged_direct["aggregate"],
        "note": (
            "No ANN index was rebuilt. Non-identical Top100 lists are retained as "
            "FP32 boundary-order evidence; pass requires bounded score error, identical "
            "aggregate exact metrics, and at least 99.9% mean Top100 overlap."
        ),
    }
    write_json(output, payload)
    return payload


def run_arm(args: argparse.Namespace) -> None:
    plan = freeze_plan(args.root)
    for step in (0, 45, 89, 178):
        checkpoint_path = Path(plan["r14_checkpoints"][args.arm][str(step)]["path"])
        audit_checkpoint(
            args,
            arm=args.arm,
            checkpoint_id=f"step_{step:06d}",
            checkpoint_path=checkpoint_path,
            reuse_index=_existing_step178_index(args.root, args.arm)
            if step == 178
            else None,
        )
    checkpoint_path = Path(plan["r14_checkpoints"][args.arm]["178"]["path"])
    trained = load_student(checkpoint_path, torch.device(args.device)).eval()
    audit_checkpoint(
        args,
        arm=args.arm,
        checkpoint_id="step_000178_adapter_off",
        checkpoint_path=checkpoint_path,
        model_override=adapter_off_model(trained),
    )
    if args.arm == "l_eoff":
        audit_linear_merge(args)


def run_i_arm(args: argparse.Namespace) -> None:
    """Audit the new evidence-off interaction checkpoints without duplicating step0."""

    arm_dir = output_root(args.root) / "stageI_interaction" / f"{args.arm}_seed13"
    output_base = arm_dir / "correctness"
    step0_path = arm_dir / "checkpoints/step_000000.pt"
    timeline: dict[str, Any] = {
        "step_000000": {
            "reused_from": str(
                (
                    output_root(args.root)
                    / "stageG_correctness"
                    / args.arm
                    / "step_000000/metrics.json"
                ).resolve()
            ),
            "identity": "R15 step0 is byte-identical to the audited R14 residual step0",
            "checkpoint_sha256": checkpoint_fingerprint(step0_path),
        }
    }
    for step in (45, 89, 178):
        checkpoint_id = f"step_{step:06d}"
        checkpoint_path = arm_dir / "checkpoints" / f"{checkpoint_id}.pt"
        checkpoint_record = json.loads(
            (arm_dir / "checkpoints" / f"{checkpoint_id}.json").read_text(
                encoding="utf-8"
            )
        )
        payload = audit_checkpoint(
            args,
            arm=args.arm,
            checkpoint_id=checkpoint_id,
            checkpoint_path=checkpoint_path,
            reuse_index=(arm_dir / "evaluation_step178/index") if step == 178 else None,
            output_base=output_base,
            step0_path=step0_path,
            optimizer_payload=checkpoint_record["optimizer"],
        )
        timeline[checkpoint_id] = {
            "path": str((output_base / checkpoint_id / "metrics.json").resolve()),
            "sha256": checkpoint_fingerprint(output_base / checkpoint_id / "metrics.json"),
            "exact_recall@10": payload["full_dev_direct"]["aggregate"]["all"][
                "exact_recall@10"
            ],
            "ann_recall@10": payload["full_dev_direct"]["aggregate"]["all"][
                "ann_recall@10"
            ],
            "ann_exact_overlap@100": payload["full_dev_direct"]["aggregate"]["all"][
                "ann_exact_overlap@100"
            ],
        }
    step0_g = json.loads(
        Path(timeline["step_000000"]["reused_from"]).read_text(encoding="utf-8")
    )
    timeline["step_000000"].update(
        {
            "source_sha256": checkpoint_fingerprint(
                Path(timeline["step_000000"]["reused_from"])
            ),
            "exact_recall@10": step0_g["full_dev_direct"]["aggregate"]["all"][
                "exact_recall@10"
            ],
            "ann_recall@10": step0_g["full_dev_direct"]["aggregate"]["all"][
                "ann_recall@10"
            ],
            "ann_exact_overlap@100": step0_g["full_dev_direct"]["aggregate"]["all"][
                "ann_exact_overlap@100"
            ],
        }
    )
    write_json(
        output_base / "timeline.json",
        {
            "format_version": 1,
            "status": "complete",
            "arm": f"{args.arm}_seed13",
            "checkpoints": timeline,
        },
    )


def finalize_gate(root: Path) -> dict[str, Any]:
    gate_path = output_root(root) / "stageG_correctness/GATE.json"
    if gate_path.is_file():
        return json.loads(gate_path.read_text(encoding="utf-8"))
    checkpoints = {}
    missing = []
    for arm in ARM_SPECS:
        checkpoints[arm] = {}
        for checkpoint_id in (
            "step_000000",
            "step_000045",
            "step_000089",
            "step_000178",
            "step_000178_adapter_off",
        ):
            path = output_root(root) / "stageG_correctness" / arm / checkpoint_id / "metrics.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            checkpoints[arm][checkpoint_id] = {
                "path": str(path.resolve()),
                "sha256": checkpoint_fingerprint(path),
                "score_path_consistency": payload["score_path_consistency"]["passed"],
                "exact_recall@10": payload["full_dev_direct"]["aggregate"]["all"]["exact_recall@10"],
                "ann_recall@10": payload["full_dev_direct"]["aggregate"]["all"]["ann_recall@10"],
                "ann_exact_overlap@100": payload["full_dev_direct"]["aggregate"]["all"]["ann_exact_overlap@100"],
            }
    merge_path = output_root(root) / "stageG_correctness/l_eoff/linear_merge_step178.json"
    if not merge_path.is_file():
        missing.append(str(merge_path))
    if missing:
        raise FileNotFoundError("Missing R15 G outputs:\n" + "\n".join(missing))
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    consistency_passed = all(
        row["score_path_consistency"]
        for by_checkpoint in checkpoints.values()
        for row in by_checkpoint.values()
    )
    if not consistency_passed or not merge["passed"]:
        status = "failed"
        decision = "Stop I: implementation/numerical consistency gate failed."
    else:
        status = "passed"
        decision = "G completed without a numerical path mismatch; I is authorized."
    classifications = {}
    for arm, rows in checkpoints.items():
        endpoint = rows["step_000178"]
        adapter_off = rows["step_000178_adapter_off"]
        exact_bad = endpoint["exact_recall@10"] < 0.05
        ann_gap = endpoint["exact_recall@10"] - endpoint["ann_recall@10"]
        classifications[arm] = {
            "endpoint": (
                "exact_and_ann_collapsed"
                if exact_bad and abs(ann_gap) < 0.01
                else "ann_specific_degradation"
                if ann_gap >= 0.01
                else "not_collapsed"
            ),
            "adapter_off_exact_delta": adapter_off["exact_recall@10"]
            - endpoint["exact_recall@10"],
        }
    gate = {
        "format_version": 1,
        "status": status,
        "decision": decision,
        "dtype_tolerance": {"dtype": "float32", "atol": FP32_ATOL, "rtol": FP32_RTOL},
        "checkpoints": checkpoints,
        "linear_merge": {
            "path": str(merge_path.resolve()),
            "sha256": checkpoint_fingerprint(merge_path),
            "passed": merge["passed"],
        },
        "classification": classifications,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(gate_path, gate)
    return gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARM_SPECS))
    parser.add_argument("--finalize-g", action="store_true")
    parser.add_argument("--stage-i", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--index-threads", type=int, default=12)
    parser.add_argument("--query-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.finalize_g:
        print(json.dumps(finalize_gate(args.root), indent=2))
        return
    if args.arm is None:
        raise ValueError("--arm is required unless --finalize-g is used")
    if args.stage_i:
        run_i_arm(args)
    else:
        run_arm(args)


if __name__ == "__main__":
    main()

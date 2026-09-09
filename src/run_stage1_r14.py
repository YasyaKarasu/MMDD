#!/usr/bin/env python
"""Run the frozen R14 Stage-1 branch and projection experiments."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.models import StudentJoinabilityModel, add_projection_residual
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.retrieval import (
    StudentANNIndices,
    build_indices,
    fuse_ranked_channels,
    load_corpus_ids,
    retrieve_zero_one_hop_detailed_many,
)
from mmdd_stage1.row_support import load_evidence_content_keys
from mmdd_stage1.scoring import score_target_batch
from mmdd_stage1.training import (
    _student_path_losses,
    _target_teacher_scores,
    checkpoint,
    student_gradient_norms,
    student_projection_drift,
    student_projection_references,
    student_relation_drift,
)
from run_stage1_r13 import (
    _DirectScorer,
    _fixed_path_metrics,
    _merge_witness_metadata,
    _paths_by_target,
    _recall_record,
    _paths as r13_paths,
    freeze_path_schedule,
    freeze_plan as freeze_r13_plan,
)
from run_stage1_r11_task_e import empty_intervention_stats
from run_stage1_r11_task_f import _accumulate, _empty, _finalize, _target_channels


CHECKPOINT_STEPS = (0, 45, 89, 178)
ARM_SPECS = {
    "m0": {"adapter": "none", "evidence_loss_weight": 1.0},
    "d_only": {"adapter": "none", "evidence_loss_weight": 0.0},
    "m_linear": {"adapter": "linear", "evidence_loss_weight": 1.0},
    "m_gelu": {"adapter": "gelu", "evidence_loss_weight": 1.0},
}


def _output_root(root: Path) -> Path:
    return root / "work/stage1_optimization_r14_20260909"


def _sha256_text(values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def freeze_plan(root: Path) -> dict[str, Any]:
    output = _output_root(root)
    path = output / "PLAN_FROZEN.json"
    plan_path = root / "stage1_optimization_r14_plan_20260909.md"
    round2_path = root / "work/gpt_pro_research_handoff_20260909/response_round2.md"
    reanalysis_path = (
        root / "work/gpt_pro_research_handoff_20260909/r13_full_reanalysis_notes.md"
    )
    expected = {
        "plan_sha256": checkpoint_fingerprint(plan_path),
        "round2_sha256": checkpoint_fingerprint(round2_path),
        "r13_reanalysis_sha256": checkpoint_fingerprint(reanalysis_path),
    }
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if payload[key] != value:
                raise ValueError(f"Frozen R14 dependency changed: {key}")
        return payload

    r13 = freeze_r13_plan(root)
    selected_path = (
        root
        / "work/stage1_optimization_r13_20260909/statistics/SELECTED_RECIPE.json"
    )
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if checkpoint_fingerprint(Path(selected["checkpoint"])) != selected[
        "checkpoint_sha256"
    ]:
        raise ValueError("R13 selected checkpoint fingerprint mismatch")
    output.mkdir(parents=True, exist_ok=True)
    for directory in (
        "stage1_A_attribution",
        "stage1_B_branch_ablation",
        "stage1_M_projection_capacity",
        "stage1_D_student_variance",
        "statistics",
    ):
        (output / directory).mkdir(exist_ok=True)
    payload = {
        "format_version": 1,
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(plan_path.resolve()),
        **expected,
        "latest_matrix_source": str(round2_path.resolve()),
        "r13_reanalysis": str(reanalysis_path.resolve()),
        "amendment": {
            "reason": (
                "Round-2 full-data analysis supersedes the plan's conditional "
                "C-G/C-RF first allocation."
            ),
            "first_batch": ["d_only", "m_linear", "m_gelu"],
            "matched_control": "R13 p_s_target_only for seed13",
            "projection_hidden_dim": 256,
            "projection_arms": {
                "m_linear": "Wz + B(Az/c)",
                "m_gelu": "Wz + B GELU(Az/c)",
            },
            "conditional_c_g": "not launched: B0 lacks the registered margin evidence",
            "conditional_c_rf": (
                "not launched: current-vs-old hard-candidate overlap evidence is absent"
            ),
        },
        "protocol": {
            **r13["protocol"],
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "student_seeds": [13, 17, 23],
            "projection_hidden_dim": 256,
            "projection_scale": (
                "RMS of raw A z on up to 4096 deterministic train-fit objects per type"
            ),
            "full_forward_for_d_only": True,
        },
        "r13_dependencies": {
            "plan": r13["plan_sha256"],
            "s0": r13["s0"],
            "schedule": r13["inputs"]["schedule"],
            "teacher_scores": r13["teacher_scores"],
            "selected_recipe": {
                "path": str(selected_path.resolve()),
                "sha256": checkpoint_fingerprint(selected_path),
                "checkpoint": selected["checkpoint"],
                "checkpoint_sha256": selected["checkpoint_sha256"],
            },
        },
    }
    write_json(path, payload)
    write_json(
        output / "TASK_DECISIONS.json",
        {
            "format_version": 1,
            "status": "frozen",
            "task_a": "reuse R13 full-data reanalysis; add weight-dependent diagnostics",
            "task_b": "run d_only seed13",
            "task_m": "run parameter-matched linear and GELU residuals seed13",
            "task_c_g": payload["amendment"]["conditional_c_g"],
            "task_c_rf": payload["amendment"]["conditional_c_rf"],
            "task_d": "decide matched seed17/23 arms after frozen seed13 endpoints",
        },
    )
    (output / "runs.jsonl").touch()
    return payload


def _schedule(root: Path, seed: int):
    manifest = freeze_path_schedule(root)
    examples = _merge_witness_metadata(root)
    if seed == 13:
        order = json.loads(Path(manifest["order"]).read_text(encoding="utf-8"))[
            "indices"
        ]
    else:
        order = list(range(len(examples)))
        random.Random(seed).shuffle(order)
    ordered = [examples[index] for index in order]
    batches = [ordered[start : start + 64] for start in range(0, len(ordered), 64)]
    if len(batches) != 178:
        raise ValueError(f"Expected 178 path batches, found {len(batches)}")
    return batches, order


def _training_ids_by_type(examples, store: FeatureStore) -> dict[str, list[str]]:
    ids = {"table": set(), "text": set(), "image": set()}
    evidence_ids = set()
    for example in examples:
        ids["table"].add(example.query_id)
        for candidate in example.candidates:
            ids["table"].add(candidate.target_id)
            evidence_ids.update(candidate.evidence_ids)
    for object_id in sorted(evidence_ids):
        ids[store.embedding_features(object_id).object_type].add(object_id)
    return {key: sorted(values) for key, values in ids.items()}


@torch.inference_mode()
def _calibrate_projection_scales(
    model: StudentJoinabilityModel,
    store: FeatureStore,
    ids_by_type: dict[str, list[str]],
    device: torch.device,
    *,
    sample_limit: int = 4096,
    batch_size: int = 256,
) -> dict[str, Any]:
    if model.projection_adapter == "none":
        return {"scales": dict(model.projection_scales), "samples": {}}
    samples = {}
    scales = {}
    for object_type, object_ids in ids_by_type.items():
        selected = object_ids[:sample_limit]
        squared_sum = 0.0
        value_count = 0
        for start in range(0, len(selected), batch_size):
            batch_ids = selected[start : start + batch_size]
            embeddings = torch.stack(
                [store.embedding_features(object_id).embedding for object_id in batch_ids]
            ).to(device=device, dtype=torch.float32)
            hidden = model.projection_residual_inputs[object_type](embeddings)
            squared_sum += float(hidden.square().sum().cpu())
            value_count += hidden.numel()
        scale = math.sqrt(squared_sum / value_count)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid projection scale for {object_type}: {scale}")
        model.projection_scales[object_type] = scale
        scales[object_type] = scale
        samples[object_type] = {
            "available": len(object_ids),
            "used": len(selected),
            "ids_sha256": _sha256_text(selected),
        }
    return {"scales": scales, "samples": samples}


@torch.inference_mode()
def _adapter_diagnostics(
    model: StudentJoinabilityModel,
    store: FeatureStore,
    ids_by_type: dict[str, list[str]],
    device: torch.device,
) -> dict[str, Any]:
    if model.projection_adapter == "none":
        return {"adapter": "none", "added_parameters": 0, "by_type": {}}
    by_type = {}
    for object_type, object_ids in ids_by_type.items():
        selected = object_ids[:256]
        embeddings = torch.stack(
            [store.embedding_features(object_id).embedding for object_id in selected]
        ).to(device=device, dtype=torch.float32)
        base = model.projections[object_type](embeddings)
        hidden = (
            model.projection_residual_inputs[object_type](embeddings)
            / model.projection_scales[object_type]
        )
        activated = torch.nn.functional.gelu(hidden)
        used = activated if model.projection_adapter == "gelu" else hidden
        residual = model.projection_residual_outputs[object_type](used)
        base_rms = float(base.square().mean().sqrt().cpu())
        residual_rms = float(residual.square().mean().sqrt().cpu())
        gelu_derivative = 0.5 * (
            1.0 + torch.erf(hidden / math.sqrt(2.0))
        ) + hidden * torch.exp(-0.5 * hidden.square()) / math.sqrt(2.0 * math.pi)
        by_type[object_type] = {
            "samples": len(selected),
            "scale": model.projection_scales[object_type],
            "hidden_rms": float(hidden.square().mean().sqrt().cpu()),
            "activated_rms": float(used.square().mean().sqrt().cpu()),
            "mean_gelu_derivative": float(gelu_derivative.mean().cpu()),
            "base_rms": base_rms,
            "residual_rms": residual_rms,
            "residual_to_base_rms": residual_rms / base_rms if base_rms else None,
            "input_weight_norm": float(
                model.projection_residual_inputs[object_type].weight.norm().cpu()
            ),
            "output_weight_norm": float(
                model.projection_residual_outputs[object_type].weight.norm().cpu()
            ),
        }
    return {
        "adapter": model.projection_adapter,
        "hidden_dim": model.projection_hidden_dim,
        "added_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("projection_residual_")
        ),
        "by_type": by_type,
    }


def _optimizer(model: StudentJoinabilityModel) -> torch.optim.AdamW:
    projection_parameters = [
        *model.projections.parameters(),
        *model.projection_residual_inputs.parameters(),
        *model.projection_residual_outputs.parameters(),
    ]
    return torch.optim.AdamW(
        [
            {"params": model.relation_parameters(), "lr": 1e-5},
            {"params": projection_parameters, "lr": 1e-6},
        ],
        weight_decay=0.01,
    )


def _arm_id(arm: str, seed: int) -> str:
    return {
        "m0": "m0_linear",
        "d_only": "b_d_e_loss_off",
        "m_linear": "m_l_linear_residual",
        "m_gelu": "m_n_gelu_residual",
    }[arm] + f"_seed{seed}"


def _arm_directory(root: Path, arm: str, seed: int) -> Path:
    task = (
        "stage1_B_branch_ablation"
        if arm == "d_only"
        else "stage1_M_projection_capacity"
        if seed == 13
        else "stage1_D_student_variance"
    )
    return _output_root(root) / task / _arm_id(arm, seed)


def _pair_counts(examples) -> dict[str, int]:
    occurrences = 0
    pairs = set()
    for example in examples:
        for candidate in example.candidates:
            occurrences += 1
            pairs.add(("direct", example.query_id, candidate.target_id))
            for evidence_id in candidate.evidence_ids:
                occurrences += 2
                pairs.add(("qe", example.query_id, evidence_id))
                pairs.add(("et", evidence_id, candidate.target_id))
    return {
        "examples": len(examples),
        "candidate_and_edge_occurrences": occurrences,
        "unique_scored_pairs": len(pairs),
    }


def _make_model(
    root: Path,
    arm: str,
    seed: int,
    store: FeatureStore,
    examples,
    device: torch.device,
) -> tuple[StudentJoinabilityModel, dict[str, Any], dict[str, list[str]]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    base = load_student(r13_paths(root)["s0"], device)
    if base.projection_mode != "shared" or base.projection_adapter != "none":
        raise ValueError("R14 requires the shared linear R13 S0")
    base.reset_projection_anchors()
    ids_by_type = _training_ids_by_type(examples, store)
    adapter = str(ARM_SPECS[arm]["adapter"])
    if adapter == "none":
        return base, {"scales": dict(base.projection_scales), "samples": {}}, ids_by_type
    model = add_projection_residual(base, adapter, hidden_dim=256)
    scale_audit = _calibrate_projection_scales(
        model, store, ids_by_type, device
    )
    with torch.inference_mode():
        sample = examples[:8]
        aggregator = PathAggregator("logsumexp", 4, path_combination="sum")
        base_scores = score_target_batch(base, sample, store, device, aggregator)
        migrated_scores = score_target_batch(model, sample, store, device, aggregator)
        direct_difference = float(
            (base_scores.direct.logits - migrated_scores.direct.logits).abs().max()
        )
        evidence_difference = float(
            (
                base_scores.evidence.logits
                - migrated_scores.evidence.logits
            ).abs().max()
        )
    scale_audit["step0_equivalence"] = {
        "queries": len(sample),
        "max_direct_abs_difference": direct_difference,
        "max_evidence_abs_difference": evidence_difference,
        "passed": direct_difference == 0.0 and evidence_difference == 0.0,
    }
    if not scale_audit["step0_equivalence"]["passed"]:
        raise RuntimeError("Projection residual migration changed step-0 scores")
    return model, scale_audit, ids_by_type


def _save_checkpoint(
    arm_dir: Path,
    step: int,
    model: StudentJoinabilityModel,
    aggregator: PathAggregator,
    fixed_batch,
    store: FeatureStore,
    ids_by_type: dict[str, list[str]],
    device: torch.device,
    gradient_norms,
) -> dict[str, Any]:
    path = arm_dir / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint(model, "student-path", aggregator), path)
    model.eval()
    scores = score_target_batch(model, fixed_batch, store, device, aggregator)
    payload = {
        "optimizer_updates": step,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": checkpoint_fingerprint(path),
        "fixed_training_batch": _fixed_path_metrics(scores, fixed_batch),
        "projection_drift_from_pca": student_projection_drift(model),
        "projection_drift_from_stage_start": student_projection_drift(
            model, reference="stage_start"
        ),
        "projection_references": student_projection_references(model),
        "projection_adapter": _adapter_diagnostics(
            model, store, ids_by_type, device
        ),
        "relation_drift": student_relation_drift(model),
        "last_gradient_norms": gradient_norms,
    }
    write_json(path.with_suffix(".json"), payload)
    return payload


def train_arm(args: argparse.Namespace) -> dict[str, Any]:
    plan = freeze_plan(args.root)
    arm_dir = _arm_directory(args.root, args.arm, args.seed)
    manifest_path = arm_dir / "manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            print(json.dumps({"status": "retained", "arm": payload["arm"]}))
            return payload
        raise FileExistsError(f"Inspect incomplete R14 arm: {arm_dir}")
    arm_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    paths = r13_paths(args.root)
    store = FeatureStore.from_path(paths["features"], cache_size=60_000)
    batches, order = _schedule(args.root, args.seed)
    examples = [example for batch in batches for example in batch]
    model, scale_audit, ids_by_type = _make_model(
        args.root, args.arm, args.seed, store, examples, device
    )
    optimizer = _optimizer(model)
    aggregator = PathAggregator("logsumexp", 4, path_combination="sum")
    evidence_loss_weight = float(ARM_SPECS[args.arm]["evidence_loss_weight"])
    started = time.monotonic()
    checkpoints = {
        0: _save_checkpoint(
            arm_dir,
            0,
            model,
            aggregator,
            batches[0],
            store,
            ids_by_type,
            device,
            None,
        )
    }
    history = []
    last_gradient_norms = None
    for step, batch in enumerate(batches, 1):
        model.train()
        student_scores = score_target_batch(
            model, batch, store, device, aggregator
        )
        teacher_scores = _target_teacher_scores(batch, device)
        objective = _student_path_losses(
            model,
            student_scores,
            teacher_scores,
            None,
            temperature=1.0,
            distillation_weight=0.3,
            anchor_weight=0.1,
            anchor_weight_evidence=0.1,
            distillation_rows=None,
            positive_loss_mode="sum_probability",
            evidence_loss_weight=evidence_loss_weight,
        )
        direct_loss = (
            objective["direct_supervised_loss"]
            + 0.3 * objective["direct_distillation_loss"]
        )
        evidence_loss = (
            objective["evidence_supervised_loss"]
            + 0.3 * objective["evidence_distillation_loss"]
        )
        optimizer.zero_grad()
        objective["loss"].backward()
        last_gradient_norms = student_gradient_norms(model)
        optimizer.step()
        history.append(
            {
                "optimizer_updates": step,
                "loss": float(objective["loss"].detach()),
                "direct_loss": float(direct_loss.detach()),
                "evidence_loss": float(evidence_loss.detach()),
                "weighted_evidence_loss": float(
                    (evidence_loss_weight * evidence_loss).detach()
                ),
                "supervised_loss": float(objective["supervised_loss"].detach()),
                "distillation_loss": float(
                    objective["distillation_loss"].detach()
                ),
                "direct_supervised_loss": float(
                    objective["direct_supervised_loss"].detach()
                ),
                "evidence_supervised_loss": float(
                    objective["evidence_supervised_loss"].detach()
                ),
                "direct_distillation_loss": float(
                    objective["direct_distillation_loss"].detach()
                ),
                "evidence_distillation_loss": float(
                    objective["evidence_distillation_loss"].detach()
                ),
                "anchor_loss": float(objective["anchor_loss"].detach()),
                "weighted_anchor_loss": float(
                    objective["weighted_anchor_loss"].detach()
                ),
            }
        )
        if step in CHECKPOINT_STEPS:
            checkpoints[step] = _save_checkpoint(
                arm_dir,
                step,
                model,
                aggregator,
                batches[0],
                store,
                ids_by_type,
                device,
                last_gradient_norms,
            )
        if step % 25 == 0 or step in CHECKPOINT_STEPS:
            print(
                json.dumps(
                    {
                        "arm": _arm_id(args.arm, args.seed),
                        "step": step,
                        "loss": history[-1]["loss"],
                        "direct_loss": history[-1]["direct_loss"],
                        "evidence_loss": history[-1]["evidence_loss"],
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    if set(checkpoints) != set(CHECKPOINT_STEPS):
        raise RuntimeError("R14 did not save every registered checkpoint")
    schedule_ids = [examples[index].query_id for index in range(len(examples))]
    payload = {
        "format_version": 1,
        "status": "complete",
        "arm": _arm_id(args.arm, args.seed),
        "arm_kind": args.arm,
        "seed": args.seed,
        "objective": (
            "L_D + anchor"
            if args.arm == "d_only"
            else "L_D + L_E + anchor"
        ),
        "unique_algorithm_change": {
            "m0": "none; matched shared-linear full-path control",
            "d_only": "evidence CE and evidence KD coefficients are zero",
            "m_linear": "zero-output 4096-256-1024 identity residual",
            "m_gelu": "zero-output 4096-256-1024 GELU residual",
        }[args.arm],
        "parent_checkpoint": plan["r13_dependencies"]["s0"],
        "plan_sha256": plan["plan_sha256"],
        "round2_sha256": plan["round2_sha256"],
        "schedule": {
            "seed": args.seed,
            "queries": len(examples),
            "batches": len(batches),
            "order_query_ids_sha256": _sha256_text(schedule_ids),
            "candidate_content": plan["r13_dependencies"]["schedule"],
        },
        "teacher_scores": plan["r13_dependencies"]["teacher_scores"],
        "projection_adapter": {
            "kind": ARM_SPECS[args.arm]["adapter"],
            "hidden_dim": 256 if ARM_SPECS[args.arm]["adapter"] != "none" else None,
            "scale_audit": scale_audit,
            "base_anchor_only": True,
        },
        "evidence_loss_weight": evidence_loss_weight,
        "full_evidence_forward_computed": True,
        "optimizer_updates": 178,
        "batch_size": 64,
        "processed": _pair_counts(examples),
        "checkpoints": checkpoints,
        "history": history,
        "cost": {
            "elapsed_seconds": time.monotonic() - started,
            "device": args.device,
            "cpu_threads": args.cpu_threads,
            "new_teacher_inference": 0,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(manifest_path, payload)
    with (_output_root(args.root) / "runs.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "task": "R14 Student training",
                    "arm": payload["arm"],
                    "status": "complete",
                    "output": str(manifest_path.resolve()),
                    "command": payload["command"],
                    "cost": payload["cost"],
                }
            )
            + "\n"
        )
    print(json.dumps({"status": "complete", "arm": payload["arm"]}, indent=2))
    return payload


def _evaluation_checkpoint(root: Path, arm: str, seed: int) -> Path:
    if arm == "m0" and seed == 13:
        return Path(
            freeze_plan(root)["r13_dependencies"]["selected_recipe"]["checkpoint"]
        )
    return _arm_directory(root, arm, seed) / "checkpoints/step_000178.pt"


@torch.inference_mode()
def evaluate_arm(args: argparse.Namespace) -> dict[str, Any]:
    plan = freeze_plan(args.root)
    arm_id = (
        "p_s_target_only_seed13_reused"
        if args.arm == "m0" and args.seed == 13
        else _arm_id(args.arm, args.seed)
    )
    checkpoint_path = _evaluation_checkpoint(args.root, args.arm, args.seed)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    arm_dir = _arm_directory(args.root, args.arm, args.seed)
    output = arm_dir / "evaluation_step178"
    metrics_path = output / "metrics.json"
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            print(json.dumps({"status": "retained", "arm": arm_id}))
            return payload
        raise FileExistsError(f"Inspect incomplete evaluation: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    paths = r13_paths(args.root)
    checkpoint_sha256 = checkpoint_fingerprint(checkpoint_path)
    model = load_student(checkpoint_path, device).eval()
    store = FeatureStore.from_path(paths["features"], cache_size=260_000)
    # Evaluation qrels come only from the frozen dev lists, never from train metadata.
    from mmdd_stage1.data import load_target_examples

    examples = load_target_examples(paths["dev_targets"], split="dev")
    ids_by_type = load_corpus_ids(paths["corpus"], store)
    preload_started = time.monotonic()
    store.preload_embeddings(
        [
            *(value for ids in ids_by_type.values() for value in ids),
            *(example.query_id for example in examples),
        ]
    )
    preload_seconds = time.monotonic() - preload_started
    index_dir = output / "index"
    started = time.monotonic()
    index_started = time.monotonic()
    corpus_sha256 = freeze_r13_plan(args.root)["inputs"]["corpus"]["sha256"]
    index_manifest = build_indices(
        model,
        store,
        ids_by_type,
        index_dir,
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=corpus_sha256,
        batch_size=4096,
        num_threads=args.index_threads,
    )
    index_seconds = time.monotonic() - index_started
    indices = StudentANNIndices(
        model,
        store,
        index_dir,
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=corpus_sha256,
        score_space="raw_logit",
    )
    content_keys, content_keys_sha256 = load_evidence_content_keys(
        paths["evidence_content_keys"]
    )
    scorer = _DirectScorer(model, store, device)
    intervention_stats = empty_intervention_stats()
    f1_accumulator = _empty()
    rrf_accumulator = _empty()
    pure_accumulator = _empty()
    ranking_path = output / "rankings.jsonl.gz"
    path_pool_path = output / "path_pool.jsonl.gz"
    processing_latencies = []
    search_vectors_per_query_max = 0
    retrieval_started = time.monotonic()
    with gzip.open(ranking_path, "wt", encoding="utf-8") as ranking_handle, gzip.open(
        path_pool_path, "wt", encoding="utf-8"
    ) as pool_handle:
        for start in range(0, len(examples), args.query_batch_size):
            batch = examples[start : start + args.query_batch_size]
            batch_started = time.monotonic()
            detailed = retrieve_zero_one_hop_detailed_many(
                [example.query_id for example in batch],
                indices,
                k=50,
                direct_k=100,
                evidence_k=20,
                targets_per_evidence=20,
                evidence_types=("text", "image"),
                evidence_aggregation="logsumexp",
                evidence_top_k=4,
                evidence_temperature=1.0,
                path_combination="sum",
                fusion_mode="rrf",
                query_batch_size=args.query_batch_size,
            )
            batch_retrieval_seconds = time.monotonic() - batch_started
            for example, retrieved in zip(batch, detailed):
                record = {
                    "query_id": example.query_id,
                    "dataset": example.dataset,
                    "split": example.split,
                    "positive_target_ids": list(example.positive_target_ids),
                    "positive_evidence_by_target": {
                        key: list(value)
                        for key, value in (
                            example.positive_evidence_by_target or {}
                        ).items()
                    },
                    "positive_evidence_rows_by_target": {
                        target_id: {
                            evidence_id: list(rows)
                            for evidence_id, rows in evidence_rows.items()
                        }
                        for target_id, evidence_rows in (
                            example.positive_evidence_rows_by_target or {}
                        ).items()
                    },
                    "query_row_count": example.query_row_count,
                    "query_kind": example.query_kind,
                    "paths_by_target": _paths_by_target(retrieved),
                }
                processing_started = time.monotonic()
                direct, evidence = _target_channels(
                    record,
                    retention="e2_row_coverage",
                    scorer=scorer,
                    store=store,
                    content_keys=content_keys,
                    top_l=20,
                    evidence_budget=4,
                    pair_batch_size=256,
                    intervention="original_mixed",
                    intervention_stats=intervention_stats,
                )
                f1 = direct
                rrf = fuse_ranked_channels(
                    direct, evidence, rrf_k=60, fusion_mode="rrf"
                )
                pure = [row for row in direct if row["original_direct_member"]]
                rankings = {k: f1 for k in (10, 20, 50)}
                rrf_rankings = {k: rrf for k in (10, 20, 50)}
                pure_rankings = {k: pure for k in (10, 20, 50)}
                own_d100 = {
                    str(row["target_id"])
                    for row in direct
                    if row["original_direct_member"]
                }
                _accumulate(f1_accumulator, record, rankings, rankings, own_d100)
                _accumulate(
                    rrf_accumulator, record, rrf_rankings, rankings, own_d100
                )
                _accumulate(
                    pure_accumulator, record, pure_rankings, rankings, own_d100
                )
                recall_record = _recall_record(record, f1, rrf, pure)
                search_vectors_per_query_max = max(
                    search_vectors_per_query_max,
                    int(recall_record["search_vectors"]),
                )
                ranking_handle.write(json.dumps(recall_record) + "\n")
                pool_handle.write(json.dumps(record) + "\n")
                processing_latencies.append(
                    batch_retrieval_seconds / len(batch)
                    + time.monotonic()
                    - processing_started
                )
    retrieval_seconds = time.monotonic() - retrieval_started
    f1_metrics = _finalize(f1_accumulator)
    rrf_metrics = _finalize(rrf_accumulator)
    pure_metrics = _finalize(pure_accumulator)
    latency_values = sorted(processing_latencies)
    index_bytes = sum(
        value.stat().st_size for value in index_dir.iterdir() if value.is_file()
    )
    payload = {
        "format_version": 1,
        "status": "complete",
        "arm": arm_id,
        "arm_kind": args.arm,
        "seed": args.seed,
        "step": 178,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "projection_adapter": model.projection_adapter,
        "evaluation_split": "dev",
        "target_lists": {
            "path": str(paths["dev_targets"].resolve()),
            "sha256": checkpoint_fingerprint(paths["dev_targets"]),
        },
        "plan_sha256": plan["plan_sha256"],
        "protocol": plan["protocol"],
        "primary": f1_metrics,
        "sensitivity_equal_union_rrf": rrf_metrics,
        "pure_direct100": pure_metrics,
        "candidate_recall@50": f1_metrics["recall@50"],
        "rankings": {
            "path": str(ranking_path.resolve()),
            "sha256": checkpoint_fingerprint(ranking_path),
        },
        "path_pool": {
            "path": str(path_pool_path.resolve()),
            "sha256": checkpoint_fingerprint(path_pool_path),
        },
        "index_manifest": index_manifest,
        "content_keys_sha256": content_keys_sha256,
        "cost": {
            "feature_preload_seconds": preload_seconds,
            "index_build_seconds": index_seconds,
            "index_bytes": index_bytes,
            "retrieval_and_ranking_seconds": retrieval_seconds,
            "elapsed_seconds_excluding_preload": time.monotonic() - started,
            "direct_supplement_pairs": scorer.scored_pairs,
            "queries": len(examples),
            "search_vectors_per_query_max": search_vectors_per_query_max,
            "online_amortized_batch_size": args.query_batch_size,
            "online_seconds_p50": statistics.median(latency_values),
            "online_seconds_p95": latency_values[
                min(len(latency_values) - 1, int(0.95 * len(latency_values)))
            ],
            "device": args.device,
            "cpu_threads": args.cpu_threads,
            "index_threads": args.index_threads,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(metrics_path, payload)
    with (_output_root(args.root) / "runs.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "task": "R14 natural Stage-1 evaluation",
                    "arm": arm_id,
                    "status": "complete",
                    "output": str(metrics_path.resolve()),
                    "command": payload["command"],
                    "cost": payload["cost"],
                }
            )
            + "\n"
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "arm": arm_id,
                "R@10": f1_metrics["recall@10"],
                "R@20": f1_metrics["recall@20"],
                "R@50": f1_metrics["recall@50"],
                "RRF_R@10": rrf_metrics["recall@10"],
                "pure_direct_R@10": pure_metrics["recall@10"],
            },
            indent=2,
        )
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--arm", choices=("freeze", *ARM_SPECS), required=True
    )
    parser.add_argument("--seed", type=int, choices=(13, 17, 23), default=13)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--index-threads", type=int, default=12)
    parser.add_argument("--query-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    if args.arm == "freeze":
        print(json.dumps(freeze_plan(args.root), indent=2))
    elif args.evaluate:
        evaluate_arm(args)
    else:
        train_arm(args)


if __name__ == "__main__":
    main()

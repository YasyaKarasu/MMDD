#!/usr/bin/env python
"""Run the frozen R15 residual/evidence interaction experiments."""

from __future__ import annotations

import argparse
import gzip
import json
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
    student_gradient_norms,
)
from run_stage1_r11_task_e import empty_intervention_stats
from run_stage1_r11_task_f import _accumulate, _empty, _finalize, _target_channels
from run_stage1_r13 import (
    _DirectScorer,
    _paths as r13_paths,
    _paths_by_target,
    _recall_record,
)
from run_stage1_r14 import (
    CHECKPOINT_STEPS,
    _optimizer,
    _pair_counts,
    _save_checkpoint,
    _schedule,
    _training_ids_by_type,
    freeze_plan as freeze_r14_plan,
)


ARM_SPECS = {
    "l_eoff": {
        "adapter": "linear",
        "r14_arm_id": "m_l_linear_residual_seed13",
    },
    "n_eoff": {
        "adapter": "gelu",
        "r14_arm_id": "m_n_gelu_residual_seed13",
    },
}


def output_root(root: Path) -> Path:
    return root / "work/stage1_optimization_r15_20260909"


def _r14_arm_dir(root: Path, arm: str) -> Path:
    return (
        root
        / "work/stage1_optimization_r14_20260909/stage1_M_projection_capacity"
        / ARM_SPECS[arm]["r14_arm_id"]
    )


def arm_directory(root: Path, arm: str) -> Path:
    return output_root(root) / "stageI_interaction" / f"{arm}_seed13"


def _dependency(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": checkpoint_fingerprint(path)}


def freeze_plan(root: Path) -> dict[str, Any]:
    """Freeze the R15 documents and exact R14 checkpoints used by R15."""

    output = output_root(root)
    frozen_path = output / "PLAN_FROZEN.json"
    documents = {
        "plan": _dependency(root / "stage1_optimization_r15_plan_20260909.md"),
        "independent_review": _dependency(
            root
            / "work/gpt_pro_research_handoff_20260909/MMDD_R14_independent_review.md"
        ),
        "response_round3": _dependency(
            root / "work/gpt_pro_research_handoff_20260909/response_round3.md"
        ),
    }
    r14 = freeze_r14_plan(root)
    checkpoints = {}
    for arm in ARM_SPECS:
        directory = _r14_arm_dir(root, arm) / "checkpoints"
        checkpoints[arm] = {
            str(step): _dependency(directory / f"step_{step:06d}.pt")
            for step in CHECKPOINT_STEPS
        }
    expected = {
        "documents": documents,
        "r14_plan_sha256": r14["plan_sha256"],
        "r14_checkpoints": checkpoints,
    }
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if frozen.get(key) != value:
                raise ValueError(f"Frozen R15 dependency changed: {key}")
        return frozen

    for directory in (
        "stageG_correctness",
        "stageI_interaction",
        "stageC_candidate_admission",
        "stageV_mechanism_validation",
        "statistics",
        "source_snapshot",
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)
    frozen = {
        "format_version": 1,
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        **expected,
        "protocol": {
            "gate_order": ["G", "I", "C", "V"],
            "student_seed": 13,
            "updates_per_new_arm": 178,
            "new_teacher_inference": 0,
            "primary_metric": "full_dev_macro_target_recall@10",
            "delivery_n": 50,
            "final_k": 10,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "evidence_loss_weight": 0.0,
            "full_evidence_forward": True,
            "optimizer": {
                "kind": "fresh AdamW",
                "relation_lr": 1e-5,
                "projection_lr": 1e-6,
                "weight_decay": 0.01,
                "gradient_clipping": None,
            },
        },
        "r14_inputs": {
            "s0": r14["r13_dependencies"]["s0"],
            "schedule": r14["r13_dependencies"]["schedule"],
            "teacher_scores": r14["r13_dependencies"]["teacher_scores"],
            "selected_recipe": r14["r13_dependencies"]["selected_recipe"],
        },
    }
    write_json(frozen_path, frozen)
    (output / "runs.jsonl").touch()
    return frozen


def optimizer_audit(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Export actual parameter groups and bounded Adam state diagnostics."""

    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = list(group["params"])
        state_steps = []
        first_moment_squared = 0.0
        second_moment_sum = 0.0
        for parameter in parameters:
            state = optimizer.state.get(parameter, {})
            step = state.get("step")
            if step is not None:
                state_steps.append(int(step.item() if torch.is_tensor(step) else step))
            if "exp_avg" in state:
                first_moment_squared += float(
                    state["exp_avg"].double().square().sum()
                )
            if "exp_avg_sq" in state:
                second_moment_sum += float(state["exp_avg_sq"].double().sum())
        groups.append(
            {
                "index": index,
                "parameter_names": [names[id(parameter)] for parameter in parameters],
                "parameter_count": sum(parameter.numel() for parameter in parameters),
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "betas": [float(value) for value in group["betas"]],
                "eps": float(group["eps"]),
                "state_entries": sum(
                    bool(optimizer.state.get(parameter)) for parameter in parameters
                ),
                "state_step_min": min(state_steps) if state_steps else None,
                "state_step_max": max(state_steps) if state_steps else None,
                "exp_avg_l2": first_moment_squared**0.5,
                "exp_avg_sq_sum": second_moment_sum,
            }
        )
    return {
        "kind": type(optimizer).__name__,
        "groups": groups,
        "gradient_clipping": None,
        "parameter_dtypes": sorted(
            {str(parameter.dtype) for parameter in model.parameters()}
        ),
    }


def _save_r15_checkpoint(
    arm_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    aggregator: PathAggregator,
    fixed_batch: list[Any],
    store: FeatureStore,
    ids_by_type: dict[str, list[str]],
    device: torch.device,
    gradient_norms: dict[str, float] | None,
) -> dict[str, Any]:
    payload = _save_checkpoint(
        arm_dir,
        step,
        model,
        aggregator,
        fixed_batch,
        store,
        ids_by_type,
        device,
        gradient_norms,
    )
    payload["optimizer"] = optimizer_audit(optimizer, model)
    write_json(arm_dir / "checkpoints" / f"step_{step:06d}.json", payload)
    return payload


def train_arm(args: argparse.Namespace) -> dict[str, Any]:
    plan = freeze_plan(args.root)
    gate_path = output_root(args.root) / "stageG_correctness/GATE.json"
    if not gate_path.is_file():
        raise FileNotFoundError("R15 G gate has not been completed")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "passed":
        raise RuntimeError("R15 G gate did not pass; I arms must not start")

    arm_dir = arm_directory(args.root, args.arm)
    manifest_path = arm_dir / "manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            print(json.dumps({"status": "retained", "arm": payload["arm"]}))
            return payload
        raise FileExistsError(f"Inspect incomplete R15 arm: {arm_dir}")
    arm_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    paths = r13_paths(args.root)
    store = FeatureStore.from_path(paths["features"], cache_size=60_000)
    batches, _order = _schedule(args.root, 13)
    examples = [example for batch in batches for example in batch]
    initial = Path(plan["r14_checkpoints"][args.arm]["0"]["path"])
    model = load_student(initial, device)
    expected_adapter = ARM_SPECS[args.arm]["adapter"]
    if model.projection_adapter != expected_adapter:
        raise ValueError(
            f"{args.arm} requires {expected_adapter}, found {model.projection_adapter}"
        )
    ids_by_type = _training_ids_by_type(examples, store)
    optimizer = _optimizer(model)
    aggregator = PathAggregator("logsumexp", 4, path_combination="sum")
    started = time.monotonic()
    checkpoints = {
        0: _save_r15_checkpoint(
            arm_dir,
            0,
            model,
            optimizer,
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
        student_scores = score_target_batch(model, batch, store, device, aggregator)
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
            evidence_loss_weight=0.0,
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
                "evidence_loss_unweighted": float(evidence_loss.detach()),
                "weighted_evidence_loss": 0.0,
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
            checkpoints[step] = _save_r15_checkpoint(
                arm_dir,
                step,
                model,
                optimizer,
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
                        "arm": f"{args.arm}_seed13",
                        "step": step,
                        "loss": history[-1]["loss"],
                        "direct_loss": history[-1]["direct_loss"],
                        "evidence_loss_unweighted": history[-1][
                            "evidence_loss_unweighted"
                        ],
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    if set(checkpoints) != set(CHECKPOINT_STEPS):
        raise RuntimeError("R15 did not save every registered checkpoint")

    payload = {
        "format_version": 1,
        "status": "complete",
        "arm": f"{args.arm}_seed13",
        "adapter": expected_adapter,
        "seed": 13,
        "objective": "L_D + anchor; E-channel CE/KD coefficient exactly zero",
        "unique_change_from_r14_full": "evidence_loss_weight: 1.0 -> 0.0",
        "initial_checkpoint": _dependency(initial),
        "plan_sha256": plan["documents"]["plan"]["sha256"],
        "g_gate_sha256": checkpoint_fingerprint(gate_path),
        "schedule": {
            "queries": len(examples),
            "batches": len(batches),
            "candidate_content": plan["r14_inputs"]["schedule"],
        },
        "teacher_scores": plan["r14_inputs"]["teacher_scores"],
        "projection_scales": dict(model.projection_scales),
        "base_anchor_only": True,
        "evidence_loss_weight": 0.0,
        "full_evidence_forward_computed": True,
        "optimizer_updates": 178,
        "batch_size": 64,
        "processed": _pair_counts(examples),
        "checkpoints": checkpoints,
        "history": history,
        "optimizer_final": optimizer_audit(optimizer, model),
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
    with (output_root(args.root) / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "task": "R15 I interaction training",
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


@torch.inference_mode()
def evaluate_arm(args: argparse.Namespace) -> dict[str, Any]:
    """Run the unchanged R14 natural retrieval and three fixed admission rules."""

    plan = freeze_plan(args.root)
    checkpoint_path = arm_directory(args.root, args.arm) / "checkpoints/step_000178.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output = arm_directory(args.root, args.arm) / "evaluation_step178"
    metrics_path = output / "metrics.json"
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            print(json.dumps({"status": "retained", "arm": payload["arm"]}))
            return payload
        raise FileExistsError(f"Inspect incomplete evaluation: {output}")
    output.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    paths = r13_paths(args.root)
    checkpoint_sha256 = checkpoint_fingerprint(checkpoint_path)
    model = load_student(checkpoint_path, device).eval()
    store = FeatureStore.from_path(paths["features"], cache_size=260_000)
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
    if index_dir.exists():
        raise FileExistsError(f"Inspect incomplete R15 evaluation index: {index_dir}")
    started = time.monotonic()
    index_started = time.monotonic()
    corpus_sha256 = checkpoint_fingerprint(paths["corpus"])
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
                _accumulate(rrf_accumulator, record, rrf_rankings, rankings, own_d100)
                _accumulate(pure_accumulator, record, pure_rankings, rankings, own_d100)
                recall_record = _recall_record(record, f1, rrf, pure)
                search_vectors_per_query_max = max(
                    search_vectors_per_query_max, int(recall_record["search_vectors"])
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
        "arm": f"{args.arm}_seed13",
        "seed": 13,
        "step": 178,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "projection_adapter": model.projection_adapter,
        "evidence_loss_weight": 0.0,
        "evaluation_split": "dev",
        "target_lists": _dependency(paths["dev_targets"]),
        "plan_sha256": plan["documents"]["plan"]["sha256"],
        "protocol": plan["protocol"],
        "primary": f1_metrics,
        "sensitivity_equal_union_rrf": rrf_metrics,
        "pure_direct100": pure_metrics,
        "candidate_recall@50": f1_metrics["recall@50"],
        "rankings": _dependency(ranking_path),
        "path_pool": _dependency(path_pool_path),
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
    with (output_root(args.root) / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "task": "R15 I natural evaluation",
                    "arm": payload["arm"],
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
                "arm": payload["arm"],
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
    parser.add_argument("--task", choices=("freeze", "train", "evaluate"), required=True)
    parser.add_argument("--arm", choices=tuple(ARM_SPECS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--index-threads", type=int, default=12)
    parser.add_argument("--query-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "freeze":
        print(json.dumps(freeze_plan(args.root), indent=2))
        return
    if args.arm is None:
        raise ValueError("--arm is required for --task train")
    if args.task == "train":
        train_arm(args)
    else:
        evaluate_arm(args)


if __name__ == "__main__":
    main()

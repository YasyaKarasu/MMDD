#!/usr/bin/env python
"""Audit and train the three pre-registered R12 edge schedules."""

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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.data import EdgeExample, load_edge_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.models import StudentJoinabilityModel
from mmdd_stage1.scoring import ListScores, edge_positive_key, global_edge_positive_ids, score_edge_batch
from mmdd_stage1.training import (
    _edge_list_metrics,
    _student_edge_losses,
    checkpoint,
    student_gradient_norms,
    student_projection_drift,
    student_projection_references,
    student_relation_drift,
)


CHECKPOINT_STEPS = (0, 45, 89, 178, 267, 356)
RELATIONS = (
    "table_to_table",
    "table_to_text",
    "table_to_image",
    "text_to_table",
    "image_to_table",
)


def _score_payload(output_root: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    root = output_root / "taskC_training/teacher_pair_scores"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = Path(manifest["scores"])
    if (
        manifest.get("status") != "complete"
        or checkpoint_fingerprint(path) != manifest.get("scores_sha256")
    ):
        raise ValueError("R12 Teacher score manifest is incomplete or stale")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    scores = payload.get("scores") if isinstance(payload, dict) else None
    if (
        not isinstance(scores, torch.Tensor)
        or scores.ndim != 1
        or scores.shape[0] != int(manifest["pair_count"])
        or not bool(torch.isfinite(scores).all())
    ):
        raise ValueError("R12 Teacher score tensor is invalid")
    return scores.float(), manifest


def _schedule_path(output_root: Path, schedule: str) -> Path:
    return (
        output_root
        / "taskC_training/candidates_seed13_steps356"
        / f"{schedule}.jsonl.gz"
    )


def _examples(
    rows: list[dict[str, Any]],
    teacher_scores: torch.Tensor,
    teacher_sha256: str,
) -> list[EdgeExample]:
    result = []
    for row in rows:
        candidate_ids = tuple(str(value) for value in row["candidate_ids"])
        pair_ids = torch.tensor(row["candidate_pair_ids"], dtype=torch.long)
        if pair_ids.numel() != len(candidate_ids):
            raise ValueError("Teacher pair IDs must align with candidate IDs")
        if bool((pair_ids < 0).any()) or bool((pair_ids >= len(teacher_scores)).any()):
            raise ValueError("Teacher pair ID is outside the frozen score tensor")
        positive_id = str(row["positive_id"])
        result.append(
            EdgeExample(
                query_id=str(row["query_id"]),
                candidate_ids=candidate_ids,
                positive_index=candidate_ids.index(positive_id),
                dataset=str(row.get("dataset", "default")),
                split="train",
                teacher_logits=tuple(float(value) for value in teacher_scores[pair_ids]),
                teacher_checkpoint_sha256=teacher_sha256,
                teacher_logit_mode="teacher",
                source_type=str(row["source_type"]),
                destination_type=str(row["destination_type"]),
                positive_ids=tuple(str(value) for value in row["positive_ids"]),
                confirmed_labels=tuple(
                    None if value is None else int(value)
                    for value in row["confirmed_labels"]
                ),
            )
        )
    return result


def _schedule_batches(
    path: Path,
    teacher_scores: torch.Tensor,
    teacher_sha256: str,
) -> Iterable[tuple[int, list[EdgeExample]]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            yield int(payload["step"]), _examples(
                payload["examples"], teacher_scores, teacher_sha256
            )


def _training_batch_order(batch_count: int, seed: int) -> list[int]:
    order = list(range(batch_count))
    if seed != 13:
        random.Random(f"r12-c1-batch-order:{seed}").shuffle(order)
    return order


def _initialize_student(root: Path, device: torch.device) -> StudentJoinabilityModel:
    pca_path = root / "work/stage1_optimization_r10_20260907/baselines/pca_entitables_v9_1024.pt"
    pca = torch.load(pca_path, map_location="cpu", weights_only=True)
    basis = pca["projection"].float()
    model = StudentJoinabilityModel(
        int(pca["input_dim"]),
        int(pca["student_dim"]),
        initialization="pca",
        initialization_basis=basis,
        freeze_projections=False,
        relation_param="full",
        relation_rank=16,
        confidence_transform=False,
    )
    historical_path = (
        root
        / "work/stage1_optimization_r11_20260908/taskA_protocol/baselines/pca_init.pt"
    )
    historical = torch.load(historical_path, map_location="cpu", weights_only=True)
    state = historical["state_dict"]
    expected_keys = [
        key
        for key in model.state_dict()
        if key.startswith(("projections.", "relations."))
    ]
    if any(key not in state for key in expected_keys):
        raise ValueError("Historical PCA checkpoint is missing model parameters")
    model.load_state_dict(
        {key: value for key, value in state.items() if key in model.state_dict()},
        strict=False,
    )
    with torch.no_grad():
        anchors = torch.stack(
            [model.projections[kind].weight for kind in model.projections]
        )
        model.initial_projection_weights.copy_(anchors)
        model.stage_initial_projection_weights.copy_(anchors)
    model.projection_reference_origin = "r12_fresh_pca_reconstruction"
    return model.to(device)


def _optimizer(model: StudentJoinabilityModel) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {"params": model.relation_parameters(), "lr": 1e-5},
            {"params": model.projections.parameters(), "lr": 1e-6},
        ],
        weight_decay=0.01,
    )


def _teacher_list_scores(
    examples: list[EdgeExample], template: ListScores, device: torch.device
) -> ListScores:
    logits = pad_sequence(
        [torch.tensor(row.teacher_logits, device=device) for row in examples],
        batch_first=True,
    )
    return ListScores(
        logits,
        template.candidate_mask,
        template.positive_indices,
        template.positive_mask,
        template.candidate_ids,
    )


def _ranking_scores(raw: ListScores) -> ListScores:
    return ListScores(
        torch.sigmoid(raw.logits) / 0.1,
        raw.candidate_mask,
        raw.positive_indices,
        raw.positive_mask,
        raw.candidate_ids,
    )


def _relation(example: EdgeExample) -> str:
    return StudentJoinabilityModel.relation_key(
        str(example.source_type), str(example.destination_type)
    )


def _accumulate_rankings(
    totals: dict[str, dict[str, list[float]]],
    examples: list[EdgeExample],
    name: str,
    logits: torch.Tensor,
) -> None:
    for row_index, example in enumerate(examples):
        values = logits[row_index, : len(example.candidate_ids)]
        positives = torch.tensor(
            [candidate_id in set(example.positive_ids) for candidate_id in example.candidate_ids],
            device=values.device,
        )
        positive_values = values[positives]
        negative_values = values[~positives]
        relation = _relation(example)
        result = totals[name][relation]
        result[0] += float(values.argmax().item() in positives.nonzero().flatten().tolist())
        result[1] += 1
        result[2] += float(positive_values.max() - negative_values.max())


def _raw_scores(
    examples: list[EdgeExample],
    store: FeatureStore,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for example in examples:
        query = store.embedding_features(example.query_id).embedding.to(device)
        candidates = torch.stack(
            [store.embedding_features(value).embedding for value in example.candidate_ids]
        ).to(device)
        rows.append(torch.mv(candidates, query))
    return pad_sequence(rows, batch_first=True)


@torch.inference_mode()
def audit_candidate_quality(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    scores, teacher_manifest = _score_payload(args.output_root)
    store = FeatureStore.from_path(
        args.root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b",
        cache_size=60_000,
    )
    device = torch.device(args.device)
    pca = _initialize_student(args.root, device).eval()
    output_dir = args.output_root / "taskC_training/candidate_quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for schedule in ("base", "candidates"):
        totals = {
            name: defaultdict(lambda: [0.0, 0.0, 0.0])
            for name in ("teacher", "raw", "pca")
        }
        lists = 0
        candidates = 0
        schedule_path = _schedule_path(args.output_root, schedule)
        for _step, batch in _schedule_batches(
            schedule_path, scores, str(teacher_manifest["teacher_checkpoint_sha256"])
        ):
            pca_list_scores = score_edge_batch(pca, batch, store, device)
            pca_scores = pca_list_scores.logits
            teacher = _teacher_list_scores(batch, pca_list_scores, device).logits
            raw = _raw_scores(batch, store, device)
            _accumulate_rankings(totals, batch, "teacher", teacher)
            _accumulate_rankings(totals, batch, "raw", raw)
            _accumulate_rankings(totals, batch, "pca", pca_scores)
            lists += len(batch)
            candidates += sum(len(row.candidate_ids) for row in batch)
        systems = {}
        for name, by_relation in totals.items():
            relation_rows = {
                relation: {
                    "hits@1": int(values[0]),
                    "lists": int(values[1]),
                    "recall@1": values[0] / values[1],
                    "mean_positive_margin": values[2] / values[1],
                }
                for relation, values in sorted(by_relation.items())
            }
            systems[name] = {
                "macro_recall@1": statistics.fmean(
                    row["recall@1"] for row in relation_rows.values()
                ),
                "micro_recall@1": sum(row["hits@1"] for row in relation_rows.values())
                / sum(row["lists"] for row in relation_rows.values()),
                "by_relation": relation_rows,
            }
        payload = {
            "format_version": 1,
            "schedule": schedule,
            "schedule_path": str(schedule_path.resolve()),
            "schedule_sha256": checkpoint_fingerprint(schedule_path),
            "lists": lists,
            "candidate_pair_occurrences": candidates,
            "systems": systems,
            "teacher_checkpoint_sha256": teacher_manifest["teacher_checkpoint_sha256"],
            "teacher_scores_sha256": teacher_manifest["scores_sha256"],
        }
        write_json(output_dir / f"{schedule}.json", payload)
        results[schedule] = payload
    summary = {
        "format_version": 1,
        "status": "pass",
        "results": results,
        "teacher_has_macro_advantage_over_pca": all(
            row["systems"]["teacher"]["macro_recall@1"]
            > row["systems"]["pca"]["macro_recall@1"]
            for row in results.values()
        ),
        "elapsed_seconds": time.monotonic() - started,
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def _hash_rank(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def _freeze_function_reference(
    args: argparse.Namespace,
    store: FeatureStore,
    model: StudentJoinabilityModel,
    device: torch.device,
) -> dict[str, Any]:
    path = args.output_root / "taskC_training/function_reference.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    supervision = (
        args.output_root / "taskA_correctness/supervision/edge_lists.train_fit.jsonl"
    )
    examples = load_edge_examples(supervision, split="train")
    rows = _sample_function_pairs(examples, store)
    model.eval()
    reference_scores = []
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            batch = rows[start : start + 256]
            sources = [
                store.embedding_features(str(row["source_id"])).for_scoring(
                    device, include_hidden=False
                )
                for row in batch
            ]
            destinations = [
                store.embedding_features(str(row["destination_id"])).for_scoring(
                    device, include_hidden=False
                )
                for row in batch
            ]
            reference_scores.extend(float(value) for value in model.score_pairs(sources, destinations))
    for row, score in zip(rows, reference_scores):
        row["pca_score"] = score
    sigma = {
        relation: max(
            statistics.pstdev(
                row["pca_score"] for row in rows
                if f"{row['source_type']}_to_{row['destination_type']}" == relation
            ),
            1e-6,
        )
        for relation in RELATIONS
    }
    payload = {
        "format_version": 1,
        "seed": 13,
        "pairs_per_relation": 256,
        "positive_neighbors_per_relation": 128,
        "random_pairs_per_relation": 128,
        "rotation_batch_per_relation": 16,
        "rotation_pairs_each_kind_per_relation": 8,
        "relations": list(RELATIONS),
        "sigma": sigma,
        "pairs": rows,
        "supervision_sha256": checkpoint_fingerprint(supervision),
        "feature_manifest_sha256": checkpoint_fingerprint(
            args.root
            / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b/manifest.jsonl"
        ),
        "random_pair_pool": "transductive unlabeled feature lake, stratified by object type",
        "model_reference": student_projection_references(model),
    }
    write_json(path, payload)
    return payload


def _sample_function_pairs(
    examples: list[EdgeExample],
    store: FeatureStore,
    *,
    pairs_per_kind: int = 128,
    seed: int = 13,
) -> list[dict[str, Any]]:
    known = global_edge_positive_ids(examples)
    positive_rows: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        relation: {} for relation in RELATIONS
    }
    for example in examples:
        relation = _relation(example)
        if relation not in positive_rows:
            continue
        for destination_id in known[edge_positive_key(example)]:
            key = (example.query_id, destination_id)
            positive_rows[relation][key] = {
                "source_id": example.query_id,
                "destination_id": destination_id,
                "source_type": example.source_type,
                "destination_type": example.destination_type,
                "known_positive": True,
                "reference_kind": "positive_neighbor",
            }
    object_ids: dict[str, list[str]] = defaultdict(list)
    for object_id in store.object_ids():
        object_ids[store.object_type(object_id)].append(object_id)
    rows = []
    for relation in RELATIONS:
        source_type, destination_type = relation.split("_to_")
        positives = sorted(
            positive_rows[relation].values(),
            key=lambda row: _hash_rank(
                f"{seed}:{relation}:positive:{row['source_id']}:{row['destination_id']}"
            ),
        )[:pairs_per_kind]
        if len(positives) != pairs_per_kind:
            raise ValueError(f"Function reference lacks positive {relation} pairs")
        generator = random.Random(f"{seed}:{relation}:random_objects")
        negatives = []
        selected = set()
        while len(negatives) < pairs_per_kind:
            source_id = generator.choice(object_ids[source_type])
            destination_id = generator.choice(object_ids[destination_type])
            key = (source_id, destination_id)
            if key in selected or source_id == destination_id:
                continue
            if destination_id in known.get(
                (source_id, source_type, destination_type), set()
            ):
                continue
            selected.add(key)
            negatives.append(
                {
                    "source_id": source_id,
                    "destination_id": destination_id,
                    "source_type": source_type,
                    "destination_type": destination_type,
                    "known_positive": False,
                    "reference_kind": "random_object",
                }
            )
        for positive, negative in zip(positives, negatives):
            rows.extend((positive, negative))
    return rows


def _function_loss(
    model: StudentJoinabilityModel,
    reference: dict[str, Any],
    store: FeatureStore,
    device: torch.device,
    step: int,
) -> torch.Tensor:
    losses = []
    for relation in RELATIONS:
        rows = [
            row
            for row in reference["pairs"]
            if f"{row['source_type']}_to_{row['destination_type']}" == relation
        ]
        start = ((step - 1) * int(reference["rotation_batch_per_relation"])) % len(rows)
        count = int(reference["rotation_batch_per_relation"])
        batch = [rows[(start + offset) % len(rows)] for offset in range(count)]
        sources = [
            store.embedding_features(str(row["source_id"])).for_scoring(
                device, include_hidden=False
            )
            for row in batch
        ]
        destinations = [
            store.embedding_features(str(row["destination_id"])).for_scoring(
                device, include_hidden=False
            )
            for row in batch
        ]
        current = model.score_pairs(sources, destinations)
        target = torch.tensor(
            [float(row["pca_score"]) for row in batch], device=device
        )
        losses.append(
            ((current - target) / float(reference["sigma"][relation])).square().mean()
        )
    return torch.stack(losses).mean()


@torch.inference_mode()
def _fixed_batch_metrics(
    model: StudentJoinabilityModel,
    examples: list[EdgeExample],
    store: FeatureStore,
    device: torch.device,
) -> dict[str, Any]:
    raw = score_edge_batch(model, examples, store, device).logits
    totals = {"student": defaultdict(lambda: [0.0, 0.0, 0.0])}
    _accumulate_rankings(totals, examples, "student", raw)
    return {
        relation: {
            "hits@1": int(values[0]),
            "lists": int(values[1]),
            "recall@1": values[0] / values[1],
            "mean_positive_margin": values[2] / values[1],
        }
        for relation, values in sorted(totals["student"].items())
    }


def _gradient_group_norms(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    gradients: tuple[torch.Tensor | None, ...],
) -> dict[str, float]:
    sums = {
        "projection": next(iter(named_parameters))[1].new_zeros(()),
        "relation": next(iter(named_parameters))[1].new_zeros(()),
    }
    for (name, _parameter), gradient in zip(named_parameters, gradients):
        if gradient is None:
            continue
        group = "projection" if name.startswith("projections.") else "relation"
        sums[group] = sums[group] + gradient.detach().square().sum()
    result = {group: float(value.sqrt().cpu()) for group, value in sums.items()}
    result["total"] = math.sqrt(sum(value * value for value in result.values()))
    return result


def _apply_function_gradients(
    model: StudentJoinabilityModel,
    protocol_loss: torch.Tensor,
    weighted_function_loss: torch.Tensor,
) -> dict[str, Any]:
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and name.startswith(("projections.", "relations.", "relation_as.", "relation_bs."))
    ]
    parameters = [parameter for _name, parameter in named_parameters]
    protocol_gradients = torch.autograd.grad(
        protocol_loss, parameters, allow_unused=True
    )
    function_gradients = torch.autograd.grad(
        weighted_function_loss, parameters, allow_unused=True
    )
    combined_gradients = []
    for parameter, protocol, function in zip(
        parameters, protocol_gradients, function_gradients
    ):
        gradients = [value for value in (protocol, function) if value is not None]
        parameter.grad = sum(gradients) if gradients else None
        combined_gradients.append(parameter.grad)
    protocol_norms = _gradient_group_norms(named_parameters, protocol_gradients)
    function_norms = _gradient_group_norms(named_parameters, function_gradients)
    combined_norms = _gradient_group_norms(
        named_parameters, tuple(combined_gradients)
    )
    return {
        "protocol": protocol_norms,
        "weighted_function": function_norms,
        "combined": combined_norms,
        "function_to_protocol_total_ratio": (
            function_norms["total"] / protocol_norms["total"]
            if protocol_norms["total"]
            else None
        ),
    }


def _save_checkpoint(
    output_dir: Path,
    step: int,
    model: StudentJoinabilityModel,
    dev_edges: list[EdgeExample],
    fixed_batch: list[EdgeExample],
    store: FeatureStore,
    device: torch.device,
    last_gradient_norms: dict[str, float | None] | None,
) -> dict[str, Any]:
    path = output_dir / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint(model, "student-edge"), path)
    model.eval()
    payload = {
        "optimizer_updates": step,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": checkpoint_fingerprint(path),
        "dev_constructed_lists": _edge_list_metrics(
            model, dev_edges, store, device, 64, "raw_logit"
        ),
        "fixed_training_batch": _fixed_batch_metrics(
            model, fixed_batch, store, device
        ),
        "projection_drift_from_pca": student_projection_drift(model),
        "projection_drift_from_stage_start": student_projection_drift(
            model, reference="stage_start"
        ),
        "projection_references": student_projection_references(model),
        "relation_drift": student_relation_drift(model),
        "last_gradient_norms": last_gradient_norms,
    }
    write_json(path.with_suffix(".json"), payload)
    return payload


def train_arm(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    scores, teacher_manifest = _score_payload(args.output_root)
    schedule = "candidates" if args.arm == "candidates" else "base"
    schedule_path = _schedule_path(args.output_root, schedule)
    candidate_manifest = json.loads(
        (
            args.output_root
            / "taskC_training/candidates_seed13_steps356/manifest.json"
        ).read_text(encoding="utf-8")
    )
    if (
        checkpoint_fingerprint(schedule_path)
        != candidate_manifest["arms"][schedule]["schedule_sha256"]
    ):
        raise ValueError("R12 schedule fingerprint mismatch")
    output_dir = args.output_root / "taskC_training" / f"c_{args.arm}_seed{args.seed}"
    if (output_dir / "manifest.json").is_file():
        raise FileExistsError(f"Training arm already has a manifest: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    store = FeatureStore.from_path(
        args.root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b",
        cache_size=60_000,
    )
    model = _initialize_student(args.root, device)
    optimizer = _optimizer(model)
    dev_edges = load_edge_examples(
        args.output_root / "taskA_correctness/supervision/edge_lists.dev.jsonl",
        split="dev",
    )
    schedule_batches = list(
        _schedule_batches(
            schedule_path,
            scores,
            str(teacher_manifest["teacher_checkpoint_sha256"]),
        )
    )
    first_step, fixed_batch = schedule_batches[0]
    if first_step != 1:
        raise ValueError("R12 schedule must begin at update 1")
    batch_order = _training_batch_order(len(schedule_batches), args.seed)
    iterator = (
        (step, schedule_batches[index][1])
        for step, index in enumerate(batch_order, 1)
    )
    function_reference = (
        _freeze_function_reference(args, store, model, device)
        if args.arm == "function"
        else None
    )
    checkpoints = {
        0: _save_checkpoint(
            output_dir, 0, model, dev_edges, fixed_batch, store, device, None
        )
    }
    history = []
    last_gradient_norms = None
    expected_step = 1
    for step, examples in iterator:
        if step > 356:
            break
        if step != expected_step:
            raise ValueError(
                f"R12 schedule step mismatch: expected {expected_step}, found {step}"
            )
        expected_step += 1
        model.train()
        raw = score_edge_batch(model, examples, store, device, student_score_space="raw_logit")
        teacher = _teacher_list_scores(examples, raw, device)
        objective = _student_edge_losses(
            model,
            examples,
            raw,
            teacher,
            _ranking_scores(raw),
            None,
            ranking_weight=1.0,
            temperature=1.0,
            distillation_weight=0.0 if args.arm == "kd_off" else 0.3,
            edge_bce_weight=0.0,
            anchor_weight=0.1,
            anchor_weight_evidence=0.1,
            positive_loss_mode="sum_probability",
        )
        function = (
            _function_loss(model, function_reference, store, device, step)
            if function_reference is not None
            else raw.logits.new_zeros(())
        )
        loss = objective["loss"] + 0.1 * function
        optimizer.zero_grad()
        gradient_balance = None
        if function_reference is None:
            loss.backward()
        else:
            gradient_balance = _apply_function_gradients(
                model, objective["loss"], 0.1 * function
            )
        last_gradient_norms = student_gradient_norms(model)
        optimizer.step()
        history.append(
            {
                "optimizer_updates": step,
                "loss": float(loss.detach()),
                "supervised_loss": float(objective["supervised_loss"].detach()),
                "distillation_loss": float(objective["distillation_loss"].detach()),
                "parameter_anchor_loss": float(objective["anchor_loss"].detach()),
                "weighted_parameter_anchor_loss": float(
                    objective["weighted_anchor_loss"].detach()
                ),
                "function_loss": float(function.detach()),
                "weighted_function_loss": float((0.1 * function).detach()),
                "gradient_balance": gradient_balance,
            }
        )
        if step in CHECKPOINT_STEPS:
            checkpoints[step] = _save_checkpoint(
                output_dir,
                step,
                model,
                dev_edges,
                fixed_batch,
                store,
                device,
                last_gradient_norms,
            )
        if step % 25 == 0 or step in CHECKPOINT_STEPS:
            print(
                json.dumps(
                    {
                        "arm": args.arm,
                        "step": step,
                        "loss": history[-1]["loss"],
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    if expected_step != 357 or set(checkpoints) != set(CHECKPOINT_STEPS):
        raise ValueError("R12 training did not save every registered checkpoint")
    payload = {
        "format_version": 1,
        "status": "complete",
        "arm": f"C-{args.arm}",
        "unique_algorithm_change": {
            "base": "corrected protocol and complete-candidate Teacher KD control",
            "candidates": "replace half of negative slots with frozen Raw ANN negatives",
            "function": "add 0.1 times the frozen score-space geometry penalty",
            "kd_off": "conditional same-base-schedule diagnostic with Teacher KD weight zero",
        }[args.arm],
        "seed": args.seed,
        "optimizer_updates": 356,
        "batch_size": 64,
        "schedule": str(schedule_path.resolve()),
        "schedule_sha256": checkpoint_fingerprint(schedule_path),
        "schedule_order": (
            "frozen_seed13_order"
            if args.seed == 13
            else "seeded permutation of frozen seed13 batches"
        ),
        "schedule_batch_order": batch_order,
        "teacher_scores_sha256": teacher_manifest["scores_sha256"],
        "teacher_checkpoint_sha256": teacher_manifest["teacher_checkpoint_sha256"],
        "student_protocol": {
            "initialization": "PCA-1024 and identity R",
            "projection_learning_rate": 1e-6,
            "relation_learning_rate": 1e-5,
            "weight_decay": 0.01,
            "ranking_score": "sigmoid(raw_logit)/0.1",
            "distillation_score": "raw_logit",
            "distillation_weight": 0.0 if args.arm == "kd_off" else 0.3,
            "distillation_temperature": 1.0,
            "parameter_anchor_weight": 0.1,
            "function_weight": 0.1 if args.arm == "function" else 0.0,
        },
        "checkpoints": checkpoints,
        "history": history,
        "function_reference": (
            str(
                (
                    args.output_root / "taskC_training/function_reference.json"
                ).resolve()
            )
            if function_reference is not None
            else None
        ),
        "elapsed_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(output_dir / "manifest.json", payload)
    with (args.output_root / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "task": "C1 screen training",
                    "arm": payload["arm"],
                    "status": "complete",
                    "command": payload["command"],
                    "output": str((output_dir / "manifest.json").resolve()),
                    "optimizer_updates": 356,
                    "elapsed_seconds": payload["elapsed_seconds"],
                }
            )
            + "\n"
        )
    print(json.dumps({"status": "complete", "arm": payload["arm"]}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--arm",
        choices=("audit", "base", "candidates", "function", "kd_off"),
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.arm == "audit":
        audit_candidate_quality(arguments)
    else:
        train_arm(arguments)

"""Read-only trajectory probes for the correctness-locked Stage-1 CQET run."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

from . import SCHEMA_VERSION
from .data import sha256_file, write_json, write_jsonl_gz
from .features import ObjectBank
from .losses import (
    aggregate_corrected_lse,
    aggregate_cqet,
    list_kl_divergence,
    rank_mass_loss,
)
from .models import FreshPathTeacher, NativeStudent, QTStudent
if TYPE_CHECKING:
    from .retrieval import PoolRecord
from .train import StudentRecipe, TeacherListScorer, _student_c2_scores, _support_loss, _support_object_ids
from .execution_layout import TEACHER_INFERENCE_CHUNK


def _stats(values: Sequence[float], *, reason: str) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "reason": reason}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p05": float(np.quantile(array, 0.05, method="linear")),
        "p50": float(np.quantile(array, 0.50, method="linear")),
        "p95": float(np.quantile(array, 0.95, method="linear")),
        "reason": None,
    }


@contextmanager
def _teacher_ablation(model: FreshPathTeacher, variant: str):
    handles = []
    if variant == "global_only":
        handles.append(model.relation.register_forward_hook(
            lambda _module, _args, output: torch.zeros_like(output)
        ))
    elif variant == "local_only":
        handles.append(model.global_relation.register_forward_hook(
            lambda _module, _args, output: torch.zeros_like(output)
        ))
    elif variant == "evidence_global_zero":
        width = model.width

        def zero_evidence_global(_module, args):
            features = args[0].clone()
            features[..., 5 * width :] = 0
            return (features,)

        handles.append(model.global_relation.register_forward_pre_hook(zero_evidence_global))
    elif variant != "normal":
        raise ValueError(f"unknown Teacher ablation: {variant}")
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def teacher_content_probe(
    teacher: FreshPathTeacher,
    bank: ObjectBank,
    pools: Mapping[str, PoolRecord],
    labels,
    split_gt: Mapping[str, dict],
    matrix: Mapping[str, Mapping[str, Mapping[str, dict]]],
    *,
    teacher_name: str,
    mode: str,
    seed: int,
    checkpoint: Path,
    output_dir: Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Export fixed-dev content and local/global ablations without changing pools."""
    if mode not in {"cqet", "lse", "qt"}:
        raise ValueError(mode)
    dev = torch.device(device)
    teacher.to(dev).eval()
    # A checkpoint is immutable during this probe: hash its bytes ONCE, not per target.
    checkpoint_sha256 = sha256_file(checkpoint)
    rows: list[dict[str, Any]] = []
    same_qt_e_stds: list[float] = []
    f0_for_paths: list[float] = []
    qet_logits: list[float] = []
    qet_minus_f0: list[float] = []
    old_lse_null_mass: list[float] = []
    witness_margins: list[float] = []
    witness_not_in_natural_bag = 0
    witness_without_natural_competitor = 0
    aggregate_by_variant: dict[str, list[float]] = {
        variant: [] for variant in ("normal", "global_only", "local_only", "evidence_global_zero")
    }

    with torch.no_grad():
        for query_id in sorted(pools, key=lambda value: value.encode("utf-8")):
            pool = pools[query_id]
            targets = list(pool.C150)
            bags = pool.retained_paths
            paths = (
                [] if mode == "qt" else
                [(i, evidence) for i, target in enumerate(targets) for evidence in bags.get(target, ())]
            )
            evidence_ids = list(dict.fromkeys(evidence for _, evidence in paths))
            all_ids = list(dict.fromkeys([query_id, *targets, *evidence_ids]))
            tokens = dict(zip(all_ids, bank.tokens_many(all_ids)))
            evidence_map = {
                evidence: (labels.modality[evidence], bank.z(evidence), tokens[evidence])
                for evidence in evidence_ids
            }
            target_index = torch.tensor(
                [target_i for target_i, _ in paths], dtype=torch.long, device=dev
            )
            per_variant: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
            for variant in aggregate_by_variant:
                with _teacher_ablation(teacher, variant):
                    f0, path_logits = teacher.score_query_lists(
                        (bank.z(query_id), tokens[query_id]),
                        (bank.z_many(targets), [tokens[target] for target in targets]),
                        evidence_map, paths, chunk=TEACHER_INFERENCE_CHUNK,
                    )
                if mode == "qt":
                    aggregate = f0
                elif mode == "cqet":
                    aggregate = aggregate_cqet(f0, path_logits, target_index)
                else:
                    aggregate = aggregate_corrected_lse(f0, path_logits, target_index)
                per_variant[variant] = (f0, path_logits, aggregate)
                aggregate_by_variant[variant].extend(float(value) for value in aggregate)

            normal_f0, normal_paths, _normal_aggregate = per_variant["normal"]
            paths_by_target: dict[int, list[tuple[str, float]]] = {}
            for slot, (target_i, evidence) in enumerate(paths):
                value = float(normal_paths[slot])
                paths_by_target.setdefault(target_i, []).append((evidence, value))
                f0_value = float(normal_f0[target_i])
                f0_for_paths.append(f0_value)
                qet_logits.append(value)
                qet_minus_f0.append(value - f0_value)
            for target_i, target in enumerate(targets):
                target_paths = paths_by_target.get(target_i, [])
                if len(target_paths) >= 2:
                    same_qt_e_stds.append(float(np.std([value for _, value in target_paths])))
                if target_paths:
                    values = np.asarray([value for _, value in target_paths], dtype=np.float64)
                    f0_value = float(normal_f0[target_i])
                    old_lse_null_mass.append(float(1.0 / (1.0 + np.exp(values - f0_value).sum())))
                witness = set(split_gt.get(query_id, {}).get("W", {}).get(target, ()))
                witness_not_in_natural_bag += len(witness - {evidence for evidence, _ in target_paths})
                for evidence, value in target_paths:
                    if evidence not in witness:
                        continue
                    modality = labels.modality[evidence]
                    competitors = [
                        other_value for other, other_value in target_paths
                        if other not in witness and labels.modality[other] == modality
                    ]
                    if competitors:
                        witness_margins.append(value - max(competitors))
                    else:
                        witness_without_natural_competitor += 1
                rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha256,
                    "teacher": teacher_name,
                    "mode": mode,
                    "query_id": query_id,
                    "target_id": target,
                    "natural_evidence_ids": [evidence for evidence, _ in target_paths],
                    "variants": {
                        variant: {
                            "f0": float(values[0][target_i]),
                            "path_logits": [
                                float(values[1][slot])
                                for slot, (path_target_i, _evidence) in enumerate(paths)
                                if path_target_i == target_i
                            ],
                            "aggregated_score": float(values[2][target_i]),
                        }
                        for variant, values in per_variant.items()
                    },
                })

    correlation = None
    correlation_reason = None
    if len(qet_logits) < 2 or np.std(qet_logits) == 0 or np.std(f0_for_paths) == 0:
        correlation_reason = "fewer_than_two_or_zero_variance_paths"
    else:
        correlation = float(np.corrcoef(f0_for_paths, qet_logits)[0, 1])

    real_swap: list[float] = []
    if mode != "qt" and "Real" in matrix.get(teacher_name, {}) and "Swap" in matrix[teacher_name]:
        for query_id in pools:
            real = matrix[teacher_name]["Real"][query_id]
            swap = matrix[teacher_name]["Swap"][query_id]
            real_scores = dict(zip(real["target_ids"], real["scores"]))
            swap_scores = dict(zip(swap["target_ids"], swap["scores"]))
            real_swap.extend(real_scores[target] - swap_scores[target] for target in real_scores)

    normal = np.asarray(aggregate_by_variant["normal"], dtype=np.float64)
    ablation_delta = {}
    for variant in ("global_only", "local_only", "evidence_global_zero"):
        values = np.asarray(aggregate_by_variant[variant], dtype=np.float64)
        ablation_delta[variant] = _stats(
            list(values - normal), reason="no_scored_targets"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "CONTENT_ABLATIONS.jsonl.gz"
    write_jsonl_gz(raw_path, rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "teacher": teacher_name,
        "mode": mode,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "query_count": len(pools),
        "target_rows": len(rows),
        "same_QT_vary_E_path_std": _stats(same_qt_e_stds, reason="fewer_than_two_natural_paths"),
        "f0_QET_correlation": {
            "count": len(qet_logits), "value": correlation, "reason": correlation_reason,
        },
        "raw_QET_minus_f0": _stats(qet_minus_f0, reason="no_natural_paths"),
        "old_LSE_null_mass": _stats(old_lse_null_mass, reason="no_nonempty_bags"),
        "W_vs_strongest_same_modality_competitor_margin": {
            **_stats(witness_margins, reason="no_natural_witness_with_competitor"),
            "witness_not_in_natural_bag": witness_not_in_natural_bag,
            "witness_without_natural_competitor": witness_without_natural_competitor,
        },
        "Real_minus_Swap": _stats(real_swap, reason="QT_mode_has_no_swap"),
        "ablation_aggregate_minus_normal": ablation_delta,
        "raw": {"path": raw_path.name, "sha256": sha256_file(raw_path)},
    }
    write_json(output_dir / "CONTENT_PROBE_SUMMARY.json", summary)
    return summary


def _gradient_vector(model: nn.Module) -> Tensor:
    values = [
        (
            parameter.grad.detach().reshape(-1).cpu()
            if parameter.grad is not None else torch.zeros(parameter.numel(), dtype=parameter.dtype)
        )
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return torch.cat(values) if values else torch.empty(0)


def _cosines(vectors: Mapping[str, Tensor]) -> dict[str, Optional[float]]:
    result = {}
    names = sorted(vectors)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a, b = vectors[left], vectors[right]
            denominator = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
            result[f"{left}__{right}"] = float(torch.dot(a, b) / denominator) if denominator else None
    return result


def _teacher_component(
    model: FreshPathTeacher,
    bank: ObjectBank,
    row: dict,
    component: str,
    mode: str,
    scale: float,
) -> Optional[tuple[float, float]]:
    query_id = row["query_id"]
    targets = list(row["targets"])
    positives = set(row["positives"])
    positive_mask = torch.tensor([target in positives for target in targets], device=bank.z(query_id).device)
    bags = row.get("natural_bags", {})
    evidence_ids = (
        list(dict.fromkeys(evidence for target in targets for evidence in bags.get(target, ())))
        if component == "aggregate" else []
    )
    auxiliary_ids = _support_object_ids(row.get("support_records", [])) if component == "support" else []
    all_ids = list(dict.fromkeys([query_id, *targets, *evidence_ids, *auxiliary_ids]))
    tokens = dict(zip(all_ids, bank.tokens_many(all_ids)))
    zq = bank.z(query_id)
    scorer = TeacherListScorer(model, 16)

    if component == "support":
        loss = _support_loss(
            bank, query_id, row.get("support_records", []), tokens[query_id], scorer, tokens
        )
    else:
        pairs = [
            ("table", zq, tokens[query_id], "table", bank.z(target), tokens[target])
            for target in targets
        ]
        keys = [((query_id, 0), (target, 1)) for target in targets]
        f0 = scorer.score_pairs(pairs, keys)
        if component == "direct":
            loss = rank_mass_loss(f0, positive_mask)
        elif component == "aggregate" and mode != "qt":
            paths = [(i, evidence) for i, target in enumerate(targets) for evidence in bags.get(target, ())]
            trips = [
                ("table", zq, tokens[query_id], bank.kind(evidence), bank.z(evidence), tokens[evidence],
                 "table", bank.z(targets[target_i]), tokens[targets[target_i]])
                for target_i, evidence in paths
            ]
            trip_keys = [
                ((query_id, 0), (evidence, 2), (targets[target_i], 1))
                for target_i, evidence in paths
            ]
            path_scores = scorer.score_triplets(trips, trip_keys)
            target_index = torch.tensor(
                [target_i for target_i, _ in paths], dtype=torch.long, device=f0.device
            )
            aggregate = (
                aggregate_cqet(f0, path_scores, target_index)
                if mode == "cqet" else aggregate_corrected_lse(f0, path_scores, target_index)
            )
            loss = rank_mass_loss(aggregate, positive_mask)
        else:
            return None
    if loss is None:
        return None
    leaves = [score for score in scorer.scores]
    score_grads = torch.autograd.grad(loss, leaves, retain_graph=True, allow_unused=True)
    shift_gradient = float(sum(
        float(gradient.sum()) for gradient in score_grads if gradient is not None
    ))
    scorer.backward(loss, scale=scale)
    return float(loss.detach()), shift_gradient


def teacher_gradient_probe(
    model: FreshPathTeacher,
    bank: ObjectBank,
    rows: Sequence[dict] | dict,
    *,
    mode: str,
    stage: str,
) -> dict[str, Any]:
    """Compare component gradients on the same fixed train128 cohort."""
    records = [rows] if isinstance(rows, dict) else list(rows)
    model.eval()
    if stage.startswith("TB_"):
        model.set_tb_trainable()
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    values = {}
    vectors = {}
    for component in ("direct", "aggregate", "support"):
        if component in {"aggregate", "support"} and mode == "qt":
            active = []
        elif component == "support":
            active = [
                row for row in records
                if any(item.get("positives") and item.get("competitors")
                       for item in row.get("support_records", ()))
            ]
        else:
            active = [
                row for row in records
                if any(target in set(row["positives"]) for target in row["targets"])
                and any(target not in set(row["positives"]) for target in row["targets"])
            ]
        if not active:
            values[component] = {
                "loss": None, "gradient_norm": None,
                "shared_score_shift_gradient": None, "parameter_count": 0,
                "active_queries": 0, "reason": "component_inactive_for_probe_or_mode",
            }
            continue
        model.zero_grad(set_to_none=True)
        losses = []
        shifts = []
        for row in active:
            result = _teacher_component(
                model, bank, row, component, mode, scale=1.0 / len(active)
            )
            if result is None:
                raise RuntimeError("prevalidated Teacher probe component became inactive")
            loss, shift = result
            losses.append(loss)
            shifts.append(shift)
        vector = _gradient_vector(model)
        values[component] = {
            "loss": float(np.mean(losses)),
            "gradient_norm": float(torch.linalg.vector_norm(vector)),
            "shared_score_shift_gradient": float(np.mean(shifts)),
            "parameter_count": int(vector.numel()),
            "active_queries": len(active),
            "reason": None,
        }
        vectors[component] = vector
    model.zero_grad(set_to_none=True)
    return {
        "probe_query_ids": [row["query_id"] for row in records],
        "probe_query_count": len(records),
        "dropout_mode": "eval",
        "components": values,
        "gradient_cosines": _cosines(vectors),
    }


def _student_component(
    model: NativeStudent | QTStudent,
    bank: ObjectBank,
    row: dict,
    component: str,
    teacher_logits: Optional[tuple[Tensor, Optional[Tensor]]],
    scale: float,
    recipe: StudentRecipe,
) -> Optional[float]:
    direct, evidence, bag_targets, _ = _student_c2_scores(model, bank, row, recipe.logit_scale)
    positives = set(row["positives"])
    direct_positive = torch.tensor(
        [target in positives for target in row["targets"]], device=direct.device
    )
    if component == "SUP":
        loss = rank_mass_loss(direct, direct_positive)
        if evidence is not None:
            evidence_positive = torch.tensor(
                [target in positives for _, target in bag_targets], device=direct.device
            )
            evidence_loss = rank_mass_loss(evidence, evidence_positive)
            if evidence_loss is not None:
                loss = evidence_loss if loss is None else loss + evidence_loss
    elif component == "direct_KD" and teacher_logits is not None and not isinstance(model, QTStudent):
        loss = list_kl_divergence(direct, teacher_logits[0].to(direct.device) / recipe.kd_temperature)
    elif (
        component == "evidence_KD" and teacher_logits is not None
        and not isinstance(model, QTStudent) and evidence is not None and teacher_logits[1] is not None
    ):
        loss = list_kl_divergence(evidence, teacher_logits[1].to(evidence.device) / recipe.kd_temperature)
    elif component == "anchor" and recipe.anchor_weight:
        loss = recipe.anchor_weight * model.anchor_loss()
    else:
        loss = None
    if loss is None:
        return None
    (loss * scale).backward()
    return float(loss.detach())


def student_gradient_probe(
    model: NativeStudent | QTStudent,
    bank: ObjectBank,
    rows: Sequence[dict] | dict,
    teacher_logits: Optional[
        Mapping[str, tuple[Tensor, Optional[Tensor]]] | tuple[Tensor, Optional[Tensor]]
    ],
    recipe: StudentRecipe = StudentRecipe(),
) -> dict[str, Any]:
    """Per-component gradient norms and cosines under the same scale/temperature as training."""
    records = [rows] if isinstance(rows, dict) else list(rows)
    if teacher_logits is None:
        logits_by_query = {}
    elif isinstance(teacher_logits, tuple):
        if len(records) != 1:
            raise ValueError("single Teacher logit tuple only aligns with one probe record")
        logits_by_query = {records[0]["query_id"]: teacher_logits}
    else:
        logits_by_query = teacher_logits
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    values = {}
    vectors = {}
    for component in ("SUP", "direct_KD", "evidence_KD", "anchor"):
        if component == "anchor":
            active = records[:1]
        elif component == "SUP":
            active = [
                row for row in records
                if any(target in set(row["positives"]) for target in row["targets"])
                and any(target not in set(row["positives"]) for target in row["targets"])
            ]
        elif isinstance(model, QTStudent):
            active = []
        elif component == "direct_KD":
            active = [
                row for row in records
                if row["query_id"] in logits_by_query and len(row["targets"]) > 1
            ]
        else:
            active = [
                row for row in records
                if row["query_id"] in logits_by_query
                and logits_by_query[row["query_id"]][1] is not None
                and logits_by_query[row["query_id"]][1].numel() > 1
            ]
        if not active:
            values[component] = {
                "loss": None, "gradient_norm": None, "parameter_count": 0,
                "active_queries": 0,
                "reason": (
                    "QT_only_independent_of_CQET" if isinstance(model, QTStudent) and "KD" in component
                    else "component_inactive_for_probe"
                ),
            }
            continue
        model.zero_grad(set_to_none=True)
        losses = []
        for row in active:
            loss = _student_component(
                model, bank, row, component,
                logits_by_query.get(row["query_id"]), scale=1.0 / len(active), recipe=recipe,
            )
            if loss is None:
                if component == "anchor" and not recipe.anchor_weight:
                    break
                raise RuntimeError("prevalidated Student probe component became inactive")
            losses.append(loss)
        if not losses:
            values[component] = {
                "loss": None, "gradient_norm": None, "parameter_count": 0,
                "active_queries": 0, "reason": "anchor_weight_zero",
            }
            continue
        vector = _gradient_vector(model)
        values[component] = {
            "loss": float(np.mean(losses)),
            "gradient_norm": float(torch.linalg.vector_norm(vector)),
            "parameter_count": int(vector.numel()),
            "active_queries": len(active),
            "reason": None,
        }
        vectors[component] = vector
    model.zero_grad(set_to_none=True)
    return {
        "probe_query_ids": [row["query_id"] for row in records],
        "probe_query_count": len(records),
        "recipe": recipe.as_dict(),
        "components": values,
        "gradient_cosines": _cosines(vectors),
    }

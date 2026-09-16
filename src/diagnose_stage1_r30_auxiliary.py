"""Complete the preregistered R30 parameter and production-COV diagnostics."""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from diagnose_stage1_r30 import BRIDGE, OUT, checkpoint_spec, stable_probe
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.row_support import load_evidence_content_keys
from run_stage1_bridge import sha256, stable_sha, write_json
from run_stage1_r11_task_e import select_evidence
from run_stage1_r30 import (
    FEATURE_ROOT,
    SCHEDULE_ROOT,
    TEACHER_ROOT,
    _edge_batches,
    _load_schedule,
    _loss,
    _teacher_scores,
)


DIAGNOSTICS = OUT / "diagnostics_repaired"
C1_NAMES = (
    "STOP356", "JOINT500", "JOINT659", "F-P500", "F-P659",
    "F-P-ETNAT500", "F-P-ETNAT659",
)
OBJECTIVES = ("D_SUP", "QE_SUP", "ET_SUP", "KD_weighted_0.3", "anchor_weighted_0.1")
GROUP_PREFIXES = {
    "P": ("projections.", "projection_residual_inputs.", "projection_residual_outputs."),
    "R": ("relations.", "relation_as.", "relation_bs.", "confidence_alphas.", "confidence_biases."),
}


def json_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parameter_group(name: str) -> str | None:
    for group, prefixes in GROUP_PREFIXES.items():
        if name.startswith(prefixes):
            return group
    return None


def spectral_norm_estimate(tensor: torch.Tensor, iterations: int = 20) -> float:
    matrix = tensor.detach().float()
    if matrix.ndim < 2:
        return float(matrix.norm().cpu())
    matrix = matrix.reshape(matrix.shape[0], -1)
    if not matrix.numel() or not float(matrix.norm()):
        return 0.0
    vector = torch.ones(matrix.shape[1], device=matrix.device, dtype=matrix.dtype)
    vector /= vector.norm()
    for _ in range(iterations):
        left = matrix @ vector
        if not float(left.norm()):
            return 0.0
        left /= left.norm()
        vector = matrix.T @ left
        if not float(vector.norm()):
            return 0.0
        vector /= vector.norm()
    return float((matrix @ vector).norm().cpu())


def parameter_drift(
    parent: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    per_parameter: dict[str, dict[str, Any]] = {}
    group_squares: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for name, before in parent.items():
        group = parameter_group(name)
        if group is None or name not in current or not torch.is_floating_point(before):
            continue
        after = current[name]
        before_float = before.detach().float()
        delta = after.detach().float() - before_float
        base_norm = float(before_float.norm())
        delta_norm = float(delta.norm())
        group_squares[group][0] += delta_norm**2
        group_squares[group][1] += base_norm**2
        before_device = before_float.to(device)
        after_device = after.detach().float().to(device)
        delta_device = after_device - before_device
        per_parameter[name] = {
            "group": group,
            "bitwise_equal_parent": torch.equal(before, after),
            "frobenius": float(after.detach().float().norm()),
            "delta_frobenius": delta_norm,
            "relative_frobenius": delta_norm / base_norm if base_norm else None,
            "spectral_norm_estimate": spectral_norm_estimate(after_device),
            "delta_spectral_norm_estimate": spectral_norm_estimate(delta_device),
            "spectral_iterations": 20,
        }
    return {
        "groups": {
            group: {
                "relative_frobenius": math.sqrt(delta_sq / base_sq) if base_sq else None,
                "bitwise_equal_parent": all(
                    row["bitwise_equal_parent"]
                    for row in per_parameter.values()
                    if row["group"] == group
                ),
            }
            for group, (delta_sq, base_sq) in group_squares.items()
        },
        "parameters": per_parameter,
        "spectral_method": "20 deterministic power iterations from an all-ones right vector",
    }


def gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool = False,
) -> list[torch.Tensor | None]:
    values = torch.autograd.grad(
        loss, parameters, allow_unused=True, retain_graph=retain_graph
    )
    return [None if value is None else value.detach().cpu() for value in values]


def cosine(left: torch.Tensor | None, right: torch.Tensor | None) -> float | None:
    if left is None or right is None:
        return None
    left_norm = float(left.double().norm())
    right_norm = float(right.double().norm())
    if not left_norm or not right_norm:
        return None
    return float(torch.sum(left.double() * right.double()) / (left_norm * right_norm))


def aggregate_gradient(
    values: list[torch.Tensor | None], names: list[str], group: str
) -> float:
    selected = [
        value.double()
        for name, value in zip(names, values)
        if value is not None and parameter_group(name) == group
    ]
    if not selected:
        return 0.0
    return math.sqrt(sum(float(value.square().sum()) for value in selected))


def aggregate_cosine(
    left: list[torch.Tensor | None],
    right: list[torch.Tensor | None],
    names: list[str],
    group: str,
) -> float | None:
    dot = left_square = right_square = 0.0
    for name, left_value, right_value in zip(names, left, right):
        if parameter_group(name) != group:
            continue
        if left_value is not None:
            left_square += float(left_value.double().square().sum())
        if right_value is not None:
            right_square += float(right_value.double().square().sum())
        if left_value is not None and right_value is not None:
            dot += float(torch.sum(left_value.double() * right_value.double()))
    if not left_square or not right_square:
        return None
    return dot / math.sqrt(left_square * right_square)


def fixed_batch_gradients(
    model: torch.nn.Module,
    batch: list[Any],
    store: FeatureStore,
    device: torch.device,
) -> dict[str, Any]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter_group(name)]
    names = [name for name, _parameter in named]
    parameters = [parameter for _name, parameter in named]
    relation_groups = {
        "D_SUP": [row for row in batch if row.source_type == "table" and row.destination_type == "table"],
        "QE_SUP": [row for row in batch if row.source_type == "table" and row.destination_type in {"text", "image"}],
        "ET_SUP": [row for row in batch if row.source_type in {"text", "image"} and row.destination_type == "table"],
    }
    objective_gradients: dict[str, list[torch.Tensor | None]] = {}
    losses: dict[str, float] = {}
    lists: dict[str, int] = {}
    for objective, examples in relation_groups.items():
        if not examples:
            raise ValueError(f"fixed batch has no {objective} examples")
        terms, _raw = _loss(model, examples, store, device)
        loss = terms["weighted_supervised_loss"]
        losses[objective] = float(loss.detach().cpu())
        lists[objective] = len(examples)
        objective_gradients[objective] = gradients(loss, parameters)
    terms, _raw = _loss(model, batch, store, device)
    kd = 0.3 * terms["distillation_loss"]
    anchor = terms["weighted_anchor_loss"]
    losses["KD_weighted_0.3"] = float(kd.detach().cpu())
    losses["anchor_weighted_0.1"] = float(anchor.detach().cpu())
    objective_gradients["KD_weighted_0.3"] = gradients(kd, parameters, retain_graph=True)
    objective_gradients["anchor_weighted_0.1"] = gradients(anchor, parameters)

    norms: dict[str, Any] = {}
    for objective, values in objective_gradients.items():
        norms[objective] = {
            "aggregate": {
                group: aggregate_gradient(values, names, group)
                for group in GROUP_PREFIXES
            },
            "per_parameter": {
                name: (None if value is None else float(value.double().norm()))
                for name, value in zip(names, values)
            },
        }

    cosines: dict[str, Any] = {}
    for left_name, right_name in itertools.combinations(OBJECTIVES, 2):
        left = objective_gradients[left_name]
        right = objective_gradients[right_name]
        cosines[f"{left_name}_vs_{right_name}"] = {
            "aggregate": {
                group: aggregate_cosine(left, right, names, group)
                for group in GROUP_PREFIXES
            },
            "per_parameter": {
                name: cosine(left_value, right_value)
                for name, left_value, right_value in zip(names, left, right)
            },
        }
    return {
        "losses": losses,
        "lists": lists,
        "gradient_norms": norms,
        "gradient_cosines": cosines,
        "interpretation_boundary": (
            "Raw fixed-batch gradients only; these are not AdamW-preconditioned update shares."
        ),
    }


def run_parameter_diagnostics(seed: int, device_name: str) -> dict[str, Any]:
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    schedule_path = SCHEDULE_ROOT / f"seed{seed}_steps659/closure_full.jsonl.gz"
    teacher_path = TEACHER_ROOT / f"teacher_edge_cache_seed{seed}.jsonl.gz"
    batches = _edge_batches(_load_schedule(schedule_path), _teacher_scores(teacher_path))
    batch = batches[356]
    batch_identity = [
        [row.query_id, row.source_type, row.destination_type, list(row.candidate_ids), list(row.teacher_logits or ())]
        for row in batch
    ]
    parent_path, _index = checkpoint_spec(seed, "STOP356")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)["state_dict"]
    store = FeatureStore.from_path(FEATURE_ROOT, cache_size=4096)
    checkpoints: dict[str, Any] = {}
    for name in C1_NAMES:
        checkpoint, _index = checkpoint_spec(seed, name)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        model = load_student(checkpoint, device).train()
        checkpoints[name] = {
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)},
            "parameter_drift_from_STOP356": parameter_drift(parent, state, device),
            "fixed_batch_gradients": fixed_batch_gradients(model, batch, store, device),
        }
        del model, state
        torch.cuda.empty_cache()
    result = {
        "status": "pass",
        "seed": seed,
        "fixed_batch": {
            "global_step": 357,
            "schedule": {"path": str(schedule_path.resolve()), "sha256": sha256(schedule_path)},
            "teacher": {"path": str(teacher_path.resolve()), "sha256": sha256(teacher_path)},
            "identity_sha256": stable_sha(batch_identity),
            "examples": len(batch),
            "scope": "common native Bridge batch; diagnostic only, no optimizer step",
        },
        "checkpoints": checkpoints,
        "runtime": {
            "seconds": time.monotonic() - started,
            "command": [sys.executable, *sys.argv],
            "device": torch.cuda.get_device_name(device),
        },
    }
    if not all(
        checkpoints[name]["parameter_drift_from_STOP356"]["groups"]["P"]["bitwise_equal_parent"]
        for name in ("F-P500", "F-P659", "F-P-ETNAT500", "F-P-ETNAT659")
    ):
        raise ValueError("a freeze-P endpoint changed projection bytes")
    destination = DIAGNOSTICS / f"PARAMETER_AND_GRADIENT_DIAGNOSTICS_seed{seed}.json"
    write_json(destination, result)
    print(json.dumps({"status": "pass", "seed": seed, "output": str(destination.resolve())}))
    return result


def ranking_path(generator: str) -> Path:
    if generator.startswith("STOP356_s"):
        seed = int(generator.rsplit("s", 1)[1])
        suffix = "" if seed == 13 else "_seed29"
        return BRIDGE / f"evaluation/rankings/B5_356{suffix}/rankings.jsonl.gz"
    if generator.startswith("JOINT500_s"):
        seed = int(generator.rsplit("s", 1)[1])
        suffix = "" if seed == 13 else "_seed29"
        return BRIDGE / f"evaluation/rankings/B5_500{suffix}/rankings.jsonl.gz"
    if generator.startswith("JOINT659_s"):
        seed = int(generator.rsplit("s", 1)[1])
        suffix = "" if seed == 13 else "_seed29"
        return BRIDGE / f"evaluation/rankings/B5_659{suffix}/rankings.jsonl.gz"
    return OUT / f"rankings/{generator}/rankings.jsonl.gz"


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()) if len(array) else None,
        "q10": float(np.quantile(array, 0.10)) if len(array) else None,
        "median": float(np.quantile(array, 0.50)) if len(array) else None,
        "q90": float(np.quantile(array, 0.90)) if len(array) else None,
    }


def cov_generators() -> list[str]:
    values = []
    for seed in (13, 29):
        values.extend((f"STOP356_s{seed}", f"JOINT500_s{seed}", f"JOINT659_s{seed}"))
        for recipe in ("F-P", "F-P-ETNAT"):
            values.extend(f"{recipe}{step}_s{seed}" for step in (500, 659))
        values.extend(f"C2-F-P{step}_s{seed}" for step in (89, 178))
    return values


def run_cov_diagnostics() -> dict[str, Any]:
    started = time.monotonic()
    content_keys_path = OUT / "RESOLVED_INPUTS.json"
    resolved = json.loads(content_keys_path.read_text(encoding="utf-8"))
    content_record = next(
        row for row in resolved["records"] if row["name"] == "shared_inputs.content_keys"
    )
    content_keys, content_sha = load_evidence_content_keys(Path(content_record["path"]))
    store = FeatureStore.from_path(FEATURE_ROOT, cache_size=20000)
    generators: dict[str, Any] = {}
    replay_total = replay_mismatches = 0
    replay_max_error = 0.0
    for generator in cov_generators():
        path = ranking_path(generator)
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = list(json_rows(path))
        probe_ids = set(stable_probe([str(row["query_id"]) for row in rows], count=8))
        values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        empty_bags = 0
        local_replay = local_mismatches = 0
        local_max_error = 0.0
        for row in rows:
            positives = {str(value) for value in row["positive_target_ids"]}
            saved = {str(value["target_id"]): value for value in row["E_paths"]}
            for target in row["E_pre_retention"]:
                target_id = str(target["target_id"])
                evidence_paths = [value for value in target["paths"] if value.get("kind") == "evidence"]
                if not evidence_paths:
                    empty_bags += 1
                    continue
                retained = saved.get(target_id)
                if retained is None or not retained.get("selected_evidence_ids"):
                    continue
                role = "known_positive" if target_id in positives else "unknown"
                values[role]["evidence_score"].append(float(retained["evidence_score"]))
                values[role]["retained_path_lse"].append(float(retained["retained_path_lse"]))
                values[role]["candidate_paths"].append(float(len(evidence_paths)))
                values[role]["selected_evidence"].append(float(len(retained["selected_evidence_ids"])))

            if str(row["query_id"]) not in probe_ids:
                continue
            by_role: dict[str, dict[str, Any]] = {}
            for target in row["E_pre_retention"]:
                target_id = str(target["target_id"])
                role = "known_positive" if target_id in positives else "unknown"
                if role not in by_role and target_id in saved:
                    by_role[role] = target
            support_cache: dict[str, list[float]] = {}
            for target in by_role.values():
                target_id = str(target["target_id"])
                selected, score = select_evidence(
                    "e2_row_coverage", target["paths"], query_id=str(row["query_id"]),
                    store=store, content_keys=content_keys, top_l=20, budget=4,
                    support_cache=support_cache,
                )
                retained = saved[target_id]
                local_replay += 1
                error = abs(float(score) - float(retained["evidence_score"])) if score is not None else math.inf
                local_max_error = max(local_max_error, error)
                if selected != [str(value) for value in retained["selected_evidence_ids"]] or error > 1e-12:
                    local_mismatches += 1
        generators[generator] = {
            "rankings": {"path": str(path.resolve()), "sha256": sha256(path)},
            "nonempty_bag_statistics": {
                role: {metric: distribution(metric_values) for metric, metric_values in role_values.items()}
                for role, role_values in values.items()
            },
            "empty_bags_excluded": empty_bags,
            "formula_replay": {
                "rows": local_replay,
                "mismatches": local_mismatches,
                "max_absolute_score_error": local_max_error,
            },
        }
        replay_total += local_replay
        replay_mismatches += local_mismatches
        replay_max_error = max(replay_max_error, local_max_error)
    result = {
        "status": "pass" if replay_total and replay_mismatches == 0 and replay_max_error <= 1e-12 else "fail",
        "formula": {
            "name": "production e2_row_coverage",
            "implementation": str((Path(__file__).resolve().parent / "run_stage1_r11_task_e.py")),
            "top_l": 20,
            "budget": 4,
            "content_keys": {"path": str(Path(content_record["path"]).resolve()), "sha256": content_sha},
            "labels_used_for": "offline known-positive/unknown stratification only",
            "empty_bag_policy": "excluded rather than represented by a zero placeholder",
        },
        "formula_replay": {
            "rows": replay_total,
            "mismatches": replay_mismatches,
            "max_absolute_score_error": replay_max_error,
        },
        "generators": generators,
        "runtime": {"seconds": time.monotonic() - started, "command": [sys.executable, *sys.argv]},
    }
    if result["status"] != "pass":
        raise ValueError(f"production COV replay failed: {result['formula_replay']}")
    destination = DIAGNOSTICS / "COV_SCORE_DIAGNOSTIC.json"
    write_json(destination, result)
    print(json.dumps({"status": "pass", "output": str(destination.resolve()), "replay": result["formula_replay"]}))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(13, 29))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cov-only", action="store_true")
    args = parser.parse_args()
    if args.cov_only:
        run_cov_diagnostics()
        return
    if args.seed is None:
        parser.error("--seed is required unless --cov-only is used")
    run_parameter_diagnostics(args.seed, args.device)


if __name__ == "__main__":
    main()

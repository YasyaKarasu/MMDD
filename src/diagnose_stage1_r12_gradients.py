#!/usr/bin/env python
"""Measure relation-specific gradient conflicts on a fixed train-fit batch."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import load_edge_examples
from mmdd_stage1.features import FeatureStore, OBJECT_TYPES
from mmdd_stage1.objectives import distillation_kl
from mmdd_stage1.pca import load_pca_projection
from mmdd_stage1.scoring import ListScores, global_edge_positive_ids, score_edge_batch, score_edge_batch_in_batch
from mmdd_stage1.teacher_logits import load_teacher_logits
from mmdd_stage1.training import _edge_teacher_scores, _teacher_edge_loss, sample_mixed_epoch


def run(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    r11 = args.root / "work/stage1_optimization_r11_20260908"
    r10 = args.root / "work/stage1_optimization_r10_20260907"
    output = args.output_root / "taskC_training/gradient_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    examples = load_edge_examples(r11 / "taskA_protocol/supervision/edge_lists.train_fit.jsonl", split="train")
    teacher_path = r11 / "taskC_clean/teacher/teacher_edge.pt"
    examples, teacher_cache, hit = load_teacher_logits(
        examples, r11 / "taskC_clean/teacher/teacher_logits/edge", checkpoint_fingerprint(teacher_path),
        teacher_score_space="raw_logit",
    )
    if not hit:
        raise FileNotFoundError(teacher_cache)
    sampled, _ = sample_mixed_epoch(examples, (), random.Random(13), hard_fraction=0.5, dataset_sampling_alpha=0)
    batch = sampled[:64]
    known = global_edge_positive_ids(examples)
    store = FeatureStore.from_path(r10 / "features_qwen3_vl_embedding_8b", cache_size=1000)
    basis = load_pca_projection(r10 / "baselines/pca_entitables_v9_1024.pt", input_dim=4096, student_dim=1024).to(device)
    checkpoints = [
        ("pca", r11 / "taskA_protocol/baselines/pca_init.pt"),
        ("c2_edge_epoch1", r11 / "taskC_clean/c2_long/student_edge.epochs/epoch_001.pt"),
        ("c2_edge_epoch2", r11 / "taskC_clean/c2_long/student_edge.last.pt"),
        ("c2_path_epoch2", r11 / "taskC_clean/c2_long/student_path.last.pt"),
    ]
    results = {}
    fixed_batch = [{"source_id": row.query_id, "source_type": row.source_type, "destination_type": row.destination_type,
                    "candidate_ids": row.candidate_ids, "positive_ids": row.positive_ids} for row in batch]
    write_json(output / "fixed_batch.json", fixed_batch)
    for name, path in checkpoints:
        model = load_student(path, device).eval()
        parameters = {f"P/{kind}": model.projections[kind].weight for kind in OBJECT_TYPES}
        parameters.update({f"R/{relation}": model.relations[relation] for relation in (
            "table_to_table", "table_to_text", "table_to_image", "text_to_table", "image_to_table",
        )})
        values = list(parameters.values())
        raw = score_edge_batch(model, batch, store, device)
        expanded = score_edge_batch_in_batch(model, batch, store, device, known_positive_ids=known,
                                             sampling_seed=13, sampling_context="epoch=0:step=1")
        ranking_scores = ListScores(torch.sigmoid(expanded.logits) / 0.1, expanded.candidate_mask,
                                    expanded.positive_indices, expanded.positive_mask)
        teacher_scores = _edge_teacher_scores(batch, device)
        p_anchor = sum((model.projections[kind].weight - basis).square().mean() for kind in OBJECT_TYPES)
        r_anchor = sum(model.relation_residual_squared_norm(key) / model.student_dim ** 2
                       for key in model.relations)
        terms = {
            "ranking": _teacher_edge_loss(ranking_scores),
            "kd_weighted_0.3": 0.3 * distillation_kl(raw.logits, teacher_scores.logits, raw.candidate_mask, 1),
            "P_anchor_weighted_0.1": 0.1 * p_anchor,
            "R_anchor_weighted_0.1": 0.1 * r_anchor,
        }
        gradients = {}
        for term, loss in terms.items():
            gradients[term] = [gradient.detach().cpu() if gradient is not None else None
                               for gradient in torch.autograd.grad(loss, values, allow_unused=True, retain_graph=True)]
        margins = {}
        for relation in ("table_to_table", "table_to_text", "table_to_image", "text_to_table", "image_to_table"):
            indices = [index for index, row in enumerate(batch) if f"{row.source_type}_to_{row.destination_type}" == relation]
            if not indices:
                margins[relation] = {"lists": 0}
                continue
            differences = []
            for index in indices:
                positive = expanded.positive_mask[index] & expanded.candidate_mask[index]
                negative = ~expanded.positive_mask[index] & expanded.candidate_mask[index]
                differences.append(expanded.logits[index, positive].mean() - expanded.logits[index, negative].max())
            margin = torch.stack(differences).mean()
            margin_grad = [gradient.detach().cpu() if gradient is not None else None
                           for gradient in torch.autograd.grad(margin, values, allow_unused=True, retain_graph=True)]
            change = {}
            for term, grads in gradients.items():
                change[term] = -sum(
                    float(torch.sum(left.double() * right.double())) * (1e-6 if key.startswith("P/") else 1e-5)
                    for key, left, right in zip(parameters, margin_grad, grads) if left is not None and right is not None
                )
            margins[relation] = {"lists": len(indices), "raw_positive_mean_minus_max_negative": float(margin.detach()),
                                 "first_order_margin_delta_under_declared_lr_sgd": change}
        norms = {term: {key: float(gradient.norm()) if gradient is not None else None
                        for key, gradient in zip(parameters, grads)} for term, grads in gradients.items()}
        cosines = {}
        for left, right in itertools.combinations(gradients, 2):
            result = {}
            for key, a, b in zip(parameters, gradients[left], gradients[right]):
                if a is None or b is None or not float(a.norm()) or not float(b.norm()):
                    result[key] = None
                else:
                    result[key] = float(torch.sum(a.double() * b.double()) / (a.double().norm() * b.double().norm()))
            cosines[f"{left} vs {right}"] = result
        results[name] = {"checkpoint": str(path), "checkpoint_sha256": checkpoint_fingerprint(path),
                         "losses": {key: float(value.detach()) for key, value in terms.items()},
                         "gradient_norms": norms, "gradient_cosines": cosines, "margins": margins,
                         "expanded_candidate_ids": expanded.candidate_ids,
                         "batch_relations": {key: value["lists"] for key, value in margins.items()}}
        write_json(output / f"{name}.json", results[name])
        del gradients, model, expanded, raw, terms, values, parameters
        torch.cuda.empty_cache()
    payload = {"identity": "C0 gradient diagnostic only; no parameter update or new training claim",
               "batch": str(output / "fixed_batch.json"), "batch_sha256": checkpoint_fingerprint(output / "fixed_batch.json"),
               "teacher": str(teacher_path), "teacher_cache": str(teacher_cache),
               "reference": "Fixed full-chain PCA for P diagnostic; identity for R",
               "kd_mask": "Historical R11 original local lists; supervised ranking uses actual globally masked in-batch expansion",
               "margin_delta_scope": "First-order SGD direction, not the stateful AdamW update",
               "results": results, "elapsed_seconds": time.monotonic() - started,
               "command": [sys.executable, *sys.argv], "code_sha256": checkpoint_fingerprint(Path(__file__))}
    write_json(output / "summary.json", payload)
    with (args.output_root / "runs.jsonl").open("a") as handle:
        handle.write(json.dumps({"task": "C0 fixed-batch gradients", "status": "pass", "command": payload["command"],
                                 "elapsed_seconds": payload["elapsed_seconds"], "output": str(output / "summary.json")}) + "\n")
    print(json.dumps({"status": "pass", "output": str(output / "summary.json"), "elapsed_seconds": payload["elapsed_seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())

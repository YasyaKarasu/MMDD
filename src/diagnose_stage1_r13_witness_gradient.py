#!/usr/bin/env python
"""Measure R13 witness-loss gradient coverage on the frozen first path batch."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.row_support import load_evidence_content_keys
from mmdd_stage1.scoring import score_target_batch
from mmdd_stage1.training import student_gradient_norms
from mmdd_stage1.witness_supervision import witness_auxiliary_loss
from run_stage1_r13 import _model, _output_root, _path_schedule, _paths, freeze_plan


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    plan = freeze_plan(args.root)
    paths = _paths(args.root)
    batch = _path_schedule(args.root)[0]
    store = FeatureStore.from_path(paths["features"], cache_size=60_000)
    model = _model(args.root, "shared", device).train()
    scores = score_target_batch(
        model,
        batch,
        store,
        device,
        PathAggregator("logsumexp", 4, path_combination="sum"),
    )
    content_keys, content_keys_sha256 = load_evidence_content_keys(
        paths["evidence_content_keys"]
    )
    witness_loss, witness_stats = witness_auxiliary_loss(
        batch, scores, content_keys=content_keys
    )
    path_tensors = [
        tensor
        for example_tensors in scores.path_logits or ()
        for tensor in example_tensors
    ]
    path_gradients = torch.autograd.grad(
        witness_loss,
        path_tensors,
        retain_graph=True,
        allow_unused=True,
    )
    categories: dict[str, list[float]] = {
        "known_positive_witness": [],
        "unlabelled_positive_target_path": [],
        "weak_unknown_target_path": [],
    }
    tensor_offset = 0
    for example, example_tensors in zip(batch, scores.path_logits or ()):
        positives = set(example.positive_target_ids)
        known_by_target = example.positive_evidence_by_target or {}
        for candidate, tensor in zip(example.candidates, example_tensors):
            gradient = path_gradients[tensor_offset]
            tensor_offset += 1
            values = (
                [0.0] * tensor.numel()
                if gradient is None
                else gradient.detach().abs().cpu().tolist()
            )
            if candidate.target_id not in positives:
                categories["weak_unknown_target_path"].extend(values)
                continue
            known = set(known_by_target.get(candidate.target_id, ()))
            for evidence_id, value in zip(candidate.evidence_ids, values):
                key = (
                    "known_positive_witness"
                    if evidence_id in known
                    else "unlabelled_positive_target_path"
                )
                categories[key].append(float(value))
    model.zero_grad(set_to_none=True)
    witness_loss.backward()
    parameter_gradients = student_gradient_norms(model)
    category_summary = {
        name: {
            "paths": len(values),
            "nonzero_gradient_paths": sum(value > 0 for value in values),
            "nonzero_fraction": (
                sum(value > 0 for value in values) / len(values) if values else None
            ),
            "mean_absolute_gradient": (
                sum(values) / len(values) if values else None
            ),
            "maximum_absolute_gradient": max(values, default=None),
        }
        for name, values in categories.items()
    }
    payload = {
        "format_version": 1,
        "status": "complete",
        "identity": "diagnostic S0 forward/backward only; no optimizer step",
        "plan_sha256": plan["plan_sha256"],
        "batch": {
            "schedule": "frozen path batch 1 of 178",
            "queries": len(batch),
            "witness_stats": witness_stats,
        },
        "witness_loss": float(witness_loss.detach()),
        "weighted_witness_loss": float(0.1 * witness_loss.detach()),
        "path_gradient_coverage": category_summary,
        "parameter_gradient_norms": parameter_gradients,
        "weak_negative_semantics": {
            "weak_unknown_target_fraction_in_W_negative_denominator": 1.0,
            "confirmed_negative_fraction": 0.0,
            "used_for_confirmed_BCE": False,
            "note": (
                "All W contrast targets are ranking-only weak/unknown; the measured "
                "local gradients are not interpreted as confirmed invalid joins."
            ),
        },
        "all_finite": (
            math.isfinite(float(witness_loss.detach()))
            and all(
                value is None or math.isfinite(float(value))
                for summary in category_summary.values()
                for value in summary.values()
                if isinstance(value, float)
            )
        ),
        "content_keys_sha256": content_keys_sha256,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    target = (
        _output_root(args.root)
        / "taskD_witness_supervision/witness_gradient_diagnostic.json"
    )
    write_json(target, payload)
    with (_output_root(args.root) / "runs.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "task": "D witness gradient diagnostic",
                    "status": "complete",
                    "output": str(target.resolve()),
                    "command": payload["command"],
                }
            )
            + "\n"
        )
    print(json.dumps({"status": "complete", "output": str(target)}, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--cpu-threads", type=int, default=2)
    run(parser.parse_args())

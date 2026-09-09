#!/usr/bin/env python
"""Replay and extend the selected R12 C-base/C-candidates pair."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.data import load_edge_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.scoring import score_edge_batch
from mmdd_stage1.training import (
    _student_edge_losses,
    student_gradient_norms,
)
from run_stage1_r12_task_c import (
    _initialize_student,
    _optimizer,
    _ranking_scores,
    _save_checkpoint,
    _schedule_batches,
    _score_payload,
    _teacher_list_scores,
)


CHECKPOINT_STEPS = (659, 1318)


def _combined_scores(
    output_root: Path,
) -> tuple[torch.Tensor, dict, dict]:
    screen_scores, screen_manifest = _score_payload(output_root)
    extension_root = output_root / "taskC_training/teacher_extension_pair_scores"
    extension_manifest = json.loads(
        (extension_root / "manifest.json").read_text(encoding="utf-8")
    )
    if extension_manifest.get("status") != "complete":
        raise ValueError("Extension Teacher scores are incomplete")
    extension_path = Path(extension_manifest["scores"])
    if checkpoint_fingerprint(extension_path) != extension_manifest["scores_sha256"]:
        raise ValueError("Extension Teacher score fingerprint mismatch")
    payload = torch.load(extension_path, map_location="cpu", weights_only=True)
    extension_scores = payload.get("scores") if isinstance(payload, dict) else None
    if (
        not isinstance(extension_scores, torch.Tensor)
        or extension_scores.ndim != 1
        or extension_scores.shape[0] != int(extension_manifest["pair_count"])
        or not bool(torch.isfinite(extension_scores).all())
    ):
        raise ValueError("Extension Teacher score tensor is invalid")
    if int(extension_manifest["global_pair_id_start"]) != len(screen_scores):
        raise ValueError("Extension Teacher scores do not follow the screen prefix")
    if (
        extension_manifest["teacher_checkpoint_sha256"]
        != screen_manifest["teacher_checkpoint_sha256"]
    ):
        raise ValueError("Screen and extension Teacher checkpoints differ")
    return (
        torch.cat((screen_scores, extension_scores.float())),
        screen_manifest,
        extension_manifest,
    )


def _verify_screen_replay(
    model: torch.nn.Module, checkpoint_path: Path
) -> dict[str, float | int | str]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = payload["state_dict"]
    actual = model.state_dict()
    mismatches = []
    max_absolute_difference = 0.0
    for name, expected_value in expected.items():
        if name not in actual:
            mismatches.append(name)
            continue
        actual_value = actual[name].detach().cpu()
        if not torch.equal(actual_value, expected_value):
            mismatches.append(name)
            max_absolute_difference = max(
                max_absolute_difference,
                float((actual_value - expected_value).abs().max()),
            )
    if mismatches:
        raise ValueError(
            "Extension optimizer replay differs from the saved screen state: "
            + ", ".join(mismatches[:5])
        )
    return {
        "status": "exact",
        "tensors_checked": len(expected),
        "mismatches": 0,
        "max_absolute_difference": max_absolute_difference,
        "screen_checkpoint": str(checkpoint_path.resolve()),
        "screen_checkpoint_sha256": checkpoint_fingerprint(checkpoint_path),
    }


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    scores, teacher_manifest, extension_teacher_manifest = _combined_scores(
        args.output_root
    )
    screen_dir = args.output_root / "taskC_training/candidates_seed13_steps356"
    extension_dir = (
        args.output_root
        / "taskC_training/candidates_seed13_extension_steps357_1318"
    )
    screen_candidate_manifest = json.loads(
        (screen_dir / "manifest.json").read_text(encoding="utf-8")
    )
    extension_candidate_manifest = json.loads(
        (extension_dir / "manifest.json").read_text(encoding="utf-8")
    )
    schedule_paths = [
        screen_dir / f"{args.arm}.jsonl.gz",
        extension_dir / f"{args.arm}.jsonl.gz",
    ]
    if (
        checkpoint_fingerprint(schedule_paths[0])
        != screen_candidate_manifest["arms"][args.arm]["schedule_sha256"]
        or checkpoint_fingerprint(schedule_paths[1])
        != extension_candidate_manifest["arms"][args.arm]["schedule_sha256"]
    ):
        raise ValueError("Extension schedule fingerprint mismatch")
    output_dir = (
        args.output_root / "taskC_training" / f"c_{args.arm}_extension_seed13"
    )
    if (output_dir / "manifest.json").is_file():
        raise FileExistsError(f"Extension arm already has a manifest: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(13)
    torch.cuda.manual_seed_all(13)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    store = FeatureStore.from_path(
        args.root
        / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b",
        cache_size=60_000,
    )
    model = _initialize_student(args.root, device)
    optimizer = _optimizer(model)
    dev_edges = load_edge_examples(
        args.output_root / "taskA_correctness/supervision/edge_lists.dev.jsonl",
        split="dev",
    )
    teacher_sha256 = str(teacher_manifest["teacher_checkpoint_sha256"])
    screen_iterator = iter(
        _schedule_batches(schedule_paths[0], scores, teacher_sha256)
    )
    first_step, fixed_batch = next(screen_iterator)
    if first_step != 1:
        raise ValueError("Screen schedule must begin at update 1")
    batches = itertools.chain(
        ((first_step, fixed_batch),),
        screen_iterator,
        _schedule_batches(schedule_paths[1], scores, teacher_sha256),
    )
    checkpoints = {}
    history = []
    replay = None
    expected_step = 1
    last_gradient_norms = None
    for step, examples in batches:
        if step != expected_step:
            raise ValueError(
                f"Extension schedule step mismatch: expected {expected_step}, found {step}"
            )
        expected_step += 1
        model.train()
        raw = score_edge_batch(
            model, examples, store, device, student_score_space="raw_logit"
        )
        objective = _student_edge_losses(
            model,
            examples,
            raw,
            _teacher_list_scores(examples, raw, device),
            _ranking_scores(raw),
            None,
            ranking_weight=1.0,
            temperature=1.0,
            distillation_weight=0.3,
            edge_bce_weight=0.0,
            anchor_weight=0.1,
            anchor_weight_evidence=0.1,
            positive_loss_mode="sum_probability",
        )
        optimizer.zero_grad()
        objective["loss"].backward()
        last_gradient_norms = student_gradient_norms(model)
        optimizer.step()
        if step == 356:
            replay = _verify_screen_replay(
                model,
                args.output_root
                / f"taskC_training/c_{args.arm}_seed13/checkpoints/step_000356.pt",
            )
        if step > 356:
            history.append(
                {
                    "optimizer_updates": step,
                    "loss": float(objective["loss"].detach()),
                    "supervised_loss": float(objective["supervised_loss"].detach()),
                    "distillation_loss": float(
                        objective["distillation_loss"].detach()
                    ),
                    "parameter_anchor_loss": float(
                        objective["anchor_loss"].detach()
                    ),
                    "weighted_parameter_anchor_loss": float(
                        objective["weighted_anchor_loss"].detach()
                    ),
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
        if step % 100 == 0 or step in CHECKPOINT_STEPS:
            print(
                json.dumps(
                    {
                        "arm": args.arm,
                        "step": step,
                        "loss": float(objective["loss"].detach()),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    if (
        expected_step != 1319
        or replay is None
        or set(checkpoints) != set(CHECKPOINT_STEPS)
    ):
        raise ValueError("R12 extension did not reach every registered checkpoint")
    payload = {
        "format_version": 1,
        "status": "complete",
        "arm": f"C-{args.arm}",
        "seed": 13,
        "optimizer_updates_total": 1318,
        "optimizer_updates_replayed": 356,
        "optimizer_updates_new": 962,
        "batch_size": 64,
        "screen_replay": replay,
        "schedules": [str(path.resolve()) for path in schedule_paths],
        "schedule_sha256s": [
            checkpoint_fingerprint(path) for path in schedule_paths
        ],
        "screen_teacher_scores_sha256": teacher_manifest["scores_sha256"],
        "extension_teacher_scores_sha256": extension_teacher_manifest[
            "scores_sha256"
        ],
        "teacher_checkpoint_sha256": teacher_sha256,
        "checkpoints": checkpoints,
        "history_after_screen": history,
        "elapsed_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(output_dir / "manifest.json", payload)
    print(json.dumps({"status": "complete", "arm": payload["arm"]}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("base", "candidates"), required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

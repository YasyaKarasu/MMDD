#!/usr/bin/env python
"""Replay pruned historical path checkpoints, validating against archived hashes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_checkpoint, load_student
from mmdd_stage1.data import load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.teacher_logits import load_teacher_logits
from mmdd_stage1.training import checkpoint, train_student_paths
from train_stage1 import _student_optimizer


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    torch.set_num_threads(2)
    r11 = args.root / "work/stage1_optimization_r11_20260908"
    folder = r11 / ("taskD_controls/c2_d1_long" if args.arm == "d1_long" else f"taskC_clean/{args.arm}")
    output = args.output_root / "taskA_correctness/reconstructed" / args.arm
    output.mkdir(parents=True, exist_ok=True)
    if (output / "verification.json").exists():
        raise FileExistsError(output / "verification.json")
    manifest = json.loads((folder / "manifest.json").read_text())
    command = manifest["commands"]["path"]

    def option(flag, default=None):
        return command[command.index(flag) + 1] if flag in command else default

    device = torch.device(option("--device"))
    seed = int(option("--seed"))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    source = Path(option("--student-checkpoint"))
    student = load_student(source, device)
    # Deliberately reproduce the old stage-local anchor, not the R12 main protocol.
    with torch.no_grad():
        student.initial_projection_weights.copy_(student.stage_initial_projection_weights)
    student.projection_reference_origin = "historical_stage_anchor_replay_only"
    aggregator = PathAggregator("logsumexp", 4, temperature=1.0, path_combination="sum")
    examples = load_target_examples(Path(option("--base-data")), split="train")
    kd_weight = float(option("--distillation-weight"))
    teacher_cache = None
    if kd_weight:
        teacher = Path(option("--teacher-checkpoint"))
        examples, teacher_cache, hit = load_teacher_logits(
            examples, Path(option("--teacher-logit-cache")), checkpoint_fingerprint(teacher),
            aggregator, teacher_score_space="raw_logit",
        )
        if not hit:
            raise FileNotFoundError(f"Archived Teacher logits unavailable: {teacher_cache}")
    store = FeatureStore.from_path(Path(option("--features")), cache_size=100000)
    store.preload_embeddings({object_id for row in examples
                              for object_id in (row.query_id, *(candidate.target_id for candidate in row.candidates),
                                                *(value for candidate in row.candidates for value in candidate.evidence_ids))})
    optimizer = _student_optimizer(
        student, projection_learning_rate=float(option("--learning-rate")),
        relation_learning_rate=float(option("--relation-learning-rate")),
        weight_decay=float(option("--weight-decay")),
    )
    original_history = json.loads((folder / "student_path.pt.history.json").read_text())["epochs"]
    expected = {row["epoch"]: row["candidate_checkpoint_sha256"] for row in original_history}
    recovered = []

    def save(epoch, model, record):
        payload = checkpoint(model, "student-path", aggregator)
        payload.pop("projection_references")
        payload["state_dict"].pop("initial_projection_weights")
        payload["state_dict"].pop("stage_initial_projection_weights")
        target = output / f"epoch_{epoch:03d}.pt"
        temporary = target.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
        digest = checkpoint_fingerprint(target)
        recovered.append({"epoch": epoch, "checkpoint": str(target), "sha256": digest,
                          "historical_sha256": expected[epoch], "exact_file_match": digest == expected[epoch],
                          "optimizer_updates": record["cumulative_optimizer_updates"]})
        print(json.dumps(recovered[-1]), flush=True)
        return False

    history = train_student_paths(
        student, examples, store, optimizer, aggregator, device=device, epochs=int(option("--epochs")),
        batch_size=int(option("--batch-size")), seed=seed, temperature=float(option("--temperature")),
        distillation_weight=kd_weight, anchor_weight=float(option("--anchor-weight")),
        anchor_weight_evidence=float(option("--anchor-weight-evidence")),
        in_batch_negatives="--in-batch-negatives" in command,
        in_batch_max_negatives=int(option("--in-batch-max-negatives")),
        use_global_positive_mask="--global-positive-mask" in command,
        max_optimizer_updates=int(option("--max-optimizer-updates")),
        dataset_sampling_alpha=float(option("--dataset-sampling-alpha")),
        student_score_space=option("--student-score-space"),
        positive_loss_mode=option("--positive-loss-mode"), epoch_callback=save,
    )
    archived = load_checkpoint(folder / "student_path.last.pt")["state_dict"]
    parameter_comparison = {
        name: {"equal": torch.equal(parameter.detach().cpu(), archived[name]),
               "max_abs_difference": float((parameter.detach().cpu() - archived[name]).abs().max())}
        for name, parameter in student.named_parameters()
    }
    status = "pass" if all(row["exact_file_match"] for row in recovered) else "unverified_reconstruction"
    payload = {"status": status, "identity": "historical replay only, original stage-local P anchor",
               "original_command": command, "actual_command": [sys.executable, *sys.argv],
               "source_checkpoint": str(source), "source_checkpoint_sha256": checkpoint_fingerprint(source),
               "teacher_cache": str(teacher_cache) if teacher_cache else None, "recovered": recovered,
               "endpoint_parameter_comparison": parameter_comparison,
               "history": history, "elapsed_seconds": time.monotonic() - started,
               "code_sha256": checkpoint_fingerprint(Path(__file__))}
    write_json(output / "verification.json", payload)
    with (args.output_root / "runs.jsonl").open("a") as handle:
        handle.write(json.dumps({"task": "A4.2 historical checkpoint reconstruction", "arm": args.arm,
                                 "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "command": [sys.executable, *sys.argv], "status": status,
                                 "elapsed_seconds": payload["elapsed_seconds"],
                                 "output": str(output / "verification.json")}) + "\n")
    print(json.dumps({"status": status, "arm": args.arm, "output": str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("c1_long", "c2_long", "d1_long"), required=True)
    run(parser.parse_args())

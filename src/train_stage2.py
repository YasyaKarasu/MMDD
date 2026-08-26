#!/usr/bin/env python
"""Train the Stage-2 RATA candidate-column head with a frozen Qwen3.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmdd_stage2.checkpoints import save_candidate_scorer
from mmdd_stage2.qwen import QwenStage2Backend
from mmdd_stage2.training import load_column_training_data, train_candidate_scorer
from mmdd_stage2.verifier import CandidateColumnScorer


def run(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    examples, objects = load_column_training_data(
        Path(args.dataset_root),
        [Path(path) for path in args.retrieval_results],
        top_k_evidence=args.top_k_evidence,
        max_targets=args.max_targets,
    )
    backend = QwenStage2Backend(
        Path(args.model_dir),
        device=args.device,
        dtype=args.dtype,
    )
    device = backend.device
    scorer = CandidateColumnScorer(backend.hidden_dim).to(device)
    history = train_candidate_scorer(
        backend,
        scorer,
        examples,
        objects,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    save_candidate_scorer(
        Path(args.output),
        scorer,
        metadata={
            "model_dir": str(Path(args.model_dir).resolve()),
            "top_k_evidence": args.top_k_evidence,
            "max_targets": args.max_targets,
        },
    )
    history_path = Path(args.output).with_suffix(Path(args.output).suffix + ".history.json")
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "checkpoint": args.output,
                "history": str(history_path),
                "examples": len(examples),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--retrieval-results", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default="hf_models/Qwen3.5-9B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--top-k-evidence", type=int, default=4)
    parser.add_argument("--max-targets", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

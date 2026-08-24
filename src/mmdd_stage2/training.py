"""Frozen-Qwen training for the RATA candidate-column head."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from mmdd_dataset.wdc_runtime import iter_dataset_artifact

from .data import Stage2ObjectIndex, iter_retrieval_results, load_stage2_index, local_column_index
from .pipeline import Stage2Backend, Stage2Verifier
from .verifier import CandidateColumnScorer, EvidenceBundle, build_evidence_bundles


@dataclass(frozen=True)
class ColumnTrainingExample:
    query_id: str
    bundles: tuple[EvidenceBundle, ...]
    positive_target_id: str
    positive_source_column: int


def load_column_training_data(
    output_dir: Path,
    retrieval_paths: Sequence[Path],
    *,
    top_k_evidence: int,
    max_targets: int,
) -> tuple[list[ColumnTrainingExample], Stage2ObjectIndex]:
    qrels = {
        str(record["query_table_id"]): record
        for record in iter_dataset_artifact(output_dir, "qrels")
        if record.get("reason") == "model_recoverable_join_column"
    }
    examples = []
    query_ids: set[str] = set()
    target_ids: set[str] = set()
    evidence_ids: set[str] = set()
    for path in retrieval_paths:
        for record in iter_retrieval_results(path):
            query_id = str(record["query_id"])
            qrel = qrels.get(query_id)
            if qrel is None:
                continue
            bundles = build_evidence_bundles(record["results"][:max_targets], top_k_evidence=top_k_evidence)
            positive_target = str(qrel.get("target_table_id", qrel.get("data_lake_table_id")))
            if positive_target not in {bundle.target_id for bundle in bundles}:
                continue
            example = ColumnTrainingExample(
                query_id=query_id,
                bundles=tuple(bundles),
                positive_target_id=positive_target,
                positive_source_column=int(qrel["join_attribute"]["source_column_index"]),
            )
            examples.append(example)
            query_ids.add(query_id)
            target_ids.update(bundle.target_id for bundle in bundles)
            evidence_ids.update(evidence_id for bundle in bundles for evidence_id in bundle.evidence_ids)
    if not examples:
        raise ValueError("No Stage-2 training examples have a retrieved positive evidence path")
    return examples, load_stage2_index(
        output_dir,
        query_ids=query_ids,
        target_ids=target_ids,
        evidence_ids=evidence_ids,
    )


def train_candidate_scorer(
    backend: Stage2Backend,
    scorer: CandidateColumnScorer,
    examples: Sequence[ColumnTrainingExample],
    objects: Stage2ObjectIndex,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=learning_rate, weight_decay=weight_decay)
    verifier = Stage2Verifier(backend, scorer)
    device = scorer.weight.weight.device
    order = list(range(len(examples)))
    generator = random.Random(seed)
    history = []
    scorer.train()
    for epoch in range(epochs):
        generator.shuffle(order)
        total_loss = 0.0
        for example_index in order:
            example = examples[example_index]
            logits = verifier.candidate_logits(
                objects.queries[example.query_id],
                example.bundles,
                objects.targets,
                objects.evidence,
            )
            target_position = next(
                index for index, bundle in enumerate(example.bundles) if bundle.target_id == example.positive_target_id
            )
            target = objects.targets[example.positive_target_id]
            local_index = local_column_index(target, example.positive_source_column)
            column_position = next(
                index for index, column in enumerate(target["columns"]) if int(column["column_index"]) == local_index
            )
            table_scores = torch.tensor(
                [bundle.retrieval_score for bundle in example.bundles], device=device, dtype=torch.float32
            )
            loss = -torch.log_softmax(table_scores, dim=0)[target_position]
            loss = loss - torch.log_softmax(logits[target_position], dim=0)[column_position]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        history.append({"epoch": epoch + 1, "loss": total_loss / len(examples), "examples": len(examples)})
    return history

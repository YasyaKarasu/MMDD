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
            positive_target = str(qrel["target_table_id"])
            positive_bundle = next((bundle for bundle in bundles if bundle.target_id == positive_target), None)
            if positive_bundle is None:
                continue
            example = ColumnTrainingExample(
                query_id=query_id,
                bundles=tuple(bundles),
                positive_target_id=positive_target,
                positive_source_column=int(qrel["join_attribute"]["source_column_index"]),
            )
            examples.append(example)
            query_ids.add(query_id)
            target_ids.add(positive_target)
            evidence_ids.update(positive_bundle.evidence_ids)
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
    order = list(range(len(examples)))
    generator = random.Random(seed)
    history = []
    scorer.train()
    for epoch in range(epochs):
        generator.shuffle(order)
        total_column_loss = 0.0
        total_table_loss = 0.0
        for example_index in order:
            example = examples[example_index]
            target_position = next(
                index for index, bundle in enumerate(example.bundles) if bundle.target_id == example.positive_target_id
            )
            positive_bundle = example.bundles[target_position]
            logits = verifier.candidate_logits(
                objects.queries[example.query_id],
                (positive_bundle,),
                objects.targets,
                objects.evidence,
            )[0]
            target = objects.targets[example.positive_target_id]
            local_index = local_column_index(target, example.positive_source_column)
            column_position = next(
                index for index, column in enumerate(target["columns"]) if int(column["column_index"]) == local_index
            )
            table_scores = torch.tensor(
                [bundle.retrieval_score for bundle in example.bundles], dtype=torch.float32
            )
            table_loss = -torch.log_softmax(table_scores, dim=0)[target_position]
            column_loss = -torch.log_softmax(logits, dim=0)[column_position]
            optimizer.zero_grad()
            column_loss.backward()
            optimizer.step()
            total_column_loss += float(column_loss.detach())
            total_table_loss += float(table_loss.detach())
        mean_column_loss = total_column_loss / len(examples)
        mean_table_loss = total_table_loss / len(examples)
        history.append(
            {
                "epoch": epoch + 1,
                "column_loss": mean_column_loss,
                "table_loss": mean_table_loss,
            }
        )
    return history

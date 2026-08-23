"""End-to-end Stage-2 verification over Stage-1 retrieval results."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

import torch

from .data import column_name, column_values, local_column_index, row_values
from .verifier import (
    CandidateColumnScorer,
    EvidenceBundle,
    SemanticJoinability,
    joint_candidate_probabilities,
    semantic_joinability,
)


@dataclass
class LocalizedEvidence:
    evidence_id: str
    evidence_type: str
    relevance: float
    text: str | None = None
    image: Any | None = None
    box: tuple[float, float, float, float] | None = None

    def record(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "relevance": self.relevance,
            "text": self.text,
            "box": self.box,
        }


@dataclass(frozen=True)
class ColumnSelection:
    target_id: str
    column_index: int
    column_name: str
    table_probability: float
    column_probability: float
    joint_probability: float


@dataclass(frozen=True)
class DirectVerification:
    target_id: str
    query_column: int
    target_column: int
    result: SemanticJoinability


@dataclass(frozen=True)
class RowPrediction:
    row_id: int
    entity: str
    attribute: str
    value: str
    evidence: dict[str, Any] | None


@dataclass(frozen=True)
class Stage2Result:
    query_id: str
    direct_candidates: tuple[DirectVerification, ...]
    selection: ColumnSelection | None
    rows: tuple[RowPrediction, ...]
    augmented_query: dict[str, Any] | None
    semantic_joinability: SemanticJoinability | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Stage2Backend(Protocol):
    hidden_dim: int

    def reader_states(
        self,
        query: dict[str, Any],
        target: dict[str, Any],
        evidence: Sequence[tuple[dict[str, Any], float]],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def localize_evidence(
        self,
        row: dict[str, str],
        *,
        entity_column: str,
        attribute_name: str,
        evidence: dict[str, Any],
    ) -> LocalizedEvidence: ...

    def generate_value(
        self,
        row: dict[str, str],
        *,
        attribute_name: str,
        evidence: LocalizedEvidence,
    ) -> str: ...

    def embed_texts(self, values: Sequence[str]) -> torch.Tensor: ...


class Stage2Verifier:
    def __init__(
        self,
        backend: Stage2Backend,
        scorer: CandidateColumnScorer,
        *,
        similarity_threshold: float = 0.8,
        min_row_coverage: float = 0.6,
    ) -> None:
        if scorer.weight.in_features != backend.hidden_dim * 2:
            raise ValueError("Candidate scorer and Stage-2 backend dimensions disagree")
        self.backend = backend
        self.scorer = scorer
        self.similarity_threshold = similarity_threshold
        self.min_row_coverage = min_row_coverage

    def candidate_logits(
        self,
        query: dict[str, Any],
        bundles: Sequence[EvidenceBundle],
        targets: dict[str, dict[str, Any]],
        evidence: dict[str, dict[str, Any]],
    ) -> list[torch.Tensor]:
        device = self.scorer.weight.weight.device
        logits = []
        for bundle in bundles:
            target = targets[bundle.target_id]
            weighted_evidence = [(evidence[item.evidence_id], item.weight) for item in bundle.evidence]
            open_states, close_states = self.backend.reader_states(query, target, weighted_evidence)
            if open_states.shape[0] != len(target["columns"]):
                raise ValueError(f"{bundle.target_id}: reader did not return one marker pair per column")
            logits.append(self.scorer(open_states.to(device), close_states.to(device)))
        return logits

    def select_column(
        self,
        query: dict[str, Any],
        bundles: Sequence[EvidenceBundle],
        targets: dict[str, dict[str, Any]],
        evidence: dict[str, dict[str, Any]],
    ) -> ColumnSelection:
        logits = self.candidate_logits(query, bundles, targets, evidence)
        device = logits[0].device
        max_columns = max(values.shape[0] for values in logits)
        column_logits = torch.full((1, len(logits), max_columns), -torch.inf, device=device)
        column_mask = torch.zeros_like(column_logits, dtype=torch.bool)
        for target_index, values in enumerate(logits):
            column_logits[0, target_index, : values.shape[0]] = values
            column_mask[0, target_index, : values.shape[0]] = True
        retrieval_scores = torch.tensor([[bundle.retrieval_score for bundle in bundles]], device=device)
        target_mask = torch.ones_like(retrieval_scores, dtype=torch.bool)
        table_probabilities, column_probabilities, joint = joint_candidate_probabilities(
            retrieval_scores, column_logits, target_mask, column_mask
        )
        flat_index = int(joint.reshape(-1).argmax())
        target_index, column_position = divmod(flat_index, max_columns)
        target = targets[bundles[target_index].target_id]
        column_index = int(target["columns"][column_position]["column_index"])
        return ColumnSelection(
            target_id=bundles[target_index].target_id,
            column_index=column_index,
            column_name=column_name(target, column_index),
            table_probability=float(table_probabilities[0, target_index].detach()),
            column_probability=float(column_probabilities[0, target_index, column_position].detach()),
            joint_probability=float(joint[0, target_index, column_position].detach()),
        )

    def _semantic_check(self, query_values: Sequence[str], target_values: Sequence[str]) -> SemanticJoinability:
        embeddings = self.backend.embed_texts([*query_values, *target_values])
        return semantic_joinability(
            query_values,
            target_values,
            query_embeddings=embeddings[: len(query_values)],
            target_embeddings=embeddings[len(query_values) :],
            similarity_threshold=self.similarity_threshold,
            min_coverage=self.min_row_coverage,
        )

    def verify_direct(
        self,
        query: dict[str, Any],
        targets: dict[str, dict[str, Any]],
        target_ids: Sequence[str],
    ) -> tuple[DirectVerification, ...]:
        verified = []
        for target_id in target_ids:
            target = targets[target_id]
            candidates = []
            for query_column in query["columns"]:
                query_index = int(query_column["column_index"])
                query_column_values = column_values(query, query_index)
                for target_column in target["columns"]:
                    target_index = int(target_column["column_index"])
                    result = self._semantic_check(query_column_values, column_values(target, target_index))
                    candidates.append((result.coverage, result.mean_similarity, query_index, target_index, result))
            _, _, query_index, target_index, result = max(candidates, key=lambda item: item[:2])
            if result.joinable:
                verified.append(DirectVerification(target_id, query_index, target_index, result))
        return tuple(verified)

    def verify(
        self,
        query: dict[str, Any],
        bundles: Sequence[EvidenceBundle],
        targets: dict[str, dict[str, Any]],
        evidence: dict[str, dict[str, Any]],
        *,
        direct_target_ids: Sequence[str] = (),
    ) -> Stage2Result:
        query_id = str(query.get("table_id", query.get("object_id")))
        direct = self.verify_direct(query, targets, direct_target_ids)
        if not bundles:
            return Stage2Result(query_id, direct, None, (), None, None)

        selection = self.select_column(query, bundles, targets, evidence)
        selected_bundle = next(bundle for bundle in bundles if bundle.target_id == selection.target_id)
        if "query_entity_col" in query:
            entity_index = local_column_index(query, int(query["query_entity_col"]))
        else:
            entity_index = int(query["columns"][0]["column_index"])
        entity_name = column_name(query, entity_index)
        predictions = []
        for row in query["rows"]:
            visible_row = row_values(query, row)
            localized = []
            for reference in selected_bundle.evidence:
                item = self.backend.localize_evidence(
                    visible_row,
                    entity_column=entity_name,
                    attribute_name=selection.column_name,
                    evidence=evidence[reference.evidence_id],
                )
                item.relevance *= reference.weight
                localized.append(item)
            best = max(localized, key=lambda item: item.relevance)
            value = self.backend.generate_value(
                visible_row,
                attribute_name=selection.column_name,
                evidence=best,
            )
            predictions.append(
                RowPrediction(
                    row_id=int(row["row_id"]),
                    entity=visible_row[entity_name],
                    attribute=selection.column_name,
                    value=value,
                    evidence=best.record(),
                )
            )

        prediction_by_row = {prediction.row_id: prediction.value for prediction in predictions}
        augmented_query = deepcopy(query)
        generated_index = max(int(column["column_index"]) for column in query["columns"]) + 1
        augmented_query["columns"].append(
            {
                "column_index": generated_index,
                "column_name": selection.column_name,
                "generated": True,
                "source_target_id": selection.target_id,
                "source_target_column_index": selection.column_index,
            }
        )
        for row in augmented_query["rows"]:
            row["cells"].append(
                {
                    "column_index": generated_index,
                    "column_name": selection.column_name,
                    "text": prediction_by_row[int(row["row_id"])],
                    "generated": True,
                }
            )
        generated_values = column_values(augmented_query, generated_index, include_empty=True)
        target_values = column_values(targets[selection.target_id], selection.column_index)
        check = self._semantic_check(generated_values, target_values)
        return Stage2Result(query_id, direct, selection, tuple(predictions), augmented_query, check)

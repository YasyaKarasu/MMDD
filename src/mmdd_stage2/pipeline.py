"""End-to-end Stage-2 verification over Stage-1 retrieval results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from mmdd_progress import progress

import torch

from .data import column_name, column_values, row_values
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
    text: str | None = None
    image: Any | None = None
    box: tuple[float, float, float, float] | None = None
    text_span_relevance: float | None = None
    image_presence_probability: float | None = None

    def record(self) -> dict[str, Any]:
        record = {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
        }
        if self.evidence_type == "text":
            if self.text_span_relevance is None:
                raise ValueError("Localized text evidence requires text_span_relevance")
            record["text_span"] = self.text or ""
            record["text_span_relevance"] = self.text_span_relevance
        elif self.evidence_type == "image":
            if self.image_presence_probability is None:
                raise ValueError("Localized image evidence requires image_presence_probability")
            record["image_presence_probability"] = self.image_presence_probability
            if self.box is not None:
                record["image_box"] = self.box
        else:
            raise ValueError(f"Unsupported localized evidence type: {self.evidence_type!r}")
        return record


@dataclass(frozen=True)
class ColumnSelection:
    target_id: str
    column_index: int
    column_name: str


@dataclass(frozen=True)
class DirectVerification:
    target_id: str
    query_column: int
    target_column: int


@dataclass(frozen=True)
class RowPrediction:
    row_id: int
    value: str
    evidence: dict[str, Any] | None


@dataclass(frozen=True)
class Stage2Result:
    query_id: str
    direct_candidates: tuple[DirectVerification, ...]
    selection: ColumnSelection | None
    rows: tuple[RowPrediction, ...]
    semantic_joinability: SemanticJoinability | None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"query_id": self.query_id}
        if self.direct_candidates:
            payload["direct_matches"] = [
                {
                    "target_id": candidate.target_id,
                    "query_column": candidate.query_column,
                    "target_column": candidate.target_column,
                }
                for candidate in self.direct_candidates
            ]
        if self.selection is not None:
            payload["selection"] = {
                "target_id": self.selection.target_id,
                "column_index": self.selection.column_index,
                "column_name": self.selection.column_name,
            }
        payload["rows"] = [
            {
                "row_id": row.row_id,
                "value": row.value,
                **({"evidence": row.evidence} if row.evidence is not None else {}),
            }
            for row in self.rows
        ]
        if self.semantic_joinability is not None:
            payload["verification"] = {
                "joinable": self.semantic_joinability.joinable,
                "coverage": self.semantic_joinability.coverage,
                "mean_similarity": self.semantic_joinability.mean_similarity,
            }
        return payload


class Stage2Backend(Protocol):
    hidden_dim: int

    def reader_states(
        self,
        query: dict[str, Any],
        target: dict[str, Any],
        evidence: Sequence[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def localize_evidence(
        self,
        row: dict[str, str],
        *,
        attribute_name: str,
        evidence: dict[str, Any],
    ) -> LocalizedEvidence: ...

    def evidence_logits(
        self,
        row: dict[str, str],
        *,
        attribute_name: str,
        candidates: Sequence[LocalizedEvidence],
    ) -> torch.Tensor: ...

    def generate_value(
        self,
        row: dict[str, str],
        *,
        attribute_name: str,
        evidence: LocalizedEvidence,
    ) -> str: ...

    def embed_texts(self, values: Sequence[str]) -> torch.Tensor: ...


class EvidenceRowRouter(Protocol):
    def assign(
        self,
        query_id: str,
        evidence_ids: Sequence[str],
        *,
        row_count: int,
    ) -> dict[str, int]: ...


class Stage2Verifier:
    def __init__(
        self,
        backend: Stage2Backend,
        scorer: CandidateColumnScorer,
        *,
        evidence_router: EvidenceRowRouter | None = None,
        similarity_threshold: float = 0.8,
        min_row_coverage: float = 0.6,
        similarity_batch_size: int = 1024,
    ) -> None:
        if scorer.weight.in_features != backend.hidden_dim * 2:
            raise ValueError("Candidate scorer and Stage-2 backend dimensions disagree")
        if similarity_batch_size <= 0:
            raise ValueError("similarity_batch_size must be positive")
        self.backend = backend
        self.scorer = scorer
        self.evidence_router = evidence_router
        self.similarity_threshold = similarity_threshold
        self.min_row_coverage = min_row_coverage
        self.similarity_batch_size = similarity_batch_size

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
            selected_evidence = [evidence[evidence_id] for evidence_id in bundle.evidence_ids]
            open_states, close_states = self.backend.reader_states(query, target, selected_evidence)
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
        column_logits = torch.full((len(logits), max_columns), -torch.inf, device=device)
        column_mask = torch.zeros_like(column_logits, dtype=torch.bool)
        for target_index, values in enumerate(logits):
            column_logits[target_index, : values.shape[0]] = values
            column_mask[target_index, : values.shape[0]] = True
        retrieval_scores = torch.tensor(
            [bundle.retrieval_score for bundle in bundles], device=device
        )
        joint = joint_candidate_probabilities(
            retrieval_scores, column_logits, column_mask
        )
        flat_index = int(joint.reshape(-1).argmax())
        target_index, column_position = divmod(flat_index, max_columns)
        target = targets[bundles[target_index].target_id]
        column_index = int(target["columns"][column_position]["column_index"])
        return ColumnSelection(
            target_id=bundles[target_index].target_id,
            column_index=column_index,
            column_name=column_name(target, column_index),
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
            similarity_batch_size=self.similarity_batch_size,
        )

    def verify_direct(
        self,
        query: dict[str, Any],
        targets: dict[str, dict[str, Any]],
        target_ids: Sequence[str],
    ) -> tuple[DirectVerification, ...]:
        if not target_ids:
            return ()
        query_columns = [
            (
                int(column["column_index"]),
                column_values(query, int(column["column_index"])),
            )
            for column in query["columns"]
        ]
        target_columns = {
            target_id: [
                (
                    int(column["column_index"]),
                    column_values(
                        targets[target_id], int(column["column_index"])
                    ),
                )
                for column in targets[target_id]["columns"]
            ]
            for target_id in target_ids
        }
        ordered_columns = [values for _, values in query_columns]
        ordered_columns.extend(
            values
            for target_id in target_ids
            for _, values in target_columns[target_id]
        )
        embeddings = iter(
            self.backend.embed_texts(
                [value for values in ordered_columns for value in values]
            ).split([len(values) for values in ordered_columns])
        )
        query_embeddings = [next(embeddings) for _ in query_columns]
        embedded_targets = {
            target_id: [
                (column_index, values, next(embeddings))
                for column_index, values in target_columns[target_id]
            ]
            for target_id in target_ids
        }

        verified = []
        for target_id in progress(
            target_ids,
            desc="Verify direct targets",
            unit="target",
            leave=False,
        ):
            candidates = []
            for (query_index, query_values), query_vectors in zip(
                query_columns, query_embeddings
            ):
                for target_index, target_values, target_vectors in embedded_targets[
                    target_id
                ]:
                    result = semantic_joinability(
                        query_values,
                        target_values,
                        query_embeddings=query_vectors,
                        target_embeddings=target_vectors,
                        similarity_threshold=self.similarity_threshold,
                        min_coverage=self.min_row_coverage,
                        similarity_batch_size=self.similarity_batch_size,
                    )
                    candidates.append(
                        (
                            result.coverage,
                            result.mean_similarity,
                            query_index,
                            target_index,
                            result,
                        )
                    )
            _, _, query_index, target_index, result = max(
                candidates, key=lambda item: item[:2]
            )
            if result.joinable:
                verified.append(
                    DirectVerification(target_id, query_index, target_index)
                )
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
        query_id = str(query["table_id"])
        direct = self.verify_direct(query, targets, direct_target_ids)
        if not bundles:
            return Stage2Result(query_id, direct, None, (), None)

        selection = self.select_column(query, bundles, targets, evidence)
        selected_bundle = next(bundle for bundle in bundles if bundle.target_id == selection.target_id)
        if self.evidence_router is None:
            raise ValueError("Stage-2 row filling requires an evidence router")
        assignment_by_evidence = self.evidence_router.assign(
            query_id,
            selected_bundle.evidence_ids,
            row_count=len(query["rows"]),
        )
        if set(assignment_by_evidence) != set(selected_bundle.evidence_ids):
            raise ValueError("Evidence router must assign every selected evidence exactly once")
        evidence_by_row: list[list[str]] = [[] for _ in query["rows"]]
        for evidence_id in selected_bundle.evidence_ids:
            row_position = assignment_by_evidence[evidence_id]
            if not 0 <= row_position < len(query["rows"]):
                raise ValueError(
                    f"Evidence router returned invalid row position {row_position}"
                )
            evidence_by_row[row_position].append(evidence_id)

        predictions = []
        rows = enumerate(query["rows"])
        for row_position, row in progress(
            rows,
            total=len(query["rows"]),
            desc="Stage-2 row filling",
            unit="row",
            leave=False,
        ):
            visible_row = row_values(query, row)
            routed = evidence_by_row[row_position]
            if not routed:
                predictions.append(
                    RowPrediction(
                        row_id=int(row["row_id"]),
                        value="",
                        evidence=None,
                    )
                )
                continue

            localized = []
            for evidence_id in routed:
                item = self.backend.localize_evidence(
                    visible_row,
                    attribute_name=selection.column_name,
                    evidence=evidence[evidence_id],
                )
                localized.append(item)
            if len(localized) == 1:
                best = localized[0]
            else:
                evidence_logits = self.backend.evidence_logits(
                    visible_row,
                    attribute_name=selection.column_name,
                    candidates=localized,
                )
                if evidence_logits.shape != (len(localized),):
                    raise ValueError("Evidence reranker must return one logit per localized candidate")
                best = localized[int(evidence_logits.argmax())]
            value = self.backend.generate_value(
                visible_row,
                attribute_name=selection.column_name,
                evidence=best,
            )
            evidence_record = best.record()
            predictions.append(
                RowPrediction(
                    row_id=int(row["row_id"]),
                    value=value,
                    evidence=evidence_record,
                )
            )

        generated_values = [prediction.value for prediction in predictions]
        target_values = column_values(targets[selection.target_id], selection.column_index)
        check = self._semantic_check(generated_values, target_values)
        return Stage2Result(query_id, direct, selection, tuple(predictions), check)

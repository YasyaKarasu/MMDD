"""End-to-end Stage-2 verification over Stage-1 retrieval results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol

from mmdd_progress import progress

import torch

from .data import (
    column_name,
    column_values,
    direct_target_ids,
    permute_table_columns,
    row_values,
    validate_retrieval_path_budget,
)
from .verifier import (
    CandidateColumnScorer,
    EvidenceBundle,
    SemanticJoinability,
    build_evidence_bundles,
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
    semantic_joinability: SemanticJoinability


@dataclass(frozen=True)
class RowPrediction:
    row_id: int
    value: str
    evidence: dict[str, Any] | None


@dataclass(frozen=True)
class EvidenceVerification:
    selection: ColumnSelection
    rows: tuple[RowPrediction, ...]
    semantic_joinability: SemanticJoinability


@dataclass(frozen=True)
class CandidateScores:
    selection: ColumnSelection
    column_indices: tuple[int, ...]
    joint_probabilities: tuple[float, ...]
    table_probability: float
    recovery_priority: float
    accepted: bool = True
    acceptance_probability: float | None = None


def joinability_sort_key(
    check: SemanticJoinability, stage1_rank: int = 0
) -> tuple[float, float, int]:
    return -check.coverage, -check.mean_similarity, stage1_rank


@dataclass(frozen=True)
class CandidateResult:
    target_id: str
    stage1_rank: int
    stage1_score: float | None
    table_prior: float | None
    bundle: EvidenceBundle | None
    scores: CandidateScores | None
    selected_for_recovery: bool
    direct: DirectVerification | None
    evidence: EvidenceVerification | None
    rerank_rank: int | None = None

    @property
    def final_branch(self) -> str | None:
        branches = {
            name: result
            for name, result in (("direct", self.direct), ("evidence", self.evidence))
            if result is not None
        }
        # An exact branch tie keeps direct; neither branch is normalized separately.
        return min(
            branches,
            key=lambda name: joinability_sort_key(branches[name].semantic_joinability),
            default=None,
        )

    @property
    def semantic_joinability(self) -> SemanticJoinability | None:
        branch = self.final_branch
        result = self.direct if branch == "direct" else self.evidence
        return result.semantic_joinability if result is not None else None

    def to_dict(self) -> dict[str, Any]:
        branches: dict[str, Any] = {}
        if self.direct is not None:
            branches["direct"] = {
                "status": "verified",
                "query_column": self.direct.query_column,
                "target_column": self.direct.target_column,
                "verification": asdict(self.direct.semantic_joinability),
            }
        if self.bundle is not None:
            if self.evidence is not None:
                not_attempted_reason = None
            elif self.scores is not None and not self.scores.accepted:
                not_attempted_reason = "column_rejected"
            else:
                not_attempted_reason = "recovery_budget"
            branches["evidence"] = {
                "status": "verified" if self.evidence is not None else "not_attempted",
                "not_attempted_reason": not_attempted_reason,
                "evidence_ids": list(self.bundle.evidence_ids),
                "verification": asdict(self.evidence.semantic_joinability) if self.evidence else None,
                "rows": [asdict(row) for row in self.evidence.rows] if self.evidence else [],
            }
        check = self.semantic_joinability
        return {
            "target_id": self.target_id,
            "stage1_rank": self.stage1_rank,
            "stage1_score": self.stage1_score,
            "table_prior": self.table_prior,
            "table_probability": self.scores.table_probability if self.scores else None,
            "selection": asdict(self.scores.selection) if self.scores else None,
            "joint_probabilities": [
                {"column_index": column, "probability": probability}
                for column, probability in zip(
                    self.scores.column_indices, self.scores.joint_probabilities, strict=True
                )
            ] if self.scores else [],
            "recovery_priority": self.scores.recovery_priority if self.scores else None,
            "column_accepted": self.scores.accepted if self.scores else None,
            "column_acceptance_probability": (
                self.scores.acceptance_probability if self.scores else None
            ),
            "selected_for_recovery": self.selected_for_recovery,
            "branches": branches,
            "status": "verified" if check is not None else "not_attempted",
            "final_branch": self.final_branch,
            "verification": asdict(check) if check is not None else None,
            "rerank_rank": self.rerank_rank,
        }


@dataclass(frozen=True)
class Stage2Result:
    query_id: str
    recovery_budget: int
    reranked_candidates: tuple[CandidateResult, ...]
    unattempted_candidates: tuple[CandidateResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "input_candidate_count": len(self.reranked_candidates) + len(self.unattempted_candidates),
            "recovery_budget": self.recovery_budget,
            "reranked_candidates": [candidate.to_dict() for candidate in self.reranked_candidates],
            "unattempted_candidates": [candidate.to_dict() for candidate in self.unattempted_candidates],
        }


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
        column_permutation_seed: int | None = None,
        column_rejection_threshold: float | None = None,
    ) -> None:
        if scorer.input_dim != backend.hidden_dim * 2:
            raise ValueError("Candidate scorer and Stage-2 backend dimensions disagree")
        if getattr(scorer, "reader_layout_version", "header_markers_v0") != getattr(backend, "reader_layout_version", "header_markers_v0"):
            raise ValueError("Candidate scorer and Stage-2 backend reader layouts disagree")
        if similarity_batch_size <= 0:
            raise ValueError("similarity_batch_size must be positive")
        if column_rejection_threshold is not None and not 0.0 <= column_rejection_threshold <= 1.0:
            raise ValueError("column_rejection_threshold must be between zero and one")
        self.backend = backend
        self.scorer = scorer
        self.evidence_router = evidence_router
        self.similarity_threshold = similarity_threshold
        self.min_row_coverage = min_row_coverage
        self.similarity_batch_size = similarity_batch_size
        self.column_permutation_seed = column_permutation_seed
        self.column_rejection_threshold = column_rejection_threshold

    def _reader_target(self, target: dict[str, Any]) -> dict[str, Any]:
        if self.column_permutation_seed is None:
            return target
        return permute_table_columns(target, seed=self.column_permutation_seed)

    def candidate_logits(
        self,
        query: dict[str, Any],
        bundles: Sequence[EvidenceBundle],
        targets: dict[str, dict[str, Any]],
        evidence: dict[str, dict[str, Any]],
    ) -> list[torch.Tensor]:
        device = next(self.scorer.parameters()).device
        requests = []
        for bundle in bundles:
            target = self._reader_target(targets[bundle.target_id])
            selected_evidence = [evidence[evidence_id] for evidence_id in bundle.evidence_ids]
            requests.append((query, target, selected_evidence))
        batch_reader = getattr(self.backend, "reader_states_batch", None)
        if callable(batch_reader):
            batch_size = int(getattr(self.backend, "reader_batch_size", len(requests)))
            if batch_size <= 0:
                raise ValueError("reader_batch_size must be positive")
            states = []
            for start in range(0, len(requests), batch_size):
                states.extend(batch_reader(requests[start : start + batch_size]))
        else:
            states = [self.backend.reader_states(*request) for request in requests]
        logits = []
        for bundle, (_query, target, _evidence), (open_states, close_states) in zip(
            bundles, requests, states, strict=True
        ):
            if open_states.shape[0] != len(target["columns"]):
                raise ValueError(f"{bundle.target_id}: reader did not return one marker pair per column")
            logits.append(self.scorer(open_states.to(device), close_states.to(device)))
        return logits

    @torch.no_grad()
    def score_candidates(
        self,
        query: dict[str, Any],
        bundles: Sequence[EvidenceBundle],
        targets: dict[str, dict[str, Any]],
        evidence: dict[str, dict[str, Any]],
    ) -> tuple[CandidateScores, ...]:
        """Score the entire evidence pool before applying any recovery budget."""

        if not bundles:
            return ()
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
        table_probabilities = retrieval_scores.softmax(dim=-1)
        scored = []
        for target_index, bundle in enumerate(bundles):
            target = self._reader_target(targets[bundle.target_id])
            column_indices = tuple(int(column["column_index"]) for column in target["columns"])
            probabilities = joint[target_index, : len(column_indices)]
            column_position = int(probabilities.argmax())
            column_index = column_indices[column_position]
            acceptance_probability = float(torch.sigmoid(logits[target_index].max()))
            accepted = (
                self.column_rejection_threshold is None
                or acceptance_probability >= self.column_rejection_threshold
            )
            scored.append(
                CandidateScores(
                    selection=ColumnSelection(
                        bundle.target_id, column_index, column_name(target, column_index)
                    ),
                    column_indices=column_indices,
                    joint_probabilities=tuple(probabilities.tolist()),
                    table_probability=float(table_probabilities[target_index]),
                    recovery_priority=float(probabilities[column_position]),
                    accepted=accepted,
                    acceptance_probability=acceptance_probability,
                )
            )
        return tuple(scored)

    def select_column(
        self,
        query: dict[str, Any],
        bundles: Sequence[EvidenceBundle],
        targets: dict[str, dict[str, Any]],
        evidence: dict[str, dict[str, Any]],
    ) -> ColumnSelection:
        scores = self.score_candidates(query, bundles, targets, evidence)
        return max(scores, key=lambda item: item.recovery_priority).selection

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
                column_values(query, int(column["column_index"]), include_empty=True),
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
            verified.append(
                DirectVerification(target_id, query_index, target_index, result)
            )
        return tuple(verified)

    @torch.no_grad()
    def verify(
        self,
        query: dict[str, Any],
        retrieval_results: Sequence[dict[str, Any]],
        targets: dict[str, dict[str, Any]],
        evidence: dict[str, dict[str, Any]],
        *,
        recovery_budget: int = 20,
        top_k_evidence: int = 4,
    ) -> Stage2Result:
        """Rerank one complete Top-N path pool using direct and budgeted evidence checks."""

        if recovery_budget < 0:
            raise ValueError("recovery_budget must be non-negative")
        if top_k_evidence <= 0:
            raise ValueError("top_k_evidence must be positive")
        validate_retrieval_path_budget(
            {"results": retrieval_results},
            max_targets=len(retrieval_results),
            top_k_evidence=top_k_evidence,
        )
        bundles = build_evidence_bundles(retrieval_results, top_k_evidence=top_k_evidence)
        scores = self.score_candidates(query, bundles, targets, evidence)
        # Each table contributes exactly one maximum. Stable sorting preserves Stage-1 ties.
        selected_ids = {
            item.selection.target_id
            for item in sorted(
                (item for item in scores if item.accepted),
                key=lambda item: -item.recovery_priority,
            )[:recovery_budget]
        }
        direct = {
            result.target_id: result
            for result in self.verify_direct(query, targets, direct_target_ids(retrieval_results))
        }
        recovered = {
            bundle.target_id: self.recover_candidate(query, bundle, item.selection, targets, evidence)
            for bundle, item in zip(bundles, scores, strict=True)
            if bundle.target_id in selected_ids
        }
        bundle_by_id = {bundle.target_id: bundle for bundle in bundles}
        score_by_id = {item.selection.target_id: item for item in scores}
        candidates = []
        for stage1_rank, result in enumerate(retrieval_results, 1):
            target_id = str(result["target_id"])
            bundle = bundle_by_id.get(target_id)
            table_prior = bundle.retrieval_score if bundle else result.get("stage2_table_score")
            candidates.append(
                CandidateResult(
                    target_id=target_id,
                    stage1_rank=stage1_rank,
                    stage1_score=float(result["score"]) if result.get("score") is not None else None,
                    table_prior=float(table_prior) if table_prior is not None else None,
                    bundle=bundle,
                    scores=score_by_id.get(target_id),
                    selected_for_recovery=target_id in selected_ids,
                    direct=direct.get(target_id),
                    evidence=recovered.get(target_id),
                )
            )
        verified = sorted(
            (candidate for candidate in candidates if candidate.semantic_joinability is not None),
            key=lambda candidate: joinability_sort_key(candidate.semantic_joinability, candidate.stage1_rank),
        )
        return Stage2Result(
            query_id=str(query["table_id"]),
            recovery_budget=recovery_budget,
            reranked_candidates=tuple(
                replace(candidate, rerank_rank=rank) for rank, candidate in enumerate(verified, 1)
            ),
            unattempted_candidates=tuple(
                candidate for candidate in candidates if candidate.semantic_joinability is None
            ),
        )

    def recover_candidate(
        self,
        query: dict[str, Any],
        selected_bundle: EvidenceBundle,
        selection: ColumnSelection,
        targets: dict[str, dict[str, Any]],
        evidence: dict[str, dict[str, Any]],
    ) -> EvidenceVerification:
        """Recover one preselected bridge column without mutating the original query."""

        query_id = str(query["table_id"])
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
        return EvidenceVerification(selection, tuple(predictions), check)

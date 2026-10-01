"""Similarity routing from each retrieved evidence object to one query row."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from mmdd_stage1.feature_cache import FeatureStore
from torch.nn import functional as F


class SimilarityEvidenceRouter:
    """Assign every evidence object to its most similar cached query-row view."""

    def __init__(self, features: FeatureStore) -> None:
        self.features = features

    @torch.inference_mode()
    def assign(
        self,
        query_id: str,
        evidence_ids: Sequence[str],
        *,
        row_count: int,
    ) -> dict[str, int]:
        if row_count <= 0:
            raise ValueError("Evidence routing requires at least one query row")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Evidence routing requires unique evidence IDs")
        if not evidence_ids:
            return {}

        query = self.features.get(query_id, include_hidden=False)
        rows = query.row_embeddings
        if rows is None:
            raise ValueError(
                f"{query_id}: feature cache has no row embeddings; rebuild it from Stage-1 objects with "
                "embedding_role='query'"
            )
        if rows.shape[0] != row_count:
            raise ValueError(
                f"{query_id}: feature cache has {rows.shape[0]} row embeddings but Stage 2 loaded {row_count} rows"
            )

        row_vectors = F.normalize(rows.float(), dim=-1)
        evidence_vectors = F.normalize(
            torch.stack(
                [
                    self.features.get(evidence_id, include_hidden=False).embedding
                    for evidence_id in evidence_ids
                ]
            ).float(),
            dim=-1,
        )
        similarities = evidence_vectors @ row_vectors.T
        row_positions = similarities.argmax(dim=-1)
        return {
            evidence_id: int(row_position)
            for evidence_id, row_position in zip(evidence_ids, row_positions)
        }

"""RATA/FOCUS second-stage verification for multimodal join discovery."""

from .pipeline import (
    ColumnSelection,
    DirectVerification,
    LocalizedEvidence,
    RowPrediction,
    Stage2Result,
    Stage2Verifier,
)
from .verifier import (
    CandidateColumnScorer,
    EvidenceBundle,
    RegionOfInterest,
    SemanticJoinability,
    best_image_region,
    best_text_span,
    build_evidence_bundles,
    build_row_evidence_query,
    focus_relevance,
    gaussian_smooth,
    joint_candidate_probabilities,
    joint_relevance,
    propose_image_regions,
    semantic_joinability,
)

__all__ = [
    "CandidateColumnScorer",
    "ColumnSelection",
    "DirectVerification",
    "EvidenceBundle",
    "LocalizedEvidence",
    "RegionOfInterest",
    "RowPrediction",
    "SemanticJoinability",
    "Stage2Result",
    "Stage2Verifier",
    "best_image_region",
    "best_text_span",
    "build_evidence_bundles",
    "build_row_evidence_query",
    "focus_relevance",
    "gaussian_smooth",
    "joint_candidate_probabilities",
    "joint_relevance",
    "propose_image_regions",
    "semantic_joinability",
]

"""Split rules that prevent evaluation labels from leaking into Stage-1 training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def validate_protocol_split(operation: str, split: str) -> None:
    expected = {
        "training": "train",
        "mining": "train",
        "dev_gate": "dev",
        "final_test": "test",
    }
    if operation not in expected:
        raise ValueError(f"Unknown Stage-1 protocol operation: {operation}")
    if split != expected[operation]:
        raise ValueError(
            f"Stage-1 {operation} must use split {expected[operation]!r}, not {split!r}"
        )


def validate_r6_readonly_invariants(
    *,
    evidence_types: Sequence[str],
    candidate_pool_fingerprints: Mapping[str, str],
    teacher_checkpoints: Mapping[str, str],
) -> None:
    """Protect the r6 image, fixed-pool, and lake-local Teacher protocol."""

    if "image" not in evidence_types:
        raise ValueError("Stage-1 r6 configurations must retain image evidence")
    if not candidate_pool_fingerprints or len(
        set(candidate_pool_fingerprints.values())
    ) != 1:
        raise ValueError("Stage-1 r6 comparisons must use one fixed candidate pool")
    if set(teacher_checkpoints) != {"entitables", "wdc"}:
        raise ValueError("Stage-1 r6 requires EntiTables and WDC Teacher provenance")
    if any(not value for value in teacher_checkpoints.values()) or len(
        set(teacher_checkpoints.values())
    ) != 2:
        raise ValueError("Stage-1 r6 requires distinct lake-local Teacher checkpoints")

"""Split rules that prevent evaluation labels from leaking into Stage-1 training."""

from __future__ import annotations


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

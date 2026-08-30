from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from summarize_stage1_r4_taskj import _comparison, _directory_bytes


def _system(
    kind: str,
    values: list[float],
    *,
    direct: list[float] | None = None,
) -> dict:
    per_query = {"recall@10": values}
    return {
        "kind": kind,
        "metrics": {
            "per_query": {
                "fused": per_query,
                "direct": {"recall@10": direct if direct is not None else values},
            }
            if kind == "zero_one_hop_weighted_rrf"
            else per_query
        },
    }


def test_taskj_summary_compares_matching_metric_channels() -> None:
    raw = _system(
        "zero_one_hop_weighted_rrf",
        [0.0, 1.0, 0.0],
        direct=[1.0, 1.0, 0.0],
    )
    ensemble = _system("direct_teacher_ensemble", [1.0, 1.0, 0.0])

    result = _comparison(
        ensemble,
        raw,
        recall_ks=(10,),
        iterations=500,
        seed=13,
        reference_channel="direct",
    )

    assert result["recall@10"]["mean"] == 0.0


def test_taskj_summary_compares_student_fused_to_raw_fused() -> None:
    raw = _system(
        "zero_one_hop_weighted_rrf",
        [0.0, 1.0, 0.0],
        direct=[1.0, 1.0, 0.0],
    )
    student = _system(
        "zero_one_hop_weighted_rrf",
        [1.0, 1.0, 0.0],
        direct=[0.0, 0.0, 0.0],
    )

    result = _comparison(
        student,
        raw,
        recall_ks=(10,),
        iterations=500,
        seed=13,
        candidate_channel="fused",
        reference_channel="fused",
    )

    assert result["recall@10"]["mean"] == 1 / 3


def test_taskj_summary_counts_only_files(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "first").write_bytes(b"123")
    (tmp_path / "nested" / "second").write_bytes(b"4567")

    assert _directory_bytes(tmp_path) == 7

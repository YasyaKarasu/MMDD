from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from summarize_stage1_r4_taskl import (
    _comparison,
    _lake_spec,
    _read_system_or_fallback,
)


def _path_system(fused: list[float], direct: list[float]) -> dict:
    return {
        "metrics": {
            "per_query": {
                "fused": {"recall@10": fused},
                "direct": {"recall@10": direct},
            }
        }
    }


def _direct_system(values: list[float]) -> dict:
    return {"metrics": {"per_query": {"recall@10": values}}}


def test_taskl_comparison_uses_explicit_matching_channels() -> None:
    raw = _path_system([0.0, 1.0, 0.0], [1.0, 1.0, 0.0])
    student = _path_system([1.0, 1.0, 0.0], [0.0, 0.0, 0.0])
    ensemble = _direct_system([1.0, 1.0, 0.0])

    student_delta = _comparison(
        student, raw, recall_ks=(10,), iterations=500, seed=13,
        candidate_channel="fused", reference_channel="fused",
    )
    teacher_delta = _comparison(
        ensemble, raw, recall_ks=(10,), iterations=500, seed=13,
        reference_channel="direct",
    )

    assert student_delta["recall@10"]["mean"] == 1 / 3
    assert teacher_delta["recall@10"]["mean"] == 0.0


def test_taskl_lake_spec_parses_evaluation_directory() -> None:
    assert _lake_spec("wdc=WDC,/tmp/wdc") == ("wdc", "WDC", Path("/tmp/wdc"))


def test_taskl_local_raw_ensemble_overrides_taskj_reference(tmp_path: Path) -> None:
    local = _direct_system([1.0, 0.0])
    taskj = _direct_system([0.0, 1.0])
    metrics = tmp_path / "raw_ensemble" / "metrics.json"
    metrics.parent.mkdir()
    metrics.write_text(
        '{"systems":{"raw_ensemble":'
        + json.dumps(local)
        + "}}",
        encoding="utf-8",
    )

    result = _read_system_or_fallback(metrics, "raw_ensemble", taskj)

    assert result == local

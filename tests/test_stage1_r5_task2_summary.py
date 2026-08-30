from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import summarize_stage1_r5_task2


def _diagnostic(path: Path, teacher_values: list[float]) -> None:
    raw_values = [1.0, 1.0, 0.0, 0.0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "queries": 4,
                "corpus_sha256": "corpus",
                "raw_top_k": 100,
                "teacher_checkpoint": str(path.with_suffix(".pt")),
                "teacher_checkpoint_sha256": path.stem,
                "raw_direct": {
                    "recall@10": sum(raw_values) / 4,
                    "per_query": {"recall@10": raw_values},
                },
                "teacher_reranked": {
                    "recall@10": sum(teacher_values) / 4,
                    "per_query": {"recall@10": teacher_values},
                },
            }
        ),
        encoding="utf-8",
    )


def test_task2_summary_enforces_same_pool_and_reports_gate(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline" / "metrics.json"
    mixed = tmp_path / "mixed" / "metrics.json"
    _diagnostic(baseline, [0.0, 0.0, 0.0, 0.0])
    _diagnostic(mixed, [1.0, 1.0, 0.0, 0.0])

    payload = summarize_stage1_r5_task2.run(
        argparse.Namespace(
            baseline_diagnostic=str(baseline),
            mixed_diagnostic=str(mixed),
            output_dir=str(tmp_path / "output"),
            raw_tolerance=0.03,
            bootstrap_iterations=100,
            bootstrap_seed=13,
        )
    )

    assert payload["gate"]["passed"] is True
    assert payload["mixed_negative_teacher"]["vs_baseline_teacher"]["mean"] == 0.5
    assert (tmp_path / "output" / "same_pool_teacher_comparison.csv").is_file()
    assert "same WDC-local raw top-100" in (
        tmp_path / "output" / "RESULTS.md"
    ).read_text()

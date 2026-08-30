from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from summarize_stage1_r4_taskn import _comparison, _markdown


def test_taskn_comparison_uses_fused_channel_for_students() -> None:
    before = {
        "metrics": {
            "per_query": {
                "fused": {"recall@10": [0.0, 1.0, 0.0]},
                "direct": {"recall@10": [1.0, 1.0, 0.0]},
            }
        }
    }
    after = {
        "metrics": {
            "per_query": {
                "fused": {"recall@10": [1.0, 1.0, 0.0]},
                "direct": {"recall@10": [0.0, 0.0, 0.0]},
            }
        }
    }

    result = _comparison(
        after, before, recall_ks=(10,), iterations=500, seed=13,
        candidate_channel="fused", reference_channel="fused",
    )

    assert result["recall@10"]["mean"] == 1 / 3


def test_taskn_skipped_markdown_records_reason() -> None:
    markdown = _markdown({"skipped": True, "reason": "chain failed"})

    assert "Skipped: chain failed" in markdown
